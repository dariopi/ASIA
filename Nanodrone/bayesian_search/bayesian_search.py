"""Bayesian hyperparameter search using Optuna (TPE) for NanoDrone 3-fold benchmark.

Search space (8 hyperparameters):
  model_class  : categorical {AutoregressiveLSTM, ResidualLSTM}
  lr           : log-uniform [1e-4, 1e-2]
  weight_decay : log-uniform [1e-5, 1e-2]
  dropout_prob : float [0.0, 0.35]
  n_hidden     : int [64, 384]
  num_layers   : int [1, 4]
  hidden_sizes : categorical {(64,), (128,), (256,), (128,64), (256,128)}
  batch_size   : categorical {32, 64, 128, 256}

Fixed: max_epochs=500, grad_clip_norm=1.0, early_stopping_patience=20,
       optimizer=adamw, loss=mae, multihorizon_loss=True

The study is persisted to optuna_study.db (SQLite) — re-running continues from
where it left off (Optuna skips already-completed trials automatically).

Usage:
    python bayesian_search/bayesian_search.py          # 30 trials (default)
    python bayesian_search/bayesian_search.py --n 50
    python bayesian_search/bayesian_search.py --seed 42 --n 30
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import optuna
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
RS_DIR    = REPO_ROOT / "random_search"
THIS_DIR  = Path(__file__).resolve().parent

sys.path.insert(0, str(RS_DIR))
sys.path.insert(0, str(REPO_ROOT))

import prepare
import rs_train as train_module

RESULTS_PATH     = THIS_DIR / "results.tsv"
CHECKPOINTS_ROOT = THIS_DIR / "checkpoints"
STUDY_DB         = THIS_DIR / "optuna_study.db"

FOLD_NAMES = ["chirp", "random", "square"]

FIXED_CONFIG = {
    "activation":               "ReLU",
    "max_epochs":               500,
    "grad_clip_norm":           1.0,
    "early_stopping_patience":  20,
    "optimizer":                "adamw",
    "loss":                     "mae",
    "multihorizon_loss":        True,
}

HIDDEN_SIZES_OPTIONS = ["[64]", "[128]", "[256]", "[128,64]", "[256,128]"]
BATCH_SIZE_OPTIONS   = [32, 64, 128, 256]


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

MODEL_CLASSES = ["AutoregressiveLSTM", "ResidualLSTM"]


def build_config(trial: optuna.Trial) -> dict:
    cfg = copy.deepcopy(FIXED_CONFIG)
    cfg["model_class"]     = trial.suggest_categorical("model_class", MODEL_CLASSES)
    cfg["lr"]              = trial.suggest_float("lr",           1e-4, 1e-2, log=True)
    cfg["weight_decay"]    = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    cfg["dropout_prob"]    = trial.suggest_float("dropout_prob", 0.0,  0.35)
    cfg["n_hidden_states"] = trial.suggest_int("n_hidden",       64,   384)
    cfg["num_layers"]      = trial.suggest_int("num_layers",     1,    4)
    cfg["hidden_sizes"]    = json.loads(trial.suggest_categorical("hidden_sizes", HIDDEN_SIZES_OPTIONS))
    cfg["batch_size"]      = trial.suggest_categorical("batch_size", BATCH_SIZE_OPTIONS)
    return cfg


def config_description(cfg: dict) -> str:
    cls_short = {"AutoregressiveLSTM": "LSTM", "ResidualLSTM": "ResLSTM"}.get(
        cfg["model_class"], cfg["model_class"]
    )
    hs = str(cfg["hidden_sizes"]).replace(" ", "")
    return (
        f"{cls_short} h={cfg['n_hidden_states']} L={cfg['num_layers']} "
        f"hs={hs} bs={cfg['batch_size']} "
        f"lr={cfg['lr']:.2e} do={cfg['dropout_prob']:.2f} wd={cfg['weight_decay']:.0e}"
    )


# ---------------------------------------------------------------------------
# 3-fold cross-validation for one trial
# ---------------------------------------------------------------------------

def run_one_trial(
    trial_num: int,
    cfg: dict,
    train_sequences: dict,
    general_config: dict,
) -> dict:
    ckpt_root = CHECKPOINTS_ROOT / f"trial_{trial_num:04d}"
    gc = copy.deepcopy(general_config)
    gc["checkpoint_path"] = str(ckpt_root)
    gc["log_dir"]         = str(THIS_DIR / "logs")
    gc["threads_per_worker"] = max(1, torch.get_num_threads() // len(FOLD_NAMES))

    mp_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(FOLD_NAMES), mp_context=mp_context) as ex:
        futures = {
            ex.submit(
                train_module.train_one_fold,
                fold_name,
                train_sequences,
                gc,
                copy.deepcopy(cfg),
            ): fold_name
            for fold_name in FOLD_NAMES
        }
        fold_results = [f.result() for f in as_completed(futures)]

    fold_results.sort(key=lambda r: r["validation_name"])
    summary = train_module.summarize_folds(fold_results)
    train_module.save_current_summary(ckpt_root, summary)

    return {
        "trial_num":  trial_num,
        "cfg":        cfg,
        "summary":    summary,
        "val_mae":    summary["validation_mae_norm_mean"],
        "fold_vals":  [r["validation_mae_norm"] for r in fold_results],
        "fold_names": [r["validation_name"]     for r in fold_results],
    }


# ---------------------------------------------------------------------------
# Results file
# ---------------------------------------------------------------------------

def init_results_file() -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_PATH.exists():
        return
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "trial", "val_MAE",
            "chirp", "random", "square",
            "model_class", "n_hidden", "num_layers", "hidden_sizes", "batch_size",
            "lr", "dropout", "weight_decay",
            "description",
        ])


def append_result(result: dict) -> None:
    cfg = result["cfg"]
    folds_by_name = dict(zip(result["fold_names"], result["fold_vals"]))
    with RESULTS_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            result["trial_num"],
            f"{result['val_mae']:.6f}",
            f"{folds_by_name.get('chirp',  float('nan')):.6f}",
            f"{folds_by_name.get('random', float('nan')):.6f}",
            f"{folds_by_name.get('square', float('nan')):.6f}",
            cfg["model_class"],
            cfg["n_hidden_states"],
            cfg["num_layers"],
            str(cfg["hidden_sizes"]),
            cfg["batch_size"],
            f"{cfg['lr']:.2e}",
            f"{cfg['dropout_prob']:.3f}",
            f"{cfg['weight_decay']:.0e}",
            config_description(cfg),
        ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n",    type=int, default=30, help="Number of trials")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train_sequences, _, general_config = prepare.load_datasets_and_config()
    prepare.set_global_seed(general_config["seed"])

    init_results_file()

    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=5)
    storage = f"sqlite:///{STUDY_DB}"
    study = optuna.create_study(
        study_name="nanodrone_bayesian",
        direction="minimize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    completed_so_far = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_remaining = max(0, args.n - completed_so_far)

    print(f"Bayesian search (TPE): {args.n} trials total, {completed_so_far} already done, "
          f"{n_remaining} to run  |  seed={args.seed}")
    print(f"Results  → {RESULTS_PATH}")
    print(f"Study DB → {STUDY_DB}\n")

    best_val = min(
        (t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
        default=float("inf"),
    )
    trial_counter = completed_so_far

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_val, trial_counter
        trial_counter += 1
        cfg  = build_config(trial)
        desc = config_description(cfg)
        print(f"[{trial_counter:>3}/{args.n}] trial={trial.number}  {desc}")

        t0     = time.perf_counter()
        result = run_one_trial(trial.number, cfg, train_sequences, general_config)
        elapsed = time.perf_counter() - t0

        val = result["val_mae"]
        folds_by_name = dict(zip(result["fold_names"], result["fold_vals"]))
        fold_str = "  ".join(f"{n}={folds_by_name[n]:.4f}" for n in FOLD_NAMES if n in folds_by_name)
        marker = " ★ NEW BEST" if val < best_val else ""
        print(f"         val={val:.6f}  [{fold_str}]  {elapsed:.0f}s{marker}")

        if val < best_val:
            best_val = val

        append_result(result)
        return val

    study.optimize(objective, n_trials=n_remaining)

    best_trial = study.best_trial
    print(f"\nBest: trial {best_trial.number}  val_MAE={best_trial.value:.6f}")
    print(f"  params: {best_trial.params}")

    best_cfg = copy.deepcopy(FIXED_CONFIG)
    best_cfg.update(best_trial.params)
    best_cfg["n_hidden_states"] = best_trial.params["n_hidden"]

    best_json = THIS_DIR / "best_config.json"
    best_json.write_text(json.dumps({
        "trial":        best_trial.number,
        "val_MAE":      f"{best_trial.value:.6f}",
        **{k: str(v) for k, v in best_trial.params.items()},
        "description":  config_description(best_cfg),
    }, indent=2), encoding="utf-8")
    print(f"Best config → {best_json}")


if __name__ == "__main__":
    main()
