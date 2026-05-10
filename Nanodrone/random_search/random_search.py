"""Random hyperparameter search for the NanoDrone 3-Fold benchmark.

Reuses the same train_one_fold infrastructure (same time budget, same 3-fold CV).
Varies: model architecture, layers, hidden sizes, lr, dropout, weight_decay.

Usage:
    python random_search/random_search.py            # 30 configs (default)
    python random_search/random_search.py --n 50
    python random_search/random_search.py --seed 0 --n 30
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import multiprocessing
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT))

import prepare
import rs_train as train_module
from model import build_model_from_config

OUT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = OUT_DIR / "results.tsv"
CHECKPOINTS_ROOT = OUT_DIR / "checkpoints"


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

HIDDEN_SIZES_OPTIONS = [
    [32],
    [64],
    [128],
    [256],
    [64, 32],
    [64, 64],
    [128, 64],
    [128, 128],
    [256, 128],
    [256, 256],
]


def sample_config(rng: random.Random) -> dict:
    return {
        "lr": float(np.exp(rng.uniform(np.log(5e-4), np.log(5e-3)))),
        "max_epochs": 500,
        "model_class": rng.choice(["AutoregressiveLSTM", "PhysicsResidualLSTM"]),
        "n_hidden_states": rng.choice([64, 128, 192, 256, 320]),
        "hidden_sizes": rng.choice(HIDDEN_SIZES_OPTIONS),
        "activation": "ReLU",
        "num_layers": rng.choice([1, 2, 3]),
        "dropout_prob": rng.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.25]),
        "weight_decay": rng.choice([1e-5, 1e-4, 1e-3]),
        "optimizer": "adamw",
        "grad_clip_norm": 1.0,
        "early_stopping_patience": 20,
        "batch_size": 64,
        "loss": "mae",
        "multihorizon_loss": True,
    }


def config_description(cfg: dict) -> str:
    cls = {"AutoregressiveLSTM": "AR-LSTM", "PhysicsResidualLSTM": "Phys+Res"}.get(
        cfg["model_class"], cfg["model_class"]
    )
    return (
        f"{cls} h={cfg['n_hidden_states']} L={cfg['num_layers']} "
        f"lr={cfg['lr']:.1e} do={cfg['dropout_prob']} wd={cfg['weight_decay']:.0e}"
    )


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------

def evaluate_test_ensemble_mae(
    checkpoint_root: Path,
    test_sequences: dict,
    general_config: dict,
) -> dict[str, float]:
    """Ensemble MAE (denormalized) for each test sequence using the saved fold models."""
    fold_dirs = sorted(d for d in checkpoint_root.iterdir() if d.is_dir())
    checkpoint_paths = [d / "model.pt" for d in fold_dirs if (d / "model.pt").exists()]
    if not checkpoint_paths:
        return {name: float("nan") for name in test_sequences}

    device = general_config["device"]
    results = {}
    for test_name, test_seq in test_sequences.items():
        fold_preds = []
        for ckpt_path in checkpoint_paths:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = build_model_from_config(
                config_pars=ckpt["model_config"],
                n_inputs=general_config["n_inputs"],
                n_states=general_config["n_states"],
                n_outputs=general_config["n_outputs"],
            ).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            normalizer = prepare.Normalizer.from_state_dict(ckpt["normalizer"])
            test_norm = normalizer.normalize_sequence(test_seq)
            with torch.no_grad():
                y_hat_norm, _ = model(
                    test_norm.u.to(device),
                    test_norm.y0.to(device),
                )
                y_hat = normalizer.denormalize_y_tensor(y_hat_norm).detach().cpu().numpy()
            fold_preds.append(y_hat)
        ensemble = np.mean(np.stack(fold_preds, axis=0), axis=0)
        results[test_name] = prepare.mae(test_seq.y.numpy(), ensemble)
    return results


# ---------------------------------------------------------------------------
# Run one config (3-fold CV + test evaluation)
# ---------------------------------------------------------------------------

def run_one_config(
    config_id: int,
    cfg: dict,
    train_sequences: dict,
    test_sequences: dict,
    general_config: dict,
) -> dict:
    checkpoint_root = CHECKPOINTS_ROOT / f"config_{config_id:04d}"
    general_config_local = copy.deepcopy(general_config)
    general_config_local["checkpoint_path"] = str(checkpoint_root)
    general_config_local["log_dir"] = str(OUT_DIR / "logs")

    fold_names = sorted(train_sequences)
    n_workers = len(fold_names)
    general_config_local["threads_per_worker"] = max(
        1, torch.get_num_threads() // n_workers
    )

    mp_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as executor:
        futures = {
            executor.submit(
                train_module.train_one_fold, name, train_sequences, general_config_local, cfg
            ): name
            for name in fold_names
        }
        fold_results = [f.result() for f in as_completed(futures)]

    fold_results.sort(key=lambda r: r["validation_name"])
    summary = train_module.summarize_folds(fold_results)
    train_module.save_current_summary(checkpoint_root, summary)

    test_maes = evaluate_test_ensemble_mae(checkpoint_root, test_sequences, general_config_local)

    return {
        "config_id": config_id,
        "cfg": cfg,
        "summary": summary,
        "val_mae": summary["validation_mae_norm_mean"],
        "test_maes": test_maes,
        "fold_vals": [r["validation_mae_norm"] for r in fold_results],
        "fold_names": [r["validation_name"] for r in fold_results],
    }


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def init_results_file(test_names: list[str]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "config_id", "val_MAE",
            *[f"test_MAE_{n}" for n in test_names],
            "chirp", "random", "square",
            "model_class", "n_hidden", "num_layers", "hidden_sizes",
            "lr", "dropout", "weight_decay",
            "description",
        ])


def append_result(result: dict, test_names: list[str]) -> None:
    cfg = result["cfg"]
    folds_by_name = dict(zip(result["fold_names"], result["fold_vals"]))
    test_maes = result.get("test_maes", {})
    with RESULTS_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            result["config_id"],
            f"{result['val_mae']:.6f}",
            *[f"{test_maes.get(n, float('nan')):.6f}" for n in test_names],
            f"{folds_by_name.get('chirp', float('nan')):.6f}",
            f"{folds_by_name.get('random', float('nan')):.6f}",
            f"{folds_by_name.get('square', float('nan')):.6f}",
            cfg["model_class"],
            cfg["n_hidden_states"],
            cfg["num_layers"],
            str(cfg["hidden_sizes"]),
            f"{cfg['lr']:.2e}",
            cfg["dropout_prob"],
            cfg["weight_decay"],
            config_description(cfg),
        ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="Number of random configs")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    train_sequences, test_sequences, general_config = prepare.load_datasets_and_config()
    prepare.set_global_seed(general_config["seed"])

    test_names = sorted(test_sequences)
    init_results_file(test_names)
    print(f"Random search: {args.n} configs, seed={args.seed}")
    print(f"Results → {RESULTS_PATH}\n")

    best_val = float("inf")
    best_id  = -1

    for i in range(1, args.n + 1):
        cfg = sample_config(rng)
        desc = config_description(cfg)
        print(f"[{i:>3}/{args.n}] {desc}")

        t0 = time.perf_counter()
        result = run_one_config(i, cfg, train_sequences, test_sequences, general_config)
        elapsed = time.perf_counter() - t0

        val = result["val_mae"]
        fold_str = "  ".join(
            f"{n}={v:.4f}"
            for n, v in zip(result["fold_names"], result["fold_vals"])
        )
        test_str = "  ".join(
            f"{n}={result['test_maes'].get(n, float('nan')):.4f}"
            for n in test_names
        )
        marker = " ★ NEW BEST" if val < best_val else ""
        print(f"         val={val:.6f}  [{fold_str}]  test=[{test_str}]  {elapsed:.0f}s{marker}")

        if val < best_val:
            best_val = val
            best_id  = i

        append_result(result, test_names)

    print(f"\nBest: config {best_id}  val_MAE={best_val:.6f}")
    print(f"Results saved to {RESULTS_PATH}")

    # Save best config as JSON
    best_json = OUT_DIR / "best_config.json"
    with RESULTS_PATH.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    best_row = min(rows, key=lambda r: float(r["val_MAE"]))
    best_json.write_text(json.dumps(best_row, indent=2), encoding="utf-8")
    print(f"Best config   → {best_json}")


if __name__ == "__main__":
    main()
