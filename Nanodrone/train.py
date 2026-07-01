from __future__ import annotations

import copy
import json
import multiprocessing
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

import prepare
from model import build_model_from_config


config_pars = {
    "lr": 2e-3,
    "max_epochs": 1200,
    "type": "LSTM",
    "n_hidden_states": 32,
    "hidden_sizes": [32],
    "activation": "ReLU",
    "num_layers": 1,
    "dropout_prob": 0.0,
    "weight_decay": 0.0,
    "grad_clip_norm": 1.0,
    "early_stopping_patience": 20,
}


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def append_log_line(log_path: Path, line: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def initialize_fold_log(log_path: Path, fold_name: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"{fold_name}\n")
        handle.write("epoch,train_loss_norm_mse,train_mae_norm,val_mae_norm\n")


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(model_config: dict, general_config: dict) -> torch.nn.Module:
    return build_model_from_config(
        config_pars=model_config,
        n_inputs=general_config["n_inputs"],
        n_states=general_config["n_states"],
        n_outputs=general_config["n_outputs"],
    ).to(general_config["device"])


# ---------------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------------

def compute_training_loss(
    model: torch.nn.Module,
    normalized_sequences: list[prepare.DroneSequence],
    device: str,
) -> torch.Tensor:
    """MSE loss summed across all windows, time steps, and outputs, then normalised."""
    total_sse = None
    total_count = 0

    for sequence in normalized_sequences:
        u = sequence.u.to(device)       # (N, 50, 4)
        y_tgt = sequence.y.to(device)   # (N, 50, 12)
        y0 = sequence.y0.to(device)     # (N, 1, 12)

        y_hat, _ = model(u, y0)
        diff = y_hat - y_tgt
        sse = torch.sum(diff ** 2)
        total_sse = sse if total_sse is None else total_sse + sse
        total_count += diff.numel()

    if total_sse is None or total_count == 0:
        raise RuntimeError("No training sequences provided.")

    return total_sse / total_count


def evaluate_one_sequence(
    model: torch.nn.Module,
    sequence_raw: prepare.DroneSequence,
    sequence_norm: prepare.DroneSequence,
    normalizer: prepare.Normalizer,
    device: str,
) -> dict[str, float]:
    """MAE on a single trajectory (all its windows)."""
    model.eval()
    with torch.no_grad():
        y_hat_norm, _ = model(sequence_norm.u.to(device), sequence_norm.y0.to(device))
        y_hat_raw = normalizer.denormalize_y_tensor(y_hat_norm)

    return {
        "mae_norm": prepare.mae(
            sequence_norm.y.detach().cpu().numpy(),
            y_hat_norm.detach().cpu().numpy(),
        ),
        "mae_raw": prepare.mae(
            sequence_raw.y.detach().cpu().numpy(),
            y_hat_raw.detach().cpu().numpy(),
        ),
    }


def aggregate_metrics_across_sequences(
    model: torch.nn.Module,
    raw_sequences: list[prepare.DroneSequence],
    norm_sequences: list[prepare.DroneSequence],
    normalizer: prepare.Normalizer,
    device: str,
) -> dict[str, float]:
    """MAE pooled over all training trajectories."""
    all_targets_norm, all_predictions_norm = [], []
    all_targets_raw, all_predictions_raw = [], []

    model.eval()
    with torch.no_grad():
        for raw_seq, norm_seq in zip(raw_sequences, norm_sequences):
            y_hat_norm, _ = model(norm_seq.u.to(device), norm_seq.y0.to(device))
            y_hat_raw = normalizer.denormalize_y_tensor(y_hat_norm)
            all_targets_norm.append(norm_seq.y.detach().cpu().numpy())
            all_predictions_norm.append(y_hat_norm.detach().cpu().numpy())
            all_targets_raw.append(raw_seq.y.detach().cpu().numpy())
            all_predictions_raw.append(y_hat_raw.detach().cpu().numpy())

    return {
        "mae_norm": prepare.mae(
            np.concatenate(all_targets_norm, axis=0),
            np.concatenate(all_predictions_norm, axis=0),
        ),
        "mae_raw": prepare.mae(
            np.concatenate(all_targets_raw, axis=0),
            np.concatenate(all_predictions_raw, axis=0),
        ),
    }


# ---------------------------------------------------------------------------
# Single-fold training (runs in a subprocess)
# ---------------------------------------------------------------------------

def train_one_fold(
    validation_name: str,
    train_sequences_raw: dict[str, prepare.DroneSequence],
    general_config: dict,
) -> dict:
    torch.set_num_threads(general_config.get("threads_per_worker", torch.get_num_threads()))
    prepare.set_global_seed(general_config["seed"])

    train_names = [name for name in train_sequences_raw if name != validation_name]
    raw_train_list = [train_sequences_raw[name] for name in train_names]
    raw_validation = train_sequences_raw[validation_name]

    normalizer = prepare.Normalizer.fit(raw_train_list)
    train_list_norm = [normalizer.normalize_sequence(seq) for seq in raw_train_list]
    validation_norm = normalizer.normalize_sequence(raw_validation)

    device = general_config["device"]
    checkpoint_root = Path(general_config["checkpoint_path"])
    log_dir = Path(general_config["log_dir"])

    fold_dir = checkpoint_root / validation_name
    fold_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(config_pars, general_config)

    wd = config_pars.get("weight_decay", 0.0)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config_pars["lr"], weight_decay=wd
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, threshold=1e-4, min_lr=1e-5,
    )

    best_state_dict = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale_evaluations = 0
    fold_start_time = time.perf_counter()
    fold_time_budget_seconds = float(general_config["fold_time_budget_seconds"])
    eval_every = int(general_config["eval_every"])

    fold_log_path = log_dir / f"train_{validation_name}.log"
    initialize_fold_log(fold_log_path, validation_name)

    with torch.no_grad():
        initial_loss = compute_training_loss(model, train_list_norm, device=device)
    initial_train = aggregate_metrics_across_sequences(
        model, raw_train_list, train_list_norm, normalizer, device
    )
    initial_val = evaluate_one_sequence(
        model, raw_validation, validation_norm, normalizer, device
    )
    best_val_mae_norm = float(initial_val["mae_norm"])

    initial_line = (
        f"0,"
        f"{initial_loss.item():.6f},"
        f"{initial_train['mae_norm']:.6f},"
        f"{initial_val['mae_norm']:.6f}"
    )
    print(f"[{validation_name}] {initial_line}")
    append_log_line(fold_log_path, initial_line)

    for epoch in range(1, config_pars["max_epochs"] + 1):
        if time.perf_counter() - fold_start_time >= fold_time_budget_seconds:
            break

        model.train()
        optimizer.zero_grad()

        loss = compute_training_loss(model, train_list_norm, device=device)
        loss.backward()

        grad_clip = config_pars.get("grad_clip_norm", 0.0)
        if grad_clip and grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        should_evaluate = epoch == 1 or epoch % eval_every == 0 or epoch == config_pars["max_epochs"]
        if not should_evaluate:
            continue

        if time.perf_counter() - fold_start_time >= fold_time_budget_seconds:
            break

        train_metrics = aggregate_metrics_across_sequences(
            model, raw_train_list, train_list_norm, normalizer, device
        )
        val_metrics = evaluate_one_sequence(
            model, raw_validation, validation_norm, normalizer, device
        )

        line = (
            f"{epoch},"
            f"{loss.item():.6f},"
            f"{train_metrics['mae_norm']:.6f},"
            f"{val_metrics['mae_norm']:.6f}"
        )
        print(f"[{validation_name}] {line}")
        append_log_line(fold_log_path, line)

        if val_metrics["mae_norm"] < best_val_mae_norm:
            best_val_mae_norm = float(val_metrics["mae_norm"])
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            stale_evaluations = 0
        else:
            stale_evaluations += 1

        scheduler.step(val_metrics["mae_norm"])

        if stale_evaluations >= config_pars["early_stopping_patience"]:
            break

    model.load_state_dict(best_state_dict)

    final_train = aggregate_metrics_across_sequences(
        model, raw_train_list, train_list_norm, normalizer, device
    )
    final_val = evaluate_one_sequence(
        model, raw_validation, validation_norm, normalizer, device
    )

    checkpoint_payload = {
        "validation_name": validation_name,
        "train_names": train_names,
        "model_state_dict": model.state_dict(),
        "model_config": copy.deepcopy(config_pars),
        "general_config": general_config,
        "normalizer": normalizer.state_dict(),
        "best_epoch": best_epoch,
        "metrics": {
            "train_mae_norm": float(final_train["mae_norm"]),
            "validation_mae_norm": float(final_val["mae_norm"]),
            "train_mae_raw": float(final_train["mae_raw"]),
            "validation_mae_raw": float(final_val["mae_raw"]),
        },
    }
    torch.save(checkpoint_payload, fold_dir / "model.pt")

    return {
        "validation_name": validation_name,
        "best_epoch": best_epoch,
        "checkpoint_path": str(fold_dir / "model.pt"),
        "train_mae_norm": float(final_train["mae_norm"]),
        "validation_mae_norm": float(final_val["mae_norm"]),
        "train_mae_raw": float(final_train["mae_raw"]),
        "validation_mae_raw": float(final_val["mae_raw"]),
    }


# ---------------------------------------------------------------------------
# Summary and best-so-far management
# ---------------------------------------------------------------------------

def summarize_folds(fold_results: list[dict]) -> dict:
    train_norm = np.asarray([r["train_mae_norm"] for r in fold_results])
    val_norm = np.asarray([r["validation_mae_norm"] for r in fold_results])
    train_raw = np.asarray([r["train_mae_raw"] for r in fold_results])
    val_raw = np.asarray([r["validation_mae_raw"] for r in fold_results])
    best_epochs = np.asarray([r["best_epoch"] for r in fold_results])

    return {
        "metric": "mae",
        "training_progress_metric": "normalized_mae",
        "num_folds": len(fold_results),
        "train_mae_norm_mean": float(train_norm.mean()),
        "train_mae_norm_std": float(train_norm.std()),
        "validation_mae_norm_mean": float(val_norm.mean()),
        "validation_mae_norm_std": float(val_norm.std()),
        "train_mae_raw_mean": float(train_raw.mean()),
        "train_mae_raw_std": float(train_raw.std()),
        "validation_mae_raw_mean": float(val_raw.mean()),
        "validation_mae_raw_std": float(val_raw.std()),
        "best_epoch_mean": float(best_epochs.mean()),
        "best_epoch_median": int(np.median(best_epochs)),
        "folds": fold_results,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_current_summary(checkpoint_root: Path, summary: dict) -> Path:
    summary_path = checkpoint_root / "cross_validation_summary.json"
    write_json(summary_path, summary)
    return summary_path


def maybe_update_best_so_far(checkpoint_root: Path, summary: dict) -> bool:
    best_root = checkpoint_root / "best_so_far"
    best_summary_path = best_root / "cross_validation_summary.json"
    current_value = float(summary["validation_mae_norm_mean"])

    if best_summary_path.exists():
        previous_summary = json.loads(best_summary_path.read_text(encoding="utf-8"))
        previous_value = float(previous_summary["validation_mae_norm_mean"])
        if current_value >= previous_value:
            return False

    updated_folds = []
    for fold_result in summary["folds"]:
        validation_name = str(fold_result["validation_name"])
        source_checkpoint = Path(str(fold_result["checkpoint_path"]))
        target_checkpoint = best_root / validation_name / "model.pt"
        target_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_checkpoint, target_checkpoint)
        updated_fold_result = dict(fold_result)
        updated_fold_result["checkpoint_path"] = str(target_checkpoint)
        updated_folds.append(updated_fold_result)

    best_summary = dict(summary)
    best_summary["folds"] = updated_folds
    write_json(best_summary_path, best_summary)
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> dict:
    train_sequences, general_config = prepare.load_train_sequences_and_config()
    prepare.set_global_seed(general_config["seed"])

    checkpoint_root = Path(general_config["checkpoint_path"])
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    fold_names = sorted(train_sequences)
    n_workers = len(fold_names)
    general_config["threads_per_worker"] = max(1, torch.get_num_threads() // n_workers)

    mp_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as executor:
        futures = {
            executor.submit(train_one_fold, name, train_sequences, general_config): name
            for name in fold_names
        }
        fold_results = [f.result() for f in as_completed(futures)]

    fold_results.sort(key=lambda r: r["validation_name"])

    summary = summarize_folds(fold_results)
    save_current_summary(checkpoint_root, summary)
    best_so_far_updated = maybe_update_best_so_far(checkpoint_root, summary)

    print("")
    print("Cross-validation summary")
    print(f"Mean train MAE norm       : {summary['train_mae_norm_mean']:.6f}")
    print(f"Mean validation MAE norm  : {summary['validation_mae_norm_mean']:.6f}")
    print(f"Median best epoch         : {summary['best_epoch_median']}")
    print(
        "Best-so-far checkpoint    : "
        f"{'updated' if best_so_far_updated else 'kept previous'}"
    )

    return summary


if __name__ == "__main__":
    main()
