from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import prepare
from model import build_model_from_config


def load_checkpoint_paths(checkpoint_root: Path, checkpoint_set: str) -> list[Path]:
    summary_root = checkpoint_root if checkpoint_set == "current" else checkpoint_root / "best_so_far"
    summary_path = summary_root / "cross_validation_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return [Path(item["checkpoint_path"]) for item in summary["folds"]]

    best_summary_path = checkpoint_root / "best_so_far" / "cross_validation_summary.json"
    if best_summary_path.exists():
        summary = json.loads(best_summary_path.read_text(encoding="utf-8"))
        return [Path(item["checkpoint_path"]) for item in summary["folds"]]

    return sorted(checkpoint_root.glob("*/model.pt"))


def load_model_and_normalizer(
    checkpoint_path: Path,
    general_config: dict,
) -> tuple[torch.nn.Module, prepare.Normalizer]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=general_config["device"],
        weights_only=False,
    )
    model = build_model_from_config(
        config_pars=checkpoint["model_config"],
        n_inputs=general_config["n_inputs"],
        n_states=general_config["n_states"],
        n_outputs=general_config["n_outputs"],
    ).to(general_config["device"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    normalizer = prepare.Normalizer.from_state_dict(checkpoint["normalizer"])
    return model, normalizer


def predict_with_checkpoint(
    checkpoint_path: Path,
    test_sequence: prepare.DroneSequence,
    general_config: dict,
) -> np.ndarray:
    """Return denormalized predictions of shape (N, seq_len, n_outputs)."""
    model, normalizer = load_model_and_normalizer(checkpoint_path, general_config)
    test_norm = normalizer.normalize_sequence(test_sequence)

    with torch.no_grad():
        y_hat_norm, _ = model(
            test_norm.u.to(general_config["device"]),
            test_norm.y0.to(general_config["device"]),
        )
        y_hat = normalizer.denormalize_y_tensor(y_hat_norm).detach().cpu().numpy()

    return y_hat  # (N, seq_len, n_outputs)


def r2_per_output(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """R² for each output dimension. Inputs shape: (N, seq_len, n_outputs) or (T, n_outputs)."""
    yt = y_true.reshape(-1, y_true.shape[-1])
    yp = y_pred.reshape(-1, y_pred.shape[-1])
    ss_res = np.sum((yt - yp) ** 2, axis=0)
    ss_tot = np.sum((yt - yt.mean(axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / np.where(ss_tot == 0, 1.0, ss_tot)


def plot_test_predictions(
    test_sequence: prepare.DroneSequence,
    fold_predictions: list[np.ndarray],
    ensemble_prediction: np.ndarray,
    plot_path: Path,
) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    y_true_flat = test_sequence.y.numpy().reshape(-1, test_sequence.y.shape[-1])
    ensemble_flat = ensemble_prediction.reshape(-1, ensemble_prediction.shape[-1])
    fold_flat = [p.reshape(-1, p.shape[-1]) for p in fold_predictions]

    T = y_true_flat.shape[0]
    time_axis = np.arange(T)
    n_out = len(prepare.output_names)
    n_cols = 3
    n_rows = (n_out + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows), sharex=True)
    axes = np.array(axes).flatten()

    for i, out_name in enumerate(prepare.output_names):
        ax = axes[i]
        ax.plot(time_axis, y_true_flat[:, i], linewidth=1.2, color="#1d3557", label="true")
        for j, fp in enumerate(fold_flat):
            ax.plot(
                time_axis, fp[:, i], linewidth=0.7, alpha=0.35,
                label="fold models" if j == 0 else None,
            )
        ax.plot(time_axis, ensemble_flat[:, i], linewidth=1.2, color="#e76f51", label="ensemble")
        ax.set_ylabel(out_name, fontsize=8)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(fontsize=7)

    for ax in axes[n_out:]:
        ax.set_visible(False)

    axes[n_out - 1].set_xlabel("time step")
    fig.suptitle(f"{test_sequence.name} — test predictions (denormalized)")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fold checkpoints on the official test trajectories."
    )
    parser.add_argument(
        "--checkpoint-set",
        choices=("current", "best_so_far"),
        default="current",
        help="Which saved fold checkpoints to evaluate.",
    )
    return parser.parse_args()


def main() -> dict:
    args = parse_args()
    train_sequences, test_sequences, general_config = prepare.load_datasets_and_config()

    checkpoint_root = Path(general_config["checkpoint_path"])
    plots_root = Path(general_config["plots_path"])

    checkpoint_paths = load_checkpoint_paths(checkpoint_root, args.checkpoint_set)
    if not checkpoint_paths:
        raise FileNotFoundError("No fold checkpoints found. Run `train.py` first.")

    output_checkpoint_root = (
        checkpoint_root if args.checkpoint_set == "current" else checkpoint_root / "best_so_far"
    )
    output_plot_root = (
        plots_root if args.checkpoint_set == "current" else plots_root / "best_so_far"
    )
    output_checkpoint_root.mkdir(parents=True, exist_ok=True)
    output_plot_root.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, object] = {}

    for test_name, test_sequence in test_sequences.items():
        fold_predictions = [
            predict_with_checkpoint(path, test_sequence, general_config)
            for path in checkpoint_paths
        ]
        ensemble_prediction = np.mean(np.stack(fold_predictions, axis=0), axis=0)

        y_true = test_sequence.y.numpy()
        test_mae = prepare.mae(y_true, ensemble_prediction)
        fold_maes = [prepare.mae(y_true, p) for p in fold_predictions]
        r2_per_out = r2_per_output(y_true, ensemble_prediction)

        metrics = {
            "metric": "mae",
            "metric_scale": "denormalized_physical_units",
            "num_models": len(checkpoint_paths),
            "fold_test_mae_denormalized": [float(v) for v in fold_maes],
            "ensemble_test_mae_denormalized": float(test_mae),
            "ensemble_r2_per_output": {
                name: float(r2_per_out[i])
                for i, name in enumerate(prepare.output_names)
            },
            "ensemble_r2_mean": float(r2_per_out.mean()),
            "ensemble_rule": "Mean of the denormalized predictions of the saved fold models.",
            "checkpoint_paths": [str(p) for p in checkpoint_paths],
        }
        all_metrics[test_name] = metrics

        np.savez_compressed(
            output_checkpoint_root / f"test_ensemble_predictions_{test_name}.npz",
            y_true=y_true.astype(np.float32),
            ensemble_prediction=ensemble_prediction.astype(np.float32),
            fold_predictions=np.stack(fold_predictions, axis=0).astype(np.float32),
        )
        (output_checkpoint_root / f"test_ensemble_metrics_{test_name}.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        plot_test_predictions(
            test_sequence=test_sequence,
            fold_predictions=fold_predictions,
            ensemble_prediction=ensemble_prediction,
            plot_path=output_plot_root / f"test_ensemble_prediction_{test_name}.png",
        )

        print(f"\nTest ensemble summary ({args.checkpoint_set}) — {test_name}")
        print(f"Number of fold models : {len(checkpoint_paths)}")
        print(f"Ensemble test MAE     : {test_mae:.6f}")
        print(f"Ensemble mean R²      : {r2_per_out.mean():.4f}")
        print("R² per output:")
        for name, r2 in zip(prepare.output_names, r2_per_out):
            print(f"  {name:6s}: {r2:.4f}")
        print(f"Fold MAEs             : {[f'{v:.6f}' for v in fold_maes]}")
        print(f"Saved metrics         : {output_checkpoint_root / f'test_ensemble_metrics_{test_name}.json'}")

    return all_metrics


if __name__ == "__main__":
    main()
