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

    return sorted(checkpoint_root.glob("fold_*/model.pt"))


def load_model_and_normalizer(
    checkpoint_path: Path,
    general_config: dict,
) -> tuple[torch.nn.Module, prepare.Normalizer]:
    # These checkpoints are produced locally by train.py and contain pickled
    # numpy state in addition to tensors, so we load them in trusted mode.
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
    test_sequence: prepare.TankSequence,
    general_config: dict,
) -> np.ndarray:
    model, normalizer = load_model_and_normalizer(checkpoint_path, general_config)
    test_sequence_norm = normalizer.normalize_sequence(test_sequence)

    with torch.no_grad():
        y_hat_norm, _ = model(
            test_sequence_norm.u.to(general_config["device"]),
            test_sequence_norm.y0.to(general_config["device"]),
        )
        y_hat = normalizer.denormalize_y_tensor(y_hat_norm).detach().cpu().numpy()

    # Test-time predictions are returned in physical units.
    return y_hat[0]


def plot_test_predictions(
    test_sequence: prepare.TankSequence,
    fold_predictions: list[np.ndarray],
    ensemble_prediction: np.ndarray,
    plot_path: Path,
) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    time_axis = np.arange(test_sequence.num_samples, dtype=np.float32) * test_sequence.sampling_time
    u_values = test_sequence.u[0].detach().cpu().numpy()
    y_true = test_sequence.y[0].detach().cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(time_axis, u_values[:, 0], linewidth=1.5, color="#006d77")
    axes[0].set_ylabel(prepare.input_names[0])
    axes[0].set_title("Test input trajectory")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(time_axis, y_true[:, 0], linewidth=2.0, color="#1d3557", label="true output")
    for index, prediction in enumerate(fold_predictions, start=1):
        axes[1].plot(
            time_axis,
            prediction[:, 0],
            linewidth=1.0,
            alpha=0.35,
            label="individual fold models" if index == 1 else None,
        )
    axes[1].plot(
        time_axis,
        ensemble_prediction[:, 0],
        linewidth=2.0,
        color="#e76f51",
        label="ensemble mean",
    )
    axes[1].axvline(
        test_sequence.warmup * test_sequence.sampling_time,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="test warmup",
    )
    axes[1].set_ylabel(prepare.output_names[0])
    axes[1].set_xlabel("time [s]")
    axes[1].set_title("Test output prediction (denormalized)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fold checkpoints on the official test set.")
    parser.add_argument(
        "--checkpoint-set",
        choices=("current", "best_so_far"),
        default="current",
        help="Which saved fold checkpoints to evaluate.",
    )
    return parser.parse_args()


def main() -> dict[str, object]:
    args = parse_args()
    _, test_sequence, general_config = prepare.load_datasets_and_config()

    checkpoint_root = Path(general_config["checkpoint_path"])
    plots_root = Path(general_config["plots_path"])

    checkpoint_paths = load_checkpoint_paths(checkpoint_root, args.checkpoint_set)
    if not checkpoint_paths:
        raise FileNotFoundError("No fold checkpoints found. Run `train.py` first.")

    fold_predictions = [
        predict_with_checkpoint(path, test_sequence=test_sequence, general_config=general_config)
        for path in checkpoint_paths
    ]
    ensemble_prediction = np.mean(np.stack(fold_predictions, axis=0), axis=0)

    y_true = test_sequence.y[0].detach().cpu().numpy()
    test_rmse = prepare.rmse(y_true[test_sequence.warmup :], ensemble_prediction[test_sequence.warmup :])
    fold_rmses = [
        prepare.rmse(y_true[test_sequence.warmup :], prediction[test_sequence.warmup :])
        for prediction in fold_predictions
    ]

    metrics = {
        "metric": "rmse",
        "metric_scale": "denormalized_physical_units",
        "plot_scale": "denormalized_physical_units",
        "warmup_test": int(test_sequence.warmup),
        "num_models": len(checkpoint_paths),
        "fold_test_rmse_denormalized": [float(value) for value in fold_rmses],
        "ensemble_test_rmse_denormalized": float(test_rmse),
        # Backward-compatible aliases.
        "fold_test_rmse": [float(value) for value in fold_rmses],
        "ensemble_test_rmse": float(test_rmse),
        "ensemble_rule": "Mean of the denormalized predictions of the saved fold models.",
        "checkpoint_paths": [str(path) for path in checkpoint_paths],
    }

    output_checkpoint_root = checkpoint_root if args.checkpoint_set == "current" else checkpoint_root / "best_so_far"
    output_plot_root = plots_root if args.checkpoint_set == "current" else plots_root / "best_so_far"

    output_checkpoint_root.mkdir(parents=True, exist_ok=True)
    output_plot_root.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_checkpoint_root / "test_ensemble_predictions.npz",
        y_true=y_true.astype(np.float32),
        ensemble_prediction=ensemble_prediction.astype(np.float32),
        fold_predictions=np.stack(fold_predictions, axis=0).astype(np.float32),
    )
    (output_checkpoint_root / "test_ensemble_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    plot_test_predictions(
        test_sequence=test_sequence,
        fold_predictions=fold_predictions,
        ensemble_prediction=ensemble_prediction,
        plot_path=output_plot_root / "test_ensemble_prediction.png",
    )

    print(f"Test ensemble summary ({args.checkpoint_set})")
    print(f"Number of fold models : {len(checkpoint_paths)}")
    print(f"Ensemble test RMSE    : {metrics['ensemble_test_rmse_denormalized']:.6f}")
    print(f"Saved metrics         : {output_checkpoint_root / 'test_ensemble_metrics.json'}")

    return metrics


if __name__ == "__main__":
    main()
