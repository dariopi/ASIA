"""Trajectory plots: true vs predicted for AR best & RS best models.

Uses the non-overlapping test sequences (from cached data) to produce
continuous predicted trajectories, concatenated across all test runs.
MAE and R² per output are computed on these same predictions.
Stride-1 group MAE is loaded from grouped_mae.json (already computed).

Usage:
    python final_analysis/trajectory_plots.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
RS_ROOT   = REPO_ROOT / "random_search"
OUT_DIR   = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import prepare
import test as test_eval

_rs_spec = importlib.util.spec_from_file_location("rs_model", RS_ROOT / "model.py")
rs_model_module = importlib.util.module_from_spec(_rs_spec)
_rs_spec.loader.exec_module(rs_model_module)


# ---------------------------------------------------------------------------
# RS prediction helper
# ---------------------------------------------------------------------------

def predict_rs(cp: Path, seq: prepare.DroneSequence, gc: dict) -> np.ndarray:
    ck = torch.load(cp, map_location=gc["device"], weights_only=False)
    model = rs_model_module.build_model_from_config(
        config_pars=ck["model_config"],
        n_inputs=gc["n_inputs"], n_states=gc["n_states"], n_outputs=gc["n_outputs"],
    ).to(gc["device"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    normalizer = prepare.Normalizer.from_state_dict(ck["normalizer"])
    seq_norm = normalizer.normalize_sequence(seq)
    with torch.no_grad():
        y_hat_norm, _ = model(seq_norm.u.to(gc["device"]), seq_norm.y0.to(gc["device"]))
        y_hat = normalizer.denormalize_y_tensor(y_hat_norm).detach().cpu().numpy()
    return y_hat  # (N, seq_len, n_out)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def r2_per_output(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = y_true.reshape(-1, y_true.shape[-1])
    yp = y_pred.reshape(-1, y_pred.shape[-1])
    ss_res = np.sum((yt - yp) ** 2, axis=0)
    ss_tot = np.sum((yt - yt.mean(axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / np.where(ss_tot == 0, 1.0, ss_tot)


def mae_per_output(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(
        y_true.reshape(-1, y_true.shape[-1]) - y_pred.reshape(-1, y_pred.shape[-1])
    ), axis=0)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

OUTPUT_LABELS = {
    "x": "x [m]", "y": "y [m]", "z": "z [m]",
    "vx": "vx [m/s]", "vy": "vy [m/s]", "vz": "vz [m/s]",
    "roll": "roll [rad]", "pitch": "pitch [rad]", "yaw": "yaw [rad]",
    "wx": "wx [rad/s]", "wy": "wy [rad/s]", "wz": "wz [rad/s]",
}
GROUPS = {
    "x": "positions", "y": "positions", "z": "positions",
    "roll": "angular pos.", "pitch": "angular pos.", "yaw": "angular pos.",
    "vx": "linear vel.", "vy": "linear vel.", "vz": "linear vel.",
    "wx": "angular vel.", "wy": "angular vel.", "wz": "angular vel.",
}


def make_plot(y_true_flat: np.ndarray,
              ar_flat: np.ndarray,
              rs_flat: np.ndarray,
              mae_ar: np.ndarray,
              mae_rs: np.ndarray,
              r2_ar: np.ndarray,
              r2_rs: np.ndarray,
              out_path: Path) -> None:
    n_out  = len(prepare.output_names)
    n_cols, n_rows = 3, 4
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(240 / 25.4, 280 / 25.4),
                             constrained_layout=True)
    fig.patch.set_facecolor("#f6f7fb")
    axes_flat = axes.flatten()
    T = y_true_flat.shape[0]
    t = np.arange(T)

    for i, name in enumerate(prepare.output_names):
        ax = axes_flat[i]
        ax.set_facecolor("#ffffff")
        ax.plot(t, y_true_flat[:, i], color="#1d3557", lw=0.8, label="true", zorder=3)
        ax.plot(t, ar_flat[:, i], color="#e76f51", lw=0.7, alpha=0.85,
                label=f"AR  MAE={mae_ar[i]:.4f}  R²={r2_ar[i]:.3f}", zorder=2)
        ax.plot(t, rs_flat[:, i], color="#2a9d8f", lw=0.7, alpha=0.75, ls="--",
                label=f"RS  MAE={mae_rs[i]:.4f}  R²={r2_rs[i]:.3f}", zorder=2)
        ax.set_ylabel(OUTPUT_LABELS[name], fontsize=7)
        ax.set_title(GROUPS[name], fontsize=7, color="#555")
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.18)
        ax.legend(fontsize=5.5, framealpha=0.85, loc="upper right")

    for ax in axes_flat[n_out:]:
        ax.set_visible(False)
    axes_flat[-2].set_xlabel("time step", fontsize=8)
    fig.suptitle(
        "Test trajectory (melon) — 50-step prediction windows\n"
        "AR best (autoresearch iter28)  vs  RS best (config_0026)",
        fontsize=9,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _, test_sequences, gc = prepare.load_datasets_and_config()
    seq = test_sequences["melon"]
    y_true = seq.y.numpy()                          # (N, seq_len, 12)
    y_true_flat = y_true.reshape(-1, y_true.shape[-1])

    # ── AR best ───────────────────────────────────────────────────────────
    ar_ckpts = test_eval.load_checkpoint_paths(
        Path(gc["checkpoint_path"]), "best_so_far"
    )
    ar_preds  = [test_eval.predict_with_checkpoint(cp, seq, gc) for cp in ar_ckpts]
    ar_ens    = np.mean(np.stack(ar_preds, axis=0), axis=0)   # (N, seq_len, 12)
    ar_flat   = ar_ens.reshape(-1, ar_ens.shape[-1])

    # ── RS best ───────────────────────────────────────────────────────────
    rs_ckpts = sorted((RS_ROOT / "checkpoints" / "config_0026").glob("*/model.pt"))
    rs_preds  = [predict_rs(cp, seq, gc) for cp in rs_ckpts]
    rs_ens    = np.mean(np.stack(rs_preds, axis=0), axis=0)
    rs_flat   = rs_ens.reshape(-1, rs_ens.shape[-1])

    # ── Metrics ───────────────────────────────────────────────────────────
    mae_ar = mae_per_output(y_true, ar_ens)
    mae_rs = mae_per_output(y_true, rs_ens)
    r2_ar  = r2_per_output(y_true, ar_ens)
    r2_rs  = r2_per_output(y_true, rs_ens)

    # ── Print table ───────────────────────────────────────────────────────
    print(f"\n{'Output':<8}  {'AR MAE':>9}  {'AR R²':>7}  {'RS MAE':>9}  {'RS R²':>7}")
    print("-" * 50)
    for i, name in enumerate(prepare.output_names):
        print(f"{name:<8}  {mae_ar[i]:9.6f}  {r2_ar[i]:7.4f}  "
              f"{mae_rs[i]:9.6f}  {r2_rs[i]:7.4f}")
    print("-" * 50)
    print(f"{'MEAN':<8}  {mae_ar.mean():9.6f}  {r2_ar.mean():7.4f}  "
          f"{mae_rs.mean():9.6f}  {r2_rs.mean():7.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────
    make_plot(y_true_flat, ar_flat, rs_flat, mae_ar, mae_rs, r2_ar, r2_rs,
              OUT_DIR / "trajectory_50step.png")

    # ── Save JSON ─────────────────────────────────────────────────────────
    metrics = {
        "seq_len": int(y_true.shape[1]),
        "num_windows": int(y_true.shape[0]),
        "autoresearch_best": {
            "per_output": {
                name: {"mae": float(mae_ar[i]), "r2": float(r2_ar[i])}
                for i, name in enumerate(prepare.output_names)
            },
            "overall_mae": float(mae_ar.mean()),
            "overall_r2":  float(r2_ar.mean()),
        },
        "random_search_best": {
            "per_output": {
                name: {"mae": float(mae_rs[i]), "r2": float(r2_rs[i])}
                for i, name in enumerate(prepare.output_names)
            },
            "overall_mae": float(mae_rs.mean()),
            "overall_r2":  float(r2_rs.mean()),
        },
    }
    out_json = OUT_DIR / "trajectory_50step_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
