"""
Compute per-group MAE at h=1, 10, 50 and cumulative h=1:50,
in physical (denormalized) units — same format as Busetto et al. (2026).

Models evaluated:
  - Naïve  : predict y0 for all steps
  - AR best: autoresearch ensemble (checkpoints/best_so_far/)
  - RS best: random search best config (checkpoints/config_0026/)

Usage:
    cd /home/dpiga/Benchmark_Autoresearch/Nanodrone
    python final_analysis/paper_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR   = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import importlib.util as _ilu
import prepare
import test as test_eval

# Load random_search/model.py explicitly to avoid name clash with main model.py
_rs_spec = _ilu.spec_from_file_location("rs_model", REPO_ROOT / "random_search" / "model.py")
rs_model_module = _ilu.module_from_spec(_rs_spec)
_rs_spec.loader.exec_module(rs_model_module)

SEQ_LEN = 50
HORIZONS = [1, 10, 50]

GROUPS = {
    "positions":          ["x", "y", "z"],
    "linear_velocities":  ["vx", "vy", "vz"],
    "angular_positions":  ["roll", "pitch", "yaw"],   # geodesic handled separately
    "angular_velocities": ["wx", "wy", "wz"],
}
GROUP_SHORT = {
    "positions":          "e_p [m]",
    "linear_velocities":  "e_v [m/s]",
    "angular_positions":  "e_R [rad]",
    "angular_velocities": "e_w [rad/s]",
}
GROUP_LABELS = {
    "positions":          r"$\bar{e}_{p,h}$ [m]",
    "linear_velocities":  r"$\bar{e}_{v,h}$ [m/s]",
    "angular_positions":  r"$\bar{e}_{R,h}$ [rad]  (geodesic)",
    "angular_velocities": r"$\bar{e}_{\omega,h}$ [rad/s]",
}

OUT_IDX = {name: i for i, name in enumerate(prepare.output_names)}


# ---------------------------------------------------------------------------
# Sliding-window sequence (stride=1, raw physical units)
# ---------------------------------------------------------------------------

def sliding_windows(data_dir: Path, name: str) -> prepare.DroneSequence:
    import pandas as pd
    u_list, y_list, y0_list = [], [], []
    for fname in sorted(data_dir.iterdir()):
        if fname.name.startswith(name) and fname.suffix == ".csv":
            df = pd.read_csv(fname)
            u_vals = df[prepare.input_names].values.astype(np.float32)
            y_vals = df[prepare.output_names].values.astype(np.float32)
            n = max(0, len(df) - SEQ_LEN)
            u_w  = np.empty((n, SEQ_LEN, len(prepare.input_names)),  dtype=np.float32)
            y_w  = np.empty((n, SEQ_LEN, len(prepare.output_names)), dtype=np.float32)
            y0_w = np.empty((n, 1,       len(prepare.output_names)), dtype=np.float32)
            for i in range(n):
                y0_w[i, 0] = y_vals[i]
                u_w[i]     = u_vals[i + 1: i + 1 + SEQ_LEN]
                y_w[i]     = y_vals[i + 1: i + 1 + SEQ_LEN]
            u_list.append(u_w); y_list.append(y_w); y0_list.append(y0_w)
    return prepare.DroneSequence(
        name=name,
        u=torch.from_numpy(np.concatenate(u_list,  axis=0)),
        y=torch.from_numpy(np.concatenate(y_list,  axis=0)),
        y0=torch.from_numpy(np.concatenate(y0_list, axis=0)),
    )


# ---------------------------------------------------------------------------
# MAE per horizon
# ---------------------------------------------------------------------------

def euler_to_rotmat(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Convert ZYX Euler angles to rotation matrices. Input shape (...,), output (..., 3, 3)."""
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    R = np.stack([
        np.stack([ cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr], axis=-1),
        np.stack([ sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr], axis=-1),
        np.stack([-sp,     cp*sr,             cp*cr            ], axis=-1),
    ], axis=-2)  # (..., 3, 3)
    return R


def geodesic_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Geodesic rotation error on SO(3) at each horizon h.
    y_true, y_pred: (N, H, 12) — attitude at indices 6,7,8 (roll,pitch,yaw).
    Returns array of shape (H,).
    """
    roll_t, pitch_t, yaw_t = y_true[..., 6], y_true[..., 7], y_true[..., 8]
    roll_p, pitch_p, yaw_p = y_pred[..., 6], y_pred[..., 7], y_pred[..., 8]
    R_t = euler_to_rotmat(roll_t, pitch_t, yaw_t)   # (N, H, 3, 3)
    R_p = euler_to_rotmat(roll_p, pitch_p, yaw_p)
    # d(R_t, R_p) = arccos((tr(R_t^T R_p) - 1) / 2)
    RtRp  = np.einsum("...ji,...jk->...ik", R_t, R_p)   # R_t^T @ R_p
    trace = RtRp[..., 0, 0] + RtRp[..., 1, 1] + RtRp[..., 2, 2]
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)   # (N, H)
    return angle.mean(axis=0)      # (H,)


def prediction_errors(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, np.ndarray]:
    """
    Mean Euclidean prediction error per group at each horizon h.
    Attitude uses geodesic distance on SO(3). Shape of each value: (SEQ_LEN,).
    """
    result = {}
    for group, outputs in GROUPS.items():
        if group == "angular_positions":
            result[group] = geodesic_error(y_true, y_pred)
        else:
            idxs = [OUT_IDX[o] for o in outputs]
            diff = y_true[:, :, idxs] - y_pred[:, :, idxs]   # (N, H, 3)
            l2   = np.linalg.norm(diff, axis=2)               # (N, H)
            result[group] = l2.mean(axis=0)                   # (H,)
    return result


def naive_curves(seq: prepare.DroneSequence) -> dict[str, np.ndarray]:
    y_true  = seq.y.numpy()
    y0      = seq.y0.numpy()
    y_naive = np.repeat(y0, SEQ_LEN, axis=1)
    return prediction_errors(y_true, y_naive)


def ensemble_curves(ckpt_dir: Path, seq: prepare.DroneSequence,
                    gc: dict) -> dict[str, np.ndarray]:
    paths = test_eval.load_checkpoint_paths(ckpt_dir.parent, ckpt_dir.name)
    preds = [test_eval.predict_with_checkpoint(p, seq, gc) for p in paths]
    y_ens  = np.mean(np.stack(preds, axis=0), axis=0)
    return prediction_errors(seq.y.numpy(), y_ens)


def rs_predict(ckpt_path: Path, seq: prepare.DroneSequence, device: str) -> np.ndarray:
    """Load a RS checkpoint and return denormalized predictions (N, H, 12)."""
    ck = torch.load(ckpt_path, map_location=device)
    normalizer = prepare.Normalizer.from_state_dict(ck["normalizer"])
    cfg = ck["model_config"]
    gc  = ck["general_config"]
    model = rs_model_module.build_model_from_config(
        cfg, gc["n_inputs"], gc["n_states"], gc["n_outputs"]
    )
    model.load_state_dict(ck["model_state_dict"])
    model.eval().to(device)
    seq_norm = normalizer.normalize_sequence(seq)
    with torch.no_grad():
        y_hat_norm, _ = model(seq_norm.u.to(device), seq_norm.y0.to(device))
    return normalizer.denormalize_y_tensor(y_hat_norm).cpu().numpy()


def rs_ensemble_curves(ckpt_root: Path, seq: prepare.DroneSequence,
                       device: str) -> dict[str, np.ndarray]:
    paths = sorted(ckpt_root.glob("*/model.pt"))
    preds = [rs_predict(p, seq, device) for p in paths]
    y_ens = np.mean(np.stack(preds, axis=0), axis=0)
    return prediction_errors(seq.y.numpy(), y_ens)


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def print_table(models: dict[str, dict[str, np.ndarray]]) -> None:
    h1, h10, h50 = 0, 9, 49   # 0-based indices

    header = f"{'Model':<12}"
    for g in GROUPS:
        s = GROUP_SHORT[g]
        header += f"  {s:>30}"
    print(header)

    subheader = f"{'':12}"
    for _ in GROUPS:
        subheader += f"  {'h=1':>7}  {'h=10':>7}  {'h=50':>7}  {'h=1:50':>7}"
    print(subheader)
    print("-" * (12 + 4 * (4 * 9 + 2)))

    for model_name, curves in models.items():
        row = f"{model_name:<12}"
        for g in GROUPS:
            c = curves[g]
            cumul = float(c.sum())
            row += f"  {c[h1]:7.4f}  {c[h10]:7.4f}  {c[h50]:7.4f}  {cumul:7.4f}"
        print(row)


# ---------------------------------------------------------------------------
# Figure (same layout as Busetto et al.)
# ---------------------------------------------------------------------------

COLORS = {
    "Naïve":   ("#d62728", "--"),
    "AR best": ("#2ca02c", "-"),
    "RS best": ("#1f78b4", "-"),
}

def make_figure(models: dict[str, dict[str, np.ndarray]], out_path: Path) -> None:
    h = np.arange(1, SEQ_LEN + 1)
    groups = list(GROUPS.keys())
    fig, axes = plt.subplots(1, len(groups),
                             figsize=(len(groups) * 55 / 25.4, 70 / 25.4),
                             constrained_layout=True)
    fig.patch.set_facecolor("#ffffff")

    for ax, group in zip(axes, groups):
        ax.set_facecolor("#ffffff")
        for model_name, curves in models.items():
            color, ls = COLORS.get(model_name, ("#888888", "-"))
            ax.plot(h, curves[group], color=color, lw=1.8, ls=ls, label=model_name)
        ax.set_xlabel(r"$h$ [-]", fontsize=9)
        ax.set_ylabel(GROUP_LABELS[group], fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(1, SEQ_LEN)
        ax.set_ylim(bottom=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(models),
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, 1.10))

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _, _, gc = prepare.load_datasets_and_config()

    print("Loading stride-1 test windows (Melon, physical units)...")
    seq = sliding_windows(prepare.TEST_DATA_DIR, "melon")
    print(f"  {seq.u.shape[0]} windows  (stride=1, H={SEQ_LEN})")

    models: dict[str, dict[str, np.ndarray]] = {}

    print("Computing Naïve baseline...")
    models["Naïve"] = naive_curves(seq)

    print("Computing AR best ensemble...")
    ar_dir = REPO_ROOT / "checkpoints" / "best_so_far"
    models["AR best"] = ensemble_curves(ar_dir, seq, gc)

    print("Computing RS best ensemble...")
    rs_dir = REPO_ROOT / "random_search" / "checkpoints" / "config_0026"
    models["RS best"] = rs_ensemble_curves(rs_dir, seq, gc.get("device", "cpu"))


    print("\n" + "=" * 80)
    print_table(models)
    print("=" * 80)

    make_figure(models, OUT_DIR / "paper_metrics_horizon.png")


if __name__ == "__main__":
    main()
