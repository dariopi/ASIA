"""Final analysis: stride-1 test evaluation + results.tsv plot.

Usage:
    python final_analysis/final_analysis.py
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import prepare
import test as test_eval

SEQ_LEN = 50
OUT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Stride-1 evaluation
# ---------------------------------------------------------------------------

def sliding_windows(data_dir: Path, name: str) -> prepare.DroneSequence:
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


def r2_per_output(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = y_true.reshape(-1, y_true.shape[-1])
    yp = y_pred.reshape(-1, y_pred.shape[-1])
    ss_res = np.sum((yt - yp) ** 2, axis=0)
    ss_tot = np.sum((yt - yt.mean(axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / np.where(ss_tot == 0, 1.0, ss_tot)


def run_stride1_eval(general_config: dict) -> dict:
    checkpoint_root = Path(general_config["checkpoint_path"])
    checkpoint_paths = test_eval.load_checkpoint_paths(checkpoint_root, "best_so_far")
    if not checkpoint_paths:
        checkpoint_paths = test_eval.load_checkpoint_paths(checkpoint_root, "current")
    if not checkpoint_paths:
        raise FileNotFoundError("No fold checkpoints found.")

    seq = sliding_windows(prepare.TEST_DATA_DIR, "melon")
    print(f"\n=== Stride-1 evaluation (melon) ===")
    print(f"Windows: {seq.u.shape[0]}  (stride=1, seq_len={SEQ_LEN})")

    fold_preds = [
        test_eval.predict_with_checkpoint(cp, seq, general_config)
        for cp in checkpoint_paths
    ]
    ensemble = np.mean(np.stack(fold_preds, axis=0), axis=0)
    y_true   = seq.y.numpy()

    mae_per_out = np.mean(np.abs(y_true - ensemble), axis=(0, 1))
    r2          = r2_per_output(y_true, ensemble)
    overall_mae = float(np.mean(mae_per_out))
    overall_r2  = float(np.mean(r2))

    print(f"\n{'Output':>8}  {'MAE':>10}  {'R²':>8}")
    print("-" * 32)
    for name, m, r in zip(prepare.output_names, mae_per_out, r2):
        print(f"{name:>8}  {m:10.6f}  {r:8.4f}")
    print("-" * 32)
    print(f"{'MEAN':>8}  {overall_mae:10.6f}  {overall_r2:8.4f}")

    metrics = {
        "num_windows": int(seq.u.shape[0]),
        "seq_len": SEQ_LEN,
        "stride": 1,
        "num_fold_models": len(checkpoint_paths),
        "overall_mae": overall_mae,
        "overall_r2": overall_r2,
        "per_output": {
            name: {"mae": float(m), "r2": float(r)}
            for name, m, r in zip(prepare.output_names, mae_per_out, r2)
        },
        "checkpoint_paths": [str(p) for p in checkpoint_paths],
    }
    out = OUT_DIR / "stride1_test_metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")
    return metrics


# ---------------------------------------------------------------------------
# Results plot
# ---------------------------------------------------------------------------

IEEE_W = 3.5


@dataclass
class ResultRow:
    iteration: int
    commit: str
    val_mae: float
    test_mae: float
    description: str
    status: str


def read_results(path: Path) -> list[ResultRow]:
    rows: list[ResultRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f, delimiter="\t"), start=1):
            try:
                val = float(row.get("val_MAE", ""))
            except (ValueError, TypeError):
                continue
            if not math.isfinite(val):
                continue
            try:
                test = float(row.get("test_MAE", ""))
            except (ValueError, TypeError):
                test = math.nan
            rows.append(ResultRow(
                iteration=i,
                commit=str(row.get("commit", "")).strip(),
                val_mae=val,
                test_mae=test,
                description=str(row.get("description", "")).strip(),
                status=str(row.get("status", "")).strip(),
            ))
    return rows


def _short_label(desc: str) -> str:
    _abbrev = {
        "num_layers": "L", "n_hidden_states": "h", "n_hidden": "h",
        "batch_size": "bs", "batch": "bs", "dropout": "do",
        "weight_decay": "wd", "eval_every": "eval@",
    }
    # 1. Extract up to 2 key=value pairs
    kv = re.findall(
        r"(num_layers|n_hidden_states|n_hidden|batch_size|batch|dropout|weight_decay|eval_every)"
        r"\s*[=:]\s*([\d.e+-]+)",
        desc,
    )
    if kv:
        parts = []
        for key, val in kv[:2]:
            short = _abbrev.get(key, key)
            parts.append(f"{short}{val}" if short.endswith("@") else f"{short}={val}")
        return " ".join(parts)
    # 2. Teacher forcing ratio
    if "teacher" in desc.lower():
        m = re.search(r"([\d.]+)\s*[-→>]+\s*([\d.]+)", desc)
        if m:
            return f"teacher {m.group(1)}→{m.group(2)}"
    # 3. Architecture keywords
    for arch in ["Transformer", "GRU", "Koopman", "BiEncoder", "physics", "kinematic"]:
        if re.search(arch, desc, re.IGNORECASE):
            m = re.search(r"(?:\w+\s+)?" + arch + r"(?:\s+\w+)?", desc, re.IGNORECASE)
            label = m.group(0).strip() if m else arch
            return label if len(label) <= 16 else arch
    # 4. Fallback
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", desc)
    return " ".join(tokens[:2]) if tokens else "—"


def _running_best(values: list[float]) -> list[float]:
    best, out = math.inf, []
    for v in values:
        if v < best:
            best = v
        out.append(best)
    return out


def plot_val_mae(rows: list[ResultRow], output_path: Path) -> None:
    iters = [r.iteration for r in rows]
    vals  = [r.val_mae  for r in rows]
    best  = _running_best(vals)

    record_pts: list[tuple[int, float, str]] = []
    cur_best = math.inf
    for r in rows:
        if r.val_mae < cur_best:
            cur_best = r.val_mae
            record_pts.append((r.iteration, r.val_mae, _short_label(r.description)))

    W_mm, H_mm = 180, 72
    fig, ax = plt.subplots(figsize=(W_mm / 25.4, H_mm / 25.4), constrained_layout=True)
    fig.patch.set_facecolor("#f6f7fb")
    ax.set_facecolor("#ffffff")

    ax.scatter(iters, vals, s=22, color="#5b9ec9", alpha=0.65, edgecolors="none", zorder=2)
    ax.step(iters, best, where="post", color="#d95f02", linewidth=2.0, zorder=3)
    ax.scatter(
        [p[0] for p in record_pts], [p[1] for p in record_pts],
        s=32, color="#a50f15", edgecolors="white", linewidths=0.7, zorder=4,
    )
    for k, (it, val, lbl) in enumerate(record_pts):
        offset_y = 7 if k % 2 == 0 else -11
        ax.annotate(
            lbl, xy=(it, val), xytext=(5, offset_y), textcoords="offset points",
            fontsize=8, color="#7f1d1d",
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#f3c7b0", alpha=0.92),
            arrowprops=dict(arrowstyle="-", color="#f0a27a", lw=0.6, alpha=0.8),
            zorder=5,
        )

    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel("Validation MAE", fontsize=10)
    ax.grid(True, alpha=0.18)
    ax.tick_params(labelsize=9)
    ax.set_ylim(min(vals) * 0.90, max(vals) * 1.05)
    ax.set_xlim(0.5, len(rows) + 0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=400)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_val_vs_test(rows: list[ResultRow], output_path: Path) -> None:
    valid = [r for r in rows if math.isfinite(r.test_mae)]
    if not valid:
        return
    vals  = np.array([r.val_mae  for r in valid])
    tests = np.array([r.test_mae for r in valid])
    iters = [r.iteration for r in valid]

    fig, ax = plt.subplots(figsize=(IEEE_W * 1.5, 2.2), constrained_layout=True)
    fig.patch.set_facecolor("#f6f7fb")
    ax.set_facecolor("#ffffff")

    sc = ax.scatter(vals, tests, c=iters, cmap="plasma", s=40,
                    edgecolors="white", linewidths=0.5, zorder=3)
    plt.colorbar(sc, ax=ax, label="iteration", pad=0.02)
    ax.set_xlabel("Validation MAE (normalized)", fontsize=8)
    ax.set_ylabel("Test MAE (physical units)", fontsize=8)
    ax.set_title("Val MAE vs Test MAE", fontsize=8.5)
    ax.grid(True, alpha=0.18)
    ax.tick_params(labelsize=7)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")


def run_results_plot() -> None:
    results_path = REPO_ROOT / "results.tsv"
    rows = read_results(results_path)
    print(f"\n=== Results plot ({len(rows)} iterations) ===")
    plot_val_mae(rows, OUT_DIR / "nanodrone_final_analysis.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _, _, general_config = prepare.load_datasets_and_config()
    run_stride1_eval(general_config)
    run_results_plot()


if __name__ == "__main__":
    main()
