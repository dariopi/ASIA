"""Generate Monte Carlo boxplot figure — val (normalized) + test (physical units)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Load data ─────────────────────────────────────────────────────────────────
mc_tsv = Path(__file__).resolve().parent / "mc_results.tsv"
runs = []
with open(mc_tsv) as fh:
    next(fh)  # skip header
    for line in fh:
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        val, test = float(parts[1]), float(parts[2])
        if val > 0 and test > 0 and test < 1.0:   # exclude outliers
            runs.append((val, test))

val_norm  = np.array([r[0] for r in runs])
test_phys = np.array([r[1] for r in runs])

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7.085, 2.3))

bp_style = dict(vert=True, patch_artist=True, widths=0.5)

axes[0].boxplot(val_norm, **bp_style,
    boxprops=dict(facecolor="#dbe9f6", color="#3182bd"),
    medianprops=dict(color="#08519c", linewidth=2.5),
    whiskerprops=dict(color="#3182bd"),
    capprops=dict(color="#3182bd"),
    flierprops=dict(marker="o", color="#3182bd", markersize=6))
axes[0].set_title("Validation RMSE")
axes[0].set_ylabel("RMSE [normalized]")
axes[0].set_xticks([])
axes[0].grid(axis="y", alpha=0.3)

axes[1].boxplot(test_phys, **bp_style,
    boxprops=dict(facecolor="#fee6ce", color="#e6550d"),
    medianprops=dict(color="#a63603", linewidth=2.5),
    whiskerprops=dict(color="#e6550d"),
    capprops=dict(color="#e6550d"),
    flierprops=dict(marker="o", color="#e6550d", markersize=6))
axes[1].set_title("Test RMSE")
axes[1].set_ylabel("RMSE [V]")
axes[1].set_xticks([])
axes[1].grid(axis="y", alpha=0.3)

fig.tight_layout()
out = Path(__file__).resolve().parent / "mc_boxplots.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved → {out}")
print(f"n_runs={len(runs)}")
print(f"Val  norm: mean={val_norm.mean():.4f}  std={val_norm.std():.4f}  min={val_norm.min():.4f}  max={val_norm.max():.4f}")
print(f"Test phys: mean={test_phys.mean():.4f}  std={test_phys.std():.4f}  min={test_phys.min():.4f}  max={test_phys.max():.4f}")
