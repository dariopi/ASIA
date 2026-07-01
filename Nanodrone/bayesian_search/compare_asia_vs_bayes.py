"""
ASIA (Autoresearch gen12) vs Bayesian Optimization (TPE) — incumbent trace + scatter.
Visual style identical to random_search/compare_rs_vs_ar.py.
"""

import csv
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent

# ─── ASIA data (results.tsv, 15 runs — gen12) ────────────────────────────────
asia_raw = [
    (1,  0.706694, "baseline"),
    (2,  0.322879, "AR-DeltaGRU"),
    (3,  0.307497, "PhysicsResidual"),
    (4,  0.310190, None),
    (5,  0.308651, None),
    (6,  0.344463, None),
    (7,  0.438221, "Koopman"),
    (8,  0.559873, "TCN"),
    (9,  2.001204, "Transformer"),   # outlier — clipped
    (10, 0.300280, "sched. sampling"),
    (11, 0.309008, None),
    (12, 0.341908, None),
    (13, 0.319838, None),
    (14, 0.304173, None),
    (15, 0.340670, None),
]

asia_k    = [e[0] for e in asia_raw]
asia_val  = [e[1] for e in asia_raw]
asia_best = [min(asia_val[:i+1]) for i in range(len(asia_val))]

asia_records = []
cur = math.inf
for k, v, lbl in asia_raw:
    if v < cur:
        cur = v
        asia_records.append((k, v, lbl or ""))

# ─── Bayesian data (from results.tsv) ────────────────────────────────────────
bayes_raw = []
with open(HERE / "results.tsv") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        bayes_raw.append((
            int(row["trial"]) + 1,        # 1-indexed
            float(row["val_MAE"]),
            row["model_class"],
        ))

bayes_k     = [e[0] for e in bayes_raw]
bayes_val   = [e[1] for e in bayes_raw]
bayes_model = [e[2] for e in bayes_raw]
bayes_best  = [min(bayes_val[:i+1]) for i in range(len(bayes_val))]

bayes_records = []
cur = math.inf
for k, v, m in bayes_raw:
    if v < cur:
        cur = v
        short = {"PhysicsResidualLSTM": "PhysRes", "ResidualLSTM": "ResLSTM",
                 "AutoregressiveLSTM": "AR-LSTM"}[m]
        bayes_records.append((k, v, short))

# ─── Colour palette ──────────────────────────────────────────────────────────
FIG_BG = "#f6f7fb"
AX_BG  = "#ffffff"

ASIA_SCATTER  = "#fc8d59"
ASIA_LINE     = "#d95f02"
ASIA_RECORD   = "#a50f15"

BAYES_SCATTER = "#5b9ec9"
BAYES_LINE    = "#1f78b4"
BAYES_RECORD  = "#08519c"

Y_LOW  = 0.275
Y_HIGH = 0.780   # Transformer (2.001) and TCN (0.560) partially visible

# ─── Figure ───────────────────────────────────────────────────────────────────
W_mm, H_mm = (2834-46)/400*25.4, (950-46)/400*25.4  # target 2834x950 px @ 400 dpi
fig, ax_inc = plt.subplots(figsize=(W_mm / 25.4, H_mm / 25.4), constrained_layout=True)
fig.patch.set_facecolor("white")
ax_inc.set_facecolor("white")
ax_inc.grid(True, alpha=0.18)
ax_inc.tick_params(labelsize=9)

asia_clipped  = [min(v, Y_HIGH * 0.995) for v in asia_val]
bayes_clipped = [min(v, Y_HIGH * 0.995) for v in bayes_val]

# ── Incumbent trace ────────────────────────────────────────────────────────────
ax = ax_inc

ax.scatter(bayes_k, bayes_clipped, s=22, color=BAYES_SCATTER, alpha=0.65,
           edgecolors="none", zorder=2)
ax.step(bayes_k, bayes_best, where="post", color=BAYES_LINE, linewidth=2.0,
        zorder=3, label="BO")
ax.scatter([p[0] for p in bayes_records], [p[1] for p in bayes_records],
           s=32, color=BAYES_RECORD, edgecolors="white", linewidths=0.7, zorder=4)

ax.scatter(asia_k, asia_clipped, s=22, color=ASIA_SCATTER, alpha=0.65,
           edgecolors="none", zorder=2)
ax.step(asia_k, asia_best, where="post", color=ASIA_LINE, linewidth=2.0,
        zorder=3, label="ASIA")
ax.scatter([p[0] for p in asia_records], [p[1] for p in asia_records],
           s=32, color=ASIA_RECORD, edgecolors="white", linewidths=0.7, zorder=4)

# ASIA annotations (significant incumbents only)
ann_cfg = {
    1:  ((5,   6), "#7f1d1d", "#f3c7b0", "#f0a27a"),
    2:  ((5,   6), "#7f1d1d", "#f3c7b0", "#f0a27a"),
    3:  ((5, -13), "#7f1d1d", "#f3c7b0", "#f0a27a"),
    10: ((5,   6), "#7f1d1d", "#f3c7b0", "#f0a27a"),
}
for k, v, lbl in asia_records:
    if k not in ann_cfg or not lbl:
        continue
    (ox, oy), fc_txt, ec_box, arr_c = ann_cfg[k]
    ax.annotate(lbl, xy=(k, v), xytext=(ox, oy), textcoords="offset points",
                fontsize=7.5, color=fc_txt,
                bbox=dict(boxstyle="round,pad=0.14", fc="white", ec=ec_box, alpha=0.92),
                arrowprops=dict(arrowstyle="-", color=arr_c, lw=0.6, alpha=0.8),
                zorder=5)


ax.set_xlabel("Iteration", fontsize=10)
ax.set_ylabel("Validation MAE", fontsize=10)
ax.set_title("")
ax.legend(fontsize=9, framealpha=0.85)
ax.set_ylim(Y_LOW, Y_HIGH)
ax.set_xlim(0.5, 15.5)
ax.set_xticks(range(1, 16))

# ─── Save ─────────────────────────────────────────────────────────────────────
out = HERE / "comparison_asia_vs_bayes.png"
fig.savefig(out, dpi=400, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

best_bayes_k = bayes_k[bayes_val.index(min(bayes_val))]
best_asia_k  = asia_k[asia_val.index(min(asia_val))]
print(f"\nASIA     best val_MAE = {min(asia_val):.4f}  (run {best_asia_k} — sched. sampling)")
print(f"Bayesian best val_MAE = {min(bayes_val):.4f}  (trial {best_bayes_k})")
print(f"Gap: ASIA better by {(min(bayes_val)-min(asia_val))/min(bayes_val)*100:.1f}%")
