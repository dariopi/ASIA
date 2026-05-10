"""
Confronto Random Search vs Autoresearch — incumbent trace (best-so-far).

Stile visivo identico a final_analysis.py.
Output: final_analysis/comparison_rs_vs_ar.png
"""

import csv
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR   = Path(__file__).resolve().parent

# ─── Dati Random Search ────────────────────────────────────────────────────────
rs_data = []
with open(REPO_ROOT / "random_search" / "results.tsv") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        rs_data.append({
            "k":       int(row["config_id"]),
            "val_MAE": float(row["val_MAE"]),
            "model":   row["model_class"],
        })

BASELINE_MAE = 0.707
rs_val   = [BASELINE_MAE] + [d["val_MAE"] for d in rs_data]
rs_best  = [min(rs_val[:i+1]) for i in range(len(rs_val))]
rs_iters = list(range(1, len(rs_val) + 1))

rs_records = []
cur = math.inf
for i, v in enumerate(rs_val, 1):
    if v < cur:
        cur = v
        rs_records.append((i, v))

# ─── Dati Autoresearch ─────────────────────────────────────────────────────────
ar_evals = [
    # (iter_k, val_MAE, annotation_label or None)
    (1,  0.707,  "LSTM-32 baseline"),
    (2,  0.447,  None),
    (3,  0.341,  "kinematic LSTM"),
    (4,  0.457,  None),
    (5,  0.412,  None),
    (6,  0.363,  None),
    (7,  0.310,  "teacher forcing"),
    (8,  0.325,  None),
    (9,  0.308,  "h=256"),
    (10, 0.320,  None),
    (11, 0.296,  "L=3"),
    (12, 0.295,  "L=4"),
    (13, 0.299,  None),
    (14, 0.300,  None),
    (15, 0.310,  None),
    (16, 0.298,  None),
    (17, 0.299,  None),
    (18, 0.297,  None),
    (19, 0.296,  None),
    (20, 0.289,  "eval_every=5"),
    (21, 0.2889, None),
    (22, 0.2911, None),
    (23, 0.2911, None),
    (24, 0.2887, None),
    (25, 0.2899, None),
    (26, 0.2896, None),
    (27, 0.2878, "bs=128"),
    (28, 0.2882, None),
    (29, 0.2858, "L=5+bs=128"),
    (30, 0.2984, None),
    (31, 0.2890, None),
]

ar_k    = [e[0] for e in ar_evals]
ar_val  = [e[1] for e in ar_evals]
ar_best = [min(ar_val[:i+1]) for i in range(len(ar_val))]

ar_records = []
cur = math.inf
for k, v, lbl in ar_evals:
    if v < cur:
        cur = v
        ar_records.append((k, v, lbl or ""))

# ─── Palette (identica a final_analysis.py) ───────────────────────────────────
FIG_BG = "#f6f7fb"
AX_BG  = "#ffffff"

RS_SCATTER = "#5b9ec9"
RS_LINE    = "#1f78b4"
RS_RECORD  = "#08519c"

AR_SCATTER = "#fc8d59"
AR_LINE    = "#d95f02"
AR_RECORD  = "#a50f15"

# ─── Figura ───────────────────────────────────────────────────────────────────
W_mm, H_mm = 180, 95          # più alta dell'originale (72mm) per dare respiro in alto
fig, ax = plt.subplots(figsize=(W_mm / 25.4, H_mm / 25.4), constrained_layout=True)
fig.patch.set_facecolor(FIG_BG)
ax.set_facecolor(AX_BG)
ax.grid(True, alpha=0.18)
ax.tick_params(labelsize=9)

# ── Random Search ─────────────────────────────────────────────────────────────
ax.scatter(rs_iters, rs_val, s=22, color=RS_SCATTER, alpha=0.65,
           edgecolors="none", zorder=2)
ax.step(rs_iters, rs_best, where="post", color=RS_LINE, linewidth=2.0, zorder=3,
        label="Random Search")
ax.scatter([p[0] for p in rs_records], [p[1] for p in rs_records],
           s=32, color=RS_RECORD, edgecolors="white", linewidths=0.7, zorder=4)

# ── Autoresearch ──────────────────────────────────────────────────────────────
ax.scatter(ar_k, ar_val, s=22, color=AR_SCATTER, alpha=0.65,
           edgecolors="none", zorder=2)
ax.step(ar_k, ar_best, where="post", color=AR_LINE, linewidth=2.0, zorder=3,
        label="ASIA")
ax.scatter([p[0] for p in ar_records], [p[1] for p in ar_records],
           s=32, color=AR_RECORD, edgecolors="white", linewidths=0.7, zorder=4)

# Annotazioni key record AR
ann_offset = {
    1:  (5,   7),
    3:  (5,  -12),
    7:  (5,   7),
    9:  (5,  -12),
    11: (5,   7),
    12: (5,  -12),
    20: (5,   7),
    27: (5,  -12),
    29: (5,   7),
}
for i, (k, v, lbl) in enumerate(ar_records):
    if not lbl:
        continue
    ox, oy = ann_offset.get(k, (5, 7 if i % 2 == 0 else -12))
    ax.annotate(
        lbl, xy=(k, v), xytext=(ox, oy), textcoords="offset points",
        fontsize=7.5, color="#7f1d1d",
        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#f3c7b0", alpha=0.92),
        arrowprops=dict(arrowstyle="-", color="#f0a27a", lw=0.6, alpha=0.8),
        zorder=5,
    )

# ── Assi ──────────────────────────────────────────────────────────────────────
all_vals = rs_val + ar_val
ax.set_xlabel("Iteration", fontsize=10)
ax.set_ylabel("Validation MAE", fontsize=10)
ax.legend(fontsize=9, framealpha=0.85, loc="upper right")
ax.set_ylim(min(all_vals) * 0.92, max(all_vals) * 1.12)   # più spazio in alto
ax.set_xlim(0.5, max(len(rs_val), len(ar_k)) + 0.5)

out = OUT_DIR / "comparison_rs_vs_ar.png"
fig.savefig(out, dpi=400, bbox_inches="tight")
plt.close(fig)
print(f"Salvato: {out}")

print("\n── Riepilogo ──────────────────────────────────────────────────")
print(f"Random Search  → best val_MAE = {min(rs_val):.4f}  "
      f"(config #{rs_val.index(min(rs_val))+1}, dopo {rs_val.index(min(rs_val))+1} eval)")
print(f"Autoresearch   → best val_MAE = {min(ar_val):.4f}  (iter28, dopo 29 eval)")
print(f"Gap: autoresearch migliore del {(min(rs_val)-min(ar_val))/min(rs_val)*100:.1f}%")
