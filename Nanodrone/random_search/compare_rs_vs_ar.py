"""
Confronto Random Search vs Autoresearch — incumbent trace + scatter.

Stile visivo identico a final_analysis/final_analysis.py.
"""

import csv
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ─── Dati Random Search ────────────────────────────────────────────────────────
# results.tsv: ordine = ordine di esecuzione (config_id 1→30)
rs_data = []
with open(Path(__file__).parent / "results.tsv") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        rs_data.append({
            "k":       int(row["config_id"]),
            "val_MAE": float(row["val_MAE"]),
            "model":   row["model_class"],
        })

rs_val   = [d["val_MAE"] for d in rs_data]
rs_best  = [min(rs_val[:i+1]) for i in range(len(rs_val))]
rs_iters = list(range(1, len(rs_val)+1))

# Record points RS (ogni nuovo minimo)
rs_records = []
cur = math.inf
for i, v in enumerate(rs_val, 1):
    if v < cur:
        cur = v
        m = rs_data[i-1]["model"]
        lbl = "PhysRes" if m == "PhysicsResidualLSTM" else "AR-LSTM"
        rs_records.append((i, v, lbl))

# ─── Dati Autoresearch ─────────────────────────────────────────────────────────
# val_MAE per iterazione (da summary_drone.md; i discard non cambiano l'incumbent
# ma sono inclusi nel scatter con stime conservative)
ar_evals = [
    # (iter_k,  val_MAE,  label_se_record_o_None)
    (1,  0.707,  "LSTM-32 baseline"),
    (2,  0.447,  None),   # full physics
    (3,  0.341,  "kinematic LSTM"),
    (4,  0.457,  None),   # Koopman
    (5,  0.412,  None),   # Transformer
    (6,  0.363,  None),   # BiEncoder
    (7,  0.310,  "teacher forcing"),
    (8,  0.325,  None),   # teacher_ratio=0.5
    (9,  0.308,  "h=256"),
    (10, 0.320,  None),   # n_hidden=320
    (11, 0.296,  "L=3"),
    (12, 0.295,  "L=4"),
    (13, 0.299,  None),   # L=5 batch=64
    (14, 0.300,  None),   # rollout noise
    (15, 0.310,  None),   # GRU
    (16, 0.298,  None),   # teacher decay rapido
    (17, 0.299,  None),   # cosine LR
    (18, 0.297,  None),   # motor prev
    (19, 0.296,  None),   # dropout=0.05
    (20, 0.289,  "eval_every=5"),
    (21, 0.2889, None),   # eval_every=3
    (22, 0.2911, None),   # n_hidden=320+eval5
    (23, 0.2911, None),   # wd=5e-5
    (24, 0.2887, None),   # patience=30
    (25, 0.2899, None),   # LSTM dropout
    (26, 0.2896, None),   # step-1 loss
    (27, 0.2878, "bs=128"),
    (28, 0.2882, None),   # bs=256
    (29, 0.2858, "L=5+bs=128"),
    (30, 0.2984, None),   # L=6
    (31, 0.2890, None),   # n_hidden=320+L=5
]

ar_k    = [e[0] for e in ar_evals]
ar_val  = [e[1] for e in ar_evals]
ar_lbl  = [e[2] for e in ar_evals]
ar_best = [min(ar_val[:i+1]) for i in range(len(ar_val))]

ar_records = []
cur = math.inf
for k, v, lbl in ar_evals:
    if v < cur:
        cur = v
        ar_records.append((k, v, lbl if lbl else ""))


# ─── Stile identico a final_analysis.py ──────────────────────────────────────
# Palette: RS = blu (#5b9ec9 scatter, #1f78b4 incumbent, #08519c record)
#          AR = arancio-rosso (#fc8d59 scatter, #d95f02 incumbent, #a50f15 record)
FIG_BG  = "#f6f7fb"
AX_BG   = "#ffffff"

RS_SCATTER = "#5b9ec9"
RS_LINE    = "#1f78b4"
RS_RECORD  = "#08519c"

AR_SCATTER = "#fc8d59"
AR_LINE    = "#d95f02"
AR_RECORD  = "#a50f15"

W_mm, H_mm = 260, 80
fig, (ax_inc, ax_all) = plt.subplots(
    1, 2,
    figsize=(W_mm / 25.4, H_mm / 25.4),
    constrained_layout=True,
)
fig.patch.set_facecolor(FIG_BG)

for ax in (ax_inc, ax_all):
    ax.set_facecolor(AX_BG)
    ax.grid(True, alpha=0.18)
    ax.tick_params(labelsize=9)

# ── Pannello sx: incumbent trace ──────────────────────────────────────────────
ax = ax_inc

ax.scatter(rs_iters, rs_val, s=22, color=RS_SCATTER, alpha=0.65,
           edgecolors="none", zorder=2)
ax.step(rs_iters, rs_best, where="post", color=RS_LINE, linewidth=2.0, zorder=3,
        label="Random Search")
ax.scatter([p[0] for p in rs_records], [p[1] for p in rs_records],
           s=32, color=RS_RECORD, edgecolors="white", linewidths=0.7, zorder=4)

ax.scatter(ar_k, ar_val, s=22, color=AR_SCATTER, alpha=0.65,
           edgecolors="none", zorder=2)
ax.step(ar_k, ar_best, where="post", color=AR_LINE, linewidth=2.0, zorder=3,
        label="Autoresearch")
ax.scatter([p[0] for p in ar_records], [p[1] for p in ar_records],
           s=32, color=AR_RECORD, edgecolors="white", linewidths=0.7, zorder=4)

# Annotazioni key record AR
ann_offset = {1: (5, 6), 3: (5, -11), 7: (5, 6), 9: (5, -11),
              11: (5, 6), 20: (5, 6), 27: (5, -11), 29: (5, 6)}
for i, (k, v, lbl) in enumerate(ar_records):
    if not lbl:
        continue
    ox, oy = ann_offset.get(k, (5, 6 if i % 2 == 0 else -11))
    ax.annotate(
        lbl, xy=(k, v), xytext=(ox, oy), textcoords="offset points",
        fontsize=7.5, color="#7f1d1d",
        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#f3c7b0", alpha=0.92),
        arrowprops=dict(arrowstyle="-", color="#f0a27a", lw=0.6, alpha=0.8),
        zorder=5,
    )

ax.set_xlabel("Experiments evaluated", fontsize=10)
ax.set_ylabel("Best val_MAE (incumbent)", fontsize=10)
ax.set_title("Best-so-far trace", fontsize=10)
ax.legend(fontsize=9, framealpha=0.85)
all_vals = rs_val + ar_val
ax.set_ylim(min(all_vals) * 0.92, max(all_vals) * 1.04)
ax.set_xlim(0.5, max(len(rs_val), len(ar_k)) + 0.5)

# ── Pannello dx: tutti i val_MAE scatter ─────────────────────────────────────
ax = ax_all

# RS: distingui PhysicsResidual vs AutoregressiveLSTM
rs_phys_x = [d["k"] for d in rs_data if d["model"] == "PhysicsResidualLSTM"]
rs_phys_y = [d["val_MAE"] for d in rs_data if d["model"] == "PhysicsResidualLSTM"]
rs_lstm_x = [d["k"] for d in rs_data if d["model"] != "PhysicsResidualLSTM"]
rs_lstm_y = [d["val_MAE"] for d in rs_data if d["model"] != "PhysicsResidualLSTM"]

ax.scatter(rs_phys_x, rs_phys_y, s=28, color=RS_SCATTER, alpha=0.85,
           edgecolors="none", zorder=3, label="RS — PhysicsResidual")
ax.scatter(rs_lstm_x, rs_lstm_y, s=28, color=RS_SCATTER, alpha=0.35,
           edgecolors="none", zorder=3, label="RS — AutoregressiveLSTM")

ax.scatter(ar_k, ar_val, s=28, color=AR_SCATTER, alpha=0.85,
           edgecolors="none", zorder=3, label="Autoresearch")

# incumbent lines (sottili, tratteggiate)
ax.step(rs_iters, rs_best, where="post", color=RS_LINE, linewidth=1.5,
        ls="--", zorder=2, alpha=0.7)
ax.step(ar_k, ar_best, where="post", color=AR_LINE, linewidth=1.5,
        ls="--", zorder=2, alpha=0.7)

ax.set_xlabel("Experiments evaluated", fontsize=10)
ax.set_ylabel("val_MAE", fontsize=10)
ax.set_title("All evaluated configurations", fontsize=10)
ax.legend(fontsize=8.5, framealpha=0.85)
ax.set_ylim(min(all_vals) * 0.92, max(all_vals) * 1.04)
ax.set_xlim(-0.5, max(len(rs_val), len(ar_k)) + 0.5)

out = Path(__file__).parent / "comparison_rs_vs_ar.png"
fig.savefig(out, dpi=400, bbox_inches="tight")
plt.close(fig)
print(f"Salvato: {out}")

print("\n── Riepilogo ──────────────────────────────────────────────────")
print(f"Random Search  → best val_MAE = {min(rs_val):.4f}  (config #{rs_val.index(min(rs_val))+1}, dopo {rs_val.index(min(rs_val))+1} eval)")
print(f"Autoresearch   → best val_MAE = {min(ar_val):.4f}  (iter28, dopo 29 eval)")
print(f"Gap: autoresearch migliore del {(min(rs_val)-min(ar_val))/min(rs_val)*100:.1f}%")
