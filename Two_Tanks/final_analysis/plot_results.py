from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

IEEE_SINGLE_COLUMN_WIDTH_IN = 3.5


@dataclass
class ResultRow:
    iteration: int
    commit: str
    val_rmse: float
    description: str
    status: str


def read_results(path: Path) -> list[ResultRow]:
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")

    rows: list[ResultRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for index, row in enumerate(reader, start=1):
            raw_val = str(row.get("val_RMSE", "")).strip()
            if not raw_val:
                continue
            try:
                val_rmse = float(raw_val)
            except ValueError:
                continue

            rows.append(
                ResultRow(
                    iteration=index,
                    commit=str(row.get("commit", "")).strip(),
                    val_rmse=val_rmse,
                    description=str(row.get("description", "")).strip(),
                    status=str(row.get("status", "")).strip(),
                )
            )

    return rows


def short_label(description: str) -> str:
    """Return a concise, human-readable tag for a new-best iteration."""
    desc = description.lower()
    if "baseline" in desc:
        return "RNN baseline\n(4 hidden units)"
    if "gru" in desc and "256" in desc and "adamw" in desc:
        return "GRU-256\nAdamW + Cosine"
    if "gru" in desc and "256" in desc:
        return "GRU-256\nTBPTT"
    if "gru" in desc and "128" in desc:
        return "GRU-128\nTBPTT"
    if "gru" in desc:
        return "GRU"
    if "lstm" in desc and "256" in desc:
        return "LSTM-256\n3-layer"
    if "lstm" in desc and "128" in desc:
        return "LSTM-128\nTBPTT"
    if "lstm" in desc:
        return "LSTM"
    if "ltc" in desc or "cfc" in desc:
        return "LTC/CfC"
    if "echo" in desc or "reservoir" in desc:
        return "EchoState"
    if "transformer" in desc:
        return "Transformer"
    if "koopman" in desc:
        return "Koopman"
    if "neural" in desc and "ode" in desc:
        return "NeuralODE"
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", description)
    return " ".join(tokens[:2]) if tokens else "best"


def running_best(values: list[float]) -> list[float]:
    best_values: list[float] = []
    current_best = math.inf
    for value in values:
        if value < current_best:
            current_best = value
        best_values.append(current_best)
    return best_values


def plot_results(rows: list[ResultRow], output_path: Path) -> None:
    if not rows:
        raise ValueError("results.tsv does not contain any numeric rows to plot.")

    iterations = [row.iteration for row in rows]
    values = [row.val_rmse for row in rows]
    best_values = running_best(values)

    record_points: list[tuple[int, float, str]] = []
    current_best = math.inf
    for row in rows:
        if row.val_rmse < current_best:
            current_best = row.val_rmse
            record_points.append((row.iteration, row.val_rmse, short_label(row.description)))

    W_mm, H_mm = 180, 72
    fig, ax = plt.subplots(
        figsize=(W_mm / 25.4, H_mm / 25.4),
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#f6f7fb")
    ax.set_facecolor("#ffffff")

    ax.scatter(
        iterations,
        values,
        s=22,
        color="#fc8d59",
        alpha=0.65,
        edgecolors="none",
        label="Validation RMSE",
        zorder=2,
    )
    ax.step(
        iterations,
        best_values,
        where="post",
        color="#d95f02",
        linewidth=2.0,
        label="best so far",
        zorder=3,
    )
    ax.scatter(
        [item[0] for item in record_points],
        [item[1] for item in record_points],
        s=32,
        color="#a50f15",
        edgecolors="white",
        linewidths=0.7,
        zorder=4,
    )

    # Offsets tuned for gen09: baseline (high → below), LSTM-128 (mid → above),
    # GRU-128 (low → above-right), GRU-256 (lowest → below).
    _offsets = [(4, -44), (4, 18), (4, 18), (4, 18)]
    for idx, (iteration, value, label) in enumerate(record_points):
        ox, oy = _offsets[idx] if idx < len(_offsets) else (5, 10)
        ax.annotate(
            label,
            xy=(iteration, value),
            xytext=(ox, oy),
            textcoords="offset points",
            fontsize=7.5,
            color="#7f1d1d",
            multialignment="center",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#f3c7b0", alpha=0.95),
            arrowprops=dict(arrowstyle="-", color="#f0a27a", lw=0.7, alpha=0.85),
            zorder=5,
        )

    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel("Validation RMSE", fontsize=10)
    ax.grid(True, which="major", alpha=0.18, linewidth=0.8)
    ax.tick_params(labelsize=9)

    ymin = min(values + best_values)
    ymax = max(values + best_values)
    margin_lo = max(0.005, 0.02 * (ymax - ymin))
    margin_hi = max(0.03, 0.06 * (ymax - ymin))
    ax.set_ylim(ymin - margin_lo, ymax + margin_hi)
    ax.set_xlim(0.5, len(rows) + 0.5)

    ax.legend(fontsize=8, framealpha=0.9, loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=400)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the evolution of results.tsv.")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results.tsv",
        help="Path to the tab-separated results log.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().with_name("cascaded_final_analysis.png"),
        help="Path of the PNG plot to write.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_results(args.results)
    plot_results(rows, args.output)


if __name__ == "__main__":
    main()
