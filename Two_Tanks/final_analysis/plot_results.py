from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


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
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", description)
    if not tokens:
        return "best"
    return " ".join(tokens[:2])


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

    fig, ax = plt.subplots(figsize=(13, 6.8), constrained_layout=True)
    fig.patch.set_facecolor("#f6f7fb")
    ax.set_facecolor("#ffffff")

    ax.scatter(
        iterations,
        values,
        s=34,
        color="#9ecae1",
        alpha=0.42,
        edgecolors="none",
        label="val_RMSE",
        zorder=2,
    )
    ax.step(
        iterations,
        best_values,
        where="post",
        color="#d95f02",
        linewidth=2.4,
        label="best so far",
        zorder=3,
    )
    ax.scatter(
        [item[0] for item in record_points],
        [item[1] for item in record_points],
        s=52,
        color="#a50f15",
        edgecolors="white",
        linewidths=0.8,
        zorder=4,
    )

    for idx, (iteration, value, label) in enumerate(record_points):
        offset_y = 11 if idx % 2 == 0 else -14
        ax.annotate(
            label,
            xy=(iteration, value),
            xytext=(8, offset_y),
            textcoords="offset points",
            fontsize=8,
            color="#7f1d1d",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#f3c7b0", alpha=0.92),
            arrowprops=dict(arrowstyle="-", color="#f0a27a", lw=0.8, alpha=0.8),
            zorder=5,
        )

    ax.set_title("Evolution of validation RMSE across iterations", fontsize=14, pad=14)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("val_RMSE")
    ax.grid(True, which="major", alpha=0.18, linewidth=0.8)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)

    ymin = min(values + best_values)
    ymax = max(values + best_values)
    margin = max(0.005, 0.02 * (ymax - ymin if ymax > ymin else 1.0))
    ax.set_ylim(ymin - margin, max(ymax + margin, 0.75))
    ax.set_xlim(0.5, len(rows) + 0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
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
        default=Path(__file__).resolve().with_name("results_plot.png"),
        help="Path of the PNG plot to write.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_results(args.results)
    plot_results(rows, args.output)


if __name__ == "__main__":
    main()
