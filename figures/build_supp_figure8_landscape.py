"""Supplementary Figure 8 — submission-landscape scatters (A: methodology; B: health).

Replaces the legacy pi_q6_*_submission_landscape.png panels, which carried an
internal "Q6:" analysis title, truncated topic labels, and overlapping label
text (flagged in review). Labels here are full topic names placed with a
deterministic collision-avoidance pass and kept inside the axes.

Data: data/visualizations_k100/pi_q5_{domain}_specialization.csv
Output: figures/supp_figure8_landscape_{methodology,health}.png
"""

from __future__ import annotations

import os
from pathlib import Path

import sys; sys.path.insert(0, str(Path(__file__).parent)); from _mirror import mirror_to_main
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "visualizations_k100"

DOT = "#3498db"
EDGE = "black"


def pick_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Label the story-carrying extremes: biggest topics and fewest-venue topics."""
    top_volume = df.nlargest(4, "total")
    few_venues = df.nsmallest(4, "n_journals")
    return pd.concat([top_volume, few_venues]).drop_duplicates(subset="topic")


def place_labels(ax, df, labelled):
    """Annotate with overlap avoidance: points close together get stacked offsets."""
    x_max = df["n_journals"].max()
    y_floor = np.log10(df["total"].min()) + 0.35
    placed = []  # (x, log_y_effective) after offsets, in label units
    for _, r in labelled.sort_values(["n_journals", "total"], ascending=False).iterrows():
        x, y = r["n_journals"], r["total"]
        near_right = x > x_max - 3
        ha = "right" if near_right else "left"
        dx = -10 if near_right else 10
        dy = 0
        ly = np.log10(y)
        # bottom-region labels stack upward so they never leave the axes;
        # everything else stacks downward in 15-pt steps
        up = ly < y_floor
        step = 0
        while any(abs(x - px) < 3.5 and abs(ly - ply) < 0.16 for px, ply in placed):
            step += 1
            dy += 15 if up else -15
            ly += 0.16 if up else -0.16
        placed.append((x, ly))
        ax.annotate(
            r["topic"], (x, y), xytext=(dx, dy - 4 if dy else 6),
            textcoords="offset points", fontsize=8.5, ha=ha, alpha=0.9,
            arrowprops=(dict(arrowstyle="-", alpha=0.35, lw=0.6) if step else None),
        )


def build(domain: str):
    df = pd.read_csv(DATA / f"pi_q5_{domain}_specialization.csv")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(df["n_journals"], df["total"], s=80, c=DOT,
               edgecolors=EDGE, linewidths=0.5, alpha=0.7)
    ax.set_xlabel("Number of journals publishing this topic",
                  fontsize=12)
    ax.set_ylabel("Total articles", fontsize=12)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.margins(x=0.06)
    place_labels(ax, df, pick_labels(df))
    fig.tight_layout()
    out = HERE / f"supp_figure8_landscape_{domain}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    mirror_to_main(str(out))
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    for domain in ("methodology", "health"):
        build(domain)
