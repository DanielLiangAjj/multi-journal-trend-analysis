#!/usr/bin/env python3
"""
Figure 3: AI / digital health publication volume across three innovation eras
(2011-2025), with era bands, an inflection marker, and a growth-percentage
annotation.

This script focuses solely on Figure 3, applying QA polish fixes:
  (a) Move the +662% growth annotation out of the LLM era band so it no longer
      collides with the era-band header label.
  (b) Render the inflection label horizontally above the chart at x=2017.5
      (instead of cramped vertical text against the dashed line).
  (c) Lift the era-band labels into the white space above the highest bar so
      they don't compete with the bar value labels.
  (d) Use proper Unicode glyphs (U+2192 RIGHTWARDS ARROW) in the annotation.

Output: figures/figure3_volume_eras_annotated.png (300 dpi, bbox_inches='tight')
"""

from __future__ import annotations

import os
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ---------------------------------------------------------------------------
PROJECT_ROOT = "/Users/danielliang/Library/CloudStorage/Dropbox/multi_journal_trend_analysis"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
CORPUS = os.path.join(DATA_DIR, "pubmed_all_journals_2011_2025_w_keywords_patched.csv")

YEARS = list(range(2011, 2026))


def compute_year_counts() -> list[int]:
    print("[F3] Computing per-year totals from corpus ...")
    year_counts: Counter[int] = Counter()
    for chunk in pd.read_csv(CORPUS, usecols=["year"], chunksize=200_000):
        chunk = chunk.dropna()
        chunk = chunk[chunk["year"].between(2011, 2025)]
        for y, c in chunk["year"].astype(int).value_counts().items():
            year_counts[y] += c
    counts = [year_counts[y] for y in YEARS]
    for y, c in zip(YEARS, counts):
        print(f"   {y}: {c}")
    return counts


def build_figure(counts: list[int], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    # ----- era bands ------------------------------------------------------
    eras = [
        (2010.5, 2015.5, "#fff3b0", "Genomics era"),
        (2015.5, 2020.5, "#d8f3dc", "Machine learning surge"),
        (2020.5, 2025.5, "#cfe2f3", "LLM / Explainable AI era"),
    ]

    # Give the y-axis enough headroom to host both era-band labels (above
    # the bars) and the inflection label (above the era labels) without
    # any of them overlapping bar value labels.
    ymax = max(counts) * 1.42
    for x0, x1, color, _label in eras:
        ax.axvspan(x0, x1, color=color, alpha=0.55, zorder=0)

    # ----- bars + connecting line -----------------------------------------
    ax.bar(
        YEARS, counts,
        color="#4169e1", alpha=0.85,
        edgecolor="#1f3b8b", linewidth=0.6, zorder=2,
    )
    ax.plot(
        YEARS, counts, "o-",
        color="#0d1b4d", lw=1.6, markersize=4, zorder=3,
    )

    # bar value labels just above each bar
    for x, y in zip(YEARS, counts):
        ax.text(
            x, y + ymax * 0.01, f"{y:,}",
            ha="center", va="bottom",
            fontsize=8, color="#222222",
            zorder=4,
        )

    # ----- inflection marker (vertical dashed line at 2017.5) -------------
    ax.axvline(2017.5, color="#b30000", linestyle="--", lw=1.6, zorder=4)

    # FIX (b): horizontal inflection label placed in the headroom ABOVE the
    # chart area, anchored at x=2017.5. Sits below the era labels.
    ax.text(
        2017.5, ymax * 0.84,
        "Inflection: deep-learning surge",
        color="#b30000", fontsize=10, fontweight="bold",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.30",
                  facecolor="white", edgecolor="#b30000", lw=1.0, alpha=0.95),
        zorder=5,
    )

    # ----- era labels (FIX c): pushed up into the headroom ABOVE the
    # tallest bar so they don't collide with the per-bar value labels.
    for x0, x1, _color, label in eras:
        ax.text(
            (x0 + x1) / 2, ymax * 0.985, label,
            ha="center", va="top",
            fontsize=11, fontweight="bold",
            color="#333333",
            zorder=4,
        )

    # ----- +X% growth annotation (FIX a + d) ------------------------------
    # Move the annotation OUT of the LLM/Explainable-AI era band by anchoring
    # the text in the lower-right corner (well below the era-band header).
    # Use a Unicode rightwards arrow in the (1,494 -> 11,384) text.
    pct_growth = (counts[-1] - counts[0]) / counts[0] * 100
    arrow = FancyArrowPatch(
        (2025, counts[-1] * 0.98),       # tail: top of 2025 bar
        (2024.0, counts[-1] * 0.55),     # head: above annotation box
        arrowstyle="->", color="#b30000",
        lw=1.4, mutation_scale=14,
        zorder=5,
    )
    ax.add_patch(arrow)
    ax.text(
        2024.6, counts[-1] * 0.45,
        f"+{pct_growth:.0f}% from 2011\n({counts[0]:,} → {counts[-1]:,})",
        fontsize=10, color="#b30000", fontweight="bold",
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.30",
                  facecolor="white", edgecolor="#b30000", lw=1.0, alpha=0.95),
        zorder=6,
    )

    # ----- axes / title ---------------------------------------------------
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of articles", fontsize=12)
    ax.set_title(
        "AI / digital health publication volume across three innovation eras (2011-2025)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(YEARS)
    ax.set_xticklabels(YEARS, rotation=0)
    ax.set_xlim(2010.5, 2025.5)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", alpha=0.3, zorder=1)
    ax.set_axisbelow(True)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[F3] Saved {out_path}")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    counts = compute_year_counts()
    out = os.path.join(FIG_DIR, "figure3_volume_eras_annotated.png")
    build_figure(counts, out)
