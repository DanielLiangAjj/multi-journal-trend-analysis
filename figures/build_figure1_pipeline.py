"""
Figure 1: Publication-grade pipeline diagram for the multi-journal trend
analysis manuscript.

Horizontal workflow / data-flow diagram: an example article record (left) is
carried through six pipeline stages, each labelled with the data that enters
and leaves the stage, yielding an example named topic with its taxonomy
placement and 2011-2025 trend (right).

Output: figures/figure1_pipeline.png (300 dpi, bbox_inches='tight')
"""

from __future__ import annotations

import os

import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent)); from _mirror import mirror_to_main
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

# Muted, publication-friendly blues for the six main stages.
STAGE_FILL = "#2F5C8A"
STAGE_EDGE = "#1F3F5F"
STAGE_TEXT = "#FFFFFF"

# Light grey example-input / example-output cards.
PANEL_FILL = "#F2F2F2"
PANEL_EDGE = "#999999"
PANEL_TITLE = "#222222"
PANEL_TEXT = "#333333"

ARROW_COLOR = "#333333"
SPARK_COLOR = "#2F5C8A"

# ---------------------------------------------------------------------------
# Stage content: each box reads as DATA-IN -> DATA-OUT, no novelty framing.
# ---------------------------------------------------------------------------

STAGES = [
    {
        "title": "1. Data\n    Collection",
        "subtext": (
            "IN: 29 PubMed\njournals, 2011-2025\n\n"
            "OUT: 78,425 articles\n90,386 author keywords"
        ),
    },
    {
        "title": "2. Keyword\n    Categorization",
        "subtext": (
            "IN: 90,386 keywords\n"
            "(BiomedBERT + GPT-5)\n\n"
            "OUT: methodology vs.\nhealth labels"
        ),
    },
    {
        "title": "3. Topic\n    Clustering",
        "subtext": (
            "IN: keyword\nembeddings\n"
            "K-means (K=100,\nsilhouette)\n\n"
            "OUT: 100 clusters"
        ),
    },
    {
        "title": "4. Topic\n    Naming",
        "subtext": (
            "IN: 100 clusters\n"
            "(GPT-5)\n\n"
            "OUT: 100 named\ntopics"
        ),
    },
    {
        "title": "5. Hierarchy\n    Construction",
        "subtext": (
            "IN: 100 named topics\n"
            "Ward linkage\n\n"
            "OUT: topic taxonomy\ntree"
        ),
    },
    {
        "title": "6. Trend\n    Analysis",
        "subtext": (
            "IN: yearly topic\ncounts\n"
            "Poisson, Holm-\nBonferroni\n\n"
            "OUT: trends +\np-values"
        ),
    },
]

# ---------------------------------------------------------------------------
# Geometry (data units; aspect = "auto" so x and y scale independently)
# ---------------------------------------------------------------------------

FIG_W, FIG_H = 21.0, 6.0  # inches

CANVAS_W = 21.0
CANVAS_H = 6.0

N_STAGES = len(STAGES)
LEFT_PAD = 0.20
RIGHT_PAD = 0.20
GAP = 0.28  # space between adjacent stage boxes (for arrows)

# Example-input / example-output side panels.
PANEL_W = 2.85
PANEL_GAP = 0.55  # gap between a side panel and the stage row

# Stage row geometry (shrunk to leave room for the two side panels).
BOX_W = (
    CANVAS_W - LEFT_PAD - RIGHT_PAD - 2 * (PANEL_W + PANEL_GAP) - GAP * (N_STAGES - 1)
) / N_STAGES
BOX_H = 3.05
BOX_Y = 1.55  # bottom of main row

# Reserved vertical band at top for the figure title.
TITLE_Y = CANVAS_H - 0.25


def stage_x(i: int) -> float:
    """Left-x of stage i (0-indexed); stages start after the left panel."""
    return LEFT_PAD + PANEL_W + PANEL_GAP + i * (BOX_W + GAP)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def draw_stage_box(ax, x: float, y: float, w: float, h: float, title: str, subtext: str):
    """Draw a rounded rectangle stage box with title and data-in/out subtext."""
    box = FancyBboxPatch(
        (x + 0.02, y + 0.02),
        w - 0.04,
        h - 0.04,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.2,
        facecolor=STAGE_FILL,
        edgecolor=STAGE_EDGE,
        zorder=2,
    )
    ax.add_patch(box)

    title_top_y = y + h - 0.18
    ax.text(
        x + 0.14,
        title_top_y,
        title,
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        color=STAGE_TEXT,
        zorder=3,
    )

    ax.text(
        x + 0.14,
        y + h - 0.92,
        subtext,
        ha="left",
        va="top",
        fontsize=8.0,
        color=STAGE_TEXT,
        linespacing=1.25,
        zorder=3,
    )


def draw_example_panel(ax, x: float, y: float, w: float, h: float, title: str, body: str):
    """Draw a light-grey example-input / example-output card."""
    box = FancyBboxPatch(
        (x + 0.02, y + 0.02),
        w - 0.04,
        h - 0.04,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.0,
        facecolor=PANEL_FILL,
        edgecolor=PANEL_EDGE,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.16,
        y + h - 0.16,
        title,
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color=PANEL_TITLE,
        zorder=3,
    )
    ax.text(
        x + 0.16,
        y + h - 0.62,
        body,
        ha="left",
        va="top",
        fontsize=7.6,
        color=PANEL_TEXT,
        linespacing=1.32,
        zorder=3,
    )


def draw_horizontal_arrow(ax, x_from: float, x_to: float, y: float):
    arrow = FancyArrowPatch(
        (x_from, y),
        (x_to, y),
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.6,
        color=ARROW_COLOR,
        zorder=1,
    )
    ax.add_patch(arrow)


def draw_trend_sparkline(ax, x0: float, x1: float, y0: float, height: float):
    """Draw a small ascending trend sparkline (illustrative) on the main axes."""
    counts = [1, 2, 3, 4, 5, 7, 9, 12, 15, 18, 22, 27, 31, 35, 40]
    n = len(counts)
    cmin, cmax = min(counts), max(counts)
    xs = [x0 + (x1 - x0) * k / (n - 1) for k in range(n)]
    ys = [y0 + height * (c - cmin) / (cmax - cmin) for c in counts]
    ax.fill_between(xs, [y0] * n, ys, color=SPARK_COLOR, alpha=0.15, zorder=3)
    ax.plot(xs, ys, color=SPARK_COLOR, lw=1.6, zorder=4)


# ---------------------------------------------------------------------------
# Build the figure
# ---------------------------------------------------------------------------


def build_figure(out_path: str):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(0, CANVAS_H)
    ax.set_aspect("auto")  # let x and y scale independently
    ax.axis("off")

    # --- Figure title (neutral: workflow + data flow, no novelty framing) ---
    ax.text(
        CANVAS_W / 2,
        TITLE_Y,
        "Workflow and data flow of the multi-journal research-trend analysis pipeline (2011-2025)",
        ha="center",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        color="#222222",
    )

    # --- Example INPUT panel (left edge) -----------------------------------
    draw_example_panel(
        ax,
        x=LEFT_PAD,
        y=BOX_Y,
        w=PANEL_W,
        h=BOX_H,
        title="Example input (one article)",
        body=(
            'Title: "A transformer-based\n'
            "model for predicting 30-day\n"
            "hospital readmission in\n"
            "heart-failure patients\n"
            'from EHR notes"\n\n'
            "Author keywords:\n"
            " • deep learning\n"
            " • electronic health records\n"
            " • hospital readmission\n"
            " • heart failure"
        ),
    )
    # Arrow: input panel -> stage 1
    draw_horizontal_arrow(
        ax, LEFT_PAD + PANEL_W + 0.04, stage_x(0) + 0.02, BOX_Y + BOX_H / 2
    )

    # --- Main horizontal stage row -----------------------------------------
    for i, stage in enumerate(STAGES):
        x = stage_x(i)
        draw_stage_box(
            ax,
            x=x,
            y=BOX_Y,
            w=BOX_W,
            h=BOX_H,
            title=stage["title"],
            subtext=stage["subtext"],
        )

        # Arrow to the next stage (if any).
        if i < N_STAGES - 1:
            x_from = x + BOX_W - 0.02
            x_to = stage_x(i + 1) + 0.02
            draw_horizontal_arrow(ax, x_from, x_to, BOX_Y + BOX_H / 2)

    # --- Example OUTPUT panel (right edge) ---------------------------------
    out_x = stage_x(N_STAGES - 1) + BOX_W + PANEL_GAP
    draw_example_panel(
        ax,
        x=out_x,
        y=BOX_Y,
        w=PANEL_W,
        h=BOX_H,
        title="Example output (one topic)",
        body=(
            "Named topic:\n"
            ' "Clinical Predictive\n'
            '  Modeling from EHR Data"\n\n'
            "Taxonomy placement:\n"
            " Machine Learning\n"
            "  └ Clinical Prediction\n"
            "     └ EHR-based Risk Models\n\n"
            "Trend 2011–2025:"
        ),
    )
    # Arrow: stage 6 -> output panel
    draw_horizontal_arrow(
        ax, stage_x(N_STAGES - 1) + BOX_W - 0.02, out_x + 0.04, BOX_Y + BOX_H / 2
    )
    # Trend sparkline + label inside the output panel.
    draw_trend_sparkline(ax, out_x + 0.25, out_x + PANEL_W - 0.30, BOX_Y + 0.55, 0.55)
    ax.text(
        out_x + 0.18,
        BOX_Y + 0.34,
        "↑ increasing  (p < 0.001)",
        ha="left",
        va="bottom",
        fontsize=7.6,
        color=SPARK_COLOR,
        fontweight="bold",
        zorder=4,
    )

    # --- Save --------------------------------------------------------------
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    mirror_to_main(out_path)
    plt.close(fig)


if __name__ == "__main__":
    out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "figure1_pipeline.png",
    )
    build_figure(out)
    print(f"Wrote {out}")
