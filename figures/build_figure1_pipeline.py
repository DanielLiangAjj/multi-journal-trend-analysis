"""
Figure 1: Publication-grade pipeline diagram for the multi-journal trend
analysis manuscript.

Mirrors the horizontal flowchart style of Fang et al. (J Biomed Inform 178
[2026] 105013), Figure 1(a): six pipeline stages, each labelled with the
action performed at that stage, plus a parallel "Evaluation" track underneath
stages 4-6.

Output: figures/figure1_pipeline.png (300 dpi, bbox_inches='tight')
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

# Muted, publication-friendly blues for the six main stages.
STAGE_FILL = "#2F5C8A"
STAGE_EDGE = "#1F3F5F"
STAGE_TEXT = "#FFFFFF"

# Evaluation track: a contrasting muted teal so it reads as a separate band.
EVAL_FILL = "#3F7C7A"
EVAL_EDGE = "#2A5A58"
EVAL_TEXT = "#FFFFFF"

# Badge palette.
BADGE_COLORS = {
    "NEW": "#2E8B57",       # green
    "MODIFIED": "#E08E2A",  # orange
    "FAITHFUL": "#7A7A7A",  # grey
}
BADGE_TEXT = "#FFFFFF"

ARROW_COLOR = "#333333"
EVAL_ARROW_COLOR = "#3F7C7A"

# ---------------------------------------------------------------------------
# Stage content
# ---------------------------------------------------------------------------

STAGES = [
    {
        "title": "1. Data\n    Collection",
        "subtext": (
            "29 PubMed journals\n"
            "2011-2025\n"
            "78,425 articles\n"
            "90,386 keywords\n"
            "Three-pass keyword\nacquisition"
        ),
        "badge": "NEW",
    },
    {
        "title": "2. Keyword\n    Categorization",
        "subtext": (
            "Few-shot BiomedBERT\n+ GPT-5-nano\n"
            "vs. MeSH 2024 reference\n"
            "4 classes: methodology /\nhealth / both / neither"
        ),
        "badge": "MODIFIED",
    },
    {
        "title": "3. Topic\n    Clustering",
        "subtext": (
            "K-means on BiomedBERT\nembeddings\n"
            "K = 100 selected by\nsilhouette analysis"
        ),
        "badge": "MODIFIED",
    },
    {
        "title": "4. Topic\n    Naming",
        "subtext": (
            "GPT-5-nano cluster\ncoherence check\n"
            "Sub-cluster splitting\n"
            "Frequency-weighted\nnaming\n"
            "61-topic LLM audit pass"
        ),
        "badge": "NEW",
    },
    {
        "title": "5. Hierarchy\n    Construction",
        "subtext": (
            "5-phase pipeline:\n"
            "stemming merge ->\n"
            "cosine similarity ->\n"
            "adaptive threshold ->\n"
            "relationship class. ->\n"
            "4-round verification +\n"
            "cycle / transitive cleanup"
        ),
        "badge": "FAITHFUL",
    },
    {
        "title": "6. Trend Analysis &\n    Statistical Testing",
        "subtext": (
            "Linear-trend Holm-\nBonferroni correction\n"
            "Poisson rate test\n(COVID-19)\n"
            "14 new analyses\n(N1-N7 ecosystem +\nPI Q1-Q7 practical impact)"
        ),
        "badge": "NEW",
    },
]

# ---------------------------------------------------------------------------
# Geometry (data units; aspect = "auto" so x and y scale independently)
# ---------------------------------------------------------------------------

FIG_W, FIG_H = 16.0, 6.0  # inches

CANVAS_W = 16.0
CANVAS_H = 6.0

N_STAGES = len(STAGES)
LEFT_PAD = 0.20
RIGHT_PAD = 0.20
GAP = 0.28  # space between adjacent stage boxes (for arrows)

# Stage row geometry.
BOX_W = (CANVAS_W - LEFT_PAD - RIGHT_PAD - GAP * (N_STAGES - 1)) / N_STAGES
BOX_H = 2.55
BOX_Y = 2.45  # bottom of main row

# Evaluation row geometry.
EVAL_BOX_H = 1.05
EVAL_BOX_Y = 0.85  # bottom of eval row

# Reserved vertical band at top for the figure title.
TITLE_Y = CANVAS_H - 0.25

# Reserved vertical band at bottom for the legend.
LEGEND_Y = 0.18


def stage_x(i: int) -> float:
    """Left-x of stage i (0-indexed)."""
    return LEFT_PAD + i * (BOX_W + GAP)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def draw_stage_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtext: str,
    badge: str | None = None,
):
    """Draw a rounded rectangle stage box with title, subtext, and optional badge."""
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

    # Badge pill in the top-right corner, placed in a reserved top strip
    # so it never overlaps the title text.
    badge_h = 0.24
    badge_w = 0.60
    if badge is not None:
        badge_x = x + w - badge_w - 0.12
        badge_y = y + h - badge_h - 0.10
        pill = FancyBboxPatch(
            (badge_x, badge_y),
            badge_w,
            badge_h,
            boxstyle="round,pad=0.01,rounding_size=0.10",
            linewidth=0.8,
            facecolor=BADGE_COLORS[badge],
            edgecolor="white",
            zorder=4,
        )
        ax.add_patch(pill)
        ax.text(
            badge_x + badge_w / 2,
            badge_y + badge_h / 2,
            badge,
            ha="center",
            va="center",
            fontsize=6.8,
            fontweight="bold",
            color=BADGE_TEXT,
            zorder=5,
        )

    # Title sits BELOW the badge strip so it can use the full inner width.
    # The reserved badge strip is ~0.40 data units tall.
    title_top_y = y + h - 0.42
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

    # Subtext below title. Fixed gap below the (possibly two-line) title
    # so all stages align visually; we always leave room for two title lines.
    ax.text(
        x + 0.14,
        y + h - 1.06,
        subtext,
        ha="left",
        va="top",
        fontsize=8.0,
        color=STAGE_TEXT,
        linespacing=1.25,
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


def draw_eval_arrow(ax, x_from: float, y_from: float, x_to: float, y_to: float):
    arrow = FancyArrowPatch(
        (x_from, y_from),
        (x_to, y_to),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.3,
        color=EVAL_ARROW_COLOR,
        linestyle=(0, (4, 2)),
        zorder=1,
    )
    ax.add_patch(arrow)


# ---------------------------------------------------------------------------
# Build the figure
# ---------------------------------------------------------------------------


def build_figure(out_path: str):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(0, CANVAS_H)
    ax.set_aspect("auto")  # let x and y scale independently
    ax.axis("off")

    # --- Figure title (drawn first so it sits in its reserved band) -------
    ax.text(
        CANVAS_W / 2,
        TITLE_Y,
        "Figure 1. Pipeline overview for multi-journal biomedical research-trend analysis (2011-2025)",
        ha="center",
        va="top",
        fontsize=12.0,
        fontweight="bold",
        color="#222222",
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
            badge=stage["badge"],
        )

        # Arrow to the next stage (if any).
        if i < N_STAGES - 1:
            x_from = x + BOX_W - 0.02
            x_to = stage_x(i + 1) + 0.02
            draw_horizontal_arrow(ax, x_from, x_to, BOX_Y + BOX_H / 2)

    # --- Evaluation track --------------------------------------------------
    eval_left = stage_x(3)
    eval_right = stage_x(5) + BOX_W
    eval_w = eval_right - eval_left

    eval_box = FancyBboxPatch(
        (eval_left + 0.02, EVAL_BOX_Y + 0.02),
        eval_w - 0.04,
        EVAL_BOX_H - 0.04,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.2,
        facecolor=EVAL_FILL,
        edgecolor=EVAL_EDGE,
        zorder=2,
    )
    ax.add_patch(eval_box)

    ax.text(
        eval_left + 0.18,
        EVAL_BOX_Y + EVAL_BOX_H - 0.18,
        "MeSH-based Automatic Evaluation",
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=EVAL_TEXT,
        zorder=3,
    )
    ax.text(
        eval_left + 0.18,
        EVAL_BOX_Y + EVAL_BOX_H - 0.50,
        (
            "Set A (NLM-assigned MeSH terms)  $\\cap$  Set B (MeSH terms predicted from topics)\n"
            "      $\\rightarrow$  overlap rate validates topic naming, hierarchy, and the downstream trend signal"
        ),
        ha="left",
        va="top",
        fontsize=8.5,
        color=EVAL_TEXT,
        linespacing=1.35,
        zorder=3,
    )

    # Upward dashed arrows from eval track to stages 4, 5, 6.
    eval_top_y = EVAL_BOX_Y + EVAL_BOX_H
    stage_bottom_y = BOX_Y
    for i in (3, 4, 5):
        x_center = stage_x(i) + BOX_W / 2
        draw_eval_arrow(
            ax,
            x_from=x_center,
            y_from=eval_top_y + 0.02,
            x_to=x_center,
            y_to=stage_bottom_y - 0.02,
        )

    # --- Legend (under the eval track, full width) ------------------------
    legend_y = LEGEND_Y
    legend_items = [
        ("NEW", "new contribution"),
        ("MODIFIED", "adapted from prior work"),
        ("FAITHFUL", "re-implemented as in Fang et al."),
    ]
    cursor_x = LEFT_PAD
    for tag, desc in legend_items:
        pill_w, pill_h = 0.65, 0.26
        pill = FancyBboxPatch(
            (cursor_x, legend_y),
            pill_w,
            pill_h,
            boxstyle="round,pad=0.01,rounding_size=0.10",
            linewidth=0.8,
            facecolor=BADGE_COLORS[tag],
            edgecolor="white",
            zorder=4,
        )
        ax.add_patch(pill)
        ax.text(
            cursor_x + pill_w / 2,
            legend_y + pill_h / 2,
            tag,
            ha="center",
            va="center",
            fontsize=7.0,
            fontweight="bold",
            color="white",
            zorder=5,
        )
        ax.text(
            cursor_x + pill_w + 0.10,
            legend_y + pill_h / 2,
            desc,
            ha="left",
            va="center",
            fontsize=8.5,
            color="#333333",
            zorder=5,
        )
        # Advance cursor: pill + label width (data-units estimate ~0.075/char at 8.5pt).
        cursor_x += pill_w + 0.12 + len(desc) * 0.085 + 0.40

    # --- Save --------------------------------------------------------------
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "figure1_pipeline.png",
    )
    build_figure(out)
    print(f"Wrote {out}")
