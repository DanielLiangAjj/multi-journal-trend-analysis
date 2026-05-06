"""
Build Figure 5 - milestone-annotated trend curves for selected K=100 case-study topics.

This is a rebuild that fixes QA-flagged issues in the previous version:

1. Milestone label collisions (especially the dense 2017-2023 cluster in panels A & B)
   are resolved with `adjustText.adjust_text` for iterative repulsion. If adjustText
   is not available we fall back to manual alternating offsets.
2. The faint green secondary "% of domain" axis was removed - the figure now focuses
   only on absolute counts, which is cleaner.
3. X-tick labels no longer collide with subplot titles - top-row x-ticks are hidden,
   x-tick rotation reduced to 30 deg, and `hspace` increased.
4. Panel letters are now bold and large in the upper-left corner of each subplot.
5. Title uses "Figure 5." (period) for consistency with figures 1-3.

Inputs (read-only):
  data/visualizations_k100/methodology_topic_year_counts.csv
  data/visualizations_k100/health_topic_year_counts.csv

Output:
  figures/figure5_milestones.png  (overwrites the QA-failed 2x2 grid)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:  # pragma: no cover
    HAS_ADJUST_TEXT = False


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT = Path("/Users/danielliang/Library/CloudStorage/Dropbox/multi_journal_trend_analysis")
DATA_DIR = ROOT / "data" / "visualizations_k100"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

METHOD_YEARS = DATA_DIR / "methodology_topic_year_counts.csv"
HEALTH_YEARS = DATA_DIR / "health_topic_year_counts.csv"

# -----------------------------------------------------------------------------
# Plot configuration
# -----------------------------------------------------------------------------
YEAR_RANGE = list(range(2011, 2026))  # 2011-2025; 2026 partial year excluded.

PANELS = [
    {
        "key": "a",
        "case_study": "Deep learning innovations",
        "domain": "methodology",
        "topic": "Transformer models",
        "milestones": [
            (2012, "AlexNet"),
            (2014, "GAN"),
            (2015, "ResNet"),
            (2017, "Transformer"),
            (2018, "BERT"),
            (2020, "GPT-3"),
            (2021, "Diffusion models"),
            (2022, "ChatGPT"),
            (2023, "GPT-4"),
        ],
    },
    {
        "key": "b",
        "case_study": "Biomedical NLP / LLM",
        "domain": "methodology",
        "topic": "Large language models",
        "milestones": [
            (2013, "Word2Vec"),
            (2014, "GloVe / Attention"),
            (2017, "Transformer"),
            (2018, "BERT / GPT-1"),
            (2020, "GPT-3"),
            (2022, "ChatGPT"),
            (2023, "Med-LLMs"),
        ],
    },
    {
        "key": "c",
        "case_study": "Privacy and de-identification",
        "domain": "methodology",
        "topic": "Privacy and de-identification",
        "milestones": [
            (2013, "FDA warning to 23andMe"),
            (2014, "NIH Genomic Data Sharing Policy"),
            (2018, "All of Us / GDPR"),
            (2024, "AI bias regulations"),
        ],
    },
    {
        "key": "d",
        "case_study": "Infectious diseases",
        "domain": "health",
        "topic": "Infectious diseases",
        "milestones": [
            (2014, "Ebola epidemic"),
            (2020, "WHO declares COVID-19 pandemic"),
            (2021, "mRNA vaccines authorized"),
            (2023, "WHO ends COVID emergency"),
        ],
    },
]


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_year_counts(domain: str) -> pd.DataFrame:
    path = METHOD_YEARS if domain == "methodology" else HEALTH_YEARS
    return pd.read_csv(path)


def get_topic_series(df: pd.DataFrame, topic: str, years: list[int]) -> np.ndarray:
    row = df.loc[df["topic"] == topic]
    if row.empty:
        raise ValueError(f"Topic not found: {topic}")
    cols = [str(y) for y in years]
    return row[cols].iloc[0].astype(float).values


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def value_at_year(years: list[int], counts: np.ndarray, target: int) -> float:
    """Get curve y-value at the target year, clipped to range."""
    if target < years[0]:
        return float(counts[0])
    if target > years[-1]:
        return float(counts[-1])
    return float(counts[years.index(target)])


def annotate_milestones(
    ax: plt.Axes,
    years: list[int],
    counts: np.ndarray,
    milestones: list[tuple[int, str]],
    *,
    label_fontsize: int = 8,
) -> tuple[list[int], list[plt.Text]]:
    """Draw triangle markers at each milestone year and place label texts.

    The label texts are returned so we can hand them to adjustText for iterative
    de-collision. Markers stay anchored on the curve.
    """
    zero_years: list[int] = []
    text_objs: list[plt.Text] = []
    y_max = max(float(counts.max()), 1.0)
    base_offset = 0.18 * y_max  # initial seed offset; adjustText will refine

    for i, (year, label) in enumerate(milestones):
        y_val = value_at_year(years, counts, year)
        if years[0] <= year <= years[-1] and counts[years.index(year)] == 0:
            zero_years.append(year)

        # Triangle marker on the curve - stays put.
        ax.plot(
            year,
            y_val,
            marker="v",
            markersize=10,
            color="#c0392b",
            markeredgecolor="black",
            markeredgewidth=0.7,
            linestyle="None",
            zorder=5,
        )

        # Seed label position - alternate above/below to give adjustText a head start.
        direction = 1 if i % 2 == 0 else -1
        seed_y = y_val + direction * base_offset
        if seed_y < 0.05 * y_max:
            seed_y = y_val + base_offset

        txt = ax.text(
            year,
            seed_y,
            f"{year}\n{label}",
            ha="center",
            va="center",
            fontsize=label_fontsize,
            color="#222222",
            bbox=dict(
                boxstyle="round,pad=0.25",
                fc="white",
                ec="#999999",
                lw=0.5,
                alpha=0.92,
            ),
            zorder=6,
        )
        text_objs.append(txt)

    return zero_years, text_objs


def deconflict_labels(
    ax: plt.Axes,
    years: list[int],
    counts: np.ndarray,
    text_objs: list[plt.Text],
) -> None:
    """Run adjustText (or fall back to manual stagger) to remove label overlaps."""
    if not text_objs:
        return

    if HAS_ADJUST_TEXT:
        # Anchor x and y points the labels should avoid: data markers + curve nodes.
        x_anchors = list(years) + [t.get_position()[0] for t in text_objs]
        y_anchors = list(counts) + [t.get_position()[1] for t in text_objs]
        adjust_text(
            text_objs,
            x=x_anchors,
            y=y_anchors,
            ax=ax,
            arrowprops=dict(
                arrowstyle="-",
                color="#888888",
                lw=0.6,
                shrinkA=2,
                shrinkB=2,
            ),
            expand=(1.4, 1.6),
            force_text=(0.6, 1.0),
            force_static=(0.4, 0.8),
            min_arrow_len=4,
            max_move=80,
            time_lim=2.0,
        )
    else:
        # Manual fallback: alternate above/below with growing offsets.
        y_max = max(float(counts.max()), 1.0)
        for i, t in enumerate(text_objs):
            x_, _ = t.get_position()
            year_idx = years.index(int(round(x_))) if int(round(x_)) in years else 0
            base = float(counts[year_idx])
            direction = 1 if i % 2 == 0 else -1
            t.set_position((x_, base + direction * (0.25 + 0.08 * i) * y_max))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    method_df = load_year_counts("methodology")
    health_df = load_year_counts("health")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes_flat = axes.flatten()
    panel_letters = ["A", "B", "C", "D"]

    report_lines: list[str] = []

    for ax_idx, panel in enumerate(PANELS):
        ax = axes_flat[ax_idx]
        df = method_df if panel["domain"] == "methodology" else health_df

        counts = get_topic_series(df, panel["topic"], YEAR_RANGE)

        # Main count curve.
        ax.plot(
            YEAR_RANGE,
            counts,
            color="#1f4e79",
            linewidth=2.6,
            marker="o",
            markersize=5,
            markerfacecolor="#1f4e79",
            markeredgecolor="white",
            markeredgewidth=0.6,
            label="Articles per year",
            zorder=3,
        )
        ax.fill_between(YEAR_RANGE, 0, counts, color="#1f4e79", alpha=0.08, zorder=1)

        ax.set_ylabel("Articles per year", color="#1f4e79")
        ax.tick_params(axis="y", labelcolor="#1f4e79")
        ax.set_xticks(YEAR_RANGE)
        # Hide x-tick labels on top row to free space; bottom row keeps them.
        is_bottom_row = ax_idx >= 2
        if is_bottom_row:
            ax.set_xticklabels(YEAR_RANGE, rotation=30, ha="right")
            ax.set_xlabel("Year")
        else:
            ax.set_xticklabels([""] * len(YEAR_RANGE))

        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax.set_xlim(YEAR_RANGE[0] - 0.5, YEAR_RANGE[-1] + 0.5)

        # Annotate milestones (markers + seed text positions).
        zero_years, text_objs = annotate_milestones(
            ax, YEAR_RANGE, counts, panel["milestones"]
        )

        # Generous headroom: adjustText will push labels into the empty band above.
        y_max = float(counts.max())
        ax.set_ylim(0, y_max * 1.85 if y_max > 0 else 1)

        # Bold panel letter in upper-left corner of subplot.
        ax.text(
            0.015,
            0.965,
            panel_letters[ax_idx],
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
            ha="left",
            color="#1a1a1a",
        )

        # Subplot title - sits to the right of the bold panel letter, no inline letter.
        title = (
            f"{panel['case_study']}  -  K=100 topic: \"{panel['topic']}\" "
            f"({panel['domain']})"
        )
        ax.set_title(title, loc="left", x=0.05, pad=10, fontsize=11)

        # Build per-panel report line.
        early = counts[:3].mean()
        late = counts[-3:].mean()
        ratio = (late / early) if early > 0 else float("inf")
        post_msg = (
            f" | zero-count years among milestones: {zero_years}" if zero_years else ""
        )
        report_lines.append(
            f"Panel {panel_letters[ax_idx]} - {panel['case_study']} -> "
            f"topic '{panel['topic']}' (early3 mean={early:.1f}, late3 mean={late:.1f}, "
            f"ratio={ratio:.2f}x){post_msg}"
        )

    # Suptitle - period style for consistency with figures 1, 2, 3.
    fig.suptitle(
        "Figure 5. Milestone-annotated trend curves for selected K=100 case-study topics",
        fontsize=14,
        y=0.995,
    )

    # Layout first so axis transforms reflect final positions, then de-collide.
    fig.subplots_adjust(left=0.06, right=0.985, top=0.93, bottom=0.08,
                        hspace=0.5, wspace=0.22)
    # Force a draw so adjustText sees correct text bboxes in display coords.
    fig.canvas.draw()

    for ax_idx, panel in enumerate(PANELS):
        ax = axes_flat[ax_idx]
        df = method_df if panel["domain"] == "methodology" else health_df
        counts = get_topic_series(df, panel["topic"], YEAR_RANGE)
        # Pull the text objects out of the axis (children added via ax.text).
        text_objs = [
            child for child in ax.get_children()
            if isinstance(child, plt.Text)
            and child.get_text().count("\n") == 1
            and child.get_text()[:4].isdigit()
        ]
        deconflict_labels(ax, YEAR_RANGE, counts, text_objs)

    out_main = FIG_DIR / "figure5_milestones.png"
    fig.savefig(out_main, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_main}")

    print("\n=== Per-panel quantitative summary ===")
    for line in report_lines:
        print(line)
    print(
        f"\nadjustText available: {HAS_ADJUST_TEXT} "
        f"({'used' if HAS_ADJUST_TEXT else 'fell back to manual stagger'})"
    )


if __name__ == "__main__":
    main()
