"""
Build Figure 5-style milestone-annotated trend curves for selected K=100 case-study topics.

Mirrors Figure 5 of Fang et al., Journal of Biomedical Informatics 178 (2026) 105013:
overlays technology / policy milestones as triangles + labels on smoothed trend curves
to verify that the K=100 pipeline recovers well-known historical inflection points.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
YEAR_RANGE = list(range(2011, 2026))  # plot 2011-2025 (2026 partial year excluded)

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
    df = pd.read_csv(path)
    return df


def get_topic_series(df: pd.DataFrame, topic: str, years: list[int]) -> np.ndarray:
    row = df.loc[df["topic"] == topic]
    if row.empty:
        raise ValueError(f"Topic not found: {topic}")
    cols = [str(y) for y in years]
    return row[cols].iloc[0].astype(float).values


def domain_total_per_year(df: pd.DataFrame, years: list[int]) -> np.ndarray:
    cols = [str(y) for y in years]
    return df[cols].sum(axis=0).astype(float).values


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
) -> list[int]:
    """Draw triangle markers + labels for each milestone. Returns list of zero-count milestone years."""
    y_max = max(counts.max(), 1.0)
    # Vertical offset for label, alternating up/down so labels don't overlap
    base_offset = 0.18 * y_max
    zero_years = []

    for i, (year, label) in enumerate(milestones):
        y_val = value_at_year(years, counts, year)
        if year >= years[0] and year <= years[-1]:
            if counts[years.index(year)] == 0:
                zero_years.append(year)
        # Triangle marker on the curve
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
        # Alternating offset direction so adjacent labels don't collide
        direction = 1 if i % 2 == 0 else -1
        offset = direction * base_offset * (1.0 + 0.25 * (i % 3))
        label_y = y_val + offset
        # keep labels within visible area roughly
        if label_y < 0:
            label_y = y_val + base_offset
        # Thin connector line drawn implicitly via annotation arrow
        ax.annotate(
            f"{year}\n{label}",
            xy=(year, y_val),
            xytext=(year, label_y),
            ha="center",
            va="bottom" if direction > 0 else "top",
            fontsize=8,
            color="#333333",
            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.6),
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec="#bbbbbb",
                lw=0.5,
                alpha=0.9,
            ),
            zorder=6,
        )
    return zero_years


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    method_df = load_year_counts("methodology")
    health_df = load_year_counts("health")

    method_totals = domain_total_per_year(method_df, YEAR_RANGE)
    health_totals = domain_total_per_year(health_df, YEAR_RANGE)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
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
        domain_totals = method_totals if panel["domain"] == "methodology" else health_totals

        counts = get_topic_series(df, panel["topic"], YEAR_RANGE)

        # Main count curve
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

        # Light fill under the curve for emphasis
        ax.fill_between(YEAR_RANGE, 0, counts, color="#1f4e79", alpha=0.08, zorder=1)

        ax.set_xlabel("Year")
        ax.set_ylabel("Articles per year", color="#1f4e79")
        ax.tick_params(axis="y", labelcolor="#1f4e79")
        ax.set_xticks(YEAR_RANGE)
        ax.set_xticklabels(YEAR_RANGE, rotation=45)
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax.set_xlim(YEAR_RANGE[0] - 0.5, YEAR_RANGE[-1] + 0.5)

        # Annotate milestones
        zero_years = annotate_milestones(ax, YEAR_RANGE, counts, panel["milestones"])

        # Add headroom for labels
        y_max = float(counts.max())
        ax.set_ylim(0, y_max * 1.55 if y_max > 0 else 1)

        # Secondary axis: % share of all publications in domain (green dashed)
        share = np.where(domain_totals > 0, counts / domain_totals * 100.0, 0.0)
        ax2 = ax.twinx()
        ax2.plot(
            YEAR_RANGE,
            share,
            color="#2e7d32",
            linewidth=1.4,
            linestyle="--",
            marker="s",
            markersize=3,
            alpha=0.75,
            label="% of domain",
            zorder=2,
        )
        ax2.set_ylabel("% of domain publications", color="#2e7d32")
        ax2.tick_params(axis="y", labelcolor="#2e7d32")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_color("#2e7d32")
        # Headroom so green doesn't overlap labels of blue line
        s_max = float(share.max()) if share.max() > 0 else 1.0
        ax2.set_ylim(0, s_max * 2.2)

        title = (
            f"{panel_letters[ax_idx]}. {panel['case_study']}\n"
            f"K=100 topic: \"{panel['topic']}\" ({panel['domain']})"
        )
        ax.set_title(title, loc="left")

        # Combined legend (in upper-left of subplot)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper left",
            fontsize=8,
            frameon=True,
            framealpha=0.9,
        )

        # Build a per-panel report line
        post_msg = ""
        if zero_years:
            post_msg = f" | zero-count years among milestones: {zero_years}"
        # surge check: compare last 3 years mean vs first 3 years mean
        early = counts[:3].mean()
        late = counts[-3:].mean()
        ratio = (late / early) if early > 0 else float("inf")
        report_lines.append(
            f"Panel {panel_letters[ax_idx]} - {panel['case_study']} -> "
            f"topic '{panel['topic']}' (early3 mean={early:.1f}, late3 mean={late:.1f}, "
            f"ratio={ratio:.2f}x){post_msg}"
        )

    fig.suptitle(
        "Figure 5. Milestone-annotated trend curves for selected K=100 case-study topics",
        fontsize=14,
        y=1.00,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_main = FIG_DIR / "figure5_milestones.png"
    fig.savefig(out_main, dpi=300, bbox_inches="tight")
    print(f"Wrote: {out_main}")

    # Save individual panels too
    for ax_idx, panel in enumerate(PANELS):
        single_fig, single_ax = plt.subplots(figsize=(9, 6))
        df = method_df if panel["domain"] == "methodology" else health_df
        domain_totals = method_totals if panel["domain"] == "methodology" else health_totals
        counts = get_topic_series(df, panel["topic"], YEAR_RANGE)

        single_ax.plot(
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
        single_ax.fill_between(YEAR_RANGE, 0, counts, color="#1f4e79", alpha=0.08, zorder=1)
        single_ax.set_xlabel("Year")
        single_ax.set_ylabel("Articles per year", color="#1f4e79")
        single_ax.tick_params(axis="y", labelcolor="#1f4e79")
        single_ax.set_xticks(YEAR_RANGE)
        single_ax.set_xticklabels(YEAR_RANGE, rotation=45)
        single_ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        single_ax.set_xlim(YEAR_RANGE[0] - 0.5, YEAR_RANGE[-1] + 0.5)
        annotate_milestones(single_ax, YEAR_RANGE, counts, panel["milestones"])
        y_max = float(counts.max())
        single_ax.set_ylim(0, y_max * 1.55 if y_max > 0 else 1)

        share = np.where(domain_totals > 0, counts / domain_totals * 100.0, 0.0)
        ax2s = single_ax.twinx()
        ax2s.plot(
            YEAR_RANGE,
            share,
            color="#2e7d32",
            linewidth=1.4,
            linestyle="--",
            marker="s",
            markersize=3,
            alpha=0.75,
            label="% of domain",
            zorder=2,
        )
        ax2s.set_ylabel("% of domain publications", color="#2e7d32")
        ax2s.tick_params(axis="y", labelcolor="#2e7d32")
        ax2s.spines["top"].set_visible(False)
        s_max = float(share.max()) if share.max() > 0 else 1.0
        ax2s.set_ylim(0, s_max * 2.2)

        title = (
            f"{panel_letters[ax_idx]}. {panel['case_study']} - "
            f"K=100 topic: \"{panel['topic']}\" ({panel['domain']})"
        )
        single_ax.set_title(title, loc="left")
        lines1, labels1 = single_ax.get_legend_handles_labels()
        lines2, labels2 = ax2s.get_legend_handles_labels()
        single_ax.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper left",
            fontsize=9,
            frameon=True,
            framealpha=0.9,
        )
        single_fig.tight_layout()
        out_single = FIG_DIR / f"figure5_milestones_panel_{panel['key']}.png"
        single_fig.savefig(out_single, dpi=300, bbox_inches="tight")
        plt.close(single_fig)
        print(f"Wrote: {out_single}")

    plt.close(fig)

    print("\n=== Per-panel quantitative summary ===")
    for line in report_lines:
        print(line)


if __name__ == "__main__":
    main()
