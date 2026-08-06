"""Figure 4: top-topic yearly trajectories (small multiples).

Replaces the dual-axis 16-panel grid in
data/visualizations_k100/rq2_*_topic_trends.png with:
  - single y-axis (raw yearly count)
  - shared y-scale within each domain (so magnitude is comparable)
  - top 12 topics by total volume per domain
  - one regression line per panel
  - slope, R^2, Holm-adjusted p annotated in-panel
  - line/marker colour encodes Holm-significance:
      blue = Holm-significant positive slope
      grey = not Holm-significant
      red  = Holm-significant negative slope (none observed in this corpus,
             but reserved for cross-K consistency)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MAIN_MIRROR = Path("/Users/danielliang/Library/CloudStorage/Dropbox/multi_journal_trend_analysis")
VIS_DIR = ROOT / "data/visualizations_k100"
OUT_DIR = ROOT / "figures"

YEARS = list(range(2011, 2026))
N_TOP = 12
GRID = (3, 4)
COLOUR_SIG_POS = "#1f77b4"
COLOUR_SIG_NEG = "#d62728"
COLOUR_NS = "#888888"


def panel_color(slope, sig):
    if not sig:
        return COLOUR_NS
    return COLOUR_SIG_POS if slope > 0 else COLOUR_SIG_NEG


def fmt_p(p):
    if p < 1e-3:
        return f"p={p:.0e}"
    return f"p={p:.3f}"


def build(domain: str, out_path: Path):
    counts = pd.read_csv(VIS_DIR / f"{domain}_topic_year_counts.csv")
    trends = pd.read_csv(VIS_DIR / f"{domain}_trend_analysis.csv")

    # Pick top-N by total volume
    top = trends.sort_values("total", ascending=False).head(N_TOP)
    topic_names = top["topic"].tolist()

    # Year columns present in counts CSV
    year_cols = [str(y) for y in YEARS if str(y) in counts.columns]
    counts_idx = counts.set_index("topic")

    rows, cols = GRID
    fig, axes = plt.subplots(rows, cols, figsize=(15, 9.5),
                             sharey=True, sharex=True)
    axes = axes.flatten()

    # Compute global y-max for shared scale
    ymax = max(counts_idx.loc[t, year_cols].max() for t in topic_names)

    for ax_i, name in enumerate(topic_names):
        ax = axes[ax_i]
        y = counts_idx.loc[name, year_cols].values.astype(float)
        x = np.array([int(c) for c in year_cols])
        row = trends[trends.topic == name].iloc[0]
        slope = row["slope"]
        rsq = row["r_squared"]
        p_holm = row["p_value_holm"]
        sig = bool(row["significant_holm"])
        col = panel_color(slope, sig)

        ax.plot(x, y, "o-", color=col, lw=1.8, markersize=4.5)
        # Linear fit overlay
        if len(x) > 1:
            xc = x - x.mean()
            yc = y - y.mean()
            m = (xc * yc).sum() / (xc ** 2).sum()
            b = y.mean() - m * x.mean()
            ax.plot(x, m * x + b, "--", color=col, alpha=0.45, lw=1.2)

        # Annotation box
        sig_label = "Holm-sig." if sig else "ns"
        ann = (f"slope={slope:+.1f}/yr\n"
               f"R²={rsq:.2f}\n"
               f"{fmt_p(p_holm)} ({sig_label})")
        ax.text(0.04, 0.96, ann, transform=ax.transAxes,
                fontsize=8.5, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#dddddd", lw=0.7, alpha=0.9))

        # Title with topic + total
        ax.set_title(f"{name}\n(total = {int(row['total']):,})",
                     fontsize=10, pad=4)
        ax.set_ylim(0, ymax * 1.08)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(
            lambda v, _: f"{int(v):,}" if v >= 1000 else f"{int(v)}"))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", alpha=0.25, lw=0.5)

    # Hide any unused axes
    for ax in axes[len(topic_names):]:
        ax.set_visible(False)

    # Shared axis labels
    fig.supxlabel("Year", fontsize=11)
    fig.supylabel("Articles per year", fontsize=11)

    domain_label = "methodology" if domain == "methodology" else "health"
    panel = "a" if domain == "methodology" else "b"
    n_sig = int(trends["significant_holm"].sum())
    n_total = len(trends)
    fig.suptitle(
        f"({panel}) Top {N_TOP} {domain_label} topics by total volume — yearly trajectories\n"
        f"{n_sig}/{n_total} topics carry a Holm-significant linear slope; line colour encodes significance",
        fontsize=13, fontweight="bold", y=0.995,
    )

    # Legend at the bottom
    handles = [
        plt.Line2D([0], [0], color=COLOUR_SIG_POS, lw=2.0,
                   marker="o", markersize=5, label="Holm-significant positive slope"),
        plt.Line2D([0], [0], color=COLOUR_NS, lw=2.0,
                   marker="o", markersize=5, label="Not Holm-significant"),
        plt.Line2D([0], [0], color=COLOUR_SIG_NEG, lw=2.0,
                   marker="o", markersize=5, label="Holm-significant negative slope"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.005))

    fig.tight_layout(rect=[0.03, 0.05, 1, 0.94])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {out_path}")

    mirror = MAIN_MIRROR / "figures" / out_path.name
    if mirror.parent.exists():
        import shutil
        shutil.copy2(out_path, mirror)
        print(f"  mirrored to {mirror}")


def main():
    for d in ("methodology", "health"):
        out = OUT_DIR / f"figure4_{d}_topic_trends.png"
        build(d, out)


if __name__ == "__main__":
    main()
