"""Cross-journal universality vs niche: per-topic entropy spectrum.

Horizontal bar chart of normalised Shannon entropy per topic, sorted from
most-niche (low entropy, journal-concentrated) to most-universal
(high entropy, evenly spread across all 29 journals). Two panels (methodology,
health). Background shading marks the universal (≥0.85) and niche (≤0.55)
zones cited in §4.2.3.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MAIN_MIRROR = Path("/Users/danielliang/Library/CloudStorage/Dropbox/multi_journal_trend_analysis")
OUT = ROOT / "figures" / "figure_entropy_spectrum.png"


def panel(ax, df, domain_label, colour):
    df2 = df.sort_values("norm_entropy")
    n = len(df2)
    y = np.arange(n)
    bars = ax.barh(y, df2["norm_entropy"], color=colour, alpha=0.78,
                   edgecolor="white", linewidth=0.4)
    # Background shading: niche (≤0.55) and universal (≥0.85)
    ax.axvspan(0, 0.55, alpha=0.07, color="#d62728", zorder=0)
    ax.axvspan(0.85, 1.0, alpha=0.07, color="#2ca02c", zorder=0)
    ax.axvline(0.55, color="#d62728", lw=0.7, ls="--", alpha=0.6)
    ax.axvline(0.85, color="#2ca02c", lw=0.7, ls="--", alpha=0.6)
    ax.text(0.275, n + 0.7, "NICHE\n(entropy ≤ 0.55)",
            fontsize=8.5, color="#d62728", ha="center", fontweight="bold")
    ax.text(0.925, n + 0.7, "UNIVERSAL\n(entropy ≥ 0.85)",
            fontsize=8.5, color="#2ca02c", ha="center", fontweight="bold")
    ax.text(0.70, n + 0.7, "INTERMEDIATE",
            fontsize=8.5, color="#666", ha="center", fontweight="bold")

    # Topic labels
    ax.set_yticks(y)
    ax.set_yticklabels(df2["topic"], fontsize=7.2)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.7, n + 1.7)
    ax.set_xlabel("Normalised Shannon entropy across 29 journals", fontsize=10.5)
    ax.tick_params(axis="x", labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.18, lw=0.4)

    # Annotate the most-niche topic and most-universal topic at right
    for ix in [0, 1, 2, n - 1, n - 2, n - 3]:
        if 0 <= ix < n:
            row = df2.iloc[ix]
            ax.text(row["norm_entropy"] + 0.005, ix,
                    f"  {int(row['n_journals'])} jrnls · "
                    f"{row['max_journal'][:18] if isinstance(row['max_journal'], str) else ''} {row['max_share']:.0f}%",
                    fontsize=6.6, va="center", ha="left", color="#444")

    ax.set_title(f"({domain_label[0]}) {domain_label[1]} topics — entropy spectrum",
                 fontsize=11.5, fontweight="bold", pad=8)


def main():
    sp_m = pd.read_csv(ROOT / "data/visualizations_k100/pi_q5_methodology_specialization.csv")
    sp_h = pd.read_csv(ROOT / "data/visualizations_k100/pi_q5_health_specialization.csv")
    # Filter to topics with at least 80 articles (matches §4.2.3 threshold)
    sp_m = sp_m[sp_m["total"] >= 80]
    sp_h = sp_h[sp_h["total"] >= 80]

    fig, axes = plt.subplots(1, 2, figsize=(15, max(11, 0.20 * max(len(sp_m), len(sp_h)) + 2)))
    panel(axes[0], sp_m, ("A", "Methodology"), "#1f4ea0")
    panel(axes[1], sp_h, ("B", "Health"), "#9c27b0")

    fig.suptitle(
        "Cross-journal topic universality vs niche concentration (K=100)\n"
        "Higher entropy → topic spread evenly across all 29 journals (universal vocabulary). "
        "Lower entropy → topic concentrated in a few journals (identity marker).",
        fontsize=12.5, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print(f"  wrote {OUT}")
    if MAIN_MIRROR.exists():
        import shutil
        shutil.copy2(OUT, MAIN_MIRROR / "figures" / OUT.name)
        print(f"  mirrored to MAIN")


if __name__ == "__main__":
    main()
