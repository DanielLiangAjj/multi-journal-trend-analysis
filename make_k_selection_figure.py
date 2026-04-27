"""
Build a single publication-grade K-selection figure combining both domains.
Reads the existing k_scan CSVs and overlays K=100 (chosen) and K=750
(paper-faithful baseline) reference lines.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("/Users/danielliang/Dropbox/multi_journal_trend_analysis")
SRC = ROOT / "data" / "clusters"
OUT = ROOT / "data" / "k_comparison"
OUT.mkdir(parents=True, exist_ok=True)


def load(domain):
    return pd.read_csv(SRC / f"{domain}_k_scan.csv")


def main():
    method = load("methodology")
    health = load("health")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    domains = [("Methodology", method), ("Health", health)]

    for row, (label, df) in enumerate(domains):
        ax_in = axes[row, 0]
        ax_in.plot(df["k"], df["inertia"], "o-", color="#1f77b4")
        ax_in.axvline(100, color="#d62728", ls="--", lw=1.5, label="K=100 (chosen)")
        ax_in.axvline(750, color="#7f7f7f", ls=":", lw=1.5, label="K=750 (paper baseline)")
        ax_in.set_title(f"Elbow Method - {label}", fontsize=12)
        ax_in.set_xlabel("Number of Clusters (K)")
        ax_in.set_ylabel("Inertia")
        ax_in.legend(loc="upper right", fontsize=9)
        ax_in.grid(alpha=0.3)

        ax_si = axes[row, 1]
        ax_si.plot(df["k"], df["silhouette_score"], "o-", color="#2ca02c")
        ax_si.axhline(0, color="black", lw=0.6, alpha=0.5)
        ax_si.axvline(100, color="#d62728", ls="--", lw=1.5, label="K=100 (chosen)")
        ax_si.axvline(750, color="#7f7f7f", ls=":", lw=1.5, label="K=750 (paper baseline)")
        s_at_100 = df.loc[df["k"] == 100, "silhouette_score"].iloc[0]
        s_at_750 = df.loc[df["k"] == 700, "silhouette_score"].iloc[0]  # nearest in scan
        ax_si.annotate(f"K=100: {s_at_100:.4f}",
                       xy=(100, s_at_100), xytext=(180, s_at_100),
                       fontsize=9, va="center",
                       arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8))
        ax_si.annotate(f"K~750: {s_at_750:.4f}",
                       xy=(700, s_at_750), xytext=(800, s_at_750 - 0.005),
                       fontsize=9, va="center",
                       arrowprops=dict(arrowstyle="->", color="#7f7f7f", lw=0.8))
        ax_si.set_title(f"Silhouette Analysis - {label}", fontsize=12)
        ax_si.set_xlabel("Number of Clusters (K)")
        ax_si.set_ylabel("Silhouette Score")
        ax_si.legend(loc="upper right", fontsize=9)
        ax_si.grid(alpha=0.3)

    fig.suptitle(
        "Empirical K Selection: K=100 maximises silhouette in both domains; "
        "scores monotonically degrade through K=1500",
        fontsize=13, y=1.00,
    )
    plt.tight_layout()
    out_path = OUT / "k_selection_combined.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
