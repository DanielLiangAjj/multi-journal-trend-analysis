"""Supplementary Figure 9 — silhouette score versus K over the FULL range, K = 2 to 1500.

Replaces the previous image, whose baked title asserted "K=100 maximises silhouette
in both domains". That scan started at K=100, so it could not see the maximum. The
sub-100 scan (data/k_comparison/k_scan_sub100.csv, run with the same MiniBatchKMeans
settings as the original) shows silhouette is highest at K=2 and near zero from K=20
upward, so K=100 is a granularity choice, not a silhouette optimum.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _mirror import mirror_to_main

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "supp_figure9_k_scan_full.png"

INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#dcdcdc"
COL = {"methodology": "#0072B2", "health": "#D55E00"}

sub = pd.read_csv(ROOT / "data/k_comparison/k_scan_sub100.csv")
series = {}
for dom in ("methodology", "health"):
    hi = pd.read_csv(ROOT / f"data/clusters/{dom}_k_scan.csv")[["k", "silhouette_score"]]
    hi = hi.rename(columns={"silhouette_score": "silhouette"})
    lo = sub[sub.domain == dom][["k", "silhouette"]]
    series[dom] = (pd.concat([lo, hi]).drop_duplicates("k")
                 .sort_values("k").reset_index(drop=True))

fig, ax = plt.subplots(figsize=(11, 5.4))
ax.axhline(0, color=MUTED, lw=1, alpha=0.6, zorder=1)

for dom, s in series.items():
    ax.plot(s.k, s.silhouette, color=COL[dom], lw=2, marker="o", ms=5,
            markerfacecolor="white", markeredgewidth=1.5, zorder=3,
            label=dom.capitalize())
    ax.annotate(f" {dom.capitalize()}", (s.k.iloc[-1], s.silhouette.iloc[-1]),
                color=INK, fontsize=9.5, va="center", ha="left",
                xytext=(4, 0), textcoords="offset points")

for k, lab, style in [(2, "K = 2\nsilhouette maximum", "max"),
                      (100, "K = 100\nchosen granularity", "chosen"),
                      (750, "K = 750\nsingle-journal value", "ref")]:
    ax.axvline(k, color=MUTED, lw=1.1,
               ls=(0, (5, 4)) if style != "chosen" else (0, (2, 2)),
               alpha=0.75 if style == "chosen" else 0.45, zorder=2)
    ax.annotate(lab, (k, ax.get_ylim()[1]), xytext=(0, -4),
                textcoords="offset points", ha="center", va="top",
                fontsize=8.8, color=MUTED)

ax.set_xscale("log")
ax.set_xticks([2, 5, 10, 20, 50, 100, 200, 500, 1000, 1500])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("Number of K-means clusters (K, log scale)", fontsize=11, color=INK)
ax.set_ylabel("Silhouette score", fontsize=11, color=INK)
ax.set_title("Silhouette score across the full K range, K = 2–1500\n"
             "Scores are near zero throughout and highest at the coarsest settings — "
             "K = 100 is a granularity choice, not a silhouette optimum",
             fontsize=12.5, fontweight="bold", color=INK, pad=12)
ax.legend(frameon=False, fontsize=10, loc="lower left", labelcolor=INK)
ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.8, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(colors=MUTED, labelsize=9.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)

fig.tight_layout()
fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"wrote {OUT}")
for dom, s in series.items():
    i = int(s.silhouette.idxmax())
    print(f"  {dom}: max silhouette {s.silhouette.iat[i]:.4f} at K={int(s.k.iat[i])}; "
          f"K=100 -> {s.loc[s.k == 100, 'silhouette'].iat[0]:.4f}")
mirror_to_main(OUT)
