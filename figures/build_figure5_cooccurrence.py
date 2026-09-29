"""Figure 5 — methodology x health co-occurrence heat-map, K=100, 29-journal corpus.

REPLACES the image previously embedded in the manuscript, which was rendered from
~/Downloads/visualizations_trend_analysis/cooccurrence_matrix.csv — the ORIGINAL
Fang et al. single-journal (JBI) output. Its axis labels ("Digital research
methods", "recommender algorithm", "Healthcare semantic interoperability", ...)
belong to that earlier taxonomy and appear nowhere in this paper's K=100 taxonomy,
so the figure was showing another study's results.

This builds from data/visualizations_k100/cooccurrence_matrix.csv, which is the
matrix Supplementary Table 2 and the Section 4.1.3 claims are computed from.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _mirror import mirror_to_main

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "figure5_cooccurrence_k100.png"
N = 15                                    # top N per domain, by co-occurrence volume

INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#e4e4e4"

co = pd.read_csv(ROOT / "data/visualizations_k100/cooccurrence_matrix.csv", index_col=0)
co = co.loc[co.sum(axis=1).nlargest(N).index, co.sum(axis=0).nlargest(N).index]
co = co.loc[co.sum(axis=1).sort_values(ascending=False).index,
            co.sum(axis=0).sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(13.5, 9.5))
# sequential = one hue, light -> dark (never a rainbow)
im = ax.imshow(co.values, cmap="Blues", aspect="auto",
               norm=matplotlib.colors.PowerNorm(gamma=0.55, vmin=0, vmax=co.values.max()))

ax.set_xticks(range(len(co.columns)))
ax.set_yticks(range(len(co.index)))
ax.set_xticklabels(co.columns, rotation=42, ha="right", fontsize=9.5, color=INK)
ax.set_yticklabels(co.index, fontsize=9.5, color=INK)
ax.set_xticks(np.arange(-.5, len(co.columns), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(co.index), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2)   # 2px surface gap between cells
ax.tick_params(which="minor", length=0)
ax.tick_params(which="major", length=0, colors=MUTED)

hi = co.values.max()
for i in range(len(co.index)):
    for j in range(len(co.columns)):
        v = co.values[i, j]
        ax.text(j, i, f"{int(v):,}", ha="center", va="center", fontsize=7.8,
                color="white" if v > hi * 0.45 else INK)

cb = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.015)
cb.set_label("Articles assigned to both topics", fontsize=10, color=INK)
cb.ax.tick_params(labelsize=9, colors=MUTED)
cb.outline.set_visible(False)

ax.set_xlabel("Health topics (K=100)", fontsize=11, color=INK, labelpad=10)
ax.set_ylabel("Methodology topics (K=100)", fontsize=11, color=INK, labelpad=8)
# (in-image title removed 2026-09-29; the caption carries it)
for s in ax.spines.values():
    s.set_visible(False)

fig.tight_layout()
fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"wrote {OUT}")
st = co.stack().sort_values(ascending=False).head(10)
print("  top 10 pairs shown:")
for (m, h), v in st.items():
    print(f"    {int(v):6,d}  {m} × {h}")
mirror_to_main(OUT)
