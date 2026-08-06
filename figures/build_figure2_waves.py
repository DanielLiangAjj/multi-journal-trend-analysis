"""Figure 2 — annual publication volume with three OVERLAPPING waves of research
attention.

Replaces figure3_volume_eras_annotated.png, whose title and three abutting shaded
blocks asserted mutually exclusive "eras" — the reading the caption explicitly
disclaims and that the round-2 review rejected ("the genomics era didn't end").
Waves are drawn as overlapping bands with feathered onsets, so no boundary is
implied.
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
OUT = ROOT / "figures" / "figure2_volume_waves.png"

# Validated categorical triple (CVD-checked: worst adjacent deutan dE 11.0).
WAVE = [
    ("Genomics & foundational informatics", 2011, 2025, "#0072B2"),
    ("Machine learning",                    2016, 2025, "#D55E00"),
    ("LLM / generative AI",                 2021, 2025, "#009E73"),
]
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d8"
BAR = "#c3cdd6"

df = pd.read_csv(ROOT / "data/pubmed_all_journals_2011_2025_w_keywords_patched.csv",
                 usecols=["year"], low_memory=False)
c = df[df.year.between(2011, 2025)].year.value_counts().sort_index()
years, vals = c.index.to_numpy(), c.to_numpy()

fig, ax = plt.subplots(figsize=(11, 5.6))
ax.bar(years, vals, width=0.72, color=BAR, edgecolor="white", linewidth=0.8, zorder=2)
ax.plot(years, vals, color=INK, lw=2, marker="o", ms=4.5, zorder=4,
        markerfacecolor="white", markeredgewidth=1.5)

top = vals.max() * 1.45
ax.set_ylim(0, top)

# Overlapping wave bands: each starts at its onset and runs to the end of the
# window, stacked in shallow lanes so the overlap is visible rather than implied.
lane_h = top * 0.052
for i, (label, onset, end, col) in enumerate(WAVE):
    y0 = top - lane_h * (i + 1) - top * 0.015 * i
    # feathered onset so the left edge does not read as a hard boundary
    grad = np.linspace(0, 1, 120) ** 0.6
    xs = np.linspace(onset - 1.1, min(onset + 2.2, end), 120)
    ax.imshow(grad.reshape(1, -1), extent=[xs[0], xs[-1], y0, y0 + lane_h],
              aspect="auto", cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                  "f", [(1, 1, 1, 0), matplotlib.colors.to_rgba(col, 0.42)]), zorder=3)
    ax.add_patch(plt.Rectangle((xs[-1], y0), end + 0.45 - xs[-1], lane_h,
                               color=col, alpha=0.42, lw=0, zorder=3))
    ax.plot([onset, onset], [0, y0 + lane_h], color=col, lw=1.4, ls=(0, (4, 3)),
            alpha=0.85, zorder=3)
    ax.text(onset + 0.12, y0 + lane_h / 2, f"  {label}  —  onset ≈ {onset}",
            va="center", ha="left", fontsize=9.5, color=INK, zorder=5)

for x, v in zip(years, vals):
    if x in (2011, 2015, 2020, 2025):
        ax.annotate(f"{v:,}", (x, v), textcoords="offset points",
                    xytext=(-14 if x == 2025 else 0, 9),
                    ha="right" if x == 2025 else "center",
                    fontsize=9, color=INK, fontweight="bold")

ax.set_xlabel("Year", fontsize=11, color=INK)
ax.set_ylabel("Articles published", fontsize=11, color=INK)
ax.set_title("Annual publication volume across 29 biomedical informatics journals, "
             "2011–2025\nThree overlapping waves of research attention — onsets, not "
             "period boundaries", fontsize=12.5, fontweight="bold", color=INK, pad=12)
ax.set_xticks(years)
ax.tick_params(colors=MUTED, labelsize=9.5)
ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)

fig.tight_layout()
fig.savefig(OUT, dpi=400, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"wrote {OUT}")
mirror_to_main(OUT)
