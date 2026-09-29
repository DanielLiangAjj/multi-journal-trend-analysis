"""Figure 8 — per-journal publication volume for all 29 journals, grouped into the
three trajectory archetypes defined in Methods 3.4.

Replaces the previous image, which plotted only the top 10 journals as
undifferentiated cycled lines while the caption promised all 29 grouped into
archetypes. 29 series cannot be told apart by hue, so this uses small multiples:
one panel per archetype, one hue per archetype, with the highest-volume members
directly labelled.

Archetypes are assigned by the rules stated in 3.4, applied to every journal:
  high-volume veteran   full-window coverage (first indexed year 2011) and >4,000 articles
  high-growth newcomer  first indexed after 2011 and >3x growth in annual output by 2025
  remaining journals    everything else (residual group; renamed from "stable specialists" in round 23)
NOTE: applying these rules reproducibly does NOT reproduce the journal lists given
in 4.2.1 - see the audit comment on that section.
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
OUT = ROOT / "figures" / "figure8_journal_archetypes.png"

ARCH = [
    ("High-volume veterans",  "#0072B2"),
    ("High-growth newcomers", "#D55E00"),
    ("Remaining journals",    "#009E73"),
]
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#dcdcdc"

SHORT = {
    "Journal of Medical Internet Research (JMIR)": "JMIR",
    "Computers in Biology and Medicine": "Comput Biol Med",
    "Briefings in Bioinformatics": "Brief Bioinform",
    "Bioinformatics": "Bioinformatics",
    "IEEE Journal of Biomedical and Health Informatics (J-BHI)": "IEEE J-BHI",
    "BMC Medical Informatics and Decision Making": "BMC MIDM",
    "Nature Medicine": "Nat Med",
    "Journal of the American Medical Informatics Association (JAMIA)": "JAMIA",
    "JMIR mHealth and uHealth": "JMIR mHealth",
    "Journal of Medical Systems": "J Med Syst",
    "Journal of Biomedical Informatics (JBI)": "JBI",
    "International Journal of Medical Informatics (IJMI)": "IJMI",
    "Nature Methods": "Nat Methods",
    "npj Digital Medicine": "npj Digit Med",
    "JMIR Medical Informatics (JMI)": "JMIR Med Inform",
    "Database: The Journal of Biological Databases and Curation": "Database",
    "Artificial Intelligence in Medicine": "Artif Intell Med",
    "Frontiers in Digital Health": "Front Digit Health",
    "Applied Clinical Informatics": "Appl Clin Inform",
    "Journal of Clinical and Translational Science": "J Clin Transl Sci",
    "Health Informatics Journal": "Health Inform J",
    "JAMIA Open": "JAMIA Open",
    "Bioinformatics Advances": "Bioinform Adv",
    "PLOS Digital Health": "PLOS Digit Health",
    "Methods of Information in Medicine": "Methods Inf Med",
    "The Lancet Digital Health": "Lancet Digit Health",
    "BMJ Health & Care Informatics": "BMJ HCI",
    "Digital Biomarkers": "Digit Biomark",
    "Journal of Innovation in Health Informatics": "JIHI",
}

df = pd.read_csv(ROOT / "data/pubmed_all_journals_2011_2025_w_keywords_patched.csv",
                 usecols=["journal", "year"], low_memory=False)
df = df[df.year.between(2011, 2025)]
piv = df.pivot_table(index="journal", columns="year", aggfunc=len, fill_value=0)
piv = piv.reindex(columns=range(2011, 2026), fill_value=0)
total = df.journal.value_counts()
first = df.groupby("journal").year.min()


def archetype(j):
    y, f = piv.loc[j], int(first[j])
    base = y[f] if y[f] > 0 else y[y > 0].iloc[0]
    growth = y[2025] / base if base > 0 else 0
    if f == 2011 and total[j] > 4000:
        return 0
    if f > 2011 and growth > 3:
        return 1
    return 2


groups = {0: [], 1: [], 2: []}
for j in total.index:
    groups[archetype(j)].append(j)

# 2x2 layout per reviewer request: veterans upper-left, newcomers lower-left,
# and the 18 stable specialists across the whole right half so every journal
# can be identified; each journal gets its own colour+marker combination.
years = list(range(2011, 2026))
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
           "#8C510A", "#5E3C99", "#1B9E77", "#E7298A", "#66A61E", "#A6611A",
           "#7570B3", "#D95F02", "#80B1D3", "#B2182B", "#4D4D4D", "#35978F"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "p", "h",
           "o", "s", "^", "D", "v", "P"]

fig = plt.figure(figsize=(15.5, 9.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], hspace=0.30, wspace=0.16)
ax_vet = fig.add_subplot(gs[0, 0])
ax_new = fig.add_subplot(gs[1, 0])
ax_sta = fig.add_subplot(gs[:, 1])
panel_axes = [ax_vet, ax_new, ax_sta]
letters = ["a", "b", "c"]

for ax, letter, (gi, (title, _col)) in zip(panel_axes, letters, enumerate(ARCH)):
    members = sorted(groups[gi], key=lambda j: -total[j])
    for rank, j in enumerate(members):
        y = piv.loc[j].reindex(years).to_numpy(dtype=float)
        y[y == 0] = float("nan")            # skip years before a journal is indexed
        ax.plot(years, y,
                color=PALETTE[rank % len(PALETTE)],
                marker=MARKERS[rank % len(MARKERS)],
                markersize=4.2, markevery=2, lw=1.5, alpha=0.9,
                label=SHORT.get(j, j[:18]), solid_capstyle="round")
    ax.set_title(f"({letter}) {title} — {len(members)} journals",
                 fontsize=11.5, fontweight="bold", color=INK, pad=8, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(2010.6, 2025.6)
    ax.set_xticks([2011, 2015, 2020, 2025])
    ax.tick_params(colors=MUTED, labelsize=9.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ncol = 1 if len(members) <= 4 else 2
    ax.legend(fontsize=7.6, ncol=ncol, frameon=False, loc="upper left",
              handlelength=2.2, labelspacing=0.35, columnspacing=0.9)

ax_vet.set_ylabel("Articles published per year", fontsize=11, color=INK)
ax_new.set_ylabel("Articles published per year", fontsize=11, color=INK)
ax_new.set_xlabel("Year", fontsize=11, color=INK)
ax_sta.set_xlabel("Year", fontsize=11, color=INK)

fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"wrote {OUT}")
for gi, (title, _) in enumerate(ARCH):
    print(f"  {title}: {len(groups[gi])} -> {[SHORT.get(j, j) for j in sorted(groups[gi], key=lambda x: -total[x])]}")
mirror_to_main(OUT)
