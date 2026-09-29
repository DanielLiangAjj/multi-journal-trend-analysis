"""Supplementary Figure S2 — per-journal coverage timeline (Gantt)."""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MAIN_MIRROR = Path("/Users/danielliang/Library/CloudStorage/Dropbox/multi_journal_trend_analysis")
OUT = ROOT / "figures" / "figure_supp_journal_timeline.png"

# Sub-domain colour mapping
SUBDOMAIN_COLOUR = {
    "clinical informatics": "#1f77b4",
    "digital/consumer health": "#ff7f0e",
    "comp. biology": "#2ca02c",
    "bridging methods": "#d62728",
}

# Same 29 journals as Supplementary Table S1
JOURNALS = [
    ("Journal of Medical Internet Research (JMIR)", "digital/consumer health"),
    ("Bioinformatics", "comp. biology"),
    ("Computers in Biology and Medicine", "bridging methods"),
    ("IEEE Journal of Biomedical and Health Informatics (J-BHI)", "bridging methods"),
    ("Briefings in Bioinformatics", "comp. biology"),
    ("BMC Medical Informatics and Decision Making", "clinical informatics"),
    ("Nature Medicine", "clinical informatics"),
    ("Journal of the American Medical Informatics Association (JAMIA)", "clinical informatics"),
    ("JMIR mHealth and uHealth", "digital/consumer health"),
    ("Journal of Medical Systems", "clinical informatics"),
    ("Journal of Biomedical Informatics (JBI)", "clinical informatics"),
    ("International Journal of Medical Informatics (IJMI)", "clinical informatics"),
    ("Nature Methods", "comp. biology"),
    ("npj Digital Medicine", "digital/consumer health"),
    ("JMIR Medical Informatics (JMI)", "digital/consumer health"),
    ("Database: The Journal of Biological Databases and Curation", "comp. biology"),
    ("Artificial Intelligence in Medicine", "bridging methods"),
    ("Frontiers in Digital Health", "digital/consumer health"),
    ("Applied Clinical Informatics", "clinical informatics"),
    ("Journal of Clinical and Translational Science", "clinical informatics"),
    ("Health Informatics Journal", "clinical informatics"),
    ("JAMIA Open", "clinical informatics"),
    ("Bioinformatics Advances", "comp. biology"),
    ("PLOS Digital Health", "digital/consumer health"),
    ("Methods of Information in Medicine", "clinical informatics"),
    ("The Lancet Digital Health", "digital/consumer health"),
    ("BMJ Health & Care Informatics", "clinical informatics"),
    ("Digital Biomarkers", "digital/consumer health"),
    ("Journal of Innovation in Health Informatics", "clinical informatics"),
]


def main():
    corp = pd.read_csv(ROOT / "data/pubmed_all_journals_2011_2025_w_keywords_patched.csv",
                       usecols=["journal", "year"])
    corp = corp[corp.year.between(2011, 2025)]
    g = corp.groupby("journal")["year"].agg(["min", "max", "count"]).to_dict("index")

    # Sort journals by start year, then by total count desc
    rows = []
    for name, sub in JOURNALS:
        if name not in g:
            continue
        d = g[name]
        rows.append({"name": name, "subdomain": sub,
                     "start": int(d["min"]), "end": int(d["max"]),
                     "count": int(d["count"])})
    rows.sort(key=lambda r: (r["start"], -r["count"]))

    n = len(rows)
    fig, ax = plt.subplots(figsize=(13, 0.34 * n + 2.8))

    for i, r in enumerate(rows):
        col = SUBDOMAIN_COLOUR[r["subdomain"]]
        # Bar from start to end
        ax.barh(i, r["end"] - r["start"] + 0.9, left=r["start"] - 0.45,
                height=0.6, color=col, alpha=0.78,
                edgecolor=col, linewidth=0.6, zorder=2)
        # Endpoint annotations
        ax.text(r["start"] - 0.30, i, f"{r['start']}",
                fontsize=7.0, ha="left", va="center", color="white", fontweight="bold", zorder=3)
        ax.text(r["end"] + 0.55, i, f"  n={r['count']:,}",
                fontsize=7.5, ha="left", va="center", color="#444")
        # Journal name
        ax.text(2010.0, i, r["name"], fontsize=8.4, ha="right", va="center",
                color="#222")

    ax.set_xlim(2007.5, 2030.0)
    ax.set_ylim(-0.7, n - 0.3)
    ax.invert_yaxis()
    ax.set_xticks(range(2011, 2026))
    ax.tick_params(axis="x", labelsize=9, rotation=45)
    ax.set_yticks([])
    ax.set_xlabel("Year", fontsize=11)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.3, lw=0.5, zorder=1)
    # Era bands
    ax.axvspan(2010.5, 2015.5, alpha=0.04, color="#1f77b4", zorder=0)
    ax.axvspan(2015.5, 2020.5, alpha=0.04, color="#2ca02c", zorder=0)
    ax.axvspan(2020.5, 2025.5, alpha=0.04, color="#d62728", zorder=0)

    # Sub-domain legend
    legend_handles = [mpatches.Patch(color=c, label=k) for k, c in SUBDOMAIN_COLOUR.items()]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        "Bars show each journal's first → last indexed year in our corpus; "
        "right-side n = total articles. Colour = sub-domain.",
        fontsize=12, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0.34, 0.04, 1.0, 0.95])
    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print(f"  wrote {OUT}")
    if MAIN_MIRROR.exists():
        import shutil
        shutil.copy2(OUT, MAIN_MIRROR / "figures" / OUT.name)
        print(f"  mirrored to MAIN")


if __name__ == "__main__":
    main()
