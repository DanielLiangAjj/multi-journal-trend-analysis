"""Figure 13 — clustered journal × topic specialisation heat-map.

Replaces the original `n2_*_journal_specialization.png` (arbitrary ordering,
no visible block structure) with a hierarchically-clustered heat-map that
groups similar journals together so the structural blocks the prose claims
(Bioinformatics-cluster / CBM / JMIR concentration triangles) are visually
obvious.

Two panels: methodology + health.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list

ROOT = Path(__file__).resolve().parent.parent
MAIN_MIRROR = Path("/Users/danielliang/Library/CloudStorage/Dropbox/multi_journal_trend_analysis")
OUT = ROOT / "figures" / "figure13_journal_topic_specialization.png"

YEARS = list(range(2011, 2026))


def load_journal_topic_share(domain):
    """Return a [journal x topic] matrix where each row sums to 1
    (each cell is the share of that journal's articles assigned to that topic)."""
    spec = pd.read_csv(ROOT / f"data/visualizations_k100/pi_q5_{domain}_specialization.csv")
    # We need a journal x topic share matrix; pi_q5 only has per-topic max-journal info.
    # Re-build from corpus + topic keyword mappings.
    import json
    from collections import defaultdict, Counter
    topics = json.load(open(
        ROOT / f"data/hierarchy_k100/{domain}_final_topics.json"))
    kw_to_topics = defaultdict(set)
    for tid, dat in topics.items():
        name = dat["name"]
        for kw in dat.get("keywords", []):
            kw_to_topics[kw.strip().lower()].add(name)

    # Walk the corpus
    journal_topic = defaultdict(Counter)
    journal_total = Counter()
    for chunk in pd.read_csv(ROOT / "data/pubmed_all_journals_2011_2025_w_keywords_patched.csv",
                             usecols=["journal", "year", "keywords"], chunksize=50_000):
        chunk = chunk.dropna(subset=["keywords", "year", "journal"])
        chunk = chunk[chunk["year"].between(2011, 2025)]
        for j, kws in zip(chunk["journal"], chunk["keywords"].astype(str)):
            tset = set()
            for kw in kws.split(";"):
                k = kw.strip().lower()
                if k in kw_to_topics:
                    tset |= kw_to_topics[k]
            journal_total[j] += 1
            for t in tset:
                journal_topic[j][t] += 1

    journals = sorted(journal_total.keys())
    topic_names = sorted({t for ctr in journal_topic.values() for t in ctr})
    M = np.zeros((len(journals), len(topic_names)))
    for i, j in enumerate(journals):
        denom = max(1, journal_total[j])
        for k, t in enumerate(topic_names):
            M[i, k] = journal_topic[j].get(t, 0) / denom
    return journals, topic_names, M


JOURNAL_ABBR = {
    "Journal of Medical Internet Research (JMIR)": "JMIR",
    "Bioinformatics": "Bioinformatics",
    "Computers in Biology and Medicine": "CBM",
    "IEEE Journal of Biomedical and Health Informatics (J-BHI)": "IEEE J-BHI",
    "Briefings in Bioinformatics": "Brief Bioinform",
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
    "BMJ Health & Care Informatics": "BMJ Health Care",
    "Digital Biomarkers": "Digit Biomark",
    "Journal of Innovation in Health Informatics": "J Innov Health Inform",
}


def cluster_order(M, axis):
    """Return the leaf order of hierarchical clustering of M along axis."""
    if axis == 0:
        Z = linkage(M, method="average", metric="cosine")
    else:
        Z = linkage(M.T, method="average", metric="cosine")
    return leaves_list(Z)


def panel(ax, journals, topics, M, title, top_n_topics=20):
    """Filter to the top-N topics by total volume, then compute log2 enrichment
    (journal_share / field_average_share) and render with divergent colormap."""
    # Filter: keep top-N topics by sum across journals
    topic_totals = M.sum(axis=0)
    keep = np.argsort(topic_totals)[-top_n_topics:][::-1]
    M_f = M[:, keep]
    topics_f = [topics[i] for i in keep]

    # Field-average share per topic (mean across journals, weighted equally)
    expected = M_f.mean(axis=0, keepdims=True)
    expected = np.where(expected < 1e-6, 1e-6, expected)
    # Enrichment: log2(observed / expected)
    M_enr = np.log2(np.where(M_f < 1e-6, 1e-6, M_f) / expected)
    # Clip extreme values for legibility
    M_enr = np.clip(M_enr, -3, 3)

    j_order = cluster_order(M_enr, axis=0)
    t_order = cluster_order(M_enr, axis=1)
    M2 = M_enr[j_order][:, t_order]
    journals2 = [JOURNAL_ABBR.get(journals[i], journals[i]) for i in j_order]
    topics2 = [topics_f[i] for i in t_order]

    im = ax.imshow(M2, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_xticks(range(len(topics2)))
    ax.set_xticklabels(topics2, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(journals2)))
    ax.set_yticklabels(journals2, fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    # Annotate cells with strong enrichment (|log2| ≥ 1, i.e., ≥2× or ≤½×)
    for i in range(M2.shape[0]):
        for j in range(M2.shape[1]):
            v = M2[i, j]
            if abs(v) >= 1.0:
                colour = "white" if abs(v) >= 1.6 else "#222"
                # Show as fold-change: 2^v
                fold = 2 ** v
                if fold >= 10:
                    label = f"{fold:.0f}×"
                elif fold >= 1.5:
                    label = f"{fold:.1f}×"
                elif fold >= 0.5:
                    label = f".{int(round(fold*10))}×"
                else:
                    label = f"{fold:.2f}×"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=6.5, color=colour, fontweight="bold")
    # Faint grid between cells
    ax.set_xticks(np.arange(-0.5, len(topics2), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(journals2), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im


def main():
    fig, axes = plt.subplots(2, 1, figsize=(16, 22),
                              gridspec_kw={"hspace": 0.55})

    print("Building methodology heat-map (top 20 topics)…")
    j_m, t_m, M_m = load_journal_topic_share("methodology")
    im1 = panel(axes[0], j_m, t_m, M_m,
                "(A) Methodology — top 20 topics × 29 journals (over-/under-representation, K=100)",
                top_n_topics=20)
    cb1 = fig.colorbar(im1, ax=axes[0], shrink=0.85, pad=0.02,
                       ticks=[-3, -2, -1, 0, 1, 2, 3])
    cb1.ax.set_yticklabels(["≤⅛×", "¼×", "½×", "1× (field avg)", "2×", "4×", "≥8×"])
    cb1.set_label("Enrichment vs field average\n(log₂ scale)", fontsize=9.5)

    print("Building health heat-map (top 20 topics)…")
    j_h, t_h, M_h = load_journal_topic_share("health")
    im2 = panel(axes[1], j_h, t_h, M_h,
                "(B) Health — top 20 topics × 29 journals (over-/under-representation, K=100)",
                top_n_topics=20)
    cb2 = fig.colorbar(im2, ax=axes[1], shrink=0.85, pad=0.02,
                       ticks=[-3, -2, -1, 0, 1, 2, 3])
    cb2.ax.set_yticklabels(["≤⅛×", "¼×", "½×", "1× (field avg)", "2×", "4×", "≥8×"])
    cb2.set_label("Enrichment vs field average\n(log₂ scale)", fontsize=9.5)

    fig.suptitle(
        "Journal × topic enrichment heat-map (K=100)\n"
        "Cell = log₂(journal's share for topic ÷ field-average share for topic).\n"
        "RED = over-represented in this journal vs the field average; "
        "BLUE = under-represented.\n"
        "Rows and columns hierarchically clustered to surface journal sub-communities; "
        "cells with ≥2× or ≤½× enrichment are annotated.",
        fontsize=12.5, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT, dpi=400, facecolor="white")
    plt.close(fig)
    print(f"  wrote {OUT}")
    if MAIN_MIRROR.exists():
        import shutil
        shutil.copy2(OUT, MAIN_MIRROR / "figures" / OUT.name)
        print(f"  mirrored to MAIN")


if __name__ == "__main__":
    main()
