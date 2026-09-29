"""Figure 13 (manuscript Figure 7) — clustered journal × topic specialisation heat-map.

2026-09-29: block labels (i)–(iv) moved out of the grid into the right margin; one bracket per contiguous run of
member rows (interleaved non-members unbracketed); fonts enlarged for print scale; 13.5 × 19.5 in; bbox_inches="tight".
Drop this file over figures/build_figure13_n2_heatmap.py in the repo to reproduce figures/Figure_7.png.

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
    load_journal_topic_share.last_totals = np.array([journal_total[j] for j in journals], dtype=float)
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

    # Field-average share per topic = unweighted mean of the 29 journals' shares (each journal counts
    # equally, i.e. the "typical journal"); the Methods text describes this definition explicitly (round 23).
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
    ax.set_xticklabels(topics2, rotation=45, ha="right", fontsize=12)
    ax.set_yticks(range(len(journals2)))
    ax.set_yticklabels(journals2, fontsize=12)
    ax.set_title(title, fontsize=14.5, fontweight="bold", pad=10, loc="left")
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
                        fontsize=7.6, color=colour, fontweight="bold")
    # Faint grid between cells
    ax.set_xticks(np.arange(-0.5, len(topics2), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(journals2), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im, journals2, topics2


def draw_block(ax, journals2, topics2, jset, tset, label, color="#111111"):
    """Mark the named journal x topic block in the clustered layout. If the members are contiguous
    (at most one non-member row and column inside their bounding box) draw one dashed box; otherwise
    (round 23: the clustering interleaves other journals) outline the member cells individually."""
    import matplotlib.patches as mpatches
    jab = {JOURNAL_ABBR.get(j, j) for j in jset} | set(jset)
    rows = [i for i, j in enumerate(journals2) if j in jab]
    cols = [i for i, t in enumerate(topics2) if t in tset]
    if not rows or not cols:
        print(f"   block {label}: MISSING members (rows={rows}, cols={cols})")
        return
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    gaps_r = (r1 - r0 + 1) - len(rows); gaps_c = (c1 - c0 + 1) - len(cols)
    print(f"   block {label}: rows {r0}-{r1} ({len(rows)} named, {gaps_r} interlopers), cols {c0}-{c1} ({len(cols)} named, {gaps_c} interlopers)")
    if gaps_r <= 1 and gaps_c <= 1:
        rect = mpatches.Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0 + 1, r1 - r0 + 1,
                                  fill=False, edgecolor=color, linewidth=2.6,
                                  linestyle=(0, (4, 2)), zorder=6)
        ax.add_patch(rect)
        lx, ly = c0 - 0.42, r0 - 0.62
    else:
        for r in rows:
            for cidx in cols:
                ax.add_patch(mpatches.Rectangle((cidx - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=color,
                                                linewidth=2.0, linestyle=(0, (3, 1.5)), zorder=6))
        lx, ly = min(cols) - 0.42, min(rows) - 0.62
    # Label outside the grid (right margin). One bracket per contiguous run of MEMBER rows (interleaved
    # non-member rows are left unbracketed); the label sits beside the longest run. No cell is covered.
    ncols = len(topics2)
    x0 = ncols - 0.5 + 0.30
    runs = []
    for r in sorted(rows):
        if runs and r == runs[-1][1] + 1: runs[-1][1] = r
        else: runs.append([r, r])
    for r0_, r1_ in runs:
        ax.plot([x0, x0], [r0_ - 0.5, r1_ + 0.5], color=color, linewidth=2.6, solid_capstyle="butt",
                clip_on=False, zorder=7)
        for yy in (r0_ - 0.5, r1_ + 0.5):
            ax.plot([x0 - 0.18, x0], [yy, yy], color=color, linewidth=2.6, clip_on=False, zorder=7)
    lr0, lr1 = max(runs, key=lambda q: q[1] - q[0])
    ax.text(x0 + 0.28, (lr0 + lr1) / 2.0, label, fontsize=13, fontweight="bold",
            color=color, ha="left", va="center", clip_on=False, zorder=7)


def main():
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 19.5),
                              gridspec_kw={"hspace": 0.50})

    COMPBIO = {"Bioinformatics", "Bioinformatics Advances",
               "Briefings in Bioinformatics",
               "Database: The Journal of Biological Databases and Curation",
               "Nature Methods"}
    # Digital/consumer-health blocks use the journals that the clustering
    # actually places adjacently in each panel (npj Digit Med and Digital
    # Biomarkers cluster elsewhere, so boxing them would span the panel).
    JMIRFAM_M = {"Journal of Medical Internet Research (JMIR)",
                 "JMIR mHealth and uHealth", "JMIR Medical Informatics (JMI)",
                 "Frontiers in Digital Health"}
    JMIRFAM_H = {"Journal of Medical Internet Research (JMIR)",
                 "JMIR mHealth and uHealth", "JMIR Medical Informatics (JMI)",
                 "Frontiers in Digital Health", "JAMIA Open"}

    print("Building methodology heat-map (top 20 topics)…")
    j_m, t_m, M_m = load_journal_topic_share("methodology")
    im1, j2_m, t2_m = panel(
        axes[0], j_m, t_m, M_m,
        "(a) Methodology topics (top 20 by volume)",
        top_n_topics=20)
    cb1 = fig.colorbar(im1, ax=axes[0], shrink=0.85, pad=0.075,
                       ticks=[-3, -2, -1, 0, 1, 2, 3])
    cb1.ax.set_yticklabels(["≤⅛×", "¼×", "½×", "1× (field avg)", "2×", "4×", "≥8×"])
    cb1.set_label("Enrichment vs field average\n(log₂ scale)", fontsize=11.5); cb1.ax.tick_params(labelsize=11)
    # Block outlines (reviewer request): (i) computational-biology block and
    # (iii) digital/consumer-health block in the methodology panel.
    draw_block(axes[0], j2_m, t2_m, COMPBIO,
               {"Software and usability research",
                "Genomics and sequence bioinformatics", "Statistical methods"},
               "(iv)", color="#111111")
    draw_block(axes[0], j2_m, t2_m, JMIRFAM_M,
               {"Mobile health and social media", "Information dissemination",
                "eHealth and mobile applications"},
               "(ii)", color="#5E3C99")
    # (iii) AI / signal- and image-processing block (round-23 reviewer request)
    AIBLOCK = {"Artificial Intelligence in Medicine",
               "Computers in Biology and Medicine",
               "IEEE Journal of Biomedical and Health Informatics (J-BHI)"}
    draw_block(axes[0], j2_m, t2_m, AIBLOCK,
               {"Medical image segmentation", "Biomedical signal processing",
                "Mixed deep learning applications", "Knowledge graphs"},
               "(iii)", color="#B2182B")

    print("Building health heat-map (top 20 topics)…")
    j_h, t_h, M_h = load_journal_topic_share("health")
    im2, j2_h, t2_h = panel(
        axes[1], j_h, t_h, M_h,
        "(b) Health topics (top 20 by volume)",
        top_n_topics=20)
    cb2 = fig.colorbar(im2, ax=axes[1], shrink=0.85, pad=0.075,
                       ticks=[-3, -2, -1, 0, 1, 2, 3])
    cb2.ax.set_yticklabels(["≤⅛×", "¼×", "½×", "1× (field avg)", "2×", "4×", "≥8×"])
    cb2.set_label("Enrichment vs field average\n(log₂ scale)", fontsize=11.5); cb2.ax.tick_params(labelsize=11)
    # (ii) gene-and-variant block and (iii) digital/consumer-health block.
    draw_block(axes[1], j2_h, t2_h, COMPBIO,
               {"Genomic variant analysis", "Gene expression regulation",
                "Protein and gene networks", "Computational cancer biology"},
               "(i)", color="#111111")
    draw_block(axes[1], j2_h, t2_h, JMIRFAM_H,
               {"Psychiatric disorders", "Digital health technologies",
                "Health behavior change"},
               "(ii)", color="#5E3C99")

    fig.tight_layout()
    fig.savefig(OUT, dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  wrote {OUT}")
    if MAIN_MIRROR.exists():
        import shutil
        shutil.copy2(OUT, MAIN_MIRROR / "figures" / OUT.name)
        print(f"  mirrored to MAIN")


if __name__ == "__main__":
    main()
