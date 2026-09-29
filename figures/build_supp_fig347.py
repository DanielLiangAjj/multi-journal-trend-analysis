"""Supplementary Figures 3, 4, and 7 rebuilds (reviewer round).

Supp Fig 3  — authorship dynamics as ONE image with subpanels (a)(b)(c)
              ordered to match the text (authors -> affiliations ->
              composition); affiliation panel starts at 2014.
Supp Fig 4  — per-journal team-science indicators recomputed on 2014-2025
              (pre-2014 affiliation metadata are sparse); prints the new
              numbers the manuscript text must cite.
Supp Fig 7  — journal-fit heat maps with methodology as panel (A) and
              health as panel (B) (they were reversed), no internal "Q6"
              titles.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _mirror import mirror_to_main

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
VIS = ROOT / "data" / "visualizations_k100"

ABBR = {
    "Journal of Medical Internet Research (JMIR)": "JMIR",
    "Computers in Biology and Medicine": "Comput Biol Med",
    "Briefings in Bioinformatics": "Brief Bioinform",
    "IEEE Journal of Biomedical and Health Informatics (J-BHI)": "IEEE J-BHI",
    "BMC Medical Informatics and Decision Making": "BMC Med Inform",
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
    "Bioinformatics": "Bioinformatics",
}


def supp3():
    df = pd.read_csv(VIS / "pi_q2_authorship_stats.csv")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    (a, b, c) = axes

    a.plot(df.year, df.mean_authors, "o-", color="#1f4ea0", lw=2, label="Mean")
    a.plot(df.year, df.median_authors, "s--", color="#e07b39", lw=2, label="Median")
    a.set_title("(a) Authors per paper", fontsize=12.5, fontweight="bold", loc="left")
    a.set_ylabel("Authors per paper", fontsize=11)

    d14 = df[df.year >= 2014]
    b.plot(d14.year, d14.mean_affs, "o-", color="#1f4ea0", lw=2, label="Mean")
    b.plot(d14.year, d14.median_affs, "s--", color="#e07b39", lw=2, label="Median")
    b.set_title("(b) Distinct affiliations per paper", fontsize=12.5,
                fontweight="bold", loc="left")
    b.set_ylabel("Affiliations per paper", fontsize=11)
    b.text(0.98, 0.04, "2014–2025 only:\npre-2014 affiliation\nmetadata are sparse",
           transform=b.transAxes, fontsize=8.5, va="bottom", ha="right",
           style="italic", color="#666666")

    c.plot(df.year, df.pct_interdisciplinary, "o-", color="#1f4ea0", lw=2,
           label="Interdisciplinary (both)")
    c.plot(df.year, df.pct_clinical_only, "s--", color="#e07b39", lw=2,
           label="Clinical only")
    c.plot(df.year, df.pct_computational_only, "^:", color="#2ca02c", lw=2,
           label="Computational only")
    c.plot(df.year, df.pct_neither, "d-.", color="#999999", lw=1.6,
           label="Neither or unknown")
    c.axvspan(2010.5, 2013.5, color="#bbbbbb", alpha=0.18, zorder=0)
    c.text(2010.7, 40, "sparse\nmetadata", fontsize=8, color="#777777", style="italic")
    c.set_title("(c) Team composition", fontsize=12.5, fontweight="bold", loc="left")
    c.set_ylabel("% of papers", fontsize=11)

    for ax in axes:
        ax.set_xlabel("Year", fontsize=11)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9, frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = ROOT / "figures" / "supp_figure3_authorship.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")
    mirror_to_main(str(out))


def supp4():
    au = pd.read_csv(ROOT / "data/articles_with_authors.csv")
    corpus = pd.read_csv(ROOT / "data/pubmed_all_journals_2011_2025_w_keywords_patched.csv",
                         usecols=["pmid", "journal", "year"])
    m = au.merge(corpus, on="pmid", how="inner")
    m = m[m.year.between(2014, 2025)]
    EXCLUDE = {"IEEE Journal of Biomedical and Health Informatics (J-BHI)"}

    rows = []
    for j, g in m.groupby("journal"):
        if len(g) < 100 or j in EXCLUDE:
            continue
        gi = g.dropna(subset=["is_interdisciplinary"])
        rows.append({"journal": j, "n": len(g),
                     "mean_authors": g.n_authors.mean(),
                     "mean_affs": g.n_unique_affiliations.mean(),
                     "pct_inter": 100 * gi.is_interdisciplinary.mean()})
    t = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 7.2))
    specs = [("mean_authors", "Mean authors/article", "#3b8bd4", "{:.1f}"),
             ("mean_affs", "Mean distinct affiliations/article", "#1b9e77", "{:.1f}"),
             ("pct_inter", "% interdisciplinary articles", "#9467bd", "{:.1f}%")]
    for ax, letter, (col, label, colr, fmt) in zip(axes, "abc", specs):
        top = t.sort_values(col, ascending=False).head(20).iloc[::-1]
        ax.barh([ABBR.get(j, j) for j in top.journal], top[col], color=colr,
                edgecolor="white")
        for i, v in enumerate(top[col]):
            ax.text(v * 1.01, i, fmt.format(v), va="center", fontsize=8)
        ax.set_title(f"({letter}) {label}", fontsize=11.5, fontweight="bold",
                     loc="left")
        ax.tick_params(labelsize=8.5)
        ax.margins(x=0.12)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.text(0.995, 0.005,
             "2014–2025; journals with ≥100 articles; IEEE J-BHI excluded "
             "(low PubMed affiliation coverage)",
             fontsize=8.5, style="italic", color="#666666", ha="right")
    fig.tight_layout()
    out = ROOT / "figures" / "supp_figure4_journal_teamscience.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")
    print("  top-5 interdisciplinary (2014-2025): " + "; ".join(
        f"{ABBR.get(r.journal, r.journal)} {r.pct_inter:.1f}%"
        for _, r in t.sort_values('pct_inter', ascending=False).head(5).iterrows()))
    print("  top-3 mean team size (2014-2025): " + "; ".join(
        f"{ABBR.get(r.journal, r.journal)} {r.mean_authors:.1f}"
        for _, r in t.sort_values('mean_authors', ascending=False).head(3).iterrows()))
    print("  top-3 mean affiliations (2014-2025): " + "; ".join(
        f"{ABBR.get(r.journal, r.journal)} {r.mean_affs:.1f}"
        for _, r in t.sort_values('mean_affs', ascending=False).head(3).iterrows()))
    mirror_to_main(str(out))


def journal_topic_share(domain):
    topics = json.load(open(ROOT / f"data/hierarchy_k100/{domain}_final_topics.json"))
    kw2t = defaultdict(set)
    for d in topics.values():
        for k in d.get("keywords", []):
            kw2t[k.strip().lower()].add(d["name"])
    jt = defaultdict(Counter)
    jn = Counter()
    for chunk in pd.read_csv(ROOT / "data/pubmed_all_journals_2011_2025_w_keywords_patched.csv",
                             usecols=["journal", "year", "keywords"], chunksize=50_000):
        chunk = chunk.dropna(subset=["keywords", "year", "journal"])
        chunk = chunk[chunk["year"].between(2011, 2025)]
        for j, kws in zip(chunk["journal"], chunk["keywords"].astype(str)):
            jn[j] += 1
            ts = set()
            for kw in kws.split(";"):
                ts |= kw2t.get(kw.strip().lower(), set())
            for tt in ts:
                jt[j][tt] += 1
    return jt, jn


def supp7():
    for domain, letter in (("methodology", "A"), ("health", "B")):
        jt, jn = journal_topic_share(domain)
        top_j = [j for j, _ in jn.most_common(15)]
        topic_tot = Counter()
        for j in top_j:
            topic_tot.update(jt[j])
        all_tot = Counter()
        for j in jn:
            all_tot.update(jt[j])
        top_t = [t for t, _ in all_tot.most_common(20)]
        M = np.zeros((len(top_j), len(top_t)))
        for i, j in enumerate(top_j):
            for k, t in enumerate(top_t):
                M[i, k] = 100 * jt[j].get(t, 0) / max(1, jn[j])
        fig, ax = plt.subplots(figsize=(16, 9))
        im = ax.imshow(M, aspect="auto", cmap="YlGnBu",
                       vmin=0, vmax=max(60, M.max()))
        ax.set_xticks(range(len(top_t)))
        ax.set_xticklabels(top_t, rotation=45, ha="right", fontsize=9.5)
        ax.set_yticks(range(len(top_j)))
        ax.set_yticklabels([ABBR.get(j, j) for j in top_j], fontsize=10)
        for i in range(M.shape[0]):
            for k in range(M.shape[1]):
                v = M[i, k]
                ax.text(k, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=7.2,
                        color="white" if v > 0.55 * max(60, M.max()) else "#333")
        ax.set_title(f"({letter}) {domain.capitalize()} topics — share of each "
                     f"journal's articles (%)", fontsize=12.5,
                     fontweight="bold", loc="left", pad=8)
        ax.set_xticks(np.arange(-0.5, len(top_t), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(top_j), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.015)
        cb.set_label("% of journal's articles carrying the topic", fontsize=10)
        fig.tight_layout()
        out = ROOT / "figures" / f"supp_figure7_journalfit_{domain}.png"
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {out}")
        mirror_to_main(str(out))


if __name__ == "__main__":
    supp3()
    supp4()
    supp7()
