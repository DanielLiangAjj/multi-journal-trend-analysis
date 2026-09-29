#!/usr/bin/env python3
"""Citation-influence layer (round 23, PI checklist): NIH iCite metrics joined to the K=100 topic assignments.

Outputs in data/citation_influence/:
  topic_influence.csv        per topic: n articles with RCR, median/mean RCR, % RCR>=2, total citations, share trend (from visualizations_k100)
  top_papers_by_rcr.csv      top 25 papers by RCR (2011-2023 window) with their topics
  top_papers_per_topic.csv   top 3 RCR papers for each of the 20 largest topics per domain
  pioneer_citation_weighted.csv  citation-weighted pioneer journals (journal whose early articles on a topic gathered most citations)
  journal_citation_network.csv   journal x journal within-corpus citation counts; betweenness on the citation graph
  bridging_papers.csv        corpus papers cited from the largest number of distinct K=100 topics (within-corpus citations)
  summary.json               headline numbers used in the manuscript
Figure: figures/supp_figure10_citation_influence.png (a: attention vs impact; b: citation-weighted vs first-to-publish pioneer ranking)
"""
import os, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parent
D = ROOT / "data"; V = D / "visualizations_k100"; OUT = D / "citation_influence"; OUT.mkdir(exist_ok=True)
YEARS = list(range(2011, 2026)); Y = [str(y) for y in YEARS]
RCR_MAX_YEAR = 2023   # RCR needs >= 2 full years of citations to be meaningful

# ---- corpus + memberships ----
c = pd.read_csv(D / "pubmed_all_journals_2011_2025_w_keywords_patched.csv", dtype={"pmid": str})
c["year"] = pd.to_numeric(c.year, errors="coerce"); c = c[c.year.between(2011, 2025)].copy(); c["year"] = c.year.astype(int)
c["kwlist"] = c.keywords.map(lambda s: [k.strip().lower() for k in str(s).split(";") if k.strip()] if isinstance(s, str) and s else [])
kw2 = {}
for dom in ["methodology", "health"]:
    t = json.load(open(D / "hierarchy_k100" / f"{dom}_final_topics.json")); m = defaultdict(set)
    for tid, v in t.items():
        for kw in v.get("keywords", []): m[kw.strip().lower()].add(v["name"])
    kw2[dom] = m
c["mt"] = c.kwlist.map(lambda l: set().union(*[kw2["methodology"].get(k, set()) for k in l]) if l else set())
c["ht"] = c.kwlist.map(lambda l: set().union(*[kw2["health"].get(k, set()) for k in l]) if l else set())
c["topics"] = [m | h for m, h in zip(c.mt, c.ht)]

# ---- iCite ----
ic = pd.read_csv(D / "icite_metrics.csv", dtype={"pmid": str})
c = c.merge(ic[["pmid", "citation_count", "relative_citation_ratio", "is_research_article"]], on="pmid", how="left")
c["rcr"] = pd.to_numeric(c.relative_citation_ratio, errors="coerce")
c["cites"] = pd.to_numeric(c.citation_count, errors="coerce").fillna(0)
summary = {"n_with_metrics": int(c.citation_count.notna().sum()), "n_with_rcr": int(c.rcr.notna().sum()),
           "n_with_rcr_to_2023": int(c[c.year <= RCR_MAX_YEAR].rcr.notna().sum()), "median_rcr_all": float(c.rcr.median()),
           "pct_rcr_ge1": float((c.rcr >= 1).mean() * 100), "pct_rcr_ge2": float((c.rcr >= 2).mean() * 100), "total_citations": int(c.cites.sum())}
rcr_year = c.groupby("year").rcr.median().round(2); summary["median_rcr_by_year"] = {int(k): float(v) for k, v in rcr_year.items()}

# ---- per-topic influence ----
sub = c[(c.year <= RCR_MAX_YEAR) & c.rcr.notna()]
rows = []
for dom in ["methodology", "health"]:
    col = "mt" if dom == "methodology" else "ht"
    st = pd.read_csv(V / f"{dom}_share_trends.csv").set_index("topic")
    if "share_logslope" in st.columns: st["pct"] = (np.exp(st.share_logslope) - 1) * 100
    tyc = pd.read_csv(V / f"{dom}_topic_year_counts.csv").set_index("topic"); tyc["n"] = tyc[Y].sum(axis=1)
    for tp in tyc.index:
        s = sub[sub[col].map(lambda x: tp in x)]
        if len(s) < 30: continue
        rows.append(dict(domain=dom, topic=tp, n_articles_2011_2025=int(tyc.n[tp]), n_with_rcr_2011_2023=int(len(s)), median_rcr=round(s.rcr.median(), 2), mean_rcr=round(s.rcr.mean(), 2),
                         pct_rcr_ge2=round((s.rcr >= 2).mean() * 100, 1), total_citations=int(s.cites.sum()), citations_per_article=round(s.cites.mean(), 1),
                         share_trend_pct_per_yr=round(float(st.pct.get(tp, np.nan)), 1) if tp in st.index else None, share_trend_sig=bool(st.sig.get(tp, False)) if tp in st.index else None))
ti = pd.DataFrame(rows); ti.to_csv(OUT / "topic_influence.csv", index=False)
summary["topic_influence"] = {}
for dom in ["methodology", "health"]:
    d = ti[ti.domain == dom]
    summary["topic_influence"][dom] = dict(top_median_rcr=[(r.topic, r.median_rcr, int(r.n_with_rcr_2011_2023)) for _, r in d.nlargest(6, "median_rcr").iterrows()],
                                          bottom_median_rcr=[(r.topic, r.median_rcr, int(r.n_with_rcr_2011_2023)) for _, r in d.nsmallest(6, "median_rcr").iterrows()],
                                          corr_median_rcr_vs_share_trend=round(float(d.median_rcr.corr(d.share_trend_pct_per_yr)), 2),
                                          corr_median_rcr_vs_log_volume=round(float(d.median_rcr.corr(np.log(d.n_articles_2011_2025))), 2),
                                          median_rcr_overall=round(float(d.median_rcr.median()), 2))
    # gainers vs losers
    g = d[(d.share_trend_pct_per_yr > 0) & (d.share_trend_sig == True)]; l = d[(d.share_trend_pct_per_yr < 0) & (d.share_trend_sig == True)]
    summary["topic_influence"][dom]["median_rcr_share_gainers"] = round(float(g.median_rcr.median()), 2) if len(g) else None
    summary["topic_influence"][dom]["median_rcr_share_losers"] = round(float(l.median_rcr.median()), 2) if len(l) else None
    summary["topic_influence"][dom]["median_rcr_nonsig"] = round(float(d[d.share_trend_sig == False].median_rcr.median()), 2)

# ---- top papers ----
tp_cols = ["pmid", "year", "journal", "title", "rcr", "cites"]
top = c[c.year <= RCR_MAX_YEAR].nlargest(25, "rcr")[tp_cols].copy(); top["topics"] = [", ".join(sorted(t))[:200] for t in c.loc[top.index, "topics"]]
top.to_csv(OUT / "top_papers_by_rcr.csv", index=False)
summary["top_papers"] = [(r.pmid, int(r.year), r.journal, r.title[:90], round(r.rcr, 1), int(r.cites)) for _, r in top.head(10).iterrows()]
rows = []
for dom in ["methodology", "health"]:
    col = "mt" if dom == "methodology" else "ht"; d = ti[ti.domain == dom].nlargest(20, "n_articles_2011_2025")
    for tp in d.topic:
        s = c[(c.year <= RCR_MAX_YEAR) & c[col].map(lambda x: tp in x)].nlargest(3, "rcr")
        for _, r in s.iterrows(): rows.append(dict(domain=dom, topic=tp, pmid=r.pmid, year=int(r.year), journal=r.journal, title=r.title, rcr=round(r.rcr, 1), citations=int(r.cites)))
pd.DataFrame(rows).to_csv(OUT / "top_papers_per_topic.csv", index=False)

# ---- citation-weighted pioneers ----
q1 = pd.concat([pd.read_csv(V / f"pi_q1_{dom}_first_emergence.csv").assign(domain=dom) for dom in ["methodology", "health"]])
rows = []
for dom in ["methodology", "health"]:
    col = "mt" if dom == "methodology" else "ht"
    for tp in q1[q1.domain == dom].topic:
        s = c[c[col].map(lambda x: tp in x)]
        if len(s) == 0: continue
        y0 = int(s.year.min()); early = s[s.year <= y0 + 2]
        jc = early.groupby("journal").cites.sum().sort_values(ascending=False)
        first_j = q1[(q1.domain == dom) & (q1.topic == tp)].first_journal.iloc[0]
        rows.append(dict(domain=dom, topic=tp, first_year=y0, first_journal=first_j, citation_weighted_pioneer=jc.index[0], early_citations_leader=int(jc.iloc[0]), early_articles=int(len(early)), same=bool(jc.index[0] == first_j)))
pw = pd.DataFrame(rows); pw.to_csv(OUT / "pioneer_citation_weighted.csv", index=False)
cw_rank = pw[pw.domain == "methodology"].citation_weighted_pioneer.value_counts(); ft_rank = pw[pw.domain == "methodology"].first_journal.value_counts()
summary["pioneers_methodology"] = dict(citation_weighted_top=[(j[:45], int(n)) for j, n in cw_rank.head(6).items()], first_to_publish_top=[(j[:45], int(n)) for j, n in ft_rank.head(6).items()],
                                       agreement_pct=round(float(pw[pw.domain == "methodology"].same.mean() * 100), 1), n_topics=int((pw.domain == "methodology").sum()))
cw_rank_h = pw[pw.domain == "health"].citation_weighted_pioneer.value_counts(); ft_rank_h = pw[pw.domain == "health"].first_journal.value_counts()
summary["pioneers_health"] = dict(citation_weighted_top=[(j[:45], int(n)) for j, n in cw_rank_h.head(6).items()], first_to_publish_top=[(j[:45], int(n)) for j, n in ft_rank_h.head(6).items()],
                                  agreement_pct=round(float(pw[pw.domain == "health"].same.mean() * 100), 1), n_topics=int((pw.domain == "health").sum()))

# ---- within-corpus citation network ----
links = json.load(open(D / "icite_links.json"))
pm2j = dict(zip(c.pmid, c.journal)); pm2t = dict(zip(c.pmid, c.topics)); pm2y = dict(zip(c.pmid, c.year))
corpus = set(c.pmid)
n_refs_total = 0; n_refs_in = 0
JJ = Counter(); cited_by_topics = defaultdict(set); in_cites = Counter()
for p in c.pmid:
    L = links.get(p);
    if not L: continue
    refs = [str(x) for x in L.get("references", [])]
    n_refs_total += len(refs)
    for r in refs:
        if r in corpus:
            n_refs_in += 1; JJ[(pm2j[p], pm2j[r])] += 1; in_cites[r] += 1
            for t in pm2t[p]: cited_by_topics[r].add(t)
summary["citation_network"] = dict(references_total=n_refs_total, references_within_corpus=n_refs_in, pct_within=round(n_refs_in / max(1, n_refs_total) * 100, 1))
journals = sorted(c.journal.unique())
jm = pd.DataFrame(0, index=journals, columns=journals)
for (a, b), n in JJ.items(): jm.loc[a, b] = n
jm.to_csv(OUT / "journal_citation_matrix.csv")
# undirected weighted graph of cross-journal citations (excluding self-citations), normalised by the two journals' sizes
G = nx.Graph(); size = c.journal.value_counts()
for a in journals:
    G.add_node(a, size=int(size[a]))
for a in journals:
    for b in journals:
        if a < b:
            w = JJ[(a, b)] + JJ[(b, a)]
            if w > 0: G.add_edge(a, b, weight=w / np.sqrt(size[a] * size[b]), raw=w)
bc = nx.betweenness_centrality(G, weight=lambda u, v, d: 1.0 / (d["weight"] + 1e-9))
summary["citation_network"]["betweenness_top"] = [(j[:45], round(v, 3)) for j, v in sorted(bc.items(), key=lambda x: -x[1])[:6]]
strength = {j: sum(d["raw"] for _, _, d in G.edges(j, data=True)) for j in G.nodes}
summary["citation_network"]["cross_journal_citations_top"] = [(j[:45], int(v)) for j, v in sorted(strength.items(), key=lambda x: -x[1])[:6]]
self_share = {j: JJ[(j, j)] / max(1, sum(n for (a, b), n in JJ.items() if a == j)) for j in journals}
summary["citation_network"]["self_citation_share_median"] = round(float(np.median(list(self_share.values()))) * 100, 1)
# bridging papers: cited from the most distinct topics
rows = []
for p, ts in cited_by_topics.items():
    rows.append(dict(pmid=p, year=pm2y[p], journal=pm2j[p], title=c.loc[c.pmid == p, "title"].iloc[0][:120], within_corpus_citations=in_cites[p], distinct_citing_topics=len(ts), rcr=float(c.loc[c.pmid == p, "rcr"].iloc[0])))
bp = pd.DataFrame(rows).sort_values(["distinct_citing_topics", "within_corpus_citations"], ascending=False)
bp.head(50).to_csv(OUT / "bridging_papers.csv", index=False)
summary["bridging_papers_top"] = [(r.pmid, int(r.year), r.journal[:30], r.title[:80], int(r.distinct_citing_topics), int(r.within_corpus_citations)) for _, r in bp.head(8).iterrows()]
summary["most_cited_within_corpus"] = [(r.pmid, int(r.year), r.journal[:30], r.title[:80], int(r.within_corpus_citations)) for _, r in bp.sort_values("within_corpus_citations", ascending=False).head(8).iterrows()]
json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=str)
print(json.dumps(summary, indent=1, default=str)[:6000])

# ---- figure: (a) attention vs impact; (b) pioneer rankings ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(ROOT / "figures")); from _mirror import mirror_to_main
fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]; texts = []
for dom, col, mk in [("methodology", "#0072B2", "o"), ("health", "#D55E00", "s")]:
    d = ti[(ti.domain == dom) & ti.share_trend_pct_per_yr.notna()]
    ax.scatter(d.share_trend_pct_per_yr, d.median_rcr, s=8 + 60 * np.log10(d.n_articles_2011_2025 / 100 + 1) ** 2, c=col, alpha=0.55, marker=mk, edgecolor="white", lw=0.5, label=f"{dom} topics")
    for _, r in d.iterrows():
        if r.median_rcr >= 1.9 or r.share_trend_pct_per_yr >= 30 or r.share_trend_pct_per_yr <= -12 or r.n_articles_2011_2025 >= 15000:
            texts.append(ax.text(r.share_trend_pct_per_yr, r.median_rcr, r.topic, fontsize=7))
try:
    from adjustText import adjust_text
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#888", lw=0.5), expand_points=(1.3, 1.5))
except Exception as e:
    print("adjustText unavailable:", e)
ax.axhline(1.0, color="#999", lw=0.8, ls="--"); ax.axvline(0, color="#999", lw=0.8, ls="--")
ax.set_xlabel("Share trend (% change in share of corpus output per year)"); ax.set_ylabel("Median relative citation ratio (RCR), articles 2011–2023")
ax.set_title("(a) Research attention versus citation impact by topic", loc="left", fontweight="bold", fontsize=11)
ax.legend(frameon=False, fontsize=9)
ax = axes[1]
top_j = list(dict.fromkeys(list(ft_rank.head(8).index) + list(cw_rank.head(8).index)))[:10]
ft = [int(ft_rank.get(j, 0)) for j in top_j]; cw = [int(cw_rank.get(j, 0)) for j in top_j]
ab = {"Journal of Biomedical Informatics (JBI)": "JBI", "Journal of the American Medical Informatics Association (JAMIA)": "JAMIA", "Journal of Medical Internet Research (JMIR)": "JMIR", "Computers in Biology and Medicine": "CBM", "Bioinformatics": "Bioinformatics", "IEEE Journal of Biomedical and Health Informatics (J-BHI)": "IEEE J-BHI", "BMC Medical Informatics and Decision Making": "BMC MIDM", "Briefings in Bioinformatics": "Brief Bioinform", "International Journal of Medical Informatics (IJMI)": "IJMI", "Journal of Medical Systems": "J Med Syst", "Nature Methods": "Nat Methods", "Nature Medicine": "Nat Med", "Artificial Intelligence in Medicine": "Artif Intell Med", "Database: The Journal of Biological Databases and Curation": "Database", "Methods of Information in Medicine": "Methods Inf Med", "Applied Clinical Informatics": "Appl Clin Inform"}
yp = np.arange(len(top_j)); ax.barh(yp - 0.2, ft, 0.4, color="#0072B2", label="first to publish (first article in the corpus)"); ax.barh(yp + 0.2, cw, 0.4, color="#009E73", label="citation-weighted (most early citations)")
ax.set_yticks(yp); ax.set_yticklabels([ab.get(j, j[:22]) for j in top_j], fontsize=9); ax.invert_yaxis(); ax.set_xlabel("Methodology topics (of 70) for which the journal leads")
ax.set_title("(b) Pioneer journals: first-to-publish vs citation-weighted", loc="left", fontweight="bold", fontsize=11); ax.legend(frameon=False, fontsize=9)
for a in axes:
    for s in ("top", "right"): a.spines[s].set_visible(False)
fig.tight_layout(); out = ROOT / "figures" / "supp_figure10_citation_influence.png"; fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white"); mirror_to_main(out); print("wrote", out)
