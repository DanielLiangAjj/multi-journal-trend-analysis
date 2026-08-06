#!/usr/bin/env python3
"""Recompute every statistic quoted in the manuscript and check it against the
claimed value. Run from the project root:  python3 verify_manuscript_numbers.py

Covers the 2011-2025 window only; the 352 records indexed for 2026 are used
solely in the MeSH evaluation, matching the manuscript's scope.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

YEARS = list(range(2011, 2026))
YC = [str(y) for y in YEARS]
CORPUS_CSV = "data/pubmed_all_journals_2011_2025_w_keywords_patched.csv"
VIS = "data/visualizations_k100"

PASS, FAIL = [], []


def chk(label, actual, claimed, tol=0.0):
    good = (abs(actual - claimed) <= tol) if isinstance(claimed, (int, float)) \
        and not isinstance(claimed, bool) else (actual == claimed)
    (PASS if good else FAIL).append(label)
    mark = "ok  " if good else "FAIL"
    print(f"  {mark} {label:58s} manuscript={claimed!s:>18s}  computed={actual!s:>18s}")
    return good


def section(t):
    print(f"\n{'=' * 96}\n{t}\n{'=' * 96}")


# ---------------------------------------------------------------- A. corpus
section("A. CORPUS")
recs = pd.read_csv(CORPUS_CSV, usecols=["journal", "year", "keywords", "keyword_source"],
                   low_memory=False)
chk("total records collected", len(recs), 78425)
w = recs[recs.year.between(2011, 2025)]
chk("records published 2011-2025", len(w), 78073)
chk("records indexed for 2026", int((recs.year == 2026).sum()), 352)
chk("journals", w.journal.nunique(), 29)
# 95,872 is the number of distinct keywords entering categorisation, after the
# pipeline's own normalisation - not a naive split of the raw `keywords` column.
kwc = pd.read_csv("data/keywords_categorized.csv", low_memory=False)
chk("distinct author keywords (categorisation input)", len(kwc), 95872)
raw = set()
for _s in recs.keywords.dropna().astype(str):
    raw.update(k.strip().lower() for k in _s.split(";") if k.strip())
print(f"       (naive lowercased split of the raw column gives {len(raw):,} - "
      f"normalisation accounts for the difference)")

vol = w.year.value_counts().sort_index()
chk("2011 volume", int(vol[2011]), 1494)
chk("2025 volume", int(vol[2025]), 11384)
chk("fold increase 2011->2025", round(vol[2025] / vol[2011], 1), 7.6, 0.05)
chk("wave 1 growth 2011->2015 (%)", round((vol[2015] / vol[2011] - 1) * 100), 121, 1)
chk("wave 2 growth 2016->2020 (%)", round((vol[2020] / vol[2016] - 1) * 100, 1), 88.6, 0.1)
chk("2016 volume", int(vol[2016]), 3525)
chk("2020 volume", int(vol[2020]), 6647)
chk("2021 volume", int(vol[2021]), 7438)

section("B. KEYWORD ACQUISITION PASSES (Methods 3.1)")
ks = recs.keyword_source.value_counts()
chk("Pass 3 GPT abstract extraction (n)", int(ks.get("abstract_gpt", 0)), 857)
chk("Pass 3 share (%)", round(100 * ks.get("abstract_gpt", 0) / len(recs), 1), 1.1, 0.05)
chk("Pass 4 MeSH substitution (n)", int(ks.get("mesh", 0)), 18409)
chk("Pass 4 share (%)", round(100 * ks.get("mesh", 0) / len(recs), 1), 23.5, 0.05)

# ---------------------------------------------------------------- C. topics & trends
section("C. TOPIC COUNTS AND LINEAR TRENDS (4.1.2)")
tyc, trend = {}, {}
for dom in ("methodology", "health"):
    tyc[dom] = pd.read_csv(f"{VIS}/{dom}_topic_year_counts.csv")
    tyc[dom]["n25"] = tyc[dom][YC].sum(axis=1).astype(int)
    trend[dom] = pd.read_csv(f"{VIS}/{dom}_trend_analysis.csv")
chk("methodology topics", len(tyc["methodology"]), 70)
chk("health topics", len(tyc["health"]), 86)
chk("methodology Holm-sig linear trends", int(trend["methodology"].significant_holm.sum()), 56)
chk("health Holm-sig linear trends", int(trend["health"].significant_holm.sum()), 72)
chk("methodology Holm-sig (%)", round(100 * trend["methodology"].significant_holm.mean(), 1), 80.0, 0.05)
chk("health Holm-sig (%)", round(100 * trend["health"].significant_holm.mean(), 1), 83.7, 0.05)
neg = sum(int(((trend[d].slope < 0) & trend[d].significant_holm).sum()) for d in trend)
chk("Holm-sig NEGATIVE absolute slopes", neg, 0)
for dom, name, claim in [("methodology", "Knowledge distillation", 98),
                         ("methodology", "Federated learning", 154),
                         ("methodology", "Transfer learning variants", 41),
                         ("methodology", "Contrastive learning", 1579),
                         ("methodology", "Data augmentation", 383)]:
    n = int(tyc[dom].loc[tyc[dom].topic == name, "n25"].iloc[0])
    chk(f"n (2011-25) {name}", n, claim)

# ---------------------------------------------------------------- D. share model
section("D. SHARE MODEL — quasi-Poisson with log-corpus offset (4.2.3)")
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
corpus = np.array([vol[y] for y in YEARS], float)
chk("corpus offset sums to", int(corpus.sum()), 78073)
share = {}
for dom in ("methodology", "health"):
    rows = []
    X = sm.add_constant(np.array(YEARS) - 2011)
    for _, r in tyc[dom].iterrows():
        y = np.array([float(r[c]) for c in YC])
        if y.sum() < 10:
            continue
        m = sm.GLM(y, X, family=sm.families.Poisson(), offset=np.log(corpus)).fit(scale="X2")
        rows.append((r.topic, m.params[1], m.pvalues[1], y.sum()))
    R = pd.DataFrame(rows, columns=["topic", "slope", "p", "n"])
    R["sig"], R["p_holm"] = multipletests(R.p, alpha=0.05, method="holm")[:2]
    R["pct"] = (np.exp(R.slope) - 1) * 100
    share[dom] = R
chk("methodology topics tested", len(share["methodology"]), 70)
chk("methodology Holm-sig share trends", int(share["methodology"].sig.sum()), 34)
chk("  ... gaining share", int((share["methodology"].sig & (share["methodology"].pct > 0)).sum()), 23)
chk("  ... losing share", int((share["methodology"].sig & (share["methodology"].pct < 0)).sum()), 11)
chk("health topics tested", len(share["health"]), 86)
chk("health Holm-sig share trends", int(share["health"].sig.sum()), 21)
chk("  ... gaining share", int((share["health"].sig & (share["health"].pct > 0)).sum()), 9)
chk("  ... losing share", int((share["health"].sig & (share["health"].pct < 0)).sum()), 12)
for dom, name, claim in [("methodology", "Large language models", 65.4),
                         ("methodology", "Knowledge distillation", 51.3),
                         ("methodology", "Federated learning", 46.5),
                         ("methodology", "Transfer learning variants", 30.8),
                         ("methodology", "Contrastive learning", 25.9),
                         ("methodology", "Pilot studies and synthetic data", -22.2),
                         ("methodology", "Microarray analysis", -19.3),
                         ("methodology", "Computer security", -15.5),
                         ("methodology", "Clinical study designs", -10.9),
                         ("health", "Health semantic similarity", 12.3),
                         ("health", "Mixed medical imaging", -13.0)]:
    v = share[dom].loc[share[dom].topic == name, "pct"].iloc[0]
    chk(f"share %/yr {name}", round(v, 1), claim, 0.05)
tot_cs = flat = losing_h = 0
for dom in ("methodology", "health"):
    m = share[dom].merge(trend[dom][["topic", "significant_holm"]], on="topic")
    cs = m[m.significant_holm]
    tot_cs += len(cs); flat += int((~cs.sig).sum())
    if dom == "health":
        losing_h = int((cs.sig & (cs.pct < 0)).sum())
chk("count-significant topics (both domains)", tot_cs, 128)
chk("  ... of those, share-flat", flat, 87)
chk("health count-risers that LOSE share", losing_h, 7)
dht = tyc["health"][tyc["health"].topic == "Digital health technologies"].iloc[0]
dy = np.array([float(dht[c]) for c in YC]); dsh = dy / corpus * 100
chk("Digital health technologies n (2011-25)", int(dy.sum()), 32491)
chk("  ... share in 2011 (%)", round(dsh[0], 1), 45.3, 0.05)
chk("  ... share in 2025 (%)", round(dsh[-1], 1), 40.6, 0.05)

# ---------------------------------------------------------------- E. breakthrough years
section("E. BREAKTHROUGH YEARS (4.2.3)")
bt = {}
for dom in ("methodology", "health"):
    rows = []
    for _, r in tyc[dom].iterrows():
        y = np.array([float(r[c]) for c in YC])
        if y.sum() < 100:
            continue
        rows.append((r.topic, YEARS[int(np.argmax(y >= 0.25 * y.max()))], y.sum()))
    bt[dom] = pd.DataFrame(rows, columns=["topic", "byear", "n"])
m, h = bt["methodology"], bt["health"]
chk("methodology topics with >=100 articles", len(m), 58)
chk("  at/above threshold at 2011 corpus start", int((m.byear == 2011).sum()), 13)
for yr, c in ((2014, 8), (2016, 5), (2018, 5), (2019, 6)):
    chk(f"  methodology breakthroughs in {yr}", int((m.byear == yr).sum()), c)
chk("methodology median breakthrough year", int(m.byear.median()), 2015)
chk("health topics with >=100 articles", len(h), 76)
chk("  at/above threshold at 2011 corpus start", int((h.byear == 2011).sum()), 12)
for yr, c in ((2012, 14), (2014, 9), (2015, 9), (2018, 8)):
    chk(f"  health breakthroughs in {yr}", int((h.byear == yr).sum()), c)
chk("health median breakthrough year", int(h.byear.median()), 2015)
chk("methodology LLM-era (2021-25) breakthroughs", int((m.byear >= 2021).sum()), 7)
chk("health LLM-era (2021-25) breakthroughs", int((h.byear >= 2021).sum()), 3)
late_h = sorted(h[h.byear >= 2021].topic.tolist())
chk("  health LLM-era topic set", late_h,
    sorted(["Sequence and structure prediction", "Eye, ENT, and oral diseases",
            "Health semantic similarity"]))

# ---------------------------------------------------------------- F. entropy
section("F. CROSS-JOURNAL ENTROPY (4.2.2)")
for dom, n_uni, n_niche in (("methodology", 5, 7), ("health", 3, 10)):
    sp = pd.read_csv(f"{VIS}/pi_q5_{dom}_specialization.csv")
    chk(f"{dom} universal topics (entropy >= 0.85)", int((sp.norm_entropy >= 0.85).sum()), n_uni)
    chk(f"{dom} niche topics (<=0.55, >=80 articles)",
        int(((sp.norm_entropy <= 0.55) & (sp.total >= 80)).sum()), n_niche)
sp = pd.read_csv(f"{VIS}/pi_q5_health_specialization.csv")
chk("  drug safety surveillance entropy", round(float(sp.loc[sp.topic == "Drug safety surveillance", "norm_entropy"].iloc[0]), 3), 0.866, 0.0006)
niche = sp[(sp.norm_entropy <= 0.55) & (sp.total >= 80)]
chk("  niche health topics whose top venue is Bioinformatics",
    int((niche.max_journal == "Bioinformatics").sum()), 7)

# ---------------------------------------------------------------- G. COVID
section("G. COVID-19 CASE STUDY (4.3.3)")
cov = pd.read_csv(f"{VIS}/pi_q7_covid_impact.csv").dropna(subset=["pct_change_during", "p_value_during"])
cov["p_holm"] = multipletests(cov.p_value_during.clip(lower=1e-300), method="holm")[1]
sig = cov[cov.p_holm < 0.05]
chk("topics tested", len(cov), 133)
chk("Holm-significant", len(sig), 90)
chk("  increasing", int((sig.pct_change_during > 0).sum()), 88)
chk("  decreasing", int((sig.pct_change_during < 0).sum()), 2)
chk("  share significant (%)", round(100 * len(sig) / len(cov)), 67, 1)
for name, claim in [("Sequence and structure prediction", 244), ("Data augmentation", 229),
                    ("Population health surveillance", 158), ("Biomedical databases", -36.2)]:
    v = float(cov.loc[cov.topic == name, "pct_change_during"].iloc[0])
    chk(f"  pct change {name}", round(v, 1) if claim % 1 else round(v), claim, 0.6)

# ---------------------------------------------------------------- H. authorship
section("H. AUTHORSHIP AND TEAM SCIENCE (4.3.1)")
a = pd.read_csv(f"{VIS}/pi_q2_authorship_stats.csv").set_index("year")
chk("mean authors 2011", round(float(a.loc[2011, "mean_authors"]), 2), 5.98, 0.005)
chk("mean authors 2025", round(float(a.loc[2025, "mean_authors"]), 2), 7.78, 0.005)
chk("max team size 2025", int(a.loc[2025, "max_authors"]), 634)
chk("mean affiliations 2014", round(float(a.loc[2014, "mean_affs"]), 2), 2.0, 0.02)
chk("mean affiliations 2025", round(float(a.loc[2025, "mean_affs"]), 2), 4.85, 0.005)
chk("interdisciplinary 2014 (%)", round(float(a.loc[2014, "pct_interdisciplinary"]), 1), 16.4, 0.05)
chk("interdisciplinary 2025 (%)", round(float(a.loc[2025, "pct_interdisciplinary"]), 1), 28.8, 0.05)
chk("interdisciplinary 2011 (%)", round(float(a.loc[2011, "pct_interdisciplinary"]), 1), 5.4, 0.05)
ts = pd.read_csv(f"{VIS}/pi_q3_team_science.csv").nlargest(5, "pct_inter")
for j, claim in (("JAMIA Open", 51.5), ("The Lancet Digital Health", 51.1)):
    chk(f"  interdisciplinary share {j} (%)",
        round(float(ts.loc[ts.journal == j, "pct_inter"].iloc[0]), 1), claim, 0.05)
pio = pd.read_csv(f"{VIS}/pi_q1_methodology_first_emergence.csv")
n_jbi = int((pio.first_journal == "Journal of Biomedical Informatics (JBI)").sum())
chk("JBI pioneer topics", n_jbi, 18)
chk("JBI pioneer share (%)", round(100 * n_jbi / len(pio), 1), 25.7, 0.05)

# ---------------------------------------------------------------- I. granularity (Supp Table 3)
section("I. GRANULARITY SENSITIVITY (Supplementary Note S1.1 / Supplementary Table 3)")
gran = [("k100", 100, 70, 86, 645, 1006, 56, 70, 72, 86),
        ("k750", 750, 335, 413, 95, 76, 149, 280, 153, 326),
        ("k1050_1100", 1050, 1050, 1100, 93, 133, 375, 1045, 399, 1098)]
for suff, k, nm, nh, mm, mh, sm_, tm_, sh_, th_ in gran:
    for dom, ntop, med, nsig, ntest in (("methodology", nm, mm, sm_, tm_), ("health", nh, med if False else mh, sh_, th_)):
        f = f"data/visualizations_{suff}/{dom}_topic_year_counts.csv"
        t = f"data/visualizations_{suff}/{dom}_trend_analysis.csv"
        if not (os.path.exists(f) and os.path.exists(t)):
            print(f"  skip {suff}/{dom} (missing)"); continue
        d = pd.read_csv(f); arts = d[YC].sum(axis=1)
        tr = pd.read_csv(t)
        chk(f"K={k} {dom} median articles/topic", int(round(arts.median())), med)
        chk(f"K={k} {dom} testable topics (>=10 articles)", len(tr), ntest)
        chk(f"K={k} {dom} Holm-significant trends", int(tr.significant_holm.sum()), nsig)
for dom, n in (("methodology", 335), ("health", 413)):
    j = f"data/hierarchy_k750/{dom}_final_topics.json"
    if os.path.exists(j):
        chk(f"K=750 {dom} named topics", len(json.load(open(j))), n)
sub = pd.read_csv("data/k_comparison/k_scan_sub100.csv")
chk("sub-100 silhouette scan grid", sorted(set(sub.k.tolist())), [2, 3, 5, 10, 15, 20, 30, 50, 75])
for dom, k2, k100 in (("methodology", 0.0865, 0.0243), ("health", 0.0812, 0.0257)):
    chk(f"silhouette K=2 {dom}",
        round(float(sub[(sub.domain == dom) & (sub.k == 2)].silhouette.iloc[0]), 4), k2, 0.0001)
    hi = pd.read_csv(f"data/clusters/{dom}_k_scan.csv")
    chk(f"silhouette K=100 {dom}",
        round(float(hi[hi.k == 100].silhouette_score.iloc[0]), 4), k100, 0.0001)
    chk(f"silhouette first negative K {dom}",
        int(hi[hi.silhouette_score < 0].k.min()), 600)

# ---------------------------------------------------------------- J. decoupling
section("J. DECOUPLING CASE STUDIES (Supplementary Note S1.2)")
from scipy import stats as sps
def series(dom, suff, col, val):
    d = pd.read_csv(f"data/visualizations_{suff}/{dom}_topic_year_counts.csv")
    r = d[d[col] == val].iloc[0]
    return np.array([float(r[c]) for c in YC])
for label, dom, suff, col, val, claims in [
        ("HCI (K=750)", "methodology", "k750", "topic", "Human-computer interaction",
         dict(n=3861, early=245, late=259)),
        ("SVM (K=1050 cl.45)", "methodology", "k1050_1100", "cluster_id", 45,
         dict(n=544, peak=58, last=27)),
        ("Hospital IS (K=1050 cl.22)", "methodology", "k1050_1100", "cluster_id", 22,
         dict(n=631, peak=104, last=24, drop=77, slope=-4.45, p=0.0009)),
        ("Interop (K=1100 cl.477)", "health", "k1050_1100", "cluster_id", 477,
         dict(n=1021, peak=128, trough=45, last=76))]:
    c = series(dom, suff, col, val)
    chk(f"{label} n (2011-25)", int(c.sum()), claims["n"])
    if "peak" in claims: chk(f"{label} peak/yr", int(c.max()), claims["peak"])
    if "last" in claims: chk(f"{label} 2025/yr", int(c[-1]), claims["last"])
    if "drop" in claims: chk(f"{label} decline from peak (%)", int(round((1 - c[-1] / c.max()) * 100)), claims["drop"])
    if "early" in claims: chk(f"{label} 2011-13 mean", int(round(c[:3].mean())), claims["early"])
    if "late" in claims: chk(f"{label} 2023-25 mean", int(round(c[-3:].mean())), claims["late"])
    if "trough" in claims: chk(f"{label} trough", int(c.min()), claims["trough"])
    if "slope" in claims:
        lr = sps.linregress(YEARS, c)
        chk(f"{label} slope/yr", round(lr.slope, 2), claims["slope"], 0.005)
        chk(f"{label} raw p", round(lr.pvalue, 4), claims["p"], 0.0002)
for name, a, b, ratio in [("Software and usability research", 227, 1329, 5.9),
                          ("Machine learning methods", 440, 5248, 11.9),
                          ("Clinical decision support", 507, 2251, 4.4),
                          ("Patient-provider communication", 229, 802, 3.5)]:
    c = series("methodology" if name != "Patient-provider communication" else "health",
               "k100", "topic", name)
    chk(f"parent {name} 2011", int(c[0]), a)
    chk(f"parent {name} 2025", int(c[-1]), b)
    chk(f"parent {name} fold growth", round(c[-1] / c[0], 1), ratio, 0.05)

# ---------------------------------------------------------------- K. summary
section("SUMMARY")
print(f"  {len(PASS)} checks passed, {len(FAIL)} failed")
if FAIL:
    print("\n  FAILURES:")
    for f in FAIL: print("   -", f)
sys.exit(1 if FAIL else 0)
