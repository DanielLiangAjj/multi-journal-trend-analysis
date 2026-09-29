"""Supplementary Figure 2 rebuild — within-topic keyword composition.

Reviewer fixes: larger text, per-panel legend arranged in rows beneath each
panel (not a microscopic side box), axes drawn only on outer subplots, no
in-image figure title, and a corrected panel set identical to the pipeline's
(top 6 topics per domain, each topic's top 5 keywords + 'others').
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
YEARS = list(range(2011, 2026))
COLORS = ["#4c72b0", "#55a868", "#c44e52", "#8172b3", "#ccb974", "#b8b8b8"]


def kw_map(domain):
    topics = json.load(open(ROOT / f"data/hierarchy_k100/{domain}_final_topics.json"))
    m = defaultdict(set)
    for d in topics.values():
        for k in d.get("keywords", []):
            m[k.strip().lower()].add(d["name"])
    return m


def build(domain, letter):
    m = kw_map(domain)
    topic_kw_year = defaultdict(lambda: defaultdict(Counter))
    for chunk in pd.read_csv(
            ROOT / "data/pubmed_all_journals_2011_2025_w_keywords_patched.csv",
            usecols=["year", "keywords"], chunksize=50_000):
        chunk = chunk.dropna(subset=["keywords", "year"])
        chunk = chunk[chunk["year"].between(2011, 2025)]
        for yr, kws in zip(chunk["year"], chunk["keywords"].astype(str)):
            yr = int(yr)
            for kw in kws.split(";"):
                k = kw.strip().lower()
                for t in m.get(k, ()):
                    topic_kw_year[t][yr][k] += 1

    totals = Counter({t: sum(sum(c.values()) for c in yd.values())
                      for t, yd in topic_kw_year.items()})
    top_topics = [t for t, _ in totals.most_common(6)]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    for ax, topic in zip(axes.flatten(), top_topics):
        yd = topic_kw_year[topic]
        kw_tot = Counter()
        for c in yd.values():
            kw_tot.update(c)
        top5 = [k for k, _ in kw_tot.most_common(5)]
        series = []
        for k in top5:
            series.append([yd[y].get(k, 0) for y in YEARS])
        others = [sum(yd[y].values()) - sum(yd[y].get(k, 0) for k in top5)
                  for y in YEARS]
        series.append(others)
        arr = np.array(series, dtype=float)
        denom = arr.sum(axis=0)
        denom[denom == 0] = 1.0
        arr = arr / denom
        ax.stackplot(YEARS, arr, labels=[*top5, "others"], colors=COLORS,
                     alpha=0.92, edgecolor="white", linewidth=0.3)
        ax.set_title(topic, fontsize=12.5, fontweight="bold", pad=6)
        ax.set_ylim(0, 1)
        ax.set_xlim(2011, 2025)
        ax.set_xticks([2012, 2016, 2020, 2024])
        ax.tick_params(labelsize=10)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
                  fontsize=8.2, ncol=2, frameon=False,
                  handlelength=1.2, columnspacing=0.8, labelspacing=0.3)
        ax.label_outer()
    for i, ax in enumerate(axes.flatten()):
        if i % 3 == 0:
            ax.set_ylabel("Share of topic's keyword occurrences", fontsize=10.5)
    fig.text(0.005, 0.995, f"({letter}) {domain.capitalize()}",
             fontsize=14, fontweight="bold", ha="left", va="top")
    fig.subplots_adjust(hspace=0.52, wspace=0.22, bottom=0.16, top=0.90)
    out = ROOT / "figures" / f"supp_figure2_keywords_{domain}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")
    for t in top_topics:
        yd = topic_kw_year[t]
        kw_tot = Counter()
        for c in yd.values():
            kw_tot.update(c)
        print(f"   {t}: top5 = {[k for k, _ in kw_tot.most_common(5)]}")
    mirror_to_main(str(out))


if __name__ == "__main__":
    build("methodology", "a")
    build("health", "b")
