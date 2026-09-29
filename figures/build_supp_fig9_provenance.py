"""Supplementary Figure 9 (round 23) — keyword-provenance decomposition of the four
fine-grained 'decline' candidates that Supplementary Note S1.2 previously showcased.

Each panel stacks the yearly article count of the fine-K cluster/topic into records
whose keywords come from authors/publishers/GPT (Passes 1, 2, 4) versus records whose
keywords are MeSH descriptors substituted in Pass 3. The apparent early peaks and
subsequent 'declines' sit entirely in the MeSH-substituted stack, which shrinks as
author keywords became available from 2013 onward; the author-keyword stack is flat
or rising in every case. Replaces figure12_decoupling_case_studies.png.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _mirror import mirror_to_main

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "supp_figure9_provenance.png"
YEARS = list(range(2011, 2026))
AUTH_C, MESH_C, INK, MUTED, GRID = "#1f4ea0", "#c9a227", "#1a1a1a", "#5c5c5c", "#dcdcdc"

df = pd.read_csv(ROOT / "data/pubmed_all_journals_2011_2025_w_keywords_patched.csv",
                 usecols=["year", "keywords", "keyword_source"], low_memory=False)
df = df[df.year.between(2011, 2025)].copy()
df["kw"] = df.keywords.map(lambda s: [k.strip().lower() for k in str(s).split(";") if k.strip()]
                           if isinstance(s, str) else [])
df["mesh"] = df.keyword_source == "mesh"


def cluster_keywords(domain, cid):
    t = pd.read_csv(ROOT / f"data/clusters_k1050_1100/{domain}_keyword_clusters.csv")
    return set(t[t.cluster_id == cid].keyword.str.lower())


def k750_topic_keywords(domain, name_pattern):
    t = json.load(open(ROOT / f"data/topics_k750/{domain}_topics.json"))
    kws = set()
    for node, v in t.items():
        if re.search(name_pattern, v["name"], re.I):
            kws |= {k.lower() for k in v["keywords"]}
    return kws


CASES = [
    ("(a) Human–computer interaction (K = 750 topic)", k750_topic_keywords("methodology", r"human.computer interaction")),
    ("(b) Support vector machines (K = 1,050 cluster)", cluster_keywords("methodology", 45)),
    ("(c) Hospital information systems (K = 1,050 cluster)", cluster_keywords("methodology", 22)),
    ("(d) Interoperability (K = 1,100 cluster)", cluster_keywords("health", 477)),
]

fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
for ax, (title, kws) in zip(axes.ravel(), CASES):
    hit = df[df.kw.map(lambda l: any(k in kws for k in l))]
    a = hit[~hit.mesh].groupby("year").size().reindex(YEARS, fill_value=0).to_numpy()
    m = hit[hit.mesh].groupby("year").size().reindex(YEARS, fill_value=0).to_numpy()
    ax.bar(YEARS, a, color=AUTH_C, width=0.75, label="Author / publisher / GPT keywords (Passes 1, 2, 4)", zorder=3)
    ax.bar(YEARS, m, bottom=a, color=MESH_C, width=0.75, label="MeSH descriptors substituted (Pass 3)", zorder=3)
    ax.plot(YEARS, a + m, color=INK, lw=1.4, marker="o", ms=3, zorder=4, label="All records")
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", color=INK)
    ax.set_xticks(YEARS[::2])
    ax.tick_params(labelsize=8.5, colors=MUTED)
    ax.set_ylabel("Articles per year", fontsize=9.5, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    tot = a + m
    print(f"{title}: total {int(tot.sum())}; author {a.tolist()}; mesh {m.tolist()}")
axes[1, 0].legend(fontsize=8.5, frameon=False, loc="upper right")
for ax in axes[1]:
    ax.set_xlabel("Year", fontsize=9.5, color=INK)
fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"wrote {OUT}")
mirror_to_main(OUT)
