"""Does the journal-network finding (two Louvain communities, JBI bridge)
survive finer clustering granularity? Supp Note S1.1 support.

Rebuilds the journal-similarity network with the exact recipe of
figures/build_figure_betweenness.py::build_network() — journal x topic
proportion vectors, pairwise cosine similarity, edges at or above the median,
betweenness on 1-cosine distances, Louvain (seed=42) on similarity weights —
at three granularities:

  K=100          hierarchy_k100 final topics   (validates the method: must
                 reproduce data/journal_similarity_graph.pkl edge-for-edge)
  K=750          hierarchy_k750 final topics
  K=1,050/1,100  clusters_k1050_1100 raw clusters (no naming at this K)

Result (2026-08-27 run):
  K=100:        2 communities [12, 17], modularity 0.123, JBI top bridge (0.362)
  K=750:        3 communities,          modularity 0.134, JBI 4th   (0.169)
  K=1050/1100:  4 communities,          modularity 0.182, JBI 13th  (0.024)

So the two-community + JBI-bridge structure is a K=100-resolution result and
is described as such in the manuscript. Data-only: no pipeline re-run.
"""
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import networkx.algorithms.community as nx_com
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "data/pubmed_all_journals_2011_2025_w_keywords_patched.csv"
JBI = "Journal of Biomedical Informatics (JBI)"


def kwmap_from_topics(path, prefix):
    topics = json.load(open(path))
    m = defaultdict(set)
    for d in topics.values():
        for k in d.get("keywords", []):
            m[k.strip().lower()].add(prefix + d["name"])
    return m


def kwmap_from_clusters(path, prefix):
    df = pd.read_csv(path)
    m = defaultdict(set)
    for _, r in df.iterrows():
        m[str(r.keyword).strip().lower()].add(f"{prefix}{int(r.cluster_id)}")
    return m


def build_network(topic_maps):
    """Mirror of figures/build_figure_betweenness.py::build_network()."""
    jt = defaultdict(Counter)
    jn = Counter()
    for chunk in pd.read_csv(CORPUS, usecols=["journal", "year", "keywords"],
                             chunksize=50_000):
        chunk = chunk.dropna(subset=["keywords", "year", "journal"])
        chunk = chunk[chunk["year"].between(2011, 2025)]
        for j, kws in zip(chunk["journal"], chunk["keywords"].astype(str)):
            jn[j] += 1
            tset = set()
            for kw in kws.split(";"):
                k = kw.strip().lower()
                for m in topic_maps:
                    tset |= m.get(k, set())
            for t in tset:
                jt[j][t] += 1
    journals = sorted(jn)
    all_topics = sorted({t for c in jt.values() for t in c})
    ti = {t: i for i, t in enumerate(all_topics)}
    M = np.zeros((len(journals), len(all_topics)))
    for i, j in enumerate(journals):
        for t, n in jt[j].items():
            M[i, ti[t]] = n / jn[j]
    Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
    S = Mn @ Mn.T
    thresh = np.percentile(S[np.triu_indices(len(journals), 1)], 50)
    G = nx.Graph()
    for j in journals:
        G.add_node(j, size=jn[j])
    for a in range(len(journals)):
        for b in range(a + 1, len(journals)):
            if S[a, b] >= thresh:
                G.add_edge(journals[a], journals[b], weight=float(S[a, b]))
    return G


def report(label, G):
    bc = nx.betweenness_centrality(G, weight=lambda u, v, d: 1 - d["weight"])
    top = sorted(bc.items(), key=lambda kv: -kv[1])
    comms = nx_com.louvain_communities(G, weight="weight", seed=42)
    mod = nx_com.modularity(G, comms, weight="weight")
    jbi_rank = [j for j, _ in top].index(JBI) + 1
    print(f"\n== {label}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"   Louvain (seed 42): {len(comms)} communities, sizes "
          f"{sorted(len(c) for c in comms)}, modularity {mod:.4f}")
    print(f"   JBI betweenness {bc[JBI]:.4f}, rank {jbi_rank}")
    print(f"   top bridges: {[(j[:40], round(b, 3)) for j, b in top[:4]]}")
    return bc, comms


if __name__ == "__main__":
    # --- K=100: validate the reconstruction against the pipeline's graph
    G100 = build_network([
        kwmap_from_topics(ROOT / "data/hierarchy_k100/methodology_final_topics.json", "M:"),
        kwmap_from_topics(ROOT / "data/hierarchy_k100/health_final_topics.json", "H:"),
    ])
    pkl = ROOT / "data/journal_similarity_graph.pkl"
    if pkl.exists():
        Gref = pickle.load(open(pkl, "rb"))
        same = set(map(frozenset, Gref.edges())) == set(map(frozenset, G100.edges()))
        wmax = max(abs(Gref[u][v]["weight"] - G100[u][v]["weight"])
                   for u, v in Gref.edges()) if same else float("nan")
        print(f"K=100 reconstruction matches journal_similarity_graph.pkl: "
              f"edges {same}, max weight diff {wmax:.2e}")
        assert same, "reconstruction does not match the published graph"
    report("K=100 (named final topics)", G100)

    G750 = build_network([
        kwmap_from_topics(ROOT / "data/hierarchy_k750/methodology_final_topics.json", "M:"),
        kwmap_from_topics(ROOT / "data/hierarchy_k750/health_final_topics.json", "H:"),
    ])
    report("K=750 (named final topics)", G750)

    Gfine = build_network([
        kwmap_from_clusters(ROOT / "data/clusters_k1050_1100/methodology_keyword_clusters.csv", "M:"),
        kwmap_from_clusters(ROOT / "data/clusters_k1050_1100/health_keyword_clusters.csv", "H:"),
    ])
    report("K=1,050/1,100 (raw clusters)", Gfine)
