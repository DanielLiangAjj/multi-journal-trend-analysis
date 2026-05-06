#!/usr/bin/env python3
"""
Replicate the neural-symbolic MeSH-based taxonomy evaluation from
Fang et al. (J. Biomed. Inform. 178, 2026, 105013) §2.3 / §4.1
on the K=100 multi-journal corpus.

Run from the project root:
    OPENAI_API_KEY=... python3 mesh_eval.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(
    "/Users/danielliang/Library/CloudStorage/Dropbox/multi_journal_trend_analysis"
)
CORPUS_CSV = PROJECT_ROOT / "data" / "pubmed_all_journals_2011_2025_w_keywords_patched.csv"
HIERARCHY_DIR = PROJECT_ROOT / "data" / "hierarchy_k100"
OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_JSON = OUTPUT_DIR / "candidate_mesh_pool.json"
TOPIC_TO_MESH_JSON = OUTPUT_DIR / "topic_to_mesh.json"
PER_ARTICLE_CSV = OUTPUT_DIR / "mesh_overlap_per_article.csv"
SUMMARY_MD = OUTPUT_DIR / "mesh_overlap_summary.md"

MODEL = "gpt-5-nano"
COVERAGE_TARGET = 0.85  # ≥85% of MeSH-tagged articles

# Heuristic substrings for "methodology-related" MeSH terms
METHOD_SUBSTRINGS = [
    "algorithm",
    "machine learning",
    "deep learning",
    "natural language processing",
    "data mining",
    "software",
    "computational biology",
    "neural network",
    "support vector",
    "bayes",
    "regression",
    "classification",
    "cluster analysis",
    "pattern recognition",
    "image processing",
    "decision support",
    "informatics",
    "text",
    "model",
    "database",
    "simulation",
    "analysis",
    "methodology",
    "electronic health records",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def split_semicolon(s) -> List[str]:
    if not isinstance(s, str):
        return []
    return [p.strip() for p in s.split(";") if p.strip()]


# ---------------------------------------------------------------------------
# Step 1 — Build the candidate MeSH pool
# ---------------------------------------------------------------------------

def build_candidate_pool(df_mesh: pd.DataFrame) -> Dict[str, dict]:
    """Returns dict with keys: pool (List[str]), frequencies, matches, coverage."""
    print("[Step 1] Counting MeSH-term frequencies across MeSH-tagged corpus")
    counter: Counter = Counter()
    for s in df_mesh["mesh_terms"]:
        for term in set(split_semicolon(s)):  # set: an article counts a term once
            counter[term] += 1

    print(f"  unique MeSH terms across corpus: {len(counter):,}")

    # Filter by methodology heuristic
    method_terms: List[tuple] = []  # (term, count, matched_substrings)
    for term, count in counter.items():
        term_lc = term.lower()
        matched = [sub for sub in METHOD_SUBSTRINGS if sub in term_lc]
        if matched:
            method_terms.append((term, count, matched))
    print(f"  methodology-heuristic-matching terms: {len(method_terms):,}")

    # Sort by frequency descending
    method_terms.sort(key=lambda t: t[1], reverse=True)

    # Cumulative coverage: an article is "covered" if it contains AT LEAST ONE term
    # in the running pool. Greedy add by frequency until coverage ≥ target.
    n_articles = len(df_mesh)
    target_n = int(np.ceil(COVERAGE_TARGET * n_articles))

    article_terms: List[set] = [
        set(split_semicolon(s)) for s in df_mesh["mesh_terms"]
    ]
    covered = np.zeros(n_articles, dtype=bool)

    pool: List[str] = []
    pool_meta: List[dict] = []
    for term, count, matched in method_terms:
        # Update coverage
        for i, terms in enumerate(article_terms):
            if not covered[i] and term in terms:
                covered[i] = True
        pool.append(term)
        pool_meta.append(
            {
                "mesh_term": term,
                "corpus_frequency": int(count),
                "matched_substrings": matched,
                "cumulative_articles_covered": int(covered.sum()),
                "cumulative_coverage_fraction": float(covered.sum() / n_articles),
            }
        )
        if covered.sum() >= target_n:
            break

    coverage = float(covered.sum() / n_articles)
    print(
        f"  pool size: {len(pool)}  coverage: {covered.sum():,}/{n_articles:,} "
        f"({coverage:.1%})"
    )
    if coverage < COVERAGE_TARGET:
        print(
            "  WARNING: coverage target not reached even after exhausting "
            "heuristic-matched terms"
        )

    out = {
        "heuristic_substrings": METHOD_SUBSTRINGS,
        "coverage_target": COVERAGE_TARGET,
        "n_mesh_tagged_articles": n_articles,
        "n_articles_covered": int(covered.sum()),
        "coverage_fraction": coverage,
        "pool_size": len(pool),
        "pool": pool_meta,
    }
    with open(CANDIDATE_JSON, "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=2, ensure_ascii=False)
    print(f"  wrote {CANDIDATE_JSON}")
    return out


# ---------------------------------------------------------------------------
# Step 2 — topic → MeSH mapping via GPT-5-nano
# ---------------------------------------------------------------------------

class MeshSubset(BaseModel):
    mesh_terms: List[str] = Field(
        default_factory=list,
        description="Subset of the candidate MeSH terms that conceptually describe the topic.",
    )


def query_topic(client, topic_name: str, top_keywords: List[str], pool: List[str]) -> List[str]:
    """Single batched GPT call for one topic."""
    prompt = (
        f"Below is a list of candidate MeSH terms. Return the subset of terms "
        f"that conceptually describe the topic '{topic_name}'. The topic's "
        f"representative keywords are: {', '.join(top_keywords)}. Reply as a "
        f"JSON list of strings, the exact MeSH term spellings as given. Do not "
        f"invent MeSH terms not in the list.\n\n"
        f"Candidate MeSH terms:\n"
        + "\n".join(f"- {t}" for t in pool)
    )
    last_err = None
    for attempt in range(4):
        try:
            response = client.responses.parse(
                model=MODEL,
                input=[{"role": "user", "content": prompt}],
                text_format=MeshSubset,
                max_output_tokens=2000,
                reasoning={"effort": "low"},
            )
            parsed = response.output_parsed
            if parsed is None:
                raise RuntimeError(
                    f"output_parsed is None (likely truncated or refused); "
                    f"status={getattr(response, 'status', '?')}"
                )
            pool_norm = {p.strip().lower(): p for p in pool}
            return [
                pool_norm[t.strip().lower()]
                for t in parsed.mesh_terms
                if t.strip().lower() in pool_norm
            ]
        except Exception as e:
            last_err = e
            wait = 2 * (attempt + 1)
            print(f"    GPT call failed (attempt {attempt + 1}): {e}; sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"GPT call failed for topic '{topic_name}' after retries: {last_err}")


def build_topic_to_mesh(
    pool: List[str], topics_by_domain: Dict[str, dict]
) -> Dict[str, Dict[str, List[str]]]:
    api_key = os.environ.get("OPENAI_API_KEY")
    cache: Dict[str, Dict[str, List[str]]] = {}
    if TOPIC_TO_MESH_JSON.exists():
        with open(TOPIC_TO_MESH_JSON, "r", encoding="utf-8") as fp:
            cache = json.load(fp)
        print(f"[Step 2] Loaded existing cache: {sum(len(v) for v in cache.values())} topics")
    for domain in topics_by_domain:
        cache.setdefault(domain, {})

    # Determine missing topics
    missing: List[tuple] = []
    for domain, topics in topics_by_domain.items():
        for topic_name in topics:
            if topic_name not in cache[domain]:
                missing.append((domain, topic_name))
    print(f"[Step 2] Topics needing GPT predictions: {len(missing)}")

    if missing:
        if not api_key:
            fail(
                "OPENAI_API_KEY not set. Cannot run GPT calls for the "
                f"{len(missing)} uncached topics. Re-run with the env var set."
            )
        from openai import OpenAI  # local import to avoid cost when fully cached
        client = OpenAI(api_key=api_key)
        for idx, (domain, topic_name) in enumerate(missing, 1):
            info = topics_by_domain[domain][topic_name]
            keywords = info.get("keywords", [])
            kw_counts = info.get("keyword_counts", {})
            # Top-10 by frequency where possible, else first 10
            if isinstance(kw_counts, dict) and kw_counts:
                ranked = sorted(kw_counts.items(), key=lambda x: x[1], reverse=True)
                top_kw = [k for k, _ in ranked[:10]]
            else:
                top_kw = list(keywords[:10])
            print(
                f"  [{idx}/{len(missing)}] {domain}::{topic_name} "
                f"(top kws: {top_kw[:3]}...)"
            )
            preds = query_topic(client, topic_name, top_kw, pool)
            cache[domain][topic_name] = preds
            # Persist after every topic for safe resume
            with open(TOPIC_TO_MESH_JSON, "w", encoding="utf-8") as fp:
                json.dump(cache, fp, indent=2, ensure_ascii=False)
            print(f"      -> {len(preds)} MeSH terms predicted")
    else:
        print("[Step 2] All topics already cached.")

    # Re-write to ensure file ordering / formatting
    with open(TOPIC_TO_MESH_JSON, "w", encoding="utf-8") as fp:
        json.dump(cache, fp, indent=2, ensure_ascii=False)
    return cache


# ---------------------------------------------------------------------------
# Step 3 — Compute Set A and Set B per article
# ---------------------------------------------------------------------------

def build_keyword_to_topics(
    topics_by_domain: Dict[str, dict]
) -> Dict[str, List[tuple]]:
    """Lowercased keyword -> list of (domain, topic_name) tuples."""
    mapping: Dict[str, List[tuple]] = {}
    for domain, topics in topics_by_domain.items():
        for topic_name, info in topics.items():
            for kw in info.get("keywords", []):
                key = kw.strip().lower()
                if not key:
                    continue
                mapping.setdefault(key, []).append((domain, topic_name))
    return mapping


def evaluate_articles(
    df_mesh: pd.DataFrame,
    pool: List[str],
    topic_to_mesh: Dict[str, Dict[str, List[str]]],
    topics_by_domain: Dict[str, dict],
) -> pd.DataFrame:
    pool_set = set(pool)
    keyword_to_topics = build_keyword_to_topics(topics_by_domain)

    rows: List[dict] = []
    n = len(df_mesh)
    for idx, (pmid, year, journal, mesh_str, kw_str) in enumerate(
        zip(
            df_mesh["pmid"].values,
            df_mesh["year"].values,
            df_mesh["journal"].values,
            df_mesh["mesh_terms"].values,
            df_mesh["keywords"].values,
        )
    ):
        # Set A
        article_mesh = {m.strip() for m in split_semicolon(mesh_str)}
        set_a = article_mesh & pool_set

        # Article topics
        article_topics: Set[tuple] = set()
        for kw in split_semicolon(kw_str):
            kl = kw.lower()
            if kl in keyword_to_topics:
                article_topics.update(keyword_to_topics[kl])

        # Set B
        set_b: Set[str] = set()
        for domain, topic_name in article_topics:
            preds = topic_to_mesh.get(domain, {}).get(topic_name, [])
            set_b.update(preds)
        set_b &= pool_set  # defensive

        intersection = set_a & set_b
        rows.append(
            {
                "pmid": pmid,
                "year": year,
                "journal": journal,
                "n_topics": len(article_topics),
                "set_a_size": len(set_a),
                "set_b_size": len(set_b),
                "intersection_size": len(intersection),
                "correct": len(intersection) > 0,
            }
        )
        if (idx + 1) % 10000 == 0:
            print(f"  evaluated {idx + 1:,}/{n:,}")

    out = pd.DataFrame(rows)
    out.to_csv(PER_ARTICLE_CSV, index=False)
    print(f"  wrote {PER_ARTICLE_CSV}")
    return out


# ---------------------------------------------------------------------------
# Step 4 — Aggregate and report
# ---------------------------------------------------------------------------

def write_summary(per_article: pd.DataFrame, candidate_pool_meta: dict) -> str:
    n = len(per_article)
    n_correct = int(per_article["correct"].sum())
    n_a = int((per_article["set_a_size"] > 0).sum())
    n_b = int((per_article["set_b_size"] > 0).sum())
    headline_pct = 100.0 * n_correct / n if n else 0.0

    # Per-year
    per_year = (
        per_article.groupby("year")["correct"]
        .agg(["count", "sum"])
        .rename(columns={"count": "n", "sum": "correct"})
    )
    per_year["pct"] = 100.0 * per_year["correct"] / per_year["n"]
    per_year = per_year.sort_index()

    # Per-journal
    per_journal = (
        per_article.groupby("journal")["correct"]
        .agg(["count", "sum"])
        .rename(columns={"count": "n", "sum": "correct"})
    )
    per_journal["pct"] = 100.0 * per_journal["correct"] / per_journal["n"]
    per_journal = per_journal.sort_values("pct")

    # Confusion-style buckets
    a_only = int(((per_article["set_a_size"] > 0) & (per_article["set_b_size"] == 0)).sum())
    b_only = int(((per_article["set_a_size"] == 0) & (per_article["set_b_size"] > 0)).sum())
    both_empty = int(((per_article["set_a_size"] == 0) & (per_article["set_b_size"] == 0)).sum())
    both_nonempty = int(((per_article["set_a_size"] > 0) & (per_article["set_b_size"] > 0)).sum())

    headline_line = (
        f"**{n_correct:,} / {n:,} articles correctly evaluated "
        f"({headline_pct:.1f}%)**"
    )

    md = []
    md.append("# MeSH-based Taxonomy Evaluation (replication of Fang et al. 2026)\n")
    md.append("## Headline\n")
    md.append(headline_line + "\n")
    md.append(f"Compared with the original paper: 2,009 / 2,379 (84.4%) on JBI.\n")
    md.append("\n## Candidate MeSH Pool\n")
    md.append(
        f"- pool size: **{candidate_pool_meta['pool_size']}** terms\n"
        f"- coverage: {candidate_pool_meta['n_articles_covered']:,} / "
        f"{candidate_pool_meta['n_mesh_tagged_articles']:,} "
        f"({candidate_pool_meta['coverage_fraction']:.1%})\n"
        f"- coverage target: ≥{int(COVERAGE_TARGET * 100)}%\n"
        f"- heuristic substrings used: "
        f"{', '.join(candidate_pool_meta['heuristic_substrings'])}\n"
    )
    md.append("Top 30 pool terms (by corpus frequency):\n\n")
    md.append("| MeSH term | corpus freq | matched substrings |\n|---|---|---|\n")
    for entry in candidate_pool_meta["pool"][:30]:
        md.append(
            f"| {entry['mesh_term']} | {entry['corpus_frequency']:,} | "
            f"{', '.join(entry['matched_substrings'])} |\n"
        )

    md.append("\n## Set populations\n")
    md.append(f"- articles evaluated (MeSH-tagged): **{n:,}**\n")
    md.append(f"- articles with non-empty Set A (NLM): {n_a:,} ({100.0*n_a/n:.1f}%)\n")
    md.append(f"- articles with non-empty Set B (predicted): {n_b:,} ({100.0*n_b/n:.1f}%)\n")
    md.append(f"- articles with non-empty Set A ∩ Set B (= correct): **{n_correct:,} ({headline_pct:.1f}%)**\n")

    md.append("\n## Confusion-style buckets\n\n")
    md.append("| condition | count | pct |\n|---|---|---|\n")
    md.append(f"| Set A non-empty, Set B non-empty | {both_nonempty:,} | {100.0*both_nonempty/n:.1f}% |\n")
    md.append(f"| Set A non-empty, Set B empty (pipeline missed coverage) | {a_only:,} | {100.0*a_only/n:.1f}% |\n")
    md.append(f"| Set A empty, Set B non-empty (predicted, NLM did not assign) | {b_only:,} | {100.0*b_only/n:.1f}% |\n")
    md.append(f"| Set A empty, Set B empty | {both_empty:,} | {100.0*both_empty/n:.1f}% |\n")

    md.append("\n## Per-year breakdown\n\n")
    md.append("| year | n | correct | pct |\n|---|---|---|---|\n")
    for y, row in per_year.iterrows():
        md.append(f"| {int(y)} | {int(row['n']):,} | {int(row['correct']):,} | {row['pct']:.1f}% |\n")

    md.append("\n## Per-journal breakdown (sorted ascending by pct)\n\n")
    md.append("| journal | n | correct | pct |\n|---|---|---|---|\n")
    for j, row in per_journal.iterrows():
        md.append(f"| {j} | {int(row['n']):,} | {int(row['correct']):,} | {row['pct']:.1f}% |\n")

    text = "".join(md)
    SUMMARY_MD.write_text(text, encoding="utf-8")
    print(f"  wrote {SUMMARY_MD}")
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"[main] reading corpus: {CORPUS_CSV}")
    df = pd.read_csv(
        CORPUS_CSV,
        usecols=["pmid", "year", "journal", "mesh_terms", "keywords"],
        dtype={"pmid": str},
    )
    print(f"  total rows: {len(df):,}")
    df_mesh = df[df["mesh_terms"].notna()].reset_index(drop=True)
    print(f"  with MeSH coverage: {len(df_mesh):,} ({len(df_mesh)/len(df):.1%})")

    # Step 1
    pool_meta = build_candidate_pool(df_mesh)
    pool: List[str] = [e["mesh_term"] for e in pool_meta["pool"]]

    # Step 2
    with open(HIERARCHY_DIR / "methodology_final_topics.json", "r", encoding="utf-8") as fp:
        meth = json.load(fp)
    with open(HIERARCHY_DIR / "health_final_topics.json", "r", encoding="utf-8") as fp:
        health = json.load(fp)
    topics_by_domain = {"methodology": meth, "health": health}
    print(f"  methodology topics: {len(meth)}, health topics: {len(health)}")

    topic_to_mesh = build_topic_to_mesh(pool, topics_by_domain)

    # Detect topics with zero predictions (worth flagging)
    empty_topics: List[tuple] = []
    for domain, mapping in topic_to_mesh.items():
        for t, preds in mapping.items():
            if not preds:
                empty_topics.append((domain, t))
    if empty_topics:
        print(f"\n[Note] {len(empty_topics)} topics returned zero MeSH predictions:")
        for d, t in empty_topics[:25]:
            print(f"   - {d}::{t}")
        if len(empty_topics) > 25:
            print(f"   ... and {len(empty_topics) - 25} more")

    # Step 3
    print("\n[Step 3] Evaluating articles")
    per_article = evaluate_articles(df_mesh, pool, topic_to_mesh, topics_by_domain)

    # Step 4
    print("\n[Step 4] Writing summary")
    write_summary(per_article, pool_meta)

    # Final headline print
    n = len(per_article)
    n_correct = int(per_article["correct"].sum())
    print("\n" + "=" * 60)
    print(
        f"HEADLINE: {n_correct:,} / {n:,} articles correctly evaluated "
        f"({100.0 * n_correct / n:.1f}%)"
    )
    print(f"Original Fang et al.: 2,009 / 2,379 (84.4%) on JBI")
    print("=" * 60)


if __name__ == "__main__":
    main()
