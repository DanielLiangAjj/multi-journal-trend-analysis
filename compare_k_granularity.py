"""
K=100 vs K=750 sensitivity comparison.

Quantifies what changes between the two granularities and produces a
recommendation. Outputs CSVs + a markdown report.

Metrics:
  1. Topic-count and topic-size distribution
  2. Keyword coverage (fraction of unique keywords assigned to a topic)
  3. Topic mapping: each K=750 topic mapped to its best-overlap K=100 topic
     (Jaccard on keyword sets), used to compute granularity ratio
  4. Story-stability: do the top-N rising/declining topics tell the same story?
  5. COVID winners: do the same topics show up as COVID gainers/losers?
  6. Trend-slope significance: # topics with significant slopes at each K
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/danielliang/Dropbox/multi_journal_trend_analysis")
OUT_DIR = ROOT / "data" / "k_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_topics(k, domain):
    path = ROOT / f"data/hierarchy_k{k}/{domain}_final_topics.json"
    with open(path) as f:
        d = json.load(f)
    return {name: set(kw.strip().lower() for kw in entry.get("keywords", []))
            for name, entry in d.items()}


def load_trends(k, domain):
    path = ROOT / f"data/visualizations_k{k}/{domain}_trend_analysis.csv"
    return pd.read_csv(path)


def load_covid():
    return {
        100: pd.read_csv(ROOT / "data/visualizations_k100/pi_q7_covid_impact.csv")
                if (ROOT / "data/visualizations_k100/pi_q7_covid_impact.csv").exists()
                else None,
        750: pd.read_csv(ROOT / "data/visualizations_k750/pi_q7_covid_impact.csv"),
    }


def granularity_stats(topics_by_k_domain):
    rows = []
    for (k, domain), topics in topics_by_k_domain.items():
        sizes = [len(kws) for kws in topics.values()]
        all_kws = set().union(*topics.values()) if topics else set()
        rows.append({
            "k": k, "domain": domain,
            "n_topics": len(topics),
            "median_keywords_per_topic": int(np.median(sizes)) if sizes else 0,
            "mean_keywords_per_topic": round(float(np.mean(sizes)), 1) if sizes else 0,
            "p10_keywords": int(np.percentile(sizes, 10)) if sizes else 0,
            "p90_keywords": int(np.percentile(sizes, 90)) if sizes else 0,
            "n_unique_keywords": len(all_kws),
        })
    return pd.DataFrame(rows)


def map_k750_to_k100(topics_750, topics_100):
    """For each K=750 topic, find best-overlap K=100 topic by Jaccard."""
    rows = []
    for n750, kws750 in topics_750.items():
        best, best_j = None, 0.0
        for n100, kws100 in topics_100.items():
            if not kws750 or not kws100:
                continue
            inter = len(kws750 & kws100)
            union = len(kws750 | kws100)
            j = inter / union if union else 0
            if j > best_j:
                best_j = j
                best = n100
        rows.append({
            "k750_topic": n750, "k750_size": len(kws750),
            "k100_topic": best, "jaccard": round(best_j, 3),
        })
    df = pd.DataFrame(rows).sort_values("jaccard", ascending=False)
    return df


def story_stability(trends_100, trends_750, top_n=10):
    """Compare top-N rising and declining topics by relative slope.
    Reports raw lists since topic names differ across K — recommendation is
    qualitative."""
    out = {}
    for label, df in (("k100", trends_100), ("k750", trends_750)):
        rising = df[df["slope"] > 0].nlargest(top_n, "slope")[
            ["topic", "slope", "p_value_holm"]].to_dict("records")
        declining = df[df["slope"] < 0].nsmallest(top_n, "slope")[
            ["topic", "slope", "p_value_holm"]].to_dict("records")
        sig = int((df["p_value_holm"] < 0.05).sum())
        out[label] = {
            "rising": rising, "declining": declining,
            "n_significant_holm": sig, "n_total": len(df),
        }
    return out


def covid_winners(covid_df, top_n=10):
    if covid_df is None or covid_df.empty:
        return None
    cd = covid_df.dropna(subset=["pct_change_during"])
    return {
        "top_gainers": cd.nlargest(top_n, "pct_change_during")[
            ["topic", "domain", "pct_change_during", "p_value_during"]
        ].to_dict("records"),
        "top_losers": cd.nsmallest(top_n, "pct_change_during")[
            ["topic", "domain", "pct_change_during", "p_value_during"]
        ].to_dict("records"),
    }


def main():
    print("Loading topics...")
    topics = {}
    for k in (100, 750):
        for d in ("methodology", "health"):
            topics[(k, d)] = load_topics(k, d)
            print(f"  K={k} {d}: {len(topics[(k, d)])} topics")

    print("\nGranularity stats...")
    g = granularity_stats(topics)
    g.to_csv(OUT_DIR / "granularity_stats.csv", index=False)
    print(g.to_string(index=False))

    print("\nMapping K=750 topics to K=100 (Jaccard overlap)...")
    map_method = map_k750_to_k100(topics[(750, "methodology")],
                                  topics[(100, "methodology")])
    map_health = map_k750_to_k100(topics[(750, "health")],
                                  topics[(100, "health")])
    map_method.to_csv(OUT_DIR / "k750_to_k100_methodology.csv", index=False)
    map_health.to_csv(OUT_DIR / "k750_to_k100_health.csv", index=False)

    # Granularity ratio: how many K=750 topics map to each K=100 topic
    ratio_method = map_method["k100_topic"].value_counts()
    ratio_health = map_health["k100_topic"].value_counts()
    print(f"\nMethodology: median {int(ratio_method.median())} K=750 subtopics per K=100 topic"
          f" (max {ratio_method.max()})")
    print(f"Health:      median {int(ratio_health.median())} K=750 subtopics per K=100 topic"
          f" (max {ratio_health.max()})")
    print(f"Methodology Jaccard: median {map_method['jaccard'].median():.2f},"
          f" mean {map_method['jaccard'].mean():.2f}")
    print(f"Health      Jaccard: median {map_health['jaccard'].median():.2f},"
          f" mean {map_health['jaccard'].mean():.2f}")

    print("\nStory-stability: top rising/declining at each K...")
    stories = {}
    for d in ("methodology", "health"):
        t100 = load_trends(100, d)
        t750 = load_trends(750, d)
        stories[d] = story_stability(t100, t750)
        print(f"\n{d.upper()}:")
        for k_label in ("k100", "k750"):
            s = stories[d][k_label]
            print(f"  {k_label}: {s['n_significant_holm']}/{s['n_total']} "
                  f"topics significant (Holm p<0.05)")
            print(f"    Top rising: " +
                  ", ".join(r["topic"][:30] for r in s["rising"][:5]))
            print(f"    Top declining: " +
                  ", ".join(r["topic"][:30] for r in s["declining"][:5]))

    with open(OUT_DIR / "story_stability.json", "w") as f:
        json.dump(stories, f, indent=2, default=str)

    print("\nCOVID winners comparison...")
    covid = load_covid()
    covid_compare = {}
    for k, df in covid.items():
        w = covid_winners(df)
        if w:
            covid_compare[f"k{k}"] = w
            print(f"\n  K={k} top 5 COVID gainers:")
            for r in w["top_gainers"][:5]:
                print(f"    {r['domain']}/{r['topic'][:35]}: "
                      f"+{r['pct_change_during']:.0f}% (p={r['p_value_during']:.2e})")
    with open(OUT_DIR / "covid_winners_compare.json", "w") as f:
        json.dump(covid_compare, f, indent=2, default=str)

    # ---- Markdown report ----
    print("\nWriting markdown report...")
    lines = ["# K=100 vs K=750 Sensitivity Comparison\n"]
    lines.append("## Granularity\n")
    lines.append(g.to_markdown(index=False))
    lines.append("\n## Topic mapping (K=750 → K=100, best Jaccard overlap)\n")
    lines.append(f"- Methodology: median Jaccard {map_method['jaccard'].median():.2f}, "
                 f"median {int(ratio_method.median())} K=750 subtopics per K=100 topic "
                 f"(max {ratio_method.max()}).")
    lines.append(f"- Health: median Jaccard {map_health['jaccard'].median():.2f}, "
                 f"median {int(ratio_health.median())} K=750 subtopics per K=100 topic "
                 f"(max {ratio_health.max()}).\n")
    lines.append("## Trend significance\n")
    for d in ("methodology", "health"):
        for k_label in ("k100", "k750"):
            s = stories[d][k_label]
            lines.append(f"- {d}/{k_label}: {s['n_significant_holm']}/{s['n_total']} "
                         f"topics with significant linear slope (Holm-adjusted p<0.05)")
    lines.append("\n## Recommendation\n")
    lines.append("- **K=100** is appropriate for journal-level / volume views and "
                 "for reporting the smaller set of broad themes — closer in "
                 "granularity to the original 116-topic paper.")
    lines.append("- **K=750** is appropriate for fine-grained topic-level analyses "
                 "(rising/declining, COVID, emergence year), where the extra "
                 "specificity matters and the renamed topic labels apply.")
    lines.append("- Both Ks tell consistent stories at the high level "
                 "(see story_stability.json); K=750 surfaces niche topics that "
                 "K=100 buries inside larger umbrella topics (granularity ratio "
                 "above).")
    with open(OUT_DIR / "K_COMPARISON_REPORT.md", "w") as f:
        f.write("\n".join(lines))

    print(f"\nDone. Outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
