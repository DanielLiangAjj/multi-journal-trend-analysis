#!/usr/bin/env python3
"""Polish Table 2 (top 10 co-occurring methodology x health topic pairs).

QA fixes applied:
- Use Unicode glyphs (× and →) instead of ASCII (x and ->).
- Sparklines share a global y-axis (max across all 10 pairs across all years),
  so visual height correctly conveys magnitude.
- Add a caption defining "co-occurrence" and stating the shared y-max.
- Right-align the "Total" numeric column.
"""
import os
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = "/Users/danielliang/Library/CloudStorage/Dropbox/multi_journal_trend_analysis"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
VIS_DIR = os.path.join(DATA_DIR, "visualizations_k100")
HIER_DIR = os.path.join(DATA_DIR, "hierarchy_k100")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
CORPUS = os.path.join(DATA_DIR, "pubmed_all_journals_2011_2025_w_keywords_patched.csv")

YEARS = list(range(2011, 2026))


def load_topics(domain):
    with open(os.path.join(HIER_DIR, f"{domain}_final_topics.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def build_kw_to_topics(topics):
    kw_to = defaultdict(set)
    for tid, data in topics.items():
        name = data["name"]
        for kw in data.get("keywords", []):
            kw_to[kw.strip().lower()].add(name)
    return kw_to


def main():
    print("[t2] Loading co-occurrence matrix to identify top 10 pairs ...")
    cooc = pd.read_csv(os.path.join(VIS_DIR, "cooccurrence_matrix.csv"), index_col=0)
    pairs = cooc.stack().sort_values(ascending=False).head(10)
    top_pairs = [(m, h, int(v)) for (m, h), v in pairs.items()]
    print("  top 10 pairs (M, H, total):")
    for p in top_pairs:
        print("   ", p)

    method_set = {m for m, _, _ in top_pairs}
    health_set = {h for _, h, _ in top_pairs}

    method_topics = load_topics("methodology")
    health_topics = load_topics("health")
    kw_to_method = build_kw_to_topics(method_topics)
    kw_to_health = build_kw_to_topics(health_topics)

    pair_year = {(m, h): Counter() for (m, h, _) in top_pairs}

    print("[t2] Walking corpus in chunks ...")
    nrows_total = 0
    chunksize = 50_000
    for chunk in pd.read_csv(CORPUS, usecols=["year", "keywords"], chunksize=chunksize):
        nrows_total += len(chunk)
        chunk = chunk.dropna(subset=["keywords", "year"])
        chunk = chunk[chunk["year"].between(2011, 2025)]
        for year_val, kw_str in zip(chunk["year"].astype(int), chunk["keywords"].astype(str)):
            mset = set()
            hset = set()
            for kw in kw_str.split(";"):
                k = kw.strip().lower()
                if k in kw_to_method:
                    for t in kw_to_method[k]:
                        if t in method_set:
                            mset.add(t)
                if k in kw_to_health:
                    for t in kw_to_health[k]:
                        if t in health_set:
                            hset.add(t)
            if not mset or not hset:
                continue
            for m in mset:
                for h in hset:
                    if (m, h) in pair_year:
                        pair_year[(m, h)][year_val] += 1
    print(f"  scanned {nrows_total} rows")

    # ---- Compute shared y-axis max across all pairs / all years ----
    global_ymax = 0
    for (m, h, _) in top_pairs:
        for y in YEARS:
            v = pair_year[(m, h)].get(y, 0)
            if v > global_ymax:
                global_ymax = v
    print(f"[t2] Shared sparkline y-axis max = {global_ymax}")

    # ---- Build figure ----
    n = len(top_pairs)
    fig, ax = plt.subplots(figsize=(12.6, 0.55 * n + 1.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n + 1.0)
    ax.invert_yaxis()
    ax.axis("off")

    fig.suptitle(
        "Table 2. Top 10 co-occurring methodology × health topic pairs (K=100)",
        fontsize=14, fontweight="bold", y=0.985,
    )

    # Column x positions (axes-fraction)
    x_method = 0.02
    x_health = 0.30
    x_spark_left = 0.58
    x_spark_right = 0.88
    x_total = 0.99  # right-edge for right-aligned "Total"

    # Header row
    header_y = 0.55
    ax.text(x_method, header_y, "Methodology topic",
            fontsize=11, fontweight="bold", va="center")
    ax.text(x_health, header_y, "Health topic",
            fontsize=11, fontweight="bold", va="center")
    ax.text((x_spark_left + x_spark_right) / 2, header_y,
            "2011 → 2025 trend",
            fontsize=11, fontweight="bold", va="center", ha="center")
    ax.text(x_total, header_y, "Total",
            fontsize=11, fontweight="bold", va="center", ha="right")

    # Header underline
    ax.plot([0.01, 0.99], [0.95, 0.95],
            color="black", lw=0.8, transform=ax.transAxes)

    # Per-row content
    bbox = ax.get_position()
    for i, (m, h, total) in enumerate(top_pairs):
        row_y = i + 1.4
        ax.text(x_method, row_y, m, fontsize=9, va="center")
        ax.text(x_health, row_y, h, fontsize=9, va="center")
        # Right-aligned numeric column
        ax.text(x_total, row_y, f"{total:,}", fontsize=9, va="center",
                ha="right", fontweight="bold")

        counts = [pair_year[(m, h)].get(y, 0) for y in YEARS]

        # Inset axes for sparkline (figure-fraction coords)
        y_center_disp = bbox.y0 + (1 - row_y / (n + 1.0)) * bbox.height
        row_h_disp = 0.55 / (n + 1.0) * bbox.height
        x_left_disp = bbox.x0 + x_spark_left * bbox.width
        x_right_disp = bbox.x0 + x_spark_right * bbox.width
        spark_ax = fig.add_axes([
            x_left_disp,
            y_center_disp - row_h_disp / 2,
            x_right_disp - x_left_disp,
            row_h_disp,
        ])
        spark_ax.plot(YEARS, counts, color="#1f4ea0", lw=1.2, marker="o",
                      markersize=2.8, markerfacecolor="#1f4ea0",
                      markeredgecolor="#1f4ea0")
        spark_ax.fill_between(YEARS, counts, color="#1f4ea0", alpha=0.10)
        spark_ax.set_xlim(2011, 2025)
        # SHARED y-axis across all 10 sparklines
        spark_ax.set_ylim(-global_ymax * 0.05, global_ymax * 1.10)

        # Endpoint annotations
        spark_ax.text(2011 - 0.3, counts[0], f"{counts[0]}",
                      fontsize=6, va="center", ha="right", color="#444444")
        spark_ax.text(2025 + 0.3, counts[-1], f"{counts[-1]}",
                      fontsize=6, va="center", ha="left", color="#444444")
        spark_ax.set_xticks([])
        spark_ax.set_yticks([])
        for s in ("top", "right", "bottom", "left"):
            spark_ax.spines[s].set_visible(False)

        if i < n - 1:
            ax.axhline(row_y + 0.5, color="#e0e0e0", lw=0.4, zorder=0)

    # Caption note (defines co-occurrence + states shared y-max)
    caption = (
        f"Co-occurrence: an article is counted once per pair if its author "
        f"keywords map to both topics in that pair. "
        f"Sparklines share a common y-axis (max = {global_ymax})."
    )
    fig.text(
        0.5, 0.005, caption,
        fontsize=8.5, color="#444444", style="italic",
        ha="center", va="bottom", wrap=True,
    )

    out = os.path.join(FIG_DIR, "table2_cooccurrence_sparklines.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[t2] Saved {out}")


if __name__ == "__main__":
    main()
