"""Table 2 v2 — top 10 methodology x health topic pairs, with sparklines that
actually show the within-pair trend.

Design changes vs v1:
- Per-sparkline y-axis auto-scales to that pair's range (not a shared 1,805
  global max). The shape of the rise/fall is now the dominant visual signal.
- Magnitude is preserved through the "Total" numeric column AND a small
  proportional bar to the right of "Total" showing each pair's share of the
  largest pair.
- Line colour encodes overall direction (start-3-yr mean vs end-3-yr mean):
    green = rising,  red = falling,  grey = flat (|change| < 10%).
- Peak year is marked with a small open circle.
- A new "Δ 2011–25" column shows percent change between the 2011–13 mean
  and 2023–25 mean — the headline trend the eye is hunting for.
"""
import os
import json
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = (
    "/Users/danielliang/Library/CloudStorage/Dropbox/"
    "multi_journal_trend_analysis (Yilun Liang's conflicted copy 2026-04-27)"
)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
VIS_DIR = os.path.join(DATA_DIR, "visualizations_k100")
HIER_DIR = os.path.join(DATA_DIR, "hierarchy_k100")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
CORPUS = os.path.join(DATA_DIR, "pubmed_all_journals_2011_2025_w_keywords_patched.csv")

YEARS = list(range(2011, 2026))

GREEN = "#0072B2"   # rising  (Okabe-Ito blue)
RED = "#D55E00"     # declining (Okabe-Ito vermillion)
GREY = "#6e6e6e"


def load_topics(domain):
    with open(os.path.join(HIER_DIR, f"{domain}_final_topics.json"), encoding="utf-8") as f:
        return json.load(f)


def kw_to_topics(topics):
    out = defaultdict(set)
    for tid, data in topics.items():
        name = data["name"]
        for kw in data.get("keywords", []):
            out[kw.strip().lower()].add(name)
    return out


def main():
    cooc = pd.read_csv(os.path.join(VIS_DIR, "cooccurrence_matrix.csv"), index_col=0)
    pairs = cooc.stack().sort_values(ascending=False).head(10)
    top_pairs = [(m, h, int(v)) for (m, h), v in pairs.items()]

    method_set = {m for m, _, _ in top_pairs}
    health_set = {h for _, h, _ in top_pairs}

    method_topics = load_topics("methodology")
    health_topics = load_topics("health")
    kw_m = kw_to_topics(method_topics)
    kw_h = kw_to_topics(health_topics)

    pair_year = {(m, h): Counter() for (m, h, _) in top_pairs}

    for chunk in pd.read_csv(CORPUS, usecols=["year", "keywords"], chunksize=50_000):
        chunk = chunk.dropna(subset=["keywords", "year"])
        chunk = chunk[chunk["year"].between(2011, 2025)]
        for yr, kws in zip(chunk["year"].astype(int), chunk["keywords"].astype(str)):
            mset, hset = set(), set()
            for kw in kws.split(";"):
                k = kw.strip().lower()
                if k in kw_m:
                    mset.update(t for t in kw_m[k] if t in method_set)
                if k in kw_h:
                    hset.update(t for t in kw_h[k] if t in health_set)
            for m in mset:
                for h in hset:
                    if (m, h) in pair_year:
                        pair_year[(m, h)][yr] += 1

    # Compute per-pair direction & % change
    pair_meta = []
    largest_total = max(t for _, _, t in top_pairs)
    for m, h, total in top_pairs:
        counts = np.array([pair_year[(m, h)].get(y, 0) for y in YEARS], dtype=float)
        early = counts[:3].mean()
        late = counts[-3:].mean()
        pct_change = (late - early) / max(1.0, early) * 100.0
        if pct_change > 10:
            direction = "up"
            colour = GREEN
        elif pct_change < -10:
            direction = "down"
            colour = RED
        else:
            direction = "flat"
            colour = GREY
        peak_idx = int(np.argmax(counts))
        pair_meta.append({
            "m": m, "h": h, "total": total, "counts": counts,
            "pct": pct_change, "dir": direction, "colour": colour,
            "peak_year": YEARS[peak_idx], "peak_val": counts[peak_idx],
            "share_of_max": total / largest_total,
        })

    # Layout: matplotlib figure with explicit per-row sparkline axes
    n = len(pair_meta)
    fig_w = 16.0
    row_h = 0.62
    title_band = 1.4
    caption_band = 1.0
    fig_h = title_band + row_h * n + caption_band
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Body axes inside the title/caption margins
    body_top = 1.0 - (title_band / fig_h)
    body_bot = caption_band / fig_h
    ax = fig.add_axes([0.0, body_bot, 1.0, body_top - body_bot])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n + 0.5)
    ax.axis("off")

    # Column x-coordinates (figure-fraction within body axes)
    x_method = 0.020
    x_health = 0.215
    x_spark_left = 0.41
    x_spark_right = 0.70
    x_change = 0.735
    x_total = 0.86
    x_bar_left = 0.880
    x_bar_right = 0.985

    # Title + subtitle (in figure coords above body axes)
    fig.text(
        0.5, 1.0 - 0.30 * (title_band / fig_h),
        "Top 10 co-occurring methodology × health topic pairs (K=100)",
        fontsize=14, fontweight="bold", ha="center", va="center",
    )
    fig.text(
        0.5, 1.0 - 0.65 * (title_band / fig_h),
        "Sparklines auto-scale per row so the within-pair trend is readable; "
        "Δ 2011–25 = percent change between 2011–13 mean and 2023–25 mean.",
        fontsize=9.5, color="#444", ha="center", va="center", style="italic",
    )

    # Header row sits inside the body axes at y = n + 0.05
    header_y = n + 0.05
    ax.text(x_method, header_y, "Methodology topic",
            fontsize=11, fontweight="bold", va="center")
    ax.text(x_health, header_y, "Health topic",
            fontsize=11, fontweight="bold", va="center")
    ax.text((x_spark_left + x_spark_right) / 2, header_y,
            "2011 → 2025 trend (per-row scale)",
            fontsize=11, fontweight="bold", va="center", ha="center")
    ax.text(x_change, header_y, "Δ 2011–25",
            fontsize=11, fontweight="bold", va="center", ha="left")
    ax.text(x_total, header_y, "Total",
            fontsize=11, fontweight="bold", va="center", ha="right")
    ax.plot([0.012, 0.992], [n - 0.32, n - 0.32], color="black", lw=0.8)

    # Per-row content (top-down ordering)
    fig.canvas.draw()  # so ax.get_position() reflects final layout
    bbox = ax.get_position()
    for i, p in enumerate(pair_meta):
        row_y = n - 0.85 - i  # row 0 just below the header underline
        # Text columns
        ax.text(x_method, row_y, p["m"], fontsize=9.5, va="center")
        ax.text(x_health, row_y, p["h"], fontsize=9.5, va="center")
        # Δ column with arrow + colour
        arrow = {"up": "▲", "down": "▼", "flat": "◆"}[p["dir"]]
        sign = "+" if p["pct"] >= 0 else ""
        ax.text(x_change, row_y,
                f"{arrow} {sign}{p['pct']:.0f}%",
                fontsize=10, fontweight="bold", va="center",
                ha="left", color=p["colour"])
        ax.text(x_total, row_y, f"{p['total']:,}", fontsize=10, va="center",
                ha="right", fontweight="bold")
        # Magnitude bar (proportional to largest pair)
        bar_w = (x_bar_right - x_bar_left) * p["share_of_max"]
        ax.add_patch(plt.Rectangle(
            (x_bar_left, row_y - 0.18), bar_w, 0.36,
            facecolor="#888", alpha=0.5, edgecolor="none",
            transform=ax.transData,
        ))
        ax.add_patch(plt.Rectangle(
            (x_bar_left, row_y - 0.18), x_bar_right - x_bar_left, 0.36,
            facecolor="none", edgecolor="#cccccc", lw=0.5,
            transform=ax.transData,
        ))

        # Sparkline inset axes (auto-scaled y)
        body_axis_h = (n + 0.5)
        y_disp = bbox.y0 + (row_y / body_axis_h) * bbox.height
        row_h_disp = 0.50 / body_axis_h * bbox.height
        x_left_disp = bbox.x0 + x_spark_left * bbox.width
        x_right_disp = bbox.x0 + x_spark_right * bbox.width
        spark_ax = fig.add_axes([
            x_left_disp,
            y_disp - row_h_disp / 2,
            x_right_disp - x_left_disp,
            row_h_disp,
        ])
        counts = p["counts"]
        spark_ax.plot(YEARS, counts, color=p["colour"], lw=1.6, marker="o",
                      markersize=3.0, markerfacecolor=p["colour"],
                      markeredgecolor=p["colour"])
        spark_ax.fill_between(YEARS, counts, color=p["colour"], alpha=0.12)
        # peak year marker
        spark_ax.plot(p["peak_year"], p["peak_val"], "o",
                      markersize=6, markerfacecolor="white",
                      markeredgecolor=p["colour"], markeredgewidth=1.5,
                      zorder=5)
        # auto-scaled y, with a little top headroom so peak marker fits
        ymin = float(counts.min())
        ymax = float(counts.max())
        yspan = max(1.0, ymax - ymin)
        spark_ax.set_ylim(ymin - yspan * 0.20, ymax + yspan * 0.25)
        spark_ax.set_xlim(2010.5, 2025.5)
        # Endpoint labels
        spark_ax.text(2011 - 0.4, counts[0], f"{int(counts[0])}",
                      fontsize=7.5, va="center", ha="right", color="#333")
        spark_ax.text(2025 + 0.4, counts[-1], f"{int(counts[-1])}",
                      fontsize=7.5, va="center", ha="left", color="#333")
        spark_ax.set_xticks([])
        spark_ax.set_yticks([])
        for s in ("top", "right", "bottom", "left"):
            spark_ax.spines[s].set_visible(False)

        if i < n - 1:
            ax.axhline(row_y - 0.5, color="#e0e0e0", lw=0.4, zorder=0)

    # Caption note
    caption = (
        "Co-occurrence: an article is counted once per pair if its author keywords map to "
        "both topics in that pair. Open circle = peak year for that pair. "
        "Per-row sparkline scaling reveals trend shape; the magnitude bar (right) and Total "
        "column preserve cross-row magnitude comparison."
    )
    fig.text(
        0.5, 0.55 * (caption_band / fig_h), caption,
        fontsize=8.5, color="#444", style="italic",
        ha="center", va="center",
    )

    out = os.path.join(FIG_DIR, "table2_cooccurrence_sparklines.png")
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
    print(f"[t2] Saved {out}")

    # Mirror to MAIN Dropbox folder
    main_mirror = "/Users/danielliang/Library/CloudStorage/Dropbox/multi_journal_trend_analysis/figures/table2_cooccurrence_sparklines.png"
    if os.path.exists(os.path.dirname(main_mirror)):
        import shutil
        shutil.copy2(out, main_mirror)
        print(f"[t2] Mirrored to {main_mirror}")
    print("[t2] Per-pair Δ:")
    for p in pair_meta:
        print(f"   {p['m']:42s} × {p['h']:35s}  total={p['total']:>6,}  Δ={p['pct']:+.0f}%  dir={p['dir']}")


if __name__ == "__main__":
    main()
