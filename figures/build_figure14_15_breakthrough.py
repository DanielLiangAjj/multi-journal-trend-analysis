"""Figure 14/15 — topic breakthrough year (first year crossing 25% of eventual peak).

Replaces the old N4 figure that mistakenly plotted "year of steepest YoY growth"
(which clusters on 2025 because 2025 was the field-wide +33% jump). The new
metric — first 25%-of-peak crossing — actually distributes across 2011–2024
and matches the manuscript text.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MAIN_MIRROR = Path("/Users/danielliang/Library/CloudStorage/Dropbox/multi_journal_trend_analysis")

YEARS = list(range(2011, 2026))
YEAR_COLS = [str(y) for y in YEARS]


def compute_breakthrough(domain: str, top_n: int = 30):
    counts = pd.read_csv(ROOT / f"data/visualizations_k100/{domain}_topic_year_counts.csv")
    cols = [c for c in YEAR_COLS if c in counts.columns]
    rows = []
    for _, r in counts.iterrows():
        v = r[cols].values.astype(float)
        if v.sum() < 100:
            continue
        peak_val = float(v.max())
        peak_year = int(cols[int(np.argmax(v))])
        threshold = 0.25 * peak_val
        v_2011 = float(v[0])
        ratio_2011 = v_2011 / peak_val if peak_val > 0 else 0
        cross_year = None
        for i, val in enumerate(v):
            if val >= threshold:
                cross_year = int(cols[i])
                break
        rows.append({
            "topic": r.iloc[0],
            "total": int(v.sum()),
            "peak_year": peak_year,
            "peak_val": peak_val,
            "ratio_2011": ratio_2011,
            "established_at_start": ratio_2011 >= 0.25,
            "breakthrough_year": cross_year,
            "value_at_breakthrough": v[cols.index(str(cross_year))],
        })
    df = pd.DataFrame(rows).sort_values("breakthrough_year")
    return df.head(top_n) if top_n else df


def build_panel(domain: str, fig_num: int, out_path: Path):
    df_all = compute_breakthrough(domain, top_n=None)
    # Two groups: established at corpus start (2011 ≥ 25% of peak) vs genuine emergents
    df_established = df_all[df_all["established_at_start"]].sort_values("total", ascending=False)
    df = df_all[~df_all["established_at_start"]].sort_values("breakthrough_year").reset_index(drop=True)
    n = len(df)
    n_est = len(df_established)

    fig_h = max(11, 0.34 * max(n, n_est) + 3.5)
    fig = plt.figure(figsize=(18, fig_h))
    ax_est = fig.add_axes([0.03, 0.06, 0.18, 0.86])
    ax = fig.add_axes([0.34, 0.06, 0.62, 0.86])

    # ===== LEFT PANEL: established-at-corpus-start topics =====
    ax_est.set_title(
        f"Established at corpus start\n(2011 count ≥ 25% of eventual peak; n={n_est})",
        fontsize=10.5, fontweight="bold", color="#666", pad=8,
    )
    if n_est > 0:
        sizes_est = 50 + 5.0 * np.sqrt(df_established["total"].values)
        y_est = np.arange(n_est)
        ax_est.scatter([0.85] * n_est, y_est, s=sizes_est,
                       c="#888", alpha=0.7, edgecolors="white", linewidth=0.6)
        for i, (_, r) in enumerate(df_established.iterrows()):
            ax_est.text(0.78, i, r["topic"], fontsize=9, ha="right", va="center", color="#222")
            ax_est.text(0.93, i, f"n={int(r['total']):,}",
                        fontsize=8, ha="left", va="center", color="#666")
    ax_est.set_xlim(0.0, 1.05)
    ax_est.set_ylim(-0.7, max(n_est, n) - 0.3)
    ax_est.invert_yaxis()
    ax_est.set_xticks([])
    ax_est.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax_est.spines[s].set_visible(False)
    # Caption note positioned above the panel
    ax_est.text(0.5, -2.0, "True breakthrough year\nis pre-2011 (unobservable)",
                fontsize=8.5, ha="center", va="bottom", color="#888", style="italic")

    # ===== RIGHT PANEL: genuinely-emergent topics, breakthrough timeline =====
    # Color by breakthrough era band
    # Era bins now match the drawn era bands exactly (2011-2015 /
    # 2016-2020 / 2021-2025); the old <=2014 / <=2019 cuts coloured
    # 2015 and 2020 breakthroughs with the NEXT era's colour (reviewer catch).
    colours = []
    for y in df["breakthrough_year"]:
        if y <= 2015:
            colours.append("#1f77b4")  # genomics-foundational era
        elif y <= 2020:
            colours.append("#2ca02c")  # ML surge
        else:
            colours.append("#d62728")  # LLM / Gen-AI era

    y_pos = np.arange(n)
    sizes = 60 + 6.0 * np.sqrt(df["total"].values)
    ax.scatter(df["breakthrough_year"], y_pos, s=sizes, c=colours,
               alpha=0.85, edgecolors="white", linewidth=0.6, zorder=3)

    # Topic labels on the left of dots; metadata on the FAR RIGHT (away from dots)
    x_meta = 2026.7  # fixed column past the rightmost era band
    for i, (_, r) in enumerate(df.iterrows()):
        ax.text(r["breakthrough_year"] - 0.35, i, r["topic"],
                fontsize=9.0, ha="right", va="center", color="#222")
        ax.text(x_meta, i,
                f"peak {int(r['peak_year'])} · n={int(r['total']):,}",
                fontsize=7.8, ha="left", va="center", color="#666")

    ax.set_yticks([])
    ax.set_xlim(2010.5, 2030.0)
    ax.set_ylim(-0.7, n - 0.3)
    ax.invert_yaxis()
    ax.set_xticks(YEARS[::1])
    ax.tick_params(axis="x", labelsize=8.5, rotation=45)
    ax.set_xlabel("Breakthrough year (first year crossing 25% of eventual peak)", fontsize=11)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.25, lw=0.5)

    # Era band guides
    ax.axvspan(2010.5, 2015.5, alpha=0.05, color="#1f77b4", zorder=0)
    ax.axvspan(2015.5, 2020.5, alpha=0.05, color="#2ca02c", zorder=0)
    ax.axvspan(2020.5, 2026.5, alpha=0.05, color="#d62728", zorder=0)
    # Era labels at top
    ax.text(2013, -0.3, "Genomics era", fontsize=9, color="#1f77b4",
            fontweight="bold", ha="center")
    ax.text(2018, -0.3, "ML surge", fontsize=9, color="#2ca02c",
            fontweight="bold", ha="center")
    ax.text(2023.5, -0.3, "LLM / Gen-AI era", fontsize=9, color="#d62728",
            fontweight="bold", ha="center")

    domain_label = "methodology" if domain == "methodology" else "health"
    letter = "a" if domain == "methodology" else "b"
    fig.text(0.01, 0.985, f"({letter}) {domain_label.capitalize()}",
             fontsize=14, fontweight="bold", ha="left", va="top")

    # Don't call tight_layout — we set axes positions manually for the dual panel.
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  wrote {out_path}")
    if MAIN_MIRROR.exists():
        import shutil
        shutil.copy2(out_path, MAIN_MIRROR / "figures" / out_path.name)
        print(f"  mirrored to MAIN/figures/{out_path.name}")


def main():
    out_dir = ROOT / "figures"
    build_panel("methodology", 13, out_dir / "figure14_methodology_breakthrough.png")
    build_panel("health", 14, out_dir / "figure15_health_breakthrough.png")

    # Also print the per-year distribution for the manuscript
    print()
    for d in ["methodology", "health"]:
        df = compute_breakthrough(d, top_n=None)
        print(f"\n=== {d} breakthrough year distribution ===")
        print(df.breakthrough_year.value_counts().sort_index().to_string())
        print(f"  n_topics with ≥100 articles: {len(df)}")
        print(f"  median breakthrough year: {df.breakthrough_year.median():.0f}")


if __name__ == "__main__":
    main()
