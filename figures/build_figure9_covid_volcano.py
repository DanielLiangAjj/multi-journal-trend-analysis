#!/usr/bin/env python3
"""Rebuild Figure 9 (COVID-19 impact volcano) with a usable y-axis.

Strategy:
- Cap displayed -log10(p_value_during) at 50; mark capped points with a small
  upward arrow ('uparrow') glyph next to them and add an axis footnote.
- Make the dashed Holm threshold line at y = -log10(0.05) clearly visible
  (red, slightly thicker, annotated near right edge).
- Use adjustText for collision-free label placement with short straight leaders.
- Top-5 gainer + top-5 decliner labels show topic name + percentage on
  separate lines (so duplicate "+211%" entries are disambiguated).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text

PROJECT_ROOT = "/Users/danielliang/Library/CloudStorage/Dropbox/multi_journal_trend_analysis"
VIS_DIR = os.path.join(PROJECT_ROOT, "data", "visualizations_k100")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")

CAP = 50.0  # cap displayed -log10(p) at this value
COLOR = {"methodology": "#1f77b4", "health": "#ff7f0e"}


def main():
    df = pd.read_csv(os.path.join(VIS_DIR, "pi_q7_covid_impact.csv"))
    df = df.dropna(subset=["pct_change_during", "p_value_during"]).copy()

    # Compute raw and capped -log10 p
    # Floor at 1e-300 so -log10 is finite; then cap at CAP for display.
    safe_p = df["p_value_during"].clip(lower=1e-300)
    df["neg_log10_p_raw"] = -np.log10(safe_p)
    df["neg_log10_p"] = df["neg_log10_p_raw"].clip(upper=CAP)
    df["capped"] = df["neg_log10_p_raw"] > CAP

    n_capped = int(df["capped"].sum())
    print(f"[fig9] {n_capped} points capped at -log10(p) = {CAP:.0f}")

    fig, ax = plt.subplots(figsize=(12, 7.5))

    # Scatter by domain
    for dom in ("methodology", "health"):
        sub = df[df["domain"] == dom]
        ax.scatter(
            sub["pct_change_during"], sub["neg_log10_p"],
            c=COLOR[dom], s=50, alpha=0.78,
            edgecolor="white", linewidth=0.6,
            label=dom.capitalize(),
            zorder=3,
        )

    # Mark capped points with an up-arrow just to the right of the marker
    capped = df[df["capped"]]
    for _, r in capped.iterrows():
        ax.annotate(
            "↑",  # up arrow
            xy=(r["pct_change_during"], r["neg_log10_p"]),
            xytext=(6, 0), textcoords="offset points",
            fontsize=10, color=COLOR[r["domain"]],
            ha="left", va="center", fontweight="bold", zorder=4,
        )

    # Reference lines
    ax.axvline(0, color="black", linestyle="--", lw=0.9, alpha=0.5, zorder=1)
    holm_y = -np.log10(0.05)
    ax.axhline(holm_y, color="red", linestyle="--", lw=1.4, alpha=0.85, zorder=2)

    # Compute padded x range to leave room for end-of-line annotation
    xmin, xmax = df["pct_change_during"].min(), df["pct_change_during"].max()
    xpad = (xmax - xmin) * 0.08
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(-2, CAP + 4)  # small headroom above cap

    # Annotate the Holm threshold near the right end of the line
    ax.text(
        ax.get_xlim()[1] * 0.99, holm_y + 1.2,
        "p = 0.05 (Holm)",
        color="red", fontsize=9, fontweight="bold",
        ha="right", va="bottom",
    )

    # Top-5 gainers and decliners (by pct_change_during)
    top_pos = df.nlargest(5, "pct_change_during")
    top_neg = df.nsmallest(5, "pct_change_during")
    callouts = pd.concat([top_pos, top_neg])

    print("[fig9] Top-5 gainers:")
    for _, r in top_pos.iterrows():
        print(f"   {r['domain']:12s} {r['topic'][:45]:45s}  pct={r['pct_change_during']:+7.1f}%  p={r['p_value_during']:.2e}")
    print("[fig9] Top-5 decliners:")
    for _, r in top_neg.iterrows():
        print(f"   {r['domain']:12s} {r['topic'][:45]:45s}  pct={r['pct_change_during']:+7.1f}%  p={r['p_value_during']:.2e}")

    # Build label text with topic name + pct on separate lines so duplicate
    # percentages are disambiguated by the topic line.
    texts = []
    for _, r in callouts.iterrows():
        label = f"{r['topic']}\n({r['pct_change_during']:+.0f}%)"
        t = ax.text(
            r["pct_change_during"], r["neg_log10_p"], label,
            fontsize=8, color="black",
            ha="center", va="center",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor=COLOR[r["domain"]],
                lw=0.8, alpha=0.93,
            ),
            zorder=5,
        )
        texts.append(t)

    # adjustText for collision avoidance with short straight leaders
    adjust_text(
        texts,
        ax=ax,
        expand=(1.4, 1.6),
        force_text=(0.6, 0.9),
        force_static=(0.3, 0.5),
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.7, alpha=0.7),
    )

    # Axis labels (verified against analyze_all_questions.py: pre=2018-2019,
    # during=2020-2021, post=2022-2025)
    ax.set_xlabel(
        "% change during COVID-19 (2020-2021 vs 2018-2019)", fontsize=12,
    )
    ax.set_ylabel("-log10 Poisson p-value (during-window)", fontsize=12)

    # Title + subtitle (period not colon to match other figures)
    fig.suptitle(
        "Figure 9. COVID-19 impact volcano: effect size × statistical significance",
        fontsize=14, fontweight="bold", y=0.995,
    )
    ax.set_title(
        "Each point = one topic (K=100). Topics above the dashed line have Poisson p < 0.05.",
        fontsize=10, color="#444444", style="italic", pad=8,
    )

    # Legend (lower right, bigger swatches for clarity)
    leg = ax.legend(
        title="Domain", loc="lower right",
        fontsize=10, title_fontsize=10,
        markerscale=1.3, framealpha=0.95,
    )
    leg.get_frame().set_edgecolor("#888888")

    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)

    # Footnote about the cap
    if n_capped > 0:
        fig.text(
            0.01, 0.005,
            f"Note: y-axis capped at -log10(p) = {CAP:.0f}; {n_capped} highly significant "
            "points marked with $\\uparrow$ extend beyond.",
            fontsize=8, color="#444444", style="italic",
            ha="left", va="bottom",
        )

    out = os.path.join(FIG_DIR, "figure9_covid_volcano.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig9] Saved {out}")


if __name__ == "__main__":
    main()
