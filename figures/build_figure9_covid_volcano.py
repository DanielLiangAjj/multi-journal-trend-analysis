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
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent)); from _mirror import mirror_to_main
import matplotlib.pyplot as plt
from adjustText import adjust_text

PROJECT_ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
VIS_DIR = os.path.join(PROJECT_ROOT, "data", "visualizations_k100")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")

CAP = 50.0  # cap displayed -log10(p) at this value
COLOR = {"methodology": "#1f77b4", "health": "#ff7f0e"}


def main():
    df = pd.read_csv(os.path.join(VIS_DIR, "pi_q7_covid_impact.csv"))
    df = df.dropna(subset=["pct_change_during", "p_value_during"]).copy()

    # Holm-Bonferroni adjust the 'during' p-values across the testable topics so the
    # plotted threshold matches the Holm-based counts reported in the text
    # (90/133 significant; 88 up, 2 down). Without this the figure plots RAW p.
    from statsmodels.stats.multitest import multipletests
    df["p_holm"] = multipletests(df["p_value_during"].clip(lower=1e-300),
                                 method="holm")[1]
    n_sig = int((df["p_holm"] < 0.05).sum())
    print(f"[fig9] Holm-significant during-window topics: {n_sig} / {len(df)}")

    # Compute raw and capped -log10 of the Holm-adjusted p.
    # Floor at 1e-300 so -log10 is finite; then cap at CAP for display.
    safe_p = df["p_holm"].clip(lower=1e-300)
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
    ax.set_xlim(xmin - xpad * 3.0, xmax + xpad * 3.2)
    ax.set_ylim(-2, CAP + 4)  # small headroom above cap

    # Annotate the Holm threshold near the right end of the line
    # Far right, just above the line: the only band that stays clear of the
    # point cloud and of the callout boxes adjustText places.
    ax.text(
        ax.get_xlim()[1] - (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.012,
        holm_y + 1.0,
        "p = 0.05 (Holm)",
        color="red", fontsize=9, fontweight="bold",
        ha="right", va="bottom",
    )

    # Top-5 gainers and decliners (by pct_change_during), plus every
    # Holm-significant decliner (the text names biomedical databases AND
    # mixed medical imaging; the latter is outside the raw top-5 and was
    # previously unlabeled — Casey's comment).
    top_pos = df.nlargest(5, "pct_change_during")
    top_neg = df.nsmallest(5, "pct_change_during")
    sig_neg = df[(df["p_holm"] < 0.05) & (df["pct_change_during"] < 0)]
    callouts = pd.concat([top_pos, top_neg, sig_neg]).drop_duplicates(subset=["topic"])

    print("[fig9] Top-5 gainers:")
    for _, r in top_pos.iterrows():
        print(f"   {r['domain']:12s} {r['topic'][:45]:45s}  pct={r['pct_change_during']:+7.1f}%  p={r['p_value_during']:.2e}")
    print("[fig9] Top-5 decliners:")
    for _, r in top_neg.iterrows():
        print(f"   {r['domain']:12s} {r['topic'][:45]:45s}  pct={r['pct_change_during']:+7.1f}%  p={r['p_value_during']:.2e}")

    # Near-floor labels (-log10 p < 2.5) pile up against the axis where
    # adjustText cannot separate them; stack those by hand in the empty
    # far-left region with leader lines. The rest go through adjustText.
    is_manual = (callouts["neg_log10_p"] < 2.5) | (callouts["topic"] == "Biomedical databases")
    manual = callouts[is_manual].sort_values("neg_log10_p", ascending=False)
    auto = callouts[~is_manual]
    x_stack = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.005
    y_slots = [22.0, 17.5, 13.0, 8.5, 4.0]
    for slot, (_, r) in zip(y_slots, manual.iterrows()):
        ax.annotate(
            f"{r['topic']}\n({r['pct_change_during']:+.0f}%)",
            xy=(r["pct_change_during"], r["neg_log10_p"]),
            xytext=(x_stack, slot),
            fontsize=8, color="black", ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=COLOR[r["domain"]], lw=0.8, alpha=0.93),
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.7, alpha=0.7),
            zorder=5,
        )

    # Build label text with topic name + pct on separate lines so duplicate
    # percentages are disambiguated by the topic line.
    texts = []
    for _, r in auto.iterrows():
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
    np.random.seed(0)  # adjust_text() draws from the global RNG
    adjust_text(
        texts,
        iter_lim=200,
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
    ax.set_ylabel("-log10 Holm-adjusted Poisson p (during-window)", fontsize=12)

    # In-image titles removed — the caption carries this information
    # (reviewer request: figure titles belong in captions).

    # Legend (lower right, bigger swatches for clarity)
    leg = ax.legend(
        title="Domain", loc="upper left",
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
    fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")
    mirror_to_main(out)
    plt.close(fig)
    print(f"[fig9] Saved {out}")


if __name__ == "__main__":
    main()
