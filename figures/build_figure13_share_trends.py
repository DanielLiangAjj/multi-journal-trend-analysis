"""Figure 13 — Holm-significant share trends from the quasi-Poisson model.

Replaces data/visualizations_k100/n5_*_rising_declining.png, which plotted an
uncorrected linear share slope in a red/green palette, showed only an arbitrary
top-10 per side, gave the two panels different x-scales, and labelled bars with
2026-inclusive article totals.

This version plots the model the Methods actually specify (§3.3.1): an
overdispersion-robust Poisson GLM of annual counts with a log-corpus-size
offset, Holm-corrected within domain. Only topics whose share trend survives
Holm are drawn, so the figure is the direct visual for the §4.2.3 counts.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _mirror import mirror_to_main

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "figure13_share_trends.png"

YEARS = np.arange(2011, 2026)
UP, DOWN, INK, MUTED, GRID = "#0072B2", "#D55E00", "#1a1a1a", "#5c5c5c", "#dcdcdc"


def corpus_by_year():
    counts = pd.Series(0, index=YEARS, dtype=int)
    for ch in pd.read_csv(ROOT / "data/pubmed_all_journals_2011_2025_w_keywords_patched.csv",
                          usecols=["year"], chunksize=100_000, low_memory=False):
        v = ch.year.value_counts()
        for y in YEARS:
            counts[y] += int(v.get(y, 0))
    return counts.values.astype(float)


def share_trends(domain, corpus):
    df = pd.read_csv(ROOT / f"data/visualizations_k100/{domain}_topic_year_counts.csv")
    ycols = [str(y) for y in YEARS]
    x = sm.add_constant(YEARS - 2011)
    rows = []
    for _, r in df.iterrows():
        y = np.array([float(r[c]) for c in ycols])
        if y.sum() < 10:
            continue
        try:
            m = sm.GLM(y, x, family=sm.families.Poisson(),
                       offset=np.log(corpus)).fit(scale="X2")
        except Exception:
            continue
        rows.append((r["topic"], m.params[1], m.pvalues[1], y.sum()))
    R = pd.DataFrame(rows, columns=["topic", "slope", "p", "n"])
    R["sig"], R["p_holm"] = multipletests(R.p, alpha=0.05, method="holm")[:2]
    R["pct"] = (np.exp(R.slope) - 1) * 100
    R.to_csv(ROOT / f"data/visualizations_k100/{domain}_share_trends.csv", index=False)
    return R


def draw(ax, R, panel, domain_label):
    sig = R[R.sig].sort_values("pct", ascending=True).reset_index(drop=True)
    y = np.arange(len(sig))
    colours = [UP if v > 0 else DOWN for v in sig.pct]
    ax.barh(y, sig.pct, color=colours, alpha=0.88, height=0.72,
            edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(0, color=INK, lw=1.0, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(sig.topic, fontsize=9)
    ax.set_ylim(-0.8, len(sig) - 0.2)

    span = max(abs(sig.pct.min()), abs(sig.pct.max()))
    # Per-side headroom: the value labels sit outside the bar ends, and on the
    # losing side they would otherwise run into the topic names.
    pad = span * 0.95
    ax.set_xlim(min(sig.pct.min(), 0) - pad, max(sig.pct.max(), 0) + pad)
    for yi, (v, n) in enumerate(zip(sig.pct, sig.n)):
        off = span * 0.045
        ax.text(v + (off if v > 0 else -off), yi,
                f"{v:+.1f}%/yr  (n={int(n):,})",
                va="center", ha="left" if v > 0 else "right",
                fontsize=8.2, color=INK, zorder=5)

    nu, nd = int((sig.pct > 0).sum()), int((sig.pct < 0).sum())
    ax.set_title(f"({panel}) {domain_label}\n"
                 f"{len(sig)} of {len(R)} topics with a Holm-significant share trend "
                 f"— {nu} gaining, {nd} losing",
                 fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=10)
    ax.set_xlabel("Change in share of annual corpus output (% per year)",
                  fontsize=10.5, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.tick_params(axis="y", length=0)
    return sig


def main():
    corpus = corpus_by_year()
    print(f"corpus 2011-2025 = {int(corpus.sum()):,}")
    meth = share_trends("methodology", corpus)
    heal = share_trends("health", corpus)

    n_rows = max(int(meth.sig.sum()), int(heal.sig.sum()))
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 1.6 + 0.30 * n_rows))
    sm_ = draw(axes[0], meth, "A", "Methodology topics")
    sh_ = draw(axes[1], heal, "B", "Health topics")

    # in-image figure title removed (caption carries it)
    fig.text(0.5, 0.008,
             "Positive values gain share of annual corpus output; negative values lose share. "
             "Bars show only topics significant after Holm correction; n = articles 2011–2025. "
             "Absolute counts rise for most topics (Figures 4a, 4b) because the corpus itself grew 7.6-fold.",
             ha="center", va="bottom", fontsize=9, style="italic", color=MUTED)

    fig.tight_layout(rect=[0, 0.028, 1, 0.955])
    fig.savefig(OUT, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT}")
    for lbl, R, s in (("methodology", meth, sm_), ("health", heal, sh_)):
        print(f"  {lbl}: {len(R)} tested, {int(R.sig.sum())} Holm-sig "
              f"({int((s.pct > 0).sum())} up / {int((s.pct < 0).sum())} down)")
        print(f"     largest gain: {s.iloc[-1].topic} {s.iloc[-1].pct:+.1f}%/yr")
        print(f"     largest loss: {s.iloc[0].topic} {s.iloc[0].pct:+.1f}%/yr")
    mirror_to_main(OUT)


if __name__ == "__main__":
    main()
