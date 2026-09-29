"""Figure 6 (combined): methodology + health topics across three 5-year windows.

Merges the former Figures 6 and 7 into one four-panel figure per Casey's
review (saves a figure slot): (a)/(b) methodology volume and share change,
(c)/(d) health. Two fixes vs the old per-domain builders:
  - bar labels now print the 2011-2025 window sum (the old labels used the
    CSV 'total' column, which includes 2026 records, so labels disagreed
    with the plotted bars by up to ~1%);
  - no in-image figure title (captions carry it).
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

CONFLICT = Path(__file__).resolve().parent.parent
DATA = CONFLICT / "data" / "visualizations_k100"
OUT = CONFLICT / "figures" / "figure6_windows_combined.png"

WINDOWS = [("2011–2015", range(2011, 2016)),
           ("2016–2020", range(2016, 2021)),
           ("2021–2025", range(2021, 2026))]
WCOLORS = ["#add8e6", "#6495ed", "#1f3a93"]
TOP_N = 20


def short(s, n=38):
    return s if len(s) <= n else s[: n - 1] + "…"


def panels(domain, axL, axR, letterL, letterR):
    df = pd.read_csv(DATA / f"{domain}_topic_year_counts.csv")
    ycols = [c for c in df.columns if str(c).strip().isdigit()]
    for wname, yrs in WINDOWS:
        cols = [c for c in ycols if int(c) in yrs]
        df[wname] = df[cols].sum(axis=1)
    wnames = [w[0] for w in WINDOWS]
    df["win_sum"] = df[wnames].sum(axis=1)          # 2011-2025 only
    win_total = {w: df[w].sum() for w in wnames}
    for w in wnames:
        df[f"share_{w}"] = df[w] / win_total[w] * 100.0

    top = df.sort_values("win_sum", ascending=False).head(TOP_N).copy()
    top = top.sort_values("win_sum", ascending=True).reset_index(drop=True)
    y = np.arange(len(top))
    names = [short(t) for t in top["topic"]]

    left = np.zeros(len(top))
    for w, c in zip(wnames, WCOLORS):
        axL.barh(y, top[w], left=left, color=c, label=w,
                 edgecolor="white", linewidth=0.4)
        left += top[w].values
    for yi, tot in zip(y, top["win_sum"]):
        axL.text(tot + win_total[wnames[0]] * 0.004, yi, f"{int(tot):,}",
                 va="center", ha="left", fontsize=8, color="#333333")
    axL.set_yticks(y)
    axL.set_yticklabels(names, fontsize=9)
    axL.set_xlabel("Article count (2011–2025)", fontsize=11)
    axL.set_title(f"({letterL}) {domain.capitalize()} — article volume by "
                  f"5-year window", fontsize=12, fontweight="bold", loc="left")
    axL.legend(title="Window", fontsize=8.5, title_fontsize=8.5,
               loc="lower right", frameon=False)
    axL.margins(x=0.14)
    for s in ("top", "right"):
        axL.spines[s].set_visible(False)
    axL.grid(True, axis="x", alpha=0.25)

    s1 = top[f"share_{wnames[0]}"].values
    s3 = top[f"share_{wnames[2]}"].values
    net = s3 - s1
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in net]
    axR.barh(y, net, color=colors, edgecolor="white", linewidth=0.4)
    axR.axvline(0, color="#555555", linewidth=0.8)
    span = max(abs(net.min()), abs(net.max())) or 1.0
    for yi, v, a, b in zip(y, net, s1, s3):
        off = span * 0.03
        axR.text(v + (off if v >= 0 else -off), yi, f"{a:.1f}%→{b:.1f}%",
                 va="center", ha="left" if v >= 0 else "right",
                 fontsize=7.5, color="#222222")
    axR.set_xlabel("Δ corpus share (pp), 2011–2015 → 2021–2025", fontsize=11)
    axR.set_title(f"({letterR}) {domain.capitalize()} — change in corpus "
                  f"share", fontsize=12, fontweight="bold", loc="left")
    axR.set_yticks(y)
    axR.set_yticklabels([])
    axR.margins(x=0.24)
    for s in ("top", "right", "left"):
        axR.spines[s].set_visible(False)
    axR.grid(True, axis="x", alpha=0.25)

    print(f"[{domain}] window totals: " +
          ", ".join(f"{w}={int(win_total[w]):,}" for w in wnames))


def main():
    fig, axes = plt.subplots(2, 2, figsize=(16, 18),
                             gridspec_kw={"width_ratios": [1.25, 1],
                                          "hspace": 0.16, "wspace": 0.04})
    panels("methodology", axes[0][0], axes[0][1], "a", "b")
    panels("health", axes[1][0], axes[1][1], "c", "d")
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT}")
    mirror_to_main(str(OUT))


if __name__ == "__main__":
    main()
