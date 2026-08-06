import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

CONFLICT = Path("/Users/danielliang/Library/CloudStorage/Dropbox/multi_journal_trend_analysis (Yilun Liang's conflicted copy 2026-04-27)")
MAIN     = Path("/Users/danielliang/Library/CloudStorage/Dropbox/multi_journal_trend_analysis")
DATA = CONFLICT / "data" / "visualizations_k100"
OUTDIR = CONFLICT / "figures" / "manuscript_v5"
OUTDIR.mkdir(parents=True, exist_ok=True)

WINDOWS = [("2011–2015", range(2011, 2016)),
           ("2016–2020", range(2016, 2021)),
           ("2021–2025", range(2021, 2026))]
WCOLORS = ["#add8e6", "#6495ed", "#1f3a93"]   # light -> dark blue
TOP_N = 20

def short(s, n=38): return s if len(s) <= n else s[:n-1] + "…"

def build(domain, fig_label, out_name):
    df = pd.read_csv(DATA / f"{domain}_topic_year_counts.csv")
    ycols = [c for c in df.columns if str(c).strip().isdigit()]
    # per-window counts per topic
    for wname, yrs in WINDOWS:
        cols = [c for c in ycols if int(c) in yrs]
        df[wname] = df[cols].sum(axis=1)
    wnames = [w[0] for w in WINDOWS]
    # domain total per window (sum across all topics) -> share denominator (matches manuscript share defn)
    win_total = {w: df[w].sum() for w in wnames}
    for w in wnames:
        df[f"share_{w}"] = df[w] / win_total[w] * 100.0
    # top-N by overall total; plot ascending so largest sits at top of barh
    top = df.sort_values("total", ascending=False).head(TOP_N).copy()
    top = top.sort_values("total", ascending=True).reset_index(drop=True)
    y = np.arange(len(top))
    names = [short(t) for t in top["topic"]]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 9.5), sharey=True,
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # ---- panel (a): article count by 5-year window (stacked) ----
    left = np.zeros(len(top))
    for w, c in zip(wnames, WCOLORS):
        axL.barh(y, top[w], left=left, color=c, label=w, edgecolor="white", linewidth=0.4)
        left += top[w].values
    for yi, tot in zip(y, top["total"]):
        axL.text(tot + win_total[wnames[0]] * 0.004, yi, f"{int(tot):,}",
                 va="center", ha="left", fontsize=8, color="#333333")
    axL.set_yticks(y); axL.set_yticklabels(names, fontsize=9)
    axL.set_xlabel("Article count", fontsize=12)
    axL.set_title("(a) Article volume by 5-year window", fontsize=13, fontweight="bold")
    axL.legend(title="Window", fontsize=9, title_fontsize=9, loc="lower right", frameon=False)
    axL.margins(x=0.12)
    for s in ("top", "right"): axL.spines[s].set_visible(False)
    axL.grid(True, axis="x", alpha=0.25)

    # ---- panel (b): rate of increase = change in corpus share (W1 -> W3) ----
    s1 = top[f"share_{wnames[0]}"].values
    s3 = top[f"share_{wnames[2]}"].values
    net = s3 - s1
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in net]
    axR.barh(y, net, color=colors, edgecolor="white", linewidth=0.4)
    axR.axvline(0, color="#555555", linewidth=0.8)
    span = max(abs(net.min()), abs(net.max())) or 1.0
    for yi, v, a, b in zip(y, net, s1, s3):
        off = span * 0.03
        ha = "left" if v >= 0 else "right"
        axR.text(v + (off if v >= 0 else -off), yi, f"{a:.1f}%→{b:.1f}%",
                 va="center", ha=ha, fontsize=7.5, color="#222222")
    axR.set_xlabel("Change in corpus share (percentage points), 2011–2015 → 2021–2025", fontsize=11)
    axR.set_title("(b) Rate of increase (Δ corpus share)", fontsize=13, fontweight="bold")
    axR.margins(x=0.22)
    for s in ("top", "right", "left"): axR.spines[s].set_visible(False)
    axR.grid(True, axis="x", alpha=0.25)

    fig.suptitle(f"{domain.capitalize()} topics across three 5-year windows "
                 f"(top {TOP_N} by volume)", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUTDIR / out_name
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    # mirror to MAIN
    mirror = MAIN / "figures" / "manuscript_v5" / out_name
    if mirror.parent.exists():
        shutil.copy2(out, mirror)
    # quick console summary
    print(f"{fig_label} [{domain}] -> {out.name}")
    print(f"   window totals: " + ", ".join(f"{w}={int(win_total[w]):,}" for w in wnames))
    risers = top.sort_values(f"share_{wnames[2]}", ascending=False)
    big = (top.assign(net=net).sort_values("net", ascending=False)).head(3)["topic"].tolist()
    drop = (top.assign(net=net).sort_values("net", ascending=True)).head(3)["topic"].tolist()
    print(f"   biggest share gainers: {big}")
    print(f"   biggest share losers : {drop}")
    return out

o1 = build("methodology", "Figure 6", "figure08_methodology_count_and_rate.png")
o2 = build("health",      "Figure 7", "figure09_health_count_and_rate.png")
print("\nSaved:", o1.name, "and", o2.name, "to", OUTDIR)
