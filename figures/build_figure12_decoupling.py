"""Figure 12 — Decoupling case studies: K=100 parent vs fine-K child trajectories.

Four panels, each showing a topic where the K=100 umbrella is rising
Holm-significantly while a fine-grained child (K=750 / K=1,050 / K=1,100) is
plateauing or declining beneath it.

Panels:
  A. HCI (K=750)            ↘ Software & usability research (K=100)
  B. SVMs (K=1,050 cl.45)   ↘ Machine learning methods       (K=100)
  C. HIS  (K=1,050 cl.22)   ↘ Clinical decision support      (K=100)
  D. Interoperability (K=1,100 cl.477) ↘ Patient-provider communication (K=100)

Each panel plots both lines on a shared time axis with a SECOND y-axis for
the child so absolute counts of both lines are visible without one swamping
the other.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MAIN_MIRROR = Path("/Users/danielliang/Library/CloudStorage/Dropbox/multi_journal_trend_analysis")
OUT = ROOT / "figures" / "figure12_decoupling_case_studies.png"

YEARS = list(range(2011, 2026))
YEAR_COLS = [str(y) for y in YEARS]

PARENT_COLOUR = "#1f4ea0"
CHILD_COLOUR = "#c12828"


def load_topic_year(domain, suff, lookup_col, lookup_val):
    df = pd.read_csv(ROOT / f"data/visualizations_{suff}/{domain}_topic_year_counts.csv")
    cols = [c for c in YEAR_COLS if c in df.columns]
    row = df[df[lookup_col] == lookup_val]
    if len(row) == 0:
        raise KeyError(f"{lookup_val} not found in {domain}/{suff}")
    return row.iloc[0][cols].values.astype(float)


# ---------------- Case-study definitions ----------------
# Annotation strings are derived from the plotted arrays; hand-written values
# drifted out of sync with the data (panel C previously claimed a 247/yr peak
# against a series that peaks at 104/yr).


def parent_stats(domain, topic):
    t = pd.read_csv(ROOT / f"data/visualizations_k100/{domain}_trend_analysis.csv")
    return t[t.topic == topic].iloc[0]


def sup(x):
    return str(x).translate(str.maketrans("-0123456789", "\u207b\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079"))


def fmt_p(v):
    m, e = f"{v:.1e}".split("e")
    return f"p_holm={m}\u00d710{sup(int(e))} \u2605"


def parent_meta(domain, topic, counts):
    r = parent_stats(domain, topic)
    return (f"n={int(counts.sum()):,} (2011\u20132025) \u00b7 {r.slope:+.1f}/yr \u00b7 "
            f"R\u00b2={r.r_squared:.2f} \u00b7 {fmt_p(r.p_value_holm)}")


def child_meta(counts, kind="drop"):
    yrs = np.array(YEARS)
    pk = int(np.argmax(counts))
    n = int(counts.sum())
    if kind == "flat":
        return (f"n={n:,} \u00b7 peak {yrs[pk]} ({counts[pk]:.0f}/yr) \u2192 "
                f"{counts[-1]:.0f}/yr in 2025 \u00b7 no sustained trend")
    if kind == "v":
        tr = int(np.argmin(counts))
        return (f"n={n:,} \u00b7 peak {yrs[pk]} ({counts[pk]:.0f}/yr) \u2192 trough "
                f"{yrs[tr]} ({counts[tr]:.0f}) \u2192 {counts[-1]:.0f} in 2025")
    return (f"n={n:,} \u00b7 peak {yrs[pk]} ({counts[pk]:.0f}/yr) \u2192 "
            f"{counts[-1]:.0f}/yr in 2025 \u00b7 {100 * (1 - counts[-1] / counts[pk]):.0f}% decline")


_A_c = load_topic_year("methodology", "k750", "topic", "Human-computer interaction")
_A_p = load_topic_year("methodology", "k100", "topic", "Software and usability research")
_B_c = load_topic_year("methodology", "k1050_1100", "cluster_id", 45)
_B_p = load_topic_year("methodology", "k100", "topic", "Machine learning methods")
_C_c = load_topic_year("methodology", "k1050_1100", "cluster_id", 22)
_C_p = load_topic_year("methodology", "k100", "topic", "Clinical decision support")
_D_c = load_topic_year("health", "k1050_1100", "cluster_id", 477)
_D_p = load_topic_year("health", "k100", "topic", "Digital health technologies")

CASES = [
    {
        "panel": "A",
        "mechanism": "Plateau under explosive growth",
        "child_label": "Human\u2013computer interaction (K=750)",
        "parent_label": "Software & usability research (K=100)",
        "child_counts": _A_c,
        "parent_counts": _A_p,
        "child_meta": child_meta(_A_c, "flat"),
        "parent_meta": parent_meta("methodology", "Software and usability research", _A_p),
    },
    {
        "panel": "B",
        "mechanism": "Algorithm displacement",
        "child_label": "Support vector machines (K=1,050 cl. 45)",
        "parent_label": "Machine learning methods (K=100)",
        "child_counts": _B_c,
        "parent_counts": _B_p,
        "child_meta": child_meta(_B_c),
        "parent_meta": parent_meta("methodology", "Machine learning methods", _B_p),
    },
    {
        "panel": "C",
        "mechanism": "Paradigm replacement",
        "child_label": "Hospital information systems (K=1,050 cl. 22)",
        "parent_label": "Clinical decision support (K=100)",
        "child_counts": _C_c,
        "parent_counts": _C_p,
        "child_meta": child_meta(_C_c),
        "parent_meta": parent_meta("methodology", "Clinical decision support", _C_p),
    },
    {
        "panel": "D",
        "mechanism": "Infrastructure assumption (V-shape)",
        "child_label": "Interoperability research (K=1,100 cl. 477)",
        "parent_label": "Digital health technologies (K=100)",
        "child_counts": _D_c,
        "parent_counts": _D_p,
        "child_meta": child_meta(_D_c, "v"),
        "parent_meta": parent_meta("health", "Digital health technologies", _D_p),
    },
]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    x = np.array(YEARS)

    for ax, case in zip(axes, CASES):
        # Parent on left axis
        parent = case["parent_counts"]
        child = case["child_counts"]

        ax.plot(x, parent, "o-", color=PARENT_COLOUR, lw=2.4, markersize=5,
                label=f"PARENT  {case['parent_label']}", zorder=3)
        ax.fill_between(x, parent, color=PARENT_COLOUR, alpha=0.10)
        ax.set_ylim(0, parent.max() * 1.15)
        ax.set_ylabel("Parent: articles/yr",
                      color=PARENT_COLOUR, fontsize=10.5, fontweight="bold")
        ax.tick_params(axis="y", colors=PARENT_COLOUR, labelsize=9)
        ax.tick_params(axis="x", labelsize=9)
        ax.set_xticks(x[::2])

        # Child on right axis
        ax2 = ax.twinx()
        ax2.plot(x, child, "s--", color=CHILD_COLOUR, lw=2.0, markersize=5,
                 label=f"CHILD  {case['child_label']}", zorder=3)
        ax2.fill_between(x, child, color=CHILD_COLOUR, alpha=0.10)
        ax2.set_ylim(0, max(child.max() * 1.30, 5))
        ax2.set_ylabel("Child: articles/yr",
                       color=CHILD_COLOUR, fontsize=10.5, fontweight="bold")
        ax2.tick_params(axis="y", colors=CHILD_COLOUR, labelsize=9)

        # Peak markers
        p_peak = int(np.argmax(parent))
        c_peak = int(np.argmax(child))
        ax.plot(x[p_peak], parent[p_peak], "o", markersize=11,
                markerfacecolor="white", markeredgecolor=PARENT_COLOUR,
                markeredgewidth=2, zorder=4)
        ax2.plot(x[c_peak], child[c_peak], "s", markersize=11,
                 markerfacecolor="white", markeredgecolor=CHILD_COLOUR,
                 markeredgewidth=2, zorder=4)

        # Panel title with mechanism
        ax.set_title(f"({case['panel']})  {case['mechanism']}",
                     fontsize=12, fontweight="bold", loc="left", pad=8)

        # Sub-annotation box: child & parent meta lines
        meta_text = f"PARENT  {case['parent_meta']}\nCHILD     {case['child_meta']}"
        ax.text(0.02, 0.98, meta_text, transform=ax.transAxes,
                fontsize=8.5, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec="#cccccc", lw=0.7, alpha=0.92))

        # Legend at the bottom of each panel
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="lower center", fontsize=8.5,
                  frameon=True, framealpha=0.92, ncol=1,
                  bbox_to_anchor=(0.5, -0.02))

        for s in ("top",):
            ax.spines[s].set_visible(False)
            ax2.spines[s].set_visible(False)
        ax.grid(axis="y", alpha=0.18, color=PARENT_COLOUR, lw=0.4)

    # in-image figure title removed (caption carries it)
    fig.supxlabel("Year", fontsize=11, y=0.02)
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.94])
    fig.savefig(OUT, dpi=400, facecolor="white")
    plt.close(fig)
    print(f"  wrote {OUT}")

    # Mirror to MAIN folder
    if MAIN_MIRROR.exists():
        import shutil
        mirror = MAIN_MIRROR / "figures" / OUT.name
        shutil.copy2(OUT, mirror)
        print(f"  mirrored to {mirror}")


if __name__ == "__main__":
    main()
