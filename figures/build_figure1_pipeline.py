"""
Figure 1: Framework overview diagram, modeled on the Fang et al. (2026) JBI
pipeline figure per Chunhua Weng's review (2026-08-17):
  - illustrated module boxes with example content instead of dense text
  - chevron stage banners naming each module
  - ownership bands marking which modules are new (this study) vs inherited
  - the three analysis layers (field-level / cross-journal / practical impact)
    shown explicitly, mirroring Methods 3.3-3.5 and Results 4.1-4.3

Drawn at true print width (7.2 in) so fonts are >= 5.4 pt at 1:1 scale on a
6.5-in text column; the previous 21-in canvas scaled fonts down ~3x, which is
what made the old figure illegible.

Casey Ta review round (2026-08-21): source-share percentages removed from
Module 1 (kept in Methods 3.1 text), pills in Module 2 labeled as example
extracted topics, and the last box renamed OUTPUT -> EXAMPLE FINDINGS since
the findings are interpretations, not literal pipeline output.

Output: figures/figure1_pipeline.png (420 dpi)
"""

from __future__ import annotations

import os

import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent)); from _mirror import mirror_to_main
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
NEW_FILL = "#2F5C8A"       # this-study ownership: publication blue
NEW_EDGE = "#1F3F5F"
OLD_FILL = "#8A93A0"       # inherited ownership: neutral slate
OLD_EDGE = "#6A7480"
IO_FILL = "#4A4F57"        # input / findings: dark neutral
IO_EDGE = "#33373D"

METH_C = "#2F6DB5"         # methodology-domain content (Fang: blue)
HLTH_C = "#D97B29"         # health-domain content (Fang: orange)

BOX_BG = "#FFFFFF"
BOX_TXT = "#26292E"
FAINT = "#79808A"
ARROW_C = "#3A3F46"

# ---------------------------------------------------------------------------
# Geometry: canvas in inches (1 data unit = 1 inch at scale 1)
# ---------------------------------------------------------------------------
CW, CH = 7.2, 4.30
M = 0.10
GAP = 0.13
BOX_Y0, BOX_Y1 = 1.30, 4.22
BAND_H = 0.235
CHEV_Y0, CHEV_Y1 = 0.68, 1.14
LEG_Y = 0.28

W_IN, W_M1, W_M2, W_M3, W_OUT = 0.86, 1.24, 1.26, 1.62, 1.50   # sum = 6.48
X_IN = M
X_M1 = X_IN + W_IN + GAP
X_M2 = X_M1 + W_M1 + GAP
X_M3 = X_M2 + W_M2 + GAP
X_OUT = X_M3 + W_M3 + GAP

FS_BAND = 6.5
FS_CHEV = 7.2


def module_box(ax, x, w, band_label, band_fill, band_edge):
    """White content box with a colored ownership band across the top."""
    ax.add_patch(FancyBboxPatch(
        (x, BOX_Y0), w, BOX_Y1 - BOX_Y0,
        boxstyle="round,pad=0,rounding_size=0.045",
        facecolor=BOX_BG, edgecolor=band_edge, linewidth=1.1, zorder=2))
    ax.add_patch(FancyBboxPatch(
        (x, BOX_Y1 - BAND_H), w, BAND_H,
        boxstyle="round,pad=0,rounding_size=0.045",
        facecolor=band_fill, edgecolor=band_fill, linewidth=0, zorder=3))
    ax.add_patch(Rectangle((x, BOX_Y1 - BAND_H), w, BAND_H / 2,
                           facecolor=band_fill, edgecolor="none", zorder=3))
    ax.text(x + w / 2, BOX_Y1 - BAND_H / 2, band_label, ha="center",
            va="center", fontsize=FS_BAND, fontweight="bold", color="white",
            zorder=4)


def chip(ax, cx, cy, w, h, text, edge, face="#FFFFFF", fs=6.2, txt=None,
         lw=0.9, z=4):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.05",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=z))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color=txt or edge, zorder=z + 1, linespacing=1.15)


def chevron(ax, x, w, label, fill):
    tip = 0.10
    y0, y1 = CHEV_Y0, CHEV_Y1
    ym = (y0 + y1) / 2
    pts = [(x, y0), (x + w - tip, y0), (x + w, ym), (x + w - tip, y1),
           (x, y1), (x + tip, ym)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=fill, edgecolor="none",
                         zorder=2))
    ax.text(x + tip / 2 + (w - tip) / 2, ym, label, ha="center", va="center",
            fontsize=FS_CHEV, fontweight="bold", color="white", zorder=3,
            linespacing=1.1)


def flow_arrow(ax, x0, x1):
    y = (BOX_Y0 + BOX_Y1 - BAND_H) / 2
    ax.add_patch(FancyArrowPatch((x0 - 0.005, y), (x1 + 0.005, y),
                                 arrowstyle="-|>", mutation_scale=8,
                                 linewidth=1.2, color=ARROW_C, zorder=5))


def build_figure(out_path: str):
    fig, ax = plt.subplots(figsize=(CW, CH))
    ax.set_xlim(0, CW)
    ax.set_ylim(0, CH)
    ax.set_aspect("equal")
    ax.axis("off")

    # ============ Box 1: input corpus =====================================
    module_box(ax, X_IN, W_IN, "INPUT", IO_FILL, IO_EDGE)
    cx = X_IN + W_IN / 2
    y = BOX_Y1 - BAND_H - 0.22
    ax.text(cx, y, "29 journals", ha="center", va="center", fontsize=7.2,
            fontweight="bold", color=BOX_TXT, zorder=4)
    y -= 0.165
    ax.text(cx, y, "PubMed-indexed", ha="center", va="center", fontsize=5.2,
            color=FAINT, zorder=4)
    y -= 0.24
    for name in ("JAMIA", "JMIR", "Bioinformatics", "npj Digit. Med.", "IJMI"):
        chip(ax, cx, y, W_IN - 0.12, 0.175, name, "#9AA3AE", face="#F4F6F8",
             txt="#3D434B", fs=5.4)
        y -= 0.225
    ax.text(cx, y + 0.015, "+ 24 more", ha="center", va="center",
            fontsize=5.6, color=FAINT, style="italic", zorder=4)
    y -= 0.42
    ax.text(cx, y, "2011–2025", ha="center", va="center", fontsize=7.0,
            fontweight="bold", color=BOX_TXT, zorder=4)

    flow_arrow(ax, X_IN + W_IN, X_M1)

    # ============ Box 2: multi-pass keyword acquisition (new) =============
    module_box(ax, X_M1, W_M1, "THIS STUDY", NEW_FILL, NEW_EDGE)
    x0 = X_M1 + 0.10
    y = BOX_Y1 - BAND_H - 0.28
    for k, label in enumerate(("PubMed API", "publisher pages",
                               "MeSH substitution", "GPT-5-nano")):
        ax.add_patch(plt.Circle((x0 + 0.06, y), 0.062,
                                facecolor=NEW_FILL, edgecolor="none",
                                zorder=4))
        ax.text(x0 + 0.06, y - 0.004, str(k + 1), ha="center", va="center",
                fontsize=6.0, fontweight="bold", color="white", zorder=5)
        ax.text(x0 + 0.165, y, label, ha="left", va="center", fontsize=5.7,
                color=BOX_TXT, zorder=4)
        y -= 0.30
    # percentages intentionally omitted here (Casey Ta review); Methods 3.1
    # carries the per-pass source shares
    y -= 0.02
    ax.text(X_M1 + W_M1 / 2, y, "each pass fills gaps\nthe previous left",
            ha="center", va="top", fontsize=5.2, color=FAINT, style="italic",
            zorder=4, linespacing=1.3)
    chip(ax, X_M1 + W_M1 / 2, BOX_Y0 + 0.26, W_M1 - 0.20, 0.36,
         "78,073 articles\n95,872 keywords", NEW_FILL, face="#EAF0F7",
         fs=6.2, lw=1.0)

    flow_arrow(ax, X_M1 + W_M1, X_M2)

    # ============ Box 3: taxonomy construction (inherited) ================
    module_box(ax, X_M2, W_M2, "FANG ET AL. 2026", OLD_FILL, OLD_EDGE)
    cx = X_M2 + W_M2 / 2
    rng = np.random.default_rng(7)
    ex, ey = X_M2 + 0.33, BOX_Y1 - BAND_H - 0.44
    pts = rng.normal(0, 1, (11, 2)) * [0.085, 0.075] + [ex, ey]
    ax.scatter(pts[:, 0], pts[:, 1], s=4, c=METH_C, zorder=5, lw=0)
    ax.add_patch(Ellipse((ex, ey), 0.42, 0.40, facecolor="none",
                         edgecolor=METH_C, lw=0.8, ls=(0, (3, 2)), zorder=4))
    ex2 = X_M2 + W_M2 - 0.33
    pts = rng.normal(0, 1, (9, 2)) * [0.08, 0.07] + [ex2, ey]
    ax.scatter(pts[:, 0], pts[:, 1], s=4, c=HLTH_C, zorder=5, lw=0)
    ax.add_patch(Ellipse((ex2, ey), 0.40, 0.38, facecolor="none",
                         edgecolor=HLTH_C, lw=0.8, ls=(0, (3, 2)), zorder=4))
    ax.text(cx, BOX_Y1 - BAND_H - 0.76, "embed · cluster · name",
            ha="center", va="center", fontsize=5.6, color=FAINT, zorder=4)
    ax.text(cx, BOX_Y1 - BAND_H - 0.89, "BiomedBERT · K-means · GPT-5-nano",
            ha="center", va="center", fontsize=3.6, color=FAINT, zorder=4)
    # label the pills as outputs, not methods (Casey Ta review comment)
    ax.text(cx, BOX_Y1 - BAND_H - 1.02, "example extracted topics",
            ha="center", va="center", fontsize=4.8, color=FAINT,
            style="italic", zorder=4)
    y = BOX_Y1 - BAND_H - 1.18
    for label, c in (("Transformer models", METH_C),
                     ("Knowledge graphs", METH_C),
                     ("Drug discovery", HLTH_C)):
        chip(ax, cx, y, W_M2 - 0.10, 0.19, label, c, fs=5.2)
        y -= 0.235
    chip(ax, cx, BOX_Y0 + 0.24, W_M2 - 0.18, 0.33,
         "70 methodology\n+ 86 health topics", OLD_EDGE, face="#F1F3F5",
         fs=5.9, txt="#3D434B", lw=1.0)

    flow_arrow(ax, X_M2 + W_M2, X_M3)

    # ============ Box 4: three-layer analysis (new) =======================
    module_box(ax, X_M3, W_M3, "THIS STUDY", NEW_FILL, NEW_EDGE)
    rows = [("Field-level trends", "counts · shares · waves\ntopic co-occurrence"),
            ("Cross-journal\necosystem", "specialization · network\ncommunities · bridges"),
            ("Practical impact", "authorship · pioneers\njournal fit · event shifts")]
    icon_w = 0.30
    row_h = (BOX_Y1 - BAND_H - BOX_Y0 - 0.16) / 3
    for k, (title, sub) in enumerate(rows):
        ry1 = BOX_Y1 - BAND_H - 0.07 - k * row_h
        ry0 = ry1 - row_h + 0.05
        rcy = (ry0 + ry1) / 2
        ax.add_patch(FancyBboxPatch(
            (X_M3 + 0.07, ry0), W_M3 - 0.14, ry1 - ry0,
            boxstyle="round,pad=0,rounding_size=0.035",
            facecolor="#F3F6FA", edgecolor="#C7D2DE", lw=0.7, zorder=3))
        tx = X_M3 + 0.12 + icon_w
        if "\n" in title:
            ax.text(tx, ry1 - 0.085, title, ha="left", va="top", fontsize=5.9,
                    fontweight="bold", color=NEW_FILL, zorder=5,
                    linespacing=1.15)
            sub_y = rcy - 0.115
        else:
            ax.text(tx, rcy + 0.135, title, ha="left", va="center",
                    fontsize=5.9, fontweight="bold", color=NEW_FILL, zorder=5)
            sub_y = rcy - 0.02 if "\n" not in sub else rcy - 0.075
        ax.text(tx, sub_y, sub, ha="left", va="top" if "\n" in sub else "center",
                fontsize=5.2, color="#4A5058", zorder=5, linespacing=1.3)
        ix0, ix1 = X_M3 + 0.13, X_M3 + 0.11 + icon_w
        if k == 0:      # trend sparkline
            xs = np.linspace(ix0, ix1, 12)
            ys = rcy - 0.15 + 0.28 * (np.linspace(0, 1, 12) ** 1.7)
            ax.plot(xs, ys, color=METH_C, lw=1.1, zorder=5)
            ax.fill_between(xs, rcy - 0.16, ys, color=METH_C, alpha=0.14,
                            zorder=4)
        elif k == 1:    # mini journal network with bridge node
            nodes = [(ix0 + 0.03, rcy - 0.08), (ix0 + 0.01, rcy + 0.09),
                     (ix0 + 0.13, rcy + 0.13), ((ix0 + ix1) / 2, rcy),
                     (ix1 - 0.10, rcy + 0.10), (ix1 - 0.02, rcy - 0.05),
                     (ix1 - 0.12, rcy - 0.12)]
            edges = [(0, 1), (1, 2), (0, 3), (1, 3), (2, 3), (3, 4), (3, 5),
                     (4, 5), (5, 6), (3, 6)]
            for a, b in edges:
                ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]],
                        color="#AEB8C4", lw=0.7, zorder=4)
            for i, (nx_, ny_) in enumerate(nodes):
                big = i == 3
                ax.scatter([nx_], [ny_], s=22 if big else 10,
                           c=NEW_FILL if big else (METH_C if i < 3 else HLTH_C),
                           zorder=5, lw=0)
        else:           # mini journal-by-topic heat map
            vals = rng.random((3, 6))
            cw_, chh = (ix1 - ix0) / 6, 0.28 / 3
            for r in range(3):
                for c in range(6):
                    ax.add_patch(Rectangle(
                        (ix0 + c * cw_, rcy - 0.14 + r * chh), cw_ - 0.008,
                        chh - 0.008,
                        facecolor=plt.cm.Blues(0.15 + 0.75 * vals[r, c]),
                        edgecolor="none", zorder=5))

    flow_arrow(ax, X_M3 + W_M3, X_OUT)

    # ============ Box 5: findings =========================================
    # "EXAMPLE FINDINGS", not "OUTPUT": these are interpreted results, not
    # literal pipeline output (Casey Ta review comment)
    module_box(ax, X_OUT, W_OUT, "EXAMPLE FINDINGS", IO_FILL, IO_EDGE)
    cx = X_OUT + W_OUT / 2
    cards = [
        ("Three waves of growth", 0.62,
         "genomics → machine learning\n→ generative AI"),
        ("Only 56 of 156 topics", 0.62,
         "gained or lost share after\ncontrolling for growth"),
        ("Four journal communities", 0.62,
         "weakly separated,\nbridged by JBI"),
        ("COVID-19 started trends", 0.48,
         "that outlasted the pandemic"),
    ]
    y1 = BOX_Y1 - BAND_H - 0.07
    for title, h, sub in cards:
        y0 = y1 - h
        ax.add_patch(FancyBboxPatch(
            (X_OUT + 0.045, y0), W_OUT - 0.09, h,
            boxstyle="round,pad=0,rounding_size=0.035",
            facecolor="#F5F6F8", edgecolor="#CDD3DA", lw=0.7, zorder=3))
        ty = y0 + h - 0.145
        ax.text(cx, ty, title, ha="center", va="center", fontsize=5.3,
                fontweight="bold", color=BOX_TXT, zorder=5)
        multi = "\n" in sub
        ax.text(cx, ty - 0.13 if multi else ty - 0.175, sub, ha="center",
                va="top" if multi else "center", fontsize=5.0,
                color="#4A5058", zorder=5, linespacing=1.35)
        y1 = y0 - 0.07

    # ============ chevron banner ==========================================
    chevron(ax, X_IN, W_IN, "Input\ncorpus", IO_FILL)
    chevron(ax, X_M1, W_M1, "Keyword\nacquisition", NEW_FILL)
    chevron(ax, X_M2, W_M2, "Taxonomy\nconstruction", OLD_FILL)
    chevron(ax, X_M3, W_M3, "Three-layer\nanalysis", NEW_FILL)
    chevron(ax, X_OUT, W_OUT, "Example\nfindings", IO_FILL)

    # ============ legend ==================================================
    lx = CW / 2 - 1.95
    ax.add_patch(Rectangle((lx, LEG_Y - 0.05), 0.15, 0.10,
                           facecolor=NEW_FILL, edgecolor="none", zorder=3))
    ax.text(lx + 0.21, LEG_Y, "developed in this study", ha="left",
            va="center", fontsize=6.6, color=BOX_TXT, zorder=3)
    lx2 = lx + 1.90
    ax.add_patch(Rectangle((lx2, LEG_Y - 0.05), 0.15, 0.10,
                           facecolor=OLD_FILL, edgecolor="none", zorder=3))
    ax.text(lx2 + 0.21, LEG_Y, "inherited from Fang et al. (2026)",
            ha="left", va="center", fontsize=6.6, color=BOX_TXT, zorder=3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=420, bbox_inches="tight", facecolor="white",
                pad_inches=0.04)
    mirror_to_main(out_path)
    plt.close(fig)


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "figure1_pipeline.png")
    build_figure(out)
    print(f"Wrote {out}")
