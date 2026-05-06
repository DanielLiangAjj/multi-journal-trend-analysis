"""Build Figure 2 taxonomy network visualizations for K=100 hierarchies.

Outputs:
- figures/figure2_methodology_taxonomy.png
- figures/figure2_health_taxonomy.png

Design notes
------------
- ONE title per figure ("Figure 2. <Domain> topic taxonomy (K=100)").
- Sparse-DAG honesty annotation under the title.
- Largest weakly-connected component drawn at the center / largest scale;
  remaining non-trivial components tiled to the right; isolated singletons
  listed below in a faded "isolates" sub-region (two columns of names).
- Top-10 topics by total article count are labelled in the main panel,
  using textwrap (no ellipses).
- Three callout sub-trees on the right column, each with a bold sans-serif
  panel letter (a), (b), (c) in the upper-left.
- Callout hub nodes are highlighted in orange in BOTH the main panel and
  the callout, with a thin orange leader line linking them.
- Small legend in the main panel explaining colors.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.lines import Line2D

ROOT = Path("/Users/danielliang/Library/CloudStorage/Dropbox/multi_journal_trend_analysis")
HIER_DIR = ROOT / "data" / "hierarchy_k100"
VIS_DIR = ROOT / "data" / "visualizations_k100"
OUT_DIR = ROOT / "figures"


# Colors
COLOR_NODE_DEFAULT = "#9ecae1"   # blue
COLOR_NODE_DEFAULT_EDGE = "#3c6e91"
COLOR_HUB = "#f4a261"            # orange
COLOR_HUB_EDGE = "#a85a13"
COLOR_SINGLETON = "#cfd8dc"      # faint grey
COLOR_SINGLETON_EDGE = "#8c9ba3"
COLOR_LEADER = "#a85a13"         # orange leader from main panel to callout border


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_domain(domain: str) -> Tuple[nx.DiGraph, Dict[str, int], List[Tuple[str, str]]]:
    """Load edges + topics + sizes; build parent->child DiGraph."""
    with open(HIER_DIR / f"{domain}_edges.json") as f:
        edges_raw = json.load(f)  # [[child, parent], ...]
    with open(HIER_DIR / f"{domain}_final_topics.json") as f:
        topics = json.load(f)
    trend = pd.read_csv(VIS_DIR / f"{domain}_trend_analysis.csv")
    sizes = dict(zip(trend["topic"], trend["total"]))

    for t in topics:
        sizes.setdefault(t, 0)

    G = nx.DiGraph()
    for t in topics:
        G.add_node(t, size=int(sizes.get(t, 0)))

    parent_child_edges: List[Tuple[str, str]] = []
    for child, parent in edges_raw:
        if parent not in G:
            G.add_node(parent, size=int(sizes.get(parent, 0)))
        if child not in G:
            G.add_node(child, size=int(sizes.get(child, 0)))
        G.add_edge(parent, child)
        parent_child_edges.append((parent, child))

    return G, sizes, parent_child_edges


# ---------------------------------------------------------------------------
# Hub selection (manual per-domain to control overlap)
# ---------------------------------------------------------------------------

def select_hubs(G: nx.DiGraph, sizes: Dict[str, int], domain: str) -> List[str]:
    """Pick three callout hubs per domain.

    For methodology, only 5 internal nodes have any children (out-degree>0):
      Clinical study designs (out=4)
      Statistical methods   (out=2, child of Clinical study designs)
      Information dissemination (out=2)
      Machine learning methods (out=1, child of Statistical methods)
      Transformer models     (out=1)
    The structure is a DAG (Large language models has TWO parents:
    Machine learning methods AND Transformer models) so any depth>=2
    callout containing 'Machine learning methods' will share 'Large
    language models' with a callout rooted at Transformer models.
    We pick three hubs that span the available structure and explicitly
    disclose the DAG nature in the figure caption / annotation.

    For health, the largest WCC has 30 nodes with several broad hubs.
    We pick the three highest-out-degree hubs.
    """
    if domain == "methodology":
        return [
            "Clinical study designs",     # out=4 (depth-2 reaches LLMs via stat methods)
            "Information dissemination",  # out=2 (separate WCC)
            "Transformer models",         # out=1 (alt path into LLMs; shows DAG)
        ]
    elif domain == "health":
        return [
            "Mixed clinical conditions and demographics",  # out=8
            "Determinants of health",                      # out=6
            "Translational and clinical research",         # out=3
        ]
    else:
        raise ValueError(domain)


def descendants_to_depth(G: nx.DiGraph, root: str, depth: int = 2) -> List[str]:
    """Return root + descendants up to `depth` levels (BFS)."""
    visited = {root: 0}
    frontier = [root]
    while frontier:
        nxt = []
        for n in frontier:
            d = visited[n]
            if d >= depth:
                continue
            for c in G.successors(n):
                if c not in visited:
                    visited[c] = d + 1
                    nxt.append(c)
        frontier = nxt
    return list(visited.keys())


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _graphviz_layout_safe(G: nx.DiGraph, prog: str = "dot"):
    """Run graphviz layout with sanitized integer node IDs."""
    name_to_id = {n: f"n{i}" for i, n in enumerate(G.nodes)}
    id_to_name = {v: k for k, v in name_to_id.items()}
    H = nx.relabel_nodes(G, name_to_id, copy=True)

    try:
        from networkx.drawing.nx_agraph import graphviz_layout as gv_layout_agraph
        raw = gv_layout_agraph(H, prog=prog)
        return {id_to_name[k]: v for k, v in raw.items()}
    except Exception:
        pass

    try:
        from networkx.drawing.nx_pydot import graphviz_layout as gv_layout_pydot
        raw = gv_layout_pydot(H, prog=prog)
        return {id_to_name[k]: v for k, v in raw.items()}
    except Exception:
        pass

    return None


def compute_main_layout(G: nx.DiGraph) -> Tuple[Dict[str, Tuple[float, float]], List[set]]:
    """Lay out the main background panel.

    Strategy:
      - Identify weakly-connected components.
      - Largest non-trivial component placed BIG and CENTER in the upper
        portion of the canvas (occupies ~70% width, ~75% height).
      - Other non-trivial (size>=2) components tiled in the right-hand area,
        smaller scale.
      - Singleton components are NOT placed in the network area; they get
        rendered as a textual "isolates" block by the caller, so we return
        them separately. The caller will render the isolates list below.

    Returns:
      pos: dict node -> (x, y). Only includes non-singleton nodes.
      singleton_components: list of single-node sets.
    """
    UG = G.to_undirected()
    components = sorted(nx.connected_components(UG), key=lambda c: -len(c))

    non_singleton = [c for c in components if len(c) >= 2]
    singletons = [c for c in components if len(c) == 1]

    pos: Dict[str, Tuple[float, float]] = {}
    if not non_singleton:
        return pos, singletons

    # Layout each non-singleton component with dot
    laid = []
    for comp in non_singleton:
        sub = G.subgraph(comp).copy()
        sub_pos = _graphviz_layout_safe(sub, prog="dot")
        if sub_pos is None:
            sub_pos = nx.kamada_kawai_layout(sub)
            sub_pos = {k: (v[0] * 100, v[1] * 100) for k, v in sub_pos.items()}
        laid.append(sub_pos)

    def _bbox(lay):
        xs = [p[0] for p in lay.values()]
        ys = [p[1] for p in lay.values()]
        return min(xs), min(ys), max(xs), max(ys)

    # Normalize each so it starts at (0,0); track bbox sizes
    norm = []
    sizes_xy = []
    for lay in laid:
        xmin, ymin, xmax, ymax = _bbox(lay)
        n = {k: (v[0] - xmin, v[1] - ymin) for k, v in lay.items()}
        norm.append(n)
        sizes_xy.append((xmax - xmin, ymax - ymin))

    # ---- Main component takes a CENTERED, LARGE region ----
    main_lay = norm[0]
    main_w, main_h = sizes_xy[0]

    # Define a target canvas in arbitrary units. We'll use [0, 1000] x [0, 800].
    # Main component occupies roughly the central 60% of width, top 70% of height.
    target_main_w = 600.0
    target_main_h = 560.0
    sx = target_main_w / max(main_w, 1.0)
    sy = target_main_h / max(main_h, 1.0)
    s = min(sx, sy)
    scaled_main_w = main_w * s
    scaled_main_h = main_h * s
    # Place main component centered horizontally, near top of canvas
    main_x0 = (1000.0 - scaled_main_w) / 2.0
    main_y0 = 800.0 - scaled_main_h - 20.0  # 20 px padding from top
    for k, (x, y) in main_lay.items():
        pos[k] = (main_x0 + x * s, main_y0 + y * s)

    # ---- Other non-singleton components tiled below the main one ----
    rest = list(zip(norm[1:], sizes_xy[1:]))
    if rest:
        # Arrange them in a row below the main component
        # Determine cell size -- use max bbox among rest
        max_w = max(w for _, (w, _) in rest)
        max_h = max(h for _, (_, h) in rest)
        # Scale them so they fit in a band of height ~120 px
        cell_h_target = 110.0
        cell_w_target = 180.0
        s_rest = min(cell_w_target / max(max_w, 1.0), cell_h_target / max(max_h, 1.0))

        # Place in a row spanning the canvas width below main
        n_rest = len(rest)
        gap = 30.0
        total_w = n_rest * cell_w_target + (n_rest - 1) * gap
        x_cursor = max((1000.0 - total_w) / 2.0, 20.0)
        y_band_top = main_y0 - 40.0  # below the main component
        for lay, (w, h) in rest:
            scaled_w = w * s_rest
            scaled_h = h * s_rest
            # Place each component within its cell, top-aligned
            cell_x = x_cursor + (cell_w_target - scaled_w) / 2.0
            cell_y = y_band_top - scaled_h
            for k, (x, y) in lay.items():
                pos[k] = (cell_x + x * s_rest, cell_y + y * s_rest)
            x_cursor += cell_w_target + gap

    return pos, singletons


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _scale_node_sizes(values: List[float], lo: float, hi: float) -> List[float]:
    if not values:
        return []
    arr = np.array(values, dtype=float)
    arr = np.sqrt(np.maximum(arr, 1.0))
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return [(lo + hi) / 2.0 for _ in values]
    norm = (arr - mn) / (mx - mn)
    return (lo + norm * (hi - lo)).tolist()


def _wrap_label(s: str, width: int = 22) -> str:
    """Word-wrap topic name to multiple lines (no ellipsis truncation)."""
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False)) or s


def draw_main_panel(
    ax,
    G: nx.DiGraph,
    pos: Dict[str, Tuple[float, float]],
    singletons: List[set],
    sizes: Dict[str, int],
    hubs: List[str],
    panel_letters: List[str],
    top_n_labels: int = 10,
):
    """Draw the main background panel.

    Nodes that participate in any edge are drawn at full visibility; hub
    nodes (callout roots) are colored orange; the rest are blue. Singletons
    are listed below as a textual block (handled separately by caller via
    `draw_isolates_band`).
    """
    laid_nodes = list(pos.keys())
    raw_sizes = [max(sizes.get(n, 0), 0) for n in laid_nodes]
    node_sizes = _scale_node_sizes(raw_sizes, lo=80, hi=900)

    # Edges
    edges_to_draw = [(u, v) for u, v in G.edges if u in pos and v in pos]
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges_to_draw,
        ax=ax,
        edge_color="#5a7d99",
        alpha=0.85,
        width=1.4,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=10,
        connectionstyle="arc3,rad=0.04",
    )

    hub_set = set(hubs)
    # Non-hub connected nodes
    non_hub_nodes = [n for n in laid_nodes if n not in hub_set]
    non_hub_sizes = [s for n, s in zip(laid_nodes, node_sizes) if n not in hub_set]
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=non_hub_nodes,
        node_size=non_hub_sizes,
        node_color=COLOR_NODE_DEFAULT,
        edgecolors=COLOR_NODE_DEFAULT_EDGE,
        linewidths=0.8,
        ax=ax,
    )
    # Hub nodes (highlighted)
    hub_nodes_in_pos = [n for n in laid_nodes if n in hub_set]
    hub_sizes_in_pos = [s for n, s in zip(laid_nodes, node_sizes) if n in hub_set]
    if hub_nodes_in_pos:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=hub_nodes_in_pos,
            node_size=hub_sizes_in_pos,
            node_color=COLOR_HUB,
            edgecolors=COLOR_HUB_EDGE,
            linewidths=1.6,
            ax=ax,
        )

    # ---- Top-N largest topic labels + hub labels (word-wrapped, no ellipsis) ----
    nodes_to_label = list({*sorted(laid_nodes, key=lambda n: -sizes.get(n, 0))[:top_n_labels], *(h for h in hubs if h in pos)})

    # Determine canvas y-extent so we can decide which labels go above vs below.
    all_ys = [p[1] for p in pos.values()]
    y_min = min(all_ys)
    y_max = max(all_ys)
    y_mid = (y_min + y_max) / 2.0

    # Pre-compute label entries: place above if node is in upper half of
    # network area, otherwise below. This prevents the upper row of root
    # nodes from having labels collide with the title-area annotation.
    label_entries = []
    label_offset = 28.0
    for n in nodes_to_label:
        x, y = pos[n]
        is_hub = n in hub_set
        wrapped = _wrap_label(n, width=18)
        n_lines = wrapped.count("\n") + 1
        label_h = 8.0 + 9.5 * n_lines
        place_above = (y >= y_mid)
        if place_above:
            y_anchor = y + label_offset + label_h  # top of label box
        else:
            y_anchor = y - label_offset            # top of label box
        label_entries.append({
            "node": n,
            "x": x,
            "y_node": y,
            "y": y_anchor,
            "h": label_h,
            "place_above": place_above,
            "is_hub": is_hub,
            "wrapped": wrapped,
        })

    # Simple anti-collision: iterate sorted by x then y desc; for each entry,
    # check if any previously placed label within x-distance 130 overlaps.
    # If so, push current label vertically away (further up if place_above
    # else further down).
    label_entries.sort(key=lambda e: (e["x"], -e["y"]))
    placed: List[dict] = []
    for e in label_entries:
        for p in placed:
            if abs(p["x"] - e["x"]) >= 125:
                continue
            # Compute overlap between rectangles (using y as top-of-box)
            e_top, e_bot = e["y"], e["y"] - e["h"]
            p_top, p_bot = p["y"], p["y"] - p["h"]
            if e_bot < p_top + 2 and e_top > p_bot - 2:
                # overlap; shift e
                if e["place_above"]:
                    # push e further up
                    new_top = p_top + p["h"] + 6.0  # actually means push so e sits above p
                    # Actually simpler: place e_bot = p_top + 4
                    e["y"] = p_top + e["h"] + 6.0
                else:
                    # push e further down: e_top = p_bot - 6
                    e["y"] = p_bot - 6.0
        placed.append(e)

    # Render labels + leader stems
    for e in placed:
        # Leader stem from node to nearest edge of label
        if e["place_above"]:
            stem_y0 = e["y_node"] + 2
            stem_y1 = e["y"] - e["h"] - 2  # bottom edge of the label
        else:
            stem_y0 = e["y_node"] - 2
            stem_y1 = e["y"] + 2  # top of label
        ax.plot(
            [e["x"], e["x"]],
            [stem_y0, stem_y1],
            color=COLOR_HUB_EDGE if e["is_hub"] else "#9ab0c1",
            linewidth=0.6,
            alpha=0.75,
            zorder=9,
        )
        if e["is_hub"]:
            face = "#fff5ed"
            edge_c = COLOR_HUB_EDGE
            text_c = "#5a2d05"
        else:
            face = "white"
            edge_c = "#7798b6"
            text_c = "#0d2c40"
        ax.text(
            e["x"],
            e["y"],
            e["wrapped"],
            fontsize=6.8,
            ha="center",
            va="top",
            color=text_c,
            fontweight="bold",
            bbox=dict(
                facecolor=face,
                edgecolor=edge_c,
                alpha=0.95,
                pad=1.4,
                boxstyle="round,pad=0.22",
            ),
            zorder=10,
        )

    # ---- Panel-letter badges next to each hub in the main panel ----
    for h, letter in zip(hubs, panel_letters):
        if h not in pos:
            continue
        x, y = pos[h]
        # Place badge to the right of the node, slightly above; if the hub
        # is at the very top, place badge below-right to avoid colliding
        # with the title-area sparse-DAG annotation.
        if y >= y_mid:
            bx, by, va = x + 24, y - 18, "top"
        else:
            bx, by, va = x + 24, y + 22, "bottom"
        ax.text(
            bx,
            by,
            f"({letter})",
            fontsize=11,
            fontweight="bold",
            family="sans-serif",
            color=COLOR_HUB_EDGE,
            ha="left",
            va=va,
            bbox=dict(
                facecolor="white",
                edgecolor=COLOR_HUB_EDGE,
                alpha=0.98,
                pad=1.5,
                boxstyle="round,pad=0.22",
            ),
            zorder=12,
        )

    # Set canvas limits so layout matches our chosen coordinate system.
    # Leave ~280 units below 0 for the isolates band + legend.
    ax.set_xlim(-30, 1030)
    ax.set_ylim(-340, 820)
    ax.set_aspect("auto")
    ax.set_axis_off()


def draw_isolates_band(ax, singletons: List[set], sizes: Dict[str, int]):
    """Render isolated topics as a faded textual block within the same axes,
    placed below the network area (negative y in the canvas).

    The list is allocated columns 220..1030 in canvas units; the lower-left
    corner (x<200) is reserved for the legend, so the two never collide.
    """
    iso_names = sorted(
        [next(iter(c)) for c in singletons], key=lambda n: -sizes.get(n, 0)
    )
    if not iso_names:
        return

    # Title for the isolates block
    ax.text(
        625,
        -15,
        f"Isolated topics (no parent or child): {len(iso_names)} singletons",
        fontsize=9,
        ha="center",
        va="top",
        color="#555555",
        style="italic",
        fontweight="bold",
    )

    # 4 columns to the right of the legend region.
    n_cols = 4
    rows_per_col = int(np.ceil(len(iso_names) / n_cols))

    # Column x positions: 230..1010
    col_xs = np.linspace(230, 1000, n_cols)
    y_top = -45
    line_h = 11
    for idx, name in enumerate(iso_names):
        col = idx // rows_per_col
        row = idx % rows_per_col
        if col >= n_cols:
            break
        x = col_xs[col]
        y = y_top - row * line_h
        size_val = sizes.get(name, 0)
        # Faint grey dot marker
        ax.plot(
            [x - 6],
            [y - 1],
            marker="o",
            markersize=3.0,
            color=COLOR_SINGLETON,
            markeredgecolor=COLOR_SINGLETON_EDGE,
            markeredgewidth=0.4,
            linestyle="None",
        )
        ax.text(
            x,
            y,
            f"{name}  ({size_val:,})",
            fontsize=6.2,
            ha="left",
            va="top",
            color="#555555",
        )


def draw_callout(
    ax,
    G: nx.DiGraph,
    hub: str,
    sizes: Dict[str, int],
    panel_letter: str,
    depth: int = 2,
):
    """Render the hub plus all its descendants up to `depth` as a small subgraph.

    Returns the figure-coordinate location of the hub node (used by the caller
    to draw a leader line from the main panel)."""
    nodes = descendants_to_depth(G, hub, depth=depth)
    sub = G.subgraph(nodes).copy()

    pos = _graphviz_layout_safe(sub, prog="dot")
    if pos is None:
        pos = nx.spring_layout(sub, seed=42, k=1.0)

    n_nodes = len(sub.nodes)
    if n_nodes <= 4:
        node_size_const = 1100
        font_size = 8.5
        wrap_width = 18
    elif n_nodes <= 8:
        node_size_const = 750
        font_size = 7.5
        wrap_width = 16
    else:
        node_size_const = 500
        font_size = 6.5
        wrap_width = 13

    node_sizes = [node_size_const for _ in sub.nodes]
    node_colors = [COLOR_HUB if n == hub else COLOR_NODE_DEFAULT for n in sub.nodes]
    edge_colors_node = [
        COLOR_HUB_EDGE if n == hub else COLOR_NODE_DEFAULT_EDGE for n in sub.nodes
    ]

    nx.draw_networkx_edges(
        sub,
        pos,
        ax=ax,
        edge_color="#5a7d99",
        width=1.3,
        alpha=0.9,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=10,
        node_size=node_sizes,
    )
    nx.draw_networkx_nodes(
        sub,
        pos,
        nodelist=list(sub.nodes),
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors=edge_colors_node,
        linewidths=1.2,
        ax=ax,
    )

    # Place labels just below each node
    if pos:
        ys = [p[1] for p in pos.values()]
        yspan = max(ys) - min(ys) if len(ys) > 1 else 1.0
        offset = max(yspan * 0.05, 8.0)
    else:
        offset = 8.0

    for n, (x, y) in pos.items():
        wrapped = _wrap_label(n, width=wrap_width)
        ax.text(
            x,
            y - offset,
            wrapped,
            fontsize=font_size,
            ha="center",
            va="top",
            color="#1a1a1a",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2),
            zorder=10,
        )

    # Pad axes
    if pos:
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        xrange = max(max(xs) - min(xs), 1.0)
        yrange = max(max(ys) - min(ys), 1.0)
        xpad = max(xrange * 0.12, 30)
        ypad_top = max(yrange * 0.15, 25)
        ypad_bot = max(yrange * 0.55, 60)
        ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
        ax.set_ylim(min(ys) - ypad_bot, max(ys) + ypad_top)

    # Title with hub info
    n_children = G.out_degree(hub)
    total = sizes.get(hub, 0)
    title_wrapped = _wrap_label(hub, width=30)
    ax.set_title(
        f"{title_wrapped}\n({n_children} direct children, {total:,} articles)",
        fontsize=9.5,
        fontweight="bold",
        pad=6,
    )

    # Panel letter in upper-left of the axes (in axes-fraction coords)
    ax.text(
        0.02,
        0.98,
        f"({panel_letter})",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        family="sans-serif",
        ha="left",
        va="top",
        color="#1a1a1a",
        zorder=20,
    )

    # Make the callout border visible
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#999999")
        spine.set_linewidth(0.7)

    ax.set_xticks([])
    ax.set_yticks([])


def add_legend(ax):
    """Add a small legend to the main panel, anchored in the lower-left
    region reserved by the canvas (columns 0..210 in data units)."""
    handles = [
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=COLOR_HUB, markeredgecolor=COLOR_HUB_EDGE,
            markersize=9, label="Callout hub (orange)",
        ),
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=COLOR_NODE_DEFAULT, markeredgecolor=COLOR_NODE_DEFAULT_EDGE,
            markersize=9, label="Topic with parent or child edge (blue)",
        ),
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=COLOR_SINGLETON, markeredgecolor=COLOR_SINGLETON_EDGE,
            markersize=7, label="Isolated topic (faint grey; listed below)",
        ),
        Line2D(
            [0], [0], color=COLOR_LEADER, linewidth=1.2,
            label="Leader to callout panel",
        ),
    ]
    # Anchor the legend at data coords (10, -50) -> just below the network
    # area, in the left column reserved for it.
    leg = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(10, -50),
        bbox_transform=ax.transData,
        frameon=True,
        fontsize=7.0,
        title="Legend",
        title_fontsize=7.5,
        framealpha=0.95,
        borderaxespad=0.0,
    )
    leg.get_frame().set_edgecolor("#999999")
    leg.get_frame().set_linewidth(0.6)


# ---------------------------------------------------------------------------
# Main figure assembly
# ---------------------------------------------------------------------------

def build_figure(domain: str, out_path: Path) -> dict:
    G, sizes, parent_child_edges = load_domain(domain)
    pos, singletons = compute_main_layout(G)
    hubs = select_hubs(G, sizes, domain)

    UG = G.to_undirected()
    n_components = nx.number_connected_components(UG)
    component_sizes = sorted(
        (len(c) for c in nx.connected_components(UG)), reverse=True
    )
    is_dag = nx.is_directed_acyclic_graph(G)

    # ---- Figure layout: 16x10 inches, 3 rows x 2 cols ----
    # Top reserved for title + sparse-DAG annotation block (~14% of height).
    fig = plt.figure(figsize=(16, 10), constrained_layout=False)
    gs = gridspec.GridSpec(
        nrows=3,
        ncols=2,
        figure=fig,
        width_ratios=[2.2, 1.0],
        height_ratios=[1, 1, 1],
        wspace=0.06,
        hspace=0.32,
        left=0.025,
        right=0.985,
        top=0.83,
        bottom=0.04,
    )

    panel_letters = ["a", "b", "c"]

    # Main panel spans all 3 rows on the left
    ax_bg = fig.add_subplot(gs[:, 0])
    draw_main_panel(
        ax_bg, G, pos, singletons, sizes, hubs,
        panel_letters=panel_letters, top_n_labels=10,
    )
    draw_isolates_band(ax_bg, singletons, sizes)
    add_legend(ax_bg)

    # Three callouts in the right column
    callout_axes = []
    for i, (hub, letter) in enumerate(zip(hubs, panel_letters)):
        ax = fig.add_subplot(gs[i, 1])
        draw_callout(ax, G, hub, sizes, panel_letter=letter, depth=2)
        callout_axes.append(ax)

    # ---- ONE title only (no duplicate subtitle) ----
    fig.suptitle(
        f"Figure 2. {domain.capitalize()} topic taxonomy (K=100)",
        fontsize=16,
        fontweight="bold",
        x=0.025,
        y=0.975,
        ha="left",
    )

    # ---- Sparse-DAG annotation under the title (manual newlines so we
    # don't depend on matplotlib's auto-wrap, which is unreliable when
    # bbox_inches='tight' is used at save time). ----
    largest_cc = component_sizes[0] if component_sizes else 0
    sparse_msg = (
        f"Sparse subsumption hierarchy: {len(G)} topics, {len(parent_child_edges)} parent-child edges, "
        f"{n_components} weakly connected components (largest = {largest_cc} topics).\n"
        f"At K=100 most topics are too broad to admit a parent-child relationship; deep hierarchy is "
        f"captured only at higher K (Supplementary Figure SX, K=750).\n"
        f"Hierarchy is a DAG, so a node may have multiple parents (e.g. 'Large language models' is "
        f"reached via both 'Machine learning methods' and 'Transformer models')."
    )
    fig.text(
        0.025,
        0.935,
        sparse_msg,
        fontsize=8.5,
        color="#333333",
        ha="left",
        va="top",
    )

    # ---- Leader lines from each hub (in main panel) to its callout panel.
    # We draw these via a thin orange annotation arrow in figure coords AFTER
    # the figure is fully laid out. We use ConnectionPatch via the axes
    # objects so the arrow is owned by the figure-level Axes graph and isn't
    # clipped by bbox_inches='tight' (which only crops to the bounding box
    # of all visible artists). ----
    from matplotlib.patches import ConnectionPatch
    fig.canvas.draw()
    for hub, ax_callout in zip(hubs, callout_axes):
        if hub not in pos:
            continue
        src_xy = pos[hub]  # data coords of ax_bg
        # Target: left edge, vertically centered on the callout panel
        # Use axes-fraction coords on ax_callout: (-0.02, 0.5).
        cp = ConnectionPatch(
            xyA=src_xy,
            coordsA=ax_bg.transData,
            xyB=(-0.02, 0.5),
            coordsB=ax_callout.transAxes,
            arrowstyle="-",
            color=COLOR_LEADER,
            linewidth=0.9,
            alpha=0.55,
            connectionstyle="arc3,rad=-0.10",
            zorder=4,
        )
        fig.add_artist(cp)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "domain": domain,
        "out_path": str(out_path),
        "n_topics": len(G),
        "n_edges": len(parent_child_edges),
        "n_components": n_components,
        "component_sizes": component_sizes,
        "is_dag": is_dag,
        "hubs": hubs,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for domain in ["methodology", "health"]:
        out_path = OUT_DIR / f"figure2_{domain}_taxonomy.png"
        result = build_figure(domain, out_path)
        results.append(result)

    print("=" * 72)
    print("REPORT")
    print("=" * 72)
    for r in results:
        print(f"\nDomain: {r['domain']}")
        print(f"  Output: {r['out_path']}")
        print(f"  Topics: {r['n_topics']},  Edges: {r['n_edges']}")
        print(
            f"  Components (weakly connected): {r['n_components']}  "
            f"sizes={r['component_sizes'][:10]}"
        )
        print(f"  Is DAG: {r['is_dag']}")
        print(f"  Hubs: {r['hubs']}")


if __name__ == "__main__":
    main()
