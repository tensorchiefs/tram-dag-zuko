"""Figures of a TRAM-DAG: the labelled DAG, the marginals, the training curve.

matplotlib is an optional dependency — ``pip install "tramdag[plots]"``. It is
imported on the first call, so importing tramdag never needs it.

::

    from tramdag import plot_dag
    plot_dag(spec)            # or plot_dag(flow): the spec is the DAG
    plot_marginals(flow, df)  # observed vs sampled, one panel per node
    plot_training(flow, frozen=plateau.frozen)   # NLL per epoch, freeze marks
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .spec import NodeSpec, node_parents, validate_and_sort

# %% global variables ------------------------------------------------------------------
# how each effect draws its edge; an unregistered effect falls back to dotted gray
EDGE_STYLE = {
    "LS": dict(color="0.25", ls="-", lw=1.3),
    "CS": dict(color="C0", ls="-", lw=2.4),
    "CI": dict(color="C1", ls="--", lw=1.8),
    "VC": dict(color="C3", ls="-", lw=2.4),
    "VC mod": dict(color="C3", ls=":", lw=1.4),
    "Fn": dict(color="C2", ls="-.", lw=1.6),
}
EDGE_LABEL = {"VCm": "VC mod"}  # to_matrix's tag for a VC modifier
NODE_FACE = {"continuous": "#e3f2fd", "ordinal": "#fff3e0"}
NODE_H, ROW_DY = 0.56, 1.1  # layout units: node height, distance between rows
BULGE = 0.5  # how far an edge that skips a layer bends out, per skipped layer


# %% private functions -----------------------------------------------------------------
def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError(
            'tramdag.plots needs matplotlib: pip install "tramdag[plots]"'
        ) from err
    return plt


def _spec_of(spec_or_flow) -> dict[str, NodeSpec]:
    return getattr(spec_or_flow, "spec", spec_or_flow)


def _node_width(name: str) -> float:
    """Wide enough for the name in bold 10 pt, never narrower than 1 unit."""
    return max(1.0, 0.13 * len(name) + 0.3)


def _layout(spec: dict[str, NodeSpec]) -> tuple[dict[str, tuple[float, float]], float]:
    """Layered left-to-right positions: a node sits one layer past its parents.

    Within a layer the nodes follow their parents' mean row (one barycenter
    sweep), which keeps most edges short and uncrossed on the DAGs this
    package is for. Rows are centered on 0. Also gives the layer distance,
    which grows with the widest node.
    """
    layer_dx = max(_node_width(n) for n in spec) + 1.0
    order = validate_and_sort(spec)
    depth: dict[str, int] = {}
    for name in order:
        depth[name] = 1 + max((depth[p] for p in node_parents(spec[name])), default=-1)
    layers: dict[int, list[str]] = defaultdict(list)
    for name in order:
        layers[depth[name]].append(name)
    pos: dict[str, tuple[float, float]] = {}
    for d in sorted(layers):
        names = layers[d]
        if d > 0:
            names.sort(
                key=lambda n: np.mean([pos[p][1] for p in node_parents(spec[n])])
            )
        for i, n in enumerate(names):
            pos[n] = (d * layer_dx, ((len(names) - 1) / 2 - i) * ROW_DY)
    return pos, layer_dx


def _term_edges(child: str, term) -> list[tuple[str, str, str, bool]]:
    """Give ``(parent, child, effect, joint)`` for the edges one term owns.

    Read off the term's adjacency ``cells``: the tag is the effect (``CI``
    for an intercept edge, ``VCm`` for a VC modifier), with the parent group
    appended for a multi-parent net — that suffix marks a ``joint`` edge.
    """
    edges = []
    for parent, tag in term.cells():
        effect, joint = tag.split("[")[0], "[" in tag
        edges.append((parent, child, EDGE_LABEL.get(effect, effect), joint))
    return edges


def _edges(spec: dict[str, NodeSpec]) -> list[tuple[str, str, str, bool]]:
    """Give every edge of the spec, term by term."""
    return [
        edge
        for child, node in spec.items()
        for term in node.terms
        for edge in _term_edges(child, term)
    ]


def _draw_node(ax, name: str, node: NodeSpec, xy: tuple[float, float]):
    from matplotlib.patches import Ellipse, FancyBboxPatch

    x, y = xy
    w = _node_width(name)
    if node.kind == "ordinal":
        patch = FancyBboxPatch(
            (x - w / 2, y - NODE_H / 2),
            w,
            NODE_H,
            boxstyle="round,pad=0.0,rounding_size=0.12",
            fc=NODE_FACE["ordinal"],
            ec="0.3",
            lw=1.2,
        )
        sub = f"ordinal · {node.levels} levels"
    else:
        patch = Ellipse((x, y), w, NODE_H, fc=NODE_FACE["continuous"], ec="0.3")
        sub = "continuous"
    ax.add_patch(patch)
    ax.text(x, y + 0.06, name, ha="center", va="center", fontsize=10, weight="bold")
    ax.text(x, y - 0.13, sub, ha="center", va="center", fontsize=6.5, color="0.4")
    return patch


def _bulge(pos, layer_dx: float, edge, lane: float) -> float:
    """How far an edge bends out of its chord (positive = upward).

    An edge that skips layers bends away from the middle row, where the
    nodes are, by ``BULGE`` per skipped layer; parallel edges of one pair (a
    VC modifier next to its CS) take separate lanes.
    """
    (x0, y0), (x1, y1) = pos[edge[0]], pos[edge[1]]
    skipped = round((x1 - x0) / layer_dx) - 1
    return BULGE * skipped * (1 if y0 + y1 >= 0 else -1) + lane


def _draw_edge(ax, patches, pos, edge, labels: bool, bulge: float) -> None:
    from matplotlib.patches import FancyArrowPatch

    parent, child, effect, joint = edge
    (x0, y0), (x1, y1) = pos[parent], pos[child]
    style = EDGE_STYLE.get(effect, dict(color="0.5", ls=":", lw=1.2))
    # arc3 bulges by rad * length / 2 to the right of its direction of travel
    dist = float(np.hypot(x1 - x0, y1 - y0))
    rad = -2 * bulge / dist
    arrow = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        patchA=patches[parent],
        patchB=patches[child],
        arrowstyle="-|>,head_length=6,head_width=3",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
        **style,
    )
    ax.add_patch(arrow)
    if labels:
        text = effect + (" joint" if joint else "")
        # the arc's midpoint: the chord's midpoint pushed out by the bulge
        mx = (x0 + x1) / 2 - bulge * (y1 - y0) / dist
        my = (y0 + y1) / 2 + bulge * (x1 - x0) / dist
        ax.text(
            mx,
            my,
            text,
            fontsize=6.5,
            color=style["color"],
            ha="center",
            va="center",
            bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85),
        )


def _legend(ax, effects: set[str]) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], label=e, **EDGE_STYLE[e]) for e in EDGE_STYLE if e in effects
    ]
    if handles:
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=len(handles),
            fontsize=7,
            frameon=False,
        )


def _freezes(rates: list) -> dict[str, int]:
    """Give ``{node: epoch}`` of the first zero rate per node in ``history["lr"]``."""
    frozen: dict[str, int] = {}
    for epoch, entry in enumerate(rates, start=1):
        if isinstance(entry, dict):
            for node, lr in entry.items():
                if lr == 0.0 and node not in frozen:
                    frozen[node] = epoch
    return frozen


def _finish(ax, fig, path):
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return ax


# %% public functions ------------------------------------------------------------------
def plot_dag(
    spec_or_flow, *, ax=None, labels: bool = True, legend: bool = True, path=None
):
    """Draw the labelled DAG of a spec (or of a fitted flow).

    Layers run left to right, a node one layer past its parents. Continuous
    nodes are ellipses, ordinal nodes rounded boxes with their level count.
    Each edge is drawn by the term that owns it: ``LS`` thin gray, ``CS`` thick
    blue, a complex intercept (``CI``) dashed orange, a ``VC`` treatment edge
    red with its modifiers dotted, ``Fn`` dash-dotted green; a multi-parent
    CS/CI is labelled ``joint``.

    Parameters
    ----------
    spec_or_flow : dict[str, NodeSpec] | CausalFlowDAG
        The DAG to draw. A flow draws its ``spec``.
    ax : matplotlib.axes.Axes | None, optional
        Draw into this axes; by default a new figure sized to the layout.
    labels : bool, optional
        Write the effect on each edge, by default True.
    legend : bool, optional
        Add a legend of the effects used, by default True.
    path : str | Path | None, optional
        Save the figure here (150 dpi) after drawing.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn into.
    """
    plt = _plt()
    spec = _spec_of(spec_or_flow)
    if not spec:
        raise ValueError("plot_dag needs a spec with at least one node")
    pos, layer_dx = _layout(spec)
    xs, ys = (np.array(v) for v in zip(*pos.values(), strict=True))
    parallel = defaultdict(list)
    for edge in _edges(spec):
        parallel[edge[:2]].append(edge)
    bulges = {
        edge: _bulge(pos, layer_dx, edge, 0.35 * (k - (len(pair) - 1) / 2))
        for pair in parallel.values()
        for k, edge in enumerate(pair)
    }
    # room for the arcs above and below the rows
    reach_up = max((b for b in bulges.values()), default=0.0)
    reach_down = max((-b for b in bulges.values()), default=0.0)
    half_w = max(_node_width(n) for n in spec) / 2
    x_lo, x_hi = xs.min() - half_w - 0.2, xs.max() + half_w + 0.2
    y_lo, y_hi = ys.min() - 0.5 - max(reach_down, 0), ys.max() + 0.5 + max(reach_up, 0)
    if ax is None:
        _, ax = plt.subplots(figsize=(0.9 * (x_hi - x_lo), 0.9 * (y_hi - y_lo) + 0.5))
    patches = {name: _draw_node(ax, name, spec[name], xy) for name, xy in pos.items()}
    for edge, bulge in bulges.items():
        _draw_edge(ax, patches, pos, edge, labels, bulge)
    if legend:
        _legend(ax, {e for _, _, e, _ in bulges})
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return _finish(ax, ax.figure, path)


def plot_marginals(
    flow, df: pd.DataFrame, *, ncols: int = 3, bins: int = 30, seed=None, path=None
):
    """Observed vs sampled marginal of every node, one panel each.

    Ordinal nodes compare level proportions side by side; continuous nodes a
    density histogram of the data with the flow's sample as a step outline.
    The sample has as many rows as ``df``.

    Parameters
    ----------
    flow : CausalFlowDAG
        The fitted flow.
    df : pd.DataFrame
        The data to compare against (the validation split, typically).
    ncols : int, optional
        Panels per row, by default 3.
    bins : int, optional
        Histogram bins of a continuous node, by default 30.
    seed : int | None, optional
        Seed of the flow's sample.
    path : str | Path | None, optional
        Save the figure here (150 dpi) after drawing.

    Returns
    -------
    numpy.ndarray of matplotlib.axes.Axes
        The panels, in the flow's node order (unused panels are switched off).
    """
    plt = _plt()
    sample = flow.sample(len(df), seed=seed)
    nrows = -(-len(flow.order) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows), squeeze=False
    )
    for ax in axes.flat[len(flow.order) :]:
        ax.set_axis_off()
    for ax, name in zip(axes.flat, flow.order, strict=False):
        node = flow.spec[name]
        if node.kind == "ordinal":
            lv, w = np.arange(node.levels), 0.4
            for x, d, label in [(lv - w / 2, df, "data"), (lv + w / 2, sample, "flow")]:
                counts = d[name].value_counts(normalize=True).reindex(lv, fill_value=0)
                ax.bar(x, counts, w, label=label)
            ax.set_xticks(lv)
            ax.set_ylabel("proportion")
        else:
            edges = np.linspace(df[name].min(), df[name].max(), bins + 1)
            ax.hist(df[name], bins=edges, density=True, alpha=0.5, label="data")
            ax.hist(
                sample[name],
                bins=edges,
                density=True,
                histtype="step",
                lw=1.5,
                label="flow",
            )
            ax.set_ylabel("density")
        ax.set_title(name)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("observed vs sampled marginals")
    _finish(axes.flat[0], fig, path)
    return axes


def plot_training(flow, *, frozen=None, ax=None, path=None):
    """Draw the summed train (and validation) NLL per epoch of the last ``fit``.

    Parameters
    ----------
    flow : CausalFlowDAG
        The fitted flow; ``flow.history`` is read.
    frozen : dict[str, int] | PerNodePlateau | None, optional
        ``{node: epoch}`` of the freezes, each a dashed mark. By default read
        off ``flow.history["lr"]`` (the first epoch a node's rate is 0, when
        the optimizer had per-node groups); pass a dict, or the
        :class:`tramdag.callbacks.PerNodePlateau` whose ``frozen`` to use.
    ax : matplotlib.axes.Axes | None, optional
        Draw into this axes; by default a new figure.
    path : str | Path | None, optional
        Save the figure here (150 dpi) after drawing.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn into.
    """
    plt = _plt()
    hist = flow.history
    if not hist.get("train"):
        raise ValueError("plot_training needs a fitted flow; its history is empty")
    curves = {"train": np.array([sum(d.values()) for d in hist["train"]])}
    if hist.get("val"):
        curves["val"] = np.array([sum(d.values()) for d in hist["val"]])
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 3.6))
    for label, curve in curves.items():
        ax.plot(np.arange(1, len(curve) + 1), curve, label=f"{label} NLL (total)")
    # zoom past the initial drop: the top is the curves' level after 10 % of the epochs
    lo = min(c.min() for c in curves.values())
    hi = max(c[len(c) // 10] for c in curves.values())
    if hi > lo:
        ax.set_ylim(lo - 0.05 * (hi - lo), hi)
    frozen = getattr(frozen, "frozen", frozen)
    if frozen is None:
        frozen = _freezes(hist.get("lr", []))
    # after the zoom, so the annotations hang from the visible top
    for name, epoch in sorted(frozen.items(), key=lambda kv: kv[1]):
        ax.axvline(epoch, ls="--", lw=1, color="gray")
        ax.annotate(
            f" {name} frozen",
            (epoch, ax.get_ylim()[1]),
            rotation=90,
            va="top",
            fontsize=8,
            color="gray",
        )
    ax.set_xlabel("epoch"), ax.set_ylabel("NLL"), ax.legend(frameon=False)
    ax.set_title("training")
    return _finish(ax, ax.figure, path)
