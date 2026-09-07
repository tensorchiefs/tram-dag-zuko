"""``tramdag.plots``: every effect draws, and the missing extra is named."""

# %% imports ---------------------------------------------------------------------------
import sys

import matplotlib as mpl
import pytest

mpl.use("Agg")

from tramdag import CI, CS, LS, VC, CausalFlowDAG, ContinuousNode, OrdinalNode, plot_dag
from tramdag.callbacks import PerNodePlateau, per_node_adam
from tramdag.plots import plot_marginals, plot_training


# %% private functions -----------------------------------------------------------------
def _every_effect_spec():
    return {
        "x1": ContinuousNode(),
        "x2": OrdinalNode(3, [CI("x1")]),
        "t": OrdinalNode(2, [LS("x1"), CS("x2")]),
        "y": ContinuousNode([CS("x1", "x2"), VC("x1", t="t")]),
    }


# %% public functions ------------------------------------------------------------------
def test_plot_dag_draws_every_node_and_edge():
    """One patch per node, one arrow per edge, a label per edge, a legend."""
    spec = _every_effect_spec()
    ax = plot_dag(spec)
    n_edges = 1 + 2 + 2 + 1 + 1  # CI, LS+CS, joint CS (2 parents), VC, VC mod
    arrows = [p for p in ax.patches if type(p).__name__ == "FancyArrowPatch"]
    assert len(ax.patches) == len(spec) + n_edges
    assert len(arrows) == n_edges
    labels = {t.get_text() for t in ax.texts}
    assert {"CI", "LS", "CS", "CS joint", "VC", "VC mod"} <= labels
    assert ax.get_legend() is not None
    # a flow draws its spec; labels and legend are optional
    flow = CausalFlowDAG(spec, seed=0)
    ax2 = plot_dag(flow, labels=False, legend=False)
    assert len(ax2.patches) == len(ax.patches)
    assert ax2.get_legend() is None
    assert {t.get_text() for t in ax2.texts} == {
        *spec,
        "continuous",
        "ordinal · 3 levels",
        "ordinal · 2 levels",
    }


def test_plot_dag_layers_children_past_their_parents():
    """A node sits one layer right of its deepest parent."""
    from tramdag.plots import _layout

    pos, layer_dx = _layout(_every_effect_spec())
    assert [pos[n][0] / layer_dx for n in ("x1", "x2", "t", "y")] == [0, 1, 2, 3]


def test_marginals_and_training_draw_from_a_fitted_flow(ls_chain, tmp_path):
    """Both figures read the flow after a short plateau fit; ``path`` saves."""
    df = ls_chain["draw"](300, 0)[["x1", "x2"]]
    spec = {"x1": ContinuousNode(), "x2": ContinuousNode([LS("x1")])}
    flow = CausalFlowDAG(spec, seed=0)
    plateau = PerNodePlateau(patience=2, freeze=4)
    flow.fit(
        df,
        epochs=30,
        batch_size=100,
        validation_data=df,
        optimizer=per_node_adam(flow, lr=1e-2),
        callbacks=plateau,
    )
    axes = plot_marginals(flow, df, ncols=2, seed=0, path=tmp_path / "m.png")
    assert axes.shape == (1, 2)
    assert (tmp_path / "m.png").exists()
    ax = plot_training(flow, path=tmp_path / "t.png")  # freezes read off history["lr"]
    assert len(ax.lines) == 2 + len(plateau.frozen)  # train, val, one mark per freeze
    assert (tmp_path / "t.png").exists()
    ax = plot_training(flow, frozen=plateau)  # or from the callback itself
    assert len(ax.lines) == 2 + len(plateau.frozen)
    # no validation history, no marks: one line
    flow2 = CausalFlowDAG(spec, seed=0)
    flow2.fit(df, epochs=3, batch_size=100)
    assert len(plot_training(flow2).lines) == 1


def test_plots_name_the_optional_dependency(monkeypatch):
    """Without matplotlib the error says what to install."""
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(ImportError, match=r"tramdag\[plots\]"):
        plot_dag(_every_effect_spec())
