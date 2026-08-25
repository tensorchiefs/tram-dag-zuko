"""``density()``: the analytic conditional density of a continuous node."""

import numpy as np
import pandas as pd
import pytest

from tramdag import LS, CausalFlowDAG, ContinuousNode, OrdinalNode


def _flow(ls_chain):
    """An untrained two-node chain on the ls_chain draw: x2 <- x1."""
    spec = {"x1": ContinuousNode(), "x2": ContinuousNode([LS("x1")])}
    return CausalFlowDAG(spec, seed=0), ls_chain["draw"](64, 0)[["x1", "x2"]]


def test_density_equals_exp_log_prob_at_the_observed_value(ls_chain):
    """At a row's own value the density is exp of that row's log-likelihood term."""
    flow, df = _flow(ls_chain)
    per_node = flow.node_log_prob(flow._tensorize(df))
    expected = per_node["x2"].exp().detach().numpy()
    got = np.array(
        [flow.density(df.iloc[[i]], "x2", [df["x2"].iloc[i]])[0, 0] for i in range(8)]
    )
    np.testing.assert_allclose(got, expected[:8], rtol=1e-5)


def test_density_integrates_to_one(ls_chain):
    flow, df = _flow(ls_chain)
    grid = np.linspace(-40, 40, 8001)
    dens = flow.density(df.iloc[:5], "x2", grid)
    assert dens.shape == (5, grid.size)
    np.testing.assert_allclose(np.trapezoid(dens, grid, axis=1), 1.0, atol=1e-3)


def test_do_overrides_the_parent_column(ls_chain):
    flow, df = _flow(ls_chain)
    grid = np.linspace(-3, 3, 7)
    with_do = flow.density(df.iloc[:3], "x2", grid, do={"x1": 0.7})
    by_hand = flow.density(df.iloc[:3].assign(x1=0.7), "x2", grid)
    np.testing.assert_allclose(with_do, by_hand)
    assert np.allclose(with_do[0], with_do[1])  # the only parent is clamped


def test_density_rejects_an_ordinal_node():
    spec = {"x": ContinuousNode(), "y": OrdinalNode(3, [LS("x")])}
    flow = CausalFlowDAG(spec, seed=0)
    with pytest.raises(ValueError, match="requires a continuous node"):
        flow.density(pd.DataFrame({"x": [0.0]}), "y", [0, 1, 2])
