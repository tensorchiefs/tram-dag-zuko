"""``net_input_scaling="minmax"``: the reference's ``scale_df`` for the nets.

Only the networks (complex intercepts, complex shifts, VC modifiers) see the
scaled parents; a linear shift stays raw so its weight keeps its units.
"""

# %% imports ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pytest
import torch

from tramdag import CI, CS, LS, VC, CausalFlowDAG, ContinuousNode, OrdinalNode

# %% global variables ------------------------------------------------------------------
SPEC = {
    "x1": ContinuousNode(),
    "x2": ContinuousNode(CI("x1", units=[4])),
    "x3": ContinuousNode(CS("x1", units=[4]) + LS("x2")),
}


# %% private functions -----------------------------------------------------------------
def _tensors(df):
    return {c: torch.tensor(df[c].to_numpy(), dtype=torch.float32) for c in df}


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(5.0, 3.0, n)  # far from [0, 1]: a raw tanh net saturates
    x2 = 0.5 * x1 + rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x2 - x1 + rng.normal(size=n)})


# %% public functions ------------------------------------------------------------------
def test_invalid_value_rejected():
    with pytest.raises(ValueError, match="minmax"):
        CausalFlowDAG(SPEC, net_input_scaling="zscore")


def test_net_inputs_span_unit_interval_after_first_fit():
    df = _frame()
    flow = CausalFlowDAG(SPEC, seed=0, net_input_scaling="minmax")
    flow.fit(df, epochs=1, batch_size=100)
    feats = flow._features(_tensors(df))
    for name in ("x2", "x3"):
        node = flow.nodes[name]
        scaled = node.net_input(feats, ("x1",))
        assert torch.allclose(scaled.min(0).values, torch.zeros(scaled.shape[1]))
        assert torch.allclose(scaled.max(0).values, torch.ones(scaled.shape[1]))
    # a second fit on other rows keeps the first calibration
    flow.fit(df + 10.0, epochs=1, batch_size=100)
    assert float(flow.nodes["x2"].net_lo[0]) == pytest.approx(df["x1"].min())


def test_linear_shift_stays_raw():
    """An all-LS model is bit-identical with and without the option."""
    df = _frame()
    spec = {"x1": ContinuousNode(), "x2": ContinuousNode(LS("x1"))}
    raw = CausalFlowDAG(spec, seed=1)
    scaled = CausalFlowDAG(spec, seed=1, net_input_scaling="minmax")
    raw.fit(df, epochs=2, batch_size=50, seed=0)
    scaled.fit(df, epochs=2, batch_size=50, seed=0)
    assert raw.ls_coefficients() == scaled.ls_coefficients()


def test_save_load_keeps_option_and_calibration(tmp_path):
    df = _frame()
    flow = CausalFlowDAG(SPEC, seed=0, net_input_scaling="minmax")
    flow.fit(df, epochs=1, batch_size=100)
    flow.save(tmp_path / "flow.pt")
    loaded = CausalFlowDAG.load(tmp_path / "flow.pt")
    assert loaded.net_input_scaling == "minmax"
    u = pd.DataFrame(np.random.default_rng(1).logistic(size=(20, 3)), columns=SPEC)
    pd.testing.assert_frame_equal(
        flow.sample(20, u=u, do={"x1": 9.0}), loaded.sample(20, u=u, do={"x1": 9.0})
    )


def test_mixed_node_scales_only_the_network_input():
    """``CS("x1") + LS("x2")``: the net sees the scaled x1, the LS the raw x2."""
    df = _frame()
    flow = CausalFlowDAG(SPEC, seed=0, net_input_scaling="minmax")
    flow.fit(df, epochs=1, batch_size=100)
    nd = flow.nodes["x3"]
    feats = flow._features(_tensors(df))
    with torch.no_grad():
        _, shift = nd.theta_shift(feats, len(df))
        expected = nd.shifts["x1"](nd.net_input(feats, ("x1",))) + nd.shifts["x2"](
            feats["x2"]
        )
    assert torch.allclose(shift, expected)


def test_read_outs_use_the_scaled_inputs():
    """``varying_coef`` and ``intercept_contributions`` see the fitted model."""
    rng = np.random.default_rng(2)
    n = 300
    x1, x2 = rng.normal(5, 3, n), rng.normal(-4, 2, n)
    t = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {"x1": x1, "x2": x2, "t": t, "y": x1 - x2 + t * x1 + rng.normal(size=n)}
    )
    spec = {
        "x1": ContinuousNode(),
        "x2": ContinuousNode(),
        "t": OrdinalNode(2, LS("x1")),
        "y": ContinuousNode(
            CI("x1", "x2", units=[4], allow_interaction=False)
            + VC("x1", t="t", units=[4])
        ),
    }
    flow = CausalFlowDAG(spec, seed=0, net_input_scaling="minmax")
    flow.fit(df, epochs=1, batch_size=100)
    nd = flow.nodes["y"]
    feats = flow._features(_tensors(df))
    with torch.no_grad():
        vc = nd.shifts["t"]
        beta = vc.beta(nd.net_input(feats, ("x1",)), len(df))
        parts = flow.intercept_contributions("y", df)
        raw = nd.intercept_nets[0](nd.net_input(feats, ("x1",)))
    np.testing.assert_allclose(flow.varying_coef("y", df), beta.numpy().ravel())
    np.testing.assert_allclose(
        parts["contributions"]["x1"], (raw - raw.mean(0)).numpy(), atol=1e-6
    )


def test_ordinal_parent_passes_through_one_hot():
    spec = {"k": OrdinalNode(3), "y": ContinuousNode(CS("k", units=[4]))}
    df = pd.DataFrame({"k": [0, 1, 2, 1], "y": [0.1, 0.5, -0.3, 0.2]})
    flow = CausalFlowDAG(spec, seed=0, net_input_scaling="minmax")
    flow.fit(df, epochs=1, batch_size=4)
    feats = flow._features(_tensors(df))
    assert torch.equal(flow.nodes["y"].net_input(feats, ("k",)), feats["k"])


def test_constant_parent_is_rejected():
    df = _frame().assign(x1=1.0)
    flow = CausalFlowDAG(SPEC, seed=0, net_input_scaling="minmax")
    with pytest.raises(ValueError, match="constant"):
        flow.fit(df, epochs=1, batch_size=100)
