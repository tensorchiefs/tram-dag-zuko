"""Per-term ``input_transform=``: the networks' input scaling, one per net.

Only the term's own network (complex intercept, complex shift, VC modifiers)
sees the transformed parents; a linear shift stays raw so its weight keeps
its units. Named strategies freeze their statistics from the training rows at
``calibrate``; a callable ``fn(x, train)`` receives the frozen raw train
column, never the batch's.
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
    "x2": ContinuousNode(CI("x1", units=[4], input_transform="minmax")),
    "x3": ContinuousNode(CS("x1", units=[4], input_transform="minmax") + LS("x2")),
}


# %% private functions -----------------------------------------------------------------
def _tensors(df):
    return {c: torch.tensor(df[c].to_numpy(), dtype=torch.float32) for c in df}


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(5.0, 3.0, n)  # far from [0, 1]: a raw tanh net saturates
    x2 = 0.5 * x1 + rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x2 - x1 + rng.normal(size=n)})


def _halve(x, train):  # module-level: picklable, so save() accepts it
    return x / 2.0


# %% public functions ------------------------------------------------------------------
def test_invalid_value_rejected():
    with pytest.raises(ValueError, match="input_transform"):
        CausalFlowDAG(
            {
                "a": ContinuousNode(),
                "b": ContinuousNode(CI("a", input_transform="zscale")),
            }
        )


def test_simple_intercept_rejects_the_option():
    from tramdag import SI

    with pytest.raises(ValueError, match="no network inputs"):
        CausalFlowDAG({"a": ContinuousNode([SI(input_transform="minmax")])})


def test_net_inputs_span_unit_interval_after_first_fit():
    df = _frame()
    flow = CausalFlowDAG(SPEC, seed=0)
    flow.fit(df, epochs=1, batch_size=100)
    feats = flow._features(_tensors(df))
    for name, key in (("x2", "@I"), ("x3", "x1")):
        node = flow.nodes[name]
        scaled = node.net_input(feats, ("x1",), key)
        assert torch.allclose(scaled.min(0).values, torch.zeros(scaled.shape[1]))
        assert torch.allclose(scaled.max(0).values, torch.ones(scaled.shape[1]))
    # a second fit on other rows keeps the first frozen statistics
    flow.fit(df + 10.0, epochs=1, batch_size=100)
    lo = float(flow.nodes["x2"].input_transforms["@I"].lo[0])
    assert lo == pytest.approx(df["x1"].min())


def test_standardize_freezes_train_mean_and_std():
    df = _frame()
    spec = {
        "x1": ContinuousNode(),
        "x2": ContinuousNode(CI("x1", units=[4], input_transform="standardize")),
    }
    flow = CausalFlowDAG(spec, seed=0)
    flow.fit(df[["x1", "x2"]], epochs=1, batch_size=100)
    feats = flow._features(_tensors(df[["x1", "x2"]]))
    z = flow.nodes["x2"].net_input(feats, ("x1",), "@I")
    x = torch.tensor(df["x1"].to_numpy(), dtype=torch.float32)
    expected = (x - x.mean()) / x.std(correction=1)
    assert torch.allclose(z.ravel(), expected, atol=1e-5)


def test_callable_gets_the_frozen_train_column():
    df = _frame()
    spec = {
        "x1": ContinuousNode(),
        "x2": ContinuousNode(
            CI("x1", units=[4], input_transform=lambda x, train: x / train.std())
        ),
    }
    flow = CausalFlowDAG(spec, seed=0)
    flow.fit(df[["x1", "x2"]], epochs=1, batch_size=100)
    nd = flow.nodes["x2"]
    # a 3-row batch is transformed with the TRAIN std, not the batch's
    batch = {"x1": torch.tensor([[1.0], [2.0], [3.0]])}
    got = nd.net_input(batch, ("x1",), "@I")
    train_std = float(torch.tensor(df["x1"].to_numpy(), dtype=torch.float32).std())
    assert torch.allclose(got, batch["x1"] / train_std, atol=1e-6)


def test_save_rejects_a_lambda_and_accepts_a_module_function(tmp_path):
    df = _frame()
    lam = {
        "x1": ContinuousNode(),
        "x2": ContinuousNode(CI("x1", units=[4], input_transform=lambda x, t: x)),
    }
    flow = CausalFlowDAG(lam, seed=0)
    flow.fit(df[["x1", "x2"]], epochs=1, batch_size=100)
    with pytest.raises(ValueError, match="picklable"):
        flow.save(tmp_path / "no.pt")

    ok = {
        "x1": ContinuousNode(),
        "x2": ContinuousNode(CI("x1", units=[4], input_transform=_halve)),
    }
    flow = CausalFlowDAG(ok, seed=0)
    flow.fit(df[["x1", "x2"]], epochs=1, batch_size=100)
    flow.save(tmp_path / "ok.pt")
    loaded = CausalFlowDAG.load(tmp_path / "ok.pt")
    u = pd.DataFrame(
        np.random.default_rng(1).logistic(size=(20, 2)), columns=["x1", "x2"]
    )
    pd.testing.assert_frame_equal(
        flow.sample(20, u=u, do={"x1": 9.0}), loaded.sample(20, u=u, do={"x1": 9.0})
    )


def test_save_load_keeps_option_and_calibration(tmp_path):
    df = _frame()
    flow = CausalFlowDAG(SPEC, seed=0)
    flow.fit(df, epochs=1, batch_size=100)
    flow.save(tmp_path / "flow.pt")
    loaded = CausalFlowDAG.load(tmp_path / "flow.pt")
    assert loaded.nodes["x2"].input_transforms["@I"].kind == "minmax"
    u = pd.DataFrame(np.random.default_rng(1).logistic(size=(20, 3)), columns=SPEC)
    pd.testing.assert_frame_equal(
        flow.sample(20, u=u, do={"x1": 9.0}), loaded.sample(20, u=u, do={"x1": 9.0})
    )


def test_mixed_node_transforms_only_its_own_term():
    """``CS("x1", input_transform=...) + LS("x2")``: net scaled, LS raw."""
    df = _frame()
    flow = CausalFlowDAG(SPEC, seed=0)
    flow.fit(df, epochs=1, batch_size=100)
    nd = flow.nodes["x3"]
    feats = flow._features(_tensors(df))
    grid = df["x1"].to_numpy()
    with torch.no_grad():
        _, shift = nd.theta_shift(feats, len(df))
    # public routes: the CS curve (through its input transform) + the raw LS
    cs = flow.shift_curve("x3", "x1", grid)
    ls = float(flow.ls_coefficients()["x3"]["x2"][0]) * df["x2"].to_numpy()
    np.testing.assert_allclose(shift.numpy(), cs + ls, atol=1e-5)


def test_read_outs_use_the_transformed_inputs():
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
            CI(
                "x1",
                "x2",
                units=[4],
                allow_interaction=False,
                input_transform="minmax",
            )
            + VC("x1", t="t", units=[4], input_transform="minmax")
        ),
    }
    flow = CausalFlowDAG(spec, seed=0)
    flow.fit(df, epochs=1, batch_size=100)
    nd = flow.nodes["y"]
    feats = flow._features(_tensors(df))
    with torch.no_grad():
        vc = nd.shifts["t"]
        beta = vc.beta(nd.net_input(feats, ("x1",), "t"), len(df))
        parts = flow.intercept_contributions(df, "y")
        raw = nd.intercept.nets[0](nd.net_input(feats, ("x1",), "@I"))
    np.testing.assert_allclose(flow.varying_coef(df, "y"), beta.numpy().ravel())
    np.testing.assert_allclose(
        parts["contributions"]["x1"], (raw - raw.mean(0)).numpy(), atol=1e-6
    )


def test_ordinal_parent_passes_through_one_hot():
    spec = {
        "k": OrdinalNode(3),
        "y": ContinuousNode(CS("k", units=[4], input_transform="minmax")),
    }
    df = pd.DataFrame({"k": [0, 1, 2, 1], "y": [0.1, 0.5, -0.3, 0.2]})
    flow = CausalFlowDAG(spec, seed=0)
    flow.fit(df, epochs=1, batch_size=4)
    feats = flow._features(_tensors(df))
    assert torch.equal(flow.nodes["y"].net_input(feats, ("k",), "k"), feats["k"])


def test_degenerate_column_is_rejected():
    """A constant column has no transform domain and no statistics: fail loudly.

    ``calibrate`` checks the node's own quantiles first, so that message wins
    over the input-transform one; either way nothing trains into NaN.
    """
    df = _frame().assign(x1=1.0)
    flow = CausalFlowDAG(SPEC, seed=0)
    with pytest.raises(ValueError, match="quantiles coincide"):
        flow.fit(df, epochs=1, batch_size=100)
