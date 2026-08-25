"""Tests for the propensity-centered VC term (issue #30):
beta(x) * (t - e_hat(x)) with cross-fitted (out-of-fold) e_hat.

Acceptance: center=False is bit-identical to #28's VC (regression guard);
gradient isolation (no gradient reaches the treatment node from the outcome
loss); the Dandl reproduction (confounded DGP + deliberately under-specified
prognostic part: centering must materially reduce the bias of beta_hat —
measured 5-10x on this protocol); the OOF plumbing is real fold bookkeeping
(an in-sample "simplification" fails these tests); do()/sample() recompute
t - e_hat under intervention (never cached).
"""

# %% imports ---------------------------------------------------------------------------
import numpy as np
import pytest
import torch

from tramdag import CS, LS, VC, CausalFlowDAG, ContinuousNode, I, OrdinalNode
from tramdag.spec import spec_from_dict, spec_to_dict, validate_and_sort


# %% private functions -----------------------------------------------------------------
def _misspecified_spec(center) -> dict:
    return {
        "X": ContinuousNode([I(transform="affine")]),
        "T": OrdinalNode(2, [LS("X")]),
        # prognostic part deliberately under-specified (linear vs true x^2)
        "Y": ContinuousNode([LS("X"), VC("X", center=center, t="T")]),
    }


def _fit(spec, df, epochs=250):
    flow = CausalFlowDAG(spec, seed=0)
    flow.fit(
        df.iloc[:5400],
        df.iloc[5400:],
        epochs=epochs,
        learning_rate=1e-2,
        batch_size=512,
        verbose=0,
        seed=0,
        restore_best=True,
    )
    return flow


# %% public functions ------------------------------------------------------------------
def test_center_validation():
    with pytest.raises(ValueError, match="center_folds"):
        VC("X", center=True, center_folds=1, t="T")
    spec = {"D": ContinuousNode(), "Y": ContinuousNode([VC(center=True, t="D")])}
    with pytest.raises(ValueError, match="binary ordinal"):
        validate_and_sort(spec)  # continuous treatment cannot center


def test_center_serialization_roundtrip():
    spec = {
        "X": ContinuousNode(),
        "T": OrdinalNode(2, [LS("X")]),
        "Y": ContinuousNode([VC("X", center=True, center_folds=3, t="T")]),
    }
    round_tripped = spec_from_dict(spec_to_dict(spec))
    t = next(t for t in round_tripped["Y"].terms if t.effect == "VC")
    assert (t.center, t.center_folds) == (True, 3)


def test_center_false_is_bit_identical_to_plain_vc(vc_hetero):
    """The default must preserve #28's behavior exactly: a VC term written
    without the kwarg and one with center=False produce bit-identical fits.
    """
    df = vc_hetero["draw"](1200, 100)

    def fit_with(term):
        spec = {
            "X1": ContinuousNode([I(transform="affine")]),
            "X2": ContinuousNode([I(transform="affine")]),
            "X3": ContinuousNode([I(transform="affine")]),
            "T": OrdinalNode(2, [LS("X1"), LS("X2")]),
            "Y": ContinuousNode([CS("X1", "X2", "X3"), term]),
        }
        flow = CausalFlowDAG(spec, seed=3)
        flow.fit(df.iloc[:1000], df.iloc[1000:], epochs=15, verbose=0, seed=3)
        return flow

    a = fit_with(VC("X2", "X3", t="T"))
    b = fit_with(VC("X2", "X3", center=False, t="T"))
    assert VC("X2", t="T") == VC("X2", center=False, t="T")  # Term equality
    for (ka, pa), (kb, pb) in zip(
        a.state_dict().items(), b.state_dict().items(), strict=True
    ):
        assert ka == kb
        assert torch.equal(pa, pb), ka
    assert a.vc_center_info == {}  # stage 1 never ran


def test_gradient_isolation(confounded):
    """With center=True the treatment node's parameters receive ZERO gradient
    from the outcome-node loss — on the live (inference) e_hat path and, a
    fortiori, on the frozen-OOF training path.
    """
    df = confounded["draw"](800, 1)
    flow = CausalFlowDAG(_misspecified_spec(True), seed=0)
    flow.fit(df, epochs=3, verbose=0, seed=0)  # stage 1 + a few steps
    flow.zero_grad()
    vals = flow._tensorize(df)
    (-flow.node_log_prob(vals)["Y"].mean()).backward()
    for p in flow.nodes["T"].parameters():
        assert p.grad is None or float(p.grad.abs().max()) == 0.0
    # the Y-node itself DID get gradients (the loss is not degenerate)
    assert any(
        p.grad is not None and float(p.grad.abs().max()) > 0
        for p in flow.nodes["Y"].parameters()
    )


def test_training_ehat_is_out_of_fold(confounded):
    """The training propensities are genuine OOF quantities: fold bookkeeping
    exists, each fold's values reproduce a proxy fitted WITHOUT that fold
    (recomputed independently here), and they differ from the in-sample
    full-data fit — a later 'simplification' to in-sample e_hat fails this.
    """
    df = confounded["draw"](1200, 2)
    flow = CausalFlowDAG(_misspecified_spec(True), seed=0)
    flow.fit(df, epochs=2, verbose=0, seed=0)
    info = flow.vc_center_info[("Y", "T")]
    assert info["folds"] == 5
    fold_id, e_oof = info["fold_id"], info["e_oof"]
    assert fold_id.shape == (len(df),)
    assert set(fold_id) == set(range(5))

    # (a) fold-0 values equal an independent refit WITHOUT fold 0 (deterministic)
    proxy_spec = {
        "X": ContinuousNode([I(transform="affine")]),
        "T": OrdinalNode(2, [LS("X")]),
    }
    proxy = CausalFlowDAG(proxy_spec, seed=0)
    proxy.fit_classical(df.iloc[fold_id != 0][["X", "T"]], verbose=False)
    np.testing.assert_allclose(
        e_oof[fold_id == 0], proxy._predict_p1("T", df.iloc[fold_id == 0]), atol=1e-7
    )

    # (b) OOF differs from the in-sample full-data fit
    full = CausalFlowDAG(proxy_spec, seed=0)
    full.fit_classical(df[["X", "T"]], verbose=False)
    e_in = full._predict_p1("T", df)
    assert np.abs(e_oof - e_in).max() > 1e-4


def test_treatment_with_a_bare_intercept_takes_the_classical_oof_path(confounded):
    """A parentless ``I()`` on the treatment does not force the Adam proxy.

    ``[I, LS("X")]`` is an all-``ls`` conditional with its simple intercept
    written out (the spelling that carries ``transform=``), so stage 1 must use
    the deterministic ``fit_classical``, exactly as ``[LS("X")]`` does. The
    per-node test used to demand ``effect == "LS"`` for every term and sent this
    spec down the 300-epoch Adam path instead.

    Determinism is the observable: the classical fit is reproducible to the last
    bits, the Adam proxy is not run twice from the same state.
    """
    df = confounded["draw"](600, 3)
    spec = {
        "X": ContinuousNode([I(transform="affine")]),
        "T": OrdinalNode(2, [I, LS("X")]),
        "Y": ContinuousNode([I("X"), VC(t="T", center=True, center_folds=3)]),
    }
    runs = []
    for _ in range(2):
        flow = CausalFlowDAG(spec, seed=0)
        flow.fit(df, epochs=2, verbose=0, seed=0)
        runs.append(np.array(flow.vc_center_info[("Y", "T")]["e_oof"]))
    assert np.array_equal(runs[0], runs[1])

    # and it agrees with the classical fit of the same conditional, out of fold
    info = CausalFlowDAG(spec, seed=0)
    info.fit(df, epochs=2, verbose=0, seed=0)
    book = info.vc_center_info[("Y", "T")]
    fold_id = book["fold_id"]
    proxy = CausalFlowDAG(
        {
            "X": ContinuousNode([I(transform="affine")]),
            "T": OrdinalNode(2, [I, LS("X")]),
        },
        seed=0,
    )
    proxy.fit_classical(df.iloc[fold_id != 0][["X", "T"]], verbose=False)
    np.testing.assert_allclose(
        book["e_oof"][fold_id == 0],
        proxy._predict_p1("T", df.iloc[fold_id == 0]),
        atol=1e-7,
    )


def test_do_recomputes_centered_regressor(confounded):
    """On FRESH rows (nothing cacheable) the centered regressor is re-derived
    from the current values: abduct at T=1 minus T=0 equals beta(x) exactly
    (e_hat(x) cancels only if the same live e_hat enters both), the e_hat term
    verifiably enters the latent, and counterfactual sampling under do
    round-trips through it.
    """
    df = confounded["draw"](2000, 4)
    flow = CausalFlowDAG(_misspecified_spec(True), seed=0)
    flow.fit(df, epochs=40, verbose=0, seed=0)
    fresh = confounded["draw"](300, 99)  # never seen in fit

    beta = flow.varying_coef("Y", fresh)
    u1 = flow.abduct(fresh.assign(T=1.0), seed=0)["Y"].values
    u0 = flow.abduct(fresh.assign(T=0.0), seed=0)["Y"].values
    np.testing.assert_allclose(u1 - u0, beta, rtol=0, atol=1e-5)

    # e_hat enters the latent: u(T=0) = h(y) + g(x) - beta * e_hat(x); compare
    # against the same evaluation with e_hat forced to 0 -> difference must be
    # exactly -beta * e_hat with e_hat from the flow's own T node (pmf)
    vals = flow._tensorize(fresh.assign(T=0.0))
    feats = flow._features(vals)
    nd = flow.nodes["Y"]
    e = flow.pmf(fresh, "T")[:, 1]
    with torch.no_grad():
        _, s_live = nd.theta_shift(
            feats, len(fresh), vc_ehat=flow._vc_ehat_live(nd, vals, len(fresh))
        )
        zero = {"T": torch.zeros(len(fresh))}
        _, s_zero = nd.theta_shift(feats, len(fresh), vc_ehat=zero)
    np.testing.assert_allclose((s_live - s_zero).numpy(), -beta * e, atol=1e-5)

    # counterfactual round-trip under do: push abducted u through do(T=1),
    # re-abduct the result at T=1 -> the same u comes back
    u = flow.abduct(fresh, seed=0)
    cf = flow.sample(u=u, do={"T": 1})
    u_back = flow.abduct(cf.assign(T=1.0), seed=0)["Y"].values
    np.testing.assert_allclose(u_back, u["Y"].values, atol=1e-3)

    # evaluating a centered term without its propensity must fail loudly
    with pytest.raises(RuntimeError, match="e_hat"):
        nd.theta_shift(feats, len(fresh), vc_ehat=None)


def test_dandl_centering_reduces_bias(confounded):
    """THE reason the feature exists (issue #30, Dandl et al. 2024): under
    strong confounding + a deliberately under-specified prognostic part, the
    centered VC must show materially lower bias in beta_hat than the uncentered
    one. Measured on this protocol (seeds 0/1/2): uncentered mean|beta_hat-tau|
    = 1.10-1.24 (the confounded misfit swallows the effect, mean beta_hat
    ~ -0.2 vs tau = -1), centered = 0.11-0.27 — a 5-10x reduction.
    """
    tau = confounded["truth"]["tau"]
    df = confounded["draw"](6000, 0)
    test = confounded["draw"](3000, 1000)
    flow_u = _fit(_misspecified_spec(False), df)
    flow_c = _fit(_misspecified_spec(True), df)
    bias_u = float(np.mean(np.abs(flow_u.varying_coef("Y", test) - tau)))
    bias_c = float(np.mean(np.abs(flow_c.varying_coef("Y", test) - tau)))
    assert bias_u > 0.5, f"trap not armed (uncentered bias {bias_u:.3f})"
    assert bias_c < 0.5 * bias_u, (bias_u, bias_c)


def test_centered_save_load_and_queries(confounded):
    df = confounded["draw"](1000, 6)
    flow = CausalFlowDAG(_misspecified_spec(True), seed=0)
    flow.fit(df, epochs=10, verbose=0, seed=0)
    assert torch.isfinite(flow.log_prob(df)).all()
    assert flow.sample(64, do={"T": 1}, seed=0).shape == (64, 3)
    psi = flow.scores(df, node="Y")
    assert {"X", "T"} <= set(psi.columns)


def test_centered_roundtrip_after_load(tmp_path, confounded):
    df = confounded["draw"](1000, 6)
    flow = CausalFlowDAG(_misspecified_spec(True), seed=0)
    flow.fit(df, epochs=10, verbose=0, seed=0)
    p = tmp_path / "c.pt"
    flow.save(p)
    flow2 = CausalFlowDAG.load(p)
    t = next(t for t in flow2.spec["Y"].terms if t.effect == "VC")
    assert t.center is True
    with torch.no_grad():
        np.testing.assert_allclose(
            flow2.log_prob(df.head(50)).numpy(),
            flow.log_prob(df.head(50)).detach().numpy(),
            atol=1e-6,
        )


def test_vc_oof_fit_reaches_the_stage_one_proxy(confounded):
    """vc_oof_fit is forwarded to the stage-1 out-of-fold proxy fits.

    The proxy uses Adam only when the treatment node is not all-``ls``
    (an all-``ls`` treatment takes the deterministic fit_classical path),
    so the treatment here carries a CS term. An unknown keyword must
    surface as a TypeError from that fit, which is what proves the
    forwarding rather than a silently ignored argument.
    """
    df = confounded["draw"](200, 0)
    spec = {
        "X": ContinuousNode(),
        "T": OrdinalNode(2, [CS("X")]),  # not all-ls -> the Adam proxy path
        "Y": ContinuousNode([LS("X"), VC(t="T", center=True, center_folds=2)]),
    }
    kw = dict(epochs=1, learning_rate=1e-2, batch_size=200, verbose=0)

    flow = CausalFlowDAG(spec, seed=0)
    flow.fit(df, **kw, vc_oof_fit={"epochs": 1, "batch_size": 64})
    info = flow.vc_center_info[("Y", "T")]
    assert info["folds"] == 2
    assert len(np.unique(info["fold_id"])) == 2

    with pytest.raises(TypeError):
        CausalFlowDAG(spec, seed=0).fit(df, **kw, vc_oof_fit={"not_a_kwarg": 1})
