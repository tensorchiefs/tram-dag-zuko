"""Tests for the propensity-centered VC term (issue #30):
beta(x) * (t - e_hat(x)) with cross-fitted (out-of-fold) e_hat.

The stage-1 propensities are the caller's: ``VC(center="col")`` names a
column of the training frame holding one out-of-fold value per row, and fit
refuses a centered spec whose frame lacks it.
Acceptance: center=False is bit-identical to #28's VC (regression guard);
gradient isolation (no gradient reaches the treatment node from the outcome
loss); the Dandl reproduction (confounded DGP + deliberately under-specified
prognostic part: centering must materially reduce the bias of beta_hat —
measured 5-10x on this protocol); do()/sample() recompute t - e_hat under
intervention (never cached).
"""

# %% imports ---------------------------------------------------------------------------
import numpy as np
import pytest
import torch

from tramdag import CS, LS, VC, CausalFlowDAG, ContinuousNode, I, OrdinalNode
from tramdag.spec import spec_from_dict, spec_to_dict, validate_and_sort

# %% global variables ------------------------------------------------------------------
T_SPEC = {"X": ContinuousNode([I(transform="affine")]), "T": OrdinalNode(2, [LS("X")])}


# %% private functions -----------------------------------------------------------------
def _misspecified_spec(center) -> dict:
    return {
        **T_SPEC,
        # prognostic part deliberately under-specified (linear vs true x^2)
        "Y": ContinuousNode(
            [LS("X"), VC("X", center="ps" if center else False, t="T")]
        ),
    }


def _oof_propensity(df, folds=5, seed=0) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fitted P(T=1|X) from the classical fit of the treatment spec.

    Six lines — what a user merges into the frame for ``VC(center=)``; each
    fold's rows are predicted by a model that never saw them.
    """
    fold_id = np.random.default_rng(seed).permutation(len(df)) % folds
    e = np.empty(len(df))
    for j in range(folds):
        proxy = CausalFlowDAG(T_SPEC, seed=0)
        proxy.fit_classical(df.iloc[fold_id != j][["X", "T"]])
        e[fold_id == j] = proxy.pmf(df.iloc[fold_id == j], "T")[:, 1]
    return e, fold_id


def _fit(spec, df, epochs=250):
    train = df.iloc[:5400]
    if any(t.effect == "VC" and t.center for t in spec["Y"].terms):
        train = train.assign(ps=_oof_propensity(train)[0])
    flow = CausalFlowDAG(spec, seed=0)
    flow.fit(train, epochs=epochs, learning_rate=1e-2, batch_size=512, seed=0)
    return flow


# %% public functions ------------------------------------------------------------------
def test_center_validation():
    spec = {"D": ContinuousNode(), "Y": ContinuousNode([VC(center="ps", t="D")])}
    with pytest.raises(ValueError, match="binary ordinal"):
        validate_and_sort(spec)  # continuous treatment cannot center
    chained = {
        "X": ContinuousNode(),
        "A": OrdinalNode(2, [LS("X")]),
        "T": OrdinalNode(2, [VC("X", center="ps_a", t="A")]),
        "Y": ContinuousNode([VC("X", center="ps_t", t="T")]),
    }
    with pytest.raises(ValueError, match="chained"):
        validate_and_sort(chained)
    # center names a COLUMN now: the pre-column spelling refuses loudly
    legacy = {
        "X": ContinuousNode(),
        "T": OrdinalNode(2, [LS("X")]),
        "Y": ContinuousNode([VC("X", center=True, t="T")]),
    }
    with pytest.raises(ValueError, match="COLUMN"):
        validate_and_sort(legacy)
    collides = {
        "X": ContinuousNode(),
        "T": OrdinalNode(2, [LS("X")]),
        "Y": ContinuousNode([VC("X", center="X", t="T")]),
    }
    with pytest.raises(ValueError, match="collides"):
        validate_and_sort(collides)


def test_center_serialization_roundtrip():
    spec = {
        "X": ContinuousNode(),
        "T": OrdinalNode(2, [LS("X")]),
        "Y": ContinuousNode([VC("X", center="ps", t="T")]),
    }
    round_tripped = spec_from_dict(spec_to_dict(spec))
    t = next(t for t in round_tripped["Y"].terms if t.effect == "VC")
    assert t.center == "ps"


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
        flow.fit(df.iloc[:1000], epochs=15, seed=3)
        return flow

    a = fit_with(VC("X2", "X3", t="T"))
    b = fit_with(VC("X2", "X3", center=False, t="T"))
    assert VC("X2", t="T") == VC("X2", center=False, t="T")  # Term equality
    for (ka, pa), (kb, pb) in zip(
        a.state_dict().items(), b.state_dict().items(), strict=True
    ):
        assert ka == kb
        assert torch.equal(pa, pb), ka


def test_fit_requires_the_out_of_fold_propensities(confounded):
    """A centered spec whose frame lacks the propensity column fails loudly."""
    df = confounded["draw"](300, 1)
    flow = CausalFlowDAG(_misspecified_spec(True), seed=0)
    with pytest.raises(ValueError, match="propensity column 'ps'"):
        flow.fit(df, epochs=1)


def test_oof_helper_is_genuinely_out_of_fold(confounded):
    """The propensities a user feeds in are OOF quantities: each fold's values
    reproduce a proxy fitted WITHOUT that fold, and they differ from the
    in-sample full-data fit.
    """
    df = confounded["draw"](1200, 2)
    e_oof, fold_id = _oof_propensity(df)
    assert set(fold_id) == set(range(5))
    proxy = CausalFlowDAG(T_SPEC, seed=0)
    proxy.fit_classical(df.iloc[fold_id != 0][["X", "T"]])
    np.testing.assert_allclose(
        e_oof[fold_id == 0], proxy.pmf(df.iloc[fold_id == 0], "T")[:, 1], atol=1e-7
    )
    full = CausalFlowDAG(T_SPEC, seed=0)
    full.fit_classical(df[["X", "T"]])
    assert np.abs(e_oof - full.pmf(df, "T")[:, 1]).max() > 1e-4


def test_gradient_isolation(confounded):
    """With center=True the treatment node's parameters receive ZERO gradient
    from the outcome-node loss — on the live (inference) e_hat path and, a
    fortiori, on the frozen-OOF training path.
    """
    df = confounded["draw"](800, 1)
    flow = CausalFlowDAG(_misspecified_spec(True), seed=0)
    flow.fit(df.assign(ps=_oof_propensity(df)[0]), epochs=3, seed=0)
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


def test_do_recomputes_centered_regressor(confounded):
    """On FRESH rows (nothing cacheable) the centered regressor is re-derived
    from the current values: abduct at T=1 minus T=0 equals beta(x) exactly
    (e_hat(x) cancels only if the same live e_hat enters both), the e_hat term
    verifiably enters the latent, and counterfactual sampling under do
    round-trips through it.
    """
    df = confounded["draw"](2000, 4)
    flow = CausalFlowDAG(_misspecified_spec(True), seed=0)
    flow.fit(df.assign(ps=_oof_propensity(df)[0]), epochs=40, seed=0)
    fresh = confounded["draw"](300, 99)  # never seen in fit

    beta = flow.varying_coef(fresh, "Y")
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
            feats | flow._side_feats(nd, vals, len(fresh)), len(fresh)
        )
        _, s_zero = nd.theta_shift(feats | {"ps": torch.zeros(len(fresh))}, len(fresh))
    np.testing.assert_allclose((s_live - s_zero).numpy(), -beta * e, atol=1e-5)

    # counterfactual round-trip under do: push abducted u through do(T=1),
    # re-abduct the result at T=1 -> the same u comes back
    u = flow.abduct(fresh, seed=0)
    cf = flow.sample(u=u, do={"T": 1})
    u_back = flow.abduct(cf.assign(T=1.0), seed=0)["Y"].values
    np.testing.assert_allclose(u_back, u["Y"].values, atol=1e-3)

    # evaluating a centered term without its propensity must fail loudly
    with pytest.raises(RuntimeError, match="propensity"):
        nd.theta_shift(feats, len(fresh))


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
    bias_u = float(np.mean(np.abs(flow_u.varying_coef(test, "Y") - tau)))
    bias_c = float(np.mean(np.abs(flow_c.varying_coef(test, "Y") - tau)))
    assert bias_u > 0.5, f"trap not armed (uncentered bias {bias_u:.3f})"
    assert bias_c < 0.5 * bias_u, (bias_u, bias_c)


def test_centered_save_load_and_queries(tmp_path, confounded):
    df = confounded["draw"](1000, 6)
    flow = CausalFlowDAG(_misspecified_spec(True), seed=0)
    flow.fit(df.assign(ps=_oof_propensity(df)[0]), epochs=10, seed=0)
    assert torch.isfinite(flow.log_prob(df)).all()
    assert flow.sample(64, do={"T": 1}, seed=0).shape == (64, 3)
    psi = flow.scores(df, node="Y")
    assert {"X", "T"} <= set(psi.columns)

    p = tmp_path / "c.pt"
    flow.save(p)
    flow2 = CausalFlowDAG.load(p)
    t = next(t for t in flow2.spec["Y"].terms if t.effect == "VC")
    assert t.center == "ps"
    with torch.no_grad():
        np.testing.assert_allclose(
            flow2.log_prob(df.head(50)).numpy(),
            flow.log_prob(df.head(50)).detach().numpy(),
            atol=1e-6,
        )


def test_propensities_must_be_probabilities(confounded):
    """A centered term takes P(t=1|pa_t): outside [0, 1] is a caller error."""
    df = confounded["draw"](200, 8)
    flow = CausalFlowDAG(_misspecified_spec(True), seed=0)
    bad = _oof_propensity(df)[0].copy()
    bad[0] = 1.4
    with pytest.raises(ValueError, match="probabilities"):
        flow.fit(df.assign(ps=bad), epochs=1)


def test_centered_fit_with_managed_validation(confounded):
    """Fit-managed validation composes with a centered spec: the frozen OOF
    column trains, the per-epoch validation NLL falls back to the LIVE
    propensity (validation frames carry no OOF column by design), and
    EarlyStopping restores best weights.
    """
    from tramdag.callbacks import EarlyStopping

    df = confounded["draw"](800, 3)
    val = confounded["draw"](300, 30)  # no 'ps' column: live e_hat by design
    flow = CausalFlowDAG(_misspecified_spec(True), seed=0)
    early = EarlyStopping()
    flow.fit(
        df.assign(ps=_oof_propensity(df)[0]),
        epochs=8,
        seed=0,
        validation_data=val,
        callbacks=early,
    )
    assert len(flow.history["val"]) == 8
    assert np.isfinite(early.best_nll)
    # restored best: the flow's val NLL equals the callback's best
    assert sum(flow.nll(val).values()) == pytest.approx(early.best_nll, rel=1e-6)


def test_stray_propensity_column_is_ignored_on_an_uncentered_fit(confounded):
    """Pin: a stray 'ps' column on an UNCENTERED spec is silently ignored.

    Frames often carry extra columns; only a declared center= reads one.
    Forgetting center= therefore fits uncentered — documented behavior.
    """
    df = confounded["draw"](400, 5)
    plain = CausalFlowDAG(_misspecified_spec(False), seed=0)
    with_stray = CausalFlowDAG(_misspecified_spec(False), seed=0)
    plain.fit(df, epochs=2, seed=0)
    with_stray.fit(df.assign(ps=0.5), epochs=2, seed=0)
    for (ka, pa), (kb, pb) in zip(
        plain.state_dict().items(), with_stray.state_dict().items(), strict=True
    ):
        assert ka == kb
        assert torch.equal(pa, pb), ka
