"""Tests for the varying-coefficient shift term VC(on, *modifiers, penalty=)
(issue #28) — construction/validation, the LS nesting (exact and fitted), the
recovery acceptance bar on the heterogeneous-effect DGP, the read-out
identities, warm start, and serialization.

The recovery bar (corr(beta_hat, beta_true) >= 0.9) is the regression guard
against "expressive but unestimated": the unregularized ``CS(on, x...)``
reduced form measures ~0.5 on this task class (tramdag-simu PR #21), so the
bar is what the term exists to clear.
"""

# %% imports ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pytest
import torch

from tramdag import CS, LS, VC, CausalFlowDAG, ContinuousNode, I, OrdinalNode
from tramdag.spec import spec_from_dict, spec_to_dict, validate_and_sort


# %% private functions -----------------------------------------------------------------
def _vc_spec(penalty: float = 1.0) -> dict:
    """The DGP's in-class spec (affine sources: their marginals are
    irrelevant to Y's conditional because the joint NLL decomposes per node).
    """
    return {
        "X1": ContinuousNode([I(transform="affine")]),
        "X2": ContinuousNode([I(transform="affine")]),
        "X3": ContinuousNode([I(transform="affine")]),
        "T": OrdinalNode(2, [LS("X1"), LS("X2")]),
        "Y": ContinuousNode(
            [CS("X1", "X2", "X3"), VC("X2", "X3", penalty=penalty, t="T")]
        ),
    }


# %% public functions ------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_fitted(vc_hetero):
    """A briefly-fitted VC flow (module-scoped: shared by the identity and
    serialization tests; the accuracy bar has its own fit).
    """
    df = vc_hetero["draw"](1500, 100)
    flow = CausalFlowDAG(_vc_spec(), seed=0)
    flow.fit(
        df.iloc[:1300],
        df.iloc[1300:],
        epochs=60,
        learning_rate=1e-2,
        batch_size=512,
        verbose=0,
        seed=0,
    )
    return flow, vc_hetero


# ------------------------------------------------------------ spec / validation
def test_vc_constructor():
    t = VC("X2", "X3", penalty=2.5, t="T")
    assert (t.effect, t.parents, t.penalty) == ("VC", ("T", "X2", "X3"), 2.5)
    assert VC(t="T").penalty == 1.0  # the documented default
    with pytest.raises(
        ValueError, match=r"cannot be both the treatment \(t\) and a modifier"
    ):
        VC("T", t="T")  # on cannot be a modifier
    with pytest.raises(ValueError, match="penalty must be >= 0"):
        VC(penalty=-1.0, t="T")  # negative penalty


def test_vc_modifier_may_repeat_but_on_owns_its_edge():
    # modifier X2 also acts prognostically (the intended pattern) -> valid
    ok = {
        "X2": ContinuousNode(),
        "T": OrdinalNode(levels=2),
        "Y": ContinuousNode([CS("X2"), VC("X2", t="T")]),
    }
    assert validate_and_sort(ok)[-1] == "Y"
    # a second edge-owning term for T -> invalid (beta0 vs main effect unidentified)
    bad = {
        "X2": ContinuousNode(),
        "T": OrdinalNode(levels=2),
        "Y": ContinuousNode([LS("T"), VC("X2", t="T")]),
    }
    with pytest.raises(ValueError, match="more than one"):
        validate_and_sort(bad)


def test_vc_rejects_multilevel_ordinal_treatment():
    spec = {"T": OrdinalNode(levels=4), "Y": ContinuousNode([VC(t="T")])}
    with pytest.raises(ValueError, match="2-level"):
        validate_and_sort(spec)


def test_vc_modifiers_are_real_dag_edges():
    """Modifiers must topologically precede the node (they are parents)."""
    spec = {
        "T": OrdinalNode(levels=2),
        "Y": ContinuousNode([VC("M", t="T")]),
        "M": ContinuousNode(),
    }
    order = validate_and_sort(spec)
    assert order.index("M") < order.index("Y")


def test_to_matrix_vc_labels():
    m = CausalFlowDAG(_vc_spec(), seed=0).to_matrix()
    assert m.loc["T", "Y"] == "VC"
    assert m.loc["X2", "Y"] == "CS['X1', 'X2', 'X3']+VCm"  # prognostic + modifier
    assert m.loc["X3", "Y"] == "CS['X1', 'X2', 'X3']+VCm"


# ------------------------------------------------------------- exact LS nesting
def test_vc_without_modifiers_equals_ls_exactly():
    """VC(on) has no net — with beta0 set to the LS weight the two models give
    bit-identical log-probs (the nesting is exact, not approximate).
    """
    rng = np.random.default_rng(3)
    t = rng.integers(0, 2, 200).astype(float)
    y = 0.7 * t + rng.logistic(size=200)
    df = pd.DataFrame({"T": t, "Y": y})
    spec_vc = {"T": OrdinalNode(levels=2), "Y": ContinuousNode([VC(t="T")])}
    spec_ls = {"T": OrdinalNode(levels=2), "Y": ContinuousNode([LS("T")])}
    fv, fl = CausalFlowDAG(spec_vc, seed=0), CausalFlowDAG(spec_ls, seed=0)
    with torch.no_grad():
        # LS enters via the 2-column one-hot; only w[1]-w[0] is identified.
        # Set w = (0, 0.4) so it equals beta(x)*t with beta0 = 0.4.
        fl.nodes["Y"].shifts["T"].fc.weight.copy_(torch.tensor([[0.0, 0.4]]))
        fv.nodes["Y"].shifts["T"].beta0.fill_(0.4)
    assert torch.equal(fv.log_prob(df), fl.log_prob(df))


def test_fit_classical_rejects_vc(vc_hetero):
    flow = CausalFlowDAG(_vc_spec())
    with pytest.raises(ValueError, match="all-`ls`"):
        flow.fit_classical(vc_hetero["draw"](100, 1))


# --------------------------------------------------------------- read-out ident
def test_varying_coef_deterministic_and_y_free(small_fitted):
    flow, dgp = small_fitted
    new = dgp["draw"](300, 900)
    b1 = flow.varying_coef("Y", new)
    b2 = flow.varying_coef("Y", new.drop(columns=["Y", "T", "X1"]))  # modifiers only
    np.testing.assert_array_equal(b1, b2)
    assert b1.shape == (300,)
    assert b1.std() > 0


def test_varying_coef_equals_abduct_difference(small_fitted):
    """For a binary treatment, beta(x) must equal the abduct-difference
    u(x, T=1, y) - u(x, T=0, y) identically (issue #28 identity check).
    """
    flow, dgp = small_fitted
    new = dgp["draw"](300, 901)
    u1 = flow.abduct(new.assign(T=1.0), seed=0)["Y"].values
    u0 = flow.abduct(new.assign(T=0.0), seed=0)["Y"].values
    np.testing.assert_allclose(flow.varying_coef("Y", new), u1 - u0, rtol=0, atol=1e-5)


def test_beta_recentered_over_training_data(small_fitted):
    """After fit, b_theta is sum-to-zero over the training rows: the mean of
    varying_coef on the training data equals beta0.
    """
    flow, dgp = small_fitted
    train = dgp["draw"](1500, 100).iloc[:1300]
    beta0 = float(flow.nodes["Y"].shifts["T"].beta0)
    assert np.mean(flow.varying_coef("Y", train)) == pytest.approx(beta0, abs=1e-5)


def test_save_load_roundtrip_vc(tmp_path, small_fitted):
    flow, dgp = small_fitted
    new = dgp["draw"](200, 902)
    p = tmp_path / "vc.pt"
    flow.save(p)
    flow2 = CausalFlowDAG.load(p)
    vc = next(t for t in flow2.spec["Y"].terms if t.effect == "VC")
    assert vc.penalty == 1.0
    assert bool(flow2.nodes["Y"].shifts["T"].warm_started)  # buffer survives
    np.testing.assert_allclose(
        flow2.varying_coef("Y", new), flow.varying_coef("Y", new), atol=1e-7
    )
    assert torch.allclose(flow2.log_prob(new), flow.log_prob(new), atol=1e-6)


def test_serialization_roundtrip_spec():
    spec2 = spec_from_dict(spec_to_dict(_vc_spec(penalty=3.0)))
    t = next(t for t in spec2["Y"].terms if t.effect == "VC")
    assert t == VC("X2", "X3", penalty=3.0, t="T")


# ------------------------------------------------------------------- warm start
def test_warm_start_matches_classical_ls(vc_hetero):
    """beta0 after warm start equals the classical (L-BFGS) all-`ls` coefficient
    of the node's conditional; b_theta stays the zero function.
    """
    df = vc_hetero["draw"](2000, 100)
    flow = CausalFlowDAG(_vc_spec(), seed=0)
    flow._vc_warm_start(df)
    beta0 = float(flow.nodes["Y"].shifts["T"].beta0)

    ls_spec = {
        **{k: ContinuousNode([I(transform="affine")]) for k in ("X1", "X2", "X3")},
        "T": OrdinalNode(2, [LS("X1"), LS("X2")]),
        "Y": ContinuousNode([LS("X1"), LS("X2"), LS("X3"), LS("T")]),
    }
    ref = CausalFlowDAG(ls_spec, seed=0)
    ref.fit_classical(df, verbose=False)
    w = ref.nodes["Y"].shifts["T"].weight.detach().numpy()
    assert beta0 == pytest.approx(w[1] - w[0], abs=0.02)
    # the head is untouched: beta(x) is still constant (float32 mean noise only)
    assert flow.varying_coef("Y", df).std() < 1e-6
    # guarded: a second call must not re-run (beta0 unchanged after perturbation)
    with torch.no_grad():
        flow.nodes["Y"].shifts["T"].beta0.fill_(123.0)
    flow._vc_warm_start(df)
    assert float(flow.nodes["Y"].shifts["T"].beta0) == 123.0


# ----------------------------------------------- acceptance: fitted LS nesting
def test_nesting_large_penalty_matches_classical_ls(vc_hetero):
    """Acceptance (issue #28): with `penalty` large the head is shrunk to the
    zero function and the fitted beta0 matches the fit_classical LS coefficient.
    Warm start is disabled so the test is not trivially satisfied by it.
    """
    df = vc_hetero["draw"](4000, 100)
    spec = {
        **{k: ContinuousNode([I(transform="affine")]) for k in ("X1", "X2", "X3")},
        "T": OrdinalNode(2, [LS("X1"), LS("X2")]),
        "Y": ContinuousNode(
            [LS("X1"), LS("X2"), LS("X3"), VC("X2", "X3", penalty=1e7, t="T")]
        ),
    }
    flow = CausalFlowDAG(spec, seed=0)
    for ep, lr in [(1500, 1e-2), (500, 1e-3)]:
        flow.fit(
            df,
            epochs=ep,
            learning_rate=lr,
            batch_size=1024,
            verbose=0,
            seed=0,
            restore_best=False,
            vc_warm_start=False,
        )
    # the head is dead: beta(x) constant
    assert flow.varying_coef("Y", df).std() < 1e-3

    ls_spec = {
        **spec,
        "Y": ContinuousNode([LS("X1"), LS("X2"), LS("X3"), LS("T")]),
    }
    ref = CausalFlowDAG(ls_spec, seed=0)
    ref.fit_classical(df, verbose=False)
    w = ref.nodes["Y"].shifts["T"].weight.detach().numpy()
    beta0 = float(flow.nodes["Y"].shifts["T"].beta0)
    assert beta0 == pytest.approx(w[1] - w[0], abs=0.03)


# --------------------------------------------------- acceptance: recovery >= 0.9
def test_recovery_bar_on_hetero_dgp(vc_hetero):
    """THE acceptance bar (issue #28): corr(beta_hat, beta_true) >= 0.9 at
    n = 5000 on the heterogeneous-effect DGP, default penalty. The
    unregularized CS(on, x...) workaround measures ~0.5 on this task class —
    this test is the regression guard against 'expressive but unestimated'.
    """
    df = vc_hetero["draw"](5000, 100)
    train, val = df.iloc[:4500], df.iloc[4500:]
    test = vc_hetero["draw"](2000, 500)

    flow = CausalFlowDAG(_vc_spec(), seed=0)
    flow.fit(
        train,
        val,
        epochs=300,
        learning_rate=1e-2,
        batch_size=512,
        verbose=0,
        seed=0,
        restore_best=True,
    )
    beta_hat = flow.varying_coef("Y", test)
    corr = float(np.corrcoef(beta_hat, vc_hetero["beta"](test))[0, 1])
    assert corr >= 0.9, f"recovery corr {corr:.3f} < 0.9"
    # beta0 is the interpretable main effect: close to the true b0
    b0 = vc_hetero["truth"]["b0"]
    assert float(flow.nodes["Y"].shifts["T"].beta0) == pytest.approx(b0, abs=0.15)


# ------------------------------------------------------- continuous treatment
def test_vc_continuous_treatment():
    """VC is linear in a continuous x_on: shift = beta(m) * d, so
    u(d=2) - u(d=0) = 2 * beta(m) (evaluated via abduct, y fixed).
    """
    rng = np.random.default_rng(5)
    n = 400
    m = rng.normal(size=n)
    d = rng.normal(size=n)
    y = 0.5 * m + (0.3 + 0.2 * m) * d + rng.logistic(size=n)
    df = pd.DataFrame({"M": m, "D": d, "Y": y})
    spec = {
        "M": ContinuousNode([I(transform="affine")]),
        "D": ContinuousNode([I(transform="affine")]),
        "Y": ContinuousNode([CS("M"), VC("M", t="D")]),
    }
    flow = CausalFlowDAG(spec, seed=0)
    flow.fit(df, epochs=30, verbose=0, seed=0)
    assert torch.isfinite(flow.log_prob(df)).all()
    beta = flow.varying_coef("Y", df)
    u2 = flow.abduct(df.assign(D=2.0), seed=0)["Y"].values
    u0 = flow.abduct(df.assign(D=0.0), seed=0)["Y"].values
    np.testing.assert_allclose(2.0 * beta, u2 - u0, rtol=0, atol=1e-4)
    assert flow.sample(50, do={"D": 1.0}, seed=0).shape == (50, 3)
