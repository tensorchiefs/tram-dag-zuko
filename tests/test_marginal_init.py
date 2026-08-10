"""Tests for ``fit(marginal_init=True)`` — calibrated initialization of each
*unconditional* (root) node's transform to the empirical marginal.

What it guarantees:
- Bernstein roots start as the linear map of the pre-scaled domain onto the
  standard-logistic 5%/95% quantiles (±2.944), not zuko's ~2.5×-too-steep zero θ.
- Ordinal roots start with cutpoints reproducing the empirical class frequencies.
- It only touches unconditional `SimpleIntercept` roots — `ci` intercepts are
  left alone.
- It is a *pure init*: a marginal-init fit and a default fit converge to the same
  optimum (so the exact-MLE property is preserved).
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tramdag import LS, CausalFlowDAG, ContinuousNode, I, OrdinalNode
from tramdag.conditioners import ComplexIntercept, SimpleIntercept
from tramdag.transforms import BernsteinUT, ordinal_marginal_init_theta, ordinal_pmf

DATA = Path(__file__).resolve().parents[1] / "data"


# ------------------------------------------------------------------ fast
def test_bernstein_marginal_init_is_calibrated_logistic_map():
    """marginal_init θ maps the pre-scaled domain onto [logit .05, logit .95]."""
    ut = BernsteinUT()
    ut.set_range(-5.0, 5.0)  # range == bound -> _scale is identity
    theta = ut.marginal_init_theta()  # q=0.05 default
    x = torch.tensor([-5.0, 0.0, 5.0])
    z0, _ = ut.forward(theta.unsqueeze(0).expand(3, -1), x)
    logit05 = math.log(0.05) - math.log(0.95)  # -2.9444
    np.testing.assert_allclose(z0.detach().numpy(), [logit05, 0.0, -logit05], atol=1e-3)


def test_ordinal_marginal_init_reproduces_class_frequencies():
    """Cutpoints from class counts reproduce the empirical class PMF."""
    counts = np.array([50, 30, 15, 5])  # K=4 levels
    tt = ordinal_marginal_init_theta(counts)
    pmf = ordinal_pmf(tt.unsqueeze(0), torch.zeros(1))[0].numpy()
    np.testing.assert_allclose(pmf, counts / counts.sum(), atol=1e-4)


def _mixed_flow_and_df():
    """Flow with a Bernstein root, an ordinal root, and a continuous node whose
    parent enters as `ci` (so its intercept is a ComplexIntercept, not a root)."""
    spec = {
        "x1": ContinuousNode(),  # Bernstein root
        "y": OrdinalNode(levels=4),  # ordinal root
        "x2": ContinuousNode(terms=[I("x1")]),  # ci -> ComplexIntercept
    }
    torch.manual_seed(0)
    flow = CausalFlowDAG(spec)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=400),
            "y": rng.integers(0, 4, size=400).astype(float),
            "x2": rng.normal(size=400),
        }
    )
    return flow, df


def test_marginal_init_only_touches_unconditional_roots():
    flow, df = _mixed_flow_and_df()
    # sanity: the ci node really has a ComplexIntercept
    assert isinstance(flow.nodes["x2"].intercept, ComplexIntercept)
    assert isinstance(flow.nodes["x1"].intercept, SimpleIntercept)

    root_x1_before = flow.nodes["x1"].intercept.theta.detach().clone()
    root_y_before = flow.nodes["y"].intercept.theta.detach().clone()
    ci_before = {
        k: v.detach().clone()
        for k, v in flow.nodes["x2"].intercept.state_dict().items()
    }

    flow._set_ranges(df, marginal_init=True)

    # the two roots are now calibrated (changed from their zero init)...
    assert not torch.allclose(flow.nodes["x1"].intercept.theta.detach(), root_x1_before)
    assert not torch.allclose(flow.nodes["y"].intercept.theta.detach(), root_y_before)
    np.testing.assert_allclose(
        flow.nodes["x1"].intercept.theta.detach().numpy(),
        flow.nodes["x1"].ut.marginal_init_theta().numpy(),
        atol=1e-6,
    )
    # ...while the ci node's ComplexIntercept is untouched
    for k, v in flow.nodes["x2"].intercept.state_dict().items():
        assert torch.equal(v.detach(), ci_before[k]), f"ci param {k} changed"


def test_marginal_init_off_by_default_leaves_roots_at_zero():
    flow, df = _mixed_flow_and_df()
    flow._set_ranges(df)  # marginal_init defaults to False
    assert torch.allclose(
        flow.nodes["x1"].intercept.theta.detach(),
        torch.zeros_like(flow.nodes["x1"].intercept.theta),
    )


# ------------------------------------------------------------------ slow
@pytest.mark.slow
def test_marginal_init_is_pure_init_same_optimum():
    """A marginal-init fit and a default fit converge to the same NLL — proving
    it only moves the starting point, not the optimum."""
    obs = pd.read_csv(DATA / "vaca" / "obs.csv")[["x1", "x2"]]
    spec = {"x1": ContinuousNode(), "x2": ContinuousNode(terms=[LS("x1")])}

    def converged_nll(marginal_init):
        torch.manual_seed(0)
        flow = CausalFlowDAG(spec)
        flow.fit(
            obs,
            epochs=1500,
            learning_rate=1e-2,
            batch_size=512,
            verbose=0,
            schedule="plateau",
            plateau_patience=15,
            freeze_patience=60,
            marginal_init=marginal_init,
        )
        return sum(flow.nll(obs).values())

    np.testing.assert_allclose(converged_nll(True), converged_nll(False), atol=1e-2)
