"""R3 validation: joint (multi-parent) terms capture interactions that the
additive decomposition cannot. We build a DGP whose latent shift is a *pure
interaction* x1*x2 (no main effects); a joint CS(x1,x2) can fit it, an additive
CS(x1)+CS(x2) cannot, and the joint model must reach a clearly lower NLL.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from tramdag import CS, CausalFlowDAG, ContinuousNode


def _interaction_df(n, seed=0):
    """x3 = u3 - x1*x2 with u3 standard-logistic — so the latent of x3 is
    h(x3) + shift with shift = x1*x2, a pure (mean-zero, non-additive) product.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    u = rng.uniform(1e-6, 1 - 1e-6, size=n)
    u3 = np.log(u) - np.log1p(-u)
    x3 = u3 - x1 * x2
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})



@pytest.mark.slow
def test_joint_cs_beats_additive_on_interaction(fit_x3_nll):
    df = _interaction_df(4000)
    train, val = df.iloc[:3500], df.iloc[3500:]
    joint = fit_x3_nll([CS("x1", "x2")], train, val)  # one net over (x1,x2)
    additive = fit_x3_nll([CS("x1"), CS("x2")], train, val)  # g1(x1)+g2(x2)
    # the additive model cannot represent x1*x2, so it must do clearly worse
    assert joint < additive - 0.05, (joint, additive)


def test_joint_cs_runs_and_decomposes():
    """Smoke test: a joint CS node evaluates a finite, shape-correct likelihood."""
    df = _interaction_df(64)
    spec = {
        "x1": ContinuousNode(),
        "x2": ContinuousNode(),
        "x3": ContinuousNode([CS("x1", "x2")]),
    }
    flow = CausalFlowDAG(spec, seed=0)
    lp = flow.node_log_prob(flow._tensorize(df))["x3"]
    assert lp.shape == (len(df),)
    assert torch.isfinite(lp).all()
