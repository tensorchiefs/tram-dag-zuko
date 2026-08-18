"""R3b validation: additive complex intercept.

Separate `I` terms reshape the transform *additively* (one network per parent,
summed in unconstrained coefficient space) — distinct from a single joint
`I("a","b")`. Because a complex intercept changes the *shape* (e.g. scale) of the
conditional, it captures heteroscedasticity that a complex *shift* (location
only) cannot.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from tramdag import CS, CausalFlowDAG, ContinuousNode, I

# --------------------------------------------------------------- structure


def test_additive_ci_runs_and_finite():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.normal(size=64) for c in ["x1", "x2", "x3"]})
    flow = CausalFlowDAG(
        {
            "x1": ContinuousNode(),
            "x2": ContinuousNode(),
            "x3": ContinuousNode([I("x1", "x2", allow_interaction=False)]),
        },
        seed=0,
    )
    lp = flow.node_log_prob(flow._tensorize(df))["x3"]
    assert lp.shape == (64,) and torch.isfinite(lp).all()


# --------------------------------------------------------------- validation
def _scale_df(n, seed=0):
    """x3 = sigma(x1,x2) * u3 with log-sigma = 0.5*(x1+x2): the *spread* of x3
    varies with the parents (additively), the mean does not. A shift can't model
    this; a complex intercept can.
    """
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    u = rng.uniform(1e-6, 1 - 1e-6, size=n)
    u3 = np.log(u) - np.log1p(-u)
    x3 = np.exp(0.5 * (x1 + x2)) * u3
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})


@pytest.mark.slow
def test_additive_ci_beats_additive_shift_on_heteroscedastic(fit_x3_nll):
    df = _scale_df(4000)
    train, val = df.iloc[:3500], df.iloc[3500:]
    ci = fit_x3_nll(
        [I("x1", "x2", allow_interaction=False)], train, val
    )  # reshape per parent
    shift = fit_x3_nll([CS("x1"), CS("x2")], train, val)  # location only
    assert ci < shift - 0.05, (ci, shift)
