"""Fixtures shared by the test modules.

Two kinds of fixture live here: helpers that are provably identical
across files, and the three **inline DGPs** the framework tests measure
against. The DGPs are numpy-only and deliberately independent of the flow
implementation — they are the ground truth. They live here (and not in a
simulation package) because the framework must be testable on its own:
the research generators and their frozen CSVs belong to ``experiments/``.

Each DGP fixture gives a dict, so a test names what it takes:

- ``draw(n, seed)`` — a fresh sample, one column per variable
- ``truth`` — the DGP's coefficients, the values a fit must recover
- ``beta(df)`` — the pointwise true effect function (VC DGPs only)

Specs stay in the test modules: those pin the one syntax variant a
property needs, and sharing them would couple unrelated acceptance bars.
"""

# %% imports ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pytest

from tramdag import CausalFlowDAG, ContinuousNode

# %% global variables ------------------------------------------------------------------
# %% DGP 1: all-ls ---------------------------------------------------------------------
# Every conditional is an exact linear-shift transformation model, so the
# outcome node is a proportional-odds model and the flow's MLE must equal
# the classical one (statsmodels / R polr).
LS_CHAIN_TRUTH = {
    "h_scale_x1": 1.5,  # h(x1) = 1.5 * x1
    "h_scale_x2": 2.0,  # h(x2) = 2.0 * x2
    "beta_x2_x1": 1.2,  # x2 <- x1
    "beta_t_x1": 0.5,  # t  <- x1
    "beta_t_x2": 0.5,  # t  <- x2
    "beta_y_x1": 0.4,  # y  <- x1
    "beta_y_x2": 0.6,  # y  <- x2
    "beta_y_t": -0.8,  # y  <- t  (w[1] - w[0] of the one-hot)
    "cutpoints_y": (-1.5, 0.0, 1.5),  # 4 ordinal levels
}

# %% DGP 2: heterogeneous effect -------------------------------------------------------
# The outcome conditional is exactly a transformation model with a
# treatment effect that varies with the covariates, so a VC term is
# in-class and its read-out is scorable against beta(x). X2 is
# deliberately both a confounder and an effect modifier — the case where
# an unregularized CS reduced form fails hardest.
VC_HETERO_TRUTH = {
    "b0": -1.0,  # beta(x) = b0 + b2*X2 + b3*X3
    "b2": 0.8,
    "b3": -0.6,
    "h_scale": 2.0,  # h(y) = 2 y
    "g": "0.5*X1^2 + X2 - 0.5*X3",
    "propensity": "sigmoid(0.4*X1 + 0.4*X2)",
}

# %% DGP 3: confounded + nonlinear -----------------------------------------------------
# One covariate, strong confounding and a quadratic prognostic part. Fitted
# with a linear prognostic term the misfit correlates with the propensity,
# which is the configuration propensity centering exists for.
CONFOUNDED_TRUTH = {
    "tau": -1.0,  # constant effect on the latent scale
    "propensity": "sigmoid(2*X)",
    "g": "1.2*X^2",
    "h_scale": 2.0,
}


# %% private functions -----------------------------------------------------------------
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _draw_ls_chain(n: int, seed: int) -> pd.DataFrame:
    """Draw the all-``ls`` chain ``x1 -> x2 -> t -> y`` (plus x1/x2 -> y).

    ``x1`` and ``x2`` are continuous with affine transforms, ``t`` is a
    binary and ``y`` a 4-level ordinal node. Both ordinal conditionals
    follow the flow convention ``P(Y <= k) = sigmoid(theta_k - shift)``,
    so a fit recovers the coefficients above with their signs.
    """
    tr = LS_CHAIN_TRUTH
    rng = np.random.default_rng(seed)
    x1 = rng.logistic(size=n) / tr["h_scale_x1"]
    x2 = (rng.logistic(size=n) - tr["beta_x2_x1"] * x1) / tr["h_scale_x2"]
    shift_t = tr["beta_t_x1"] * x1 + tr["beta_t_x2"] * x2
    t = (rng.logistic(size=n) > -shift_t).astype(int)  # P(t=1) = sigmoid(shift_t)
    shift_y = tr["beta_y_x1"] * x1 + tr["beta_y_x2"] * x2 + tr["beta_y_t"] * t
    cdf = _sigmoid(np.asarray(tr["cutpoints_y"])[None, :] - shift_y[:, None])
    y = (rng.uniform(size=n)[:, None] > cdf).sum(axis=1)
    return pd.DataFrame({"x1": x1, "x2": x2, "t": t, "y": y})


def _draw_vc_hetero(n: int, seed: int) -> pd.DataFrame:
    """Draw the heterogeneous-effect DGP with a confounded treatment."""
    tr = VC_HETERO_TRUTH
    rng = np.random.default_rng(seed)
    x1, x2, x3 = (rng.normal(size=n) for _ in range(3))
    t = (rng.logistic(size=n) > -(0.4 * x1 + 0.4 * x2)).astype(float)
    g = 0.5 * x1**2 + x2 - 0.5 * x3
    beta = tr["b0"] + tr["b2"] * x2 + tr["b3"] * x3
    y = (rng.logistic(size=n) - g - beta * t) / tr["h_scale"]
    return pd.DataFrame({"X1": x1, "X2": x2, "X3": x3, "T": t, "Y": y})


def _beta_vc_hetero(df) -> np.ndarray:
    """Give the true effect function on the latent (log-odds) scale."""
    tr = VC_HETERO_TRUTH
    return (
        tr["b0"]
        + tr["b2"] * np.asarray(df["X2"], dtype=float)
        + tr["b3"] * np.asarray(df["X3"], dtype=float)
    )


def _draw_confounded(n: int, seed: int) -> pd.DataFrame:
    """Draw the strongly-confounded DGP with a quadratic prognostic part."""
    tr = CONFOUNDED_TRUTH
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = (rng.logistic(size=n) > -2.0 * x).astype(float)
    y = (rng.logistic(size=n) - 1.2 * x**2 - tr["tau"] * t) / tr["h_scale"]
    return pd.DataFrame({"X": x, "T": t, "Y": y})


# %% public functions ------------------------------------------------------------------
@pytest.fixture(scope="session")
def ls_chain():
    """The all-``ls`` DGP: exact-MLE and classical-agreement tests."""
    return {"draw": _draw_ls_chain, "truth": LS_CHAIN_TRUTH}


@pytest.fixture(scope="session")
def vc_hetero():
    """The heterogeneous-effect DGP: VC recovery, read-out and nesting."""
    return {
        "draw": _draw_vc_hetero,
        "beta": _beta_vc_hetero,
        "truth": VC_HETERO_TRUTH,
    }


@pytest.fixture(scope="session")
def confounded():
    """The confounded DGP with a prognostic misfit: centering tests."""
    return {"draw": _draw_confounded, "truth": CONFOUNDED_TRUTH}


# %% helpers ---------------------------------------------------------------------------
@pytest.fixture
def fit_x3_nll():
    """Fit ``x1 -> x3 <- x2`` with the given terms; give x3's validation NLL.

    The budget (300 epochs at lr 1e-2, batch 512) is shared by the joint-vs-
    additive and additive-intercept comparisons, which only read the NLL
    difference between two fits of the same shape.
    """

    def _fit(terms, train, val) -> float:
        flow = CausalFlowDAG(
            {
                "x1": ContinuousNode(),
                "x2": ContinuousNode(),
                "x3": ContinuousNode(terms),
            },
            seed=0,
        )
        flow.fit(train, val, epochs=300, learning_rate=1e-2, batch_size=512, verbose=0)
        return flow.nll(val)["x3"]

    return _fit
