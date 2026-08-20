"""Tests for fit()'s learning-rate schedules and per-node freezing.

The critical guard is `test_plateau_freeze_preserves_exact_mle`: plateau
decay and per-node freezing must NOT break the exact-MLE property of
all-`ls` models, which is checked against statsmodels on the inline
all-`ls` DGP (see conftest).
"""

import numpy as np
import pytest
import torch

from tramdag import LS, CausalFlowDAG, ContinuousNode, OrdinalNode


def _two_node_spec():
    return {"x1": ContinuousNode(), "x2": ContinuousNode([LS("x1")])}


def _ls_spec():
    return {
        "x1": ContinuousNode(),
        "x2": ContinuousNode([LS("x1")]),
        "t": OrdinalNode(2, [LS("x1"), LS("x2")]),
        "y": OrdinalNode(4, [LS("x1"), LS("x2"), LS("t")]),
    }


@pytest.mark.parametrize("schedule", [None, "plateau"])
def test_schedules_smoke_and_improve(ls_chain, schedule):
    df = ls_chain["draw"](800, 0)[["x1", "x2"]]
    torch.manual_seed(0)
    flow = CausalFlowDAG(_two_node_spec())
    nll0 = sum(flow.nll(df).values())  # untrained (ranges set lazily in fit)
    flow.fit(df, epochs=60, learning_rate=1e-2, verbose=0, schedule=schedule)
    nll1 = sum(flow.nll(df).values())
    assert np.isfinite(nll1) and nll1 < nll0
    assert len(flow.history["lr"]) == len(flow.history["val"])


def test_unknown_schedule_raises(ls_chain):
    flow = CausalFlowDAG(_two_node_spec())
    with pytest.raises(ValueError, match="unknown schedule"):
        flow.fit(
            ls_chain["draw"](200, 0)[["x1", "x2"]], epochs=1, schedule="exponential"
        )


def test_freeze_stops_early_and_records(ls_chain):
    df = ls_chain["draw"](800, 1)[["x1", "x2"]]
    torch.manual_seed(0)
    flow = CausalFlowDAG(_two_node_spec())
    flow.fit(
        df,
        epochs=3000,
        learning_rate=1e-2,
        batch_size=256,
        verbose=0,
        schedule="plateau",
        freeze_patience=25,
    )
    n_epochs = len(flow.history["val"])
    assert n_epochs < 3000, "expected early exit once all nodes froze"
    assert set(flow.history["frozen"]) == {"x1", "x2"}
    for _name, ep in flow.history["frozen"].items():
        assert 1 <= ep <= n_epochs


def test_a_fresh_fit_call_unfreezes(ls_chain):
    df = ls_chain["draw"](800, 2)[["x1", "x2"]]
    torch.manual_seed(0)
    flow = CausalFlowDAG(_two_node_spec())
    # freeze aggressively so x1 (a fast source node) freezes mid-run
    flow.fit(
        df,
        epochs=1500,
        learning_rate=1e-2,
        batch_size=256,
        verbose=0,
        freeze_patience=10,
    )
    if len(flow.history.get("frozen", {})) == 0:
        pytest.skip("nothing froze within the budget (unexpected but not a bug)")
    name = next(iter(flow.history["frozen"]))
    snap = {k: v.clone() for k, v in flow.nodes[name].state_dict().items()}
    flow.fit(
        df, epochs=5, learning_rate=1e-2, batch_size=256, verbose=0
    )  # fresh call, no freeze
    moved = any(
        not torch.equal(snap[k], v) for k, v in flow.nodes[name].state_dict().items()
    )
    assert moved, "sanity: a fresh fit call must unfreeze (state is per-call)"


def test_plateau_freeze_preserves_exact_mle(ls_chain):
    """The headline guard: all-`ls` + plateau + freezing must still land on
    the classical MLE (outcome coefficients vs statsmodels on the same data).
    """
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    obs = ls_chain["draw"](2500, 3)
    torch.manual_seed(3)
    flow = CausalFlowDAG(_ls_spec())
    X = flow.design_matrix(obs, "y", drop_first=True)
    res = OrderedModel(obs["y"].astype(int), X, distr="logit").fit(
        method="bfgs", disp=False
    )
    flow.fit(
        obs,
        epochs=4000,
        learning_rate=1e-2,
        batch_size=512,
        verbose=0,
        schedule="plateau",
        plateau_patience=40,
        freeze_patience=200,
    )
    coefs = flow.ls_coefficients()["y"]
    w_t = np.asarray(coefs["t"]).ravel()
    assert float(coefs["x1"][0]) == pytest.approx(res.params["x1"], abs=0.03)
    assert float(coefs["x2"][0]) == pytest.approx(res.params["x2"], abs=0.03)
    assert (w_t[1] - w_t[0]) == pytest.approx(res.params["t[1]"], abs=0.06)
    # and it should have converged well before the 4000-epoch budget
    assert len(flow.history["val"]) < 4000


def test_plateau_factor_sets_the_decay_step(ls_chain):
    """Each plateau step multiplies the node lr by exactly plateau_factor.

    min_delta is absurdly high so every epoch after the first counts as
    "no improvement", which makes the trajectory deterministic. Two
    details of the recorded schedule: history["lr"] is appended before
    that epoch's decay, and epoch 0 always improves (on the initial
    infinite best), so the first decay lands after epoch 1.
    """
    df = ls_chain["draw"](200, 4)[["x1", "x2"]]
    epochs = 4
    for factor in (0.5, 0.1):
        flow = CausalFlowDAG(_two_node_spec(), seed=0)
        flow.fit(
            df,
            epochs=epochs,
            learning_rate=1e-1,
            batch_size=200,
            verbose=0,
            schedule="plateau",
            plateau_patience=1,
            min_delta=1e9,
            plateau_factor=factor,
        )
        expected = [1e-1] + [1e-1 * factor**k for k in range(epochs - 1)]
        assert np.allclose(flow.history["lr"], expected), (
            factor,
            flow.history["lr"],
            expected,
        )
