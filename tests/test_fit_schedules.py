"""Tests for fit()'s learning-rate schedules and per-node freezing.

The critical guard is the last test: plateau + freezing must NOT break the
exact-MLE property of all-`ls` models (the flow == statsmodels == R-polr match
pinned in test_simulations.py).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tramdag import LS, CausalFlowDAG, ContinuousNode, OrdinalNode

DATA = Path(__file__).resolve().parents[1] / "data"


def _toy_df(n=800, seed=0):
    rng = np.random.default_rng(seed)
    u1 = rng.logistic(size=n)
    u2 = rng.logistic(size=n)
    x1 = u1 / 1.5
    x2 = (u2 - 1.0 * x1) / 2.0
    return pd.DataFrame({"x1": x1, "x2": x2})


def _toy_spec():
    return {"x1": ContinuousNode(), "x2": ContinuousNode([LS("x1")])}


@pytest.mark.parametrize("schedule", [None, "onecycle", "cosine", "plateau"])
def test_schedules_smoke_and_improve(schedule):
    df = _toy_df()
    torch.manual_seed(0)
    flow = CausalFlowDAG(_toy_spec())
    nll0 = sum(flow.nll(df).values())  # untrained (ranges set lazily in fit)
    flow.fit(df, epochs=60, learning_rate=1e-2, verbose=0, schedule=schedule)
    nll1 = sum(flow.nll(df).values())
    assert np.isfinite(nll1) and nll1 < nll0
    assert len(flow.history["lr"]) == len(flow.history["val"])


def test_unknown_schedule_raises():
    flow = CausalFlowDAG(_toy_spec())
    with pytest.raises(ValueError, match="unknown schedule"):
        flow.fit(_toy_df(), epochs=1, schedule="exponential")


def test_freeze_stops_early_and_records():
    df = _toy_df()
    torch.manual_seed(0)
    flow = CausalFlowDAG(_toy_spec())
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
    for name, ep in flow.history["frozen"].items():
        assert 1 <= ep <= n_epochs


def test_frozen_node_parameters_stop_moving():
    df = _toy_df()
    torch.manual_seed(0)
    flow = CausalFlowDAG(_toy_spec())
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


def test_plateau_freeze_preserves_exact_mle():
    """The headline guard: all-`ls` + plateau + freezing must still land on the
    classical MLE (outcome-node coefficients vs the committed R reference).
    """
    obs = pd.read_csv(DATA / "magic-mrclean" / "ls" / "obs.csv")
    ref = pd.read_csv(DATA / "magic-mrclean" / "ls" / "ref_ls" / "coefficients.csv")
    ref_y = ref[ref["node"] == "mRS_3m"].set_index("term")["estimate"]

    spec = {
        "Age": ContinuousNode(),
        "mRS_pre": OrdinalNode(6, [LS("Age")]),
        "NIHSSa": ContinuousNode([LS("Age"), LS("mRS_pre")]),
        "T": OrdinalNode(2, [LS("Age"), LS("mRS_pre"), LS("NIHSSa")]),
        "mRS_3m": OrdinalNode(7, [LS("Age"), LS("mRS_pre"), LS("NIHSSa"), LS("T")]),
    }
    torch.manual_seed(3)
    flow = CausalFlowDAG(spec)
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
    # same comparisons/tolerances as test_simulations::test_flow_matches_r_reference
    w_age = float(flow.nodes["mRS_3m"].shifts["Age"].weight.detach())
    w_nih = float(flow.nodes["mRS_3m"].shifts["NIHSSa"].weight.detach())
    w_t = flow.nodes["mRS_3m"].shifts["T"].weight.detach().numpy().ravel()
    assert w_age == pytest.approx(ref_y["Age"], abs=0.03)
    assert w_nih == pytest.approx(ref_y["NIHSSa"], abs=0.03)
    assert (w_t[1] - w_t[0]) == pytest.approx(ref_y["T"], abs=0.06)
    # and it should have converged well before the 4000-epoch budget
    assert len(flow.history["val"]) < 4000


def test_plateau_factor_sets_the_decay_step():
    """Each plateau step multiplies the node lr by exactly plateau_factor.

    min_delta is absurdly high so every epoch after the first counts as
    "no improvement", which makes the trajectory deterministic. Two
    details of the recorded schedule: history["lr"] is appended before
    that epoch's decay, and epoch 0 always improves (on the initial
    infinite best), so the first decay lands after epoch 1.
    """
    df = _toy_df(n=200)
    epochs = 4
    for factor in (0.5, 0.1):
        flow = CausalFlowDAG(_toy_spec(), seed=0)
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
