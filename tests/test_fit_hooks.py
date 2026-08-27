"""Tests for fit()'s two hooks, ``optimizer=`` and ``callback=``.

The critical guard is `test_torch_plateau_scheduler_preserves_exact_mle`: a
learning-rate schedule attached through the hooks must NOT break the exact-MLE
property of all-`ls` models, which is checked against statsmodels on the inline
all-`ls` DGP (see conftest).
"""

# %% imports ---------------------------------------------------------------------------
import copy

import numpy as np
import pytest
import torch

from tramdag import LS, CausalFlowDAG, ContinuousNode, OrdinalNode


# %% private functions -----------------------------------------------------------------
def _two_node_spec():
    return {"x1": ContinuousNode(), "x2": ContinuousNode([LS("x1")])}


def _ls_spec():
    return {
        "x1": ContinuousNode(),
        "x2": ContinuousNode([LS("x1")]),
        "t": OrdinalNode(2, [LS("x1"), LS("x2")]),
        "y": OrdinalNode(4, [LS("x1"), LS("x2"), LS("t")]),
    }


# %% public functions ------------------------------------------------------------------
def test_fit_improves_and_records_train_nll(ls_chain):
    df = ls_chain["draw"](800, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    flow.calibrate(df)
    nll0 = sum(flow.nll(df).values())
    flow.fit(df, epochs=60, learning_rate=1e-2)
    nll1 = sum(flow.nll(df).values())
    assert np.isfinite(nll1)
    assert nll1 < nll0
    assert len(flow.history["train"]) == 60
    assert set(flow.history["train"][-1]) == {"x1", "x2"}


def test_epochs_is_required(ls_chain):
    flow = CausalFlowDAG(_two_node_spec())
    with pytest.raises(TypeError, match="epochs"):
        flow.fit(ls_chain["draw"](200, 0)[["x1", "x2"]])


def test_callback_runs_once_per_epoch_and_can_stop(ls_chain):
    """The callback sees the live flow and optimizer after each epoch (from 1);
    returning True ends the fit.
    """
    df = ls_chain["draw"](200, 0)[["x1", "x2"]]
    seen = []
    flow = CausalFlowDAG(_two_node_spec(), seed=0)

    def cb(f, epoch, opt):
        seen.append((f is flow, epoch, isinstance(opt, torch.optim.Adam)))
        return epoch == 3

    flow.fit(df, epochs=10, callback=cb)
    assert seen == [(True, 1, True), (True, 2, True), (True, 3, True)]
    assert len(flow.history["train"]) == 3


def test_user_optimizer_is_used_and_keeps_its_state(ls_chain):
    """``optimizer=`` replaces the default Adam; its lr, not ``learning_rate``,
    drives the fit, and its state survives into a second call.
    """
    df = ls_chain["draw"](400, 1)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    flow.calibrate(df)
    opt = torch.optim.SGD(flow.parameters(), lr=0.0)  # a zero step: nothing moves
    before = copy.deepcopy(flow.state_dict())
    flow.fit(df, epochs=2, learning_rate=1e-2, optimizer=opt)
    assert all(torch.equal(before[k], v) for k, v in flow.state_dict().items())
    opt = torch.optim.Adam(flow.parameters(), lr=1e-2)
    flow.fit(df, epochs=3, optimizer=opt)
    assert opt.state[flow.nodes["x2"].shifts["x1"].fc.weight]["step"] == 3 * 1


def test_restore_best_is_a_six_line_callback(ls_chain):
    """Best-validation snapshots, the former ``restore_best``, through the hook."""
    df = ls_chain["draw"](800, 2)[["x1", "x2"]]
    val = ls_chain["draw"](400, 3)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    best = {"nll": float("inf"), "state": None, "epoch": 0}

    def keep_best(f, epoch, opt):
        nll = sum(f.nll(val).values())
        if nll < best["nll"]:
            best.update(nll=nll, state=copy.deepcopy(f.state_dict()), epoch=epoch)

    flow.fit(df, epochs=40, learning_rate=1e-2, callback=keep_best)
    flow.load_state_dict(best["state"])
    assert sum(flow.nll(val).values()) == pytest.approx(best["nll"], rel=1e-6)
    assert 1 <= best["epoch"] <= 40


def test_torch_plateau_scheduler_preserves_exact_mle(ls_chain):
    """The headline guard: all-`ls` + torch's ReduceLROnPlateau through the
    hooks must still land on the classical MLE (outcome coefficients vs
    statsmodels on the same data).
    """
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    obs = ls_chain["draw"](2500, 3)
    flow = CausalFlowDAG(_ls_spec(), seed=3)
    X = flow.design_matrix(obs, "y", drop_first=True)
    res = OrderedModel(obs["y"].astype(int), X, distr="logit").fit(
        method="bfgs", disp=False
    )
    opt = torch.optim.Adam(flow.parameters(), lr=1e-2)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.3, patience=40, min_lr=1e-5
    )

    def step(f, epoch, opt):
        plateau.step(sum(f.history["train"][-1].values()))
        return opt.param_groups[0]["lr"] <= 1e-5 and epoch > 500

    flow.fit(obs, epochs=4000, batch_size=512, optimizer=opt, callback=step)
    coefs = flow.ls_coefficients()["y"]
    w_t = np.asarray(coefs["t"]).ravel()
    assert float(coefs["x1"][0]) == pytest.approx(res.params["x1"], abs=0.03)
    assert float(coefs["x2"][0]) == pytest.approx(res.params["x2"], abs=0.03)
    assert (w_t[1] - w_t[0]) == pytest.approx(res.params["t[1]"], abs=0.06)
    assert len(flow.history["train"]) < 4000  # the callback stopped it


def test_history_accumulates_across_fit_calls(ls_chain):
    """A second fit continues the record instead of replacing it."""
    df = ls_chain["draw"](200, 5)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    flow.fit(df, epochs=2, batch_size=100)
    flow.fit(df, epochs=3, batch_size=100)
    assert len(flow.history["train"]) == 5
