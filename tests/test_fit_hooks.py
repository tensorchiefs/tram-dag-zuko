"""Tests for fit()'s hooks: ``optimizer=`` and the three callback lists.

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
from tramdag.callbacks import Logger, PerNodePlateau, RestoreBest, per_node_adam


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

    flow.fit(df, epochs=10, after_epoch_callbacks=cb)
    assert seen == [(True, 1, True), (True, 2, True), (True, 3, True)]
    assert len(flow.history["train"]) == 3


def test_callback_lists_all_run_and_any_stops(ls_chain):
    """Every after-epoch callback runs even on the stop epoch (no
    short-circuit), and the before/after hooks fire once around the loop.
    """
    df = ls_chain["draw"](200, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    calls = []

    def stopper(f, epoch, opt):
        calls.append(("stop?", epoch))
        return epoch == 2

    def logger(f, epoch, opt):
        calls.append(("log", epoch))

    flow.fit(
        df,
        epochs=10,
        before_fit_callbacks=lambda f, opt: calls.append(("before", 0)),
        after_epoch_callbacks=[stopper, logger],
        after_fit_callbacks=lambda f, opt: calls.append(("after", 0)),
    )
    assert calls == [
        ("before", 0),
        ("stop?", 1),
        ("log", 1),
        ("stop?", 2),
        ("log", 2),  # the logger still ran on the stop epoch
        ("after", 0),
    ]


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


def test_restore_best_matches_the_manual_six_line_callback(ls_chain):
    """``callbacks.RestoreBest`` lands exactly where the manual snapshot
    recipe (docs/fitting.md) does, and ``restore`` runs through
    ``after_fit_callbacks``.
    """
    df = ls_chain["draw"](800, 2)[["x1", "x2"]]
    val = ls_chain["draw"](400, 3)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    manual = {"nll": float("inf"), "epoch": 0}

    def keep_best(f, epoch, opt):
        nll = sum(f.nll(val).values())
        if nll < manual["nll"]:
            manual.update(nll=nll, epoch=epoch)

    best = RestoreBest(val)
    flow.fit(
        df,
        epochs=40,
        learning_rate=1e-2,
        after_epoch_callbacks=[keep_best, best],
        after_fit_callbacks=[best.restore],
    )
    assert (best.best_nll, best.best_epoch) == (manual["nll"], manual["epoch"])
    assert sum(flow.nll(val).values()) == pytest.approx(best.best_nll, rel=1e-6)


def test_restore_best_without_an_epoch_refuses(ls_chain):
    """``restore`` before any epoch is a bug in the caller's loop — loud."""
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    with pytest.raises(RuntimeError, match="no epoch"):
        RestoreBest(ls_chain["draw"](50, 0)).restore(flow)


def test_logger_prints_every_nth_epoch(ls_chain, capsys):
    """``callbacks.Logger`` prints train (and val) NLL on the ``every`` grid."""
    df = ls_chain["draw"](200, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    flow.fit(df, epochs=5, after_epoch_callbacks=Logger(df, every=2))
    lines = capsys.readouterr().out.strip().splitlines()
    assert [ln.split()[1] for ln in lines] == ["2", "4"]
    assert all("train NLL" in ln and "val NLL" in ln for ln in lines)


def test_per_node_plateau_stops_early_and_keeps_the_mle(ls_chain):
    """``callbacks.PerNodePlateau`` over ``per_node_adam`` freezes every node,
    stops the fit before the epoch ceiling, and still lands on the known
    truth (x2 <- x1 weight 1.2 in the inline DGP).
    """
    df = ls_chain["draw"](2000, 4)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    opt = per_node_adam(flow, lr=1e-2)
    sched = PerNodePlateau(df, patience=10, freeze=40)
    flow.fit(
        df, epochs=4000, batch_size=512, optimizer=opt, after_epoch_callbacks=sched
    )
    assert sched.frozen == {"x1", "x2"}
    assert all(g["lr"] == 0.0 for g in opt.param_groups)
    assert len(flow.history["train"]) < 4000
    assert float(flow.ls_coefficients()["x2"]["x1"][0]) == pytest.approx(1.2, abs=0.1)


def test_per_node_plateau_rejects_an_untagged_optimizer(ls_chain):
    """A plain optimizer (one group, no ``node`` tag) is refused loudly."""
    df = ls_chain["draw"](200, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    flow.calibrate(df)
    opt = torch.optim.Adam(flow.parameters(), lr=1e-2)
    with pytest.raises(ValueError, match="per_node_adam"):
        PerNodePlateau(df, patience=5, freeze=10).step(flow.nll(df), opt)


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

    flow.fit(
        obs, epochs=4000, batch_size=512, optimizer=opt, after_epoch_callbacks=step
    )
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
