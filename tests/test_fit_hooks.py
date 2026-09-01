"""Tests for fit()'s hooks: ``optimizer=`` and the ``callbacks=`` list.

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
from tramdag.callbacks import (
    Callback,
    EarlyStopping,
    PerNodePlateau,
    RestoreBest,
    per_node_adam,
)


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


def test_callbacks_all_run_and_any_stops(ls_chain):
    """Every callback runs even on the stop epoch (no short-circuit); a
    Callback instance gets all three hooks, a bare callable is on_epoch_end.
    """
    df = ls_chain["draw"](200, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    calls = []

    class Recorder(Callback):
        def on_fit_begin(self, f, opt):
            calls.append(("begin", 0))

        def on_epoch_end(self, f, epoch, opt):
            # also pins the callback contract: live flow, 1-based epoch, optimizer
            assert f is flow
            assert isinstance(opt, torch.optim.Adam)
            calls.append(("stop?", epoch))
            return epoch == 2

        def on_fit_end(self, f, opt):
            calls.append(("end", 0))

    def logger(f, epoch, opt):
        calls.append(("log", epoch))

    flow.fit(df, epochs=10, callbacks=[Recorder(), logger])
    assert calls == [
        ("begin", 0),
        ("stop?", 1),
        ("log", 1),
        ("stop?", 2),
        ("log", 2),  # the logger still ran on the stop epoch
        ("end", 0),
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
    recipe (docs/fitting.md) does — restoration is automatic at fit end.
    """
    df = ls_chain["draw"](800, 2)[["x1", "x2"]]
    val = ls_chain["draw"](400, 3)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    manual = {"nll": float("inf"), "epoch": 0}

    def keep_best(f, epoch, opt):
        nll = sum(f.nll(val).values())
        if nll < manual["nll"]:
            manual.update(nll=nll, epoch=epoch)

    best = RestoreBest()
    flow.fit(
        df,
        epochs=40,
        learning_rate=1e-2,
        validation_data=val,
        callbacks=[keep_best, best],
    )
    assert (best.best_nll, best.best_epoch) == (manual["nll"], manual["epoch"])
    assert sum(flow.nll(val).values()) == pytest.approx(best.best_nll, rel=1e-6)


def test_restore_best_resets_between_fits(ls_chain):
    """A reused instance starts fresh: the second fit restores its own best,
    never the first fit's snapshot.
    """
    df = ls_chain["draw"](400, 6)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    best = RestoreBest()
    flow.fit(df, epochs=5, validation_data=df, callbacks=best)
    first = (best.best_nll, best.best_epoch)
    flow.fit(df, epochs=3, validation_data=df, callbacks=best)
    assert best.best_epoch <= 3  # counted within the second fit
    assert best.best_nll <= first[0] + 1e-9  # training continued, no stale state


def test_early_stopping_stops_and_composes_with_restore_best(ls_chain):
    """EarlyStopping halts the fit once the best epoch is ``patience`` old,
    in either registration order relative to RestoreBest.
    """
    df = ls_chain["draw"](800, 7)[["x1", "x2"]]
    val = ls_chain["draw"](400, 8)[["x1", "x2"]]
    for order in (lambda b, e: [b, e], lambda b, e: [e, b]):
        flow = CausalFlowDAG(_two_node_spec(), seed=0)
        best, early = RestoreBest(), EarlyStopping(patience=5)
        flow.fit(
            df,
            epochs=4000,
            validation_data=val,
            callbacks=order(best, early),
        )
        ran = len(flow.history["train"])
        assert ran < 4000
        assert ran - early.best_epoch == 5
        assert sum(flow.nll(val).values()) == pytest.approx(best.best_nll, rel=1e-6)


def test_misregistered_callback_fails_before_training(ls_chain):
    """A bare callable with the wrong arity (or a non-callable) must raise
    up front, not after the last epoch of a long run.
    """
    df = ls_chain["draw"](200, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    with pytest.raises(TypeError, match="flow, epoch, optimizer"):
        flow.fit(df, epochs=10, callbacks=[lambda f, opt: None])  # 2-arg hook
    assert len(flow.history["train"]) == 0  # nothing trained
    with pytest.raises(TypeError, match="Callback instances or callables"):
        flow.fit(df, epochs=10, callbacks=[42])


def test_epochs_must_be_positive(ls_chain):
    """epochs=0 would skip the loop but still calibrate and run the
    after-fit hooks — refuse it instead.
    """
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    with pytest.raises(ValueError, match="epochs"):
        flow.fit(ls_chain["draw"](100, 0)[["x1", "x2"]], epochs=0)


def test_restore_best_without_an_epoch_refuses(ls_chain):
    """Restoring before any epoch is a bug in the caller's loop — loud."""
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    with pytest.raises(RuntimeError, match="no epoch"):
        RestoreBest().on_fit_end(flow, None)


def test_callbacks_demand_fit_managed_validation(ls_chain):
    """RestoreBest without validation_data/-split fails loudly at epoch 1."""
    df = ls_chain["draw"](100, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    with pytest.raises(RuntimeError, match="validation_data"):
        flow.fit(df, epochs=2, callbacks=RestoreBest())


def test_verbose_prints_every_nth_and_final_epoch(ls_chain, capsys):
    """``fit(verbose=N)`` prints every Nth epoch plus the final one."""
    df = ls_chain["draw"](200, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    flow.fit(df, epochs=5, verbose=2, validation_data=df.head(50))
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3  # epochs 2, 4 and the final 5
    assert all("train" in ln and "val" in ln for ln in lines)
    assert "5" in lines[-1]


def test_validation_split_takes_the_tail(ls_chain):
    """A float split trains on the head, validates on the tail (Keras rule)."""
    df = ls_chain["draw"](200, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    flow.fit(df, epochs=2, validation_split=0.25, batch_size=50)
    assert len(flow.history["val"]) == 2
    # calibration saw only the head: the range is the head's quantiles
    head = df.iloc[:150]
    lo = float(flow.nodes["x1"].ut.xmin)
    assert lo == pytest.approx(head["x1"].quantile(0.05), abs=1e-6)
    with pytest.raises(ValueError, match="not both"):
        flow.fit(df, epochs=1, validation_data=df, validation_split=0.5)


def test_per_node_plateau_stops_early_and_keeps_the_mle(ls_chain):
    """``callbacks.PerNodePlateau`` over ``per_node_adam`` freezes every node,
    stops the fit before the epoch ceiling, and still lands on the known
    truth (x2 <- x1 weight 1.2 in the inline DGP).
    """
    df = ls_chain["draw"](2000, 4)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    opt = per_node_adam(flow, lr=1e-2)
    sched = PerNodePlateau(patience=10, freeze=40)
    flow.fit(
        df,
        epochs=4000,
        batch_size=512,
        validation_data=df,
        optimizer=opt,
        callbacks=sched,
    )
    assert sched.frozen == {"x1", "x2"}
    assert all(g["lr"] == 0.0 for g in opt.param_groups)
    assert len(flow.history["train"]) < 4000
    assert float(flow.ls_coefficients()["x2"]["x1"][0]) == pytest.approx(1.2, abs=0.1)


def test_per_node_plateau_reuse_restores_the_optimizer_rates(ls_chain):
    """A reused instance with a reused optimizer must not re-baseline on the
    decayed (or zeroed) rates — fit begin restores each node's start rate.
    """
    df = ls_chain["draw"](2000, 4)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    opt = per_node_adam(flow, lr=1e-2)
    sched = PerNodePlateau(patience=10, freeze=40)
    flow.fit(df, epochs=4000, validation_data=df, optimizer=opt, callbacks=sched)
    assert all(g["lr"] == 0.0 for g in opt.param_groups)  # everything froze
    flow.fit(df, epochs=1, validation_data=df, optimizer=opt, callbacks=sched)
    assert sched.lr0 == {"x1": 1e-2, "x2": 1e-2}  # baselines are the starts
    assert all(g["lr"] > 0.0 or g["node"] in sched.frozen for g in opt.param_groups)


def test_per_node_plateau_respects_a_fresh_optimizer_rate(ls_chain):
    """A reused callback must not clobber a fresh optimizer's different lr —
    the restore reads the group's own initial_lr stamp, not callback state.
    """
    df = ls_chain["draw"](2000, 4)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    sched = PerNodePlateau(patience=10, freeze=40)
    flow.fit(
        df,
        epochs=4000,
        validation_data=df,
        optimizer=per_node_adam(flow, lr=1e-2),
        callbacks=sched,
    )
    opt2 = per_node_adam(flow, lr=1e-3)  # a fresh optimizer, 10x smaller rate
    flow.fit(df, epochs=1, validation_data=df, optimizer=opt2, callbacks=sched)
    assert sched.lr0 == {"x1": 1e-3, "x2": 1e-3}


def test_fit_classical_marks_validation_stale(ls_chain):
    """After fit_classical the last history["val"] entry is pre-classical —
    a manually driven callback must refuse it, not treat it as current.
    """
    df = ls_chain["draw"](400, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    flow.fit(df, epochs=2, validation_data=df, callbacks=RestoreBest())
    flow.fit_classical(df)
    with pytest.raises(RuntimeError, match="validation_data"):
        RestoreBest().on_epoch_end(flow, 1, None)


def test_callbacks_reject_stale_validation_from_an_earlier_fit(ls_chain):
    """After a validated fit, an unvalidated fit must not let a callback read
    the old history["val"] entry as the current epoch.
    """
    df = ls_chain["draw"](200, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    flow.fit(df, epochs=2, validation_data=df, callbacks=RestoreBest())
    with pytest.raises(RuntimeError, match="validation_data"):
        flow.fit(df, epochs=2, callbacks=RestoreBest())  # no validation now
    flow.fit(df, epochs=2, validation_data=df, callbacks=RestoreBest())  # fine again


def test_callbacks_reject_the_class_instead_of_an_instance(ls_chain):
    """`callbacks=RestoreBest` (forgotten parens) fails with the fix named."""
    df = ls_chain["draw"](200, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    with pytest.raises(TypeError, match="instantiate it: RestoreBest"):
        flow.fit(df, epochs=2, callbacks=RestoreBest)


def test_per_node_plateau_rejects_a_zero_start_rate(ls_chain):
    """A node-tagged group at lr 0 without the initial_lr stamp (a reused
    hand-built optimizer whose node froze) must fail, not train at rate 0.
    """
    df = ls_chain["draw"](200, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    flow.calibrate(df)
    opt = torch.optim.Adam(
        [
            {"params": list(flow.nodes[n].parameters()), "lr": 0.0, "node": n}
            for n in flow.order
        ]
    )
    with pytest.raises(ValueError, match="learning rate 0"):
        PerNodePlateau().step(flow.nll(df), opt)


def test_per_node_plateau_rejects_an_untagged_optimizer(ls_chain):
    """A plain optimizer (one group, no ``node`` tag) is refused loudly."""
    df = ls_chain["draw"](200, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    flow.calibrate(df)
    opt = torch.optim.Adam(flow.parameters(), lr=1e-2)
    with pytest.raises(ValueError, match="per_node_adam"):
        PerNodePlateau(patience=5, freeze=10).step(flow.nll(df), opt)


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

    flow.fit(obs, epochs=4000, batch_size=512, optimizer=opt, callbacks=step)
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
