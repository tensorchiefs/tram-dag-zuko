"""Predefined ``fit`` callbacks: ``RestoreBest``, ``EarlyStopping``, ``PerNodePlateau``.

``fit`` owns validation and progress printing; the callbacks here read the
per-node validation NLL that ``fit`` appends to ``flow.history["val"]`` after
every epoch — computed once, shared by all of them. One ``callbacks=`` list
is the whole registration (the ``fit`` docstring shows it); anything not
covered here is a :class:`Callback` subclass of your own (docs/fitting.md).
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

import copy
import math

import torch


# %% private functions -------------------------------------------------------------
def _last_val(flow) -> dict[str, float]:
    """Give the current epoch's per-node validation NLL, or fail loudly.

    A stale entry from an earlier validated fit does not count: THIS fit
    must validate (``fit`` records that on the flow), so the last entry is
    the current epoch's.
    """
    if not getattr(flow, "_fit_validated", False) or not flow.history.get("val"):
        raise RuntimeError(
            "this callback reads flow.history['val'] — pass validation_data= "
            "or validation_split= to fit()"
        )
    return flow.history["val"][-1]


# %% public functions ------------------------------------------------------------------
def per_node_adam(flow, lr: float = 1e-2, **adam_kwargs) -> torch.optim.Adam:
    """Give an Adam with one ``node``-tagged parameter group per node.

    The per-node NLLs have independent gradients, so a learning rate per
    group is exactly independent per-node training. This is the optimizer
    :class:`PerNodePlateau` needs.
    """
    return torch.optim.Adam(
        [
            # initial_lr (torch's scheduler convention) lets PerNodePlateau
            # restore a decayed group to its start at the next fit begin
            {
                "params": list(flow.nodes[n].parameters()),
                "lr": lr,
                "initial_lr": lr,
                "node": n,
            }
            for n in flow.order
        ],
        **adam_kwargs,
    )


# %% public classes --------------------------------------------------------------------
class Callback:
    """Base class of ``fit(callbacks=)`` entries — override any of the hooks.

    ``on_fit_begin(flow, optimizer)`` runs once after calibration, before the
    first epoch (the shipped callbacks reset their state here, so one
    instance is safe to reuse across fits). ``on_epoch_end(flow, epoch,
    optimizer)`` runs after every epoch, once the epoch's train NLLs are in
    ``flow.history["train"]``; the fit stops after an epoch in which any
    callback returned ``True``. ``on_fit_end(flow, optimizer)`` runs once
    after the loop and **before** the VC re-centering, so a hook that swaps
    the weights hands them to the re-centering.
    """

    def on_fit_begin(self, flow, optimizer) -> None:
        """Run once before the first epoch."""

    def on_epoch_end(self, flow, epoch: int, optimizer):
        """Run after every epoch; return ``True`` to stop the fit."""

    def on_fit_end(self, flow, optimizer) -> None:
        """Run once after the loop, before the VC re-centering."""


class RestoreBest(Callback):
    """Keep the weights of the best summed validation NLL; restore them at the end.

    The key empirical finding behind it: flexible (CI/CS) models overfit
    observational confounding at the MLE and need best-validation weights to
    recover the causal effect (docs/fitting.md). The restoration is
    automatic (``on_fit_end``, which ``fit`` runs before the VC
    re-centering) — registering the instance in ``callbacks=`` is the whole
    recipe. Reads ``flow.history["val"]``, so the fit needs
    ``validation_data=`` or ``validation_split=``.

    Attributes
    ----------
    best_nll, best_epoch
        The best summed validation NLL seen this fit and its epoch.
    """

    def __init__(self):
        self._reset()

    def _reset(self) -> None:
        self.best_nll = math.inf
        self.best_epoch = 0
        self._state = None

    def on_fit_begin(self, flow, optimizer) -> None:
        """Start fresh — the snapshot never leaks into the next fit."""
        self._reset()

    def on_epoch_end(self, flow, epoch: int, optimizer) -> None:
        """Snapshot the weights when the validation NLL improves."""
        nll = sum(_last_val(flow).values())
        if nll < self.best_nll:
            self.best_nll, self.best_epoch = nll, epoch
            self._state = copy.deepcopy(flow.state_dict())

    def on_fit_end(self, flow, optimizer) -> None:
        """Load the best weights back into the flow."""
        if self._state is None:
            raise RuntimeError("RestoreBest has seen no epoch; nothing to restore")
        flow.load_state_dict(self._state)


class EarlyStopping(Callback):
    """Stop the fit after ``patience`` epochs without validation improvement.

    Tracks the summed validation NLL itself, so it composes with
    :class:`RestoreBest` in any order — the usual pair is
    ``callbacks=[RestoreBest(), EarlyStopping(patience=200)]``. Reads
    ``flow.history["val"]``, so the fit needs ``validation_data=`` or
    ``validation_split=``.

    Parameters
    ----------
    patience : int
        Epochs without a ``min_delta`` improvement before stopping.
    min_delta : float, optional
        Improvement below this is flat, by default 0.
    """

    def __init__(self, *, patience: int, min_delta: float = 0.0):
        if patience < 1:
            raise ValueError(f"patience must be at least 1, got {patience}")
        self.patience, self.min_delta = patience, min_delta
        self._reset()

    def _reset(self) -> None:
        self.best_nll = math.inf
        self.best_epoch = 0

    def on_fit_begin(self, flow, optimizer) -> None:
        """Start fresh — patience never carries into the next fit."""
        self._reset()

    def on_epoch_end(self, flow, epoch: int, optimizer) -> bool:
        """Give ``True`` once the last improvement is ``patience`` epochs old."""
        nll = sum(_last_val(flow).values())
        if nll < self.best_nll - self.min_delta:
            self.best_nll, self.best_epoch = nll, epoch
        return epoch - self.best_epoch >= self.patience


class PerNodePlateau(Callback):
    """Per-node plateau decay and freezing on the validation NLL.

    A node's learning rate decays by ``factor`` after every ``patience``
    epochs without a ``min_delta`` improvement of its own validation NLL,
    floored at ``1e-3`` of its start; once it has decayed to ``1e-2`` of the
    start and stayed flat for ``freeze`` epochs the node leaves training
    (rate 0). The callback stops the fit when every node has left. Valid
    because the per-node NLLs have independent gradients — build the
    optimizer with :func:`per_node_adam` (one ``node``-tagged group per
    node), and give ``fit`` a validation set (the callback reads
    ``flow.history["val"]``).

    A frozen node's rate is 0 but its forward/backward still runs, so the
    saving is in epochs, not per-epoch wall clock. Do not attach a torch lr
    scheduler to the same optimizer — two controllers would steer the same
    group rates (a ``LambdaLR`` even resets frozen nodes to ``initial_lr``).
    This is the pre-0.4 ``fit(schedule="plateau", freeze_patience=)`` recipe,
    back as an opt-in callback; `docs/training-speed.md` has its
    measurements.

    Parameters
    ----------
    patience, freeze : int
        Flat epochs before a decay, and before a decayed node freezes. The
        defaults (15/50) are the training-speed benchmark's VACA settings;
        its stroke workload runs 30/120
        (``experiments/benchmarks/bench_training.py``).
    min_delta : float, optional
        Improvement below this is flat, by default 1e-4.
    factor : float, optional
        Learning-rate decay per plateau, by default 0.3.
    """

    def __init__(
        self,
        *,
        patience: int = 15,
        freeze: int = 50,
        min_delta: float = 1e-4,
        factor: float = 0.3,
    ):
        if patience < 1 or freeze < 1:
            raise ValueError(
                f"patience and freeze must be at least 1, got {patience}/{freeze}"
            )
        self.patience, self.freeze = patience, freeze
        self.min_delta, self.factor = min_delta, factor
        self._reset()

    def _reset(self) -> None:
        self.lr0: dict = {}
        self.best: dict = {}
        self.bad: dict = {}
        self.frozen: set = set()

    def on_fit_begin(self, flow, optimizer) -> None:
        """Start fresh — rates and frozen nodes never carry into the next fit.

        A reused optimizer's decayed (or zeroed) group rates go back to the
        ``initial_lr`` that ``per_node_adam`` stamped on each group; without
        that, the new baseline would be the old decayed rate and a frozen
        node would "train" at rate 0. The stamp lives on the group, so a
        fresh optimizer, a fresh callback or a second flow all stay correct.
        """
        if optimizer is not None:
            for g in optimizer.param_groups:
                if "initial_lr" in g:
                    g["lr"] = g["initial_lr"]
        self._reset()

    def on_epoch_end(self, flow, epoch: int, optimizer) -> bool:
        """Step on the epoch's validation NLL; ``True`` once every node froze."""
        return self.step(_last_val(flow), optimizer)

    def step(self, nll: dict[str, float], optimizer) -> bool:
        """Step every unfrozen node on its own NLL; ``True`` when all are frozen."""
        for g in optimizer.param_groups:
            if "node" not in g:
                raise ValueError(
                    "PerNodePlateau needs one 'node'-tagged parameter group "
                    "per node — build the optimizer with per_node_adam(flow, lr)"
                )
            lr0 = self.lr0.setdefault(g["node"], g.get("initial_lr", g["lr"]))
            if lr0 == 0.0:
                raise ValueError(
                    f"node {g['node']!r} starts at learning rate 0 — build a "
                    "fresh per_node_adam(flow, lr): its initial_lr stamp lets "
                    "a reused optimizer restore its rates"
                )
            if g["node"] not in self.frozen:
                self._step_node(g, nll[g["node"]])
        return len(self.frozen) == len(optimizer.param_groups)

    def _step_node(self, g: dict, nll: float) -> None:
        name = g["node"]
        if nll < self.best.get(name, math.inf) - self.min_delta:
            self.best[name], self.bad[name] = nll, 0
        else:
            self.bad[name] = self.bad.get(name, 0) + 1
        if self.bad[name] and self.bad[name] % self.patience == 0:
            g["lr"] = max(g["lr"] * self.factor, self.lr0[name] * 1e-3)
        decayed = g["lr"] <= self.lr0[name] * 1e-2 * (1 + 1e-9)
        if decayed and self.bad[name] >= self.freeze:
            self.frozen.add(name)
            g["lr"] = 0.0
