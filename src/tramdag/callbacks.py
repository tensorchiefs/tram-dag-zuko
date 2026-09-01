"""Predefined ``fit`` callbacks: best-weight restoration, per-node plateau.

``fit`` owns validation (``validation_data=`` / ``validation_split=``) and
progress printing (``verbose=``); the callbacks here read the per-node
validation NLL that ``fit`` appends to ``flow.history["val"]`` after every
epoch — computed once, shared by all of them::

    from tramdag.callbacks import RestoreBest

    best = RestoreBest()
    flow.fit(
        train_df,
        epochs=4000,
        validation_data=val_df,
        verbose=50,
        after_epoch_callbacks=[best],
        after_fit_callbacks=[best.restore],
    )

Anything not covered here is a few lines of your own (docs/fitting.md).
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

import copy
import math

import torch


# %% private functions -------------------------------------------------------------
def _last_val(flow) -> dict[str, float]:
    """Give the current epoch's per-node validation NLL, or fail loudly."""
    val = flow.history.get("val")
    if not val:
        raise RuntimeError(
            "this callback reads flow.history['val'] — pass validation_data= "
            "or validation_split= to fit()"
        )
    return val[-1]


# %% public functions ------------------------------------------------------------------
def per_node_adam(flow, lr: float = 1e-2, **adam_kwargs) -> torch.optim.Adam:
    """Give an Adam with one ``node``-tagged parameter group per node.

    The per-node NLLs have independent gradients, so a learning rate per
    group is exactly independent per-node training. This is the optimizer
    :class:`PerNodePlateau` needs.
    """
    return torch.optim.Adam(
        [
            {"params": list(flow.nodes[n].parameters()), "lr": lr, "node": n}
            for n in flow.order
        ],
        **adam_kwargs,
    )


# %% public classes --------------------------------------------------------------------
class RestoreBest:
    """Keep the weights of the best summed validation NLL; ``restore`` loads them.

    The key empirical finding behind it: flexible (CI/CS) models overfit
    observational confounding at the MLE and need best-validation weights to
    recover the causal effect (docs/fitting.md). Register the instance in
    ``after_epoch_callbacks`` and its :meth:`restore` in
    ``after_fit_callbacks``; ``fit`` runs the latter *before* the VC
    re-centering, so the restored weights are what gets re-centered. One
    instance per fit and per flow — the snapshot is a full ``state_dict``.

    Reads ``flow.history["val"]``, so the fit needs ``validation_data=`` or
    ``validation_split=``.

    Attributes
    ----------
    best_nll, best_epoch
        The best summed validation NLL seen and the epoch it came from.
    """

    def __init__(self):
        self.best_nll = math.inf
        self.best_epoch = 0
        self._state = None

    def __call__(self, flow, epoch: int, optimizer) -> None:
        """Snapshot the weights when the validation NLL improves."""
        nll = sum(_last_val(flow).values())
        if nll < self.best_nll:
            self.best_nll, self.best_epoch = nll, epoch
            self._state = copy.deepcopy(flow.state_dict())

    def restore(self, flow, optimizer=None) -> None:
        """Load the best weights back into the flow.

        Raises
        ------
        RuntimeError
            If no epoch has run through the callback yet.
        """
        if self._state is None:
            raise RuntimeError("RestoreBest has seen no epoch; nothing to restore")
        flow.load_state_dict(self._state)


class PerNodePlateau:
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
    saving is in epochs, not per-epoch wall clock. This is the pre-0.4
    ``fit(schedule="plateau", freeze_patience=)`` recipe, back as an opt-in
    callback; `docs/training-speed.md` has its measurements. One instance
    per fit — ``frozen``/``best``/``lr0`` carry over and would stop a second
    fit immediately.

    Parameters
    ----------
    patience, freeze : int
        Flat epochs before a decay, and before a decayed node freezes.
        The training-speed benchmark runs ``patience=30, freeze=120`` on its
        stroke workload and ``patience=15, freeze=50`` on the VACA one
        (``experiments/benchmarks/bench_training.py``).
    min_delta : float, optional
        Improvement below this is flat, by default 1e-4.
    factor : float, optional
        Learning-rate decay per plateau, by default 0.3.
    """

    def __init__(
        self,
        *,
        patience: int,
        freeze: int,
        min_delta: float = 1e-4,
        factor: float = 0.3,
    ):
        if patience < 1 or freeze < 1:
            raise ValueError(
                f"patience and freeze must be at least 1, got {patience}/{freeze}"
            )
        self.patience, self.freeze = patience, freeze
        self.min_delta, self.factor = min_delta, factor
        self.lr0: dict = {}
        self.best: dict = {}
        self.bad: dict = {}
        self.frozen: set = set()

    def __call__(self, flow, epoch: int, optimizer) -> bool:
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
            self.lr0.setdefault(g["node"], g["lr"])
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
