"""The two fitting paths, as free functions over a flow.

`fit` is one minibatch Adam loop (validation, verbose printing and the
callback hooks included); `fit_classical` is the float64 full-batch L-BFGS
exact-MLE route for all-`ls` specs. `CausalFlowDAG.fit`/`.fit_classical` are
one-line delegates into this module, so the public API lives on the flow.
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

import inspect
import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .callbacks import Callback
from .terms import ShiftTerm

if TYPE_CHECKING:
    from .flow import CausalFlowDAG


# %% private functions -----------------------------------------------------------------
class _FnCallback(Callback):
    """A bare callable in ``callbacks=``, adapted to an ``on_epoch_end`` hook."""

    def __init__(self, fn):
        self.fn = fn

    def on_epoch_end(self, flow, epoch: int, optimizer):
        return self.fn(flow, epoch, optimizer)


def _check_fit_sizes(
    epochs: int, batch_size: int, verbose: int, validation_batch_size: int | None
) -> None:
    """Reject a non-positive epoch, batch or verbose value before anything runs."""
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if verbose < 0 or int(verbose) != verbose:
        raise ValueError(f"verbose must be a non-negative int, got {verbose!r}")
    if validation_batch_size is not None and validation_batch_size < 1:
        raise ValueError(
            f"validation_batch_size must be at least 1, got {validation_batch_size}"
        )


def _split_validation(
    train_df: pd.DataFrame,
    validation_data: pd.DataFrame | None,
    validation_split: float | None,
    vc_ehat: dict | None,
):
    """Resolve fit's validation arguments (the Keras rules).

    ``validation_split`` takes the LAST fraction of ``train_df`` as
    validation without shuffling, exactly like Keras — deterministic, no
    hidden RNG. ``vc_ehat`` rows are sliced with the same split, so the
    caller supplies propensities for the frame they passed.
    """
    if validation_split is None:
        return train_df, validation_data, vc_ehat
    if validation_data is not None:
        raise ValueError("pass validation_data OR validation_split, not both")
    if not 0.0 < validation_split < 1.0:
        raise ValueError(f"validation_split must be in (0, 1), got {validation_split}")
    cut = round(len(train_df) * (1.0 - validation_split))
    if cut < 1 or cut >= len(train_df):
        raise ValueError(
            f"validation_split={validation_split} leaves no rows on one side "
            f"of the {len(train_df)}-row frame"
        )
    vc_ehat = _slice_vc_ehat(vc_ehat, len(train_df), cut)
    return train_df.iloc[:cut], train_df.iloc[cut:], vc_ehat


def _slice_vc_ehat(vc_ehat: dict | None, n_full: int, cut: int) -> dict | None:
    """Slice the caller's propensities with the validation split.

    The rows must cover the FULL frame passed to ``fit`` — a wrong length
    fails here instead of being silently truncated by the slice.
    """
    if vc_ehat is None:
        return None
    for node, d in vc_ehat.items():
        for t, e in d.items():
            if len(np.asarray(e)) != n_full:
                raise ValueError(
                    f"vc_ehat[{node!r}][{t!r}] has {len(np.asarray(e))} rows, "
                    f"not the {n_full} of the frame passed to fit — supply "
                    "propensities for that frame; the split slices them"
                )
    return {
        node: {t: np.asarray(e)[:cut] for t, e in d.items()}
        for node, d in vc_ehat.items()
    }


def _normalize_callbacks(cbs) -> list[Callback]:
    """Give ``callbacks=`` as a list of ``Callback``s, or fail loudly now.

    A :class:`~tramdag.callbacks.Callback` instance is trusted — the base
    class defines all three hooks. A bare callable is an ``on_epoch_end``
    hook and must accept ``(flow, epoch, optimizer)``; checked here so a
    wrong entry fails before the first epoch, not after the last one.
    """
    if cbs is None:
        return []
    if isinstance(cbs, Callback) or callable(cbs):
        cbs = [cbs]
    out = []
    for cb in cbs:
        if isinstance(cb, Callback):
            out.append(cb)
            continue
        if isinstance(cb, type) and issubclass(cb, Callback):
            raise TypeError(
                f"callbacks= got the class {cb.__name__} — instantiate it: "
                f"{cb.__name__}()"
            )
        if not callable(cb):
            raise TypeError(
                f"callbacks entries must be Callback instances or callables, got {cb!r}"
            )
        _check_epoch_hook(cb)
        out.append(_FnCallback(cb))
    return out


def _check_epoch_hook(cb) -> None:
    """Reject a bare callable of the wrong arity before training starts."""
    try:
        sig = inspect.signature(cb, follow_wrapped=False)
    except (TypeError, ValueError):  # a callable without a signature
        return
    try:
        sig.bind(None, None, None)
    except TypeError:
        raise TypeError(
            "a bare callable in callbacks= is called as "
            f"cb(flow, epoch, optimizer); {cb!r} does not accept these "
            "arguments — for the other hooks subclass tramdag.callbacks.Callback"
        ) from None


def _slice_ehat(
    vc_ehat: dict[str, dict[str, Tensor]] | None, idx: Tensor
) -> dict[str, dict[str, Tensor]] | None:
    """Slice the frozen out-of-fold propensities down to one minibatch."""
    if vc_ehat is None:
        return None
    return {nm: {on: e[idx] for on, e in d.items()} for nm, d in vc_ehat.items()}


# %% public functions ------------------------------------------------------------------
def fit(
    flow: CausalFlowDAG,
    train_df: pd.DataFrame,
    *,
    epochs: int,
    learning_rate: float = 1e-2,
    batch_size: int = 512,
    validation_data: pd.DataFrame | None = None,
    validation_split: float | None = None,
    validation_batch_size: int | None = None,
    verbose: int = 0,
    seed: int | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    callbacks=None,
    vc_ehat: dict | None = None,
) -> CausalFlowDAG:
    """Fit by minibatch Adam — the full contract is on ``CausalFlowDAG.fit``."""
    _check_fit_sizes(epochs, batch_size, verbose, validation_batch_size)
    cbs = _normalize_callbacks(callbacks)
    if seed is not None:
        torch.manual_seed(seed)
    train_df, validation_data, vc_ehat = _split_validation(
        train_df, validation_data, validation_split, vc_ehat
    )
    # vc_ehat is validated BEFORE calibrate: a malformed dict must fail
    # while the flow is still untouched (calibrate sets ranges and the
    # marginal start — a half-mutated flow after an error would be worse
    # than no fit at all)
    ehat = flow._vc_ehat_train(train_df, vc_ehat)
    flow.calibrate(train_df)
    vals = flow._tensorize(train_df)
    val_vals = flow._tensorize(validation_data) if validation_data is not None else None
    # the shipped callbacks check this: an unvalidated fit in between must
    # not let them read a stale history["val"] entry as the current epoch
    flow._fit_validated = val_vals is not None
    opt = optimizer or torch.optim.Adam(flow.parameters(), lr=learning_rate)
    penalized = [
        m
        for nd in flow.nodes.values()
        for m in nd.shifts.values()
        if isinstance(m, ShiftTerm) and m.has_regularizer
    ]
    for cb in cbs:
        cb.on_fit_begin(flow, opt)
    for epoch in range(1, epochs + 1):
        _epoch_pass(
            flow,
            vals,
            ehat,
            opt,
            batch_size,
            penalized,
            val_vals,
            validation_batch_size,
        )
        # every callback runs (a stop must not skip a monitoring one)
        stops = [bool(cb.on_epoch_end(flow, epoch, opt)) for cb in cbs]
        _log_epoch(
            flow,
            epoch,
            epochs,
            verbose,
            stopped=any(stops),
            has_val=val_vals is not None,
        )
        if any(stops):
            break
    for cb in cbs:
        cb.on_fit_end(flow, opt)  # before re-centering: restored weights re-center
    flow._recenter_vc(vals)
    flow.eval()
    return flow


def _epoch_pass(
    flow, vals, ehat, opt, batch_size, penalized, val_vals, validation_batch_size
) -> None:
    """Run one training epoch and, when configured, the validation pass."""
    flow.train()
    flow.history["train"].append(
        _fit_epoch(flow, vals, ehat, opt, batch_size, penalized)
    )
    flow.eval()
    if val_vals is not None:
        flow.history.setdefault("val", []).append(
            _val_nll(flow, val_vals, validation_batch_size)
        )


def _log_epoch(
    flow, epoch: int, epochs: int, verbose: int, stopped: bool, has_val: bool
):
    """Print one ``verbose`` progress line on the Nth and the final epoch."""
    last = stopped or epoch == epochs
    if not verbose or (epoch % verbose and not last):
        return
    line = f"epoch {epoch}/{epochs}"
    line += f"  train {sum(flow.history['train'][-1].values()):.4f}"
    if has_val:  # THIS fit's validation, not a stale earlier one
        line += f"  val {sum(flow.history['val'][-1].values()):.4f}"
    print(line)


def _val_nll(flow, vals: dict[str, Tensor], batch_size: int | None) -> dict[str, float]:
    """Give the per-node mean validation NLL, chunked by validation batch size."""
    n = len(next(iter(vals.values())))
    chunk = batch_size or n
    acc = dict.fromkeys(flow.order, 0.0)
    with torch.no_grad():
        for start in range(0, n, chunk):
            batch = {k: v[start : start + chunk] for k, v in vals.items()}
            weight = len(next(iter(batch.values()))) / n
            for k, v in flow.node_log_prob(batch).items():
                acc[k] += float(-v.mean()) * weight
    return acc


def _fit_epoch(
    flow,
    vals: dict[str, Tensor],
    ehat: dict[str, dict[str, Tensor]] | None,
    opt: torch.optim.Optimizer,
    batch_size: int,
    penalized: list,
) -> dict[str, float]:
    """One shuffled pass over the rows; give the epoch-mean train NLL per node."""
    n = len(next(iter(vals.values())))
    acc = dict.fromkeys(flow.order, 0.0)
    for idx in torch.randperm(n, device=flow.device).split(batch_size):
        batch = {k: v[idx] for k, v in vals.items()}
        per_node = flow.node_log_prob(batch, vc_ehat=_slice_ehat(ehat, idx))
        nlls = {k: -v.mean() for k, v in per_node.items()}
        loss = torch.stack(list(nlls.values())).sum()
        for m in penalized:  # the penalty joins the loss, not the history
            loss = loss + m.penalty * m.l2() / n
        opt.zero_grad()
        loss.backward()
        opt.step()
        w = len(idx) / n
        for k, v in nlls.items():
            acc[k] += float(v.detach()) * w
    return acc


def fit_classical(
    flow: CausalFlowDAG,
    train_df: pd.DataFrame,
    *,
    max_iter: int = 400,
    tol: float = 1e-9,
    history_size: int = 50,
) -> dict:
    """Fit an all-``ls`` model by L-BFGS — contract on the flow method."""
    if not flow._is_classical():
        raise ValueError(
            "fit_classical requires an all-`ls` spec, that is every edge "
            "term 'ls'. This spec has cs, ci or vc terms. Use fit() for "
            "flexible models."
        )
    flow.calibrate(train_df, marginal_init=False)  # L-BFGS needs no warm start
    # a callback used manually afterwards must not read a pre-classical
    # validation entry as current — this fit computes none
    flow._fit_validated = False

    flow.double()  # parameters + buffers (xmin/xmax) -> float64, one call
    t0 = time.perf_counter()
    try:
        vals = flow._tensorize(train_df)
        flow.train()
        opt = torch.optim.LBFGS(
            flow.parameters(),
            lr=1.0,
            max_iter=max_iter,
            history_size=history_size,
            tolerance_grad=0.0,  # |grad| never settles on the flat ridges
            tolerance_change=tol,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad()
            nll = torch.stack(
                [-lp.mean() for lp in flow.node_log_prob(vals).values()]
            ).sum()
            nll.backward()
            return nll

        opt.step(closure)
        n_iter = next(iter(opt.state.values()))["n_iter"]
        converged = n_iter < max_iter  # torch stopped on a tolerance
        with torch.no_grad():
            final_nll = float(
                torch.stack(
                    [-lp.mean() for lp in flow.node_log_prob(vals).values()]
                ).sum()
            )
        grad_norm = float(
            torch.nn.utils.get_total_norm(
                [p.grad for p in flow.parameters() if p.grad is not None]
            )
        )
        coefs = flow.ls_coefficients()  # read while still float64
    finally:
        flow.float()  # restore canonical float32 (lossy ~1e-7, harmless)
    flow.eval()

    report = {
        "converged": converged,
        "n_iter": n_iter,
        "final_nll": final_nll,
        "grad_norm": grad_norm,
        "seconds": time.perf_counter() - t0,
        "coefficients": coefs,
    }
    flow.history["classical"] = {k: v for k, v in report.items() if k != "coefficients"}
    return report
