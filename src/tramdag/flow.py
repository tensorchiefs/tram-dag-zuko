"""CausalFlowDAG — a single triangular normalizing flow on a user-defined DAG.

The flow maps iid standard-logistic latents ``U`` to the observed variables ``X``
in topological order; its Jacobian sparsity is exactly the DAG adjacency. The
joint log-likelihood decomposes per node, so one optimizer fits all nodes at once.

Causal queries:
    flow.sample(n)                    observational sampling
    flow.sample(n, do={"T": 1})       interventional sampling (graph mutilation)
    u = flow.abduct(df)               Pearl step 1 (latents from observations)
    flow.sample(do={"T": 1}, u=u)     Pearl steps 2+3 (counterfactuals)
    flow.pmf(df, node, do=...)        analytic per-row interventional PMF
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from .conditioners import (
    ComplexIntercept,
    ComplexShift,
    LinearShift,
    SimpleIntercept,
    VaryingCoef,
)
from .spec import (
    LS,
    ContinuousNode,
    NodeSpec,
    OrdinalNode,
    node_parents,
    node_terms,
    spec_from_dict,
    spec_to_dict,
    validate_and_sort,
)
from .transforms import (
    StandardLogistic,
    make_univariate_transform,
    ordinal_abduct,
    ordinal_log_prob,
    ordinal_pmf,
    ordinal_sample,
)

__all__ = ["CausalFlowDAG"]


class _VCGroup(NamedTuple):
    """One VC term of a node.

    The fields hold the treatment name, the modifier names, and whether the
    treatment is binary ordinal. ``center`` is ``False``, ``True``, or the name
    of a column in the training DataFrame. ``folds`` is the out-of-fold count.
    """

    on: str
    mods: tuple[str, ...]
    on_is_ord: bool
    center: bool | str
    folds: int


class _Node(nn.Module):
    """One dimension of the flow: intercept (transform params) + additive shifts."""

    def __init__(self, name: str, node: NodeSpec, spec: dict[str, NodeSpec]):
        super().__init__()
        self.name = name
        self.kind = node.kind
        terms = node_terms(node)
        self.parents = tuple(node_parents(node))  # ordered parent names
        i_groups = [tuple(t.parents) for t in terms if t.effect == "I" and t.parents]
        self._intercept_groups = i_groups
        self.ci_parents = [
            p for grp in i_groups for p in grp
        ]  # flat, for introspection

        if isinstance(node, ContinuousNode):
            self.ut = make_univariate_transform(node.transform, **node.transform_kwargs)
            n_params = self.ut.n_params
            self.levels = None
        else:
            self.ut = None
            self.levels = node.levels
            n_params = node.levels - 1

        def width(parent: str) -> int:
            pn = spec[parent]
            return pn.levels if isinstance(pn, OrdinalNode) else 1

        # intercept: no I-terms -> free SimpleIntercept theta_0; one I-term (single
        # or joint multi-parent) -> one ComplexIntercept IS theta (unchanged); two+
        # separate I-terms -> additive CI: one net per term summed in unconstrained
        # coefficient space (each parent reshapes the transform independently).
        if not i_groups:
            self.intercept = SimpleIntercept(n_params)
            self.intercept_nets = None
        elif len(i_groups) == 1:
            self.intercept = ComplexIntercept(
                sum(width(p) for p in i_groups[0]), n_params
            )
            self.intercept_nets = None
        else:
            self.intercept = None
            self.intercept_nets = nn.ModuleList(
                ComplexIntercept(sum(width(p) for p in grp), n_params)
                for grp in i_groups
            )

        # shift terms: one network per term, over the term's (possibly joint)
        # parents. Single-parent terms key the ModuleDict by the parent name (so
        # ls_coefficients/introspection keep working); a joint CS over several
        # parents keys by "a+b" and runs over their concatenated features. A VC
        # term keys by its treatment (on) name — validation guarantees `on` owns
        # that edge — and carries (on, modifiers, on-is-ordinal) in _vc_groups.
        self.shifts = nn.ModuleDict()
        self._shift_groups: list[tuple[str, tuple[str, ...]]] = []
        self._vc_groups: list[_VCGroup] = []
        for term in terms:
            if term.effect == "VC":
                on, mods = term.parents[0], tuple(term.parents[1:])
                self.shifts[on] = VaryingCoef(
                    sum(width(p) for p in mods), penalty=term.penalty
                )
                self._vc_groups.append(
                    _VCGroup(
                        on,
                        mods,
                        isinstance(spec[on], OrdinalNode),
                        term.center,
                        term.center_folds,
                    )
                )
                continue
            if term.effect not in ("LS", "CS"):
                continue
            ps = tuple(term.parents)
            key = ps[0] if len(ps) == 1 else "+".join(ps)
            feat_width = sum(width(p) for p in ps)
            self.shifts[key] = (
                LinearShift(feat_width)
                if term.effect == "LS"
                else ComplexShift(feat_width)
            )
            self._shift_groups.append((key, ps))

    def theta_shift(
        self, feats: dict[str, Tensor], n: int, vc_ehat: dict[str, Tensor] | None = None
    ) -> tuple[Tensor, Tensor]:
        """Transform parameters (n, P) and total shift (n,) from parent features.

        ``vc_ehat`` supplies the propensity ``e_hat(pa_on)`` per centered VC
        treatment (required whenever a term has ``center``): training passes the
        frozen out-of-fold values, inference paths the live full-fit ones.
        """
        if self.intercept_nets is not None:  # additive complex intercept
            theta = sum(
                net(torch.cat([feats[p] for p in grp], dim=1))
                for net, grp in zip(self.intercept_nets, self._intercept_groups)
            )
        elif self.ci_parents:  # single or joint complex intercept
            theta = self.intercept(
                torch.cat([feats[p] for p in self.ci_parents], dim=1)
            )
        else:  # simple (free) intercept
            theta = self.intercept(n)
        shift = torch.zeros(n, dtype=theta.dtype, device=theta.device)
        for key, ps in self._shift_groups:
            feat = (
                feats[ps[0]]
                if len(ps) == 1
                else torch.cat([feats[p] for p in ps], dim=1)
            )
            shift = shift + self.shifts[key](feat)
        for g in self._vc_groups:
            # treatment column raw: one-hot level-1 indicator for a binary
            # ordinal on; the (n, 1) value itself for a continuous on
            t = feats[g.on][:, -1:] if g.on_is_ord else feats[g.on]
            if g.center:
                if vc_ehat is None or g.on not in vc_ehat:
                    raise RuntimeError(
                        f"centered VC term on {g.on!r} needs e_hat; internal "
                        "callers must supply vc_ehat (never evaluate a centered "
                        "term without its propensity)."
                    )
                t = t - vc_ehat[g.on].view(-1, 1)  # regressor t - e_hat(x)
            mod_feat = torch.cat([feats[p] for p in g.mods], dim=1) if g.mods else None
            shift = shift + self.shifts[g.on](t, mod_feat)
        return theta, shift


class CausalFlowDAG(nn.Module):
    """A causal normalizing flow defined by ``spec = {name: NodeSpec}``."""

    def __init__(
        self, spec: dict[str, NodeSpec], device: str = "cpu", seed: int | None = None
    ):
        """Build the flow from ``spec``.

        Args:
            seed: if given, seeds weight initialisation deterministically
                (``torch.manual_seed`` is called before the nodes are
                constructed). Because init happens here, this is the single
                obvious knob for a reproducible model — ``fit(seed=...)`` only
                controls minibatch shuffling.
        """
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.spec = spec
        self.order = validate_and_sort(spec)
        self.nodes = nn.ModuleDict(
            {name: _Node(name, spec[name], spec) for name in self.order}
        )
        self.device = torch.device(device)
        self.history: dict = {"train": [], "val": [], "lr": [], "time": []}
        self.meta: dict = {}  # provenance attached at save() (machine, versions)
        self.vc_center_info: dict = {}  # OOF bookkeeping of centered VC terms (fit)
        self.to(self.device)

    # ------------------------------------------------------------------ data
    def _encode_parent(self, name: str, values: Tensor) -> Tensor:
        """Encode the values of a node for use as a parent feature.

        This follows the original TRAM-DAG convention. A continuous parent stays
        raw, shape ``(n, 1)``. An ordinal parent is one-hot encoded, shape
        ``(n, levels)``.
        """
        node = self.spec[name]
        if isinstance(node, OrdinalNode):
            return torch.nn.functional.one_hot(
                values.long(), num_classes=node.levels
            ).to(values.dtype)
        return values.view(-1, 1)

    @property
    def _dtype(self) -> torch.dtype:
        """Current model dtype (float32 normally; float64 inside fit_classical)."""
        return next(self.parameters()).dtype

    def _tensorize(self, df: pd.DataFrame) -> dict[str, Tensor]:
        np_dtype = np.float64 if self._dtype == torch.float64 else np.float32
        out = {}
        for name in self.order:
            vals = torch.as_tensor(
                df[name].to_numpy(dtype=np_dtype), device=self.device
            )
            out[name] = vals
        return out

    def _features(self, values: dict[str, Tensor]) -> dict[str, Tensor]:
        return {name: self._encode_parent(name, vals) for name, vals in values.items()}

    # ------------------------------------------- centered-VC propensity (e_hat)
    def _vc_ehat_live(
        self, nd: _Node, values: dict[str, Tensor], n: int
    ) -> dict[str, Tensor] | None:
        """Recompute ``e_hat(pa_on) = P(on = 1 | pa_on)`` for the centered VC terms.

        The value comes from this flow's own fitted ``on`` node, as a full-data
        propensity fit. That is the DML prediction convention. Training uses
        frozen out-of-fold values instead, see :meth:`fit`.

        The result is detached, so no gradient reaches the ``on`` node from the
        loss of this node.

        The function derives the value from the current parent values, so
        ``do``-mutilated sampling uses ``t - e_hat(x)`` with the intervened ``t``
        and the observed ``x``. It never reads a cached value.
        """
        out = {}
        for g in nd._vc_groups:
            if not g.center:
                continue
            on_nd = self.nodes[g.on]
            feats = self._features({p: values[p] for p in on_nd.parents})
            theta, shift = on_nd.theta_shift(
                feats, n, vc_ehat=self._vc_ehat_live(on_nd, values, n)
            )
            # binary ordinal on: P(on <= 0) = sigmoid(theta_0 - s),
            # so e = sigmoid(s - theta_0)
            out[g.on] = torch.sigmoid(shift - theta[:, 0]).detach()
        return out or None

    def _vc_ehat_columns(self, nd: _Node) -> list[str]:
        """List the extra columns needed for the centered VC terms of ``nd``.

        These are the columns beyond ``nd.parents``, namely the parents of the
        treatment nodes, found recursively.
        """
        cols: list[str] = []
        for g in nd._vc_groups:
            if not g.center:
                continue
            on_nd = self.nodes[g.on]
            cols += [p for p in on_nd.parents] + self._vc_ehat_columns(on_nd)
        return [c for c in dict.fromkeys(cols) if c not in nd.parents]

    # ------------------------------------------------------------- likelihood
    def node_log_prob(
        self,
        values: dict[str, Tensor],
        nodes: list[str] | None = None,
        vc_ehat: dict[str, dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        """Per-node log-likelihood contributions, each (n,).

        ``nodes`` restricts computation to a subset (used to skip frozen nodes
        during training — valid because the per-node losses are independent).
        ``vc_ehat`` ({node: {on: e_hat}}) overrides the propensity used by
        centered VC terms — ``fit`` passes the frozen **out-of-fold** values for
        the training rows; when omitted, the live full-fit propensity is
        recomputed from the flow's own treatment node.
        """
        feats = self._features(values)
        n = next(iter(values.values())).shape[0]
        out = {}
        for name in self.order if nodes is None else nodes:
            node = self.nodes[name]
            ehat = (
                vc_ehat.get(name)
                if vc_ehat is not None
                else self._vc_ehat_live(node, values, n)
            )
            theta, shift = node.theta_shift(feats, n, vc_ehat=ehat)
            x = values[name]
            if node.kind == "continuous":
                z0, ladj = node.ut.forward(theta, x)
                z = z0 + shift
                out[name] = StandardLogistic.log_prob(z) + ladj
            else:
                out[name] = ordinal_log_prob(theta, shift, x)
        return out

    def log_prob(self, df: pd.DataFrame) -> Tensor:
        """Joint log-likelihood log p(x) per row, shape (n,)."""
        per_node = self.node_log_prob(self._tensorize(df))
        return torch.stack(list(per_node.values()), dim=0).sum(dim=0)

    def nll(self, df: pd.DataFrame) -> dict[str, float]:
        """Mean negative log-likelihood per node (diagnostic)."""
        with torch.no_grad():
            per_node = self.node_log_prob(self._tensorize(df))
        return {k: float(-v.mean()) for k, v in per_node.items()}

    # ------------------------------------------------------------------- fit
    def _set_ranges(self, train_df: pd.DataFrame, marginal_init: bool = False) -> None:
        """Map the train 5%/95% quantiles onto the transform domain.

        This is the min-max scaling of the original implementation.

        ``marginal_init``: opt-in calibrated Bernstein init (see ``fit``). Applied only
        on the first fit (the same ``not ut._fitted`` guard as range-setting), so a
        multi-phase fit does not reset a partially-trained intercept.
        """
        from .transforms import BernsteinUT, ordinal_marginal_init_theta

        for name in self.order:
            node = self.nodes[name]
            if node.kind == "continuous" and not node.ut._fitted:
                q = train_df[name].quantile([0.05, 0.95])
                node.ut.set_range(q.iloc[0], q.iloc[1])
                if (
                    marginal_init
                    and isinstance(node.ut, BernsteinUT)
                    and isinstance(node.intercept, SimpleIntercept)
                ):
                    with torch.no_grad():
                        node.intercept.theta.copy_(node.ut.marginal_init_theta())
            elif (
                node.kind == "ordinal"
                and marginal_init
                and isinstance(node.intercept, SimpleIntercept)
                and not getattr(node.intercept, "_marginal_inited", False)
            ):
                # calibrate unconditional cutpoints to the marginal class log-odds
                counts = np.bincount(
                    train_df[name].to_numpy().astype(np.int64),
                    minlength=self.spec[name].levels,
                )
                with torch.no_grad():
                    node.intercept.theta.copy_(ordinal_marginal_init_theta(counts))
                node.intercept._marginal_inited = True

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
        epochs: int = 500,
        learning_rate: float = 1e-2,
        batch_size: int = 512,
        verbose: int = 50,
        seed: int | None = None,
        restore_best: bool = False,
        schedule: str | None = None,
        plateau_patience: int = 15,
        freeze_patience: int | None = None,
        min_delta: float = 1e-4,
        marginal_init: bool = False,
        vc_warm_start: bool = True,
    ) -> CausalFlowDAG:
        """Jointly fit all nodes by maximum likelihood.

        By default training keeps the **final** (converged) weights, so an
        all-``ls`` model trained to convergence reproduces the classical maximum
        likelihood estimate exactly (e.g. matches ``statsmodels``/``polr``).

        The optimizer holds one parameter group per node. Because the joint NLL
        decomposes per node with independent gradients, per-node learning rates
        and freezing are exactly equivalent to independent per-node training.

        Args:
            val_df: optional held-out set, used only for monitoring (and for
                ``restore_best``, ``schedule="plateau"`` and ``freeze_patience``).
                If omitted, the training set is used for the validation metric.
            restore_best: if True, snapshot each node's best-validation weights
                during training and restore them at the end. This is a mild
                early-stopping regularization and the convention of the original
                implementation. The fit is then *not* the training-data MLE, so
                leave it False for an exact classical comparison. Default False.
            schedule: learning-rate schedule. ``None`` = constant (the classic
                behavior); ``"onecycle"`` = ``OneCycleLR`` (warmup to
                ``learning_rate``, then anneal; stepped per batch);
                ``"cosine"`` = ``CosineAnnealingLR`` over ``epochs``;
                ``"plateau"`` = **per-node** decay: a node's lr is multiplied by
                0.3 whenever its own validation NLL hasn't improved by
                ``min_delta`` for ``plateau_patience`` epochs (floor 1e-3 ×
                ``learning_rate``).
            freeze_patience: if set, a node whose validation NLL hasn't improved
                by ``min_delta`` for this many epochs is **frozen** — excluded
                from the loss and backward pass (a real compute saving, since
                per-node losses are independent). When every node is frozen the
                fit returns early. Freeze epochs are recorded in
                ``history["frozen"]``.
            marginal_init: if True, calibrate each *unconditional* node's intercept
                to its marginal at init, instead of zuko's default zero init.
                Bernstein continuous nodes -> the linear map of the pre-scaled
                domain onto the standard-logistic 5%/95% quantiles (default is
                ~2.5x too steep); ordinal nodes -> cutpoints set to the empirical
                class log-odds (default zeros = near-uniform). Pure init — the
                converged MLE is unchanged — applied once (first fit only).
                Opt-in; default off. Affects only ``SimpleIntercept`` nodes
                (conditional ci intercepts are left untouched).
            vc_warm_start: if True (default), each ``VC`` term's ``beta0`` is
                initialised from the classical all-``ls`` solution of its node's
                conditional (deterministic L-BFGS on a throwaway proxy) before
                training, so the penalized head starts at the classical answer
                and only learns deviations. Applied once per term (a buffer that
                survives ``save``/``load`` guards re-runs). No-op without VC terms.

        For ``VC`` terms the objective is the **penalized** NLL on the
        total-likelihood scale — each term adds ``penalty * ||b_theta weights||^2``
        to the summed NLL, i.e. ``penalty * ||w||^2 / n_train`` to the mean loss
        (a fixed Gaussian prior: the shrinkage vanishes as n grows, the classical
        penalized-likelihood convention; ``beta0`` unpenalized). The recorded
        ``history`` NLLs stay pure likelihoods. After training, each ``b_theta``
        is re-centered to mean zero over the training data (function-preserving;
        the constant moves into ``beta0``). ``VC(center=...)`` terms run a
        stage-1 out-of-fold propensity computation before the loop
        (:meth:`_vc_oof_stage`); the training loss uses those frozen OOF values,
        while the epoch-level validation monitor (and every post-fit query)
        uses the live full-fit treatment node.

        Calling ``fit`` again continues training (e.g. a second phase with a
        lower learning rate); freezing state does not carry across calls.
        """
        if schedule not in (None, "onecycle", "cosine", "plateau"):
            raise ValueError(f"unknown schedule {schedule!r}")
        if seed is not None:
            torch.manual_seed(seed)
        self._set_ranges(train_df, marginal_init=marginal_init)
        if vc_warm_start:
            self._vc_warm_start(train_df)
        # VC effect heads whose L2 penalty joins the loss, per owning node
        vc_penalized = {
            name: [
                self.nodes[name].shifts[g.on]
                for g in self.nodes[name]._vc_groups
                if g.mods and self.nodes[name].shifts[g.on].penalty > 0
            ]
            for name in self.order
        }
        # stage 1 for centered VC terms (issue #30): frozen OUT-OF-FOLD e_hat
        # for the training rows — a plain tensor, so the Y-node loss has no
        # gradient path into the treatment node (per-node factorization intact).
        vc_ehat_train = self._vc_oof_stage(train_df)

        train_vals = self._tensorize(train_df)
        val_vals = self._tensorize(val_df) if val_df is not None else train_vals
        n = len(train_df)
        steps_per_epoch = (n + batch_size - 1) // batch_size

        opt = torch.optim.Adam(
            [
                {
                    "params": list(self.nodes[name].parameters()),
                    "lr": learning_rate,
                    "node": name,
                }
                for name in self.order
            ]
        )
        sched = None
        if schedule == "onecycle":
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=learning_rate, total_steps=epochs * steps_per_epoch
            )
        elif schedule == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=epochs, eta_min=learning_rate * 1e-3
            )

        if restore_best and not hasattr(self, "_best"):
            self._best = {name: (float("inf"), None) for name in self.order}
        best = self._best if restore_best else None
        # per-node plateau/freeze bookkeeping (local to this fit call)
        node_best = {name: float("inf") for name in self.order}
        node_bad = {name: 0 for name in self.order}
        frozen: set[str] = set()
        t0 = time.perf_counter()
        t_offset = self.history["time"][-1] if self.history.get("time") else 0.0
        prev_train: dict[str, float] = {}

        for epoch in range(epochs):
            self.train()
            active = [name for name in self.order if name not in frozen]
            perm = torch.randperm(n, device=self.device)
            train_acc = {name: prev_train.get(name, float("nan")) for name in frozen}
            train_acc.update({name: 0.0 for name in active})
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                batch = {k: v[idx] for k, v in train_vals.items()}
                ehat_batch = (
                    None
                    if vc_ehat_train is None
                    else {
                        nm: {on: e[idx] for on, e in d.items()}
                        for nm, d in vc_ehat_train.items()
                    }
                )
                per_node = self.node_log_prob(batch, nodes=active, vc_ehat=ehat_batch)
                node_nlls = {k: -v.mean() for k, v in per_node.items()}
                loss = torch.stack(list(node_nlls.values())).sum()
                for name in active:  # VC penalty (excluded from history)
                    for m in vc_penalized[name]:
                        loss = loss + m.penalty * m.l2() / n
                opt.zero_grad()
                loss.backward()
                opt.step()
                if schedule == "onecycle":
                    sched.step()
                w = len(idx) / n
                for k, v in node_nlls.items():
                    train_acc[k] += float(v.detach()) * w
            if schedule == "cosine":
                sched.step()
            prev_train = train_acc

            self.eval()
            with torch.no_grad():
                val_per_node = {
                    k: float(-v.mean()) for k, v in self.node_log_prob(val_vals).items()
                }
            self.history["train"].append(train_acc)
            self.history["val"].append(val_per_node)
            self.history.setdefault("lr", []).append(
                max(g["lr"] for g in opt.param_groups)
            )
            self.history.setdefault("time", []).append(
                t_offset + time.perf_counter() - t0
            )

            # per-node improvement tracking (plateau decay + freezing)
            for g in opt.param_groups:
                name = g["node"]
                if name in frozen:
                    continue
                if val_per_node[name] < node_best[name] - min_delta:
                    node_best[name] = val_per_node[name]
                    node_bad[name] = 0
                else:
                    node_bad[name] += 1
                if (
                    schedule == "plateau"
                    and node_bad[name] > 0
                    and node_bad[name] % plateau_patience == 0
                ):
                    g["lr"] = max(g["lr"] * 0.3, learning_rate * 1e-3)
                # under "plateau", only freeze nodes whose lr has already been
                # decayed substantially — otherwise a node can freeze while a
                # smaller lr would still make progress toward the optimum
                lr_decayed = schedule != "plateau" or g[
                    "lr"
                ] <= learning_rate * 1e-2 * (1 + 1e-9)
                if (
                    freeze_patience is not None
                    and lr_decayed
                    and node_bad[name] >= freeze_patience
                ):
                    frozen.add(name)
                    self.history.setdefault("frozen", {}).setdefault(
                        name, len(self.history["val"])
                    )  # 1-based global epoch

            if restore_best:
                for name in self.order:
                    if val_per_node[name] < best[name][0]:
                        best[name] = (
                            val_per_node[name],
                            copy.deepcopy(self.nodes[name].state_dict()),
                        )

            if verbose and (epoch % verbose == 0 or epoch == epochs - 1):
                tot_t = sum(train_acc.values())
                tot_v = sum(val_per_node.values())
                print(
                    f"[epoch {epoch + 1:5d}/{epochs}] train NLL {tot_t:.4f}  "
                    f"val NLL {tot_v:.4f}"
                    + (f"  frozen {sorted(frozen)}" if frozen else "")
                )

            if len(frozen) == len(self.order):  # everything converged
                if verbose:
                    print(f"[epoch {epoch + 1:5d}] all nodes frozen — stopping.")
                break

        if restore_best:  # restore per-node best-validation weights
            for name, (_, state) in best.items():
                if state is not None:
                    self.nodes[name].load_state_dict(state)
        self._recenter_vc(train_vals)
        self.eval()
        return self

    # ------------------------------------------------- varying-coefficient (VC)
    def _vc_warm_start(self, train_df: pd.DataFrame) -> None:
        """Initialise every VC term's ``beta0`` from the classical solution.

        The value comes from the all-``ls`` solution of the node's conditional.
        Issue #28 recommends this warm start.

        A throwaway proxy of the node (same kind/transform, every parent an LS
        term, parent marginals irrelevant to the conditional because the joint
        NLL decomposes per node) is fitted with the deterministic
        :meth:`fit_classical`, and the ``on`` coefficient copied into ``beta0``
        (for a binary ordinal treatment, the identified one-hot difference
        ``w[1] - w[0]``). ``b_theta`` already starts at the zero function
        (zero-initialised output layer). Runs once per term — the
        ``warm_started`` buffer survives ``save``/``load``.
        """
        for name in self.order:
            nd = self.nodes[name]
            todo = [g for g in nd._vc_groups if not bool(nd.shifts[g[0]].warm_started)]
            if not todo:
                continue
            node_spec = self.spec[name]
            proxy_spec: dict[str, NodeSpec] = {}
            for p in nd.parents:
                pn = self.spec[p]
                proxy_spec[p] = (
                    OrdinalNode(levels=pn.levels)
                    if isinstance(pn, OrdinalNode)
                    else ContinuousNode(transform="affine")
                )
            ls_terms = [LS(p) for p in nd.parents]
            if isinstance(node_spec, OrdinalNode):
                proxy_spec[name] = OrdinalNode(levels=node_spec.levels, terms=ls_terms)
            else:
                proxy_spec[name] = ContinuousNode(
                    transform=node_spec.transform,
                    transform_kwargs=dict(node_spec.transform_kwargs),
                    terms=ls_terms,
                )
            proxy = CausalFlowDAG(proxy_spec, device=str(self.device))
            proxy.fit_classical(train_df[list(nd.parents) + [name]], verbose=False)
            for g in todo:
                w = proxy.nodes[name].shifts[g.on].weight.detach()
                b0 = float(w[-1] - w[0]) if g.on_is_ord else float(w[0])
                m = nd.shifts[g.on]
                with torch.no_grad():
                    m.beta0.fill_(b0)
                m.warm_started.fill_(True)

    @torch.no_grad()
    def _predict_p1(self, on: str, df: pd.DataFrame) -> np.ndarray:
        """Give ``P(on = 1 | pa_on)`` from this flow's ``on`` node.

        The treatment is binary ordinal, so the value is
        ``sigmoid(shift - theta_0)``.
        """
        nd = self.nodes[on]
        np_dtype = np.float64 if self._dtype == torch.float64 else np.float32
        values = {
            p: torch.as_tensor(df[p].to_numpy(dtype=np_dtype), device=self.device)
            for p in nd.parents
        }
        feats = self._features(values)
        theta, shift = nd.theta_shift(
            feats, len(df), vc_ehat=self._vc_ehat_live(nd, values, len(df))
        )
        return torch.sigmoid(shift - theta[:, 0]).cpu().numpy()

    def _vc_oof_stage(
        self, train_df: pd.DataFrame
    ) -> dict[str, dict[str, Tensor]] | None:
        """Compute stage 1 of the two-stage centered-VC design, issue #30.

        The result holds the frozen training-time propensities, as
        ``{node: {on: (n,) tensor}}``.

        For ``center=True`` the values are **out-of-fold** — K refits of the
        treatment node only, each predicting its held-out fold (the DML
        cross-fitting requirement; in-sample e_hat reintroduces the
        own-observation bias and can be worse than no centering). For
        ``center="col"`` the user-supplied cross-fitted column is taken as-is.
        Bookkeeping lands in ``self.vc_center_info[(node, on)]`` (``e_oof``,
        ``fold_id``, ``folds``, ``source``) so tests can assert the fold
        structure — a later "simplification" to in-sample e_hat fails CI.
        """
        jobs = [
            (name, g)
            for name in self.order
            for g in self.nodes[name]._vc_groups
            if g.center
        ]
        if not jobs:
            return None
        self.vc_center_info = {}
        np_dtype = np.float64 if self._dtype == torch.float64 else np.float32
        out: dict[str, dict[str, Tensor]] = {}
        rng_state = torch.get_rng_state()  # proxies reseed; keep fit reproducible
        try:
            for name, g in jobs:
                if isinstance(g.center, str):  # user-supplied cross-fitted col
                    if g.center not in train_df.columns:
                        raise KeyError(f"center column {g.center!r} not in train_df.")
                    e = train_df[g.center].to_numpy(dtype=np.float64)
                    if not ((e > 0.0) & (e < 1.0)).all():
                        raise ValueError(
                            f"center column {g.center!r} must hold propensities "
                            "strictly inside (0, 1)."
                        )
                    fold_id = None
                else:
                    e, fold_id = self._vc_oof_propensity(g.on, train_df, g.folds)
                out.setdefault(name, {})[g.on] = torch.as_tensor(
                    e.astype(np_dtype), device=self.device
                )
                self.vc_center_info[(name, g.on)] = {
                    "source": g.center if isinstance(g.center, str) else "oof-refit",
                    "folds": None if fold_id is None else int(g.folds),
                    "fold_id": fold_id,
                    "e_oof": e.copy(),
                    "n": len(train_df),
                }
        finally:
            torch.set_rng_state(rng_state)
        return out

    def _vc_oof_propensity(
        self, on: str, train_df: pd.DataFrame, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the out-of-fold ``P(on=1|pa_on)``.

        The function refits the ``on`` node K times, and only that node. Each
        refit uses a single-node proxy whose parents are sources, because their
        marginals cannot influence the conditional. Each refit then predicts the
        fold it never saw.

        A treatment with all-``ls`` terms uses :meth:`fit_classical`, which is
        deterministic and takes seconds. Any other treatment uses a
        fixed-budget Adam fit.
        """
        on_nd = self.nodes[on]
        if any(g.center for g in on_nd._vc_groups):
            raise NotImplementedError(
                f"treatment node {on!r} itself has a centered VC term; "
                "chained centering is not supported."
            )
        node_spec = self.spec[on]
        proxy_spec: dict[str, NodeSpec] = {}
        for p in on_nd.parents:
            pn = self.spec[p]
            proxy_spec[p] = (
                OrdinalNode(levels=pn.levels)
                if isinstance(pn, OrdinalNode)
                else ContinuousNode(transform="affine")
            )
        terms = list(node_spec.terms) if node_spec.terms else None
        proxy_spec[on] = OrdinalNode(levels=2, terms=terms)
        all_ls = all(t.effect == "LS" for t in (terms or []))
        cols = list(on_nd.parents) + [on]

        n = len(train_df)
        fold_id = np.random.default_rng(0).permutation(n) % k
        e = np.empty(n, dtype=np.float64)
        for j in range(k):
            proxy = CausalFlowDAG(proxy_spec, device=str(self.device), seed=0)
            held_in = train_df.iloc[fold_id != j][cols]
            if all_ls:
                proxy.fit_classical(held_in, verbose=False)
            else:
                proxy.fit(
                    held_in,
                    epochs=300,
                    learning_rate=1e-2,
                    verbose=0,
                    seed=0,
                    restore_best=False,
                )
            e[fold_id == j] = proxy._predict_p1(on, train_df.iloc[fold_id == j])
        return e, fold_id

    @torch.no_grad()
    def _recenter_vc(self, values: dict[str, Tensor]) -> None:
        """Re-split every VC term so ``b_theta`` sums to zero over the train rows.

        The removed constant moves into ``beta0``, so the modelled function does
        not change.
        """
        feats: dict[str, Tensor] | None = None
        for name in self.order:
            nd = self.nodes[name]
            for g in nd._vc_groups:
                if not g.mods:
                    continue
                if feats is None:
                    feats = self._features(values)
                nd.shifts[g.on].recenter(torch.cat([feats[p] for p in g.mods], dim=1))

    @torch.no_grad()
    def varying_coef(
        self, node: str, data: pd.DataFrame, on: str | None = None
    ) -> np.ndarray:
        """Evaluate the fitted effect function ``beta(x)`` of a ``VC`` term.

        The function reads the rows of ``data``. It is the first-class read-out
        of issue #28.

        The value comes in closed form from the fitted term, as
        ``beta0 + b_theta(modifiers)``. It is deterministic, it is free of ``y``
        because only the modifier columns of ``data`` are read, and it needs no
        abduction. For a binary treatment it is identical to the abduction
        difference ``u(x, t=1, y) - u(x, t=0, y)``.

        The value lives on the latent, log-odds scale of the node. A continuous
        node adds it. An ordinal node subtracts it from the cutpoints.

        For a centered term, that is ``center=...``, the form of the returned
        ``beta`` does not change, but ``beta0`` then reads as the effect at the
        treatment margin, which is the observed propensities.

        Args:
            node: name of the node carrying the VC term.
            on: the VC term's treatment name; optional when the node has exactly
                one VC term.

        Returns an ``(n,)`` array of ``beta`` values (constant when the term has
        no modifiers).
        """
        if node not in self.nodes:
            raise KeyError(f"unknown node {node!r}")
        nd = self.nodes[node]
        vcs = {g.on: g.mods for g in nd._vc_groups}
        if not vcs:
            raise ValueError(f"node {node!r} has no VC term.")
        if on is None:
            if len(vcs) > 1:
                raise ValueError(
                    f"node {node!r} has several VC terms ({sorted(vcs)}); "
                    "pass on=<treatment name>."
                )
            on = next(iter(vcs))
        if on not in vcs:
            raise KeyError(
                f"node {node!r} has no VC term on {on!r} (has {sorted(vcs)})."
            )
        mods = vcs[on]
        missing = [p for p in mods if p not in data.columns]
        if missing:
            raise KeyError(f"data is missing modifier column(s): {missing}")
        mod_feat = None
        if mods:
            np_dtype = np.float64 if self._dtype == torch.float64 else np.float32
            vals = {
                p: torch.as_tensor(data[p].to_numpy(dtype=np_dtype), device=self.device)
                for p in mods
            }
            feats = self._features(vals)
            mod_feat = torch.cat([feats[p] for p in mods], dim=1)
        return nd.shifts[on].beta(mod_feat, len(data)).cpu().numpy()

    # --------------------------------------------------------- classical fit
    def _is_all_ls(self) -> bool:
        return all(
            term.effect == "LS"
            for node in self.spec.values()
            for term in node_terms(node)
        )

    def ls_coefficients(self) -> dict[str, dict[str, np.ndarray]]:
        """Give the per-node linear-shift weights, as ``{node: {parent: array}}``.

        For an all-``ls`` model these are the interpretable log-odds-ratio
        coefficients.
        """
        out: dict[str, dict[str, np.ndarray]] = {}
        for name in self.order:
            shifts = self.nodes[name].shifts
            if shifts:
                out[name] = {
                    p: m.weight.detach().cpu().numpy().ravel().copy()
                    for p, m in shifts.items()
                }
        return out

    def to_matrix(self) -> pd.DataFrame:
        """Give the labelled adjacency matrix of term effects.

        Rows are parents and columns are children. This is the meta-adjacency
        view of the paper. A cell holds ``"LS"``, ``"CS"`` or ``"CI"``, and an
        empty cell means there is no edge. A multi-parent term carries its parent
        group as a suffix.
        """
        labels = {"I": "CI", "LS": "LS", "CS": "CS"}
        m = pd.DataFrame("", index=list(self.order), columns=list(self.order))
        for child in self.order:
            for term in node_terms(self.spec[child]):
                if term.effect == "VC":  # treatment cell "VC", modifiers "VCm"
                    cells = [(term.parents[0], "VC")] + [
                        (p, "VCm") for p in term.parents[1:]
                    ]
                else:
                    tag = labels[term.effect]
                    if len(term.parents) > 1:
                        tag = f"{tag}{list(term.parents)}"
                    cells = [(p, tag) for p in term.parents]
                for p, tag in cells:  # a VC modifier may share its cell with
                    cur = m.loc[p, child]  # a prognostic term -> join with "+"
                    m.loc[p, child] = f"{cur}+{tag}" if cur else tag
        return m

    @torch.no_grad()
    def intercept_contributions(self, node: str, data: pd.DataFrame) -> dict:
        """Decompose a complex intercept into mean-centered per-term parts.

        The parts are contributions to the transform parameters of the node. Use
        them to plot additive partial effects.

        An additive complex intercept ``terms=[I("x1"), I("x2")]`` builds one
        network per ``I``-term and **sums their outputs in unconstrained
        parameter space**: ``theta(pa) = net_1(x1) + net_2(x2)``. The sum is
        identified (so every L1/L2/L3 query is correct), but each term's output is
        identified only up to a constant — a constant moves freely between the
        nets. This makes the *raw* per-term outputs not directly comparable.

        Following the usual additive-model / GAM convention, this resolves the
        ambiguity by a **sum-to-zero (mean-centering) constraint applied over the
        rows of** ``data``: each term's contribution is centered to mean zero
        (per parameter), and the removed constants are collected into a single
        ``baseline``. The decomposition is exact —

            ``theta(pa) = baseline + sum_terms contribution_term(pa)``

        — so ``baseline`` plus the (uncentered) row sum of the contributions
        reproduces the model's transform parameters. This is **post-hoc only**:
        it reads the fitted weights and changes nothing about the model or any
        frozen number (issue #20, Option A). Shift terms (``LS``/``CS``) are a
        separate, already-interpretable slot — see :meth:`ls_coefficients`.

        Args:
            node: name of a node with at least one complex-intercept (``I``) term
                that has parents.
            data: rows over which to center (and at which to evaluate the
                contributions); must contain every intercept-parent column.

        Returns a dict with:
            ``"baseline"``: ``(P,)`` array — the absorbed constant (sum of the
                per-term means), where ``P`` is the node's transform-parameter
                count: ``ut.n_params`` for a continuous node (e.g. Bernstein
                coefficients, the [widths|heights|derivatives] block of an RQ
                spline, or the 2 affine parameters), ``levels - 1`` cutpoint
                parameters for an ordinal node. The contributions live in the
                transform's **unconstrained** parameter space (where the model
                sums the additive terms, before the monotonicity constraint), so
                they are exact partial effects on those parameters but not, in
                general, an additive shift of the curve itself.
            ``"contributions"``: ``{term_label: (n, P) array}`` — each term's
                mean-centered contribution at each row (columns sum to ~0 over
                rows). ``term_label`` is the term's parents joined by ``"+"``.
            ``"parents"``: ``{term_label: tuple(parent_names)}``.
        """
        if node not in self.nodes:
            raise KeyError(f"unknown node {node!r}")
        nd = self.nodes[node]
        groups = nd._intercept_groups
        if not groups:
            raise ValueError(
                f"node {node!r} has no complex-intercept (I) terms with parents; "
                "there is nothing to decompose (its intercept is unconditional)."
            )
        missing = [p for p in nd.ci_parents if p not in data.columns]
        if missing:
            raise KeyError(f"data is missing intercept-parent column(s): {missing}")

        np_dtype = np.float64 if self._dtype == torch.float64 else np.float32
        vals = {
            p: torch.as_tensor(data[p].to_numpy(dtype=np_dtype), device=self.device)
            for p in nd.ci_parents
        }
        feats = self._features(vals)
        # one net per group: the additive case stores them in intercept_nets;
        # a single (possibly joint) I-term is the lone `intercept` network.
        nets = (
            list(nd.intercept_nets) if nd.intercept_nets is not None else [nd.intercept]
        )

        contributions: dict[str, np.ndarray] = {}
        parents: dict[str, tuple] = {}
        baseline = None
        for net, grp in zip(nets, groups):
            raw = net(torch.cat([feats[p] for p in grp], dim=1))  # (n, P)
            mean = raw.mean(dim=0, keepdim=True)  # (1, P)
            label = "+".join(grp)
            contributions[label] = (raw - mean).cpu().numpy()
            parents[label] = grp
            baseline = mean if baseline is None else baseline + mean
        return {
            "baseline": baseline.cpu().numpy().ravel(),
            "contributions": contributions,
            "parents": parents,
        }

    def fit_classical(
        self,
        train_df: pd.DataFrame,
        *,
        max_iter: int = 400,
        tol: float = 1e-6,
        verbose: bool = True,
    ) -> dict:
        """Fit an all-``ls`` model the classical way.

        The fit uses full batches, float64, and L-BFGS with a strong-Wolfe line
        search. There are no minibatches, no schedule and no early stopping, so
        the fit is deterministic and bit-reproducible. It lands on the exact
        maximum-likelihood estimate and matches classical software, that is
        ``statsmodels`` ``OrderedModel`` and R ``polr`` or ``Colr``. It is much
        faster than minibatch Adam.

        This method is valid only when every edge is ``ls``, because each
        node-conditional is then a classical transformation model. Any other
        model raises. For a ``cs`` or ``ci`` model use :meth:`fit`, where the
        minibatch noise also regularizes the MLPs.

        float64 is a transient compute mode. The model is upcast for the fit,
        and ``self.double()`` converts the parameters and the range buffers of
        the transforms in one call. Afterwards the model returns to float32, so
        the stored model and ``save``/``load`` stay float32. Double precision is
        what lets the line search resolve the optimum cleanly.

        Convergence is judged by **NLL flatness** (relative change < ``tol``
        between L-BFGS rounds). Note that ``|grad|`` and individual coefficients
        do *not* settle to machine precision: a continuous node's Bernstein
        intercept, and weakly-identified directions like rare one-hot levels or a
        flat treatment-effect ridge, keep drifting along near-zero-curvature
        valleys long after the likelihood (and the well-identified coefficients)
        have reached the MLE. Correctness is therefore verified by comparison to
        classical software (see ``experiments/validate_ls.py``), not by this flag.

        Returns a convergence report (iterations, final NLL, gradient norm,
        max coefficient change at the last round, wall-time, and the fitted
        :meth:`ls_coefficients`).
        """
        if not self._is_all_ls():
            raise ValueError(
                "fit_classical requires an all-`ls` spec (every edge term 'ls'); "
                "this spec has cs/ci/vc terms. Use fit() for flexible models."
            )
        self._set_ranges(train_df)

        self.double()  # parameters + buffers (xmin/xmax) -> float64, one call
        assert next(self.parameters()).dtype == torch.float64
        t0 = time.perf_counter()
        chunk = 25  # inner L-BFGS iterations per round; we stop on NLL change
        try:
            vals = self._tensorize(train_df)
            self.train()
            opt = torch.optim.LBFGS(
                self.parameters(),
                lr=1.0,
                max_iter=chunk,
                history_size=50,
                tolerance_grad=0.0,
                tolerance_change=0.0,
                line_search_fn="strong_wolfe",
            )

            def closure():
                opt.zero_grad()
                nll = torch.stack(
                    [-lp.mean() for lp in self.node_log_prob(vals).values()]
                ).sum()
                nll.backward()
                return nll

            def flat_coefs() -> np.ndarray:
                cs = self.ls_coefficients()
                return (
                    np.concatenate([w for node in cs.values() for w in node.values()])
                    if cs
                    else np.zeros(1)
                )

            prev_nll, prev_c, final_nll, n_iter, converged, coef_delta = (
                float("inf"),
                flat_coefs(),
                float("nan"),
                0,
                False,
                float("inf"),
            )
            for _ in range(max(1, max_iter // chunk)):
                final_nll = float(opt.step(closure))
                n_iter += chunk
                cur_c = flat_coefs()
                coef_delta = float(np.abs(cur_c - prev_c).max())
                prev_c = cur_c
                if abs(prev_nll - final_nll) < tol * (1.0 + abs(final_nll)):
                    converged = True
                    break
                prev_nll = final_nll
            grad_norm = float(
                torch.cat(
                    [
                        p.grad.reshape(-1)
                        for p in self.parameters()
                        if p.grad is not None
                    ]
                ).norm()
            )
            coefs = self.ls_coefficients()  # read while still float64
        finally:
            self.float()  # restore canonical float32 (lossy ~1e-7, harmless)
        self.eval()

        report = {
            "converged": converged,
            "n_iter": n_iter,
            "final_nll": final_nll,
            "grad_norm": grad_norm,
            "coef_delta": coef_delta,
            "seconds": time.perf_counter() - t0,
            "coefficients": coefs,
        }
        if verbose:
            print(
                f"fit_classical: {n_iter} L-BFGS iters, NLL {final_nll:.6f}, "
                f"{report['seconds']:.2f}s"
                + ("" if converged else f"  (NLL still moving at {max_iter} iters)")
            )
        return report

    # ------------------------------------------------------- causal queries
    @torch.no_grad()
    def sample(
        self,
        n: int | None = None,
        *,
        do: dict[str, float] | None = None,
        u: pd.DataFrame | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Sample from the (optionally mutilated) flow.

        Args:
            n: number of samples (ignored if ``u`` is given).
            do: interventions {node: value}; intervened nodes are clamped and
                their parent dependence removed (graph mutilation).
            u: latent variables (as returned by :meth:`abduct`). If given, they
                are pushed through the flow — together with ``do`` this yields
                counterfactuals (Pearl's abduction -> action -> prediction).
        """
        do = do or {}
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(seed)

        np_dtype = np.float64 if self._dtype == torch.float64 else np.float32
        if u is not None:
            n = len(u)
            u_vals = {
                name: torch.as_tensor(
                    u[name].to_numpy(dtype=np_dtype, copy=True), device=self.device
                )
                for name in self.order
            }
        elif n is not None:
            u_vals = {
                name: StandardLogistic.sample((n,), device=self.device)
                if gen is None
                else StandardLogistic.icdf(
                    torch.rand((n,), device=self.device, generator=gen)
                )
                for name in self.order
            }
        else:
            raise ValueError("Provide either n or u.")

        values: dict[str, Tensor] = {}
        for name in self.order:
            if name in do:
                values[name] = torch.full(
                    (n,), float(do[name]), dtype=self._dtype, device=self.device
                )
                continue
            node = self.nodes[name]
            feats = self._features({p: values[p] for p in node.parents})
            # centered VC: e_hat(pa_on) is re-derived from the already-sampled
            # ancestor values — under do the regressor is t_do - e_hat(x), never
            # a cached training value
            theta, shift = node.theta_shift(
                feats, n, vc_ehat=self._vc_ehat_live(node, values, n)
            )
            z = u_vals[name]
            if node.kind == "continuous":
                values[name] = node.ut.inverse(theta, z - shift)
            else:
                values[name] = ordinal_sample(theta, shift, z)
        return pd.DataFrame({k: v.cpu().numpy() for k, v in values.items()})

    @torch.no_grad()
    def abduct(self, df: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
        """Pearl abduction: recover the latent variables ``u`` from observations.

        Continuous nodes are inverted exactly (``u = h(x) + shift``); for ordinal
        nodes the latent is only interval-identified, so it is sampled from the
        standard logistic truncated to the observed level's interval.
        """
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(seed)
        values = self._tensorize(df)
        feats = self._features(values)
        n = len(df)
        u = {}
        for name in self.order:
            node = self.nodes[name]
            theta, shift = node.theta_shift(
                feats, n, vc_ehat=self._vc_ehat_live(node, values, n)
            )
            x = values[name]
            if node.kind == "continuous":
                z0, _ = node.ut.forward(theta, x)
                u[name] = z0 + shift
            else:
                u[name] = ordinal_abduct(theta, shift, x, generator=gen)
        return pd.DataFrame({k: v.cpu().numpy() for k, v in u.items()})

    @torch.no_grad()
    def pmf(
        self, df: pd.DataFrame, node: str, do: dict[str, float] | None = None
    ) -> np.ndarray:
        """Give the analytic class probabilities of an ordinal node.

        The result has shape ``(n, levels)``. The parents of the node come from
        ``df``, after the ``do`` overrides are applied.
        """
        if not isinstance(self.spec[node], OrdinalNode):
            raise ValueError(f"pmf() requires an ordinal node, '{node}' is continuous.")
        df_local = df.copy()
        for col, val in (do or {}).items():
            df_local[col] = val
        nd = self.nodes[node]
        np_dtype = np.float64 if self._dtype == torch.float64 else np.float32
        cols = list(nd.parents) + self._vc_ehat_columns(nd)  # + e_hat inputs
        values = {
            p: torch.as_tensor(df_local[p].to_numpy(dtype=np_dtype), device=self.device)
            for p in cols
        }
        feats = self._features({p: values[p] for p in nd.parents})
        theta, shift = nd.theta_shift(
            feats, len(df_local), vc_ehat=self._vc_ehat_live(nd, values, len(df_local))
        )
        return ordinal_pmf(theta, shift).cpu().numpy()

    # ------------------------------------------------------------------ scores
    @torch.no_grad()
    def scores(
        self, df: pd.DataFrame, node: str, params: str = "shift"
    ) -> pd.DataFrame:
        """Give the per-observation scores ``psi_i = d l_i / d theta``, issue #29.

        The scores belong to the interpretable shift coefficients of a node and
        are analytic and exact, see ``tramdag.scores``. ``params="shift"`` is the
        only option and covers every ``LS`` weight and the ``beta0`` of every
        ``VC`` term.

        At a fitted MLE each column sums to about zero. Order the rows by a
        covariate that truly modifies the treatment effect and the cumulative sum
        of the treatment column drifts. :meth:`effect_modifier_scan` measures
        that drift.

        This is a pure read-out. It touches no fitting or sampling code path.
        """
        if params != "shift":
            raise ValueError(f"params='shift' is the only option, got {params!r}.")
        from .scores import node_scores

        return node_scores(self, df, node)

    @torch.no_grad()
    def effect_modifier_scan(
        self, df: pd.DataFrame, node: str, on: str, candidates: list[str] | None = None
    ) -> pd.DataFrame:
        """Rank candidate effect modifiers with a Zeileis-Hornik fluctuation scan.

        Issue #29 describes the method. Each candidate covariate is ranked by how
        strongly the scores of the ``on`` coefficient drift when the rows are
        ordered by it. A cheap all-``ls`` fit is enough, so this gives a measured
        shortlist for ``VC`` modifiers.

        Returns
        -------
        pd.DataFrame
            One row per candidate, with ``stat``, ``p_value``, ``crit_5pct`` and
            ``flag``. See ``tramdag.scores.effect_modifier_scan``.
        """
        from .scores import effect_modifier_scan

        return effect_modifier_scan(self, df, node, on, candidates=candidates)

    # ------------------------------------------------------------------- io
    def save(self, path: str | Path) -> None:
        """Write the model, its history and its provenance to a checkpoint.

        The file holds the spec and the weights, the training ``history``, and a
        ``meta`` block with the tramdag version, the save time, the device, and
        the machine that trained the model. A cached run therefore stays
        self-describing: the file alone is enough to rebuild a training-curve
        plot or to compare timings.
        """
        from datetime import datetime, timezone

        from . import __version__
        from .env import machine_info

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "tramdag_version": __version__,
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "device": str(self.device),
            "machine": machine_info(),
        }
        torch.save(
            {
                "spec": spec_to_dict(self.spec),
                "state_dict": self.state_dict(),
                "history": self.history,
                "meta": meta,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> CausalFlowDAG:
        """Restore a model from a checkpoint.

        ``flow.history`` and ``flow.meta`` are refilled, so a cached model can
        still produce training and diagnostic plots, and can report the machine
        that trained it.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        flow = cls(spec_from_dict(ckpt["spec"]), device=device)
        for name in flow.order:  # mark transforms as fitted before loading buffers
            node = flow.nodes[name]
            if node.kind == "continuous":
                node.ut._fitted = True
        flow.load_state_dict(ckpt["state_dict"])
        flow.history = ckpt.get(
            "history", {"train": [], "val": [], "lr": [], "time": []}
        )
        flow.meta = ckpt.get("meta", {})
        flow.eval()
        return flow
