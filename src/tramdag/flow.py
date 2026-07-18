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
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from .conditioners import (ComplexIntercept, ComplexShift, LinearShift,
                           SimpleIntercept, VaryingCoef)
from .spec import (LS, ContinuousNode, NodeSpec, OrdinalNode, node_parents,
                   node_terms, spec_from_dict, spec_to_dict, validate_and_sort)
from .transforms import (StandardLogistic, make_univariate_transform,
                         ordinal_abduct, ordinal_log_prob, ordinal_pmf,
                         ordinal_sample)

__all__ = ["CausalFlowDAG"]


class _Node(nn.Module):
    """One dimension of the flow: intercept (transform params) + additive shifts."""

    def __init__(self, name: str, node: NodeSpec, spec: dict[str, NodeSpec]):
        super().__init__()
        self.name = name
        self.kind = node.kind
        terms = node_terms(node)
        self.parents = tuple(node_parents(node))   # ordered parent names
        i_groups = [tuple(t.parents) for t in terms if t.effect == "I" and t.parents]
        self._intercept_groups = i_groups
        self.ci_parents = [p for grp in i_groups for p in grp]   # flat, for introspection

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
            self.intercept = ComplexIntercept(sum(width(p) for p in i_groups[0]), n_params)
            self.intercept_nets = None
        else:
            self.intercept = None
            self.intercept_nets = nn.ModuleList(
                ComplexIntercept(sum(width(p) for p in grp), n_params) for grp in i_groups)

        # shift terms: one network per term, over the term's (possibly joint)
        # parents. Single-parent terms key the ModuleDict by the parent name (so
        # ls_coefficients/introspection keep working); a joint CS over several
        # parents keys by "a+b" and runs over their concatenated features. A VC
        # term keys by its treatment (on) name — validation guarantees `on` owns
        # that edge — and carries (on, modifiers, on-is-ordinal) in _vc_groups.
        self.shifts = nn.ModuleDict()
        self._shift_groups: list[tuple[str, tuple[str, ...]]] = []
        self._vc_groups: list[tuple[str, tuple[str, ...], bool]] = []
        for term in terms:
            if term.effect == "VC":
                on, mods = term.parents[0], tuple(term.parents[1:])
                self.shifts[on] = VaryingCoef(sum(width(p) for p in mods),
                                              penalty=term.penalty)
                self._vc_groups.append((on, mods, isinstance(spec[on], OrdinalNode)))
                continue
            if term.effect not in ("LS", "CS"):
                continue
            ps = tuple(term.parents)
            key = ps[0] if len(ps) == 1 else "+".join(ps)
            feat_width = sum(width(p) for p in ps)
            self.shifts[key] = (LinearShift(feat_width) if term.effect == "LS"
                                else ComplexShift(feat_width))
            self._shift_groups.append((key, ps))

    def theta_shift(self, feats: dict[str, Tensor], n: int) -> tuple[Tensor, Tensor]:
        """Transform parameters (n, P) and total shift (n,) from parent features."""
        if self.intercept_nets is not None:          # additive complex intercept
            theta = sum(net(torch.cat([feats[p] for p in grp], dim=1))
                        for net, grp in zip(self.intercept_nets, self._intercept_groups))
        elif self.ci_parents:                        # single or joint complex intercept
            theta = self.intercept(torch.cat([feats[p] for p in self.ci_parents], dim=1))
        else:                                        # simple (free) intercept
            theta = self.intercept(n)
        shift = torch.zeros(n, dtype=theta.dtype, device=theta.device)
        for key, ps in self._shift_groups:
            feat = feats[ps[0]] if len(ps) == 1 else torch.cat([feats[p] for p in ps], dim=1)
            shift = shift + self.shifts[key](feat)
        for on, mods, on_is_ord in self._vc_groups:
            # treatment column raw: one-hot level-1 indicator for a binary
            # ordinal on; the (n, 1) value itself for a continuous on
            t = feats[on][:, -1:] if on_is_ord else feats[on]
            mod_feat = torch.cat([feats[p] for p in mods], dim=1) if mods else None
            shift = shift + self.shifts[on](t, mod_feat)
        return theta, shift


class CausalFlowDAG(nn.Module):
    """A causal normalizing flow defined by ``spec = {name: NodeSpec}``."""

    def __init__(self, spec: dict[str, NodeSpec], device: str = "cpu",
                 seed: int | None = None):
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
        self.nodes = nn.ModuleDict({name: _Node(name, spec[name], spec) for name in self.order})
        self.device = torch.device(device)
        self.history: dict = {"train": [], "val": [], "lr": [], "time": []}
        self.meta: dict = {}   # provenance attached at save() (machine, versions)
        self.to(self.device)

    # ------------------------------------------------------------------ data
    def _encode_parent(self, name: str, values: Tensor) -> Tensor:
        """Encode a node's values for use as a parent feature (original TRAM-DAG convention:
        continuous raw (n, 1); ordinal one-hot (n, levels))."""
        node = self.spec[name]
        if isinstance(node, OrdinalNode):
            return torch.nn.functional.one_hot(
                values.long(), num_classes=node.levels).to(values.dtype)
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
                df[name].to_numpy(dtype=np_dtype), device=self.device)
            out[name] = vals
        return out

    def _features(self, values: dict[str, Tensor]) -> dict[str, Tensor]:
        return {name: self._encode_parent(name, vals) for name, vals in values.items()}

    # ------------------------------------------------------------- likelihood
    def node_log_prob(self, values: dict[str, Tensor],
                      nodes: list[str] | None = None) -> dict[str, Tensor]:
        """Per-node log-likelihood contributions, each (n,).

        ``nodes`` restricts computation to a subset (used to skip frozen nodes
        during training — valid because the per-node losses are independent)."""
        feats = self._features(values)
        n = next(iter(values.values())).shape[0]
        out = {}
        for name in (self.order if nodes is None else nodes):
            node = self.nodes[name]
            theta, shift = node.theta_shift(feats, n)
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
        """Train 5%/95% quantiles -> transform domain (the original implementation's min_max scaling).

        ``marginal_init``: opt-in calibrated Bernstein init (see ``fit``). Applied only
        on the first fit (the same ``not ut._fitted`` guard as range-setting), so a
        multi-phase fit does not reset a partially-trained intercept."""
        from .transforms import BernsteinUT, ordinal_marginal_init_theta
        for name in self.order:
            node = self.nodes[name]
            if node.kind == "continuous" and not node.ut._fitted:
                q = train_df[name].quantile([0.05, 0.95])
                node.ut.set_range(q.iloc[0], q.iloc[1])
                if (marginal_init and isinstance(node.ut, BernsteinUT)
                        and isinstance(node.intercept, SimpleIntercept)):
                    with torch.no_grad():
                        node.intercept.theta.copy_(node.ut.marginal_init_theta())
            elif (node.kind == "ordinal" and marginal_init
                    and isinstance(node.intercept, SimpleIntercept)
                    and not getattr(node.intercept, "_marginal_inited", False)):
                # calibrate unconditional cutpoints to the marginal class log-odds
                counts = np.bincount(train_df[name].to_numpy().astype(np.int64),
                                     minlength=self.spec[name].levels)
                with torch.no_grad():
                    node.intercept.theta.copy_(ordinal_marginal_init_theta(counts))
                node.intercept._marginal_inited = True

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None,
            epochs: int = 500, learning_rate: float = 1e-2, batch_size: int = 512,
            verbose: int = 50, seed: int | None = None,
            restore_best: bool = False, schedule: str | None = None,
            plateau_patience: int = 15, freeze_patience: int | None = None,
            min_delta: float = 1e-4, marginal_init: bool = False,
            vc_warm_start: bool = True) -> "CausalFlowDAG":
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
                during training and restore them at the end (mild early-stopping
                regularization, the original implementation's convention). This makes the fit
                *not* the training-data MLE, so leave it False for an exact
                classical comparison. Default False.
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
        the constant moves into ``beta0``).

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
        vc_penalized = {name: [self.nodes[name].shifts[on]
                               for on, mods, _ in self.nodes[name]._vc_groups
                               if mods and self.nodes[name].shifts[on].penalty > 0]
                        for name in self.order}

        train_vals = self._tensorize(train_df)
        val_vals = self._tensorize(val_df) if val_df is not None else train_vals
        n = len(train_df)
        steps_per_epoch = (n + batch_size - 1) // batch_size

        opt = torch.optim.Adam(
            [{"params": list(self.nodes[name].parameters()), "lr": learning_rate,
              "node": name} for name in self.order])
        sched = None
        if schedule == "onecycle":
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=learning_rate, total_steps=epochs * steps_per_epoch)
        elif schedule == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=epochs, eta_min=learning_rate * 1e-3)

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
            train_acc = {name: prev_train.get(name, float("nan"))
                         for name in frozen}
            train_acc.update({name: 0.0 for name in active})
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                batch = {k: v[idx] for k, v in train_vals.items()}
                per_node = self.node_log_prob(batch, nodes=active)
                node_nlls = {k: -v.mean() for k, v in per_node.items()}
                loss = torch.stack(list(node_nlls.values())).sum()
                for name in active:            # VC penalty (excluded from history)
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
                val_per_node = {k: float(-v.mean())
                                for k, v in self.node_log_prob(val_vals).items()}
            self.history["train"].append(train_acc)
            self.history["val"].append(val_per_node)
            self.history.setdefault("lr", []).append(
                max(g["lr"] for g in opt.param_groups))
            self.history.setdefault("time", []).append(
                t_offset + time.perf_counter() - t0)

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
                if (schedule == "plateau" and node_bad[name] > 0
                        and node_bad[name] % plateau_patience == 0):
                    g["lr"] = max(g["lr"] * 0.3, learning_rate * 1e-3)
                # under "plateau", only freeze nodes whose lr has already been
                # decayed substantially — otherwise a node can freeze while a
                # smaller lr would still make progress toward the optimum
                lr_decayed = (schedule != "plateau"
                              or g["lr"] <= learning_rate * 1e-2 * (1 + 1e-9))
                if (freeze_patience is not None and lr_decayed
                        and node_bad[name] >= freeze_patience):
                    frozen.add(name)
                    self.history.setdefault("frozen", {}).setdefault(
                        name, len(self.history["val"]))  # 1-based global epoch

            if restore_best:
                for name in self.order:
                    if val_per_node[name] < best[name][0]:
                        best[name] = (val_per_node[name],
                                      copy.deepcopy(self.nodes[name].state_dict()))

            if verbose and (epoch % verbose == 0 or epoch == epochs - 1):
                tot_t = sum(train_acc.values())
                tot_v = sum(val_per_node.values())
                print(f"[epoch {epoch + 1:5d}/{epochs}] train NLL {tot_t:.4f}  "
                      f"val NLL {tot_v:.4f}"
                      + (f"  frozen {sorted(frozen)}" if frozen else ""))

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
        """Initialise each VC term's ``beta0`` from the classical all-``ls``
        solution of its node's conditional (issue #28's recommended warm start).

        A throwaway proxy of the node (same kind/transform, every parent an LS
        term, parent marginals irrelevant to the conditional because the joint
        NLL decomposes per node) is fitted with the deterministic
        :meth:`fit_classical`, and the ``on`` coefficient copied into ``beta0``
        (for a binary ordinal treatment, the identified one-hot difference
        ``w[1] - w[0]``). ``b_theta`` already starts at the zero function
        (zero-initialised output layer). Runs once per term — the
        ``warm_started`` buffer survives ``save``/``load``."""
        for name in self.order:
            nd = self.nodes[name]
            todo = [g for g in nd._vc_groups
                    if not bool(nd.shifts[g[0]].warm_started)]
            if not todo:
                continue
            node_spec = self.spec[name]
            proxy_spec: dict[str, NodeSpec] = {}
            for p in nd.parents:
                pn = self.spec[p]
                proxy_spec[p] = (OrdinalNode(levels=pn.levels)
                                 if isinstance(pn, OrdinalNode)
                                 else ContinuousNode(transform="affine"))
            ls_terms = [LS(p) for p in nd.parents]
            if isinstance(node_spec, OrdinalNode):
                proxy_spec[name] = OrdinalNode(levels=node_spec.levels, terms=ls_terms)
            else:
                proxy_spec[name] = ContinuousNode(
                    transform=node_spec.transform,
                    transform_kwargs=dict(node_spec.transform_kwargs),
                    terms=ls_terms)
            proxy = CausalFlowDAG(proxy_spec, device=str(self.device))
            proxy.fit_classical(train_df[list(nd.parents) + [name]], verbose=False)
            for on, mods, on_is_ord in todo:
                w = proxy.nodes[name].shifts[on].weight.detach()
                b0 = float(w[-1] - w[0]) if on_is_ord else float(w[0])
                m = nd.shifts[on]
                with torch.no_grad():
                    m.beta0.fill_(b0)
                m.warm_started.fill_(True)

    @torch.no_grad()
    def _recenter_vc(self, values: dict[str, Tensor]) -> None:
        """Re-split every VC term so ``b_theta`` is sum-to-zero over the training
        rows (function-preserving; the constant moves into ``beta0``)."""
        feats: dict[str, Tensor] | None = None
        for name in self.order:
            nd = self.nodes[name]
            for on, mods, _ in nd._vc_groups:
                if not mods:
                    continue
                if feats is None:
                    feats = self._features(values)
                nd.shifts[on].recenter(torch.cat([feats[p] for p in mods], dim=1))

    @torch.no_grad()
    def varying_coef(self, node: str, data: pd.DataFrame,
                     on: str | None = None) -> np.ndarray:
        """Fitted effect function ``beta(x)`` of a ``VC`` term, evaluated at the
        rows of ``data`` — the first-class read-out of issue #28.

        Closed-form from the fitted term (``beta0 + b_theta(modifiers)``):
        deterministic, y-free (only the modifier columns of ``data`` are read),
        no abduction. For a binary treatment it equals the abduct-difference
        ``u(x, t=1, y) - u(x, t=0, y)`` identically. The value lives on the
        node's latent (log-odds) scale — additive for a continuous node,
        subtracted from the cutpoints for an ordinal node.

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
        vcs = {o: mods for o, mods, _ in nd._vc_groups}
        if not vcs:
            raise ValueError(f"node {node!r} has no VC term.")
        if on is None:
            if len(vcs) > 1:
                raise ValueError(
                    f"node {node!r} has several VC terms ({sorted(vcs)}); "
                    "pass on=<treatment name>.")
            on = next(iter(vcs))
        if on not in vcs:
            raise KeyError(f"node {node!r} has no VC term on {on!r} "
                           f"(has {sorted(vcs)}).")
        mods = vcs[on]
        missing = [p for p in mods if p not in data.columns]
        if missing:
            raise KeyError(f"data is missing modifier column(s): {missing}")
        mod_feat = None
        if mods:
            np_dtype = np.float64 if self._dtype == torch.float64 else np.float32
            vals = {p: torch.as_tensor(data[p].to_numpy(dtype=np_dtype),
                                       device=self.device) for p in mods}
            feats = self._features(vals)
            mod_feat = torch.cat([feats[p] for p in mods], dim=1)
        return nd.shifts[on].beta(mod_feat, len(data)).cpu().numpy()

    # --------------------------------------------------------- classical fit
    def _is_all_ls(self) -> bool:
        return all(term.effect == "LS" for node in self.spec.values()
                   for term in node_terms(node))

    def ls_coefficients(self) -> dict[str, dict[str, np.ndarray]]:
        """Per-node linear-shift weights {node: {parent: weight array}} — the
        interpretable log-odds-ratio coefficients of an all-``ls`` model."""
        out: dict[str, dict[str, np.ndarray]] = {}
        for name in self.order:
            shifts = self.nodes[name].shifts
            if shifts:
                out[name] = {p: m.weight.detach().cpu().numpy().ravel().copy()
                             for p, m in shifts.items()}
        return out

    def to_matrix(self) -> pd.DataFrame:
        """Labelled adjacency matrix (rows = parent, cols = child) of term
        effects — the paper's meta-adjacency view. Cells hold "LS"/"CS"/"CI"
        (empty = no edge); a multi-parent term is suffixed with its parent group."""
        labels = {"I": "CI", "LS": "LS", "CS": "CS"}
        m = pd.DataFrame("", index=list(self.order), columns=list(self.order))
        for child in self.order:
            for term in node_terms(self.spec[child]):
                if term.effect == "VC":   # treatment cell "VC", modifiers "VCm"
                    cells = [(term.parents[0], "VC")] + [
                        (p, "VCm") for p in term.parents[1:]]
                else:
                    tag = labels[term.effect]
                    if len(term.parents) > 1:
                        tag = f"{tag}{list(term.parents)}"
                    cells = [(p, tag) for p in term.parents]
                for p, tag in cells:      # a VC modifier may share its cell with
                    cur = m.loc[p, child]  # a prognostic term -> join with "+"
                    m.loc[p, child] = f"{cur}+{tag}" if cur else tag
        return m

    @torch.no_grad()
    def intercept_contributions(self, node: str, data: pd.DataFrame) -> dict:
        """Mean-centered per-term contributions to a node's **complex intercept**
        (the transform parameters), for plotting additive partial effects.

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
                "there is nothing to decompose (its intercept is unconditional).")
        missing = [p for p in nd.ci_parents if p not in data.columns]
        if missing:
            raise KeyError(f"data is missing intercept-parent column(s): {missing}")

        np_dtype = np.float64 if self._dtype == torch.float64 else np.float32
        vals = {p: torch.as_tensor(data[p].to_numpy(dtype=np_dtype),
                                   device=self.device) for p in nd.ci_parents}
        feats = self._features(vals)
        # one net per group: the additive case stores them in intercept_nets;
        # a single (possibly joint) I-term is the lone `intercept` network.
        nets = list(nd.intercept_nets) if nd.intercept_nets is not None else [nd.intercept]

        contributions: dict[str, np.ndarray] = {}
        parents: dict[str, tuple] = {}
        baseline = None
        for net, grp in zip(nets, groups):
            raw = net(torch.cat([feats[p] for p in grp], dim=1))   # (n, P)
            mean = raw.mean(dim=0, keepdim=True)                    # (1, P)
            label = "+".join(grp)
            contributions[label] = (raw - mean).cpu().numpy()
            parents[label] = grp
            baseline = mean if baseline is None else baseline + mean
        return {"baseline": baseline.cpu().numpy().ravel(),
                "contributions": contributions, "parents": parents}

    def fit_classical(self, train_df: pd.DataFrame, *, max_iter: int = 400,
                      tol: float = 1e-6, verbose: bool = True) -> dict:
        """Fit an **all-``ls``** model classically: full-batch, **float64**,
        L-BFGS (strong-Wolfe line search). No minibatches, no schedule, no
        early stopping — so the fit is **deterministic** (bit-reproducible) and
        lands on the exact maximum-likelihood estimate, matching classical
        software (``statsmodels`` ``OrderedModel`` / R ``polr``/``Colr``) far
        faster than minibatch Adam.

        Only valid when every edge is ``ls`` (each node-conditional is then a
        classical transformation model); raises otherwise — for ``cs``/``ci``
        models use :meth:`fit`, where minibatch noise also regularizes the MLPs.

        float64 is a *transient compute mode*: the model is upcast for the fit
        (``self.double()`` converts parameters **and** the transforms' range
        buffers in one call) and restored to float32 afterwards, so the stored
        model and ``save``/``load`` stay float32. Double precision is what lets
        the line search resolve the optimum cleanly.

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
                "this spec has cs/ci/vc terms. Use fit() for flexible models.")
        self._set_ranges(train_df)

        self.double()  # parameters + buffers (xmin/xmax) -> float64, one call
        assert next(self.parameters()).dtype == torch.float64
        t0 = time.perf_counter()
        chunk = 25  # inner L-BFGS iterations per round; we stop on NLL change
        try:
            vals = self._tensorize(train_df)
            self.train()
            opt = torch.optim.LBFGS(
                self.parameters(), lr=1.0, max_iter=chunk, history_size=50,
                tolerance_grad=0.0, tolerance_change=0.0,
                line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                nll = torch.stack([-lp.mean() for lp
                                   in self.node_log_prob(vals).values()]).sum()
                nll.backward()
                return nll

            def flat_coefs() -> np.ndarray:
                cs = self.ls_coefficients()
                return np.concatenate([w for node in cs.values()
                                       for w in node.values()]) if cs \
                    else np.zeros(1)

            prev_nll, prev_c, final_nll, n_iter, converged, coef_delta = (
                float("inf"), flat_coefs(), float("nan"), 0, False, float("inf"))
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
            grad_norm = float(torch.cat([p.grad.reshape(-1)
                                         for p in self.parameters()
                                         if p.grad is not None]).norm())
            coefs = self.ls_coefficients()  # read while still float64
        finally:
            self.float()  # restore canonical float32 (lossy ~1e-7, harmless)
        self.eval()

        report = {"converged": converged, "n_iter": n_iter,
                  "final_nll": final_nll, "grad_norm": grad_norm,
                  "coef_delta": coef_delta,
                  "seconds": time.perf_counter() - t0, "coefficients": coefs}
        if verbose:
            print(f"fit_classical: {n_iter} L-BFGS iters, NLL {final_nll:.6f}, "
                  f"{report['seconds']:.2f}s"
                  + ("" if converged else f"  (NLL still moving at {max_iter} iters)"))
        return report

    # ------------------------------------------------------- causal queries
    @torch.no_grad()
    def sample(self, n: int | None = None, *, do: dict[str, float] | None = None,
               u: pd.DataFrame | None = None, seed: int | None = None) -> pd.DataFrame:
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
            u_vals = {name: torch.as_tensor(u[name].to_numpy(dtype=np_dtype, copy=True),
                                            device=self.device) for name in self.order}
        elif n is not None:
            u_vals = {name: StandardLogistic.sample((n,), device=self.device)
                      if gen is None else
                      StandardLogistic.icdf(torch.rand((n,), device=self.device, generator=gen))
                      for name in self.order}
        else:
            raise ValueError("Provide either n or u.")

        values: dict[str, Tensor] = {}
        for name in self.order:
            if name in do:
                values[name] = torch.full((n,), float(do[name]),
                                          dtype=self._dtype, device=self.device)
                continue
            node = self.nodes[name]
            feats = self._features({p: values[p] for p in node.parents})
            theta, shift = node.theta_shift(feats, n)
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
            theta, shift = node.theta_shift(feats, n)
            x = values[name]
            if node.kind == "continuous":
                z0, _ = node.ut.forward(theta, x)
                u[name] = z0 + shift
            else:
                u[name] = ordinal_abduct(theta, shift, x, generator=gen)
        return pd.DataFrame({k: v.cpu().numpy() for k, v in u.items()})

    @torch.no_grad()
    def pmf(self, df: pd.DataFrame, node: str, do: dict[str, float] | None = None) -> np.ndarray:
        """Analytic class probabilities (n, levels) for an ordinal node, with the
        node's parents taken from ``df`` after applying ``do`` overrides."""
        if not isinstance(self.spec[node], OrdinalNode):
            raise ValueError(f"pmf() requires an ordinal node, '{node}' is continuous.")
        df_local = df.copy()
        for col, val in (do or {}).items():
            df_local[col] = val
        nd = self.nodes[node]
        np_dtype = np.float64 if self._dtype == torch.float64 else np.float32
        values = {p: torch.as_tensor(df_local[p].to_numpy(dtype=np_dtype),
                                     device=self.device) for p in nd.parents}
        feats = self._features(values)
        theta, shift = nd.theta_shift(feats, len(df_local))
        return ordinal_pmf(theta, shift).cpu().numpy()

    # ------------------------------------------------------------------ scores
    @torch.no_grad()
    def scores(self, df: pd.DataFrame, node: str,
               params: str = "shift") -> pd.DataFrame:
        """Per-observation scores ``psi_i = d l_i / d theta`` (issue #29) for a
        node's interpretable shift coefficients — analytic and exact (see
        ``tramdag.scores``). ``params="shift"`` (the only option) covers every
        ``LS`` weight and every ``VC`` term's ``beta0``. At a fitted MLE the
        per-column sums are ~0; ordered by a covariate that truly modifies the
        treatment effect, the treatment column's cumulative sum drifts — see
        :meth:`effect_modifier_scan`. Pure read-out; touches no fitting or
        sampling code path."""
        if params != "shift":
            raise ValueError(f"params='shift' is the only option, got {params!r}.")
        from .scores import node_scores
        return node_scores(self, df, node)

    @torch.no_grad()
    def effect_modifier_scan(self, df: pd.DataFrame, node: str, on: str,
                             candidates: list[str] | None = None) -> pd.DataFrame:
        """Zeileis–Hornik fluctuation scan (issue #29): rank candidate
        covariates by how strongly the ``on``-coefficient scores drift when
        ordered by them — the measured shortlist for ``VC`` modifiers, from a
        cheap all-``ls`` fit. Returns stat / p_value / crit_5pct / flag per
        candidate (see ``tramdag.scores.effect_modifier_scan``)."""
        from .scores import effect_modifier_scan
        return effect_modifier_scan(self, df, node, on, candidates=candidates)

    # ------------------------------------------------------------------- io
    def save(self, path: str | Path) -> None:
        """Checkpoint the model (spec + weights), its training ``history``, and a
        provenance ``meta`` block (tramdag version, save time, device, and the
        machine/environment it was trained on) so cached runs stay
        self-describing — training-curve plots and timing comparisons can be
        reconstructed from the file alone."""
        from datetime import datetime, timezone

        from . import __version__
        from .env import machine_info
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"tramdag_version": __version__,
                "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "device": str(self.device),
                "machine": machine_info()}
        torch.save({"spec": spec_to_dict(self.spec),
                    "state_dict": self.state_dict(),
                    "history": self.history,
                    "meta": meta}, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "CausalFlowDAG":
        """Restore a model. ``flow.history`` and ``flow.meta`` are repopulated, so
        a cached model can still produce training/diagnostic plots and report the
        machine it was trained on."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        flow = cls(spec_from_dict(ckpt["spec"]), device=device)
        for name in flow.order:  # mark transforms as fitted before loading buffers
            node = flow.nodes[name]
            if node.kind == "continuous":
                node.ut._fitted = True
        flow.load_state_dict(ckpt["state_dict"])
        flow.history = ckpt.get("history", {"train": [], "val": [], "lr": [], "time": []})
        flow.meta = ckpt.get("meta", {})
        flow.eval()
        return flow
