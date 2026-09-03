"""The internal node model: one sub-model per variable.

`_Node` bundles a variable's intercept (transform parameters), monotone
transform and shift modules; `CausalFlowDAG` holds one per node and the DAG
lives in which parents each node reads.

Internal-but-stable surface
---------------------------
scores.py, callbacks recipes, the read-outs and the test suite read these
names by design; renaming any of them is an API change, not a cleanup:
``kind``, ``parents``, ``shifts`` (term modules with ``key``/``parents``/
``mods``…), ``intercept`` (+ ``.groups``/``.ci_parents``/``.nets``), ``ut``,
``net_input``, ``theta_shift``.
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from .spec import (
    ContinuousNode,
    NodeSpec,
    node_parents,
)
from .terms import ShiftTerm, VCTerm, get_term
from .transforms import (
    StandardLogistic,
    make_univariate_transform,
    ordinal_abduct,
    ordinal_log_prob,
    ordinal_marginal_init_theta,
    ordinal_sample,
)


# %% private functions -----------------------------------------------------------------
def _init_linear(m: nn.Linear, init: str) -> None:
    """Keras' two initializers on one linear layer: ``glorot`` or ``normal``."""
    if init == "glorot":
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    else:
        nn.init.normal_(m.weight, std=0.05)
        if m.bias is not None:
            nn.init.normal_(m.bias, std=0.05)


# %% private classes -------------------------------------------------------------------
class _Node(nn.Module):
    """One dimension of the flow: an intercept plus additive shift terms.

    The intercept produces the transform parameters ``theta``. The shift
    terms add up on the latent scale.

    Parameters
    ----------
    node : NodeSpec
        Specification of the node.
    spec : dict[str, NodeSpec]
        The full DAG specification. Needed for the parent feature widths.
    """

    def __init__(
        self,
        node: NodeSpec,
        spec: dict[str, NodeSpec],
    ):
        super().__init__()
        self.kind = node.kind
        terms = node.terms
        self.parents = tuple(node_parents(node))  # ordered parent names
        if isinstance(node, ContinuousNode):
            self.ut = make_univariate_transform(node.transform, **node.transform_kwargs)
            n_params = self.ut.n_params
        else:
            self.ut = None
            self.levels = node.levels
            n_params = node.levels - 1
        self._build_intercept(terms[0], n_params, spec)
        self._build_shifts(terms, spec)

    def _build_intercept(self, i_term, n_params: int, spec: dict[str, NodeSpec]):
        """Build the node's one intercept term module, via the registry.

        The term class decides its own shape: the free SimpleIntercept
        theta_0, one joint ComplexIntercept, or one net per parent summed in
        unconstrained coefficient space (``allow_interaction=False``).
        """
        self.intercept = get_term(i_term.effect).build(i_term, spec, n_params)

    def _build_shifts(self, terms, spec: dict[str, NodeSpec]):
        """Build one shift term module per LS/CS/VC term, via the registry.

        Each term class constructs itself exactly as this method used to
        (same widths, same order — the seeded RNG stream is pinned) and
        names its own ModuleDict key: the parent for single-parent terms,
        "a+b" for a joint CS, the treatment for a VC.
        """
        self.shifts = nn.ModuleDict()
        for term in terms:
            entry = get_term(term.effect)
            if entry.slot != "shift":
                continue
            assert issubclass(entry, ShiftTerm), term.effect
            m = entry.build(term, spec)
            self.shifts[m.key] = m

    def net_input(self, feats: dict[str, Tensor], parents, key: str) -> Tensor:
        """Concatenate parent features for one term's network.

        ``key`` names the term ("@I" for the intercept, the shift key
        otherwise); a term with an ``input_transform`` gets its continuous
        parent columns transformed with the statistics frozen at
        ``calibrate``. Every network input goes through here — training and
        the read-outs (``varying_coef``, ``intercept_contributions``) alike —
        so the model seen at inference is the model that was fitted. Linear
        shifts and the VC treatment column are not network inputs and never
        pass through.
        """
        term = self.intercept if key == "@I" else self.shifts[key]
        tr = term.input_transform
        cols = []
        for p in parents:
            x = feats[p]
            if tr is not None and p in tr.cols:
                x = tr(x, tr.cols.index(p))
            cols.append(x)
        return torch.cat(cols, dim=1)

    def theta_shift(self, feats: dict[str, Tensor], n: int) -> tuple[Tensor, Tensor]:
        """Compute the transform parameters and the total shift of the node.

        Parameters
        ----------
        feats : dict[str, Tensor]
            Encoded parent features keyed by parent name, plus the terms'
            side columns (a centered VC's propensities — frozen from the
            training frame, injected live by the flow at query time).
        n : int
            Batch size.

        Returns
        -------
        tuple[Tensor, Tensor]
            The transform parameters, shape ``(n, P)``, and the total
            shift, shape ``(n,)``.

        Raises
        ------
        RuntimeError
            If a centered VC term is evaluated without its propensity
            column in ``feats``.
        """
        theta = self.intercept.theta_value(self, feats, n)
        shift = torch.zeros(n, dtype=theta.dtype, device=theta.device)
        # plain shifts first, then VC (stable sort keeps the pinned order)
        for m in sorted(self.shifts.values(), key=lambda m: isinstance(m, VCTerm)):
            shift = shift + m.shift_value(self, feats)
        return theta, shift


# %% per-kind operations ---------------------------------------------------------------
# The ONLY continuous-vs-ordinal branches of the package live in these four
# adjacent functions. A third node kind earns a protocol; two stay an if/else
# in one place.
def kind_log_prob(node: _Node, theta: Tensor, shift: Tensor, x: Tensor) -> Tensor:
    """``log p(x | pa)`` from one node's transform parameters and shift."""
    if node.kind == "continuous":
        u0, ladj = node.ut.forward(theta, x)
        return StandardLogistic.log_prob(u0 + shift) + ladj
    return ordinal_log_prob(theta, shift, x)


def kind_sample(node: _Node, theta: Tensor, shift: Tensor, u: Tensor) -> Tensor:
    """Push one node's latent ``u`` forward to an observed value."""
    if node.kind == "continuous":
        return node.ut.inverse(theta, u - shift)
    return ordinal_sample(theta, shift, u)


def kind_abduct(
    node: _Node, theta: Tensor, shift: Tensor, x: Tensor, generator=None
) -> Tensor:
    """Recover one node's latent: exact (continuous) or truncated-sampled (ordinal)."""
    if node.kind == "continuous":
        u0, _ = node.ut.forward(theta, x)
        return u0 + shift
    return ordinal_abduct(theta, shift, x, generator=generator)


def kind_marginal_theta(node: _Node, column: np.ndarray):
    """Give the marginal-start theta of a simple intercept, or ``None``.

    Ordinal: the empirical class log-odds. Continuous: the transform's own
    calibrated start (``None`` for spline/affine — nothing to set).
    """
    if node.kind == "ordinal":
        counts = np.bincount(column.astype(np.int64), minlength=node.levels)
        return ordinal_marginal_init_theta(counts)
    return node.ut.marginal_init_theta()
