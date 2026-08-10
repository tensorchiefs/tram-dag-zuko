"""Conditioner networks for the per-edge term types.

Architectures replicate the original Keras/TF implementation's defaults
(``tram_models.py`` in https://github.com/tensorchiefs/tram-dag)
so that fitted models are directly comparable:

- ``LinearShift``        — Linear(n, 1, bias=False)            (term "ls")
- ``ComplexShift``       — 64-128-64 ReLU MLP -> 1, no bias    (term "cs",
                            original ``ComplexShiftDefaultTabular``)
- ``ComplexIntercept``   — 8-8 ReLU MLP -> n_params, no bias   (term "ci",
                            original ``ComplexInterceptDefaultTabular``)
- ``SimpleIntercept``    — free parameter vector (no parent dependence)
- ``VaryingCoef``        — beta0 + small penalized 16-unit MLP (term "VC",
                            issue #28; no original-implementation counterpart)

Parent features follow the original implementation's encoding: continuous parents enter raw
(one column), ordinal parents are one-hot encoded (``levels`` columns).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SimpleIntercept(nn.Module):
    """Free (data-independent) transform parameters, broadcast over the batch."""

    def __init__(self, n_params: int):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(n_params))

    def forward(self, n: int) -> Tensor:
        return self.theta.unsqueeze(0).expand(n, -1)


class ComplexIntercept(nn.Module):
    """Transform parameters as a function of the (joint) ci-parent features."""

    def __init__(self, n_features: int, n_params: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, n_params, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class LinearShift(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.fc = nn.Linear(n_features, 1, bias=False)

    @property
    def weight(self) -> Tensor:
        return self.fc.weight.squeeze(0)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x).squeeze(-1)


class ComplexShift(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


class VaryingCoef(nn.Module):
    """Varying-coefficient effect head: ``beta(x) = beta0 + b_theta(x)`` (issue #28).

    ``b_theta`` is deliberately small (one 16-unit hidden layer) and its weights
    carry an L2 ``penalty`` in the fitting objective (:meth:`l2`; ``fit`` adds
    ``penalty * l2()`` on the total-NLL scale); ``beta0`` is unpenalized. The
    output layer is **zero-initialised**, so ``beta(x) = beta0``
    exactly at construction — the head only ever learns *deviations* from the
    constant effect, which is what makes the arm-difference estimated rather than
    a by-product (the unregularized ``CS(on, x...)`` reduced form measures
    corr ≈ 0.5 against the true effect function; see issue #28).

    Identification: a constant can move freely between ``beta0`` and ``b_theta``.
    The penalty resolves it during training (shrinking ``b_theta`` toward the
    zero function); :meth:`recenter` re-splits exactly afterwards so that
    ``b_theta`` is sum-to-zero over the training data (the GAM convention, as in
    ``intercept_contributions``) — a pure reparameterization via the ``center``
    buffer, the modelled function is unchanged. With ``n_features == 0`` (no
    modifiers) there is no net and the term is exactly ``LS(on)``.
    """

    def __init__(self, n_features: int, penalty: float = 1.0, hidden: int = 16):
        super().__init__()
        self.penalty = float(penalty)
        self.beta0 = nn.Parameter(torch.zeros(()))
        self.register_buffer("center", torch.zeros(()))
        self.register_buffer("warm_started", torch.tensor(False))
        if n_features > 0:
            out = nn.Linear(hidden, 1, bias=False)
            nn.init.zeros_(out.weight)  # beta(x) == beta0 at init
            self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), out)
        else:
            self.net = None

    def beta(self, mod_feats: Tensor | None, n: int) -> Tensor:
        """Effect values beta(x), shape (n,). ``mod_feats`` is ``None`` iff the
        term has no modifiers.
        """
        if self.net is None:
            return (self.beta0 - self.center).expand(n)
        return self.beta0 + self.net(mod_feats).squeeze(-1) - self.center

    def forward(self, t: Tensor, mod_feats: Tensor | None) -> Tensor:
        """Shift contribution beta(mod_feats) * t, shape (n,). ``t`` is the raw
        treatment column (n, 1).
        """
        return self.beta(mod_feats, t.shape[0]) * t.squeeze(-1)

    def l2(self) -> Tensor:
        """Sum of squared ``b_theta`` weights (the penalized quantity; 0 without
        modifiers, ``beta0`` never included).
        """
        if self.net is None:
            return torch.zeros((), device=self.beta0.device, dtype=self.beta0.dtype)
        return sum(p.pow(2).sum() for p in self.net.parameters())

    @torch.no_grad()
    def recenter(self, mod_feats: Tensor | None) -> None:
        """Re-split beta0 + b_theta so b_theta is mean-zero over ``mod_feats``
        (function-preserving: the removed constant moves into ``beta0``).
        """
        if self.net is None:
            return
        delta = (self.net(mod_feats).squeeze(-1) - self.center).mean()
        self.center += delta
        self.beta0 += delta
