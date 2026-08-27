"""Conditioner networks, one per edge term type.

Each architecture copies a default of the PyTorch reference this package grew
out of: ``tramdag/models/tram_models.py`` in
https://github.com/buehlpa/TramDag, whose ``ComplexShiftDefaultTabular`` is
64-128-64 ReLU into a bias-free ``Linear(64, 1)`` and whose
``ComplexInterceptDefaultTabular`` is 8-8 ReLU into a bias-free
``Linear(8, n_thetas)`` with ``n_thetas=20``. A fitted model is therefore
directly comparable with that implementation.

Those defaults are **not** the TRAM-DAG paper's own nets. The paper's R
implementation (https://github.com/tensorchiefs/tram-dag) uses
``hidden_features_I = hidden_features_CS = c(2, 25, 25, 2)`` with sigmoid
activations for the triangle experiments, and a 10-100 tanh net for the
CAREFL/VACA comparisons. Every replication in ``experiments/paper/`` therefore
sets ``units=`` and ``activation=`` explicitly from its own reference script,
and none of them relies on the defaults here.

===================== ============================================ ======
Conditioner           Architecture                                 Term
===================== ============================================ ======
``LinearShift``       ``Linear(n, 1, bias=False)``                 ``LS``
``ComplexShift``      64-128-64 ReLU MLP to 1, no bias             ``CS``
``ComplexIntercept``  8-8 ReLU MLP to ``n_params``, bias-free out  ``I``
``SimpleIntercept``   free parameter vector, no parent             none
``VaryingCoef``       ``beta0`` + penalized 16-unit MLP            ``VC``
===================== ============================================ ======

``ComplexShift`` and ``ComplexIntercept`` correspond to
``ComplexShiftDefaultTabular`` and ``ComplexInterceptDefaultTabular``.
``VaryingCoef`` (issue #28) has no counterpart in the original code.

Parent features use the encoding of the original implementation. A continuous
parent enters raw, in one column. An ordinal parent is one-hot encoded, in
``levels`` columns.
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

import torch
from torch import Tensor, nn

# %% global variables ------------------------------------------------------------------
# the activations the reference implementations use: relu in the PyTorch
# reference's default classes, sigmoid in the paper's create_param_net, tanh in
# the paper's make_model for the CAREFL/VACA comparisons.
ACTIVATIONS = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
# relu, because the architectures these conditioners copy use relu -- the
# default net and the default activation come from the same source.
DEFAULT_ACTIVATION = "relu"


# %% private functions -----------------------------------------------------------------
def _mlp(
    n_in: int,
    units: tuple[int, ...],
    n_out: int,
    *,
    activation: str | None = None,
    zero_init_last: bool = False,
) -> nn.Sequential:
    """Build the one MLP shape every conditioner uses.

    Hidden layers of the given ``units``, each followed by ``activation``,
    then a bias-free output layer.

    Parameters
    ----------
    n_in : int
        Input width.
    units : tuple[int, ...]
        Hidden layer widths.
    n_out : int
        Output width.
    activation : str | None, optional
        Key of :data:`ACTIVATIONS`: ``"relu"`` (the default, and what the
        PyTorch reference's default classes use), ``"sigmoid"`` (the paper's
        ``create_param_net``) or ``"tanh"`` (the paper's ``make_model``, used
        for its CAREFL/VACA comparisons). ``None`` takes
        :data:`DEFAULT_ACTIVATION`.
    zero_init_last : bool, optional
        Zero the output layer, by default ``False``.

    Returns
    -------
    nn.Sequential
        The network.

    Raises
    ------
    KeyError
        From :data:`ACTIVATIONS` if the name is not one of its keys.
    """
    name = activation or DEFAULT_ACTIVATION
    if name not in ACTIVATIONS:
        raise ValueError(
            f"unknown activation {name!r}; choose one of {sorted(ACTIVATIONS)}"
        )
    make_activation = ACTIVATIONS[name]
    layers: list[nn.Module] = []
    width = n_in
    for u in units:
        layers += [nn.Linear(width, u), make_activation()]
        width = u
    out = nn.Linear(width, n_out, bias=False)
    if zero_init_last:
        nn.init.zeros_(out.weight)
    return nn.Sequential(*layers, out)


# %% public classes --------------------------------------------------------------------
class SimpleIntercept(nn.Module):
    """Free transform parameters that do not depend on the data.

    Parameters
    ----------
    n_params : int
        Number of transform parameters.
    """

    def __init__(self, n_params: int):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(n_params))

    def forward(self, n: int) -> Tensor:
        """Broadcast the parameters over a batch.

        Parameters
        ----------
        n : int
            Batch size.

        Returns
        -------
        Tensor
            The parameters, shape ``(n, n_params)``.
        """
        return self.theta.unsqueeze(0).expand(n, -1)


class ComplexIntercept(nn.Module):
    """Transform parameters as a function of the intercept-parent features.

    Several parents given to one term feed a single network, so they interact.

    Parameters
    ----------
    n_features : int
        Width of the encoded parent features.
    n_params : int
        Number of transform parameters to produce.
    units : tuple[int, ...] | None, optional
        Hidden layers of the network, by default ``(8, 8)`` — the two hidden
        layers of ``ComplexInterceptDefaultTabular`` (see the module
        docstring). The paper's own nets are wider; a replication should set
        this explicitly.
    activation : str | None, optional
        Key of :data:`ACTIVATIONS`, by default :data:`DEFAULT_ACTIVATION`.
    """

    def __init__(
        self,
        n_features: int,
        n_params: int,
        units: tuple[int, ...] | None = None,
        activation: str | None = None,
    ):
        super().__init__()
        self.net = _mlp(n_features, units or (8, 8), n_params, activation=activation)

    def forward(self, x: Tensor) -> Tensor:
        """Map parent features to transform parameters.

        Parameters
        ----------
        x : Tensor
            Encoded parent features, shape ``(n, n_features)``.

        Returns
        -------
        Tensor
            Transform parameters, shape ``(n, n_params)``.
        """
        return self.net(x)


class LinearShift(nn.Module):
    """Linear shift ``beta * x``, one weight per feature and no bias.

    For an ordinal child, ``exp(beta)`` is an odds ratio.

    Parameters
    ----------
    n_features : int
        Width of the encoded parent features.
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.fc = nn.Linear(n_features, 1, bias=False)

    @property
    def weight(self) -> Tensor:
        """Tensor: the shift coefficients, shape ``(n_features,)``."""
        return self.fc.weight.squeeze(0)

    def forward(self, x: Tensor) -> Tensor:
        """Compute the shift contribution.

        Parameters
        ----------
        x : Tensor
            Encoded parent features, shape ``(n, n_features)``.

        Returns
        -------
        Tensor
            The shift, shape ``(n,)``.
        """
        return self.fc(x).squeeze(-1)


class ComplexShift(nn.Module):
    """Additive shift ``g(x)`` from an MLP, still additive on the latent scale.

    Parameters
    ----------
    n_features : int
        Width of the encoded parent features.
    units : tuple[int, ...] | None, optional
        Hidden layers of the network, by default ``(64, 128, 64)`` — the three
        hidden layers of ``ComplexShiftDefaultTabular`` (see the module
        docstring). The paper's own nets are narrower; a replication should
        set this explicitly.
    activation : str | None, optional
        Key of :data:`ACTIVATIONS`, by default :data:`DEFAULT_ACTIVATION`.
    """

    def __init__(
        self,
        n_features: int,
        units: tuple[int, ...] | None = None,
        activation: str | None = None,
    ):
        super().__init__()
        self.net = _mlp(n_features, units or (64, 128, 64), 1, activation=activation)

    def forward(self, x: Tensor) -> Tensor:
        """Compute the shift contribution.

        Parameters
        ----------
        x : Tensor
            Encoded parent features, shape ``(n, n_features)``.

        Returns
        -------
        Tensor
            The shift, shape ``(n,)``.
        """
        return self.net(x).squeeze(-1)


class VaryingCoef(nn.Module):
    """Varying-coefficient effect head ``beta(x) = beta0 + b_theta(x)``.

    ``b_theta`` is deliberately small: one hidden layer by default. Its
    weights carry an L2 ``penalty`` in the fitting objective (see
    :meth:`l2`; ``fit`` adds ``penalty * l2()`` on the total-NLL scale).
    ``beta0`` is not penalized.

    The output layer starts at zero, so ``beta(x)`` equals ``beta0``
    exactly at construction. The head therefore learns only the deviation
    from a constant effect, which makes the arm difference an estimate
    instead of a by-product. The unpenalized reduced form ``CS(on, x...)``
    reaches a correlation of only about 0.5 against the true effect
    function (issue #28).

    With ``n_features == 0`` there are no modifiers, there is no network, and the
    term is exactly ``LS(on)``.

    Parameters
    ----------
    n_features : int
        Width of the encoded modifier features. Use 0 for no modifiers.
    penalty : float, optional
        L2 weight on ``b_theta``, by default ``1.0``.
    units : tuple[int, ...] | None, optional
        Hidden layers of ``b_theta``, by default ``(16,)``. One layer of 16 is
        the head ``tests/test_vc_term.py`` recovers a known ``beta(x)`` with
        at corr ~ 0.99; this term has no counterpart in the reference
        implementations, so the size comes from that measurement.
    activation : str | None, optional
        Key of :data:`ACTIVATIONS`, by default :data:`DEFAULT_ACTIVATION`.

    Notes
    -----
    A constant can move freely between ``beta0`` and ``b_theta``, so the split is
    not identified by the likelihood alone. The penalty resolves it during
    training, because it shrinks ``b_theta`` toward the zero function. After
    training, :meth:`recenter` re-splits the two exactly: ``b_theta`` then sums
    to zero over the training data, the GAM convention that
    ``intercept_contributions`` also uses. Recentering is a reparameterization
    through the ``center`` buffer and leaves the modelled function unchanged.
    """

    def __init__(
        self,
        n_features: int,
        penalty: float = 1.0,
        units: tuple[int, ...] | None = None,
        activation: str | None = None,
    ):
        super().__init__()
        self.penalty = float(penalty)
        self.beta0 = nn.Parameter(torch.zeros(()))
        self.register_buffer("center", torch.zeros(()))
        if n_features > 0:
            # zero-initialised output: beta(x) == beta0 at init
            self.net = _mlp(
                n_features,
                units or (16,),
                1,
                activation=activation,
                zero_init_last=True,
            )
        else:
            self.net = None

    def beta(self, mod_feats: Tensor | None, n: int) -> Tensor:
        """Compute the effect values ``beta(x)``.

        Parameters
        ----------
        mod_feats : Tensor | None
            Encoded modifier features. ``None`` if, and only if, the term has no
            modifiers.
        n : int
            Batch size, used when the term has no modifiers.

        Returns
        -------
        Tensor
            The effect values, shape ``(n,)``.
        """
        if self.net is None:
            return (self.beta0 - self.center).expand(n)
        return self.beta0 + self.net(mod_feats).squeeze(-1) - self.center

    def forward(self, t: Tensor, mod_feats: Tensor | None) -> Tensor:
        """Compute the shift contribution ``beta(mod_feats) * t``.

        Parameters
        ----------
        t : Tensor
            The raw treatment column, shape ``(n, 1)``.
        mod_feats : Tensor | None
            Encoded modifier features, or ``None`` without modifiers.

        Returns
        -------
        Tensor
            The shift, shape ``(n,)``.
        """
        return self.beta(mod_feats, t.shape[0]) * t.squeeze(-1)

    def l2(self) -> Tensor:
        """Sum the squared ``b_theta`` weights, the penalized quantity.

        ``beta0`` is never included.

        Returns
        -------
        Tensor
            The sum of squares, 0 without modifiers.
        """
        if self.net is None:
            return torch.zeros((), device=self.beta0.device, dtype=self.beta0.dtype)
        return sum(p.pow(2).sum() for p in self.net.parameters())

    @torch.no_grad()
    def recenter(self, mod_feats: Tensor | None) -> None:
        """Re-split ``beta0`` and ``b_theta`` so ``b_theta`` has mean zero.

        The mean is taken over ``mod_feats``. The removed constant moves into
        ``beta0``, so the modelled function does not change.

        Parameters
        ----------
        mod_feats : Tensor | None
            Encoded modifier features. Without modifiers this does nothing.

        Returns
        -------
        None
        """
        if self.net is None:
            return
        delta = (self.net(mod_feats).squeeze(-1) - self.center).mean()
        self.center += delta
        self.beta0 += delta
