"""Univariate transforms for CausalFlowDAG nodes.

Each continuous node carries a monotone 1-D transform ``h`` (zuko-backed) that maps
the observed value to the latent scale; ordinal nodes carry a cutpoint ("ordered
logit") transform. Together with the additive shift terms they form one triangular
flow from the standard-logistic latent to the observed variables.

Conventions follow the original TRAM-DAG implementation
(Keras/TF, https://github.com/tensorchiefs/tram-dag):

- continuous: ``z = h(x) + s(parents)`` with ``h`` Bernstein / RQ-spline / affine,
  fitted on the value range scaled from the train 5%/95% quantiles to ``[-B, B]``
  and linearly extrapolated outside.
- ordinal:    ``P(x <= k) = sigmoid(theta_k - s(parents))`` with increasing
  cutpoints ``theta``. This is the parametrization of
  ``transform_intercepts_ordinal`` in the original implementation.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from zuko.transforms import (
    BernsteinTransform,
    MonotonicAffineTransform,
    MonotonicRQSTransform,
)

__all__ = [
    "StandardLogistic",
    "BernsteinUT",
    "SplineUT",
    "AffineUT",
    "make_univariate_transform",
    "ordinal_cutpoints",
    "ordinal_log_prob",
    "ordinal_pmf",
    "ordinal_sample",
    "ordinal_abduct",
]


class StandardLogistic:
    """Standard logistic base distribution (the TRAM latent)."""

    @staticmethod
    def log_prob(z: Tensor) -> Tensor:
        """Give the log density at ``z``.

        Parameters
        ----------
        z : Tensor
            Evaluation points.

        Returns
        -------
        Tensor
            The log density, same shape as ``z``.
        """
        return -z - 2.0 * torch.nn.functional.softplus(-z)

    @staticmethod
    def sample(shape, device=None, eps: float = 1e-7) -> Tensor:
        """Draw samples of the given ``shape``.

        Parameters
        ----------
        shape : tuple[int, ...]
            Shape of the sample tensor.
        device : torch.device | str | None, optional
            Target device, by default ``None``.
        eps : float, optional
            Clamp margin that keeps the uniform draw off 0 and 1, by
            default 1e-7.

        Returns
        -------
        Tensor
            The samples.
        """
        u = torch.rand(shape, device=device).clamp(eps, 1.0 - eps)
        return torch.log(u) - torch.log1p(-u)

    @staticmethod
    def icdf(u: Tensor, eps: float = 1e-7) -> Tensor:
        """Give the quantile at probability ``u``.

        Parameters
        ----------
        u : Tensor
            Probabilities in (0, 1).
        eps : float, optional
            Clamp margin that keeps ``u`` off 0 and 1, by default 1e-7.

        Returns
        -------
        Tensor
            The quantiles, same shape as ``u``.
        """
        u = u.clamp(eps, 1.0 - eps)
        return torch.log(u) - torch.log1p(-u)


def _expanding_bisection(
    f, z: Tensor, lo: Tensor, hi: Tensor, max_expand: int = 60, iters: int = 80
) -> Tensor:
    """Solve ``f(t) = z`` element-wise for a monotone increasing ``f``.

    The search starts from the bracket ``[lo, hi]`` and doubles it outward
    until the root is bracketed. This handles latent samples far in the
    tails, where zuko's built-in bisection bound clips.

    Parameters
    ----------
    f : callable
        Monotone increasing function of one tensor.
    z : Tensor
        Target values.
    lo, hi : Tensor
        Initial bracket, same shape as ``z``.
    max_expand : int, optional
        Upper limit on bracket doublings, by default 60.
    iters : int, optional
        Bisection iterations after bracketing, by default 80.

    Returns
    -------
    Tensor
        The roots, same shape as ``z``.
    """
    width = hi - lo
    for _ in range(max_expand):
        too_high = f(lo) > z
        too_low = f(hi) < z
        if not (too_high.any() or too_low.any()):
            break
        lo = torch.where(too_high, lo - width, lo)
        hi = torch.where(too_low, hi + width, hi)
        width = hi - lo
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        below = f(mid) < z
        lo = torch.where(below, mid, lo)
        hi = torch.where(below, hi, mid)
    return 0.5 * (lo + hi)


class _ScaledUT(torch.nn.Module):
    """Base class for the scaled univariate transforms.

    An affine pre-map takes ``[xmin, xmax]`` to ``[-B, B]``, then a zuko
    transform maps to the latent scale. Subclasses define ``n_params`` and
    ``_build(theta) -> zuko Transform``.

    Parameters
    ----------
    bound : float, optional
        Half-width ``B`` of the pre-scaled domain, by default 5.0.
    """

    def __init__(self, bound: float = 5.0):
        super().__init__()
        self.bound = bound
        self.register_buffer("xmin", torch.tensor(0.0))
        self.register_buffer("xmax", torch.tensor(1.0))
        self._fitted = False

    @property
    def n_params(self) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    def _build(self, theta: Tensor):  # pragma: no cover - abstract
        raise NotImplementedError

    def set_range(self, xmin: float, xmax: float) -> None:
        """Set the data range that maps onto the pre-scaled domain.

        ``fit`` calls this once with the train 5%/95% quantiles. The call
        marks the transform as fitted.

        Parameters
        ----------
        xmin, xmax : float
            The range ends. They map to ``-B`` and ``+B``.
        """
        self.xmin.fill_(float(xmin))
        self.xmax.fill_(float(xmax))
        self._fitted = True

    def _scale(self, x: Tensor) -> Tensor:
        return (x - self.xmin) / (self.xmax - self.xmin) * (2 * self.bound) - self.bound

    def _unscale(self, t: Tensor) -> Tensor:
        return (t + self.bound) / (2 * self.bound) * (self.xmax - self.xmin) + self.xmin

    @property
    def _log_dt_dx(self) -> Tensor:
        # derive dtype from the range buffers so float64 stays pure (no float32
        # literal promotion) inside fit_classical
        two_b = torch.as_tensor(
            2.0 * self.bound, dtype=self.xmin.dtype, device=self.xmin.device
        )
        return torch.log(two_b) - torch.log(self.xmax - self.xmin)

    def forward(self, theta: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
        """Map observed values to the latent scale, before the shift.

        Parameters
        ----------
        theta : Tensor
            Transform parameters, shape ``(n, P)``.
        x : Tensor
            Observed values in original units, shape ``(n,)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            The pre-shift latent ``z0`` and the log absolute Jacobian
            ``log|dz0/dx|``, both shape ``(n,)``.
        """
        t = self._scale(x)
        T = self._build(theta)
        z0, ladj = T.call_and_ladj(t)
        return z0, ladj + self._log_dt_dx

    def inverse(self, theta: Tensor, z0: Tensor) -> Tensor:
        """Map pre-shift latents back to original units.

        The inverse uses expanding-bracket bisection, so it also covers
        latents far outside the pre-scaled domain.

        Parameters
        ----------
        theta : Tensor
            Transform parameters, shape ``(n, P)``.
        z0 : Tensor
            Pre-shift latents, shape ``(n,)``.

        Returns
        -------
        Tensor
            The values in original units, shape ``(n,)``.
        """
        T = self._build(theta)
        B = torch.tensor(self.bound, dtype=z0.dtype, device=z0.device)
        with torch.no_grad():
            t = _expanding_bisection(
                T, z0, -B.expand_as(z0).clone(), B.expand_as(z0).clone()
            )
        return self._unscale(t)


class BernsteinUT(_ScaledUT):
    """TRAM-style Bernstein polynomial transform (zuko ``BernsteinTransform``).

    Parameters
    ----------
    n_coeffs : int, optional
        Number of Bernstein coefficients, by default 20.
    bound : float, optional
        Half-width of the pre-scaled domain, by default 5.0.
    """

    def __init__(self, n_coeffs: int = 20, bound: float = 5.0):
        super().__init__(bound=bound)
        self._n = n_coeffs

    @property
    def n_params(self) -> int:
        """int: number of transform parameters this transform needs."""
        return self._n

    def _build(self, theta: Tensor):
        return BernsteinTransform(theta, bound=self.bound)

    def marginal_init_theta(self, q: float = 0.05) -> Tensor:
        """Give the unconstrained Bernstein coefficients of the calibrated map.

        The coefficients describe the linear map from the pre-scaled domain
        ``[-B, B]`` onto the standard-logistic quantiles
        ``[logit(q), logit(1-q)]``.

        Parameters
        ----------
        q : float, optional
            Quantile level of the calibration, by default 0.05.

        Returns
        -------
        Tensor
            The coefficients, shape ``(n_params,)``.

        Notes
        -----
        After ``set_range``, each node's 5%/95% data quantiles already sit
        at the domain bounds -+B. A single canonical theta therefore maps
        every node's body onto the latent's 5%/95% quantiles — the right
        *scale* from step 0. zuko's default (zero) theta instead maps -+B
        onto about -6.93/+7.63, about 2.5x too steep, so early training is
        spent on rescaling. This is a pure initialization: the converged
        MLE is unchanged. See the inversion of
        ``BernsteinTransform._constrain_theta`` (cumsum of softplus
        diffs).
        """
        import math

        n = self._n
        a = math.log(q) - math.log(1.0 - q)  # logit(q), e.g. -2.9444 at q=.05
        span = -2.0 * a  # logit(1-q) - logit(q)
        order = n + 1  # constrained control points: n+2
        b = span / order  # per-step increment (constant)
        shift = math.log(2.0) * n / 2.0  # zuko's centering offset
        theta = torch.full(
            (n,),
            math.log(math.expm1(b)),
            dtype=self.xmin.dtype,
            device=self.xmin.device,
        )
        theta[0] = a + shift
        return theta


class SplineUT(_ScaledUT):
    """Monotone rational-quadratic spline (zuko ``MonotonicRQSTransform``).

    Parameters
    ----------
    bins : int, optional
        Number of spline bins, by default 8.
    bound : float, optional
        Half-width of the pre-scaled domain, by default 5.0.
    """

    def __init__(self, bins: int = 8, bound: float = 5.0):
        super().__init__(bound=bound)
        self.bins = bins

    @property
    def n_params(self) -> int:
        """int: number of transform parameters this transform needs."""
        return 3 * self.bins - 1

    def _build(self, theta: Tensor):
        K = self.bins
        widths, heights, derivs = (
            theta[..., :K],
            theta[..., K : 2 * K],
            theta[..., 2 * K :],
        )
        return MonotonicRQSTransform(widths, heights, derivs, bound=self.bound)


class AffineUT(_ScaledUT):
    """Monotone affine transform: the node-conditional is a logistic GLM."""

    @property
    def n_params(self) -> int:
        """int: number of transform parameters this transform needs."""
        return 2

    def _build(self, theta: Tensor):
        return MonotonicAffineTransform(theta[..., 0], theta[..., 1])


_TRANSFORMS = {"bernstein": BernsteinUT, "spline": SplineUT, "affine": AffineUT}


def make_univariate_transform(name: str, **kwargs) -> _ScaledUT:
    """Build a scaled univariate transform by name.

    Parameters
    ----------
    name : str
        One of the registered names: ``"bernstein"``, ``"spline"``, ``"affine"``.
    **kwargs
        Passed to the transform class.

    Returns
    -------
    _ScaledUT
        The transform.

    Raises
    ------
    ValueError
        If ``name`` is not registered.
    """
    try:
        cls = _TRANSFORMS[name]
    except KeyError:
        raise ValueError(
            f"Unknown transform '{name}'. Choose from {sorted(_TRANSFORMS)}."
        )
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Ordinal ("ordered logit") transform — exact port of the original parametrization
# ---------------------------------------------------------------------------


def ordinal_cutpoints(theta_tilde: Tensor) -> Tensor:
    """Constrain unconstrained parameters to increasing cutpoints.

    Port of the original implementation's
    ``transform_intercepts_ordinal``:
    ``[-inf, t0, t0 + cumsum(exp(t1:)), +inf]``.

    Parameters
    ----------
    theta_tilde : Tensor
        Unconstrained cutpoint parameters, shape ``(n, K-1)``.

    Returns
    -------
    Tensor
        Increasing cutpoints with ``-inf``/``+inf`` ends, shape
        ``(n, K+1)``.
    """
    n = theta_tilde.shape[0]
    neg_inf = torch.full(
        (n, 1), -torch.inf, device=theta_tilde.device, dtype=theta_tilde.dtype
    )
    pos_inf = torch.full(
        (n, 1), torch.inf, device=theta_tilde.device, dtype=theta_tilde.dtype
    )
    first = theta_tilde[:, :1]
    if theta_tilde.shape[1] > 1:
        rest = first + torch.cumsum(torch.exp(theta_tilde[:, 1:]), dim=1)
        return torch.cat([neg_inf, first, rest, pos_inf], dim=1)
    return torch.cat([neg_inf, first, pos_inf], dim=1)


def ordinal_marginal_init_theta(counts, eps: float = 1e-3) -> Tensor:
    """Give the unconstrained cutpoint parameters that match class counts.

    The marginal ``P(Y<=k) = sigmoid(cutpoint_k)`` of the result matches
    the empirical class frequencies.

    Parameters
    ----------
    counts : array-like
        Per-class count vector, length K.
    eps : float, optional
        Clamp margin for the empirical CDF and the cutpoint differences,
        by default 1e-3. It guards empty classes.

    Returns
    -------
    Tensor
        The unconstrained parameters ``theta_tilde``, shape ``(K-1,)``.

    Notes
    -----
    This inverts ``ordinal_cutpoints``. The finite cutpoints are
    ``c_0 = tt[0]`` and ``c_i = c_0 + sum_{j<=i} exp(tt[j])``. Given the
    target ``c_k = logit(F(k))`` (empirical CDF, clamped off 0/1),
    recover ``tt[0] = c_0`` and ``tt[i] = log(c_i - c_{i-1})``. Like the
    Bernstein marginal-init, this is a pure initialization: the converged
    MLE is unchanged.
    """
    import numpy as np

    counts = np.asarray(counts, dtype=np.float64)
    p = counts / counts.sum()
    F = np.clip(np.cumsum(p)[:-1], eps, 1 - eps)  # P(Y<=k), k=0..K-2
    c = np.log(F) - np.log1p(-F)  # logit -> increasing
    c = np.maximum.accumulate(c)  # guard ties (empty classes)
    diffs = np.maximum(np.diff(c), eps)
    tt = np.empty_like(c)
    tt[0] = c[0]
    tt[1:] = np.log(diffs)
    return torch.as_tensor(tt)


def _bounds(theta_tilde: Tensor, shift: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """Give the shifted cutpoint interval of each observed level."""
    cut = ordinal_cutpoints(theta_tilde) - shift.view(-1, 1)
    idx = torch.arange(theta_tilde.shape[0], device=theta_tilde.device)
    y = y.long()
    return cut[idx, y], cut[idx, y + 1]


def _log1mexp(x: Tensor) -> Tensor:
    """log(1 - exp(x)) for x <= 0, numerically stable (Maechler 2012)."""
    branch = x > -math.log(2.0)
    # mask each branch's input so the unused branch cannot produce inf/NaN grads
    x_hi = x.clamp(min=-math.log(2.0))
    x_lo = x.clamp(max=-math.log(2.0))
    return torch.where(
        branch, torch.log(-torch.expm1(x_hi)), torch.log1p(-torch.exp(x_lo))
    )


def ordinal_log_prob(theta_tilde: Tensor, shift: Tensor, y: Tensor) -> Tensor:
    """Give ``log P(Y = y | cutpoints, shift)``.

    The model is ``P(Y <= k) = sigmoid(theta_k - shift)``.

    Parameters
    ----------
    theta_tilde : Tensor
        Unconstrained cutpoint parameters, shape ``(n, K-1)``.
    shift : Tensor
        Total shift, shape ``(n,)``.
    y : Tensor
        Observed levels in ``0..K-1``, shape ``(n,)``.

    Returns
    -------
    Tensor
        The log probabilities, shape ``(n,)``.

    Notes
    -----
    The computation stays in log-space. Both of these identities hold::

        log(sigmoid(u) - sigmoid(l))
            = logsigmoid(u) + log1mexp(logsigmoid(l) - logsigmoid(u))
            = logsigmoid(-l) + log1mexp(logsigmoid(-u) - logsigmoid(-l))

    For each element the function takes the side whose logsigmoids are far from
    zero, because that side is better conditioned.

    **Do not replace this with the direct difference of two sigmoids.** That
    form loses all gradient when the sigmoids saturate in float32, which happens
    for ``|t| > 17`` or so. The gradient is then exactly zero and a badly
    initialised node freezes at its starting values forever. The log-space form
    keeps the gradient non-zero, so such a node recovers.
    """
    lower, upper = _bounds(theta_tilde, shift, y)
    ls = torch.nn.functional.logsigmoid
    cdf_side = ls(upper) + _log1mexp((ls(lower) - ls(upper)).clamp(max=-1e-7))
    srv_side = ls(-lower) + _log1mexp((ls(-upper) - ls(-lower)).clamp(max=-1e-7))
    return torch.where(upper + lower > 0, srv_side, cdf_side)


def ordinal_pmf(theta_tilde: Tensor, shift: Tensor) -> Tensor:
    """Give the class probabilities of every row.

    Parameters
    ----------
    theta_tilde : Tensor
        Unconstrained cutpoint parameters, shape ``(n, K-1)``.
    shift : Tensor
        Total shift, shape ``(n,)``.

    Returns
    -------
    Tensor
        The class probabilities, shape ``(n, K)``.
    """
    cdf = torch.sigmoid(ordinal_cutpoints(theta_tilde) - shift.view(-1, 1))
    return cdf[:, 1:] - cdf[:, :-1]


def ordinal_sample(theta_tilde: Tensor, shift: Tensor, z: Tensor) -> Tensor:
    """Map latents to ordinal levels.

    The rule is ``x = #{finite cutpoints theta_j - shift < z}``.

    Parameters
    ----------
    theta_tilde : Tensor
        Unconstrained cutpoint parameters, shape ``(n, K-1)``.
    shift : Tensor
        Total shift, shape ``(n,)``.
    z : Tensor
        Latent values, shape ``(n,)``.

    Returns
    -------
    Tensor
        The levels as floats, shape ``(n,)``.
    """
    finite = ordinal_cutpoints(theta_tilde)[:, 1:-1] - shift.view(-1, 1)
    return (z.view(-1, 1) > finite).sum(dim=1).float()


def ordinal_abduct(
    theta_tilde: Tensor,
    shift: Tensor,
    y: Tensor,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Abduct the latent of an ordinal node (Pearl step 1).

    The latent is drawn from the standard logistic, truncated to the
    interval that is consistent with the observed level.

    Parameters
    ----------
    theta_tilde : Tensor
        Unconstrained cutpoint parameters, shape ``(n, K-1)``.
    shift : Tensor
        Total shift, shape ``(n,)``.
    y : Tensor
        Observed levels in ``0..K-1``, shape ``(n,)``.
    generator : torch.Generator | None, optional
        Random source for the truncated draw, by default ``None``.

    Returns
    -------
    Tensor
        The latents, shape ``(n,)``.
    """
    lower, upper = _bounds(theta_tilde, shift, y)
    u_lo, u_hi = torch.sigmoid(lower), torch.sigmoid(upper)
    u = u_lo + (u_hi - u_lo) * torch.rand(
        lower.shape, device=lower.device, generator=generator
    )
    return StandardLogistic.icdf(u)
