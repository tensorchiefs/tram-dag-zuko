# Upstream candidates for zuko

What tramdag works around in [zuko](https://zuko.readthedocs.io/stable/)
(1.6.0), ranked by value/effort as candidate upstream PRs/issues. tramdag uses
zuko surgically — only `zuko.transforms` (`BernsteinTransform`,
`MonotonicRQSTransform`, `MonotonicAffineTransform`) behind the three wrappers
in `src/tramdag/transforms.py`; the flow machinery, base distribution and DAG
structure are tramdag's own. Suggested order of attack: 4 → 1 → 3 → 2 → 5.

## 1. Analytic `call_and_ladj` for `BernsteinTransform`

`BernsteinTransform` inherits `MonotonicTransform.call_and_ladj`, which gets
the Jacobian by `torch.autograd.grad` in the forward pass. That roughly
doubles the training graph and makes the per-node loss un-`torch.compile`-able
(double backward; see docs/research/REPORT.md). The derivative is closed form
(`f'(x) = order·Δθ` against the order−1 basis — zuko already computes it in
`_offset_and_slope` for the tail slopes), so an override is ~25 lines with no
API change, consistent with `MonotonicRQSTransform` which already has one.
tramdag gets a free speedup and the `torch.compile` axis back.

## 2. Learnable/linear tails for `MonotonicRQSTransform`

zuko pads the knot derivatives with `exp(0)=1` and extrapolates as the
identity outside `[-B, B]`, so the tail slope is fixed at 1 regardless of θ —
the structural reason `spline` consistently trails `bernstein` here (~10% of
data sits beyond the 5%/95% pre-scaling range; CLAUDE.md, `spec.py`, demo
notebook section 6). Upstream: accept boundary derivatives (shape `(*, K+1)`)
or an opt-in `tails="linear"`; identity tails are deliberate in the NSF
design, so the framing must be back-compatible. Would delete the caveats and
make `spline` a first-class basis choice.

## 3. Public inverse of `_constrain_theta`

`BernsteinUT.marginal_init_theta` closed-form-inverts zuko's *private*
`_constrain_theta` (cumsum-of-softplus), hard-coding its internal
`log(2)·n/2` centering — any upstream reparametrization silently breaks the
calibrated start. A public `unconstrain_theta(theta)` (or a
`linear_theta(a, b, n, bound)` classmethod) is ~15 lines upstream and removes
the private-detail coupling. Framing: "identity/linear initialization
support", a standard flow trick.

## 4. Docstring fix: the Bernstein θ-shape off-by-one

zuko's docstring says θ has shape `(*, M-2)` for a degree-M polynomial; `n`
unconstrained coefficients become `n+2` control points = degree `n+1`, so the
correct claim is `(*, M-1)`. This off-by-one is a real replication trap
(order 21 vs the paper's `len_theta=20` = order 19) that costs tramdag prose
in three docs. Trivial PR, near-certain accept.

## 5. `Logistic` in `zuko.distributions`

Neither zuko nor torch ships a Logistic distribution; tramdag hand-rolls
`StandardLogistic` (~50 lines: `log_prob`, `sample`, `icdf`). Logistic latents
are standard in transformation models and discrete flows, and zuko has
precedent (`GeneralizedNormal`) — but zuko may point at
`TransformedUniform(SigmoidTransform().inv)`, and tramdag would keep its
`_U_EPS` clamp locally either way. Lowest value of the five.

## Anti-candidates (look upstreamable, are not)

- **Quantile pre-scaling / `_ScaledUT`**: a modeling choice; zuko's
  `ComposedTransform` with `MonotonicAffineTransform` already composes it.
- **The ordinal ordered-logit transform and log-space `ordinal_log_prob`**:
  not a bijection, outside zuko's flow scope (torch.distributions material,
  if anywhere).
- **LS/CS/CI/VC terms, marginal-init policy, propensity centering, scores**:
  TRAM-DAG semantics. zuko's `MaskedMLP` already supports arbitrary
  adjacency; tramdag doesn't use it because it needs per-term
  interpretability.
- **Spline `slope` / Bernstein `eps` knobs**: already exposed upstream.
