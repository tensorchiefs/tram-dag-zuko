# Varying-coefficient treatment effects: `VC(on, *modifiers, penalty=)`

A `VC` term gives a node a **treatment-effect head with its own bias–variance
budget** (issue #28): it contributes

```
beta(modifiers) * x_on        with    beta(x) = beta0 + b_theta(x)
```

to the node's shift, where `b_theta` is a deliberately small (one 16-unit
hidden layer), **penalized** network. The point is not expressiveness — it is
that the effect function is *estimated with care* instead of falling out as a
by-product:

```python
import tramdag as td

spec = {
    "X1": td.ContinuousNode(),
    "X2": td.ContinuousNode(),
    "X3": td.ContinuousNode(),
    "T": td.OrdinalNode(2, td.LS("X1") + td.LS("X2")),
    "Y": td.ContinuousNode(
        td.CS("X1", "X2", "X3")  # prognostic part g(x): as flexible as you like
        + td.VC("T", "X2", "X3", penalty=1.0)  # effect beta(X2, X3) * T
    ),
}
flow = td.CausalFlowDAG(spec, seed=0).fit(train, val, restore_best=True)

beta = flow.varying_coef("Y", df_new)  # (n,) array beta(x) — deterministic, y-free
beta0 = float(flow.nodes["Y"].shifts["T"].beta0)  # interpretable main effect
```

Note that `X2`/`X3` appear **twice**: prognostically through `CS` and as effect
modifiers through `VC`. That is the intended pattern — only the treatment `on`
owns its edge (declaring it in a second term raises), modifiers may repeat.

## Why not `CS("T", "X2", "X3")`? (anti-pattern)

For a binary treatment the multi-parent `CS` is *expressively* equivalent —
any shift decomposes exactly as `s(x,t) = s(x,0) + [s(x,1) − s(x,0)]·t`. But it
has **no effect-specific regularization**: the likelihood rewards fitting
`s(x,t)` on average and nothing rewards a smooth arm-difference, so the read-out
is the difference of two jointly-fitted unregularized networks — noise-amplifying.
Measured on the `vc-shift` validation DGP task class, the `CS` reduced form
reaches corr ≈ 0.5 against the true effect function **even when the model is
exactly in-class** (tramdag-simu#18 / PR #21), while the `VC` term reaches
corr ≈ 0.99 on the same protocol (`tests/test_vc_term.py`, acceptance bar 0.9).
What makes causal forests / R-learners work is not that they *target* the
effect but that they **regularize** it (Nie & Wager 2021; Athey–Tibshirani–Wager
2019); `VC` brings that ingredient into the TRAM framework.

## Semantics

- **Scale**: `beta(x)` lives on the node's latent (log-odds) scale — added for a
  continuous node (`z = h(y) + …`), subtracted from the cutpoints for an ordinal
  node, exactly like an `LS` weight. With no modifiers it *is* `LS(on)`
  (identical model, testably bit-exact), so `VC` vs `LS` is a nested question.
- **Penalty**: the fitting objective is the penalized likelihood
  `Σᵢ NLLᵢ + penalty · ‖b_theta weights‖²` (total-NLL scale — a fixed Gaussian
  prior whose shrinkage vanishes as n grows; `beta0` is never penalized).
  `penalty → ∞` shrinks `b_theta` to the zero function and recovers the
  classical `LS` fit. Default `penalty=1.0`; raise it when modifiers are many or
  n is small.
- **Identification / centering**: a constant moves freely between `beta0` and
  `b_theta`. The head's output layer is zero-initialised (`beta(x) = beta0` at
  step 0), and after `fit` the head is re-centered to mean zero over the
  training data (function-preserving), so `beta0` is the training-population
  main effect — the `Colr` reading when `beta` is constant.
- **Warm start**: `fit(vc_warm_start=True)` (default) initialises `beta0` from
  the classical all-`ls` solution of the node's conditional (deterministic
  L-BFGS on a throwaway proxy) once per term, so training starts at the
  classical answer and only learns deviations.
- **Treatments**: `x_on` continuous or binary (2-level) ordinal (the term is
  linear in `x_on`; a binary ordinal enters as its 0/1 level, so `beta` is the
  identified level-1-vs-0 contrast). Multi-level ordinal treatments are a
  planned follow-up.
- **Read-out**: `flow.varying_coef(node, data, on=...)` evaluates
  `beta0 + b_theta(modifiers)` closed-form — deterministic, y-free, no
  abduction; for a binary treatment it equals the abduct-difference
  `u(x, t=1, y) − u(x, t=0, y)` identically (pinned by a test).

## Propensity-centered VC: `center=True` (R-learner orthogonalization)

```python
td.VC("T", "X2", "X3", penalty=1.0, center=True, center_folds=5)
# contributes  beta(x) * (t - e_hat(x))  to the shift
```

Robinson/R-learner centering inside the likelihood — Dandl et al. (2024) found
treatment-centering to be *the* decisive ingredient for effect estimation under
confounding in model-based forests, and it reproduces here: on a strongly
confounded DGP whose prognostic part the model deliberately under-specifies
(true g(x) quadratic, model linear), the uncentered β̂ absorbs the confounded
misfit (mean |β̂ − τ| ≈ 1.1–1.2, effectively destroying the effect estimate)
while the centered β̂ stays near truth (≈ 0.1–0.3) — a **5–10× bias reduction**
(`tests/test_vc_centered.py`). If the prognostic part is correctly specified,
centering changes little; it is insurance against the misspecification you
don't know you have.

The naive implementations are wrong, so this is a **two-stage frozen** design:

- **Training** uses **out-of-fold** ê: `center_folds` refits of the treatment
  node *only*, each predicting the fold it never saw (the DML cross-fitting
  requirement — in-sample ê reintroduces the own-observation bias and can be
  *worse* than no centering). The OOF values enter the outcome loss as frozen
  data, so **no gradient reaches the treatment node** from the outcome node
  (per-node factorization intact; pinned by a gradient-isolation test). Fold
  bookkeeping is exposed in `flow.vc_center_info` and pinned by tests.
  `center="colname"` supplies your own cross-fitted propensities instead.
- **Inference** (`log_prob` / `sample` / `abduct` / `pmf` / `scores`) recomputes
  ê from the flow's **own fitted treatment node** — the full-data fit (the
  standard DML train/predict split) — detached, from the current parent
  values. Under `do(T=t)` the regressor is re-derived as `t − ê(x)`; nothing is
  ever cached.
- **Interpretation**: `beta0` is now the effect at the treatment margin (the
  observed propensities); `varying_coef` is unchanged (centering moves the
  regressor, not β). The LS-nesting reading applies to the uncentered term
  only. Requires a binary ordinal treatment (continuous-treatment centering
  E[T|x] is a follow-up). `center=False` (default) is bit-identical to the
  uncentered term.

## Validation

`tramdag.simulations.VCLogisticShift` (`data/vc-shift/`, frozen contract) is a
logistic-shift SCM with known `beta_true(x) = −1 + 0.8·X2 − 0.6·X3`, a nonlinear
prognostic part, and confounded assignment (X2 is confounder *and* modifier).
Acceptance (`tests/test_vc_term.py`): recovery corr ≥ 0.9 at n = 5000 (measured
≈ 0.99; min over 3 seeds 0.986), fitted `beta0` matches `fit_classical` under a
large penalty, and the read-out identities. Candidate follow-ups (separate
issues): propensity-centered `beta(x)·(t − ê(x))` (#30), per-observation scores
for effect-modifier scans (#29).
