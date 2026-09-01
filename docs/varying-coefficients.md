# Varying-coefficient treatment effects: `VC(*modifiers, t=, penalty=)`

A `VC` term gives a node a **treatment-effect head with its own bias–variance
budget** (issue #28). The term contributes

```
beta(modifiers) * x_on        with    beta(x) = beta0 + b_theta(x)
```

to the node's shift. Here `b_theta` is a deliberately small (one 16-unit
hidden layer), **penalized** network. The point is not expressiveness. The
point is that the flow estimates the effect function *with care*, not as a
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
        + td.VC("X2", "X3", t="T", penalty=1.0)  # effect beta(X2, X3) * T
    ),
}
flow = td.CausalFlowDAG(spec, seed=0).fit(
    train, epochs=500, learning_rate=1e-2, batch_size=512
)  # best-validation weights: callbacks.EarlyStopping, see fitting.md

beta = flow.varying_coef(df_new, "Y")  # (n,) array beta(x) — deterministic, y-free
beta0 = float(flow.nodes["Y"].shifts["T"].beta0)  # interpretable main effect
```

`X2` and `X3` appear **twice**: as prognostic parents through `CS` and as
effect modifiers through `VC`. This pattern is intended. Only the treatment
node named by `t=` owns its edge, and a second term that declares it raises
an error. Modifiers can repeat.

## Why not `CS("T", "X2", "X3")`? (anti-pattern)

For a binary treatment, the multi-parent `CS` is equivalent in
*expressiveness*. Any shift decomposes exactly as
`s(x,t) = s(x,0) + [s(x,1) − s(x,0)]·t`. But the `CS` form has **no
effect-specific regularization**. The likelihood rewards a good average fit
of `s(x,t)`, and nothing rewards a smooth difference between the arms. The
read-out is thus the difference of two jointly fitted, unregularized
networks, which amplifies noise.

On the heterogeneous-effect validation DGP (see below), the `CS` reduced form reaches
corr ≈ 0.5 against the true effect function (tramdag-simu#18 / PR #21). This
holds **even when the model is exactly in-class**. The `VC` term reaches
corr ≈ 0.99 on the same protocol (`tests/test_vc_term.py`, acceptance bar
0.9). Causal forests and R-learners work not because they *target* the
effect, but because they **regularize** it (Nie & Wager 2021,
Athey–Tibshirani–Wager 2019). `VC` brings that ingredient into the TRAM
framework.

## Semantics

- **Scale**: `beta(x)` lives on the node's latent (log-odds) scale. The flow
  adds it for a continuous node (`u = h(y) + …`) and subtracts it from the
  cutpoints for an ordinal node, exactly like an `LS` weight. With no
  modifiers, `VC(t="T")` *is* `LS("T")` (identical model, testably
  bit-exact). Thus `VC` versus `LS` is a nested question.
- **Penalty**: the fitting objective is the penalized likelihood
  `Σᵢ NLLᵢ + penalty · ‖b_theta weights‖²` on the total-NLL scale. This is a
  fixed Gaussian prior whose shrinkage vanishes as n grows. The penalty
  never applies to `beta0`. `penalty → ∞` shrinks `b_theta` to the zero
  function and recovers the classical `LS` fit. The default is
  `penalty=1.0`. If the modifiers are many or n is small, raise the penalty.
- **Identification / centering**: a constant moves freely between `beta0`
  and `b_theta`. The head's output layer is zero-initialized
  (`beta(x) = beta0` at step 0). After `fit`, the flow re-centers the head
  to mean zero over the training data, which preserves the function. Thus
  `beta0` is the main effect in the training population, the `Colr` reading
  when `beta` is constant.
- **Warm start** (optional, two lines): fit the all-`ls` version of the
  node classically and copy the treatment weight into
  `flow.nodes[node].shifts[t].beta0` before `fit`, so training starts at the
  classical answer and only learns deviations. Measured on the `vc_hetero`
  DGP: `beta0` lands within 0.15 of the truth with it, 0.16 from the zero
  start; the recovery correlation is 0.99 either way.
- **Treatments**: the treatment `x_on` can be continuous or binary (2-level)
  ordinal. The term is linear in `x_on`. A binary ordinal enters as its 0/1
  level, so `beta` is the identified level-1-vs-0 contrast. Multi-level
  ordinal treatments are a planned follow-up.
- **Read-out**: `flow.varying_coef(df, node, t=...)` evaluates
  `beta0 + b_theta(modifiers)` in closed form. The result is deterministic
  and y-free, and it needs no abduction. For a binary treatment, it equals
  the abduct-difference `u(x, t=1, y) − u(x, t=0, y)` identically (pinned by
  a test).

## Propensity-centered VC: `center=True` (R-learner orthogonalization)

```python
td.VC("X2", "X3", t="T", penalty=1.0, center=True)
# contributes  beta(x) * (t - e_hat(x))  to the shift; e_hat comes from you
```

This is Robinson/R-learner centering inside the likelihood. Dandl et al.
(2024) found treatment-centering to be *the* decisive ingredient for effect
estimation under confounding in model-based forests. That finding reproduces
here. The test DGP is strongly confounded, and the model deliberately
under-specifies its prognostic part (true g(x) quadratic, model linear). On
that DGP, the uncentered β̂ absorbs the confounded misfit. The mean
|β̂ − τ| is ≈ 1.1–1.2, which effectively destroys the effect estimate. The
centered β̂ stays near truth (≈ 0.1–0.3), a **5–10× bias reduction**
(`tests/test_vc_centered.py`). If the prognostic part is correctly
specified, centering changes little. Centering is insurance against the
misspecification that you do not know you have.

The naive implementations are wrong. Thus this is a **two-stage frozen**
design:

- **Training** uses **out-of-fold** ê that *you* compute and pass as
  `fit(vc_ehat={"Y": {"T": e_oof}})`, one value per training row: any
  propensity model, each fold predicted by a fit that never saw it. This is
  the DML cross-fitting requirement — in-sample ê reintroduces the
  own-observation bias and can be *worse* than no centering. Six lines with
  the flow's own classical fit:

  ```python
  fold_id = np.random.default_rng(0).permutation(len(train)) % 5
  e_oof = np.empty(len(train))
  for j in range(5):
      proxy = td.CausalFlowDAG(t_spec, seed=0)          # the treatment node's spec
      proxy.fit_classical(train.iloc[fold_id != j][["X", "T"]])
      e_oof[fold_id == j] = proxy.pmf(train.iloc[fold_id == j], "T")[:, 1]
  ```

  The OOF values enter the outcome loss as frozen data. Thus **no gradient
  reaches the treatment node** from the outcome node, and the per-node
  factorization stays intact (pinned by a gradient-isolation test). `fit`
  refuses a centered spec without `vc_ehat` and a `vc_ehat` that does not
  match the spec.
- **Inference** (`log_prob` / `sample` / `abduct` / `pmf` / `scores`)
  recomputes ê from the flow's **own fitted treatment node**, the full-data
  fit (the standard DML train/predict split). The computation is detached
  and uses the current parent values. Under `do(T=t)`, the flow re-derives
  the regressor as `t − ê(x)`. The flow caches nothing.
- **Interpretation**: with centering, `beta0` is the effect at the treatment
  margin (the observed propensities). `varying_coef` is unchanged, because
  centering moves the regressor, not β. The LS-nesting reading applies to
  the uncentered term only. Centering requires a binary ordinal treatment.
  Continuous-treatment centering with E[T|x] is a follow-up. `center=False`
  (the default) is bit-identical to the uncentered term.

## Validation

The `vc_hetero` DGP in [`tests/conftest.py`](../tests/conftest.py) is a
logistic-shift SCM with known `beta_true(x) = −1 + 0.8·X2 − 0.6·X3`. It has a
nonlinear prognostic part and confounded assignment (X2 is confounder *and*
modifier), which is the configuration where the `CS` reduced form fails
hardest. Acceptance (`tests/test_vc_term.py`) requires three results:
recovery corr ≥ 0.9 at n = 5000 (measured ≈ 0.99, min over 3 seeds 0.986), a
fitted `beta0` that matches `fit_classical` under a large penalty, and the
read-out identities. The centering claims are measured against the
`confounded` DGP in the same file. Both former follow-ups have since
shipped: propensity centering (#30) is `center=True`, documented above, and the
per-observation scores for effect-modifier scans (#29) are `flow.scores` and
`flow.effect_modifier_scan` — see [scores.md](scores.md).
