# Per-observation scores & the effect-modifier scan

`flow.scores(df, node)` returns the score ψᵢ = ∂ℓᵢ/∂θ of each observation
for the node's interpretable shift coefficients. These coefficients are every
`LS` weight and every `VC` term's `beta0`. An `LS` weight gives one column per
continuous parent and one column per one-hot level of an ordinal parent. The
`beta0` column is named after the treatment.

The computation is **analytic and exact**, with no autograd. Shifts enter the
latent additively. Thus ∂ℓᵢ/∂β = (∂ℓᵢ/∂sᵢ)·xᵢ, with ∂ℓᵢ/∂s in closed form
(`src/tramdag/scores.py`). The function is a pure read-out. It does not touch
the fitting or sampling code paths. At a fitted MLE, the sum of each column is
≈ 0. Tests pin this behavior and include a float64 finite-difference check.

## Worked example: where does β(x) need to bend?

The score is the cheapest effect-modifier detector available. It uses the
structural-change logic of model-based recursive partitioning (Zeileis &
Hornik; Dandl et al. 2024). Before you fit an expensive model, fit the
**cheap all-`LS` model**. This fit takes seconds and is deterministic. Then
scan the treatment-coefficient scores:

```python
flow = td.CausalFlowDAG(all_ls_spec, seed=0)
flow.fit_classical(df)  # seconds, exact MLE

scan = flow.effect_modifier_scan(df, node="Y", t="T")
#         stat   p_value  crit_5pct   flag
# X2      4.21    <0.001     1.3581   True     <- true modifier
# X3      3.05    <0.001     1.3581   True     <- true modifier
# X1      0.74     0.64      1.3581  False     <- prognostic only
```

For each candidate covariate, the scan orders the scores by that covariate
and forms the scaled cumulative sum `B_j = Σ_{i≤j} ψ_(i) / (sd(ψ)·√n)`. Under
a stable coefficient, `B` is a Brownian bridge. Thus `sup|B|` has the
Kolmogorov distribution (5% critical value 1.358). A coefficient that truly
varies with the covariate makes the ordered scores drift. The flagged
covariates are the measured shortlist of `VC` modifiers
(`docs/varying-coefficients.md`):

```python
spec["Y"] = td.ContinuousNode(
    td.CS("X1", "X2", "X3") + td.VC("X2", "X3", t="T")
)  # scan-informed
```

Notes: For a binary ordinal `LS` treatment, `t` resolves to the identified
level-1-vs-0 contrast. For a `VC` treatment, `t` resolves to `beta0`.
Candidates default to every column of `df` except the node and `t`.
Modifiers do not need to be parents. For few-level (heavily tied) candidates,
the ordering is only partial. Then read the scan as a ranking diagnostic, not
as an exact-size test. You can also use the scores later for
influence-function analyses and robust standard errors.
