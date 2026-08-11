# Per-observation scores & the effect-modifier scan

`flow.scores(df, node)` returns each observation's score ψᵢ = ∂ℓᵢ/∂θ for the
node's interpretable shift coefficients — every `LS` weight (one column per
continuous parent, one per one-hot level for an ordinal parent) and every `VC`
term's `beta0` (column named after the treatment). The computation is
**analytic and exact** (no autograd): shifts enter the latent additively, so
∂ℓᵢ/∂β = (∂ℓᵢ/∂sᵢ)·xᵢ with ∂ℓᵢ/∂s in closed form (`src/tramdag/scores.py`).
Pure read-out — no fitting or sampling code path is touched. At a fitted MLE
the per-column sums are ≈ 0 (pinned by tests, incl. a float64
finite-difference check).

## Worked example: where does β(x) need to bend?

The score is the cheapest effect-modifier detector there is (model-based
recursive partitioning / structural-change logic — Zeileis & Hornik; Dandl et
al. 2024). Before fitting anything expensive: fit the **cheap all-`LS` model**
(seconds, deterministic), and scan the treatment-coefficient scores —

```python
flow = td.CausalFlowDAG(all_ls_spec, seed=0)
flow.fit_classical(df)  # seconds, exact MLE

scan = flow.effect_modifier_scan(df, node="Y", on="T")
#         stat   p_value  crit_5pct   flag
# X2      4.21    <0.001     1.3581   True     <- true modifier
# X3      3.05    <0.001     1.3581   True     <- true modifier
# X1      0.74     0.64      1.3581  False     <- prognostic only
```

For each candidate covariate the scores are ordered by it and the scaled
cumulative sum `B_j = Σ_{i≤j} ψ_(i) / (sd(ψ)·√n)` is formed; under a stable
coefficient `B` is a Brownian bridge, so `sup|B|` has the Kolmogorov
distribution (5% critical value 1.358). A coefficient that truly varies with
the covariate makes the ordered scores drift — flagged covariates are the
measured shortlist of `VC` modifiers (`docs/varying-coefficients.md`):

```python
spec["Y"] = td.ContinuousNode(
    td.CS("X1", "X2", "X3") + td.VC("X2", "X3", t="T")
)  # scan-informed
```

Notes: `on` resolves to the identified level-1-vs-0 contrast for a binary
ordinal `LS` treatment and to `beta0` for a `VC` treatment. Candidates default
to every column of `df` except the node and `on` — modifiers need not be
parents. For few-level (heavily tied) candidates the ordering is only partial:
read the scan as a ranking diagnostic rather than an exact-size test. Scores
also enable influence-function analyses and robust standard errors later.
