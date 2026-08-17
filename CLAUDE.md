# CLAUDE.md — working context for tramdag

## What this is

A causal normalizing-flow implementation of **TRAM-DAG** (transformation models on a
DAG) built on [zuko](https://zuko.readthedocs.io/stable/). One triangular flow from iid
standard-logistic latents to the observed variables; Jacobian sparsity = the DAG.
Supports the do-operator, Pearl abduction (counterfactuals), analytic interventional
PMFs, and per-node configurable monotone transforms (Bernstein / RQ-spline / affine).

Origin: extracted from the private `tensorchiefs/tram-dag-stroke` paper repo (as
`zuko_dag`; renamed to `tramdag` in June 2026, repo `tensorchiefs/tramdag`). The paper analyzed the MAGIC stroke cohort against the MR CLEAN RCT;
that **clinical data is NOT in this repo** and never should be. The synthetic
`data/magic-mrclean/` cohort is the public stand-in (same schema, known ground truth).

## Commands

```bash
uv sync                          # install (uv.lock pinned: zuko, torch, statsmodels, ...)
uv run pytest tests/ -q          # full suite; -m "not slow" is ~2 min
cd experiments
uv run python sim_flow.py nl     # headline storyline (all-ls vs flexible vs known truth)
uv run python validate_ls.py     # spot-on flow == statsmodels == R polr check
uv run python paper_triangle.py atan cs  # CLeaR replications (paper_*.py);
                                 # also pinned by tests/test_paper_dgps.py
```

Experiments default to the synthetic data (`magic-mrclean/nl`). The `magic` source
(private clinical data) only works inside the original paper monorepo.

## Architecture (src/tramdag/)

- `spec.py` — user-facing DAG spec: `{name: ContinuousNode|OrdinalNode}`, each node
  declares its transformation as the first positional argument — a list of terms
  or a `+` sum (`I("a") + LS("b")`), since 0.4.0 (the 0.3 `terms=` keyword and
  node-level `transform=` were removed; 0.3 checkpoints still load). Term constructors: `LS(parent)`
  (linear shift), `CS(*parents)` (complex shift MLP), `I(*parents)` (complex
  intercept — transform params from parents), `VC(*modifiers, t=, penalty=)`
  (varying-coefficient effect head `beta(modifiers)·x_t`, small penalized
  zero-init net; read out with `flow.varying_coef` — see
  docs/varying-coefficients.md). `I(..., transform=)` picks the monotone basis,
  `units=[...]` on I/CS/VC sizes the term's network. A node takes at most ONE
  I term with parents; a **multi-parent** `I("a","b")` is one *joint* network
  (interaction on the thetas, the default), and
  `I("a","b", allow_interaction=False)` is the *additive* intercept (one net
  per parent, coefficient vectors summed). For shifts, grouping decides:
  `CS("a","b")` is joint, `CS("a")+CS("b")` additive. `term(effect, *parents)` builds a term from
  a data-driven label. No intercept term → `SimpleIntercept` baseline. Every
  parent enters through exactly one edge-owning term (VC modifiers exempt — they
  may also appear prognostically).
- `transforms.py` — monotone 1-D transforms wrapping zuko (`BernsteinUT`, `SplineUT`,
  `AffineUT`; pre-scaled from train 5%/95% quantiles to [-5,5], expanding-bracket
  bisection inverse) + the ordinal ordered-logit transform
  (`P(Y<=k) = sigmoid(theta_k - shift)`, cutpoints `[t0, t0+cumsum(exp(...))]`).
- `conditioners.py` — ls/cs/ci networks (widths replicate the original Keras/TF implementation).
- `flow.py` — `CausalFlowDAG`: `fit`, `fit_classical` (float64 full-batch
  L-BFGS, exact MLE for all-`ls` specs), `sample(n, do=, u=)`, `abduct`, `pmf`,
  `log_prob`, `save/load`, `varying_coef` (VC read-out), `scores` /
  `effect_modifier_scan` (analytic per-row ∂ℓᵢ/∂θ + CUSUM modifier scan,
  `scores.py`). NLL decomposes per node → one Adam fits all nodes jointly.
- `simulations/` — numpy-only SCM generators with known ground truth, looked up via
  `REGISTRY`; each module has a CLI that regenerates its frozen `data/<name>/` CSVs:
  `magic_mrclean.py` (stroke SCM, `ls`/`nl`), `triangle.py` (paper §6 continuous +
  ordinal triangles, f variants linear/cubic/exp/atan/sin), `vaca.py` (App. C.1
  bimodal L1/L2 benchmark), `carefl.py` (App. C.2 Laplace SCM, **analytic**
  counterfactuals), `vc_shift.py` (issue #28 heterogeneous-effect DGP with
  known `beta(x)`).

## Conventions that matter (easy to get wrong)

- **Latent scale**: continuous `z = h(x) + shift` (shifts ADDED); ordinal
  `P(Y<=k) = sigmoid(theta_k − shift)` (shift SUBTRACTED). Both follow the original TRAM-DAG
  conventions; tests pin them.
- **Parent encoding**: continuous parents enter RAW (no standardization); ordinal
  parents one-hot (all levels). With cutpoints, only shift *differences* between
  one-hot levels are identified — compare `w[k] − w[0]` against classical references.
- **Ordinal log-prob is computed in log-space** (`logsigmoid` + stable `log1mexp`,
  better-conditioned side chosen per element). The naive sigmoid difference saturates
  in float32 → *exactly zero* gradients → a node can freeze at init forever. Do not
  "simplify" it back.
- **Seeding**: weight init happens at construction. Use `CausalFlowDAG(spec, seed=...)`
  (the one obvious knob) — or call `torch.manual_seed` BEFORE `CausalFlowDAG(spec)`.
  `fit(seed=...)` only seeds minibatch shuffling, not init.
- **Spline tails are slope-clamped**: zuko's RQS extrapolates with a *fixed* slope
  outside [-5,5] regardless of θ, so the ~10% of data beyond the 5%/95% pre-scaling
  range is misweighted whenever the true tail slope differs — the structural reason
  `spline` consistently trails `bernstein` (whose linear extrapolation follows the
  boundary derivative). Demonstrated in `notebooks/stale/transforms_tram_dag.py` (parked).
- **`fit(restore_best=False)` is the default** (keeps final converged weights = exact
  MLE; an all-`ls` model then matches statsmodels/R-polr to ~1e-3). `restore_best=True`
  = per-node best-validation restoration (early stopping). Key empirical finding:
  **flexible (ci/cs) models overfit observational confounding at the MLE and need
  `restore_best=True` to recover the causal effect; all-`ls` models don't.**
  `run_experiment` defaults per style. See CHANGELOG.md.

## Ground truth & reference numbers (seed 7 synthetic data)

- `data/magic-mrclean/{ls,nl}/truth.json` — true ATE from the SCM: `ls` +0.132,
  `nl` +0.104; naive confounded contrast +0.26/+0.30.
- `nl` storyline: all-`ls` flow ≈ +0.076 (biased — can't extrapolate the age-fading
  treatment effect to the younger RCT population), flexible flow ≈ +0.10 (recovers).
- Spot-on check (`ls` variant, full-data, restore_best=False): flow = statsmodels =
  R polr at Age 0.0526, NIHSSa 0.1630, T −0.9424; ATE +0.1429 vs +0.1428.
- R reference: `data/magic-mrclean/fit_ls.R` (needs `tram`, `MASS`); its committed
  `ref_ls/` outputs let tests run without R.
- Original clinical-data numbers (context only, not reproducible here): TRAM-DAG
  nihss6 +0.108, md_dag_ls +0.054, MR CLEAN RCT +0.135 [0.057, 0.213].
- **Paper DGPs** (seed 42, arXiv:2503.16206): `triangle` true coefficients β12=+2,
  β13=−0.2 (+0.3 on x2 for `linear`); a fitted `cs` learns −f(x2)+const.
  `triangle-mixed` cutpoints θ=(−2, 0.42, 1.02); **ordinal sign flip**: the paper
  ADDS the ordinal shift, the flow SUBTRACTS → fitted weights −0.2 / +0.3; the C.4
  odds-ratio check gives OR ≈ e² ≈ 7.4. `vaca`: E[x3|do(x2=a)] = −0.25 + 0.25a
  (do(x2=−3) is off-manifold extrapolation — looser tolerance). `carefl`:
  counterfactuals are analytic (`Carefl4.true_counterfactual`); the paper's x_obs has
  a ~4σ abducted noise, so tests score 300 typical rows instead of that single point.

## Testing policy

- Frozen CSVs in `data/` (`magic-mrclean`, `triangle*`, `vaca`, `carefl`,
  `vc-shift`) are a
  contract — **never regenerate silently**; a new seed/equations → new folder
  (sim2-style), regenerate `ref_ls/` with R where applicable, update
  truth-dependent tests. `test_paper_dgps.py::test_frozen_csv_contract` pins the
  paper-DGP CSVs to their generators bit-exactly.
- Fit tests for the paper DGPs train on **regenerated n=20k** (deterministic
  `observational(n, seed_offset=100)`), not the frozen n=5k — β13 multiplies the
  low-variance x1 ∈ [0.25, 0.73] and is too weakly identified at n=5k.
- New causal features should be validated against the simulator's known truth
  (`MagicMrClean.true_ate`, `counterfactual_pair` gives true individual
  counterfactuals via shared latents).

## Roadmap notes

- ~~Generalize `simulations/` registry beyond the stroke DAG~~ — done for the
  TRAM-DAG paper's DGPs (triangle/triangle-mixed/vaca/carefl, June 2026). Still
  open: hidden confounding à la DeCaFlow.
- ~~Package for PyPI~~ — published as `tramdag` (latest 0.3.0, June 2026);
  release flow:
  bump `version` in pyproject (`__init__` now reads it back from the installed
  metadata, so there is only one place to edit), `uv build`, `uv publish`
  (Oliver's PyPI token), CHANGELOG section.
