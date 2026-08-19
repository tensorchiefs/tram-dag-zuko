# CLAUDE.md — working context for tramdag

## What this is

A causal normalizing-flow implementation of **TRAM-DAG** (transformation models on a
DAG) built on [zuko](https://zuko.readthedocs.io/stable/). One triangular flow from iid
standard-logistic latents to the observed variables; Jacobian sparsity = the DAG.
Supports the do-operator, Pearl abduction (counterfactuals), analytic interventional
PMFs, and per-node configurable monotone transforms (Bernstein / RQ-spline / affine).

Origin: extracted from the private `tensorchiefs/tram-dag-stroke` paper repo (as
`zuko_dag`; renamed to `tramdag` in June 2026, repo `tensorchiefs/tramdag`). The
clinical stroke storyline of that paper lives in its own repository — it is **not**
here, and neither is any patient data.

**The split that matters:** `src/tramdag/` is framework code only. Research code —
the SCM generators, frozen datasets and paper replications — lives in
`experiments/`, outside the installed package (`pre-experiments-cut` is the tag
before that separation; use it to recover deleted research code). The test suite
does not import `experiments/`: it measures against three inline DGPs in
`tests/conftest.py`.

## Commands

```bash
uv sync                              # install (uv.lock pinned: zuko, torch, ...)
uv run pytest tests/ -q              # full suite; -m "not slow" is ~1 min
cd experiments                       # experiments run as modules, per area
uv run python -m paper.triangle atan-cs      # one replication (config in the YAML)
uv run python -m misc.validate_ls classical  # flow == statsmodels == R polr
uv run python -m check paper triangle-atan-cs  # metrics vs committed ground truth
uv run python -m paper.check_data            # frozen data still regenerates
```

Every experiment reads its hyperparameters from its sibling `<script>.yaml` and
has **no defaults in code**; `tramdag.load_config` rejects a variant with a
missing or unknown key. `experiments/` is split into `paper/`, `benchmarks/` and
`misc/`, each with its own `data/`, `ground_truth/`, `tests/` and `results/`;
only `common.py` (output layout) and `check.py` (ground-truth comparison) are
shared. The area tests run in the ordinary `uv run pytest`.
See `experiments/README.md`.

## Architecture (src/tramdag/)

- `spec.py` — user-facing DAG spec: `{name: ContinuousNode|OrdinalNode}`, each node
  declares its transformation as the first positional argument — a list of terms
  or a `+` sum (`I("a") + LS("b")`). Term constructors: `SI()` (simple
  intercept, no parents), `CI(*parents)` (complex intercept — transform params
  from parents), `I(*parents)` (the fallback, dispatching on its arguments),
  `LS(parent)` (linear shift), `CS(*parents)` (complex shift MLP),
  `VC(*modifiers, t=, penalty=)` (varying-coefficient effect head
  `beta(modifiers)·x_t`, small penalized zero-init net; read out with
  `flow.varying_coef` — see docs/varying-coefficients.md). Each has a pythonic
  long name (`simple_intercept`, `complex_shift`, ...) aliased to the same
  object. `transform=` on an intercept picks the monotone basis and **extra
  keyword arguments pass straight to the transform class**
  (`SI(transform="spline", bins=16)`); `units=[...]` on CI/CS/VC sizes the
  term's network. A node takes at most ONE intercept term with parents; a
  **multi-parent** `CI("a","b")` is one *joint* network (interaction on the
  thetas, the default), and `CI("a","b", allow_interaction=False)` is the
  *additive* intercept (one net per parent, coefficient vectors summed). For
  shifts, grouping decides: `CS("a","b")` is joint, `CS("a")+CS("b")` additive.
  When the effect type comes from config or the CLI, put the constructor itself
  in the table (`{"x2": CS}`). No intercept term → `SimpleIntercept` baseline.
  Every parent enters through exactly one edge-owning term (VC modifiers exempt —
  they may also appear prognostically).
- `transforms.py` — monotone 1-D transforms wrapping zuko (`BernsteinUT`, `SplineUT`,
  `AffineUT`; pre-scaled from train 5%/95% quantiles to [-5,5], expanding-bracket
  bisection inverse) + the ordinal ordered-logit transform
  (`P(Y<=k) = sigmoid(theta_k - shift)`, cutpoints `[t0, t0+cumsum(exp(...))]`).
- `conditioners.py` — the LS/CS/intercept networks (widths replicate the reference
  Keras implementation).
- `utils.py` — `load_config`: read a YAML mapping and require an exact key
  set, so a missing key cannot become a hidden default. PyYAML is lazy and
  declared as the `config` extra, so the wheel does not depend on it.
- `flow.py` — `CausalFlowDAG`: `fit`, `fit_classical` (float64 full-batch
  L-BFGS, exact MLE for all-`ls` specs), `sample(n, do=, u=)`, `abduct`, `pmf`,
  `log_prob`, `save/load`, `ls_coefficients` (LS weights only — network shifts
  are skipped), `varying_coef` (VC read-out), `scores` /
  `effect_modifier_scan` (analytic per-row ∂ℓᵢ/∂θ + CUSUM modifier scan,
  `scores.py`). NLL decomposes per node → one Adam fits all nodes jointly.

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
  boundary derivative). Demonstrated in `notebooks/demo_tram_dag_colab.py` §6,
  which also shows the consequence: the interventional *mean* survives a wrong
  transform while tail probabilities do not.
- **`fit(restore_best=False)` is the default** (keeps final converged weights = exact
  MLE; an all-`ls` model then matches statsmodels/R-polr to ~1e-3). `restore_best=True`
  = per-node best-validation restoration (early stopping). Key empirical finding:
  **flexible (CI/CS) models overfit observational confounding at the MLE and need
  `restore_best=True` to recover the causal effect; all-`ls` models don't.**
  See CHANGELOG.md.

## Ground truth & reference numbers

Framework tests (inline DGPs, `tests/conftest.py`):

- `ls_chain` — every conditional an exact linear shift; the outcome node is a
  proportional-odds model, so the flow's MLE must equal `statsmodels`
  `OrderedModel` on the same design matrix. True weights: x2←x1 +1.2,
  y←(x1 +0.4, x2 +0.6, t −0.8), cutpoints (−1.5, 0, +1.5).
- `vc_hetero` — known `beta(x) = −1 + 0.8·X2 − 0.6·X3`, confounded assignment;
  the VC acceptance bar is corr ≥ 0.9 (measured ≈ 0.99).
- `confounded` — constant effect τ = −1 with a quadratic prognostic part;
  propensity centering must cut the bias of `beta_hat` by ≥ 2× (measured 5–10×).

Experiments (`experiments/`, seed 42 unless stated, arXiv:2503.16206):

- **Paper DGPs**: `triangle` true coefficients β12=+2, β13=−0.2 (+0.3 on x2 for
  `linear`); a fitted `cs` learns −f(x2)+const. `triangle-mixed` cutpoints
  θ=(−2, 0.42, 1.02); **ordinal sign flip**: the paper ADDS the ordinal shift, the
  flow SUBTRACTS → fitted weights −0.2 / +0.3; the C.4 odds-ratio check gives
  OR ≈ e² ≈ 7.4. `vaca`: E[x3|do(x2=a)] = −0.25 + 0.25a (do(x2=−3) is off-manifold
  extrapolation — looser tolerance). `carefl`: counterfactuals are analytic
  (`Carefl4.true_counterfactual`); the paper's x_obs has a ~4σ abducted noise, so
  typical held-out rows are scored instead of that single point.
- **`validate_ls`** (`experiments/data/magic-mrclean/ls`, seed 7, n=1275, full data,
  `restore_best=False`): flow = statsmodels = R polr at Age 0.0526, NIHSSa 0.1630,
  T −0.9424; ATE +0.1429 vs +0.1428, true ATE +0.132. The R reference
  (`fit_ls.R`, needs `tram`/`MASS`) has its outputs committed under `ref_ls/`, so
  nothing needs R installed.
- Committed expectations live in `experiments/ground_truth/<name>.json`, one
  `{value, atol}` entry per metric; `check.py` enforces them.

## Testing policy

- Framework tests must not depend on `experiments/`. A new causal feature is
  validated against an inline DGP's known truth (add one to `conftest.py` if
  none fits), never with "runs without error".
- Frozen CSVs in `experiments/data/` are a contract — **never regenerate
  silently**; a new seed or new equations means a **new folder**. `check_data.py`
  regenerates each from the seed in its `truth.json` and compares at **1e-9**, not
  bit equality: numpy's transcendental functions move their last bits between
  releases (measured ~1e-15 after the 2026-08 dependency bump).
- `experiments/data/magic-mrclean/ls` has no generator here (it left with the
  stroke storyline); it is frozen input data. Recover the generator from
  `pre-experiments-cut` if it ever needs regenerating.
- Fit checks for the paper DGPs train on the paper protocol (n=40k, 500 epochs),
  not the frozen n=5k CSVs — β13 multiplies the low-variance x1 ∈ [0.25, 0.73] and
  is too weakly identified at n=5k.

## Roadmap notes

- ~~Generalize the generators beyond the stroke DAG~~ — done for the TRAM-DAG
  paper's DGPs (triangle/triangle-mixed/vaca/carefl, June 2026). Still open:
  hidden confounding à la DeCaFlow.
- ~~Package for PyPI~~ — published as `tramdag` (latest 0.3.0, June 2026);
  release flow:
  bump `version` in pyproject (`__init__` now reads it back from the installed
  metadata, so there is only one place to edit), `uv build`, `uv publish`
  (Oliver's PyPI token), CHANGELOG section.
- The `experiments/` tree is the candidate for a companion repository
  (`tensorchiefs/tramdag-simu`); it is already self-contained, so the move is a
  directory copy plus a workflow.
