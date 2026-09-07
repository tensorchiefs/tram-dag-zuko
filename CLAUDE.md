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
has **no defaults in code**; `experiments/common.py::load_variant` parses it.
The blueprint (2026-09, experiments/README.md): each variant carries the FULL
model as `spec:` (`tramdag.spec_from_dict` form — transform, n_coeffs, range_q,
units, activation, input_transform are term options there, not config keys),
`flow_kwargs:` (CausalFlowDAG construction) and `fit_kwargs:` (flow.fit
verbatim); learning_rate/schedule stay top-level (they build the optimizer).
Deliberately verbose — duplication over indirection. bench_training.yaml is
workloads-shaped, same rule; perf_machine.py is exempt (curl-and-run single
file). `experiments/` is split into `paper/`, `benchmarks/` and
`misc/`, with `experiments/tests/` for the shared `check.py`. `paper` and `misc`
each own their `data/`, `ground_truth/`, `tests/` and `results/`; `benchmarks/` measures speed on the other two's data and pins
no ground truth, writing up its numbers in `docs/` instead. Only `common.py`
(output layout) and `check.py` (ground-truth comparison) are shared. The area tests run in the ordinary `uv run pytest`.
See `experiments/README.md`.

## Architecture (src/tramdag/)

Since the 1.0-RC refactor (branch rc/1.0-architecture, docs/adr/001):
term-specific behavior lives on the effect's registry entry in `terms.py`
(validation, build, shift_value/theta_value, post_init, regularizer,
finalize, score_columns, side inputs, cells, term_is_classical,
option_defaults — built-ins subclass their conditioners so checkpoints and
the seeded RNG stream stay bit-stable); node-kind branches live ONLY in
nodes.py's four kind_* functions; `fitting.py`/`readouts.py` are mixins
CausalFlowDAG composes (methods defined once, no delegate layer); new public:
`flow.shift_curve`, `fn_shift`/`Fn`, `register_term`, `ordinal_bounds`,
spec exports (`spec_to_dict`/`spec_from_dict`/`validate_and_sort`/
`node_parents`), `effect_modifier_scan(column=)`. Custom effects:
subclass `tramdag.terms.ShiftTerm`, register under a new effect name.
docs/architecture.md carries the module map and the term-contract diagram.

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
  object. `transform=` on an intercept picks the monotone transform class and **extra
  keyword arguments pass straight to the transform class**
  (`SI(transform="spline", bins=16)`); `units=[...]` on CI/CS/VC sizes the
  term's network. A node takes at most ONE intercept term with parents; a
  **multi-parent** `CI("a","b")` is one *joint* network (interaction on the
  thetas, the default), and `CI("a","b", allow_interaction=False)` is the
  *additive* intercept (one net per parent, coefficient vectors summed). For
  shifts, grouping decides: `CS("a","b")` is joint, `CS("a")+CS("b")` additive.
  When the effect type comes from config or the CLI, put the constructor itself
  in the table (`{"x2": CS}`). A formula without an intercept gets `SI()` prepended during normalization, so `node.terms[0]` is always the intercept.
  Every parent enters through exactly one edge-owning term (VC modifiers exempt —
  they may also appear prognostically).
- `transforms.py` — monotone 1-D transforms wrapping zuko (`BernsteinUT`, `SplineUT`,
  `AffineUT`; pre-scaled from the train `range_q`/1−`range_q` quantiles to
  [-5,5] — an intercept option, default 0.05, `SI(range_q=0.0)` = the
  reference comparisons' min-max `scale_df` domain — zuko's own
  inverse with its closed-form tail) + the ordinal ordered-logit transform
  (`P(Y<=k) = sigmoid(theta_k - shift)`, cutpoints `[t0, t0+cumsum(exp(...))]`).
- `conditioners.py` — the LS/CS/intercept networks. Default widths and `relu`
  replicate the PyTorch reference (`buehlpa/TramDag`, `tram_models.py`:
  `ComplexShiftDefaultTabular` 64-128-64, `ComplexInterceptDefaultTabular` 8-8,
  `n_thetas=20`) — **not** the paper's R nets, which every `experiments/paper/`
  config sets explicitly instead.
- `callbacks.py` — the shipped `fit` callbacks on the `Callback` base
  (`on_fit_begin`/`on_epoch_end`/`on_fit_end`, state reset at fit begin):
  `EarlyStopping` (best-validation weights restored automatically at fit
  end; optional stopping `patience`) and
  `PerNodePlateau` + `per_node_adam` (per-node lr decay and freezing, the
  pre-0.4 plateau recipe; `frozen = {node: epoch}` afterwards). All read
  `history["val"]`, which `fit(validation_data=|validation_split=)` fills
  per epoch; `fit` also records the optimizer's rate per epoch in
  `history["lr"]`; `verbose=` owns progress printing. Optional; `fit` itself
  stays one plain loop.
- `plots.py` — `plot_dag(spec|flow)` (the labelled DAG, layered, one edge
  style per effect), `plot_marginals`, `plot_training(frozen=)`. matplotlib is
  the optional extra `tramdag[plots]`, imported on first call — the package
  import never needs it. `plot_dag` is exported at top level.
- (no `utils.py` any more: `config_section` moved to
  `experiments/common.py`, `machine_info` to
  `experiments/benchmarks/perf_machine.py` — each next to its only caller, so
  the package is modelling code only.)
- `flow.py` — `CausalFlowDAG`: `fit`, `fit_classical` (float64 full-batch
  L-BFGS, exact MLE for all-`ls` specs), `sample(n, do=, u=)`, `abduct`, `pmf`,
  `density` (its continuous counterpart, on a grid),
  `log_prob`, `save/load`, `ls_coefficients` (LS weights only — network shifts
  are skipped), `varying_coef` (VC read-out), `scores` /
  `effect_modifier_scan` (analytic per-row ∂ℓᵢ/∂θ + CUSUM modifier scan,
  `scores.py`). NLL decomposes per node → one Adam fits all nodes jointly.

## Conventions that matter (easy to get wrong)

- **Latent scale**: continuous `z = h(x) + shift` (shifts ADDED); ordinal
  `P(Y<=k) = sigmoid(theta_k − shift)` (shift SUBTRACTED). Both follow the original TRAM-DAG
  conventions; tests pin them.
- **Parent encoding**: continuous parents enter RAW (no standardization) unless
  a term-level `input_transform=` ("minmax", "standardize", or a callable
  `fn(x, train)` over frozen train columns), which feeds that term's *network*
  (CI/CS/VC modifiers) the transformed parent (minmax like the reference's
  `scale_df`, standardize, or the callable over frozen train columns) — LS and the VC treatment stay raw either way; ordinal
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
  boundary derivative). Demonstrated in `notebooks/demo_tram_dag_colab.py` section 6.
- **`fit` keeps the final weights and is one minibatch Adam loop** — an
  all-`ls` model then matches statsmodels/R-polr to ~1e-3. Validation, lr
  schedules, early stopping / best-weight restoration and logging are the
  caller's, through `fit(validation_data=|validation_split=,
  validation_batch_size=, verbose=, optimizer=, callbacks=)` — fit fills
  `history["val"]` per epoch and `verbose=N` prints every Nth line;
  `callbacks=` takes `Callback` instances
  (`on_fit_begin`/`on_epoch_end`/`on_fit_end`; epoch hooks get
  `(flow, epoch, opt)`, any `True` stops, `on_fit_end` runs before the VC
  re-centering) or bare `on_epoch_end` callables; `tramdag/callbacks.py`
  ships `EarlyStopping` (auto-restores best weights; optional patience),
  `PerNodePlateau`+`per_node_adam` (they read `history["val"]`); `flow.calibrate(train_df)`
  takes the data-dependent state once (each term calibrates itself: ranges,
  input-transform stats — never the weights) and is called by the first fit;
  `init_marginals(train_df)` applies the calibrated start — always an
  explicit call, nothing runs it for you. Key empirical finding (stroke storyline):
  **flexible (CI/CS) models overfit observational confounding at the MLE and
  need best-validation weights to recover the causal effect; all-`ls` models
  don't** — `callbacks.EarlyStopping` now, see docs/fitting.md.

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

Experiments (`experiments/`, seed 42 unless stated, arXiv:2503.16206). The
paper states only three training numbers — n=40000, 500 epochs, Bernstein
order 20; the Adam lr 1e-3 is the R code's `optimizer_adam()` default. The configs follow the paper's own R code 1:1 where the
framework allows: the triangle scripts train one continuous Adam run with a
separate validation draw (40k / 10k mixed) and read the coefficients after
every epoch (`fit(callbacks=)`) — at batch 256 / lr 0.004 for 300
epochs (linear-cs 500, mixed exp-cs 350, mixed linear-ls 200 @ lr 0.002)
instead of the paper's 500 at Keras-default batch 32 / lr 0.001, the
deviations taken for CI runtime (every metric kept; grid, epoch floors and
the 2026-09-01 tuning round in docs/paper-replication.md); the
VACA/CAREFL comparisons take one full-batch step per epoch on nTrain = 2500 —
VACA 10000 epochs at lr 0.001 with the reference's ReduceLROnPlateau
(factor 0.1, patience 50, min_lr 1e-7; torch's scheduler on the summed
validation NLL, global as in `update_learning_rate`; restored 1:1
2026-09-02), CAREFL the reference run 1:1 since 2026-09-03 — trained on
CAREFL's own committed 2500 rows (frozen under
`experiments/paper/data/carefl-cf` with xObs and the truth/pred curves,
sd-standardized units) with `val = train`, 7000 @ 0.001, the same plateau
rule, and `range_q: 0` (the reference's min-max Bernstein domain), which
puts the Fig. 6 curves on the paper's (fig6 x4 max 0.204 vs CAREFL's own
0.174; the earlier 3000@0.002-vs-7000 trade-off was an artifact of the
fresh-draw data; minibatch and raw-parent alternatives measurably fail,
see docs/paper-replication.md).
Seeds: the triangle scripts run
unseeded, the comparison scripts seed R's RNG with 42 (not replayable in
torch), so every seed here is a repo choice. Init follows each reference:
`init: normal` (Keras `random_normal`, the triangle scripts' `LinearMasked`
layers) and `init: glorot` (Keras `Dense`, `make_model`) — under the
full-batch protocol the init decides the fit (VACA do(x2) errors
0.52/0.33/0.13 with torch's default init, 0.098/0.159/0.026 with glorot at
the config's seed — both measured on the earlier −3/−2/0 grid; on the shipped
−3/−1/0 grid glorot scored 0.097/0.088/0.019 under the old 10000-epoch
plateau protocol and 0.096/0.086/0.018 under the restored reference
protocol). Known, documented deviations: the triangle scripts also
use 5%/95% quantiles for the Bernstein domain (a match; the comparison
scripts use min-max, `scale_df` — matched for CAREFL via `range_q: 0`,
kept at the quantiles for VACA where min-max measures worse,
0.289/0.040/0.067 vs 0.096/0.080/0.022); both comparisons scale the
*network inputs* min-max (`input_transform: minmax` on the CI terms; raw parents saturate
the tanh nets: `do(x2=-3)` error 0.731 → 0.098, and the 2026-09-01 relu/sigmoid
raw-parent attempts fail too); a bias-free intercept
output layer; Adam eps 1e-8 vs Keras 1e-7 (no effect);
no marginal init anywhere in the comparisons (`validate_ls` calls
`init_marginals` explicitly — the framework never does).

**Each config takes its architecture from *its own* reference script**, and the
reference uses two different ones. The triangle experiments
(`summerof24/triangle_structured_*.R`): `hidden_features_I = hidden_features_CS`
= `c(2,25,25,2)` continuous / `c(2,2,2,2)` mixed, **sigmoid** (the ReLU line is
commented out), `len_theta = 20`. CORRECTED 2026-09-02: the c(...) vector
reads as in/out dims around the hidden stack — hidden (25,25) continuous,
(2,2) mixed. The earlier literal [2,25,25,2] reading put a 2-sigmoid
bottleneck on the input: sin could not reproduce paper Fig. 18 at any
protocol (curve err 1.22), (25,25) lands on the figure (0.24); linear-cs
0.13 -> 0.025, atan edge flattening gone. The VACA/CAREFL comparisons
(`comparison/utils.R::make_model`): one net per node, `dense(10, tanh) ->
dense(100, tanh) -> dense(len_theta)`, `M = 30`. Applying the triangle net to
CAREFL — which an earlier revision did — cost an order of magnitude on the
counterfactual MAE (x4, measured on the old 18k-row protocol, under the
misread bottleneck net and M = 20).
Note also that `n_coeffs` counts *unconstrained* coefficients: zuko ties two
extra control points on, so `n_coeffs=20` is order 21 where the reference's
`len_theta=20` is order 19. The free-parameter count is what matches.

- **Paper DGPs**: `triangle` true coefficients β12=+2, β13=−0.2 (+0.3 on x2 for
  `linear`); a fitted `cs` learns −f(x2)+const. `triangle-mixed` cutpoints
  θ=(−2, 0.42, 1.02) from `triangle_structured_mixed.R`; the paper's text does
  not state them;
  **ordinal sign flip**: the paper ADDS the ordinal shift, the
  flow SUBTRACTS → fitted weights −0.2 / +0.3; the C.4 odds-ratio check gives
  OR ≈ e² ≈ 7.4. `vaca`: E[x3|do(x2=a)] = −0.25 + 0.25a (do(x2=−3) is off-manifold
  extrapolation — looser tolerance). `carefl`: trains on CAREFL's own
  committed rows (`data/carefl-cf`: X.csv, xObs, the analytic truth curves
  and CAREFL's own predictions, everything in CAREFL's sd-standardized
  units, x3/x4 divided by 6.0104/1.9114 — external frozen input, no
  generator here) and scores the Fig. 6 curves point by point against the
  committed truth; held-out rows (fresh `Carefl4` draws scaled by the
  committed sds) are scored next to the single xObs because one point is
  a noisy yardstick.
- **`validate_ls`** (`experiments/misc/data/magic-mrclean/ls`, seed 7, n=1275, full data,
  final weights): flow = statsmodels = R polr at Age 0.0526, NIHSSa 0.1630,
  T −0.9424; ATE +0.1428 vs +0.1428, true ATE +0.132. The R reference
  (`fit_ls.R`, needs `tram`/`MASS`) has its outputs committed under `ref_ls/`, so
  nothing needs R installed.
- Committed expectations live in `experiments/<area>/ground_truth/<name>.json`;
  `check.py` enforces them. Two entry forms: `{value, atol}` (two-sided) and
  `{max}` (an upper bound, for error measures, so a better fit cannot fail).
  A `{max}` bound belongs in a band — 1.5x to 4x its measurement — and
  `check.py` notes one that is not, unless the entry carries a `"why"` saying
  why it is deliberately wide. Centers are re-pinned whenever the code moves
  them: a stale center is how a variant ends up passing while describing a
  model that no longer runs.

## Testing policy

- Framework tests must not depend on `experiments/`. A new causal feature is
  validated against an inline DGP's known truth (add one to `conftest.py` if
  none fits), never with "runs without error".
- Frozen CSVs in `experiments/<area>/data/` are a contract — **never regenerate
  silently**; a new seed or new equations means a **new folder**. `check_data.py`
  regenerates each from the seed in its `truth.json` and compares at **1e-9**, not
  bit equality: numpy's transcendental functions move their last bits between
  releases (measured ~1e-15 after the 2026-08 dependency bump).
- `experiments/misc/data/magic-mrclean/ls` has no generator here (it left with the
  stroke storyline); it is frozen input data. Recover the generator from
  `pre-experiments-cut` if it ever needs regenerating.
- Fit checks for the paper DGPs train on the paper protocol (n=40k; epochs per
  the tuned configs, 200-500),
  not the frozen n=5k CSVs — β13 multiplies x1, whose two mixture components sit at 0.25 and 0.73
  (sd 0.254, against 0.375 for x2 and 2.918 for x3), so it
  is too weakly identified at n=5k.

## Roadmap notes

- Upstream PRs to zuko: five ranked candidates (analytic Bernstein
  `call_and_ladj`, linear spline tails, public `_constrain_theta` inverse,
  θ-shape docstring fix, `Logistic` distribution) in docs/zuko-upstream.md.
- ~~Generalize the generators beyond the stroke DAG~~ — done for the TRAM-DAG
  paper's DGPs (triangle/triangle-mixed/vaca/carefl, June 2026). Still open:
  hidden confounding à la DeCaFlow.
- ~~Package for PyPI~~ — published as `tramdag` (latest 0.3.0, June 2026);
  release flow since the 1.0-RC: tag-driven (skeleton convention) — the
  version IS the git tag (hatch-vcs; `cz bump` derives it from the
  conventional commits, or tag `vX.Y.Z` by hand), pushing the tag runs
  `.github/workflows/release.yaml` (uv build → PyPI via trusted publishing →
  sigstore-signed GitHub release). One-time prerequisite: register the
  GitHub repo as a trusted publisher on pypi.org/manage/project/tramdag
  (needs the PyPI project owner — Oliver) and create the `pypi` environment
  in the repo settings. CHANGELOG section stays hand-written.
- The `experiments/` tree is the candidate for a companion repository
  (`tensorchiefs/tramdag-simu`); it is already self-contained, so the move is a
  directory copy plus a workflow.
