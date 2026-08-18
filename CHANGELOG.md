# Changelog

## 0.4.0 (unreleased)

### Added

- **Transformation syntax**: a node's additive formula is now its first
  positional argument and can be written as a `+` sum — the formula reads
  like the math (`ContinuousNode(I("x1") + CS("x2"))`,
  `OrdinalNode(4, [I, LS("x1")])`). New on `I`:
  `allow_interaction=False` (multi-parent intercept becomes additive,
  `I("a","b", allow_interaction=False) == I("a") + I("b")`) and
  `transform=`/`transform_kwargs=` (the monotone basis moves onto the
  intercept term, e.g. `I("x1", transform="spline")`). Everything
  normalizes to the same internal term list: checkpoints, `flow.py` and
  the serialized format are unchanged, and equivalence is pinned by
  state-dict-identical tests (`tests/test_transformation_syntax.py`).

- **Transparent training internals**: the previously hardcoded training
  knobs are optional kwargs with unchanged defaults —
  `fit(plateau_factor=0.3)` (per-node plateau decay multiplier) and
  `fit(vc_oof_fit={...})` (settings of the stage-1 out-of-fold proxy fits
  behind `VC(center=True)`, default
  `{"epochs": 300, "learning_rate": 1e-2, "batch_size": 512}`), plus
  `fit_classical(chunk=25, history_size=50)` (L-BFGS round length and
  memory). The plateau lr floor (`1e-3 * learning_rate`) and the freeze
  guard (`1e-2 * learning_rate`) are documented in the `fit` docstring.

- **`intercept()`** is the pythonic name of the intercept-term
  constructor; `I` stays as the exported alias (the notation of the docs
  and the paper).

- **`units=` on `I`, `CS` and `VC`** sizes the term's network directly,
  e.g. `units=[16]` for one hidden layer (defaults: I `[8, 8]`,
  CS `[64, 128, 64]`, VC `[16]`); serialized per term. All conditioner
  networks now build through one `_mlp()` helper whose module indices
  match the historical Sequentials, so 0.3 checkpoints still load.

### Changed (breaking)

- **`VC(*modifiers, t=...)`**: the positional arguments are the
  covariates that enter `b_theta`; the treatment `t` is a required
  keyword. `VC("X2", "X3", t="T")` reads as
  `(beta0 + b_theta(x2, x3)) * x_t`. (0.3 wrote `VC("T", "X2", "X3")`.)

### Removed (breaking)

- The `terms=` keyword (use the first positional argument), node-level
  `ContinuousNode(transform=/transform_kwargs=)` (choose the basis on the
  intercept term, `I(..., transform="spline")`), the unused
  `Intercept`/`LinShift`/`CShift` aliases, and the pre-0.3
  `parents={...}` checkpoint loader. 0.3 checkpoints (`terms` layout)
  still load; their node-level basis is converted on load.
- The unmaintained notebooks and experiment scripts moved to
  `notebooks/stale/` and `experiments/stale/`; the maintained set is
  the intro and Colab demo notebooks plus `sim_flow.py` and
  `validate_ls.py`.

### Added (staged earlier as an unreleased 0.3.1)

- **Propensity-centered VC: `VC(..., center=True, center_folds=5)`** (issue
  #30): the R-learner orthogonalization `beta(x)·(t − ê(x))` inside the
  likelihood, as a **two-stage frozen** design — training uses **out-of-fold**
  ê (K refits of the treatment node only; DML cross-fitting, bookkeeping in
  `flow.vc_center_info`, pinned by tests so an in-sample "simplification"
  fails CI), frozen as data (zero gradient into the treatment node from the
  outcome loss — tested); inference recomputes ê from the flow's own fitted
  treatment node and re-derives `t − ê(x)` under `do` (never cached — tested
  on fresh rows). `center="colname"` supplies user cross-fitted propensities;
  binary ordinal treatments only; `center=False` (default) is bit-identical
  to the uncentered term (tested). Measured (the Dandl et al. 2024
  reproduction, `tests/test_vc_centered.py`): under strong confounding + an
  under-specified prognostic part, centering cuts β̂ bias **5–10×**
  (1.10–1.24 → 0.11–0.27 over 3 seeds). Docs:
  `docs/varying-coefficients.md`.

- **`flow.scores(df, node)` + `flow.effect_modifier_scan(df, node, t)`**
  (issue #29): per-observation scores ψᵢ = ∂ℓᵢ/∂θ for every `LS` weight and
  `VC` `beta0` — **analytic and exact** (shifts enter the latent additively, so
  ∂ℓᵢ/∂β = (∂ℓᵢ/∂sᵢ)·xᵢ with the latent derivative in closed form; pinned by a
  float64 finite-difference test and the score-sums≈0-at-MLE property) — and
  the Zeileis–Hornik fluctuation scan packaged on top: order the treatment
  coefficient's scores by each candidate covariate, `sup|CUSUM|` vs the
  Kolmogorov 5% critical value ranks candidates for `VC` modifiers from a
  seconds-long all-`ls` `fit_classical`. Pure read-out, no fitting/sampling
  path touched. End-to-end test: the scan flags the true (X2, X3) modifiers of
  a heterogeneous-effect SCM and not the inert prognostic X1. Docs:
  `docs/scores.md`.

- **`VC(on, *modifiers, penalty=)` — varying-coefficient shift term** (issue #28):
  a treatment-effect head `beta(x) = beta0 + b_theta(x)` with a small (16-unit),
  **penalized**, zero-initialised network that only multiplies `x_on`, plus the
  first-class read-out `flow.varying_coef(node, data)` (closed-form,
  deterministic, y-free; equals the abduct-difference for binary treatments).
  The objective is the penalized likelihood `Σ NLL + penalty·‖w‖²` (total-NLL
  scale, `beta0` unpenalized); `penalty → ∞` — or `modifiers=()` exactly —
  nests `LS(on)`, and `fit(vc_warm_start=True)` (default) starts `beta0` at the
  classical all-`ls` solution. VC modifiers may also appear in prognostic terms
  (only `on` owns its edge). Motivation, measured: the `CS(on, x…)` reduced form
  is *expressive but unestimated* — corr ≈ 0.5 against the true effect function
  even in-class (tramdag-simu#18/PR #21) because nothing in the NLL rewards a
  smooth arm-difference; the regularized head reaches ≈ 0.99 on the same task
  class. New validation DGP `simulations/vc_shift.py` (`VCLogisticShift`,
  frozen `data/vc-shift/`, registry `"vc-shift"`) with known
  `beta_true = −1 + 0.8·X2 − 0.6·X3`; acceptance tests in `tests/test_vc_term.py`
  (recovery bar corr ≥ 0.9 at n = 5000, measured min-over-seeds 0.986). Docs:
  `docs/varying-coefficients.md`.

- **`flow.intercept_contributions(node, data)`** (issue #20, Option A) — post-hoc,
  mean-centered decomposition of an **additive complex intercept**
  (`terms=[I("x1"), I("x2")]`). The per-term networks are summed in unconstrained
  parameter space, so the sum is identified but each term's contribution only up to
  a constant; this returns each term's **sum-to-zero** (GAM-style mean-centered over
  `data`) contribution to the transform parameters plus the absorbed `baseline`, for
  plotting per-parent partial effects. Exact (`baseline + Σ contributions == theta`)
  and purely interpretive — it reads the fitted weights and changes nothing about the
  model or any frozen number. Shift terms remain a separate slot (`ls_coefficients`).

### Changed

- The **observational ITE study** (the `ITEObservational` DGP, its didactic
  notebook, and the train-size experiment) has **moved out of tramdag** to the
  simulation-study companion repo `tensorchiefs/tramdag-simu`. The package keeps
  only DGPs that validate the implementation; a benchmark comparing tramdag to
  other methods belongs in the (method-neutral) paper repo.

## 0.3.0 (2026-06-19)

### Removed (breaking)

- **The legacy `parents={parent: "ls"|"cs"|"ci"}` constructor argument** is gone.
  Use the term-formula notation `terms=[I(...), LS(...), CS(...)]` (see below);
  `tramdag.term(effect, *parents)` helps when the effect is data-driven. Old
  *checkpoints* saved with the dict layout still load.

### Added

- **Term-formula spec notation** — declare a node's transformation as an additive
  list of terms, `terms=[I(...), LS(...), CS(...)]`, replacing the per-edge
  `parents={parent: "ls"|"cs"|"ci"}` dict (now **deprecated**, still accepted with
  a `DeprecationWarning`). Each term names the parent(s) it depends on; **joint
  (multi-parent) terms** express interactions — `CS("x1", "x2")` is one shift
  network over both parents and `I("x1", "x2")` one joint intercept — while
  separate terms stay additive: `CS("x1") + CS("x2")` are two additive shifts, and
  `I("x1") + I("x2")` is an **additive complex intercept** (each parent reshapes the
  transform independently, the per-term coefficient vectors summed in unconstrained
  space). The grouping *is* the joint/additive choice. A `term(effect, *parents)`
  factory helps data-driven specs, and `flow.to_matrix()` renders the paper's
  meta-adjacency view.

- **API papercuts (issue #12):** `CausalFlowDAG(spec, seed=...)` seeds weight
  initialization deterministically (one obvious reproducibility knob — `fit(seed=)`
  only seeds shuffling); `save`/`load` now also carry a provenance `meta` block
  (tramdag version, save time, device, and a machine/environment snapshot) and
  `flow.meta` is repopulated on load, so cached models are self-describing;
  `tramdag.machine_info()` exposes that snapshot (host, OS, CPU/GPU, cores, RAM,
  python/torch/zuko/tramdag versions); a dev-install one-liner
  (`pip install "git+https://github.com/tensorchiefs/tramdag.git@main"`) is
  documented in the README and the Colab demo. (Training `history` already
  round-tripped through `save`/`load`; now covered by a regression test.)

- **`fit(marginal_init=True)`** — opt-in calibrated initialization for *unconditional*
  (`SimpleIntercept`) nodes, replacing zuko's default zero init. Bernstein roots
  start at the linear map of the pre-scaled domain onto the standard-logistic
  5/95 quantiles (the default is ~2.5× too steep); ordinal roots start at the
  empirical class log-odds (default zeros ≈ uniform). A **pure init** — the
  converged MLE is unchanged (the exact-`ls` MLE / R-`polr` equivalence is
  preserved), applied once on the first fit, conditional `ci` intercepts untouched.
  Large time-to-target win where a root's marginal shape dominates the NLL gap
  (vaca-ci ~2.5× faster to target over 6 seeds); small where convergence is
  coefficient-bound. Defaults unchanged (off). See `docs/research/REPORT.md`.

- **`CausalFlowDAG.fit_classical()`** — deterministic, full-batch, **float64**
  L-BFGS for all-`ls` models (each node-conditional is then a classical
  transformation model). Bit-reproducible, reaches the exact MLE, matches
  `statsmodels` ordered-logit / R `polr`/`Colr` to ~1e-3 on well-identified
  coefficients; raises on `cs`/`ci` specs (use `fit()`). Plus `ls_coefficients()`
  to read the per-node shift weights. float64 is a transient compute mode
  (`self.double()/.float()`), so the stored model stays float32; as a side effect
  the data path (`_tensorize`/`sample`/`pmf`) is now dtype-agnostic.
- `notebooks/classical_fit_tram_dag.py` (didactic) and a `--classical` flag for
  `experiments/validate_ls.py`.
- **Next:** standard-error table from the float64 Hessian at the MLE (the float64
  bracket here is the groundwork); needs a reference-level constraint for the
  one-hot ordinal-parent flat directions.

## 0.2.0 (2026-06-12)

First PyPI release: `pip install tramdag`.

### Changed (naming & packaging)

- **Renamed**: Python package `zuko_dag` → **`tramdag`** (conventional alias
  `import tramdag as td`); GitHub repo `tram-dag-zuko` → `tensorchiefs/tramdag`
  (old URLs redirect). The package implements TRAM-DAGs; zuko names the backend.
  No API changes; old checkpoints still load. References to the original
  Keras/TF implementation (tensorchiefs/tram-dag) reworded to avoid
  self-reference.
- **MIT license** added; PyPI metadata (authors, urls, classifiers); runtime
  dependencies trimmed to `torch`, `zuko`, `numpy`, `pandas` (pytest/scipy/
  statsmodels/scikit-learn/matplotlib moved to the `dev` dependency group).
- **README rewritten method-first**: the repo is the reference implementation of
  the CLeaR 2025 paper (arXiv:2503.16206); the stroke analysis is the case study
  (arXiv:2606.12623) with its detail moved to `docs/stroke-case-study.md`.
  Citation BibTeX added for both papers.

### Added

- **`fit(schedule=..., freeze_patience=...)`** — learning-rate schedules and
  per-node early stopping (defaults unchanged). The optimizer now holds one
  param group per node; `schedule="plateau"` decays each node's lr off its own
  validation NLL, and `freeze_patience` drops converged nodes from the loss
  (real FLOP savings — per-node gradients are independent) with early exit when
  all nodes froze. Also `"onecycle"`/`"cosine"`. Benchmarks + recommendation in
  `docs/training-speed.md` (`experiments/bench_training.py`): plateau+freeze
  matches the hand-tuned two-phase recipe's time-to-accuracy with **no budget
  tuning and ~3× less total compute**; full-batch LBFGS solves the classical
  all-`ls` MLE in <2 s (2/3 seeds). Existing defaults intentionally untouched.
- **Colab demo** `notebooks/demo_tram_dag_colab.py` (+ tracked output-stripped
  `.ipynb` for the badge): the paper's bimodal VACA benchmark fitted live
  (cuda/cpu auto-detect), L1 pairs plot, analytic do-checks, per-individual
  counterfactuals vs DGP truth, GPU-vs-CPU race.

- **The TRAM-DAG paper's DGPs** (Sick & Dürr, CLeaR 2025, arXiv:2503.16206) as
  simulation registry families, each a numpy-only SCM with known/analytic ground
  truth + frozen n=5000 CSVs (`data/<name>/`, the test contract) and CLIs:
  - `simulations/triangle.py` — `TriangleContinuous` (§6.1: logistic-latent TRAM
    DGP, h₂=5x₂+2x₁, h₃=0.63x₃−0.2x₁−f(x₂)) and `TriangleMixed` (§6.2: ordinal x₃,
    θ=(−2, 0.42, 1.02)); f variants `linear`/`cubic`/`exp`/`atan`/`sin`; supports
    array-valued `do` (C.4 soft interventions).
  - `simulations/vaca.py` — `VacaTriangle` (App. C.1 bimodal Gaussian L1/L2
    benchmark vs CNF).
  - `simulations/carefl.py` — `Carefl4` (App. C.2 Laplace SCM; **analytic**
    counterfactuals via `abduct_noise`/`true_counterfactual`).
- `experiments/paper_{triangle,triangle_mixed,vaca,carefl}.py` (+ `paper_common.py`)
  — replicate the paper's figures: coefficient trajectories (Fig. 14/15/19), CS-curve
  recovery (Fig. 7), L1/L2 distribution overlays (Fig. 4/5/9/16/20), counterfactual
  curves at the paper's x_obs (Fig. 6), and the C.4 odds-ratio check (OR ≈ 7.4).
- `tests/test_paper_dgps.py` — generator pinning (KS TRAM-identities, frozen-CSV
  contract, analytic ground truth) + flow recovery (coefficients with the ordinal
  sign-flip, CS curve, VACA do-moments, CAREFL counterfactual MAE).

### Changed (behavior)

- **`CausalFlowDAG.fit(..., restore_best=False)` is now the default.** Training keeps
  the **final converged weights** instead of restoring per-node best-validation
  weights. Rationale:
  - *Least surprise* — `fit()` returns the model you trained, not a silently
    swapped earlier epoch.
  - *Exact classical comparison* — an all-`ls` model trained to convergence is now
    exactly the maximum-likelihood (proportional-odds) estimate, matching
    `statsmodels` `OrderedModel` and R `MASS::polr` to ~1e-3 (see
    `experiments/validate_ls.py`, `tests/test_simulations.py::test_all_ls_flow_is_exact_mle`).
    This was **not achievable before**: best-validation restoration pinned the fit
    off the training optimum.
  - Early stopping is now an explicit, opt-in regularization choice.

  To restore the previous behavior, pass `restore_best=True`.

  **Note for flexible (`ci`/`cs`) models:** their MLE *overfits the observational
  confounding*, so they need `restore_best=True` to recover the causal effect (lower
  validation NLL confirms it generalizes better). `experiments/run_experiment`
  therefore defaults `restore_best` per style — off for all-`ls`, on for flexible.

### Added

- `src/tramdag/simulations/magic_mrclean.py` — synthetic stroke cohort (SCM with
  known ground truth); `ls`/`nl` variants; CLI to (re)generate frozen CSVs.
- `data/magic-mrclean/` — frozen public CSVs + `fit_ls.R` classical R reference and
  committed `ref_ls/` outputs. The public, reproducible substitute for the private
  clinical data.
- `experiments/common.py::load_data(source)` — switch between `"magic"` (private) and
  `"magic-mrclean/{ls,nl}"` (synthetic, default).
- `experiments/sim_flow.py` — known-truth recovery storyline; `validate_ls.py`
  rewritten as a spot-on flow-vs-MLE-vs-R comparison.
- `tests/test_simulations.py` — generator, known-truth recovery, the all-`ls`
  spot-on MLE check, and the Python-vs-R regression.
