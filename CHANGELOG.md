# Changelog

## 1.0.0-rc (unreleased, branch rc/1.0-architecture)

### Changed — tag-driven releases (the skeleton convention)

The version is now the git tag: hatch-vcs derives it (`no-local-version`),
`pyproject` carries no version field any more, and pushing a `v*` tag runs
the new release workflow — build with uv, publish to PyPI via trusted
publishing (no token secret), sigstore-sign and create the GitHub release.
`cz bump` computes the next tag from the conventional commits. One-time
setup outside the repo: the PyPI trusted-publisher registration and the
`pypi` GitHub environment.

### Changed — term-owned architecture (docs/adr/001, docs/architecture.md)

The 0.4 monolith split along its seams and every per-effect behavior moved
onto one registry entry per effect in the new `terms.py` — no effect
string-switch survives outside it. Public behavior, error messages, sign
conventions, seeding and all ten experiment ground truths are unchanged
(verified per step by a seeded state-dict smoke and replications); the fit
API, queries and read-outs keep their exact signatures as flow methods.

- **`fit(vc_ehat=)` is gone**: a centered VC names its out-of-fold
  propensity COLUMN — `VC("X", t="T", center="ps")` reads column ``ps`` of
  the training frame (``df.assign(ps=e_oof)``), which splits, shuffles and
  minibatches like any data; queries still recompute the propensity live
  from the treatment node. ``center=True`` refuses with the new spelling
  named. No term-specific argument crosses ``fit`` any more.
- `fitting.py` and `readouts.py` are mixins `CausalFlowDAG` composes —
  every method defined once, no delegate layer; `shift_curve` is a flow
  METHOD (`flow.shift_curve(node, parent, grid)`), and it evaluates
  through the term's own `shift_value`, so `Fn` and custom terms plot
  too. The internal legacy views died with the compat pass: `_VCGroup`,
  `node.vc_column` (now `VCTerm.regressor`), `node._vc_groups`,
  `node._shift_groups`, and the `ci_parents`/`_intercept_groups`
  pass-through properties (read `node.intercept.ci_parents`/`.groups`).
- New modules: `nodes.py` (the node model + the ONLY four continuous/ordinal
  branches: `kind_log_prob`/`kind_sample`/`kind_abduct`/`kind_marginal_theta`),
  `fitting.py` (`_FitMixin`: fit/fit_classical), `readouts.py`
  (`_ReadoutsMixin`: the stateless read-outs), `terms.py` (the registry and the term classes:
  LSTerm/CSTerm/VCTerm/FnTerm, SITerm/CITerm/AdditiveCITerm on
  ShiftTerm/InterceptTerm hooks).
- New public API: `flow.shift_curve(node, parent, grid)` (replaces reaching
  into `nd.shifts[..]`/`net_input`); `fn_shift(*parents, fn=)`/`Fn` (a
  callable or trainable `nn.Module` in the additive shifts);
  `register_term` (whole custom effects as `ShiftTerm` subclasses);
  `effect_modifier_scan(column=)` (scan a named level contrast of a
  multi-level ordinal treatment); `transforms.ordinal_bounds`; `I(transform=)` accepts
  a `_ScaledUT` subclass; `node_parents`/`validate_and_sort`/
  `spec_to_dict`/`spec_from_dict` exported; `flow._fit_validated` declared.
- Stricter spec: a wrong-effect option errors at construction instead of
  silently defaulting; `spec_from_dict` rejects stale/misspelled option
  keys; the unknown-effect error names `register_term`; `spec_to_dict`
  documents the callable/`fn`/class pickle-only caveat; `node_terms` is
  gone (`node.terms` is canonical); `_fit_validated` is set `False` by
  `fit_classical` so callbacks cannot read a pre-classical validation entry.
- BREAKING (0.4 was unreleased): additive-CI checkpoints re-key,
  `nodes.*.intercept_nets.*` → `nodes.*.intercept.nets.*` (parameters
  bit-identical under the rename — refit or re-key, no shim);
  `RestoreBest` had already merged into `EarlyStopping`.

## 0.4.0 (unreleased)

### Changed (breaking) — `fit` is one loop, the strategies are yours

`CausalFlowDAG.fit` shrank from 18 keyword arguments to a minibatch Adam loop
with an `optimizer=` hook and one `callbacks=` list (the shipped recipes in
their own `callbacks.py`). Three independent reviews of the
file agreed on the cut: what left was training *strategy* — choices tuned on
this repository's own DGPs — not the TRAM-DAG model, and each of them had a
default the paper replication or the tests had to switch off.

- **`fit(train_df, *, epochs, learning_rate=1e-2, batch_size=512, seed=None,
  optimizer=None, callbacks=None)`.** One optimizer over all
  parameters (the per-node NLLs have independent gradients, so this equals
  per-node training); `optimizer=` takes any torch optimizer, which is how a
  `torch.optim.lr_scheduler` attaches. `callbacks=` takes one entry or a
  list: a `tramdag.callbacks.Callback` hooks `on_fit_begin(flow,
  optimizer)`, `on_epoch_end(flow, epoch, optimizer)` and `on_fit_end(flow,
  optimizer)`; a bare callable is an `on_epoch_end` hook. Epoch hooks run
  after every epoch — all of them, every epoch — and the fit stops after an
  epoch in which any returned `True`; `on_fit_end` runs before the VC
  re-centering (so restored weights get re-centered). (The three-list shape
  this section first shipped as — `before_fit_callbacks=` /
  `after_epoch_callbacks=` / `after_fit_callbacks=` — collapsed into
  `callbacks=` before release, after a usability review flagged the
  two-hook best-weights registration as a silent footgun.) `flow.history`
  holds the per-node train NLL per epoch — and, with validation configured,
  the per-node validation NLL under `"val"`. `epochs` is a required
  keyword.
- **Gone from `fit`, with what replaces them** (all shown in
  `docs/fitting.md`): `val_df` — which later RETURNED in Keras shape
  (see the validation bullet below) after the callback round measured
  every shipped callback recomputing the same validation NLL;
  `schedule="plateau"`, `plateau_patience`, `plateau_factor`,
  `plateau_min_lr`, `min_delta` (torch's `ReduceLROnPlateau` on the
  optimizer you pass — the paper's VACA/CAREFL replication now runs the
  reference's *global* rule instead of the per-node approximation that was
  a documented deviation); `freeze_patience` and the per-node freeze
  (back as the opt-in `tramdag.callbacks.PerNodePlateau`, the recipe
  `experiments/benchmarks/bench_training.py` measures); `restore_best`
  (`tramdag.callbacks.EarlyStopping`, or a six-line snapshot callback);
  `verbose` and the `tramdag.flow` logger — `verbose=` also returned,
  Keras-style (below), and the module logger stayed gone;
  `epoch_callback` (now an entry in `callbacks=`, receives the
  optimizer); `marginal_init` (moved to `calibrate`); `vc_warm_start` and
  the hidden classical proxy fit (two lines of user code, see
  `docs/varying-coefficients.md`; measured on `vc_hetero`: `beta0` 0.16
  from the truth without it, recovery correlation 0.99 either way);
  `vc_oof_fit` and the hidden five-fold stage-1 fits behind
  `VC(center=True)` (see next bullet).
- **`tramdag.utils` is gone.** `config_section` moved to
  `experiments/common.py::_config_section`, its only caller; `machine_info`
  had already left for the benchmark. The installed package is modelling
  code only: `spec`, `transforms`, `conditioners`, `flow`, `scores` — plus
  the opt-in `callbacks` recipes added below.
- **Degenerate columns fail loudly instead of training into NaN.**
  `calibrate` raises when a continuous node's 5%/95% quantiles coincide (a
  constant or 95%-constant column: the transform has no domain) and when an
  ordinal column is not an integer level index in `0..levels-1` — both used
  to run and produce `nan` or silently truncated levels. An unknown
  `activation=` and an unknown `init=` now raise with the valid choices
  instead of a bare `KeyError` / a silent fallback.
- **`abduct` keeps the caller's index**, and `fit_classical` records its
  report (minus the coefficients) in `flow.history["classical"]`, so a
  classically fitted checkpoint says how it was fitted.
- **Every query and read-out is `(df, node, *, ...)`.** `varying_coef` and
  `intercept_contributions` took the node first and called the frame `data`;
  they now match `pmf`, `density`, `scores`, `design_matrix` and
  `effect_modifier_scan`. `do=`, `seed=`, `t=`, `candidates=` and
  `calibrate(marginal_init=)` are keyword-only, as every `fit` knob already
  was.
- **`flow.calibrate(train_df, marginal_init=True)`** — the data-dependent
  state, taken once: transform ranges, the network-input min-max, the
  calibrated start. The first `fit`/`fit_classical` calls it; a single
  `calibrated` buffer replaces the three per-module first-fit latches
  (`ut._fitted`, `_net_scaled`, `_marginal_inited`) that `load` had to
  re-close by hand. Checkpoints saved before this change do not load
  (the buffer is missing) — refit.
- **`fit(vc_ehat={node: {t: array}})`** — a centered `VC` term needs its
  out-of-fold propensities from the caller, one per training row; `fit`
  refuses a centered spec without them and a `vc_ehat` that does not match
  the spec. `VC(center_folds=)` is gone with the built-in stage (the fold
  count is the caller's), as is `flow.vc_center_info`.
- **`fit_classical`** lost `verbose` (the report dict is the summary) and
  computes its gradient norm with `torch.nn.utils.get_total_norm`.
- **`sample` returns ordinal columns as int64 level indices** (they were
  float32), matching what `fit` requires on the way in; **`log_prob` runs
  under `no_grad`** like every other query — differentiate through
  `node_log_prob` instead.
- **`save`** no longer records the machine, and **`tramdag.utils` is gone**:
  `machine_info` moved to `experiments/benchmarks/perf_machine.py` and
  `config_section` to `experiments/common.py`, each next to its only caller.
  `meta` keeps version, time and device; the package ships modelling code only.
- **`transforms`**: the inverse is zuko's own (`Transform.inv`: bisection
  inside the bound, closed-form linear/identity tail) — the 85-line
  expanding-bracket bisection duplicated it; measured identical residuals,
  spline inverse 166 ms → 0.3 ms on 4000 rows. `StandardLogistic.icdf` is
  `torch.logit(u, eps)`.

### Added

- **Specs serialize YAML-readably.** `spec_to_dict` emits nested kwargs
  tuples (`transform_kwargs`) as mappings and `spec_from_dict` accepts both
  the mapping and the old list-of-pairs form, so a spec can be written by
  hand in YAML and round-trips through JSON/YAML unchanged. This is what
  the experiments' blueprint builds on: every experiment variant now
  carries its full DAG as a `spec:` block in its sibling YAML
  (`experiments/README.md`).

- **`range_q` — the transform-domain quantile is an intercept option.**
  `SI(range_q=0.0)` (any intercept, any basis) calibrates that node's
  domain to the train min/max instead of the default 5%/95% quantiles —
  the original implementation's `scale_df` scaling, exactly. The option
  rides the existing transform-kwargs path, serializes with the spec, and
  validates to `[0, 0.5)`. The CAREFL replication uses it (measured there:
  Fig. 6 max errors 0.37 → 0.20 for x4, 0.20 → 0.07 for x3); VACA measured
  worse with it and keeps the quantile default — the tails matter
  per-problem, which is why it is per-node.

- **`notebooks/classical_fit_tram_dag.py` is back**, ported to the 0.4 term
  syntax. It had been deleted along with `notebooks/stale/`; the deletion
  moved its ordinal half to `experiments/misc/validate_ls.py` and its
  warm-start lesson to `docs/fitting.md`, but the notebook itself is the only
  place `fit_classical` is walked through end to end. The port drops the
  `tramdag.simulations` import (the VACA triangle is simulated inline, as the
  other notebooks do), reads the stroke cohort from `experiments/misc/data/`,
  and replaces the hand-rolled one-hot design with
  `flow.design_matrix(..., drop_first=True)`. It joins the docs workflow's
  `NOTEBOOKS` list, so it is executed on every push to `main` and `dev-*`.

- **Section 1 opens on a real continuous outcome.** `birthwt.csv` gains `bwt`,
  the birth weight in grams that Section 0's binary `low` is cut from, so the
  same three predictors are fitted continuously and checked against R
  `tram::Colr` at matched degree. The `smoke` log-odds ratio comes out at +0.664
  (flow), +0.669 (`Colr`) and +0.671 (Section 0's `glm` on the dichotomized
  outcome) — the proportional-odds property made concrete: a linear shift moves
  the whole latent distribution, so cutting the outcome at 2500 g discards
  information about `h` but leaves the shift alone.

- **Section 1 now reproduces the continuous fit outside the flow too.** R
  `tram::Colr` is shown with its real output rather than as untested code,
  and `statsmodels` gets a route it was previously said to lack: a continuous
  transformation model is the limit of an ordered logit, so cutting the
  outcome into `K` quantile bins and fitting `OrderedModel` converges to the
  flow's shift coefficients. Neither is an exact-MLE check, unlike Sections 0
  and 2: the two libraries implement `h` differently, so the log-likelihoods
  differ by ~0.7-0.9 nats and in **opposite directions** on the two VACA nodes,
  which proves neither function class contains the other. The cause is the
  basis range — the flow pre-scales onto [-5, 5] and leaves 10% of the rows in
  linear extrapolation tails, while `Colr` uses the full support. The section
  also writes its 1000-row sample to `notebooks/data/vaca.csv` and reads it
  back, so the R snippet runs on the identical rows the flow was fitted on, and
  it makes the per-node-kind sign convention explicit: a continuous node adds
  its shift, so `OrderedModel`'s coefficients need negating, while an ordinal
  node's do not.

- **Standard errors and confidence intervals, as a notebook prototype.**
  `fit_classical` leaves the model at the MLE in float64, which is what a
  Hessian-based standard error needs, so the notebook computes the observed
  Fisher information by double backward and inverts it. Sections 1 and 2 both
  reproduce their classical reference: `Colr`'s standard errors to three
  decimals on `bwt`, `statsmodels`' to four on the stroke outcome. The
  information matrix is **singular** in both — 13 of 24 parameters on `bwt`,
  and one flat direction per one-hot parent on the stroke node — so the helper
  uses a pseudo-inverse and reports how far each contrast leaks into the flat
  subspace. A contrast with a non-zero leak still gets a finite, plausible
  number that means nothing, which is why `flow.conf_int` is still not API: one
  row per parameter would be mostly nonsense.

- **Section 2 now says which coefficient is weakly identified, and it is not
  the treatment.** `T` sits 6.6 standard errors from zero. The weak one is
  `mRS_pre` level 5, carried by 7 of 1275 rows, with a standard error six times
  level 1's — matching what `validate_ls.py`'s docstring already recorded. The
  flow-vs-`statsmodels` coefficient gaps turn out to measure something else
  entirely: the optimizer stopping while the likelihood is still flat. The
  displacement `c' I+ grad` predicts them exactly, +1.4e-03 for `Age` and
  +7.4e-03 for `T`.

- **Logistic regression is the zeroth example**, opening
  `classical_fit_tram_dag.py`: a two-level `OrdinalNode` with `LS` terms *is*
  logistic regression, `logit P(Y=1) = -theta_0 + w_0 + sum_p w_p x_p`. It
  runs on `MASS::birthwt` (new `notebooks/data/`, exported verbatim from
  MASS, with a pasteable `glm` snippet that needs no file because the data
  ships with MASS) and agrees with `statsmodels.Logit` and R `glm` on all
  four coefficients to ~1e-8, on the log-likelihood to 1e-5, and on the
  fitted probabilities to 9e-8. It makes two conventions concrete on the simplest
  possible model: an ordinal node *subtracts* its shift, and an ordinal
  parent's one-hot level-0 column is part of the intercept.
- **Keras-shaped validation and progress in `fit`** —
  `validation_data=` (a DataFrame) or `validation_split=` (a float: the
  LAST fraction of `train_df`, unshuffled like Keras; only the head trains
  and calibrates) make `fit` compute the per-node validation NLL once per
  epoch into `flow.history["val"]`, in `validation_batch_size=` chunks.
  `verbose=N` prints every Nth epoch plus the final one (0, the default,
  is silent — no progress bars).
- **`tramdag.callbacks`** — the common training recipes as shipped, opt-in
  callbacks on a small `Callback` base (`on_fit_begin` / `on_epoch_end` /
  `on_fit_end`, each resetting its state at fit begin so instances are
  reusable): `EarlyStopping` (best-validation weights, restored
  automatically at fit end — the recipe the stroke finding needs, since
  flexible CI/CS models overfit confounding at the MLE; an optional
  `patience=` also stops the fit once the best epoch is that many epochs
  old, `restore_best=False` keeps the final weights) and `PerNodePlateau`
  with `per_node_adam` (per-node lr decay and freezing on each node's own
  validation NLL, the pre-0.4 `fit(schedule="plateau")` strategy; its
  `patience`/`freeze` now default to the benchmark's 15/50). All read
  `history["val"]`, so a fit that registers them needs
  `validation_data=`/`validation_split=` — the NLL is computed once,
  shared. Two earlier callbacks existed briefly: `Logger` dissolved into
  `verbose=`, and a separate `RestoreBest` merged into `EarlyStopping`
  (whose restoration is the default). `fit` itself stays one plain loop.
- **`src/tramdag/py.typed` ships (PEP 561)** — the package is fully
  annotated, and pip users' type checkers now see the inline types.
- **`CausalFlowDAG(spec, init="glorot")`** — Keras' `Dense` default
  initialization (glorot-uniform weights, zero biases) for every linear
  layer, the paper's reference; default stays torch's Kaiming-uniform.
  Under the reference's full-batch VACA/CAREFL protocol with its global
  plateau rule the init decides the fit: `do(x2)` errors 0.52 / 0.33 / 0.13
  with torch's init, 0.098 / 0.159 / 0.026 with glorot at the config's seed
  (another init draw: 0.035 / 0.006 / 0.007; Adam's eps made no
  difference). `init="normal"` is Keras' `RandomNormal` (sd 0.05 on weights
  and biases), the paper's triangle scripts' initializer. The VACA/CAREFL
  configs set glorot, the triangle configs normal. Stored in the checkpoint.
- **`CI/CS/VC(input_transform=)`** — every term's network chooses its own
  input transform: `"minmax"`, `"standardize"`, or a callable `fn(x, train)`
  that receives the frozen raw training column (never the batch's — batch
  statistics would encode the same value differently per call). Statistics
  freeze at `calibrate` and live in per-term buffers, so they checkpoint;
  a callable must be a picklable module-level function to `save` (a lambda
  raises with instructions). Linear shifts and the VC treatment stay raw,
  so `ls_coefficients` keep their units. This replaces the earlier
  flow-level `CausalFlowDAG(net_input_scaling=)` draft of the same idea
  (one setting for ALL nets); its motivation carries over unchanged — raw
  parents saturate a tanh net whenever `|x| > 2` (40% of the VACA rows),
  which put the VACA/CAREFL replications 2-20x off the paper until the
  minmax transform (`do(x2=-3)` error 0.731 -> 0.098). Default off;
  checkpoints saved under the flow-level draft do not load — refit.
- **`init_marginals(train_df)`** — the calibrated start as an explicit,
  repeatable step: resets every Bernstein/ordinal simple intercept to its
  column's marginal (spline and affine have no calibrated start)
  (`calibrate(marginal_init=True)` delegates to it). Unlike `calibrate` it
  is not once-guarded, so a loaded or already-trained flow can be restarted
  at the marginal; an uncalibrated flow takes its ranges from the same rows
  first.
- **`density(df, node, grid, do=)`** — the analytic conditional density of a
  continuous node on a grid, the continuous counterpart of `pmf`; closed
  form from the transform, no sampling. Pinned against `exp(log_prob)` at
  the observed value, unit mass, and `do=` ≡ column substitution.
- **`fit(batch_size=)` below 1 raises** with a plain message instead of
  reaching `range()` with a cryptic one.

- **A source node is canonical too.** `ContinuousNode()` and `OrdinalNode(k)`
  now hold `[SI()]` instead of `None`, so `node.terms[0]` really is always
  the intercept, `ContinuousNode() == ContinuousNode([SI()])`, and node
  specs are hashable (a set of nodes used to raise `TypeError`).
- **Specs round-trip through JSON.** `spec_to_dict` was already JSON-safe on
  the way out; `spec_from_dict` now turns the lists JSON gives back into the
  tuples `Term` stores, so a spec saved with `json` compares equal and
  hashes the same as the one written.
- **Paper-aligned intercept constructors**: `simple_intercept`/`SI` (the
  parentless baseline) and `complex_intercept`/`CI` (needs at least one
  parent), matching the paper's SI/CI notation. `intercept`/`I` stays as
  the fallback and dispatches on its arguments, so every existing spelling
  keeps working, and both `I` and `SI` work as bare names in a term list.

- **The experiments are self-contained and configuration-driven.** One
  script per dataset (`triangle`, `triangle_mixed`, `vaca`, `carefl`,
  `validate_ls`), each with the same shape — imports, function
  definitions, a `run(variant)` function, a `__main__` block whose
  argparse call selects the variant — and each reading **every**
  hyperparameter from a sibling `<script>.yaml`;
  `experiments/paper/tests/test_configs.py` checks that every key in a
  variant is read, so a leftover key cannot quietly become a default. `experiments/paper/PAPER_COVERAGE.md` maps every figure of
  arXiv:2503.16206 to the variant that reproduces it, including the
  paper's misspecified case (Fig. 17, new variant `triangle linear-cs`)
  and the two competing baselines that are deliberately not reimplemented.

- **The replication protocol is tuned for runtime (2026-09-01)** — with the
  quality gates unchanged: every ground-truth bound holds under 30-71% less
  training. Triangle runs 300 epochs (linear-cs keeps 500 — its cs curve
  needs them), mixed 350/200, VACA 4800 full-batch steps at lr 0.002 with
  the plateau schedule dropped (the old anneal's only job was freezing
  inside the do-error sweet spot; the tuned lr hits the same window
  directly), CAREFL 3000 at lr 0.002, the validate-ls Adam descent 2000
  epochs instead of 7000. The attempted bigger deviations are documented
  negative results: raw network inputs fail on VACA/CAREFL (sigmoid
  saturates like tanh, relu underfits and grows fragile Bernstein tails)
  and minibatching biases the interventional decomposition (a -0.11
  common-mode offset of the do-means at a perfect observational fit), so
  those two keep tanh + `net_input_scaling: minmax`. Ground truths
  re-pinned; details per experiment in the YAML headers and
  docs/paper-replication.md.

- **An experiments workflow** (`.github/workflows/experiments.yaml`) runs
  all replication variants as a matrix on every push and on demand,
  compares each run's `metrics.json` against the committed
  `experiments/<area>/ground_truth/<name>.json` (a `{value, atol}` entry
  per metric, or `{max}` for an error measure), and posts the run's report — metrics table plus figures — as a
  commit comment through CML. Where an analytic or generator truth
  exists, the table carries `DGP truth` and `abs. error` columns next to
  the fitted value, and every run records `fit_seconds` against a
  `{max, why}` tripwire pinned at 3x the CI-runner measurement — a
  gross-regression alarm (a lost `no_grad`), not a benchmark. `experiments/paper/check_data.py` additionally
  verifies that every frozen dataset still regenerates from its stored
  seed, at 1e-9 rather than bit equality.

- **Transformation syntax**: a node's additive formula is now its first
  positional argument and can be written as a `+` sum — the formula reads
  like the math (`ContinuousNode(I("x1") + CS("x2"))`,
  `OrdinalNode(4, [I, LS("x1")])`). New on `I`:
  `allow_interaction=False` (a multi-parent intercept becomes additive: one net
  per parent, their coefficient vectors summed — written
  `CI("a","b", allow_interaction=False)`, since a node takes at most one
  intercept term with parents) and
  `transform=` (the monotone basis moves onto the intercept term,
  e.g. `I("x1", transform="spline")`; extra keyword arguments pass
  straight to the transform class, `I("x1", transform="spline", bins=6)`).
  Everything
  normalizes to the same internal term list, and equivalence is pinned by
  state-dict-identical tests (`tests/test_transformation_syntax.py`).

- **`fit_classical(history_size=50)`** — the L-BFGS memory is a keyword
  (the other training knobs this bullet once listed left `fit` altogether,
  see *Changed (breaking) — `fit` is one loop*).

- **Pythonic names for every term constructor**, with the short
  notation kept as an alias of the same object: `intercept`/`I`,
  `linear_shift`/`LS`, `complex_shift`/`CS` and
  `varying_coefficient`/`VC`. Both spellings are exported, so
  `LS is linear_shift` and existing code reads unchanged.

- **`units=` on `I`, `CS` and `VC`** sizes the term's network directly,
  e.g. `units=[16]` for one hidden layer (defaults: I `[8, 8]`,
  CS `[64, 128, 64]`, VC `[16]`); serialized per term. All conditioner
  networks now build through one `_mlp()` helper.

### Fixed

- **The docs site published every formula as raw LaTeX.** `pdoc` renders maths
  only with `--math`, which defaults to false and the docs workflow never
  passed. Every `$...$` in the notebooks and guides — around 110 of them, plus
  122 in the new `classical_fit_tram_dag.py` — reached the site as literal
  source. One word in `.github/workflows/docs.yaml`.

- **`check.py` had no test, and the escape hatch it grew hid the thing it was
  meant to expose.** A `"why"` on a `{max}` entry silenced *both* edges of the
  band check, so three `validate_ls` bounds sat 12x, 141x and 566x above their
  measurements — one of them the entry introduced as "the precision claim" —
  and a fabricated 560x regression passed with an `ok`. A `"why"` now excuses
  width only; no argument survives a bound below 1.5x its measurement. The
  three bounds are re-derived from a stated floor (1e-3, the accuracy any
  comparison here claims) rather than inherited from the classical variant,
  whose scale they do not share.

  The other decay mode had no check at all: a `{value, atol}` center keeps
  passing while drifting through its tolerance, which is exactly how two
  centers reached 62% and 75% before anyone noticed. A measurement past half
  its `atol` is now reported the same way.

  `experiments/tests/test_check.py` covers all of it — including that a better
  fit never fails a `{max}`, that a `"why"` cannot silence a too-tight bound,
  and that a truth entry whose metric disappeared is an error. It is the first
  test the file has had, and the band logic had never run on a real input:
  every committed bound was inside the band.

- **vaca's ground truth pinned each flow mean *and* bounded its error against
  the analytic truth.** Those are two windows on one number, offset by
  `|center - analytic|`, and they disagreed: a run landing exactly on the
  pinned `do(x2=-3)` mean would have failed its own paired bound (error 0.0605
  against a bound of 0.0324). Setting the `atol` to the bound, as a first
  attempt did, does not fix it unless the center *is* the analytic value. The
  three centers are gone; the analytic value and the error bound determine the
  flow mean between them, and the run report still prints it.

- **The autoresearch write guard did not cover the numbers.** It denied edits
  to `tests/`, `data/` and the benchmark harness — but the target values moved
  to `experiments/*/ground_truth/` in this release, and that path was
  unguarded.

- **The ordinal counterfactual score had a reference point that was not a
  bound.** `cf_prob_true_level_ceiling` was documented as "the best any model
  could do". It is `E[p_true] = E[sum_i p_i^2]`: what a model that knew the
  identifiable law exactly would score. The largest *expected* score is
  `E[max_i p_i]`, from always naming the *modal* level — a strictly worse
  distribution estimate. The metric is now `cf_prob_true_level_analytic`, with
  `cf_prob_true_level_mode_bound` alongside it (0.921 and 0.954 on the mixed
  DGP). This surfaced when the corrected architecture pushed the flow to 0.924,
  i.e. *above* its own stated ceiling.

  Both references are expectations while the metric is one finite draw, so
  neither is a per-run ceiling either: on the `linear` DGP the mode predictor
  scores **0.829** against its own 0.806 expectation, an overshoot larger than
  the 0.003 gap the metric was introduced to explain. Measured, after a review
  called the first wording ("the attainable maximum") false. Read the two as
  reference points a run sits between, and `cf_pmf_tv_vs_analytic` as the
  metric that cannot be gamed by sharpening a prediction.

- **Ground-truth centers now follow the code.** The architecture change moved
  every complex-shift variant, but only two files were re-pinned, so several
  `{value}` centers described a net that no longer runs — `triangle-atan-cs`'s
  interventional mean had consumed 62% of its tolerance, `triangle-sin-cs`'s
  beta13 75%. All centers are re-measured. `{max}` bounds are now kept inside a
  band instead of hand-tuned: useful between 1.5x and 4x the measurement, set
  to 2.5x outside it. Below 1.5x is not hypothetical — one bound at 1.7x passed
  locally at 0.028 and failed CI at 0.113, because its maximum is over a
  coefficient with 7 of 1275 observations. That comparison is now split into
  `max_abs_diff_named_coefs` (Age, NIHSSa, the treatment contrast: the
  precision claim, ~1e-3) and the all-coefficient maximum (a sanity bound).

- **Two of the four paper replications used the wrong reference
  architecture.** The reference implementation has two: the triangle
  scripts use `hidden_features_I = hidden_features_CS = c(2,25,25,2)` with
  sigmoid, while its own CAREFL and VACA comparisons use
  `comparison/utils.R::make_model` — one net per node,
  `dense(10, tanh) -> dense(100, tanh) -> dense(len_theta)`, with `M = 30`.
  `vaca.yaml` and `carefl.yaml` cited the first while replicating the
  second. On the correct net CAREFL improves on every previously committed
  number (counterfactual MAE 0.078/0.059/0.086 against bounds
  0.216/0.174/0.219) and VACA's off-manifold `do(x2=-3)` mean lands 0.037
  from the analytic truth instead of 0.21. Ground truth re-pinned, and the
  `{max}` bounds re-pinned into a band — see the later entry on ground-truth
  centers for the rule that replaced this pass's first attempt at one.

- **The `conditioners` provenance claim was wrong.** The module, README,
  CLAUDE.md and `docs/code-map.md` all said the default architectures come
  from "the original Keras implementation (`tram_models.py` in
  tensorchiefs/tram-dag)". That repository is pure R and has no Python in
  it. The defaults come from the PyTorch reference this package grew out
  of (buehlpa/TramDag, `tram_models.py`:
  `ComplexShiftDefaultTabular` 64-128-64 ReLU,
  `ComplexInterceptDefaultTabular` 8-8 ReLU, `n_thetas=20`) — which is
  also why `DEFAULT_ACTIVATION` is relu. Corrected in all four places,
  each of which now also says these are *not* the paper's nets, so a
  replication states `units=` and `activation=` itself.

- **The `n_coeffs` documentation was off by two control points.** zuko
  constrains `n` unconstrained coefficients into `n + 2` monotone control
  points, duplicating the end differences for a smooth extrapolation, so
  `n_coeffs=20` is a degree-21 polynomial where the reference's
  `len_theta=20` is degree 19. The configs claimed "order M = 20 from the
  paper"; they now state the mapping and that the free-parameter count is
  what matches.

- **`ls_coefficients()` crashed on a node mixing `LS` and `CS` terms.** It
  read `.weight` off every shift module, but a `CS` shift is a network and
  a `VC` shift is an effect head — neither has one, so any such node raised
  `AttributeError`. That broke the paper's headline complex-shift
  replication (the old `paper_triangle.py`'s documented default was
  `atan cs`). The method now returns the linear-shift weights it is named
  for and skips network shifts; a node with no `LS` term is absent from the
  result. The bug predates 0.4 (`tests/test_api_papercuts.py`).


### Removed (breaking)

- **`config_section(require=)` and the key-set check** — the loader picks
  the section and nothing more; a script's YAML is the single source of its
  numbers, and the check bought a second copy of every key list in code.
- **`fit_classical(chunk=)`** — L-BFGS runs once with torch's own
  `tolerance_change` (the `tol` argument, now 1e-9) instead of in
  hand-rolled rounds with a relative NLL-flatness test; `n_iter` is torch's
  count. Measured on the classical anchor: 1e-9 reproduces the old
  convergence exactly (coefficients within 0.0027 of statsmodels), 1e-6
  would stop on a plateau step.
- **Two demo-notebook sections** — the misspecification excursion and the
  GPU-vs-CPU race — so the demo stays the five-minute L1/L2/L3 walk.

- **`tramdag.simulations` is no longer part of the package.** The SCM
  generators are research code and moved to `experiments/paper/simulations/`,
  together with the frozen datasets (`data/` → `experiments/paper/data/`). The
  wheel now contains framework code only, and `import tramdag.simulations`
  fails. The stroke storyline left with them: the `magic_mrclean` generator,
  the `magic-mrclean/nl` cohort, `sim_flow.py`, the `vc_shift` DGP,
  `experiments/stale/` and `docs/stroke-case-study.md` are deleted, and the
  clinical case study lives in its own repository.
  `experiments/misc/data/magic-mrclean/ls` stays as the frozen input of
  `validate_ls`. **Everything deleted is recoverable at the
  `pre-experiments-cut` tag**: `git checkout pre-experiments-cut -- <path>`.

- **`transform_kwargs=`** on the intercept constructors. Extra keyword
  arguments now pass straight to the transform class:
  `I("x1", transform="spline", bins=6)`. The canonical storage inside
  `Term.options` is unchanged, so serialized specs are unaffected.

- **The `"onecycle"` and `"cosine"` schedules.** They had no caller
  outside one parametrized test, and the June 2026 benchmark measured both
  behind plateau on every workload (`docs/training-speed.md` keeps the
  numbers). The plateau schedule itself left `fit` later in this release;
  a schedule is now a torch `lr_scheduler` on the optimizer you pass.

- **The `bound` knob on the univariate transforms.** Nothing ever set it;
  the pre-scaled domain is fixed at `[-5, 5]` (`transforms.BOUND`).

- **`VC(center=)` is a plain bool.** The `center="colname"` variant
  (user-supplied cross-fitted propensities) had a self-test and no other
  caller; it was staged-unreleased, so its Added entry is corrected in
  place.

- **All backward compatibility.** Pre-1.0, one API and one checkpoint
  format: the lowercase `"ls"`/`"cs"`/`"ci"` effect aliases are gone (a
  checkpoint carries `"I"`, `"LS"`, `"CS"`, `"VC"`), the two
  0.3-checkpoint shims in `spec_from_dict` (the multi-`I` merge and the
  node-level-basis carry) are gone with the redundant node-level
  `transform`/`transform_kwargs` keys that fed them, VC terms read
  `center` directly, and `load` requires a complete
  checkpoint (spec, weights, history, meta) instead of tolerating missing
  blocks. Checkpoints and specs written by earlier versions no longer
  load; regenerate them.

- **`Term.slot`** — derived from `effect`, and its only user in the repo
  was a test assertion.

- Node-level `ContinuousNode(transform=/transform_kwargs=)` (choose the
  basis on the intercept term instead, `I(..., transform="spline")`), the
  unused `Intercept`/`LinShift`/`CShift` aliases, and the `parents={...}`
  checkpoint loader. The terms argument itself is unchanged and still
  accepts its name: `ContinuousNode(terms=[...])` and
  `ContinuousNode([...])` are the same call.
- The unmaintained notebooks and experiment scripts were first moved to
  `notebooks/stale/` and `experiments/stale/`, then deleted in this same
  release (see Removed); the maintained set is the intro and Colab demo
  notebooks plus the `experiments/` tree.

- **`term(effect, *parents)`** — the string-label term factory. It was a
  second, weaker way to build a term, its `VC` branch and `penalty=`
  keyword were exercised only by its own tests, and a generic dispatcher
  carrying one effect's parameter is the shape this release removed from
  `Term` itself. When the effect type comes from config or the CLI, hold
  the constructor in the table instead of a label to dispatch on:
  `{"Age": I, "NIHSSa": CS}` then `t["Age"]("Age")` (see
  `experiments/paper/helpers.py::shift_term`).

### Changed (breaking)

- **The paper replications follow the paper's R code 1:1.** Triangle: one
  Adam run at lr 0.001, Keras' default batch size 32, 500 epochs, a separate
  validation draw, coefficients read every epoch. VACA/CAREFL: nTrain 2500,
  one full-batch step per epoch, 10000 / 7000 epochs, ReduceLROnPlateau
  (0.1 / 50 / 1e-7) on a separate 5000-row draw. Gone: the 90/10 split,
  `chunk_epochs` (fresh Adam per chunk — a warm-restart schedule the
  reference never had), the `polish_*` phase, `fit_in_chunks`. Every
  ground-truth file is re-pinned from these runs; the deviations that
  remain are framework-level and listed in each YAML and CLAUDE.md.
- **`calibrate(marginal_init=True)` is the default start.** Every unconditional
  intercept starts at its marginal (Bernstein: the linear map onto the
  latent 5%/95% quantiles; ordinal: the empirical class log-odds) instead
  of zuko's zero start. A pure initialization — the MLE is unchanged — but
  a fixed-budget Adam fit lands elsewhere, so the paper replications'
  ground truth is re-pinned from the CI run that follows. `fit_classical`
  keeps the zero start: L-BFGS does not need it, and the classical anchor
  stays bit-identical.

- **`fit(epochs=)` is required.** There is no default training budget any
  more. `docs/training-speed.md` measures a fixed budget going wrong in both
  directions on this repo's own workloads — the stroke fit converges after
  ~1500 of 4000 budgeted epochs, the vaca fit is 0.03 nats short at 520 — so
  a package-level number could only be arbitrary. `epochs` is a bare
  keyword-only argument (`TypeError` without it); a generous budget with a
  stopping after-epoch callback is the alternative.

- **`make_univariate_transform` raises `KeyError`, not `ValueError`.** The
  registry lookup speaks for itself; this package no longer re-words the
  failure of the dict it owns. The docstring said `ValueError` even before,
  so an `except ValueError` around a spec built from a hand-edited checkpoint
  never caught anything.

- **Single-value keyword arguments that no caller ever set became
  constants**: `StandardLogistic.sample(eps=)`/`icdf(eps=)` (`_U_EPS`),
  `BernsteinUT.marginal_init_theta(q=)` (`RANGE_Q`),
  `ordinal_marginal_init_theta(eps=)` (`_CDF_EPS`) and
  `scores.sup_bb_pvalue(terms=)`. `RANGE_Q` is now shared with
  `_set_ranges`, which hard-coded the same 0.05: the two only calibrate
  each other when they agree, so any other `q` was a silent
  miscalibration rather than a setting.

- **`VC(*modifiers, t=...)`**: the positional arguments are the
  covariates that enter `b_theta`; the treatment `t` is a required
  keyword. `VC("X2", "X3", t="T")` reads as
  `(beta0 + b_theta(x2, x3)) * x_t`. (0.3 wrote `VC("T", "X2", "X3")`.)

- **No progress output from the package.** `fit` and `fit_classical`
  print and log nothing (the `verbose=` gate and the `tramdag.flow`
  logger of an intermediate state are gone); `verbose=` or a bare
  `on_epoch_end` callable reports what you want, `fit_classical` returns its report dict.

- **The node formula argument is `terms`, not `transformation`.**
  `ContinuousNode(terms=...)` / `OrdinalNode(levels, terms=...)`, and the
  attribute is `node.terms`. It is still the first positional argument, so
  positional calls are unaffected; the word now matches what it holds and
  what `Term`/`node_terms` already say.

- **A `+` sum nested inside a list is rejected.** `+` already returns a
  flat list, so `[I("x1") + LS("x2")]` was a list of lists that the
  normalizer silently flattened. Write either a list or a sum; the error
  says so, because the usual cause is expecting `+` to combine list
  entries.

### Changed (internal, no API surface)

- **The docs site moves from pdoc to MkDocs** (Material, mkdocstrings for the
  numpy docstrings, mkdocs-jupyter for the executed notebooks, mike for one
  version per branch). The 233-line workflow with its stub modules, link
  rewriter and jinja template becomes `mkdocs.yml` plus a 40-line link hook;
  math renders through MathJax, and the code map's symbols link to the API
  pages. The PDF is typeset by pandoc + XeLaTeX from README, guides and the
  executed notebooks, so its math is real math.

- `experiments/paper/helpers.py`: the per-epoch coefficient read-out is
  `fit(callbacks=)` inside one `fit_paper(generator, spec, config, out,
  record)` call — there is no chunked or snapshotting fit helper any more.

- **Every complexity hotspot is dissolved into named stages** — `fit`
  (cognitive complexity 103 → 10), `validate_and_sort` (65 → 1),
  `_Node.__init__`, `to_matrix`, `check.compare`, `bench_training.main` and
  seven more — and every module follows one `# %% <section>` layout. Verified
  behavior-identical: a fixed-seed harness compares state dicts, history and
  samples bit-equal before and after, and all error messages moved verbatim.

- **The framework tests carry their own data.** Three inline numpy DGPs in
  `tests/conftest.py` (an all-`ls` chain, a heterogeneous-effect DGP and a
  confounded DGP with a prognostic misfit) replace the generator package the
  suite used to import. The external-software anchor is unchanged in
  substance — an all-`ls` outcome node is an ordered-logit model, so the
  flow's MLE must equal `statsmodels` on the same design matrix — but it is
  now measured on inline data at test time instead of against a committed R
  reference; the R comparison lives on in `experiments/misc/validate_ls.py`. The
  generator-pinning and frozen-CSV tests moved to the experiments workflow.

- **The stacked ternaries in the `vaca` and `carefl` generators' `simulate`
  became one if/else per variable** (readability of experiment code).
  Verified behaviour-neutral: `vaca` regenerates bit-identically and
  `carefl` within 7e-15, the same machine-epsilon drift the untouched
  `triangle` generator shows after the dependency bump.

- **The serialized term is `{effect, parents, options}`.** `Term.options`
  is already canonical (sorted, defaults dropped), so `spec_to_dict` emits
  it whole and the per-key reader disappears. `spec_from_dict` now builds
  `Term` directly, which makes `validate_and_sort` the only guard on the
  load path — a malformed checkpoint is rejected there
  (`tests/test_transformation_syntax.py`).

- **The SCM generators share one layer.** `simulations/_common.py` holds
  `logistic`, `sigmoid`, `resolve_latents`, and the `DatasetDraws` mixin
  (`observational`, `interventional`, `counterfactual_pair`) — with it the
  seed offsets behind the frozen CSVs in `data/` (`+1`, `+501`, `+2`) are
  defined once instead of once per generator. Every generator exposes the
  same three named draws. What is left of the package after the stroke
  storyline moved out is 916 lines for the four paper generators, with the
  frozen-CSV contract unchanged.

### Added (staged earlier as an unreleased 0.3.1)

- **Propensity-centered VC: `VC(..., center=True)`** (issue
  #30): the R-learner orthogonalization `beta(x)·(t − ê(x))` inside the
  likelihood, as a **two-stage frozen** design — training uses **out-of-fold**
  ê, one value per training row passed as `fit(vc_ehat=)` (DML cross-fitting;
  the caller computes them — six lines with `fit_classical`, see
  `docs/varying-coefficients.md`), frozen as data (zero gradient into the
  treatment node from the outcome loss — tested); inference recomputes ê from the flow's own fitted
  treatment node and re-derives `t − ê(x)` under `do` (never cached — tested
  on fresh rows). binary ordinal treatments only; `center=False` (default) is bit-identical
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

- **`VC(*modifiers, t=, penalty=)` — varying-coefficient shift term** (issue #28;
  first staged as `VC(on, *modifiers, penalty=)`, see Changed (breaking)):
  a treatment-effect head `beta(x) = beta0 + b_theta(x)` with a small (16-unit),
  **penalized**, zero-initialised network that only multiplies `x_on`, plus the
  first-class read-out `flow.varying_coef(df, node)` (closed-form,
  deterministic, y-free; equals the abduct-difference for binary treatments).
  The objective is the penalized likelihood `Σ NLL + penalty·‖w‖²` (total-NLL
  scale, `beta0` unpenalized); `penalty → ∞` — or `modifiers=()` exactly —
  nests `LS(on)`; a classical warm start of `beta0` is two lines of user code.
  VC modifiers may also appear in prognostic terms
  (only `on` owns its edge). Motivation, measured: the `CS(on, x…)` reduced form
  is *expressive but unestimated* — corr ≈ 0.5 against the true effect function
  even in-class (tramdag-simu#18/PR #21) because nothing in the NLL rewards a
  smooth arm-difference; the regularized head reaches ≈ 0.99 on the same task
  class. Validation DGP with known `beta_true = −1 + 0.8·X2 − 0.6·X3` — the
  inline `vc_hetero` DGP in `tests/conftest.py` (the earlier
  `simulations/vc_shift.py` + frozen `data/vc-shift/` left with the stroke
  storyline, see Removed); acceptance tests in `tests/test_vc_term.py`
  (recovery bar corr ≥ 0.9 at n = 5000, measured min-over-seeds 0.986). Docs:
  `docs/varying-coefficients.md`.

- **`flow.intercept_contributions(df, node)`** (issue #20, Option A) — post-hoc,
  mean-centered decomposition of an **additive complex intercept**
  (`CI("x1", "x2", allow_interaction=False)`). The per-term networks are summed in unconstrained
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
