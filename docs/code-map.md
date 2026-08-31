# Code map — every class and function in `src/tramdag/`

One entry per name, with its role and its place in the pipeline. Names in
parentheses are private machinery: useful to know, not part of the API.
The last section lists every training hyperparameter and where it lives.

## `spec.py` — declare the model

Every term constructor has a pythonic name and a short alias; the two
are the same object, so `LS is linear_shift`.

| Name | Role |
|---|---|
| `Term` | One additive term of a node's transformation: the frozen triple `(effect, parents, options)`. `+` on terms builds plain lists. Effect-specific settings live in `options` and read as attributes (`term.penalty`, `term.units`, ...). |
| `simple_intercept()` / `SI` | The parentless intercept — the paper's SI. Free transform parameters, the same for every row. Carries the basis choice (`transform=`, default `"bernstein"`); extra keyword arguments pass straight to the transform class. |
| `complex_intercept()` / `CI` | The parent-conditioned intercept — the paper's CI: the parents reshape the monotone transform. Needs at least one parent. Also carries `units=` and `allow_interaction=` (joint vs. additive multi-parent intercept). |
| `intercept()` / `I` | The fallback: dispatches on its arguments to `SI` (no parents) or `CI` (parents). The bare names `I` and `SI` in a term list both mean the simple intercept. |
| `linear_shift()` / `LS` | Linear shift `beta * x` — the interpretable log-odds coefficient. Exactly one parent. |
| `complex_shift()` / `CS` | Complex shift: an MLP `g(x)`, additive on the latent scale. Several parents form one joint network. |
| `varying_coefficient()` / `VC` | Varying-coefficient shift `(beta0 + b_theta(mods)) * x_t` — the penalized treatment-effect head (issue #28). `center=` adds propensity centering (issue #30). |
| `ContinuousNode` | Continuous variable: monotone 1-D transform plus shifts. `terms` is the first positional argument. |
| `OrdinalNode` | Ordinal variable with `levels` classes: ordered logit (cutpoints) plus shifts. |
| `node_terms()` / `node_parents()` | Canonical term list / ordered de-duplicated parent names of a node. |
| `validate_and_sort()` | Edge-ownership validation plus Kahn topological sort. The returned order makes the flow triangular. |
| `spec_to_dict()` / `spec_from_dict()` | Checkpoint (de)serialization. A term serializes as `{effect, parents, options}` and nothing else, since `options` is already canonical. No compatibility shims: `spec_from_dict` rejects a term without `options`, the node constructors normalize the formula, and `validate_and_sort` checks the DAG. |
| (`_normalize_terms`, `_as_term`, `_intercept_basis`, `_options`, `_OPTION_DEFAULTS`) | Formula flattening and per-entry validation (a `+` sum nested in a list is rejected), the one-parented-`I` rule plus basis hoisting in one pass, canonical option storage. |
| (`_check_term`, `_check_node`, `_check_vc_term`, `_kahn_sort`) | The stages behind `validate_and_sort`: per-term validation (effect, LS arity, unknown parents), per-node edge-ownership bookkeeping, the VC-specific checks, and the topological sort. |

## `transforms.py` — the monotone map h and the ordinal transform

| Name | Role |
|---|---|
| `StandardLogistic` | The TRAM base distribution: `log_prob`, `sample` (generator-aware), `icdf`. |
| `BernsteinUT` | Bernstein-polynomial transform (default basis, `n_coeffs=20`). Linear tail extrapolation follows the boundary derivative. `marginal_init_theta()` gives the calibrated start used by `calibrate(marginal_init=True)`. |
| `SplineUT` | Monotone rational-quadratic spline (`bins=8`). Tails extrapolate with a *fixed* slope — the structural reason spline trails Bernstein on tail-heavy data. |
| `AffineUT` | Monotone affine transform: the node-conditional is a logistic GLM. |
| `make_univariate_transform()` | Basis registry: name → transform instance. |
| `ordinal_cutpoints()` | Unconstrained `(n, K-1)` → increasing cutpoints with ±inf ends. Port of the original parametrization. |
| `ordinal_log_prob()` | `log P(Y=y)`, computed in log-space. Load-bearing: the naive sigmoid difference saturates in float32 and freezes nodes at init. Do not simplify. |
| `ordinal_pmf()` / `ordinal_sample()` / `ordinal_abduct()` | Class probabilities / latent → level / truncated-logistic latent recovery (Pearl step 1) for ordinal nodes. |
| `ordinal_marginal_init_theta()` | Cutpoint start that matches the empirical class frequencies (`calibrate(marginal_init=True)`). |
| (`_ScaledUT`, `_bounds`, `_log1mexp`) | Quantile pre-scaling base class (the inverse is zuko's, with its closed-form tail); per-level cutpoint intervals; stable `log(1-exp(x))`. |

## `conditioners.py` — the networks behind the terms

Default architectures replicate the PyTorch reference this package grew out of
([buehlpa/TramDag](https://github.com/buehlpa/TramDag), `tram_models.py`), so a
fitted model stays comparable to it. They are **not** the TRAM-DAG paper's nets:
the paper's R code uses `c(2, 25, 25, 2)` with sigmoid for the triangle
experiments and a 10-100 tanh net for its CAREFL/VACA comparisons, so each
config in `experiments/paper/` states `units=` and `activation=` itself.

| Name | Term | Role |
|---|---|---|
| `SimpleIntercept` | bare `I` | Free parameter vector; no parents. |
| `ComplexIntercept` | `I(...)` | 8-8 ReLU MLP from parent features to the transform parameters. |
| `LinearShift` | `LS` | `Linear(n, 1, bias=False)`. `.weight` is the interpretable coefficient; no bias because the intercept slot owns the constant. |
| `ComplexShift` | `CS` | 64-128-64 ReLU MLP to one shift value. |
| `VaryingCoef` | `VC` | `beta0 + b_theta(mods)` with a zero-initialized output layer and the L2 hook `l2()`. `beta()` evaluates the effect, `recenter()` re-splits `beta0`/`b_theta` after training (function-preserving). |
| (`_mlp`) | — | The one MLP builder: a stack of the given `units` with the term's `activation` (relu by default), then a bias-free output layer. |

## `flow.py` — the model

| Name | Role |
|---|---|
| `CausalFlowDAG` | The flow: one `_Node` per variable in topological order. Construction seeds the weights (`seed=` is the reproducibility knob). |
| `calibrate()` | Once, from the training rows: transform ranges (train 5%/95% quantiles onto the domain), network-input min-max, the calibrated start (`marginal_init=True`). Called by the first fit; a checkpoint carries the flag. |
| `init_marginals()` | The calibrated start as an explicit step, callable any time: resets every simple intercept to its column's marginal (Bernstein map / ordinal class log-odds). Not once-guarded — on a trained flow it restarts those intercepts. Calibrates a fresh flow's ranges itself. |
| `fit()` | Joint maximum likelihood: one minibatch Adam loop over all parameters (exact per node, because the NLL decomposes), final weights kept. Hooks: `optimizer=` (any torch optimizer, for schedulers), `after_epoch_callbacks=` (one callable or a list, `cb(flow, epoch, opt)` after every epoch, any `True` stops) and `before_fit_callbacks=`/`after_fit_callbacks=` around the loop — validation, schedules, early stopping and logging attach here; the common recipes ship in `callbacks.py`. `vc_ehat=` carries the out-of-fold propensities of centered VC terms. A second call continues training. |
| `fit_classical()` | Float64 full-batch L-BFGS for all-`ls` specs: deterministic, exact MLE, matches `statsmodels`/R `polr`. Refuses flexible specs. |
| `sample()` | Observational, interventional (`do=`, graph mutilation) and counterfactual (`u=`) sampling. |
| `abduct()` | Pearl step 1: recover the latents. Continuous exactly, ordinal by truncated draw. |
| `pmf()` | Analytic class probabilities of an ordinal node, with `do=` overrides. |
| `density()` | Analytic conditional density of a continuous node on a grid, with `do=` overrides — the continuous counterpart of `pmf`. |
| `log_prob()` / `nll()` | Joint per-row log-likelihood / mean per-node NLL diagnostic. |
| `node_log_prob()` | The per-node decomposition everything trains and evaluates through. |
| `varying_coef()` | Closed-form read-out `beta(x)` of a fitted VC term. Deterministic, y-free. |
| `scores()` / `effect_modifier_scan()` | Analytic per-observation scores and the CUSUM modifier scan (delegate to `scores.py`). |
| `intercept_contributions()` | Post-hoc GAM-style decomposition of a complex intercept into mean-centered per-term parts. |
| `ls_coefficients()` | The per-node linear-shift weights — the interpretable coefficients. |
| `design_matrix()` | Parent encoding as a DataFrame (`drop_first=` gives the classical statsmodels/`polr` design). |
| `to_matrix()` | The labeled meta-adjacency matrix of term effects. |
| `save()` / `load()` | Checkpoints with history and provenance (version, time, device). `load` requires a complete checkpoint and fails loudly otherwise. |
| (`_Node`, `_VCGroup`) | Per-node module (intercept + shift `ModuleDict` + VC bookkeeping); construction is `_build_intercept`/`_build_shifts`, and `theta_shift()` computes `(theta, shift)` through `_theta`/`_vc_shift`/`vc_column`. |
| (`_node`, `_encode_parent`, `_features`, `_tensorize`, `_generator`, `_dtype`, `_np_dtype`, `_feat_width`, `_slice_ehat`, `_term_cells`) | Node lookup with one shared error; parent encoding (continuous raw, ordinal one-hot); `_tensorize(df, cols=None)` for any column subset; seeded-generator, dtype, feature-width and adjacency-cell plumbing. |
| (`_fit_epoch`) | One shuffled pass over the rows; the epoch-mean train NLL per node goes to `history["train"]`. |
| (`_vc_ehat_train`, `_binary_p1`, `_vc_ehat_live`, `_vc_ehat_columns`, `_recenter_vc`) | The VC machinery: the caller's out-of-fold propensities as frozen training tensors (DML); live full-fit propensities for inference (recomputed under `do`, never cached); post-fit re-centering. |
| (`_is_all_ls`, `_covered_by_classical`) | Guard for `fit_classical`: every term an `LS`, or a parentless `I()` basis carrier. |

## `scores.py` — effect-modifier detection (issue #29)

| Name | Role |
|---|---|
| `node_scores()` | Analytic, exact per-observation scores `psi_i = d l_i / d theta` for every `LS` weight and VC `beta0`. No autograd. |
| `effect_modifier_scan()` | Zeileis-Hornik fluctuation scan: order the treatment scores by each candidate, `sup|CUSUM|` against the Kolmogorov 5% value. A measured shortlist for VC modifiers from a seconds-long classical fit. |
| `sup_bb_pvalue()` | `P(sup |Brownian bridge| > stat)`, the Kolmogorov series. |
| (`_dl_ds`, `_ls_score_columns`, `CRIT_5PCT`) | Closed-form latent-scale derivative; the LS/one-hot score-column builder; the 5% critical value 1.3581. |

## `callbacks.py` — the shipped `fit` callbacks

| Name | Role |
|---|---|
| `Logger` | Prints one line per `every` epochs: summed train NLL, plus validation NLL when `val_df` is given. |
| `RestoreBest` | Snapshots the weights of the best summed validation NLL per epoch; `restore` (registered in `after_fit_callbacks`) loads them back before the VC re-centering. |
| `PerNodePlateau` | Per-node lr decay and freezing on each node's own validation NLL; stops the fit once every node froze. The pre-0.4 `fit(schedule="plateau")` recipe, opt-in. `step(nll, opt)` for a hand-computed NLL. |
| `per_node_adam()` | Adam with one `node`-tagged parameter group per node — the optimizer `PerNodePlateau` needs. |

## What is *not* in the package

The SCM generators, the frozen datasets and the replication scripts are
research code and live in [`experiments/`](../experiments/), outside the
installed package — see
[`experiments/README.md`](../experiments/README.md). The framework's own
tests do not depend on them: they measure against three inline DGPs in
[`tests/conftest.py`](../tests/conftest.py).

## Where every training hyperparameter lives

Everything that shapes a fit is either a keyword you pass or a documented
default you can read at the call site. Nothing numeric is buried.

| Knob | Where | Default |
|---|---|---|
| learning rate, batch size | `fit()` | 1e-2 / 512 (in-repo callers state them explicitly anyway) |
| schedules, early stopping, logging | `fit(optimizer=, after_epoch_callbacks=)` | `tramdag.callbacks` ships `Logger`, `RestoreBest`, `PerNodePlateau`; anything else is torch's `lr_scheduler` and a few lines of callback ([fitting.md](fitting.md)) |
| calibrated init | `calibrate(marginal_init=)` | True (pure init, MLE unchanged; `False` = zuko's zero start; called by the first fit) |
| VC stage-1 propensities | `fit(vc_ehat=)` | required for a centered VC term, out of fold, computed by the caller |
| VC penalty and centering | `VC(penalty=, center=)` | 1.0 / False (centering needs `fit(vc_ehat=)`) |
| L-BFGS budget | `fit_classical(max_iter=, tol=, history_size=)` | 400 / 1e-9 (torch `tolerance_change`; `tolerance_grad` is off) / 50 — one full-batch run, no chunks |
| training budget | `fit(epochs=)` | **required** — a fixed default is wrong in both directions ([training-speed](training-speed.md)) |
| network widths | `units=` on `I`/`CS`/`VC` | (8, 8) / (64, 128, 64) — parity with the PyTorch reference's default classes; VC's (16,) has no counterpart there and comes from the recovery measurement |
| activation | `activation=` on `I`/`CS`/`VC` | `"relu"` (the reference default classes); `"sigmoid"` and `"tanh"` are the paper's |
| transform basis | `I(transform=, **kwargs)` (extra kwargs go to the transform class) | `"bernstein"`, `n_coeffs=20` unconstrained coefficients (zuko ties two more control points on, so order 21); spline `bins=8` = zuko's NSF default (the domain is fixed at [-5, 5], `transforms.BOUND`) |
| shuffling / weight init | `fit(seed=)` / `CausalFlowDAG(seed=)` | init happens at construction — the constructor seed is the reproducibility knob |
| weight init | `CausalFlowDAG(init=)` | `"torch"` (`nn.Linear` Kaiming-uniform); `"glorot"` = Keras `Dense` default, glorot-uniform weights and zero biases — the paper's reference; decisive under its full-batch protocol |
| network inputs | `CausalFlowDAG(net_input_scaling=)` | `None` (raw parents); `"minmax"` scales the continuous parents of every net (CI/CS/VC modifiers) to [0, 1] by the training min–max, the reference's `scale_df` — LS and the VC treatment stay raw |
