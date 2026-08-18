# Code map — every class and function in `src/tramdag/`

One entry per name, with its role and its place in the pipeline. Names in
parentheses are private machinery: useful to know, not part of the API.
The last section lists every training hyperparameter and where it lives.

## `spec.py` — declare the model

| Name | Role |
|---|---|
| `Term` | One additive term of a node's transformation: the frozen triple `(effect, parents, options)`. `slot` is derived from the effect. `+` on terms builds plain lists. Effect-specific settings live in `options` and read as attributes (`term.penalty`, `term.units`, ...). |
| `intercept()` / `I` | Intercept term: the parents reshape the monotone transform. `I()` or the bare name `I` is the simple-intercept baseline. Carries the basis choice (`transform=`, default `"bernstein"`) and `allow_interaction=` (joint vs. additive multi-parent intercept). |
| `LS()` | Linear shift `beta * x` — the interpretable log-odds coefficient. Exactly one parent. |
| `CS()` | Complex shift: an MLP `g(x)`, additive on the latent scale. Several parents form one joint network. |
| `VC()` | Varying-coefficient shift `(beta0 + b_theta(mods)) * x_t` — the penalized treatment-effect head (issue #28). `center=` adds propensity centering (issue #30). |
| `term()` | Build a term from a data-driven label (`"LS"`, or legacy `"ls"`/`"cs"`/`"ci"`). For sweeps that read the effect type from config. |
| `ContinuousNode` | Continuous variable: monotone 1-D transform plus shifts. The transformation is the first positional argument. |
| `OrdinalNode` | Ordinal variable with `levels` classes: ordered logit (cutpoints) plus shifts. |
| `node_terms()` / `node_parents()` | Canonical term list / ordered de-duplicated parent names of a node. |
| `validate_and_sort()` | Edge-ownership validation plus Kahn topological sort. The returned order makes the flow triangular. |
| `spec_to_dict()` / `spec_from_dict()` | Checkpoint (de)serialization. The loader carries the two 0.3-format shims: multi-`I` merge and node-level-transform carry. |
| (`_normalize_transformation`, `_check_single_intercept`, `_hoist_transform`, `_options`, `_OPTION_DEFAULTS`, `_LEGACY`) | Input normalization, the one-parented-`I` rule, basis hoisting onto the node, canonical option storage, legacy label map for `term()`. |

## `transforms.py` — the monotone map h and the ordinal transform

| Name | Role |
|---|---|
| `StandardLogistic` | The TRAM base distribution: `log_prob`, `sample`, `icdf`. |
| `BernsteinUT` | Bernstein-polynomial transform (default basis, `n_coeffs=20`). Linear tail extrapolation follows the boundary derivative. `marginal_init_theta()` gives the calibrated start used by `fit(marginal_init=True)`. |
| `SplineUT` | Monotone rational-quadratic spline (`bins=8`). Tails extrapolate with a *fixed* slope — the structural reason spline trails Bernstein on tail-heavy data. |
| `AffineUT` | Monotone affine transform: the node-conditional is a logistic GLM. |
| `make_univariate_transform()` | Basis registry: name → transform instance. |
| `ordinal_cutpoints()` | Unconstrained `(n, K-1)` → increasing cutpoints with ±inf ends. Port of the original parametrization. |
| `ordinal_log_prob()` | `log P(Y=y)`, computed in log-space. Load-bearing: the naive sigmoid difference saturates in float32 and freezes nodes at init. Do not simplify. |
| `ordinal_pmf()` / `ordinal_sample()` / `ordinal_abduct()` | Class probabilities / latent → level / truncated-logistic latent recovery (Pearl step 1) for ordinal nodes. |
| `ordinal_marginal_init_theta()` | Cutpoint start that matches the empirical class frequencies (`marginal_init`). |
| (`_ScaledUT`, `_expanding_bisection`, `_bounds`, `_log1mexp`) | Quantile pre-scaling base class; tail-safe inverse; per-level cutpoint intervals; stable `log(1-exp(x))`. |

## `conditioners.py` — the networks behind the terms

Architectures replicate the original Keras implementation, so fitted models
stay comparable to it.

| Name | Term | Role |
|---|---|---|
| `SimpleIntercept` | bare `I` | Free parameter vector; no parents. |
| `ComplexIntercept` | `I(...)` | 8-8 ReLU MLP from parent features to the transform parameters. |
| `LinearShift` | `LS` | `Linear(n, 1, bias=False)`. `.weight` is the interpretable coefficient; no bias because the intercept slot owns the constant. |
| `ComplexShift` | `CS` | 64-128-64 ReLU MLP to one shift value. |
| `VaryingCoef` | `VC` | `beta0 + b_theta(mods)` with a zero-initialized output layer and the L2 hook `l2()`. `beta()` evaluates the effect, `recenter()` re-splits `beta0`/`b_theta` after training (function-preserving). |
| (`_mlp`) | — | The one MLP builder. Its module indices match the historical Sequentials, so old checkpoints load. |

## `flow.py` — the model

| Name | Role |
|---|---|
| `CausalFlowDAG` | The flow: one `_Node` per variable in topological order. Construction seeds the weights (`seed=` is the reproducibility knob). |
| `fit()` | Joint maximum likelihood with Adam, one parameter group per node (exact, because the NLL decomposes per node). Options: schedules (`plateau` decays per node), per-node freezing, `restore_best`, `marginal_init`, `vc_warm_start`, `plateau_factor`, `vc_oof_fit`. A second call continues training. |
| `fit_classical()` | Float64 full-batch L-BFGS for all-`ls` specs: deterministic, exact MLE, matches `statsmodels`/R `polr`. Refuses flexible specs. |
| `sample()` | Observational, interventional (`do=`, graph mutilation) and counterfactual (`u=`) sampling. |
| `abduct()` | Pearl step 1: recover the latents. Continuous exactly, ordinal by truncated draw. |
| `pmf()` | Analytic class probabilities of an ordinal node, with `do=` overrides. |
| `log_prob()` / `nll()` | Joint per-row log-likelihood / mean per-node NLL diagnostic. |
| `node_log_prob()` | The per-node decomposition everything trains and evaluates through. |
| `varying_coef()` | Closed-form read-out `beta(x)` of a fitted VC term. Deterministic, y-free. |
| `scores()` / `effect_modifier_scan()` | Analytic per-observation scores and the CUSUM modifier scan (delegate to `scores.py`). |
| `intercept_contributions()` | Post-hoc GAM-style decomposition of a complex intercept into mean-centered per-term parts. |
| `ls_coefficients()` | The per-node linear-shift weights — the interpretable coefficients. |
| `to_matrix()` | The labeled meta-adjacency matrix of term effects. |
| `save()` / `load()` | Checkpoints with history and machine provenance. 0.3 checkpoints load. |
| (`_Node`, `_VCGroup`) | Per-node module (intercept + shift `ModuleDict` + VC bookkeeping); `theta_shift()` computes `(theta, shift)`. |
| (`_encode_parent`, `_features`, `_tensorize`, `_dtype`, `_np_dtype`) | Parent encoding (continuous raw, ordinal one-hot), DataFrame → tensors, dtype plumbing. |
| (`_set_ranges`) | Train 5%/95% quantiles onto the transform domain, plus the optional marginal initialization. First fit only. |
| (`_vc_warm_start`, `_vc_oof_stage`, `_vc_oof_propensity`, `_predict_p1`, `_vc_ehat_live`, `_vc_ehat_columns`, `_recenter_vc`) | The VC machinery: classical `beta0` warm start; frozen out-of-fold propensities for training (DML); live full-fit propensities for inference (recomputed under `do`, never cached); post-fit re-centering. |
| (`_is_all_ls`) | Guard for `fit_classical`. |

## `scores.py` — effect-modifier detection (issue #29)

| Name | Role |
|---|---|
| `node_scores()` | Analytic, exact per-observation scores `psi_i = d l_i / d theta` for every `LS` weight and VC `beta0`. No autograd. |
| `effect_modifier_scan()` | Zeileis-Hornik fluctuation scan: order the treatment scores by each candidate, `sup|CUSUM|` against the Kolmogorov 5% value. A measured shortlist for VC modifiers from a seconds-long classical fit. |
| `sup_bb_pvalue()` | `P(sup |Brownian bridge| > stat)`, the Kolmogorov series. |
| (`_dl_ds`, `CRIT_5PCT`) | Closed-form latent-scale derivative; the 5% critical value 1.3581. |

## `env.py`

| Name | Role |
|---|---|
| `machine_info()` | Machine/software snapshot stored by `save()`, so timings stay comparable across machines. Never raises. |

## `simulations/` — numpy-only ground truth

Each generator is independent of the flow implementation and has a CLI that
regenerates its frozen `data/<name>/` CSVs (a test contract — never
regenerate silently). `REGISTRY` maps name → class.

| Class (module) | DGP | Ground-truth read-outs |
|---|---|---|
| `MagicMrClean` (`magic_mrclean`) | Synthetic stroke cohort, `ls`/`nl` variants | `true_ate()`, `counterfactual_pair()`, `observational()`, `rct()` |
| `TriangleContinuous` / `TriangleMixed` (`triangle`) | Paper §6 triangles, f variants linear/cubic/exp/atan/sin | `paper_truth()`, `zuko_expectations()`, `true_shift_curve()`, `true_pmf()` (mixed), `interventional()`, `counterfactual_pair()` |
| `VacaTriangle` (`vaca`) | App. C.1 bimodal Gaussian benchmark | `true_moments()` (analytic do-moments), `interventional()` |
| `Carefl4` (`carefl`) | App. C.2 Laplace SCM | `abduct_noise()`, `true_counterfactual()`, `true_cf_curves()` — all analytic |
| `VCLogisticShift` (`vc_shift`) | Issue #28 heterogeneous-effect DGP | `true_beta()` (known effect function), `counterfactual_pair()` |

## Where every training hyperparameter lives

Everything that shapes a fit is either a keyword you pass or a documented
default you can read at the call site. Nothing numeric is buried.

| Knob | Where | Default |
|---|---|---|
| epochs, learning rate, batch size | `fit()` | 500 / 1e-2 / 512 (in-repo examples always state them explicitly) |
| schedule, plateau patience, plateau decay factor | `fit(schedule=, plateau_patience=, plateau_factor=)` | None / 15 / 0.3 (lr floor `1e-3 * learning_rate`, stated in the docstring) |
| per-node freezing | `fit(freeze_patience=, min_delta=)` | off / 1e-4 (freeze guard `1e-2 * learning_rate`) |
| early stopping | `fit(restore_best=)` | False = exact MLE |
| calibrated init | `fit(marginal_init=)` | False (pure init, MLE unchanged) |
| VC warm start | `fit(vc_warm_start=)` | True (classical `beta0` start) |
| VC stage-1 proxy fits | `fit(vc_oof_fit=)` | `{"epochs": 300, "learning_rate": 1e-2, "batch_size": 512}` |
| VC penalty and centering | `VC(penalty=, center=, center_folds=)` | 1.0 / False / 5 |
| L-BFGS budget | `fit_classical(max_iter=, tol=, chunk=, history_size=)` | 400 / 1e-6 / 25 / 50 |
| network widths | `units=` on `I`/`CS`/`VC` | (8, 8) / (64, 128, 64) / (16,) — Keras-parity architecture defaults |
| transform basis | `I(transform=, transform_kwargs=)` | `"bernstein"`, `n_coeffs=20`; spline `bins=8`; domain bound 5.0 |
| shuffling / weight init | `fit(seed=)` / `CausalFlowDAG(seed=)` | init happens at construction — the constructor seed is the reproducibility knob |
