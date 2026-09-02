# Code map — every module of `src/tramdag/` — the public names and the private plumbing

One entry per name, with its role and its place in the pipeline. Names in
parentheses are private machinery: useful to know, not part of the API.
The last section lists every training hyperparameter and where it lives.

## `spec.py` — declare the model

Every term constructor has a pythonic name and a short alias; the two
are the same object, so `LS is linear_shift`.

| Name | Role |
|---|---|
| [`Term`][tramdag.spec.Term] | One additive term of a node's transformation: the frozen triple `(effect, parents, options)`. `+` on terms builds plain lists. Effect-specific settings live in `options` and read as attributes (`term.penalty`, `term.units`, ...). |
| [`simple_intercept()`][tramdag.spec.simple_intercept] / [`SI`][tramdag.spec.simple_intercept] | The parentless intercept — the paper's SI. Free transform parameters, the same for every row. Carries the basis choice (`transform=`, default `"bernstein"`); extra keyword arguments pass straight to the transform class. |
| [`complex_intercept()`][tramdag.spec.complex_intercept] / [`CI`][tramdag.spec.complex_intercept] | The parent-conditioned intercept — the paper's CI: the parents reshape the monotone transform. Needs at least one parent. Also carries `units=` and `allow_interaction=` (joint vs. additive multi-parent intercept). |
| [`intercept()`][tramdag.spec.intercept] / [`I`][tramdag.spec.intercept] | The fallback: dispatches on its arguments to `SI` (no parents) or `CI` (parents). The bare names `I` and `SI` in a term list both mean the simple intercept. |
| [`linear_shift()`][tramdag.spec.linear_shift] / [`LS`][tramdag.spec.linear_shift] | Linear shift `beta * x` — the interpretable log-odds coefficient. Exactly one parent. |
| [`complex_shift()`][tramdag.spec.complex_shift] / [`CS`][tramdag.spec.complex_shift] | Complex shift: an NN `g(x)`, additive on the latent scale. Several parents form one joint network. |
| [`varying_coefficient()`][tramdag.spec.varying_coefficient] / [`VC`][tramdag.spec.varying_coefficient] | Varying-coefficient shift `(beta0 + b_theta(mods)) * x_t` — the penalized treatment-effect head (issue #28). `center=` adds propensity centering (issue #30). |
| [`ContinuousNode`][tramdag.spec.ContinuousNode] | Continuous variable: monotone 1-D transform plus shifts. `terms` is the first positional argument. |
| [`OrdinalNode`][tramdag.spec.OrdinalNode] | Ordinal variable with `levels` classes: ordered logit (cutpoints) plus shifts. |
| [`node_parents()`][tramdag.spec.node_parents] | Ordered de-duplicated parent names of a node (the canonical term list is `node.terms`). |
| [`validate_and_sort()`][tramdag.spec.validate_and_sort] | Edge-ownership validation plus Kahn topological sort. The returned order makes the flow triangular. |
| [`spec_to_dict()`][tramdag.spec.spec_to_dict] / [`spec_from_dict()`][tramdag.spec.spec_from_dict] | Checkpoint (de)serialization. A term serializes as `{effect, parents, options}` and nothing else, since `options` is already canonical. No compatibility shims: `spec_from_dict` rejects a term without `options`, the node constructors normalize the formula, and `validate_and_sort` checks the DAG. |
| (`_normalize_terms`, `_as_term`, `_intercept_basis`, `_options`) | Formula flattening and per-entry validation (a `+` sum nested in a list is rejected), the one-parented-`I` rule plus basis hoisting in one pass, canonical option storage against each effect's `option_defaults` (a wrong-effect option errors). |
| (`_check_term`, `_check_node`, `_check_input_transform`, `_kahn_sort`) | The stages behind `validate_and_sort`: the generic checks (effect known, parents exist, `input_transform` shape) plus each entry's own rules (LS arity, the VC treatment/centering block — on the term classes), edge-ownership bookkeeping, the topological sort. |

## `transforms.py` — the monotone map h and the ordinal transform

| Name | Role |
|---|---|
| [`StandardLogistic`][tramdag.transforms.StandardLogistic] | The TRAM base distribution: `log_prob`, `sample` (generator-aware), `icdf`. |
| [`BernsteinUT`][tramdag.transforms.BernsteinUT] | Bernstein-polynomial transform (default basis, `n_coeffs=20`). Linear tail extrapolation follows the boundary derivative. `marginal_init_theta()` gives the calibrated start used by `calibrate(marginal_init=True)`. |
| [`SplineUT`][tramdag.transforms.SplineUT] | Monotone rational-quadratic spline (`bins=8`). Tails extrapolate with a *fixed* slope — the structural reason spline trails Bernstein on tail-heavy data. |
| [`AffineUT`][tramdag.transforms.AffineUT] | Monotone affine transform: the node-conditional is a logistic GLM. |
| [`make_univariate_transform()`][tramdag.transforms.make_univariate_transform] | Basis registry: name → transform instance. |
| [`ordinal_cutpoints()`][tramdag.transforms.ordinal_cutpoints] | Unconstrained `(n, K-1)` → increasing cutpoints with ±inf ends. Port of the original parametrization. |
| [`ordinal_log_prob()`][tramdag.transforms.ordinal_log_prob] | `log P(Y=y)`, computed in log-space. Load-bearing: the naive sigmoid difference saturates in float32 and freezes nodes at init. Do not simplify. |
| [`ordinal_pmf()`][tramdag.transforms.ordinal_pmf] / [`ordinal_sample()`][tramdag.transforms.ordinal_sample] / [`ordinal_abduct()`][tramdag.transforms.ordinal_abduct] | Class probabilities / latent → level / truncated-logistic latent recovery (Pearl step 1) for ordinal nodes. |
| [`ordinal_marginal_init_theta()`][tramdag.transforms.ordinal_marginal_init_theta] | Cutpoint start that matches the empirical class frequencies (`calibrate(marginal_init=True)`). |
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
| [`SimpleIntercept`][tramdag.conditioners.SimpleIntercept] | bare `I` | Free parameter vector; no parents. |
| [`ComplexIntercept`][tramdag.conditioners.ComplexIntercept] | `I(...)` | 8-8 ReLU NN from parent features to the transform parameters. |
| [`LinearShift`][tramdag.conditioners.LinearShift] | `LS` | `Linear(n, 1, bias=False)`. `.weight` is the interpretable coefficient; no bias because the intercept slot owns the constant. |
| [`ComplexShift`][tramdag.conditioners.ComplexShift] | `CS` | 64-128-64 ReLU NN to one shift value. |
| [`VaryingCoef`][tramdag.conditioners.VaryingCoef] | `VC` | `beta0 + b_theta(mods)` with a zero-initialized output layer and the L2 hook `l2()`. `beta()` evaluates the effect, `recenter()` re-splits `beta0`/`b_theta` after training (function-preserving). |
| (`_nn`) | — | The one NN builder: a stack of the given `units` with the term's `activation` (relu by default), then a bias-free output layer. |

## `flow.py` — the model

| Name | Role |
|---|---|
| [`CausalFlowDAG`][tramdag.flow.CausalFlowDAG] | The flow: one `_Node` per variable in topological order. Construction seeds the weights (`seed=` is the reproducibility knob). |
| [`calibrate()`][tramdag.flow.CausalFlowDAG.calibrate] | Once, from the training rows: transform ranges (train 5%/95% quantiles onto the domain), network-input min-max, the calibrated start (`marginal_init=True`). Called by the first fit; a checkpoint carries the flag. |
| [`init_marginals()`][tramdag.flow.CausalFlowDAG.init_marginals] | The calibrated start as an explicit step, callable any time: resets every simple intercept to its column's marginal (Bernstein map / ordinal class log-odds; spline and affine have no calibrated start). Not once-guarded — on a trained flow it restarts those intercepts. Calibrates a fresh flow's ranges itself. |
| [`fit()`][tramdag.flow.CausalFlowDAG.fit] | Joint maximum likelihood: one minibatch Adam loop over all parameters (exact per node, because the NLL decomposes), final weights kept. Keras-shaped validation (`validation_data=`/`validation_split=` fill `history["val"]` per epoch, `validation_batch_size=` chunks the pass) and progress (`verbose=`). Hooks: `optimizer=` (any torch optimizer, for schedulers) and `callbacks=` (one entry or a list; a `Callback` hooks `on_fit_begin`/`on_epoch_end`/`on_fit_end`, a bare callable is an `on_epoch_end` hook `cb(flow, epoch, opt)` — any `True` stops); the recipes in `callbacks.py` read `history["val"]`. `vc_ehat=` carries the out-of-fold propensities of centered VC terms. A second call continues training. |
| [`fit_classical()`][tramdag.flow.CausalFlowDAG.fit_classical] | Float64 full-batch L-BFGS for all-`ls` specs: deterministic, exact MLE, matches `statsmodels`/R `polr`. Refuses flexible specs. |
| [`sample()`][tramdag.flow.CausalFlowDAG.sample] | Observational, interventional (`do=`, graph mutilation) and counterfactual (`u=`) sampling. |
| [`abduct()`][tramdag.flow.CausalFlowDAG.abduct] | Pearl step 1: recover the latents. Continuous exactly, ordinal by truncated draw. |
| [`pmf()`][tramdag.flow.CausalFlowDAG.pmf] | Analytic class probabilities of an ordinal node, with `do=` overrides. |
| [`density()`][tramdag.flow.CausalFlowDAG.density] | Analytic conditional density of a continuous node on a grid, with `do=` overrides — the continuous counterpart of `pmf`. |
| [`log_prob()`][tramdag.flow.CausalFlowDAG.log_prob] / [`nll()`][tramdag.flow.CausalFlowDAG.nll] | Joint per-row log-likelihood / mean per-node NLL diagnostic. |
| [`node_log_prob()`][tramdag.flow.CausalFlowDAG.node_log_prob] | The per-node decomposition everything trains and evaluates through. |
| [`varying_coef()`][tramdag.flow.CausalFlowDAG.varying_coef] | Closed-form read-out `beta(x)` of a fitted VC term. Deterministic, y-free. |
| [`scores()`][tramdag.flow.CausalFlowDAG.scores] / [`effect_modifier_scan()`][tramdag.flow.CausalFlowDAG.effect_modifier_scan] | Analytic per-observation scores and the CUSUM modifier scan (delegate to `scores.py`). |
| [`intercept_contributions()`][tramdag.flow.CausalFlowDAG.intercept_contributions] | Post-hoc GAM-style decomposition of a complex intercept into mean-centered per-term parts. |
| [`ls_coefficients()`][tramdag.flow.CausalFlowDAG.ls_coefficients] | The per-node linear-shift weights — the interpretable coefficients. |
| [`design_matrix()`][tramdag.flow.CausalFlowDAG.design_matrix] | Parent encoding as a DataFrame (`drop_first=` gives the classical statsmodels/`polr` design). |
| [`to_matrix()`][tramdag.flow.CausalFlowDAG.to_matrix] | The labeled meta-adjacency matrix of term effects. |
| [`save()`][tramdag.flow.CausalFlowDAG.save] / [`load()`][tramdag.flow.CausalFlowDAG.load] | Checkpoints with history and provenance (version, time, device). `load` requires a complete checkpoint and fails loudly otherwise. |
| (`_node`, `_encode_parent`, `_features`, `_tensorize`, `_generator`, `_dtype`, `_np_dtype`) | Node lookup with one shared error; parent encoding (continuous raw, ordinal one-hot); `_tensorize(df, cols=None)` for any column subset; seeded-generator and dtype plumbing. |
| (`_vc_ehat_train`, `_binary_p1`, `_vc_ehat_live`, `_vc_ehat_columns`, `_recenter_vc`) | The generic side-input plumbing (each term declares/validates/recomputes its own inputs via the `ShiftTerm` hooks) plus the binary propensity fit and the post-fit `finalize` loop. |
| (`_is_classical`) | Guard for `fit_classical`: every term's `term_is_classical` — `LS`, or a parentless `I()` basis carrier. |


## `terms.py` — the effect registry (the 1.0 architecture's core)

One `TermDef` per effect; see [architecture.md](architecture.md) for the
contract diagram.

| Name | Role |
|---|---|
| [`register_term()`][tramdag.terms.register_term] / [`get_term()`][tramdag.terms.get_term] | The registry: custom effects register a `ShiftTerm` subclass under a new effect name; collisions refuse. |
| [`ShiftTerm`][tramdag.terms.ShiftTerm] / [`InterceptTerm`][tramdag.terms.InterceptTerm] | The behavior hooks a term owns: validation, `build`, `shift_value`/`theta_value`, `post_init`, `regularizer`, post-fit `finalize`, `score_columns`, the side-input contract, `cells`, `term_is_classical`, `option_defaults`. |
| [`LSTerm`][tramdag.terms.LSTerm] / [`CSTerm`][tramdag.terms.CSTerm] / [`VCTerm`][tramdag.terms.VCTerm] / [`FnTerm`][tramdag.terms.FnTerm] | The built-in shift terms, subclassing their conditioners (state-dict paths and the seeded RNG stream stay bit-stable). `VCTerm.regressor` is both the forward regressor and the `beta0` score. |
| [`SITerm`][tramdag.terms.SITerm] / [`CITerm`][tramdag.terms.CITerm] / [`AdditiveCITerm`][tramdag.terms.AdditiveCITerm] | The intercept slot: free theta, one joint net, or one net per parent summed in coefficient space. |

## `nodes.py` — the node model

| Name | Role |
|---|---|
| (`_Node`) | One sub-model per variable: builds its intercept and shift terms via the registry; `theta_shift()` sums the terms' `shift_value`s (plain shifts first, then VC); `net_input()` feeds every term network, `input_transform` applied. |
| (`_InputTransform`) | One term's frozen network-input transform (minmax / standardize / callable over frozen train columns). |
| (`kind_log_prob`, `kind_sample`, `kind_abduct`, `kind_marginal_theta`) | The ONLY continuous-vs-ordinal branches in the package, adjacent. |
| (`_init_linear`) | Keras' `glorot`/`normal` initializers on one linear layer. |

## `fitting.py` — `_FitMixin`

| Name | Role |
|---|---|
| [`fit()`][tramdag.flow.CausalFlowDAG.fit] / [`fit_classical()`][tramdag.flow.CausalFlowDAG.fit_classical] | Defined here once, methods of the flow via the mixin. |
| (`_split_validation`, `_slice_vc_ehat`, `_slice_ehat`, `_normalize_callbacks`, `_check_epoch_hook`, `_check_fit_sizes`, `_epoch_pass`, `_log_epoch`, `_val_nll`, `_fit_epoch`, `_FnCallback`) | The loop plumbing: Keras-shaped validation split, callback normalization and pre-fit signature checks, the epoch/validation passes, verbose printing. |

## `readouts.py` — `_ReadoutsMixin`

| Name | Role |
|---|---|
| [`shift_curve()`][tramdag.flow.CausalFlowDAG.shift_curve] | One fitted shift term on a 1-D grid, through the term's own `shift_value` — the public replacement for reaching into `nd.shifts[..]`. |
| the read-out methods | `varying_coef`, `ls_coefficients`, `to_matrix`, `intercept_contributions`, `design_matrix` — defined here once, methods of the flow via the mixin. |

## `scores.py` — effect-modifier detection (issue #29)

| Name | Role |
|---|---|
| [`node_scores()`][tramdag.scores.node_scores] | Analytic, exact per-observation scores `psi_i = d l_i / d theta` for every `LS` weight and VC `beta0`. No autograd. |
| [`effect_modifier_scan()`][tramdag.scores.effect_modifier_scan] | Zeileis-Hornik fluctuation scan: order the treatment scores by each candidate, `sup|CUSUM|` against the Kolmogorov 5% value. A measured shortlist for VC modifiers from a seconds-long classical fit. |
| [`sup_bb_pvalue()`][tramdag.scores.sup_bb_pvalue] | `P(sup |Brownian bridge| > stat)`, the Kolmogorov series. |
| (`_dl_ds`, `CRIT_5PCT`) | Closed-form latent-scale derivative; the 5% critical value 1.3581. The per-term columns come from each term's `score_columns` hook. |

## `callbacks.py` — the shipped `fit` callbacks

| Name | Role |
|---|---|
| [`EarlyStopping`][tramdag.callbacks.EarlyStopping] | Snapshots the weights of the best summed validation NLL (read from `history["val"]`) and restores them automatically at fit end, before the VC re-centering; an optional `patience` also stops the fit once the best is that many epochs old. |
| [`PerNodePlateau`][tramdag.callbacks.PerNodePlateau] | Per-node lr decay and freezing on each node's own validation NLL (from `history["val"]`); stops the fit once every node froze. The pre-0.4 `fit(schedule="plateau")` recipe, opt-in. `step(nll, opt)` for a hand-computed NLL. |
| [`per_node_adam()`][tramdag.callbacks.per_node_adam] | Adam with one `node`-tagged parameter group per node — the optimizer `PerNodePlateau` needs. |

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
| validation, progress | `fit(validation_data=, validation_split=, validation_batch_size=, verbose=)` | per-node val NLL into `history["val"]` each epoch; `verbose=N` prints every Nth + final epoch (default 0, silent) |
| schedules, early stopping | `fit(optimizer=, callbacks=)` | `tramdag.callbacks` ships `EarlyStopping`, `PerNodePlateau`; anything else is torch's `lr_scheduler` and a few lines of callback ([fitting.md](fitting.md)) |
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
| network inputs | `CI/CS/VC(input_transform=)` | `None` (raw parents); `"minmax"` / `"standardize"` transform that term's continuous parents with statistics frozen at `calibrate`, and a callable `fn(x, train)` gets the frozen raw train column — LS and the VC treatment stay raw |
