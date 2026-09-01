# Tests — what they guarantee

This file gives an overview of the tests and how to run them.

## Running the tests

```bash
uv run pytest tests/ -q            # everything (the slow fits dominate; ~25-40
                                   #   min on 2-core CI, less on a workstation)
uv run pytest tests/ -q -m "not slow"   # fast subset (~2-3 min) — unit + contracts
uv run pytest tests/test_flow.py -q     # one file
```

- **`slow` marker** — five long fits carry `@pytest.mark.slow`, which
  `-m "not slow"` skips. It is not "everything that trains a flow": a feature's
  acceptance number (the `VC` recovery bar, the centering bias reduction) trains
  one deliberately in the fast subset, so every run measures it.
- **CI** ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)) runs the fast
  subset on every pull request and on pushes to `main` and `dev-*` (a feature-branch push with no open PR runs `pre-commit` and `experiments`, not `ci`), and the **full** suite
  nightly and on demand (Actions → CI → *Run workflow*). The split exists
  because the full suite is ~25–40 min on the 2-core runners. (No count here:
  `pytest --collect-only -q | tail -1` is always right, a number in prose goes
  stale within a week — this line has been wrong twice.)
- **Determinism** — tests seed `torch` before constructing the flow (weight init
  happens at construction), so fits are reproducible.

## Testing principles

Five kinds of test, in rough order of how much trust they carry:

1. **Known mathematical identities** — properties that must hold by the math,
   independent of any reference implementation:
   - the monotone transforms invert exactly (`test_univariate_roundtrip`);
   - abduction → push-forward reproduces the data (bijective round-trip,
     `test_abduction_roundtrip`);
   - the joint `log_prob` equals the sum of per-node terms
     (`test_log_prob_finite_and_decomposes`);
   - a `do` intervention changes only descendants
     (`test_counterfactual_only_changes_descendants`).

2. **Equivalence to independent implementations** — the strongest external check:
   an all-`ls` model *is* a classical transformation model, so it must match
   software written by other people in other languages.
   - vs. **`statsmodels`** `OrderedModel` (computed at test time):
     `test_ls_node_equals_proportional_odds`, `test_matches_statsmodels_mle`,
     `test_torch_plateau_scheduler_preserves_exact_mle`;
   - the two optimizers agree on the same optimum: `test_agrees_with_adam_mle`.

3. **Known-truth recovery** — because the inline DGPs (see `conftest.py`) *are*
   the ground truth, we can check the flow recovers quantities no real dataset
   would expose:
   - the true linear-shift coefficients of the all-`ls` chain
     (`test_continuous_only_all_ls_recovers_the_true_shift`,
     `test_matches_statsmodels_mle`);
   - the pointwise effect function `beta(x)` of the heterogeneous-effect DGP
     (`test_recovery_bar_on_hetero_dgp`, the corr >= 0.9 acceptance bar);
   - the bias reduction propensity centering exists for
     (`test_dandl_centering_reduces_bias`).

   The **research** DGPs (the paper replications and their frozen CSVs) are not
   tested here: they live in [`experiments/`](../experiments/) and are checked by
   the experiments workflow against its committed ground truth.

4. **Numerical-stability & invariant guards** — regressions we've been bitten by:
   - the ordinal log-likelihood keeps non-zero gradients under float32 saturation
     (`test_ordinal_log_prob_gradient_survives_saturation`) — the naive sigmoid
     difference would freeze a node at init;
   - cutpoints stay increasing and PMFs sum to one
     (`test_ordinal_cutpoints_increasing_and_pmf_sums_to_one`);
   - `save`/`load` round-trips a fitted model (`test_save_load_roundtrip`);
   - a schedule through the hooks doesn't break the exact-MLE property
     (`test_torch_plateau_scheduler_preserves_exact_mle`);
   - DAG validation catches cycles and orders correctly
     (`test_cycle_detected`, `test_topological_order`).

## How the ground truth is obtained

The reference values the tests compare against come from two sources:

- **By construction.** The three inline DGPs in `conftest.py` are *built as*
  transformation models with fixed coefficients, cutpoints and effect
  functions, so the true parameters are simply the numbers used to generate
  the data. They are numpy-only and deliberately independent of the flow.
- **Independent software.** The classical-equivalence tests fit `statsmodels`
  `OrderedModel` at test time on the same design matrix the flow builds. An
  all-`ls` outcome node *is* an ordered-logit model, so this is an equality
  claim against software written by other people, not a tuned similarity.

## The test files

`conftest.py` holds the three **inline DGPs** (all-`ls` chain,
heterogeneous effect, confounded-with-misfit) plus helpers that are
provably identical across modules. Specs stay per-module: each pins the
one syntax variant its property needs, and sharing them would couple
unrelated acceptance bars.

| file | what it covers |
|---|---|
| [`test_flow.py`](test_flow.py) | core unit tests — transforms, ordinal log-prob, DAG validation, abduction/counterfactual mechanics, `save`/`load`, the proportional-odds identity |
| [`test_fit_hooks.py`](test_fit_hooks.py) | `fit(optimizer=, callbacks=)` the Keras-shaped validation/verbose options and the shipped `tramdag.callbacks` — stop, lists, `EarlyStopping`, `PerNodePlateau`, a torch scheduler; the guard that it still lands on the MLE |
| [`test_density.py`](test_density.py) | `density()` integrates to one and matches sampling |
| [`test_input_transform.py`](test_input_transform.py) | per-term `input_transform=` — minmax/standardize statistics frozen at calibrate, the callable's train column on a fresh batch, lambda save rejection, checkpoint round-trip |
| [`test_fit_classical.py`](test_fit_classical.py) | `fit_classical` — guard on non-`ls` specs, determinism, float64 round-trip, agreement with `statsmodels` and Adam |
| [`test_spec_terms.py`](test_spec_terms.py) | term constructors, edge ownership, the meta-adjacency view |
| [`test_transformation_syntax.py`](test_transformation_syntax.py) | the formula syntax — every spelling normalizes identically, the constructor aliases, `units=`, round-trips, and rejection of a malformed serialized spec |
| [`test_joint_terms.py`](test_joint_terms.py) | joint multi-parent CS/I terms |
| [`test_additive_ci.py`](test_additive_ci.py) | the additive intercept (`allow_interaction=False`) |
| [`test_intercept_contributions.py`](test_intercept_contributions.py) | the post-hoc GAM decomposition of complex intercepts |
| [`test_vc_term.py`](test_vc_term.py) | the VC effect head — spec, penalty, recovery of `beta(x)` |
| [`test_vc_centered.py`](test_vc_centered.py) | propensity-centered VC — out-of-fold structure, zero-gradient freeze, bias reduction |
| [`test_scores.py`](test_scores.py) | analytic scores vs finite differences, the effect-modifier scan |
| [`test_marginal_init.py`](test_marginal_init.py) | calibrated marginal initialization — pure-init property |
| [`test_api_papercuts.py`](test_api_papercuts.py) | error messages, `save`/`load` meta, small API contracts |

## Adding tests

- New causal features should be validated against an inline DGP's **known
  truth** (add one to `conftest.py` if none fits), not just "runs without
  error".
- Mark a long fit `@pytest.mark.slow` so PR CI stays fast — unless it *is* the
  acceptance measurement for a feature, which is worth having on every run.
- Framework tests must not depend on `experiments/`: the research generators
  and their frozen CSVs live there and are checked by the experiments
  workflow — see the testing policy in [`CLAUDE.md`](../CLAUDE.md).
