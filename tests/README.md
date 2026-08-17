# Tests — what they guarantee

This file gives an overview of the tests and how to run them.

## Running the tests

```bash
uv run pytest tests/ -q            # everything (the slow fits dominate; ~25-40
                                   #   min on 2-core CI, less on a workstation)
uv run pytest tests/ -q -m "not slow"   # fast subset (~2-3 min) — unit + contracts
uv run pytest tests/test_flow.py -q     # one file
```

- **`slow` marker** — the long fit/training tests (they train flows) are marked
  `@pytest.mark.slow`. `-m "not slow"` skips them.
- **CI** ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)) runs the fast
  subset on every pull request and push to `main`, and the **full** suite nightly
  and on demand (Actions → CI → *Run workflow*). The split exists because the full
  suite is ~25–40 min on the 2-core runners.
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
   - each triangle DGP really is a TRAM — the reconstructed latent is standard
     logistic by a KS test (`test_triangle_continuous_tram_identity`);
   - a `do` intervention changes only descendants
     (`test_counterfactual_only_changes_descendants`).

2. **Equivalence to independent implementations** — the strongest external check:
   an all-`ls` model *is* a classical transformation model, so it must match
   software written by other people in other languages.
   - vs. **`statsmodels`** `OrderedModel` (computed at test time):
     `test_all_ls_flow_is_exact_mle`, `test_ls_node_equals_proportional_odds`,
     `test_matches_statsmodels_mle`;
   - vs. **R** `MASS::polr` / `tram::Colr` / `glm` (committed reference, see below):
     `test_flow_matches_r_reference`;
   - the two optimizers agree on the same optimum: `test_agrees_with_adam_mle`.

3. **Known-truth recovery** — because the simulators *are* the ground truth, we
   can check the flow recovers quantities no real dataset would expose:
   - the true ATE from the SCM (`test_flow_recovers_true_ate`,
     `test_true_ate_positive_and_confounded`);
   - the paper DGPs' true coefficients / shift curves / cutpoints
     (`test_triangle_*_recovers_*`);
   - interventional moments (`test_vaca_ci_flow_matches_interventional_moments`)
     and individual counterfactuals
     (`test_carefl_ci_flow_recovers_counterfactuals`);
   - the headline storyline: all-`ls` underestimates, flexible recovers
     (`test_nl_storyline_all_ls_underestimates_flexible_recovers`).

4. **Frozen-data contracts** — the committed CSVs under `data/` are a contract,
   not a convenience. Tests assert each regenerates **bit-identically** from its
   stored seed, so the data can never drift silently underneath the truth files
   (`test_frozen_csvs_present_and_match_truth`, `test_frozen_csv_contract`).

5. **Numerical-stability & invariant guards** — regressions we've been bitten by:
   - the ordinal log-likelihood keeps non-zero gradients under float32 saturation
     (`test_ordinal_log_prob_gradient_survives_saturation`) — the naive sigmoid
     difference would freeze a node at init;
   - cutpoints stay increasing and PMFs sum to one
     (`test_ordinal_cutpoints_increasing_and_pmf_sums_to_one`);
   - `save`/`load` round-trips a fitted model (`test_save_load_roundtrip`);
   - schedules/freezing don't break the exact-MLE property
     (`test_plateau_freeze_preserves_exact_mle`);
   - DAG validation catches cycles and orders correctly
     (`test_cycle_detection`, `test_topological_order`).

## How the ground truth is obtained

The reference values the tests compare against come from four sources, in
decreasing order of exactness:

- **By construction.** The paper DGPs are *built as* TRAMs with fixed
  coefficients, cutpoints and shift functions
  ([`simulations/triangle.py`](../src/tramdag/simulations/triangle.py)), so the
  true parameters are simply the numbers used to generate the data.
- **Analytically.** Where the SCM allows a closed form: the VACA benchmark's
  interventional moments (Gaussian; `E[x₃ | do(x₂=a)] = −0.25 + 0.25a`) and the
  CAREFL benchmark's individual counterfactuals (additive Laplace noise recovered
  exactly by abduction) — [`simulations/vaca.py`](../src/tramdag/simulations/vaca.py),
  [`simulations/carefl.py`](../src/tramdag/simulations/carefl.py).
- **Monte Carlo from the SCM.** Where no closed form exists, the truth is a
  large-sample estimate from the generator itself — e.g. the stroke cohort's true
  ATE is the do-effect averaged over a 500k-row interventional draw
  (`MagicMrClean.true_ate`), stored in each `truth.json`. The numpy simulators are
  deliberately independent of the flow, so this is a genuine external reference.
- **Independent software.** The classical-equivalence tests fit `statsmodels`
  `OrderedModel` at test time, and compare against an **R** reference
  (`MASS::polr` / `tram::Colr` / `glm`, script
  [`data/magic-mrclean/fit_ls.R`](../data/magic-mrclean/fit_ls.R)) whose outputs
  are committed under `data/magic-mrclean/*/ref_ls/` so the tests run without an R
  installation. Details: [`data/magic-mrclean/README.md`](../data/magic-mrclean/README.md).

## The test files

| file | what it covers |
|---|---|
| [`test_flow.py`](test_flow.py) | core unit tests — transforms, ordinal log-prob, DAG validation, abduction/counterfactual mechanics, `save`/`load`, the proportional-odds identity |
| [`test_simulations.py`](test_simulations.py) | the stroke simulator — schema/determinism, RCT de-confounding, known-truth ATE recovery, exact-MLE vs `statsmodels`, the R-reference regression |
| [`test_paper_dgps.py`](test_paper_dgps.py) | the paper DGPs — generator pinning (KS identities, frozen-CSV contract, analytic ground truth) + flow recovery of each family's truth |
| [`test_fit_schedules.py`](test_fit_schedules.py) | lr schedules and per-node freezing, incl. the guard that plateau+freeze still lands on the MLE |
| [`test_fit_classical.py`](test_fit_classical.py) | `fit_classical` — guard on non-`ls` specs, determinism, float64 round-trip, agreement with `statsmodels` and Adam |
| [`test_spec_terms.py`](test_spec_terms.py) | term constructors, edge ownership, the meta-adjacency view |
| [`test_transformation_syntax.py`](test_transformation_syntax.py) | the 0.4 formula syntax — every spelling normalizes identically, 0.3-dict shims, `units=` |
| [`test_joint_terms.py`](test_joint_terms.py) | joint multi-parent CS/I terms |
| [`test_additive_ci.py`](test_additive_ci.py) | the additive intercept (`allow_interaction=False`) |
| [`test_intercept_contributions.py`](test_intercept_contributions.py) | the post-hoc GAM decomposition of complex intercepts |
| [`test_vc_term.py`](test_vc_term.py) | the VC effect head — spec, penalty, warm start, recovery of `beta(x)` |
| [`test_vc_centered.py`](test_vc_centered.py) | propensity-centered VC — out-of-fold structure, zero-gradient freeze, bias reduction |
| [`test_scores.py`](test_scores.py) | analytic scores vs finite differences, the effect-modifier scan |
| [`test_marginal_init.py`](test_marginal_init.py) | calibrated marginal initialization — pure-init property |
| [`test_api_papercuts.py`](test_api_papercuts.py) | error messages, `save`/`load` meta, small API contracts |

## Adding tests

- New causal features should be validated against a simulator's **known truth**
  (`MagicMrClean.true_ate`; `counterfactual_pair` gives true individual
  counterfactuals via shared latents), not just "runs without error".
- Mark anything that trains a flow `@pytest.mark.slow` so PR CI stays fast.
- Treat `data/` as frozen: a new seed or new equations means a **new folder**
  (regenerate any R reference), never an in-place edit — see the project testing
  policy in [`CLAUDE.md`](../CLAUDE.md).
