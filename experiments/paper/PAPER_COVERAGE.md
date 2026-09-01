# Paper coverage — what of arXiv:2503.16206 is reproduced here

Checked figure by figure against the paper (v1). "Schematic" means the figure
is a drawing of a model or DAG, not a result: nothing to compute.

## Main paper

| Figure | Subject | Reproduced by |
|---|---|---|
| 1 | TRAM-DAG architecture | schematic |
| 2 | DAG with noise, and the post-interventional DAG | schematic |
| 3 | Meta-adjacency matrix specification | schematic — but the object itself is `flow.to_matrix()` |
| 4 | VACA observational joint: DGP vs TRAM-DAG vs CNF | `vaca.py flexible` → `plots/pairs.png` (DGP vs flow; **the CNF baseline is not reimplemented**, see gaps) |
| 5 | VACA interventional `p(x3 \| do(x2))` | `vaca.py flexible` → `plots/interventional.png`, scored against the analytic means |
| 6 | CAREFL counterfactual queries | `carefl.py flexible` → `plots/cf_curves.png`, scored against the analytic counterfactuals |
| 7 | Continuous triangle DAG + the complex shift for `atan` | `triangle.py atan-cs` → `plots/cs_curve.png` (left panel schematic) |
| 8 | Mixed-data DAG | schematic |
| 9 | Mixed data, observational and interventional | `triangle_mixed.py linear-ls` → `plots/distributions.png` |

## Appendix

| Figure | Subject | Reproduced by |
|---|---|---|
| 10 | Why counterfactuals fail for interval-censored discrete variables | schematic (App. B) — the measured version is gap 2 below: `triangle_mixed.py` → `plots/counterfactual_pmf.png`, scored against the analytic counterfactual law |
| 11 | DAG of the original VACA DGP | schematic |
| 12 | VACA observational fit with a Neural Spline Flow | **not reproduced** — a competing method, see gaps |
| 13 | DAG of the four-variable counterfactual experiment | schematic |
| 14 | Coefficient convergence, linear-shift continuous case | `triangle.py linear-ls` → `plots/coefficients.png` |
| 15 | Coefficient convergence with one CS term | `triangle.py atan-cs` → `plots/coefficients.png` |
| 16 | Observational and interventional match, CS case | `triangle.py atan-cs` → `plots/distributions.png` |
| 17 | Misspecified model: fitted CS vs the true linear function | `triangle.py linear-cs` → `plots/cs_curve.png` + `plots/distributions.png` |
| 18 | Non-monotone DGP `f(x2) = 2 sin(3 x2) + x2` | `triangle.py sin-cs` → `plots/coefficients.png`, `plots/cs_curve.png` |
| 19 | Coefficient estimates, mixed case with linear shifts | `triangle_mixed.py linear-ls` → `plots/coefficients.png` |
| 20 | Mixed case with an exponential shift function | `triangle_mixed.py exp-cs` → `plots/distributions.png` |

Numerical results beyond the figures: the App. C.4 odds-ratio check is computed
by `triangle_mixed.py` (predicted `exp(beta12)` against the odds ratio measured
in the DGP under `do(x1 += 1)`, theory `e² ≈ 7.39`).

## Gaps, and why

1. **Competing methods are not reimplemented.** Figure 4's CNF panel and
   Figure 12's Neural Spline Flow are baselines from other libraries
   (Causal Normalizing Flows, `zuko`'s NSF). This repository is the TRAM-DAG
   implementation; it reproduces the TRAM-DAG side of each comparison and the
   ground truth both sides are measured against. Reproducing the baselines
   would mean vendoring two more model families.
2. **Figure 10's point is now measured, not just argued.** It illustrates why
   an individual counterfactual is not identified for a discretized variable.
   `triangle_mixed.py` turns that into a number: an observed ordinal level pins
   the latent to an interval, so the generator can state the exact
   counterfactual *distribution* (`TriangleMixed.true_counterfactual_pmf`, itself
   checked against realised counterfactuals in `paper/tests/`), and the flow's
   averaged abduction draws are scored against that distribution rather than against
   a level no model could predict.
3. **Section 7's application** (the clinical case study) is not here: that data
   is private and its storyline lives in its own repository.
