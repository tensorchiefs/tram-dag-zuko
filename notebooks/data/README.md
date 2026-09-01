# Notebook data

Small datasets the notebooks read. Anything larger, or tied to an experiment
rather than a notebook, belongs in `experiments/<area>/data/` instead.

## `birthwt.csv`

Four columns of `MASS::birthwt` — the low-birth-weight study of Hosmer &
Lemeshow (1989), 189 births at Baystate Medical Center — used by
`classical_fit_tram_dag.py` as a logistic-regression example that a reader can
re-fit in R.

| column | meaning |
|---|---|
| `low` | birth weight below 2.5 kg (the outcome, 0/1) |
| `age` | mother's age in years |
| `lwt` | mother's weight at last menstrual period, lbs |
| `smoke` | smoking during pregnancy (0/1) |
| `bwt` | birth weight in grams — the continuous outcome `low` is cut from |

Exported verbatim from MASS 7.3-58.2 under R 4.2.3:

```r
library(MASS)
write.csv(birthwt[, c("low", "age", "lwt", "smoke", "bwt")],
          "notebooks/data/birthwt.csv", row.names = FALSE)
```

The copy exists only so the notebook runs without R. **The R side of the
comparison needs no file at all** — `birthwt` ships with MASS, so the notebook's
pasteable snippet reads `data = birthwt` directly, which is what makes the
three-way agreement checkable by anyone with an R install. `MASS` is GPL-2/GPL-3
and its datasets are redistributable on those terms.

Regenerating this file is a **contract change**: `classical_fit_tram_dag.py`
compares against R coefficients hard-coded from the fit above, so a regenerated
CSV means re-running the R snippet and updating `R_GLM` / `R_LOGLIK` there.

## `vaca.csv`

The bimodal VACA triangle of Section 1 of `classical_fit_tram_dag.py`: 1000 rows
of `x1`, `x2`, `x3` from

```
x1 ~ 0.5 N(-2, 1.5) + 0.5 N(1.5, 1)
x2 = -x1 + N(0, 1)
x3 =  x1 + 0.25 x2 + N(0, 1)
```

The notebook carries its generator in an `if False:` block, so a normal run
reads this file and does not rewrite it. Run that block by hand to regenerate
the sample, which is seeded and therefore reproducible. The file is tracked so
that the R snippet in the same section reads the identical rows the flow was
fitted on.

The notebook pins R's reference coefficients from that file:

```r
library(tram)
d <- read.csv("notebooks/data/vaca.csv")
m <- Colr(x3 ~ x1 + x2, data = d, order = 19)
coef(m)     # -> x1 -1.778313   x2 -0.455721
logLik(m)   # -> -1412.012824
```

(R 4.2.3, tram 1.0.4.) A change to `n` or to the seed changes those numbers, so
re-run the snippet and update `R_COLR` in the notebook when you change either.

Not to be confused with `experiments/paper/data/vaca/` — that is the frozen
5000-row benchmark of the TRAM-DAG paper replications, under the testing
contract described in CLAUDE.md. This file is neither frozen nor a contract.
