# The R half of notebooks/classical_fit_tram_dag.py.
#
# Every classical reference the notebook prints is fitted here, so the numbers
# it hard-codes can be re-checked rather than trusted. Run from the repo root:
#
#     Rscript notebooks/classical_fit_tram_dag.R
#
# Needs MASS (ships with R) and tram. `birthwt` comes from MASS, so Sections 0
# and 1 read no file at all; Section 1b reads the CSV the notebook tracks.
#
# A note on the Bernstein degree. The two libraries count the basis
# differently: the flow's `n_coeffs` unconstrained coefficients become
# `n_coeffs + 2` control points, that is a polynomial of degree `n_coeffs + 1`.
# So the notebook's `n_coeffs = 20` is tram's `order = 21`, not `order = 19`.

suppressMessages(library(MASS))
suppressMessages(library(tram))

cat("== Section 0: logistic regression on the dichotomized outcome ==\n")
# birthwt: 189 births, `low` = birth weight under 2.5 kg
m0 <- glm(low ~ age + lwt + smoke, data = birthwt, family = binomial)
print(round(coef(m0), 6))
cat("logLik", sprintf("%.6f", as.numeric(logLik(m0))), "\n")
# (Intercept)         age         lwt       smoke
#    1.368225   -0.038995   -0.012139    0.670764
# logLik -111.439676

cat("\n== Section 1: the same predictors against the continuous outcome ==\n")
d <- birthwt
d$bwt <- as.numeric(d$bwt) # tram rejects the integer column
m1 <- Colr(bwt ~ age + lwt + smoke, data = d, order = 21)
print(round(coef(m1), 6))
cat("logLik", sprintf("%.4f", as.numeric(logLik(m1))), "\n")
#        age        lwt      smoke
#  -0.021971  -0.010233   0.668698
# logLik -1499.6678

# glm cannot fit the model above: `binomial` needs a binary outcome, and there
# is no family for "flexible monotone h, logistic latent". `gaussian` is lm --
# h forced linear, latent normal -- so its coefficients are in GRAMS and only
# its log-likelihood is comparable with m1.
g1 <- glm(bwt ~ age + lwt + smoke, data = d, family = gaussian)
print(round(coef(g1), 4))
cat("logLik", sprintf("%.4f", as.numeric(logLik(g1))), "\n")
# (Intercept)         age         lwt       smoke
#   2362.4955      7.1538      4.0155   -269.2570
# logLik -1506.6629
#
# The notebook fits the flow both ways for the comparison: with a Bernstein
# baseline it reaches -1500.55, with a linear one -1507.62. The linear flow and
# lm differ only in the latent (logistic vs normal) and land within 1 nat.
#
# The `smoke` coefficient is +0.669 here and +0.671 from m0 above. A linear
# shift moves the whole latent distribution, so cutting the outcome at 2500 g
# discards information about the baseline transformation but not the shift.

cat("\n== Section 1b: the bimodal VACA triangle ==\n")
v <- read.csv("notebooks/data/vaca.csv")

m2 <- Colr(x2 ~ x1, data = v, order = 21)
cat(sprintf("x2 node : x1 %.6f   logLik %.6f\n", coef(m2), as.numeric(logLik(m2))))
# x2 node : x1 1.800042   logLik -1401.213542

m3 <- Colr(x3 ~ x1 + x2, data = v, order = 21)
cat(sprintf(
  "x3 node : x1 %.6f  x2 %.6f   logLik %.6f\n",
  coef(m3)[1], coef(m3)[2], as.numeric(logLik(m3))
))
# x3 node : x1 -1.777289  x2 -0.455282   logLik -1411.901958
#
# The flow reaches logLik -1410.97 on x3 and -1401.88 on x2: better on one node,
# worse on the other. The flow pre-scales the 5%/95% quantiles onto [-5, 5] and
# runs Bernstein there, leaving 10% of rows in linear tails, while Colr spends
# its whole basis on the observed support. Neither function class contains the
# other, so which fits better is a property of the node.

cat("\nR", as.character(getRversion()),
    "| MASS", as.character(packageVersion("MASS")),
    "| tram", as.character(packageVersion("tram")), "\n")
