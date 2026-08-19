#!/usr/bin/env Rscript
# Classical all-`ls` reference for the magic-mrclean synthetic cohort.
#
# Fits the all-linear-shift stroke DAG node-by-node with standard R, mirroring
# the zuko_dag all-`ls` flow exactly (logistic latent; continuous parents enter
# raw, the ordinal parent mRS_pre enters as a factor / one-hot):
#
#   mRS_pre ~ Age                          MASS::polr   (ordered logit)
#   NIHSSa  ~ Age + mRS_pre                tram::Colr   (continuous logistic TM)
#   T       ~ Age + mRS_pre + NIHSSa       glm(binomial) (logit)
#   mRS_3m  ~ Age + mRS_pre + NIHSSa + T   MASS::polr   (ordered logit)
#
# The interventional ATE is computed analytically: predict mRS_3m class
# probabilities on the RCT covariates under do(T=0) and do(T=1), average
# P(mRS_3m <= 2), take the difference -- the same recipe as the Python
# evaluate_rct().
#
# Usage:  Rscript fit_ls.R <variant>      # variant = ls | nl
# Writes: <variant>/ref_ls/coefficients.csv   (node, term, estimate, std_error)
#         <variant>/ref_ls/ate.csv            (p_good_do_T0, p_good_do_T1, ate)

suppressMessages({
  library(MASS)
  library(tram)
})

args <- commandArgs(trailingOnly = TRUE)
variant <- if (length(args) >= 1) args[1] else "ls"
base <- file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE),
                                                  value = TRUE))), variant)
if (!dir.exists(base)) base <- variant  # fallback: run from the data dir
stopifnot(dir.exists(base))

obs <- read.csv(file.path(base, "obs.csv"))
rct <- read.csv(file.path(base, "rct.csv"))

# ordinal columns as ordered factors with the full level set (so missing levels
# in a particular sample still get a column)
for (d in c("obs", "rct")) {
  df <- get(d)
  df$mRS_pre <- factor(df$mRS_pre, levels = 0:5)
  df$mRS_3m  <- ordered(df$mRS_3m, levels = 0:6)
  assign(d, df)
}

coef_rows <- list()
add_coef <- function(node, est, se) {
  coef_rows[[length(coef_rows) + 1]] <<- data.frame(
    node = node, term = names(est), estimate = as.numeric(est),
    std_error = as.numeric(se), row.names = NULL)
}

## --- mRS_pre ~ Age  (ordered logit) ---------------------------------------
m_pre <- polr(ordered(mRS_pre, levels = 0:5) ~ Age, data = obs,
              method = "logistic", Hess = TRUE)
sp <- summary(m_pre)$coefficients
add_coef("mRS_pre", c(Age = coef(m_pre)["Age"]), sp["Age", "Std. Error"])

## --- NIHSSa ~ Age + mRS_pre  (continuous logistic transformation model) -----
m_nih <- Colr(NIHSSa ~ Age + mRS_pre, data = obs)
cn <- coef(m_nih)                         # shift coefficients (baseline excluded)
vn <- sqrt(diag(vcov(m_nih)))[names(cn)]
add_coef("NIHSSa", cn, vn)

## --- T ~ Age + mRS_pre + NIHSSa  (logistic regression) ----------------------
m_t <- glm(T ~ Age + mRS_pre + NIHSSa, data = obs, family = binomial())
st <- summary(m_t)$coefficients
add_coef("T", coef(m_t), st[, "Std. Error"])

## --- mRS_3m ~ Age + mRS_pre + NIHSSa + T  (ordered logit) -------------------
m_y <- polr(mRS_3m ~ Age + mRS_pre + NIHSSa + T, data = obs,
            method = "logistic", Hess = TRUE)
sy <- summary(m_y)$coefficients
yb <- coef(m_y)
add_coef("mRS_3m", yb, sy[names(yb), "Std. Error"])

## --- analytic ATE on the RCT covariates ------------------------------------
good <- function(newdata) {
  p <- predict(m_y, newdata = newdata, type = "probs")  # n x 7 (levels 0..6)
  rowSums(p[, c("0", "1", "2"), drop = FALSE])           # P(mRS_3m <= 2)
}
rct0 <- rct; rct0$T <- 0
rct1 <- rct; rct1$T <- 1
p0 <- mean(good(rct0))
p1 <- mean(good(rct1))

out <- file.path(base, "ref_ls")
dir.create(out, showWarnings = FALSE)
write.csv(do.call(rbind, coef_rows), file.path(out, "coefficients.csv"),
          row.names = FALSE)
write.csv(data.frame(p_good_do_T0 = p0, p_good_do_T1 = p1, ate = p1 - p0),
          file.path(out, "ate.csv"), row.names = FALSE)

cat(sprintf("[%s] R reference: P(good|do T0)=%.4f  P(good|do T1)=%.4f  ATE=%+.4f\n",
            variant, p0, p1, p1 - p0))
cat(sprintf("        mRS_3m coefs: Age=%+.4f  NIHSSa=%+.4f  T=%+.4f\n",
            yb["Age"], yb["NIHSSa"], yb["T"]))
cat(sprintf("        wrote %s/{coefficients,ate}.csv\n", out))
