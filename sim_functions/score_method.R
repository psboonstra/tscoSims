# Scores one fitted method against the truth on the evaluation set. Every
# metric reported by the simulation is defined here and nowhere else.
#
# `fit` is the standardised list returned by any methods/*.R function:
#   fit_ok   (logical)      the model(s) were fit without error
#   p_hat    (matrix or NA) predicted probabilities on test_dat
#   stat, df, p_value       likelihood-ratio test of the A effect
#   warnings (character)    warnings raised while fitting
#
# Because p_true is known, every metric below is an *excess risk* relative to
# the oracle, estimated directly rather than through realised outcomes:
#
#   sum_k (p_hat_k - p_k)^2            = excess Brier risk
#   sum_k (F_hat(k) - F(k))^2          = excess ranked probability score
#   sum_k p_k log(p_k / p_hat_k)       = excess log risk (KL)
#
# Brier and KL are invariant to permuting the outcome categories, so neither
# can distinguish an error between adjacent categories from one that moves
# mass across the whole scale. RPS, built on the cumulative probabilities, is
# the ordinal analogue and is the primary accuracy metric here.

prob_brier <- function(p_hat, p_true) {
  mean(rowSums((as.matrix(p_hat) - as.matrix(p_true))^2))
}

prob_rps <- function(p_hat, p_true) {
  F_hat <- t(apply(as.matrix(p_hat), 1, cumsum))
  F_true <- t(apply(as.matrix(p_true), 1, cumsum))

  K <- ncol(F_true)

  # The final cumulative probability is 1 in both by construction.
  mean(rowSums((F_hat[, -K, drop = FALSE] - F_true[, -K, drop = FALSE])^2))
}

prob_mae <- function(p_hat, p_true) {
  mean(abs(as.matrix(p_hat) - as.matrix(p_true)))
}

# No flooring of p_hat. A fit that puts zero probability on a category the
# truth gives positive probability has infinite excess log risk, and that is
# the honest answer: flooring at some eps replaces the infinity with a large
# number that is a pure function of eps, which then dominates any average
# taken over replicates. Summarise this column with the median and report how
# often it is infinite.
prob_kl <- function(p_hat, p_true) {
  p_hat <- as.matrix(p_hat)
  p_true <- as.matrix(p_true)

  r <- p_true * log(p_true / p_hat)
  r[p_true == 0] <- 0

  mean(rowSums(r))
}

score_method <- function(fit, p_true, alpha = 0.05) {

  # Three distinct things can go wrong, and conflating them biases the
  # summaries: a model that fits and tests cleanly but fails to predict must
  # still contribute to the rejection rate.
  fit_ok <- isTRUE(fit$fit_ok)
  pred_ok <- is.matrix(fit$p_hat)

  p_value <- as.numeric(fit$p_value)
  test_ok <- is.finite(p_value)

  tibble(
    fit_ok = fit_ok,
    test_ok = test_ok,
    pred_ok = pred_ok,
    p_value = p_value,
    df = as.numeric(fit$df),
    stat = as.numeric(fit$stat),
    reject = test_ok && p_value < alpha,
    brier = if (pred_ok) prob_brier(fit$p_hat, p_true) else NA_real_,
    rps = if (pred_ok) prob_rps(fit$p_hat, p_true) else NA_real_,
    mae = if (pred_ok) prob_mae(fit$p_hat, p_true) else NA_real_,
    kl = if (pred_ok) prob_kl(fit$p_hat, p_true) else NA_real_,
    n_warnings = length(fit$warnings),
    warnings = paste(unique(fit$warnings), collapse = " | ")
  )
}

# Returned by a methods/*.R function when the model could not be fit at all.
null_fit <- function(warnings = character()) {
  list(
    fit_ok = FALSE,
    p_hat = NA,
    stat = NA_real_,
    df = NA_real_,
    p_value = NA_real_,
    warnings = warnings
  )
}
