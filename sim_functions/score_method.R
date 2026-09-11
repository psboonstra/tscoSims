# Scores one fitted method against the truth on the evaluation set. Every
# metric reported by the simulation is defined here and nowhere else.
#
# `fit` is the standardised list returned by any methods/*.R function:
#   p_hat    (matrix or NA) predicted probabilities on test_dat
#   stat, df, p_value       likelihood-ratio test of the A effect
#   warnings (character)    warnings raised while fitting

prob_rmse <- function(p_hat, p_true) {
  p_hat <- as.matrix(p_hat)
  p_true <- as.matrix(p_true)

  sqrt(mean((p_hat - p_true)^2))
}

prob_mae <- function(p_hat, p_true) {
  mean(abs(as.matrix(p_hat) - as.matrix(p_true)))
}

prob_kl <- function(p_hat, p_true, eps = 1e-12) {
  p_hat <- pmax(as.matrix(p_hat), eps)
  p_hat <- p_hat / rowSums(p_hat)

  p_true <- pmax(as.matrix(p_true), eps)
  p_true <- p_true / rowSums(p_true)

  mean(rowSums(p_true * log(p_true / p_hat)))
}

score_method <- function(fit, p_true, alpha = 0.05) {

  fit_ok <- is.matrix(fit$p_hat)
  p_value <- as.numeric(fit$p_value)

  tibble(
    fit_ok = fit_ok,
    p_value = p_value,
    df = as.numeric(fit$df),
    stat = as.numeric(fit$stat),
    reject = is.finite(p_value) && p_value < alpha,
    rmse = if (fit_ok) prob_rmse(fit$p_hat, p_true) else NA_real_,
    mae = if (fit_ok) prob_mae(fit$p_hat, p_true) else NA_real_,
    kl = if (fit_ok) prob_kl(fit$p_hat, p_true) else NA_real_,
    n_warnings = length(fit$warnings),
    warnings = paste(unique(fit$warnings), collapse = " | ")
  )
}

# Returned by a methods/*.R function when the model could not be fit at all.
null_fit <- function(warnings = character()) {
  list(
    p_hat = NA,
    stat = NA_real_,
    df = NA_real_,
    p_value = NA_real_,
    warnings = warnings
  )
}
