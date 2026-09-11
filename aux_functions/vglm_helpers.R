# Log-likelihood, model rank and likelihood-ratio test for VGAM fits.

vglm_loglik <- function(fit) {
  out <- tryCatch(
    as.numeric(fit@criterion["loglikelihood"]),
    error = function(e) NA_real_
  )

  if (length(out) != 1L || !is.finite(out)) NA_real_ else out
}

vglm_df <- function(fit) {
  length(stats::coef(fit))
}

vglm_lrt <- function(fit_full, fit_red) {
  ll_full <- vglm_loglik(fit_full)
  ll_red <- vglm_loglik(fit_red)

  df_full <- vglm_df(fit_full)
  df_red <- vglm_df(fit_red)

  stat <- 2 * (ll_full - ll_red)
  df <- df_full - df_red

  if (is.finite(stat) && stat < 0 && stat > -1e-7) stat <- 0

  p <- if (is.finite(stat) && df > 0) {
    stats::pchisq(stat, df = df, lower.tail = FALSE)
  } else {
    NA_real_
  }

  c(stat = stat, df = df, p = p)
}
