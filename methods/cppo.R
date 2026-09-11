# Constrained partial proportional odds (Peterson and Harrell, 1990). The A
# effect is allowed to depart from proportionality at the highest cutpoint
# only, through the prespecified contrast G.

cppo_constraints <- function(levels_y, G = c(0, 0, 0, 1), with_A = TRUE) {
  M <- length(levels_y) - 1L

  if (with_A && length(G) != M) {
    stop("G must have length K - 1.")
  }

  if (with_A) {
    list(
      "(Intercept)" = diag(M),
      A = cbind(common = rep(1, M), deviation = G),
      W = matrix(1, nrow = M, ncol = 1)
    )
  } else {
    list(
      "(Intercept)" = diag(M),
      W = matrix(1, nrow = M, ncol = 1)
    )
  }
}

fxn_cppo <- function(dat, test_dat, levels_y, G = c(0, 0, 0, 1)) {

  fit_full <- safe_fit(
    VGAM::vglm(
      Y ~ A + W,
      data = dat,
      family = VGAM::cumulative(link = "logitlink", parallel = FALSE, reverse = FALSE),
      constraints = cppo_constraints(levels_y, G, with_A = TRUE)
    )
  )

  fit_red <- safe_fit(
    VGAM::vglm(
      Y ~ W,
      data = dat,
      family = VGAM::cumulative(link = "logitlink", parallel = FALSE, reverse = FALSE),
      constraints = cppo_constraints(levels_y, G, with_A = FALSE)
    )
  )

  warnings <- c(fit_full$warnings, fit_red$warnings)

  if (!fit_full$ok || !fit_red$ok) {
    return(null_fit(warnings))
  }

  tst <- vglm_lrt(fit_full$fit, fit_red$fit)

  p_hat <- tryCatch(
    align_prob(
      VGAM::predictvglm(fit_full$fit, newdata = test_dat, type = "response"),
      levels_y
    ),
    error = function(e) NA
  )

  list(
    fit_ok = TRUE,
    p_hat = p_hat,
    stat = tst["stat"],
    df = tst["df"],
    p_value = tst["p"],
    warnings = warnings
  )
}
