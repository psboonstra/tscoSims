# Conventional proportional odds. The A effect is tested by a likelihood-ratio
# test against the model that omits A.

fxn_po <- function(dat, test_dat, levels_y) {

  fit_full <- safe_fit(
    VGAM::vglm(
      Y ~ A + W,
      data = dat,
      family = VGAM::cumulative(link = "logitlink", parallel = TRUE, reverse = FALSE)
    )
  )

  fit_red <- safe_fit(
    VGAM::vglm(
      Y ~ W,
      data = dat,
      family = VGAM::cumulative(link = "logitlink", parallel = TRUE, reverse = FALSE)
    )
  )

  warnings <- c(fit_full$warnings, fit_red$warnings)

  if (!fit_full$ok || !fit_red$ok) {
    return(null_fit(warnings))
  }

  # A returned object is not a converged fit (review issue 12). Non-convergence
  # is a failure of the fitter and is reported as one; in 300 surveyed draws at
  # n = 200 it did not occur, so this guards a rare event rather than reshaping
  # the denominator.
  if (!fit_converged(fit_full$fit) || !fit_converged(fit_red$fit)) {
    return(null_fit(c(warnings, "po: VGAM did not converge")))
  }

  tst <- vglm_lrt(fit_full$fit, fit_red$fit)

  # PO keeps its single A coefficient when a level is unobserved, so df does not
  # move, but the fit is still conditional on the observed support.
  warnings <- c(
    warnings,
    unobserved_level_tag("po", unobserved_levels(dat, levels_y)),
    df_shortfall_tag("po", tst["df"], 1L)
  )

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
