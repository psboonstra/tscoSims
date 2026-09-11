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

  tst <- vglm_lrt(fit_full$fit, fit_red$fit)

  p_hat <- tryCatch(
    align_prob(
      VGAM::predictvglm(fit_full$fit, newdata = test_dat, type = "response"),
      levels_y
    ),
    error = function(e) NA
  )

  list(
    p_hat = p_hat,
    stat = tst["stat"],
    df = tst["df"],
    p_value = tst["p"],
    warnings = warnings
  )
}
