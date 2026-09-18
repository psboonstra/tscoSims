# Multinomial regression, baseline-category logits with Y = 1 as reference.

fxn_mr <- function(dat, test_dat, levels_y) {

  fit_full <- safe_fit(
    VGAM::vglm(Y ~ A + W, data = dat, family = VGAM::multinomial(refLevel = 1))
  )

  fit_red <- safe_fit(
    VGAM::vglm(Y ~ W, data = dat, family = VGAM::multinomial(refLevel = 1))
  )

  warnings <- c(fit_full$warnings, fit_red$warnings)

  if (!fit_full$ok || !fit_red$ok) {
    return(null_fit(warnings))
  }

  # A returned object is not a converged fit (review issue 12). Non-convergence
  # is a failure of the fitter and is reported as one; in 300 surveyed draws at
  # n = 200 it did not occur, so this guards a rare event rather than reshaping
  # the denominator.
  conv_tags <- c(convergence_failure_tag("mr", fit_full$fit),
                 convergence_failure_tag("mr", fit_red$fit))
  if (length(conv_tags) > 0L) {
    return(null_fit(c(warnings, unique(conv_tags))))
  }

  tst <- vglm_lrt(fit_full$fit, fit_red$fit)

  # VGAM drops an unobserved outcome level silently, so the A test quietly
  # becomes a (K - 2)-df test of a different null. Tag both the cause and the
  # consequence; the df column then says which hypothesis was tested.
  warnings <- c(
    warnings,
    unobserved_level_tag("mr", unobserved_levels(dat, levels_y)),
    df_shortfall_tag("mr", tst["df"], length(levels_y) - 1L)
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
