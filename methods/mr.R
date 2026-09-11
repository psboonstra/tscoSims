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
