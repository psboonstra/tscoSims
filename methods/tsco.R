# Two-stage conditional odds. Both variants share the joint likelihood-ratio
# test that tsco::summary() reports for the A term, which carries the
# stage-1 and stage-2 coefficients for A together.

tsco_lrt <- function(fit) {
  s <- summary(fit, joint_test = "LRT")

  if (!("A" %in% rownames(s$joint_tests))) {
    return(c(stat = NA_real_, df = NA_real_, p = NA_real_))
  }

  c(
    stat = s$joint_tests["A", "Chisq"],
    df = s$joint_tests["A", "df"],
    p = s$joint_tests["A", "Pr(>Chisq)"]
  )
}

fxn_tsco <- function(dat, test_dat, levels_y, cutoff_level, stage1, stage2) {

  fit_obj <- safe_fit(
    tsco::tsco(
      Y ~ A + W,
      data = dat,
      levels = levels_y,
      cutoff_level = cutoff_level,
      stage1 = stage1,
      stage2 = stage2,
      warn_degenerate = FALSE
    )
  )

  if (!fit_obj$ok) {
    return(null_fit(fit_obj$warnings))
  }

  tst <- tsco_lrt(fit_obj$fit)

  p_hat <- tryCatch(
    align_prob(
      predict(fit_obj$fit, newdata = test_dat, type = "prob"),
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
    warnings = fit_obj$warnings
  )
}

fxn_tsco_pomr <- function(dat, test_dat, levels_y, cutoff_level) {
  fxn_tsco(dat, test_dat, levels_y, cutoff_level,
           stage1 = "po", stage2 = "multinomial")
}

fxn_tsco_popo <- function(dat, test_dat, levels_y, cutoff_level) {
  fxn_tsco(dat, test_dat, levels_y, cutoff_level,
           stage1 = "po", stage2 = "po")
}
