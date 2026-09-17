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

  # Since the 2026-09-17 package fix, tsco() warns and fits over the observed
  # levels rather than erroring, so this replicate now stays in the denominator.
  # `safe_fit` has already captured that warning; add the shared tag so all four
  # methods are greppable the same way.
  warnings <- c(
    fit_obj$warnings,
    unobserved_level_tag(
      # Match the harness's method labels: "multinomial" is reported as "mr".
      paste0("tsco_", sub("multinomial", "mr", stage1), sub("multinomial", "mr", stage2)),
      unobserved_levels(dat, levels_y)
    )
  )

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
    warnings = warnings
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
