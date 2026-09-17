# Item 2: what each method does when a declared outcome level is unobserved.
#
# Runs all five methods over cppo_alt draws at n = 200 (the worst case) and
# records df, p-value and the tags, stratified by the emptiness of the
# departure cell. Accuracy metrics are skipped -- this is about the test.

suppressMessages({library(tsco); library(VGAM); library(dplyr); library(tibble); library(glue)})
options(warn = -1)

levels_y <- as.character(0:5); cutoff_level <- "5"; K <- length(levels_y)
master_seed <- 20260710; array_id <- 1L; alpha <- 0.05
n <- as.integer(Sys.getenv("N", "200"))
scenario <- Sys.getenv("SCENARIO", "cppo_alt")
nsim <- as.integer(Sys.getenv("NSIM", "400"))

source("sim_functions/score_method.R"); source("sim_functions/true_probs.R")
source("sim_functions/dataset_diagnostics.R"); source("sim_functions/method_setup.R")
source("aux_functions/safe_fit.R"); source("aux_functions/align_prob.R")
source("aux_functions/vglm_helpers.R"); source("aux_functions/unobserved_levels.R")
source("methods/po.R"); source("methods/mr.R"); source("methods/cppo.R"); source("methods/tsco.R")

method_list <- build_method_list(levels_y, cutoff_level)
methods_seq <- c("tsco_popo", "tsco_pomr", "po", "mr", "cppo")

# A small test set: p_hat is not scored here, but the methods still want one.
set.seed(master_seed + 999); test_dat <- make_covariates(50)

out <- NULL
for (i in 1:nsim) {
  seed_i <- master_seed + array_id * 100000L + i
  set.seed(seed_i)
  dat <- make_covariates(n)
  dat$Y <- draw_ordinal(true_prob_scenario(dat, scenario), levels_y = levels_y)
  dg <- dataset_diagnostics(dat, levels_y)

  for (m in methods_seq) {
    f <- method_list[[m]](dat, test_dat)
    out <- bind_rows(out, tibble(
      i = i, method = m,
      fit_ok = isTRUE(f$fit_ok),
      df = as.numeric(f$df),
      stat = as.numeric(f$stat),
      p = as.numeric(f$p_value),
      test_ok = is.finite(as.numeric(f$p_value)),
      reject = is.finite(as.numeric(f$p_value)) && as.numeric(f$p_value) < alpha,
      n_empty_levels = dg$n_empty_levels,
      cell_dep_exp = dg$cell_dep_exp,
      cell_dep_unexp = dg$cell_dep_unexp,
      tags = paste(unique(f$warnings), collapse = " | ")
    ))
  }
  if (i %% 25 == 0) cat(glue("  ... {i}/{nsim}\n\n"))
}

out <- out |>
  mutate(stratum = case_when(
    n_empty_levels > 0            ~ "both arms empty",
    cell_dep_exp == 0             ~ "exposed arm only",
    cell_dep_unexp == 0           ~ "unexposed arm only",
    TRUE                          ~ "neither arm empty"
  ))

saveRDS(out, sprintf("verification/empty_policy_%s_n%d_%d.rds", scenario, n, nsim))
cat("\n================ saved ================\n")
print(count(out, stratum) |> mutate(pct = 100 * n / sum(n) * length(methods_seq)))
