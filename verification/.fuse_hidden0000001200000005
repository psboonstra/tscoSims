# Lean type I error driver. Uses the harness's own method functions and seeding
# scheme unchanged, so p_value / reject / warnings are exactly what run_sims.R
# would produce. The evaluation set is deliberately SMALL (n_test = 200): the
# LRT does not depend on it, and the accuracy metrics from this run are
# therefore not to be used -- only the test columns.
#
# Checkpoints every block of 50 replicates to scratch_typeI/<array>_<block>.rds
# so it can be re-run to resume.
suppressMessages({library(tsco); library(VGAM); library(dplyr); library(tibble); library(glue)})
options(warn = -1)

levels_y <- as.character(0:5); cutoff_level <- "5"; K <- length(levels_y)
master_seed <- 20260710; alpha <- 0.05; n <- 200; scenario <- "null"
methods_seq <- c("po", "tsco_popo", "mr", "cppo")

source("sim_functions/score_method.R"); source("sim_functions/true_probs.R")
source("aux_functions/safe_fit.R"); source("aux_functions/align_prob.R")
source("aux_functions/vglm_helpers.R")
source("methods/po.R"); source("methods/mr.R"); source("methods/cppo.R"); source("methods/tsco.R")

array_id <- as.integer(Sys.getenv("ARRAY_ID", "1"))
reps     <- as.integer(Sys.getenv("REPS", "350"))
budget   <- as.numeric(Sys.getenv("BUDGET_S", "480"))
dir.create("scratch_typeI", showWarnings = FALSE)

set.seed(master_seed + 999)
test_dat <- make_covariates(200)
p_true_test <- true_prob_scenario(test_dat, scenario)

G <- cppo_G(levels_y)
t0 <- Sys.time()
blocks <- split(seq_len(reps), ceiling(seq_len(reps) / 50))

for (bi in names(blocks)) {
  f <- sprintf("scratch_typeI/a%d_b%s.rds", array_id, bi)
  if (file.exists(f)) next
  if (as.numeric(difftime(Sys.time(), t0, units = "secs")) > budget) {
    cat("budget reached; stopping before block", bi, "\n"); break
  }
  acc <- list()
  for (i in blocks[[bi]]) {
    seed_i <- master_seed + array_id * 100000L + i
    set.seed(seed_i)
    dat <- make_covariates(n)
    p_true <- true_prob_scenario(dat, scenario)
    dat$Y <- draw_ordinal(p_true, levels_y = levels_y)

    # dataset-level diagnostics, independent of any fit
    n_exp_cell <- sum(dat$A == 1 & dat$Y == as.character(K - 2L))
    n_cell     <- sum(dat$Y == as.character(K - 2L))

    fits <- list()
    if ("po"        %in% methods_seq) fits$po        <- fxn_po(dat, test_dat, levels_y)
    if ("tsco_popo" %in% methods_seq) fits$tsco_popo <- fxn_tsco_popo(dat, test_dat, levels_y, cutoff_level)
    if ("mr"        %in% methods_seq) fits$mr        <- fxn_mr(dat, test_dat, levels_y)
    if ("cppo"      %in% methods_seq) fits$cppo      <- fxn_cppo(dat, test_dat, levels_y)

    for (m in names(fits)) {
      s <- score_method(fits[[m]], p_true_test, alpha)
      acc[[length(acc) + 1]] <- bind_cols(
        tibble(array_id = array_id, sim_num = i, data_seed = seed_i,
               n = n, scenario = scenario, method = m,
               n_exp_cell = n_exp_cell, n_cell = n_cell),
        s |> select(fit_ok, test_ok, pred_ok, p_value, df, stat, reject,
                    n_warnings, warnings))
    }
  }
  saveRDS(bind_rows(acc), f)
  cat(sprintf("  wrote %s  (%.0f s elapsed)\n", f,
              as.numeric(difftime(Sys.time(), t0, units = "secs"))))
}
