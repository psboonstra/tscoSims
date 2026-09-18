# Checks the rewritten constrained direct fitter (2026-09-18) against the
# previous exp()-reparameterised one and against VGAM where VGAM is valid.
#
#   NSIM=100 Rscript verification/r_cppo_fitter_check.R
#
# What must hold:
#   1. Interior draws: new direct == VGAM == old direct in loglik (~1e-5).
#   2. Boundary draws: new loglik >= old loglik (the supremum is now attained),
#      converged = TRUE, on_boundary = TRUE, active face named correctly.
#   3. stat >= 0 on every draw (start from the reduced solution).
#   4. No fit failures.

suppressMessages({library(tsco); library(VGAM); library(dplyr); library(tibble)})
options(warn = -1)

levels_y <- as.character(0:5); K <- length(levels_y); M <- K - 1L
master_seed <- 20260710; array_id <- 1L
n <- as.integer(Sys.getenv("N", "200"))
scenario <- Sys.getenv("SCENARIO", "cppo_alt")
nsim <- as.integer(Sys.getenv("NSIM", "100"))

source("sim_functions/score_method.R"); source("sim_functions/true_probs.R")
source("sim_functions/dataset_diagnostics.R")
source("aux_functions/safe_fit.R"); source("aux_functions/align_prob.R")
source("aux_functions/vglm_helpers.R"); source("aux_functions/unobserved_levels.R"); source("aux_functions/fit_converged.R")
source("methods/cppo.R")
G <- cppo_G(levels_y)

# The old fitter, in its own environment.
old <- new.env()
sys.source("verification/cppo_fitter_pre_20260918.R", envir = old)

set.seed(master_seed + 999); test_dat <- make_covariates(50)

out <- NULL
for (i in 1:nsim) {
  set.seed(master_seed + array_id * 100000L + i)
  dat <- make_covariates(n)
  dat$Y <- draw_ordinal(true_prob_scenario(dat, scenario), levels_y = levels_y)
  dg <- dataset_diagnostics(dat, levels_y)

  Yidx <- match(as.character(dat$Y), levels_y)
  X <- cbind(A = dat$A, W = dat$W)

  # New fitter
  t0 <- Sys.time()
  red_new <- po_direct_fit(X[, "W", drop = FALSE], Yidx, K)
  full_new <- cppo_direct_fit(X, Yidx, 1L, G, K, red = red_new)
  t_new <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  # Old fitter
  t0 <- Sys.time()
  red_old <- old$po_direct_fit(X[, "W", drop = FALSE], Yidx, K)
  full_old <- old$cppo_direct_fit(X, Yidx, 1L, G, K)
  t_old <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  # VGAM, where it is valid and no level is globally empty
  ll_vgam <- NA_real_
  if (dg$n_empty_levels == 0L) {
    ff <- safe_fit(VGAM::vglm(Y ~ A + W, data = dat,
            family = VGAM::cumulative(link = "logitlink", parallel = FALSE, reverse = FALSE),
            constraints = cppo_constraints(levels_y, G, with_A = TRUE)))
    if (ff$ok && cppo_vglm_valid(ff$fit)) ll_vgam <- vglm_loglik(ff$fit)
  }

  # What fxn_cppo returns
  res <- fxn_cppo(dat, test_dat, levels_y)

  out <- bind_rows(out, tibble(
    i = i,
    cell_dep_exp = dg$cell_dep_exp, cell_dep_unexp = dg$cell_dep_unexp,
    n_empty_levels = dg$n_empty_levels,
    ll_new = full_new$loglik, ll_old = full_old$loglik, ll_vgam = ll_vgam,
    llred_new = red_new$loglik, llred_old = red_old$loglik,
    stat_new = 2 * (full_new$loglik - red_new$loglik),
    stat_old = 2 * (full_old$loglik - red_old$loglik),
    conv_new = full_new$converged, conv_old = full_old$converged,
    kkt_new = full_new$kkt_max_violation, code_new = full_new$optim_code,
    on_bdry = full_new$on_boundary,
    act_exp = full_new$active_exposed, act_unexp = full_new$active_unexposed,
    slack_exp_new = full_new$slack_exposed, slack_exp_old = full_old$slack_exposed,
    slack_unexp_new = full_new$slack_unexposed, slack_unexp_old = full_old$slack_unexposed,
    gamma_new = full_new$gamma, gamma_old = full_old$gamma,
    t_new = t_new, t_old = t_old,
    fxn_fit_ok = res$fit_ok, fxn_boundary = res$boundary, fxn_engine = res$engine,
    fxn_stat = as.numeric(res$stat), fxn_p = as.numeric(res$p_value)
  ))
  if (i %% 25 == 0) cat("  ...", i, "/", nsim, "\n")
}

saveRDS(out, sprintf("verification/cppo_fitter_check_%s_n%d_%d.rds", scenario, n, nsim))

out <- out |> mutate(stratum = case_when(
  n_empty_levels > 0 ~ "level absent",
  cell_dep_exp == 0 ~ "exposed dep empty",
  cell_dep_unexp == 0 ~ "unexposed dep empty",
  TRUE ~ "interior"))

cat("\n=== 1. Interior draws: agreement of new direct with VGAM and old direct ===\n")
int <- out |> filter(stratum == "interior", is.finite(ll_vgam))
cat(sprintf("  n = %d   max|ll_new - ll_vgam| = %.2e   max|ll_new - ll_old| = %.2e\n",
    nrow(int), max(abs(int$ll_new - int$ll_vgam)), max(abs(int$ll_new - int$ll_old))))
cat(sprintf("  on_boundary flagged among interior draws: %d\n", sum(int$on_bdry)))

cat("\n=== 2. Boundary draws: new attains at least the old value ===\n")
bd <- out |> filter(stratum != "interior")
cat(sprintf("  n = %d   min(ll_new - ll_old) = %.2e   max(ll_new - ll_old) = %.2e\n",
    nrow(bd), min(bd$ll_new - bd$ll_old), max(bd$ll_new - bd$ll_old)))
print(bd |> count(stratum, on_bdry, act_exp, act_unexp) |> as.data.frame())

cat("\n=== 3. stat >= 0 and convergence ===\n")
cat(sprintf("  min stat_new = %.2e   (old had %d negative)\n", min(out$stat_new), sum(out$stat_old < -1e-7)))
cat(sprintf("  converged_new: %d/%d   converged_old: %d/%d\n",
    sum(out$conv_new), nrow(out), sum(out$conv_old), nrow(out)))
cat(sprintf("  KKT max violation: median %.2e, max %.2e\n", median(out$kkt_new), max(out$kkt_new)))
print(table(optim_code = out$code_new))

cat("\n=== 4. fxn_cppo ===\n")
print(out |> count(stratum, fxn_fit_ok, fxn_engine, fxn_boundary) |> as.data.frame())

cat("\n=== 5. Timing ===\n")
cat(sprintf("  median secs: new %.3f  old %.3f\n", median(out$t_new), median(out$t_old)))

cat("\n=== 6. Old vs new statistic where they disagree by > 0.01 ===\n")
print(out |> filter(abs(stat_new - stat_old) > 0.01) |>
  select(i, stratum, stat_old, stat_new, ll_old, ll_new, slack_exp_old, slack_exp_new,
         slack_unexp_old, slack_unexp_new, conv_old) |> as.data.frame(), digits = 5)
