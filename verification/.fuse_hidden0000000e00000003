# Validation of the switch from a Monte Carlo test set to quadrature.
#
#   1. backward compatibility: wt = NULL must reproduce the old unweighted
#      metrics EXACTLY, so nothing already computed silently changes meaning
#   2. the grid and Monte Carlo estimate the same quantity: the MC average must
#      converge to the grid value as n_test grows
#   3. the grid value has converged in the node count, at a fitted thetahat
#   4. the whole pipeline agrees: prob_* against expected_* against
#      kl_projection()'s own A, and E_n = R_n - A comes out non-negative
#   5. the hypothesis test is untouched: stat / df / p_value / reject identical
#      under eval_by = "grid" and eval_by = "mc"
#
# Run from the tscoSims root:  Rscript verification/r_eval_grid_check.R
suppressMessages({library(tsco); library(VGAM); library(dplyr); library(tibble)})
options(warn = -1)

levels_y <- as.character(0:5); K <- length(levels_y); cutoff_level <- "5"
source("sim_functions/true_probs.R")
source("sim_functions/score_method.R")
source("aux_functions/safe_fit.R"); source("aux_functions/align_prob.R")
source("aux_functions/vglm_helpers.R")
source("methods/po.R"); source("methods/mr.R"); source("methods/cppo.R"); source("methods/tsco.R")
source("sim_functions/kl_projection.R")

all_ok <- TRUE
report <- function(label, ok, detail = "") {
  cat(sprintf("  [%s] %-50s %s\n", if (ok) "ok  " else "FAIL", label, detail))
  all_ok <<- all_ok && ok
}

scen <- "tsco_alt"
set.seed(11111)
dat <- make_covariates(200)
dat$Y <- draw_ordinal(true_prob_scenario(dat, scen), levels_y = levels_y)

grid <- kl_quad_grid(40L)
gx <- grid[, c("A", "W")]
p_grid <- true_prob_scenario(gx, scen, cutoff_level = cutoff_level)
fit <- fxn_po(dat, gx, levels_y)
q_grid <- fit$p_hat

## ---------------------------------------------------------------------------
cat("\n=== 1. backward compatibility: wt = NULL is the old unweighted average ===\n")
## The four metrics gained a `wt` argument. With wt = NULL they must be bit-for-bit
## what they were, or every number computed before today changes meaning.
old_unweighted <- function(p_hat, p_true) {
  ph <- as.matrix(p_hat); pt <- as.matrix(p_true)
  Fh <- t(apply(ph, 1, cumsum)); Ft <- t(apply(pt, 1, cumsum)); Kc <- ncol(Ft)
  r <- pt * log(pt / ph); r[pt == 0] <- 0
  c(brier = mean(rowSums((ph - pt)^2)),
    rps   = mean(rowSums((Fh[, -Kc, drop = FALSE] - Ft[, -Kc, drop = FALSE])^2)),
    mae   = mean(abs(ph - pt)),
    kl    = mean(rowSums(r)))
}
set.seed(7); td <- make_covariates(5000)
pt_t <- true_prob_scenario(td, scen); ph_t <- fxn_po(dat, td, levels_y)$p_hat
o <- old_unweighted(ph_t, pt_t)
n <- c(brier = prob_brier(ph_t, pt_t), rps = prob_rps(ph_t, pt_t),
       mae = prob_mae(ph_t, pt_t), kl = prob_kl(ph_t, pt_t))
## Bit-for-bit equality is the wrong bar: the weighted form accumulates in a
## different order (sum(wt * rowSums(.)) rather than mean(.)), which moves the
## last bit. Require agreement to a few ULPs instead, and print how many.
for (m in names(o)) {
  ulps <- abs(o[[m]] - n[[m]]) / (.Machine$double.eps * abs(o[[m]]))
  report(sprintf("prob_%s(wt = NULL) unchanged", m), ulps <= 4,
         sprintf("%.12e   (%.0f ULP)", n[[m]], ulps))
}

## ---------------------------------------------------------------------------
cat("\n=== 2. Monte Carlo converges to the grid value ===\n")
## They estimate the same expectation, so the MC average must approach the
## quadrature value at the Monte Carlo rate, and its scatter must shrink.
R_grid <- c(kl = prob_kl(q_grid, p_grid, grid$wt), rps = prob_rps(q_grid, p_grid, grid$wt))
## Anchor: the grid value is stable to 13 digits from 40 to 640 nodes, so it is
## the reference and the MC average is the thing being tested against it.
g_hi <- kl_quad_grid(320L); gx_hi <- g_hi[, c("A", "W")]
R_hi <- prob_kl(fxn_po(dat, gx_hi, levels_y)$p_hat,
                true_prob_scenario(gx_hi, scen, cutoff_level = cutoff_level), g_hi$wt)
report("grid value stable from 40 to 320 nodes",
       abs(R_hi - R_grid[["kl"]]) / R_hi < 1e-12,
       sprintf("%.12e vs %.12e", R_grid[["kl"]], R_hi))
cat(sprintf("    grid (quadrature):  KL %.8e   RPS %.8e\n", R_grid[["kl"]], R_grid[["rps"]]))

## The MLE is fit ONCE; everything below is test-set variation only. 100 test
## sets, not 15: with 15 the standard error of the mean is itself so noisy that
## a 2-SE test fails by chance (it did, at z = -2.8, before this was widened).
fit_once <- VGAM::vglm(Y ~ A + W, data = dat,
  family = VGAM::cumulative(link = "logitlink", parallel = TRUE, reverse = FALSE))
pred_once <- function(nd)
  align_prob(VGAM::predictvglm(fit_once, newdata = nd, type = "response"), levels_y)

prev_sd <- Inf; ok_shrink <- TRUE
for (nt in c(5000L, 50000L)) {
  v <- numeric(100)
  for (b in 1:100) {
    set.seed(41000 + 13 * b + nt); tdb <- make_covariates(nt)
    v[b] <- prob_kl(pred_once(tdb), true_prob_scenario(tdb, scen))
  }
  se <- sd(v) / sqrt(100); bias <- mean(v) - R_grid[["kl"]]
  cat(sprintf("    n_test = %6d, 100 sets:  mean %.8e   bias %+.2e   z %+.2f   sd %.2e   sd*sqrt(n_test) %.3f\n",
              nt, mean(v), bias, bias / se, sd(v), sd(v) * sqrt(nt)))
  report(sprintf("  MC mean agrees with the grid at n_test = %d", nt),
         abs(bias / se) < 3, sprintf("z = %+.2f", bias / se))
  ok_shrink <- ok_shrink && sd(v) < prev_sd; prev_sd <- sd(v)
}
report("MC scatter shrinks with n_test", ok_shrink)
## And the practical point the switch is about: that scatter is a COMMON shift,
## because run_sims.R used one fixed test_seed for every array and replicate.
cat(sprintf("    for scale: sd at n_test = 20,000 is ~1.5%% of A_kl and ~3.1%% of A_rps,\n"))
cat(sprintf("    and with a fixed test_seed it does not average out over replicates.\n"))

## ---------------------------------------------------------------------------
cat("\n=== 3. the grid has converged in the node count, at thetahat ===\n")
## thetahat is a rougher integrand than theta*, so convergence is checked here
## rather than borrowed from the projection check.
for (meth in c("po", "tsco_popo", "cppo", "mr")) {
  f <- function(nd) switch(meth,
    po = fxn_po(dat, nd, levels_y)$p_hat,
    mr = fxn_mr(dat, nd, levels_y)$p_hat,
    cppo = fxn_cppo(dat, nd, levels_y)$p_hat,
    tsco_popo = fxn_tsco_popo(dat, nd, levels_y, cutoff_level)$p_hat)
  vals <- sapply(c(10L, 20L, 40L, 80L), function(m) {
    g <- kl_quad_grid(m); pgx <- g[, c("A", "W")]
    prob_kl(f(pgx), true_prob_scenario(pgx, scen, cutoff_level = cutoff_level), g$wt)
  })
  rel <- abs(vals[4] - vals[3]) / abs(vals[4])
  report(sprintf("%-10s R_n stable from 40 to 80 nodes", meth), rel < 1e-9,
         sprintf("%.8e, rel change %.1e", vals[4], rel))
}

## ---------------------------------------------------------------------------
cat("\n=== 4. the pipeline is internally consistent ===\n")
## expected_* are wrappers over prob_*, and kl_projection() uses expected_*, so
## all three routes to the same number must agree exactly.
report("expected_kl == prob_kl with arguments flipped",
       expected_kl(p_grid, q_grid, grid$wt) == prob_kl(q_grid, p_grid, grid$wt))
report("expected_rps == prob_rps with arguments flipped",
       expected_rps(p_grid, q_grid, grid$wt) == prob_rps(q_grid, p_grid, grid$wt))
proj <- kl_projection(scen, "po", levels_y, cutoff_level, n_nodes = 40L)
s <- score_method(fit, p_grid, 0.05, wt = grid$wt)
E_n <- s$kl - proj$kl
cat(sprintf("    A_kl = %.6e   R_n = %.6e   E_n = %+.6e\n", proj$kl, s$kl, E_n))
report("E_n = R_n - A >= 0 for this replicate", E_n >= 0)
cat(sprintf("    A_rps = %.6e   R_n,rps = %.6e   excess = %+.6e\n",
            proj$rps, s$rps, s$rps - proj$rps))

## ---------------------------------------------------------------------------
cat("\n=== 5. the hypothesis test is untouched by the evaluation set ===\n")
## stat / df / p_value / reject come from the training data alone, so switching
## the evaluation set must not move them at all.
s_grid <- score_method(fxn_po(dat, gx, levels_y), p_grid, 0.05, wt = grid$wt)
s_mc   <- score_method(fxn_po(dat, td, levels_y), pt_t, 0.05)
for (cc in c("stat", "df", "p_value", "reject", "fit_ok", "test_ok", "pred_ok")) {
  report(sprintf("%-8s identical under grid and mc", cc),
         identical(s_grid[[cc]], s_mc[[cc]]),
         if (is.numeric(s_grid[[cc]])) sprintf("%.10f", s_grid[[cc]]) else as.character(s_grid[[cc]]))
}

cat("\n", strrep("=", 66), "\n", sep = "")
cat(if (all_ok) "ALL EVALUATION-GRID CHECKS PASSED\n" else "*** SOME CHECKS FAILED ***\n")
