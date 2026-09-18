# Checks the CPPO Rao score test (methods/cppo.R, cppo_score_test) added
# 2026-09-18.
#
#   NSIM=400 Rscript verification/r_cppo_score_check.R            # null, n = 200
#   SCENARIO=cppo_alt N=1000 NSIM=100 Rscript verification/r_cppo_score_check.R
#
# What must hold:
#   0. The observed information used by the test agrees with a direct
#      second-difference Hessian of the nll (independent of the gradient code).
#   1. Nuisance components of the score are ~0 at the reduced fit.
#   2. Under the null, the score statistic is chi-square_2 to within Monte
#      Carlo error, INCLUDING on draws where the LRT sits on the boundary.
#   3. On interior draws at large n, score and LRT statistics agree (first-order
#      equivalence); the gap shrinks with n.
#   4. The EXPECTED information used by the test equals the covariance of the
#      score under the null: simulate Y | X at theta0 many times, compare the
#      empirical score covariance with I, and check the mean score is ~0.
#   5. The Schur-complement efficient-score statistic equals the full-vector
#      S' I^-1 S at the restricted MLE (where S_lambda = 0), to numerical
#      precision.

suppressMessages({library(tsco); library(VGAM); library(dplyr); library(tibble)})
options(warn = -1)

levels_y <- as.character(0:5); K <- length(levels_y); M <- K - 1L
master_seed <- 20260710; alpha <- 0.05
array_id <- as.integer(Sys.getenv("ARRAY", "1"))     # different ARRAY -> different seeds; pool for big runs
n <- as.integer(Sys.getenv("N", "200"))
scenario <- Sys.getenv("SCENARIO", "null")
nsim <- as.integer(Sys.getenv("NSIM", "400"))

source("sim_functions/score_method.R"); source("sim_functions/true_probs.R")
source("sim_functions/dataset_diagnostics.R")
source("aux_functions/safe_fit.R"); source("aux_functions/align_prob.R")
source("aux_functions/vglm_helpers.R"); source("aux_functions/unobserved_levels.R")
source("aux_functions/fit_converged.R")
source("methods/cppo.R")
G <- cppo_G(levels_y)
set.seed(master_seed + 999); test_dat <- make_covariates(50)

## 0. Information matrix against a direct second-difference Hessian of the nll
cat("=== 0. observed information vs second differences of the nll ===\n")
nll_nat <- function(theta, X, Yidx, a_col, w_cols, G, M) {
  alpha <- theta[1:M]; beta_w <- theta[M + seq_along(w_cols)]
  b1 <- theta[M + length(w_cols) + 1L]; b2 <- theta[M + length(w_cols) + 2L]
  eta <- drop(X[, w_cols, drop = FALSE] %*% beta_w)
  lin <- outer(-eta, alpha, "+") - outer(X[, a_col], b1 + b2 * G)
  pr <- cppo_direct_probs_from_q(plogis(lin))
  -sum(log(pmax(pr[cbind(seq_along(Yidx), Yidx)], 1e-300)))
}
set.seed(11); dat0 <- make_covariates(n)
dat0$Y <- draw_ordinal(true_prob_scenario(dat0, scenario), levels_y = levels_y)
Y0 <- match(as.character(dat0$Y), levels_y); X0 <- cbind(A = dat0$A, W = dat0$W)
red0 <- po_direct_fit(X0[, "W", drop = FALSE], Y0, K)
theta0 <- c(red0$alpha, red0$beta, 0, 0); p <- length(theta0)
# Hessian from the gradient (what the test uses)
h <- 1e-5 * pmax(1, abs(theta0)); Hg <- matrix(NA_real_, p, p)
for (k in 1:p) { tp <- theta0; tp[k] <- tp[k] + h[k]; tm <- theta0; tm[k] <- tm[k] - h[k]
  Hg[, k] <- (.cppo_natural_grad(tp, X0, Y0, 1L, 2L, G, M) - .cppo_natural_grad(tm, X0, Y0, 1L, 2L, G, M)) / (2 * h[k]) }
Hg <- (Hg + t(Hg)) / 2
# Hessian from second differences of the nll (independent of gradient code)
h2 <- 1e-3 * pmax(1, abs(theta0)); Hn <- matrix(NA_real_, p, p)
f0 <- nll_nat(theta0, X0, Y0, 1L, 2L, G, M)
for (i in 1:p) for (j in 1:p) {
  e_i <- e_j <- numeric(p); e_i[i] <- h2[i]; e_j[j] <- h2[j]
  Hn[i, j] <- (nll_nat(theta0 + e_i + e_j, X0, Y0, 1L, 2L, G, M) - nll_nat(theta0 + e_i - e_j, X0, Y0, 1L, 2L, G, M) -
               nll_nat(theta0 - e_i + e_j, X0, Y0, 1L, 2L, G, M) + nll_nat(theta0 - e_i - e_j, X0, Y0, 1L, 2L, G, M)) / (4 * h2[i] * h2[j])
}
rel <- max(abs(Hg - Hn)) / max(abs(Hn))
cat(sprintf("  max |H_grad - H_nll| / max|H| = %.2e   (min eigenvalue of H: %.3f)\n", rel, min(eigen(Hg, symmetric = TRUE)$values)))
# The 4-point second difference at h = 1e-3 carries ~1e-4 relative truncation
# error itself; a wrong gradient would be O(1) off, so 1e-3 catches bugs.
stopifnot("information matrix disagrees with second differences" = rel < 1e-3)

## 4. Expected information = Var(score) under the null, by simulation
cat("\n=== 4. expected information vs empirical score covariance (Y | X resampled at theta0) ===\n")
sc0 <- cppo_score_test(X0, Y0, 1L, G, K, red0)
I_exp <- {  # rebuild I exactly as cppo_score_test does, to compare against
  pr <- .cppo_natural_probs(theta0, X0, 1L, 2L, G, M)
  H <- matrix(0, p, p)
  for (k in seq_len(K)) { gk <- .cppo_natural_grad_rows(theta0, X0, rep(k, n), 1L, 2L, G, M)
    H <- H + crossprod(gk * sqrt(pmax(pr[, k], 0))) }
  H }
set.seed(12)
B <- 4000L
pr0 <- .cppo_natural_probs(theta0, X0, 1L, 2L, G, M)
cum0 <- t(apply(pr0, 1, cumsum))
scores <- matrix(NA_real_, B, p)
for (b in seq_len(B)) {
  u <- runif(n)
  Yb <- 1L + rowSums(cum0 < u)                # draw Y_i from pr0[i, ]
  Yb <- pmin(Yb, K)
  scores[b, ] <- -.cppo_natural_grad(theta0, X0, Yb, 1L, 2L, G, M)
}
V_emp <- stats::cov(scores)
# Entrywise standardized differences. Under normality of the scores the
# sampling variance of a covariance entry is (I_ii I_jj + I_ij^2) / (B - 1);
# the scores here are sums over n = 200 rows so that is a fair approximation.
# With p(p+1)/2 = 36 distinct entries, max |z| < 4 is a bug-catching bound
# (P(max of 36 |N(0,1)| > 4) ~ 0.002), not a loose smoke test.
se_cov <- sqrt((outer(diag(I_exp), diag(I_exp)) + I_exp^2) / (B - 1))
z_cov <- (V_emp - I_exp) / se_cov
rel_I <- max(abs(V_emp - I_exp)) / max(abs(I_exp))
mean_sc <- colMeans(scores) / sqrt(diag(I_exp) / B)     # standardized mean score
cat(sprintf("  B = %d   max|Var_emp(S) - I| / max|I| = %.3f;  entrywise standardized: max |z| = %.2f over %d entries (bound 4)\n",
    B, rel_I, max(abs(z_cov[upper.tri(z_cov, diag = TRUE)])), p * (p + 1) / 2))
cat(sprintf("  standardized mean score, max |z| over %d components = %.2f (bound 4)\n", p, max(abs(mean_sc))))
stopifnot("expected information disagrees with the empirical score covariance" =
            max(abs(z_cov[upper.tri(z_cov, diag = TRUE)])) < 4)
stopifnot("mean score not zero under the null" = max(abs(mean_sc)) < 4)
cat(sprintf("  I_eff: min eigenvalue %.3f, condition number %.1f;  I_nuis condition number %.1f\n",
    sc0$min_eig_eff, sc0$cond_eff, sc0$cond_nuis))

## 5. Schur-complement form vs full-vector form
cat("\n=== 5. efficient-score (Schur) statistic vs full-vector S' I^-1 S ===\n")
S_full <- sc0$score
stat_full <- drop(crossprod(S_full, solve(I_exp, S_full)))
cat(sprintf("  Schur %.6f   full-vector %.6f   |diff| = %.2e   (nuisance score / n = %.2e)\n",
    sc0$stat, stat_full, abs(sc0$stat - stat_full), sc0$max_nuisance_score_per_n))
stopifnot("Schur-complement and full-vector score statistics disagree" = abs(sc0$stat - stat_full) < 1e-6)

## Main loop
out <- NULL
for (i in 1:nsim) {
  set.seed(master_seed + array_id * 100000L + i)
  dat <- make_covariates(n)
  dat$Y <- draw_ordinal(true_prob_scenario(dat, scenario), levels_y = levels_y)
  dg <- dataset_diagnostics(dat, levels_y)
  Yidx <- match(as.character(dat$Y), levels_y); X <- cbind(A = dat$A, W = dat$W)

  red <- po_direct_fit(X[, "W", drop = FALSE], Yidx, K)
  sc <- cppo_score_test(X, Yidx, 1L, G, K, red)                    # expected information
  sc_obs <- cppo_score_test(X, Yidx, 1L, G, K, red, info = "observed")
  res <- fxn_cppo(dat, test_dat, levels_y)

  out <- bind_rows(out, tibble(
    i = i, n_empty_levels = dg$n_empty_levels, cell_dep_exp = dg$cell_dep_exp, cell_dep_unexp = dg$cell_dep_unexp,
    score_stat = sc$stat, score_p = sc$p, nuis = sc$max_nuisance_score,
    min_eig_eff = sc$min_eig_eff, cond_eff = sc$cond_eff, cond_nuis = sc$cond_nuis,
    stat_fullvec = if (all(is.finite(sc$score)) && !is.null(sc$I_eff)) {
      # full-vector form, rebuilt from the same expected information
      pr <- .cppo_natural_probs(c(red$alpha, red$beta, 0, 0), X, 1L, 2L, G, M)
      Hf <- matrix(0, length(sc$score), length(sc$score))
      for (k in seq_len(K)) { gk <- .cppo_natural_grad_rows(c(red$alpha, red$beta, 0, 0), X, rep(k, length(Yidx)), 1L, 2L, G, M)
        Hf <- Hf + crossprod(gk * sqrt(pmax(pr[, k], 0))) }
      tryCatch(drop(crossprod(sc$score, solve(Hf, sc$score))), error = function(e) NA_real_) } else NA_real_,
    score_obs_stat = sc_obs$stat, score_obs_p = sc_obs$p, red_boundary = red$on_boundary,
    lrt_stat = as.numeric(res$stat_alt), lrt_p = as.numeric(res$p_value_alt),
    boundary = isTRUE(res$boundary), fit_ok = res$fit_ok,
    fxn_score_p = as.numeric(res$p_value), fxn_test = res$test
  ))
  if (i %% 100 == 0) cat("  ...", i, "/", nsim, "\n")
}
saveRDS(out, sprintf("verification/cppo_score_check_%s_n%d_%d_a%d.rds", scenario, n, nsim, array_id))

cat("\n=== 1. nuisance score at the reduced fit ===\n")
cat(sprintf("  max |nuisance score| over draws: %.2e   (score components for b1, b2 are O(1)-O(10))\n", max(out$nuis)))
cat(sprintf("  draws with the reduced fit on the boundary (score test returned NA by fxn_cppo): %d\n", sum(out$red_boundary)))
reg <- out |> filter(!red_boundary)
cat(sprintf("  max |nuisance score| among regular draws: %.2e\n", max(reg$nuis)))
cat(sprintf("  fxn_cppo()$p_value_score == direct call on regular draws: %s\n",
    isTRUE(all.equal(reg$fxn_score_p, reg$score_p))))

schur_diff <- max(abs(reg$score_stat - reg$stat_fullvec), na.rm = TRUE)
cat(sprintf("  Schur vs full-vector statistic over regular draws: max |diff| = %.2e  (bound 1e-4)\n", schur_diff))
stopifnot("Schur-complement and full-vector forms disagree on some regular draw" = schur_diff < 1e-4)
cat(sprintf("  fxn_cppo primary test label: %s\n", paste(unique(reg$fxn_test), collapse = ", ")))
stopifnot("fxn_cppo primary test is not the score test" = all(reg$fxn_test == "score"))
cat(sprintf("  I_eff min eigenvalue over draws: min %.3f, median %.3f;  condition number: median %.1f, max %.1f\n",
    min(reg$min_eig_eff), median(reg$min_eig_eff), median(reg$cond_eff), max(reg$cond_eff)))
cat(sprintf("  I_nuis condition number: median %.1f, max %.1f;  score tests returned NA: %d\n",
    median(reg$cond_nuis), max(reg$cond_nuis), sum(!is.finite(reg$score_stat))))

cat("\n=== 2. rejection at alpha = 0.05 ===\n")
se <- function(p, m) sqrt(p * (1 - p) / m)
r_all <- reg |> summarize(m = n(), lrt = mean(lrt_p < alpha), score = mean(score_p < alpha), score_obs = mean(score_obs_p < alpha))
cat(sprintf("  regular draws (%d):   LRT %.3f (se %.3f)   score[expected info] %.3f (se %.3f)   score[observed info] %.3f\n",
    r_all$m, r_all$lrt, se(r_all$lrt, r_all$m), r_all$score, se(r_all$score, r_all$m), r_all$score_obs))
r_b <- reg |> group_by(boundary) |> summarize(m = n(), lrt = mean(lrt_p < alpha), score = mean(score_p < alpha),
                                              score_obs = mean(score_obs_p < alpha), .groups = "drop")
for (k in seq_len(nrow(r_b))) cat(sprintf("  LRT boundary = %-5s (%d):  LRT %.3f   score[exp] %.3f   score[obs] %.3f   [diagnostic split, not a reported stratum]\n",
    r_b$boundary[k], r_b$m[k], r_b$lrt[k], r_b$score[k], r_b$score_obs[k]))
if (scenario == "null") {
  ks <- suppressWarnings(ks.test(reg$score_stat, "pchisq", df = 2))
  cat(sprintf("  KS test of score statistic [expected info] against chi-square_2: D = %.3f, p = %.3f\n", ks$statistic, ks$p.value))
  ks2 <- suppressWarnings(ks.test(reg$lrt_stat, "pchisq", df = 2))
  cat(sprintf("  KS test of LRT statistic against chi-square_2:                  D = %.3f, p = %.3f\n", ks2$statistic, ks2$p.value))
}

cat("\n=== 3. score vs LRT statistic, interior draws ===\n")
int <- reg |> filter(!boundary)
cat(sprintf("  interior draws: %d   median |score - LRT| = %.3f   median |score - LRT| / LRT = %.3f   max = %.3f\n",
    nrow(int), median(abs(int$score_stat - int$lrt_stat)),
    median(abs(int$score_stat - int$lrt_stat) / pmax(int$lrt_stat, 1e-8)),
    max(abs(int$score_stat - int$lrt_stat))))
