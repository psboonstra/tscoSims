suppressMessages({library(tsco); library(VGAM); library(dplyr); library(tibble)})
options(warn = -1)
levels_y <- as.character(0:5); K <- 6L
source("sim_functions/true_probs.R"); source("aux_functions/safe_fit.R")
source("aux_functions/align_prob.R"); source("aux_functions/vglm_helpers.R")
source("methods/cppo.R")
G <- cppo_G(levels_y)
r <- readRDS("scratch_vgam_boundary.rds")
crit <- qchisq(0.95, 2)

cat("=== exact counts, nrep =", nrow(r), "===\n")
cat("VGAM errored:      ", sum(r$vgam_err), "\n")
cat("VGAM invalid:      ", sum(!r$vgam_valid), "  (includes the errors)\n")
cat("fallback would fire:", sum(!r$vgam_valid), sprintf(" = %.4f\n", mean(!r$vgam_valid)))
cat("empty exposed cell:", sum(r$n_exp_cell == 0), sprintf(" = %.4f\n", mean(r$n_exp_cell == 0)))
cat("\nrejections (chisq_2):\n")
cat(sprintf("  direct, all %d reps:            %3d  %.4f\n",
            nrow(r), sum(r$direct_stat > crit), mean(r$direct_stat > crit)))
ok <- !r$vgam_err
cat(sprintf("  direct, VGAM-ok subset (%d):   %3d  %.4f\n",
            sum(ok), sum(r$direct_stat[ok] > crit), mean(r$direct_stat[ok] > crit)))
cat(sprintf("  vgam,   VGAM-ok subset (%d):   %3d  %.4f\n",
            sum(ok), sum(r$vgam_stat[ok] > crit), mean(r$vgam_stat[ok] > crit)))
cat(sprintf("  direct, VGAM-ERRORED (%d):      %3d  %.4f   <- the ones VGAM loses\n",
            sum(!ok), sum(r$direct_stat[!ok] > crit), mean(r$direct_stat[!ok] > crit)))

cat("\n=== agreement, VGAM vs constrained direct, on the reps VGAM fit ===\n")
g <- abs(r$vgam_stat[ok] - r$direct_stat[ok])
cat(sprintf("  max |gap| = %.3e   median = %.3e   (n = %d)\n", max(g), median(g), length(g)))
cat(sprintf("  max |gap| among VGAM-INVALID-but-not-errored (%d) = %.3e\n",
            sum(!r$vgam_valid & ok), max(abs(r$vgam_stat[!r$vgam_valid & ok] - r$direct_stat[!r$vgam_valid & ok]))))
cat(sprintf("  max |gap| among empty-cell & fit (%d) = %.3e\n",
            sum(r$n_exp_cell == 0 & ok), max(g[r$n_exp_cell[ok] == 0])))
cat(sprintf("  any rep where the two straddle the critical value? %s\n",
            any((r$vgam_stat[ok] > crit) != (r$direct_stat[ok] > crit))))

cat("\n=== how negative do VGAM's fitted probabilities get? ===\n")
inv <- !r$vgam_valid & ok
cat(sprintf("  invalid-but-fit reps: %d\n", sum(inv)))
cat(sprintf("  min fitted prob over those: %.3e   median: %.3e\n",
            min(r$vgam_minmu[inv]), median(r$vgam_minmu[inv])))
cat(sprintf("  how many have min fitted prob < -1e-6 ? %d\n", sum(r$vgam_minmu[inv] < -1e-6)))
cat(sprintf("  how many have min fitted prob < -1e-4 ? %d\n", sum(r$vgam_minmu[inv] < -1e-4)))
cat(sprintf("  how many were flagged for iter >= maxit only? %d\n",
            sum(r$vgam_iter[inv] >= 30)))
cat("  quantiles of min fitted prob among invalid fits:\n")
print(signif(quantile(r$vgam_minmu[inv], c(0,.1,.25,.5,.75,.9,1)), 3))

cat("\n=== boundary vs empty cell (constrained fit) ===\n")
print(table(empty = r$n_exp_cell == 0, on_boundary = r$dist_bdry < 1e-6))

cat("\n=== the VGAM-errored replicates: what went wrong ===\n")
bad_i <- r$i[r$vgam_err]
cat("  i =", paste(bad_i, collapse = ", "), "\n\n")
for (i in bad_i) {
  set.seed(20260710 + 100000L + i)
  dat <- make_covariates(200)
  dat$Y <- draw_ordinal(true_prob_scenario(dat, "null"), levels_y = levels_y)
  tb <- table(A = dat$A, Y = factor(as.character(dat$Y), levels = levels_y))
  ff <- safe_fit(VGAM::vglm(Y ~ A + W, data = dat,
          family = VGAM::cumulative(link = "logitlink", parallel = FALSE, reverse = FALSE),
          constraints = cppo_constraints(levels_y, G, with_A = TRUE)))
  fr <- safe_fit(VGAM::vglm(Y ~ W, data = dat,
          family = VGAM::cumulative(link = "logitlink", parallel = FALSE, reverse = FALSE),
          constraints = cppo_constraints(levels_y, G, with_A = FALSE)))
  which_failed <- paste(c(if (!ff$ok) "full", if (!fr$ok) "reduced"), collapse = "+")
  msg <- c(if (!ff$ok) ff$fit$error, if (!fr$ok) fr$fit$error)[1]
  cat(sprintf("  i = %-4d %s failed | counts A=1: %s | A=0: %s\n", i, which_failed,
              paste(tb["1", ], collapse = ","), paste(tb["0", ], collapse = ",")))
  cat(sprintf("           %s\n", substr(msg, 1, 150)))
  cat(sprintf("           direct LRT = %.4f  reject = %s  dist_bdry = %.2e\n",
              r$direct_stat[r$i == i], r$direct_stat[r$i == i] > crit, r$dist_bdry[r$i == i]))
}

cat("\n=== what are VGAM's coefficient names (gamma extraction failed) ===\n")
set.seed(20260710 + 100001L)
dat <- make_covariates(200)
dat$Y <- draw_ordinal(true_prob_scenario(dat, "null"), levels_y = levels_y)
f <- VGAM::vglm(Y ~ A + W, data = dat,
       family = VGAM::cumulative(link = "logitlink", parallel = FALSE, reverse = FALSE),
       constraints = cppo_constraints(levels_y, G, with_A = TRUE))
print(round(coef(f), 4))
