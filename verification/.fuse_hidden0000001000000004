# Validation of sim_functions/kl_projection.R.
#
# Five checks, in increasing order of how much they would hurt if they failed:
#
#   1. the projection fitters really are the same model classes the methods fit
#   2. A(M) = 0 when the class contains the truth
#   3. A(M) reproduces the independently computed Python values
#   4. the quadrature has converged
#   5. the real fitter's risk approaches A(M) as n grows, from above
#
# Run from the tscoSims root:  Rscript verification/r_kl_projection_check.R
suppressMessages({library(tsco); library(VGAM); library(dplyr); library(tibble)})
options(warn = -1)

levels_y <- as.character(0:5); K <- length(levels_y); cutoff_level <- "5"
source("sim_functions/true_probs.R")
source("sim_functions/score_method.R")
source("aux_functions/safe_fit.R"); source("aux_functions/align_prob.R")
source("aux_functions/vglm_helpers.R")
source("methods/po.R"); source("methods/mr.R"); source("methods/cppo.R"); source("methods/tsco.R")
source("sim_functions/kl_projection.R")

pass <- function(ok) if (ok) "ok  " else "FAIL"
all_ok <- TRUE
report <- function(label, ok, detail = "") {
  cat(sprintf("  [%s] %-52s %s\n", pass(ok), label, detail))
  all_ok <<- all_ok && ok
}

## ---------------------------------------------------------------------------
cat("\n=== 0. the quadrature rule itself ===\n")
## Gauss-Hermite with m nodes is exact for polynomials of degree 2m-1, so the
## moments of the standard normal are the sharpest available test.
gh <- gauss_hermite_normal(20L)
mom <- function(j) sum(gh$weights * gh$nodes^j)
truth <- c(1, 0, 1, 0, 3, 0, 15, 0, 105)          # E[Z^0..Z^8]
got <- sapply(0:8, mom)
report("E[Z^j] for j = 0..8", max(abs(got - truth)) < 1e-12,
       sprintf("max dev %.2e", max(abs(got - truth))))
report("weights sum to 1", abs(sum(gh$weights) - 1) < 1e-14)
g <- kl_quad_grid(20L)
report("grid mass sums to 1", abs(sum(g$wt) - 1) < 1e-14,
       sprintf("%d grid points", nrow(g)))

## ---------------------------------------------------------------------------
cat("\n=== 1. do the projection fitters match the methods' own fits? ===\n")
## The projection fitters in kl_projection.R duplicate the model specification
## from methods/*.R. If they ever drift apart, A(M) is the projection onto the
## wrong class and every regret is wrong by an unknown amount. So: fit both on
## the same UNWEIGHTED dataset and compare coefficients. Equality here is the
## only thing tying the two files together.
set.seed(4321)
dat <- make_covariates(4000)
dat$Y <- draw_ordinal(true_prob_scenario(dat, "po_alt"), levels_y = levels_y)
dat$wt <- 1

cmp <- function(label, proj_fit, method_fit) {
  a <- coef(proj_fit); b <- coef(method_fit)
  ok <- length(a) == length(b) && max(abs(a - b)) < 1e-6
  report(label, ok, sprintf("max |diff| %.2e over %d coefs",
                            if (length(a) == length(b)) max(abs(a - b)) else NA, length(a)))
}
cmp("po   projection fitter == methods/po.R fit_full",
    kl_fit_po(dat, levels_y)$fit,
    VGAM::vglm(Y ~ A + W, data = dat,
      family = VGAM::cumulative(link = "logitlink", parallel = TRUE, reverse = FALSE)))
cmp("mr   projection fitter == methods/mr.R fit_full",
    kl_fit_mr(dat, levels_y)$fit,
    VGAM::vglm(Y ~ A + W, data = dat, family = VGAM::multinomial(refLevel = 1)))
cmp("cppo projection fitter == methods/cppo.R fit_full",
    kl_fit_cppo(dat, levels_y)$fit,
    VGAM::vglm(Y ~ A + W, data = dat,
      family = VGAM::cumulative(link = "logitlink", parallel = FALSE, reverse = FALSE),
      constraints = cppo_constraints(levels_y, cppo_G(levels_y), with_A = TRUE)))
t_proj <- kl_fit_tsco(dat, levels_y, cutoff_level, "po")$fit
t_meth <- tsco::tsco(Y ~ A + W, data = dat, levels = levels_y,
                     cutoff_level = cutoff_level, stage1 = "po", stage2 = "po",
                     warn_degenerate = FALSE)
cmp("tsco projection fitter == methods/tsco.R fit", t_proj, t_meth)

## Weight scale-invariance: A(M) must not depend on how the weights are scaled.
p1 <- kl_projection("po_alt", "po", levels_y, cutoff_level, n_nodes = 20L)
grid20 <- kl_quad_grid(20L)
ps <- kl_pseudo_data(grid20, true_prob_scenario(grid20, "po_alt"), levels_y)
ps$wt <- ps$wt * 1e4
f_scaled <- kl_fit_po(ps, levels_y)
kl_scaled <- expected_kl(true_prob_scenario(grid20, "po_alt"),
                         f_scaled$predict(grid20[, c("A","W")]), grid20$wt)
report("A(M) invariant to weight scaling (x 1e4)", abs(kl_scaled - p1$kl) < 1e-12,
       sprintf("%.6e vs %.6e", p1$kl, kl_scaled))

## ---------------------------------------------------------------------------
cat("\n=== 2. A(M) = 0 when the class contains the truth ===\n")
## These are the cases where the answer is known exactly, so they test the
## whole pipeline end to end rather than any one piece.
##   PO truth      -> PO exact; CPPO exact (set the deviation parameter to 0)
##   CPPO truth    -> CPPO exact
##   TsCO truth    -> the matching TsCO exact
zero_cases <- list(
  c("po_alt",   "po"),   c("po_alt",   "cppo"),
  c("cppo_alt", "cppo"),
  c("tsco_alt", "tsco_popo"), c("tsco_alt_stage2", "tsco_popo"),
  c("null",     "po"),   c("null",     "cppo")
)
## NOT in the list, deliberately: null -> tsco_popo. Under `null` the treatment
## effect is zero but W still enters proportionally, and a PO effect in a
## CONTINUOUS covariate is not representable by PO|5|PO. So TsCO is
## misspecified under `null` and A > 0. Check 2b below shows that the whole of
## that misspecification is the W effect, not the (absent) treatment effect.
## This is the population version of the size result in FINDINGS.md section 1.
for (cs in zero_cases) {
  r <- kl_projection(cs[1], cs[2], levels_y, cutoff_level, n_nodes = 40L)
  report(sprintf("%-16s -> %-10s A_kl = 0", cs[1], cs[2]), abs(r$kl) < 1e-9,
         sprintf("%.2e", r$kl))
}

cat("\n  and the cases that must be strictly positive:\n")
pos_cases <- list(c("po_alt","mr"), c("po_alt","tsco_popo"),
                  c("cppo_alt","po"), c("cppo_alt","tsco_popo"),
                  c("tsco_alt","po"), c("tsco_alt","cppo"),
                  c("none_true","po"), c("none_true","cppo"), c("none_true","tsco_popo"),
                  c("none_true","mr"))
for (cs in pos_cases) {
  r <- kl_projection(cs[1], cs[2], levels_y, cutoff_level, n_nodes = 40L)
  report(sprintf("%-16s -> %-10s A_kl > 0", cs[1], cs[2]), r$kl > 1e-12,
         sprintf("%.3e", r$kl))
}

## ---------------------------------------------------------------------------
cat("\n=== 2b. TsCO's misspecification under a PO truth is entirely the W effect ===\n")
## At BETA_W = 0 the design is the manuscript's: a single binary Z. There
## s takes two values and the stage-1 linear predictor has two free parameters,
## so PO is contained in PO|C|PO EXACTLY. Every nonzero A below is bought by
## the continuous W, and it grows with its coefficient.
bw_rows <- list()
for (bw in c(0, 0.25, 0.5, 1.0)) {
  r_null <- kl_projection("(direct)", "tsco_popo", levels_y, cutoff_level, 40L,
              p_true_fn = function(g) true_prob_po(g, b1 = 0,   beta_W = bw))
  r_alt  <- kl_projection("(direct)", "tsco_popo", levels_y, cutoff_level, 40L,
              p_true_fn = function(g) true_prob_po(g, b1 = 0.5, beta_W = bw))
  cat(sprintf("    BETA_W = %.2f   A(b1 = 0) = %.4e   A(b1 = 0.5) = %.4e\n",
              bw, r_null$kl, r_alt$kl))
  bw_rows[[length(bw_rows) + 1L]] <- c(bw, r_null$kl, r_alt$kl)
}
b <- do.call(rbind, bw_rows)
report("exact at BETA_W = 0 (single binary Z)", max(abs(b[1, 2:3])) < 1e-10,
       sprintf("max %.1e", max(abs(b[1, 2:3]))))
report("A increasing in BETA_W", all(diff(b[, 2]) > 0) && all(diff(b[, 3]) > 0))
report("the b1 = 0 and b1 = 0.5 columns are the same order",
       all(b[-1, 3] / b[-1, 2] < 4), "so it is W, not the treatment effect")

## ---------------------------------------------------------------------------
cat("\n=== 3. reproduce the independent Python values ===\n")
## From tscosims-nesting-correction, computed 2026-09-11 by a separate Python
## implementation (quadrature + analytic-gradient MLEs). PO truth, K = 6, ECMO
## baseline, beta_Z = 0.4, beta_W = 0.5. This is the only check here that is
## against an outside source rather than internal consistency.
stopifnot(BETA_W == 0.5)
po04 <- function(grid) true_prob_po(grid, b1 = 0.4)
targets <- list(
  list("cppo",      "3", 2.2e-16, "CPPO contains PO exactly"),
  list("tsco_pomr", "3", 1.9e-04, "PO|3|MR"),
  list("tsco_popo", "5", 1.3e-04, "PO|5|PO"),
  list("mr",        "5", 1.8e-04, "MR with linear W")
)
for (t in targets) {
  r <- kl_projection("(direct)", t[[1]], levels_y, cutoff_level = t[[2]],
                     n_nodes = 60L, p_true_fn = po04)
  tgt <- t[[3]]
  ok <- if (tgt < 1e-10) abs(r$kl) < 1e-10 else abs(r$kl - tgt) / tgt < 0.06
  report(sprintf("%-22s A_kl ~ %.1e", t[[4]], tgt), ok,
         sprintf("got %.3e", r$kl))
}

## ---------------------------------------------------------------------------
cat("\n=== 4. quadrature convergence ===\n")
## If A(M) still moves with the node count, the number is a property of the
## rule rather than of the model class.
conv_cases <- list(c("tsco_alt","po"), c("po_alt","tsco_popo"), c("none_true","mr"))
for (cs in conv_cases) {
  vals <- sapply(c(10L, 20L, 40L, 80L), function(m)
    kl_projection(cs[1], cs[2], levels_y, cutoff_level, n_nodes = m)$kl)
  rel <- abs(vals[4] - vals[3]) / max(abs(vals[4]), 1e-30)
  cat(sprintf("    %-16s -> %-10s  %s\n", cs[1], cs[2],
              paste(sprintf("%.6e", vals), collapse = "  ")))
  report(sprintf("  %s -> %s stable from 40 to 80 nodes", cs[1], cs[2]), rel < 1e-6,
         sprintf("rel change %.1e", rel))
}

## ---------------------------------------------------------------------------
cat("\n=== 5. does the real fitter's risk approach A(M) from above? ===\n")
## The decisive external check. A(M) is the infimum of E_X[KL(p||q_theta)] over
## the class, so for a consistent fitter R_n must fall towards it and never
## below. Asymptotically E[R_n - A] ~ tr(J H^{-1}) / (2n), so n*(R_n - A)
## should flatten at a constant -- Takeuchi's effective dimension over 2, which
## equals d/2 only under correct specification. R_n is computed as an exact
## expectation over the grid given thetahat, so the only stochastic input is
## the training sample.
grid <- kl_quad_grid(40L)
check5 <- function(scenario, method, fitter, ns = c(5000, 20000, 80000)) {
  A <- kl_projection(scenario, method, levels_y, cutoff_level, n_nodes = 40L)$kl
  p_grid <- true_prob_scenario(grid, scenario, cutoff_level = cutoff_level)
  cat(sprintf("\n  %s -> %s   A = %.6e\n", scenario, method, A))
  ok <- TRUE
  for (n in ns) {
    set.seed(90210 + n)
    d <- make_covariates(n)
    d$Y <- draw_ordinal(true_prob_scenario(d, scenario), levels_y = levels_y)
    fit <- fitter(d)
    if (!isTRUE(fit$fit_ok)) { cat(sprintf("    n = %7d  fit failed\n", n)); ok <- FALSE; next }
    R_n <- expected_kl(p_grid, fit$p_hat, grid$wt)
    E_n <- R_n - A
    cat(sprintf("    n = %7d  R_n = %.6e   E_n = %+.3e   n*E_n = %6.2f\n",
                n, R_n, E_n, n * E_n))
    ok <- ok && E_n > -1e-9
  }
  report(sprintf("  %s -> %s: E_n >= 0 at every n", scenario, method), ok)
}
check5("po_alt",   "tsco_popo", function(d) fxn_tsco_popo(d, grid[, c("A","W")], levels_y, cutoff_level))
check5("tsco_alt", "po",        function(d) fxn_po(d, grid[, c("A","W")], levels_y))
check5("po_alt",   "po",        function(d) fxn_po(d, grid[, c("A","W")], levels_y))
check5("none_true","mr",        function(d) fxn_mr(d, grid[, c("A","W")], levels_y))

cat("\n", strrep("=", 70), "\n", sep = "")
cat(if (all_ok) "ALL KL PROJECTION CHECKS PASSED\n" else "*** SOME CHECKS FAILED ***\n")
