# Checks on the data-generating mechanisms. Run this after any change to
# true_probs.R:
#
#   source("sim_functions/true_probs.R"); source("sim_functions/check_dgm.R")
#
# Three things are verified:
#   1. every DGM sits exactly on BASELINE_PROBS at A = 0, W = 0;
#   2. the PO, CPPO and TsCO parameterisations reproduce the manuscript's
#      Tables 3 and 4 to the three decimals printed there;
#   3. the implied MARGINAL distribution of each scenario is reported, because
#      matching at W = 0 does not make the marginals match -- and it is the
#      marginal that determines realised cell counts and hence separation.

ok <- TRUE

report <- function(label, pass, detail = "") {
  cat(sprintf("  [%s] %s%s\n", if (pass) "ok  " else "FAIL", label,
              if (nzchar(detail)) paste0("  ", detail) else ""))
  pass
}

at_origin <- function(f) {
  as.numeric(f(data.frame(A = 0, W = 0)))
}

cat("\n=== 1. common baseline at A = 0, W = 0 ===\n")

baseline_checks <- list(
  PO           = function(d) true_prob_po(d, b1 = 0.5),
  `PO (null)`  = function(d) true_prob_po(d, b1 = 0),
  CPPO         = function(d) true_prob_cppo(d, b1 = 0.5, b2 = -0.1),
  `TsCO PO|5|PO` = function(d) true_prob_tsco(d, b1 = 0.5, b2 = 0.15, stage2 = "po"),
  `TsCO PO|5|MR` = function(d) true_prob_tsco(d, b1 = 0.5, b2 = 0.15, stage2 = "mr"),
  `TsCO PO|3|MR` = function(d) true_prob_tsco(d, b1 = 0.5, b2 = c(0.1, 0.2, 0.15),
                                              cutoff_level = "3", stage2 = "mr"),
  `vectors`    = function(d) true_prob_vectors(
                                d, p_treat = c(0.290, 0.110, 0.050, 0.090, 0.020, 0.440))
)

for (nm in names(baseline_checks)) {
  got <- at_origin(baseline_checks[[nm]])
  dev <- max(abs(got - BASELINE_PROBS))
  ok <- report(sprintf("%-14s baseline", nm), dev < 1e-10,
               sprintf("max dev %.2e", dev)) && ok
}

cat("\n=== 2. reproduce manuscript Table 4 (CPPO) ===\n")
cat("    Pr(Y >= k | Z = 1) = expit(a_k - b1 - 1[k = K-1] b2)\n")

tab4 <- list(
  list(0.0,  0.00, c(0.270, 0.060, 0.060, 0.110, 0.030, 0.470)),
  list(0.5,  0.00, c(0.379, 0.069, 0.065, 0.109, 0.028, 0.350)),
  list(1.0,  0.00, c(0.501, 0.071, 0.062, 0.096, 0.023, 0.246)),
  list(1.5,  0.00, c(0.624, 0.064, 0.053, 0.076, 0.017, 0.165)),
  list(0.0, -0.05, c(0.270, 0.060, 0.060, 0.110, 0.018, 0.482)),
  list(1.0, -0.05, c(0.501, 0.071, 0.062, 0.096, 0.014, 0.255)),
  list(0.0, -0.12, c(0.270, 0.060, 0.060, 0.110, 0.000, 0.500)),
  list(1.5, -0.12, c(0.624, 0.064, 0.053, 0.076, 0.000, 0.182))
)

for (row in tab4) {
  got <- as.numeric(true_prob_cppo(data.frame(A = 1, W = 0),
                                   b1 = row[[1]], b2 = row[[2]], beta_W = 0))
  dev <- max(abs(got - row[[3]]))
  ok <- report(sprintf("b1 = %-4s b2 = %-6s", row[[1]], row[[2]]), dev < 6e-4,
               sprintf("max dev %.1e", dev)) && ok
}

cat("\n=== 3. reproduce manuscript Table 3 (PO|5|MR) ===\n")
cat("    Pr(Y >= k | Y < 5, Z = 1) = expit(a_k - b1);  Pr(Y = 5 | Z = 1) = expit(a_5 - b2)\n")

tab3 <- list(
  list(0.000, 0.00, c(0.270, 0.060, 0.060, 0.110, 0.030, 0.470)),
  list(0.296, 0.50, c(0.379, 0.069, 0.065, 0.109, 0.028, 0.350)),
  list(0.648, 1.00, c(0.501, 0.071, 0.062, 0.096, 0.023, 0.246)),
  list(1.046, 1.50, c(0.624, 0.064, 0.053, 0.076, 0.017, 0.165)),
  list(0.500, 0.00, c(0.335, 0.053, 0.048, 0.076, 0.019, 0.470)),
  list(1.500, 0.15, c(0.467, 0.033, 0.026, 0.035, 0.007, 0.433))
)

for (row in tab3) {
  for (s2 in c("po", "mr")) {
    got <- as.numeric(true_prob_tsco(data.frame(A = 1, W = 0),
                                     b1 = row[[1]], b2 = row[[2]],
                                     cutoff_level = "5", stage2 = s2,
                                     beta1_W = 0, beta2_W = 0))
    dev <- max(abs(got - row[[3]]))
    ok <- report(sprintf("b1 = %-6s b2 = %-5s stage2 = %s", row[[1]], row[[2]], s2),
                 dev < 1e-3, sprintf("max dev %.1e", dev)) && ok
  }
}

cat("\n  (PO|5|* has a single level in the upper partition, so the MR and PO\n")
cat("   stage-2 families coincide there -- the manuscript writes this as\n")
cat("   'P O|5|M R (and P O|5|P O)'. They differ only for a smaller cutoff.)\n")

cat("\n=== 4. scenarios reproduce manuscript Table 5 ===\n")
cat("    These are the rows the simulation actually uses, at W = 0.\n")

# NOTE ON A TYPO IN THE MANUSCRIPT. Table 5 row 5 (CPPO {0.5, -0.1}) prints its
# first entry as 0.370. That must be 0.379: it is the only one of the twelve
# rows that does not sum to 1 (it sums to 0.991), and the paper's own text
# states that b2 does not affect any category below the second-to-highest, so
# entry 0 has to equal its b2 = 0 value of 0.379. Asserted as 0.379 here.
tab5 <- list(
  list("null",            c(0.270, 0.060, 0.060, 0.110, 0.030, 0.470)),
  list("po_alt",          c(0.379, 0.069, 0.065, 0.109, 0.028, 0.350)),
  list("cppo_alt",        c(0.379, 0.069, 0.065, 0.109, 0.005, 0.373)),
  list("tsco_alt",        c(0.358, 0.057, 0.051, 0.081, 0.020, 0.433)),
  list("tsco_alt_stage2", c(0.301, 0.067, 0.067, 0.123, 0.033, 0.409)),
  list("none_true",       c(0.290, 0.110, 0.050, 0.090, 0.020, 0.440))
)

old_beta_W <- BETA_W
BETA_W <<- 0   # Table 5 has no W; compare at the manuscript's design
for (row in tab5) {
  got <- as.numeric(true_prob_scenario(data.frame(A = 1, W = 0), row[[1]]))
  dev <- max(abs(got - row[[2]]))
  ok <- report(sprintf("%-16s treatment arm", row[[1]]), dev < 1e-3,
               sprintf("max dev %.1e", dev)) && ok
}
BETA_W <<- old_beta_W

cat("\n=== 5. implied marginal distributions (NOT the baseline) ===\n")
cat("    Matching at W = 0 does not match the marginals. These are what drive\n")
cat("    realised cell counts, and so separation and finite-sample behaviour.\n\n")

set.seed(1)
big <- make_covariates(400000)
scenarios <- c("null", "po_alt", "cppo_alt", "tsco_alt", "tsco_alt_stage2", "none_true")

cat(sprintf("  %-17s %-7s %s\n", "scenario", "arm", "marginal Pr(Y = k), k = 0..5"))
cat("  ", strrep("-", 62), "\n", sep = "")
for (sc in scenarios) {
  p <- true_prob_scenario(big, sc)
  for (a in c(0, 1)) {
    m <- colMeans(p[big$A == a, , drop = FALSE])
    cat(sprintf("  %-17s %-7s %s\n", if (a == 0) sc else "", paste0("A=", a),
                paste(sprintf("%.3f", m), collapse = " ")))
  }
}

cat(sprintf("\n  Baseline for reference:      %s\n",
            paste(sprintf("%.3f", BASELINE_PROBS), collapse = " ")))
cat(sprintf("  BETA_W = %s (set to 0 to recover the manuscript's Z-only design)\n", BETA_W))

cat("\n  Smallest expected cell count at n = 200, by scenario and arm:\n")
for (sc in scenarios) {
  p <- true_prob_scenario(big, sc)
  mn <- min(colMeans(p[big$A == 0, , drop = FALSE]),
            colMeans(p[big$A == 1, , drop = FALSE]))
  cat(sprintf("    %-17s %5.1f  (per 100 per arm)\n", sc, mn * 100))
}

cat("\n", if (ok) "ALL DGM CHECKS PASSED" else "SOME DGM CHECKS FAILED", "\n\n", sep = "")
