# Coefficient-based admissibility check vs the fitted-probability check.
#
# On VGAM's scale (cumulative(reverse = FALSE), so eta_j = logit Pr(Y <= j) with
# INCREASING intercepts b0_1..b0_M), the CPPO model is admissible iff, for every
# row, eta is nondecreasing in j. With eta_j = b0_j + A(b_common + G_j b_dev) +
# W b_W the W and common-A terms cancel out of every difference, leaving
#
#   unexposed rows:  diff(b0)_j                        >= 0   for j = 1..M-1
#   exposed rows:    diff(b0)_j + (G_{j+1} - G_j) b_dev >= 0   for j = 1..M-1
#
# Two numbers, no n, no W, works for any G. This script checks that the minimum
# of those slacks classifies replicates the same way the probability check does,
# and that it also detects the face the probability check misses.
suppressMessages({library(tsco); library(VGAM); library(dplyr); library(tibble)})
options(warn = -1)
levels_y <- as.character(0:5); K <- 6L; M <- K - 1L
source("sim_functions/true_probs.R"); source("aux_functions/safe_fit.R")
source("aux_functions/align_prob.R"); source("aux_functions/vglm_helpers.R")
source("methods/cppo.R")
G <- cppo_G(levels_y)
nrep <- as.integer(Sys.getenv("NREP", "800"))

# The two slacks, from the coefficients alone.
cppo_coef_slacks <- function(fit, G) {
  cf <- coef(fit)
  b0 <- cf[grep("^\\(Intercept\\)", names(cf))]
  dev <- if ("A:2" %in% names(cf)) cf[["A:2"]] else 0   # deviation column of the A constraint
  list(unexposed = min(diff(b0)),
       exposed   = min(diff(b0) + diff(G) * dev))
}

out <- vector("list", nrep)
for (i in seq_len(nrep)) {
  set.seed(20260710 + 100000L + i)
  dat <- make_covariates(200)
  dat$Y <- draw_ordinal(true_prob_scenario(dat, "null"), levels_y = levels_y)
  ne <- sum(dat$A == 1 & dat$Y == "4"); nu <- sum(dat$A == 0 & dat$Y == "4")

  ff <- safe_fit(VGAM::vglm(Y ~ A + W, data = dat,
          family = VGAM::cumulative(link = "logitlink", parallel = FALSE, reverse = FALSE),
          constraints = cppo_constraints(levels_y, G, with_A = TRUE)))

  s_un <- s_ex <- p_min <- p_min_exposed <- NA_real_
  prob_valid <- NA
  if (ff$ok) {
    sl <- cppo_coef_slacks(ff$fit, G)
    s_un <- sl$unexposed; s_ex <- sl$exposed
    mu <- ff$fit@fitted.values
    p_min <- min(mu)
    p_min_exposed <- min(mu[dat$A == 1, K - 1L])   # what fxn_cppo actually looks at
    prob_valid <- cppo_vglm_valid(ff$fit)
  }
  out[[i]] <- tibble(i = i, n_exp = ne, n_unexp = nu, vgam_ok = ff$ok,
                     slack_unexp = s_un, slack_exp = s_ex,
                     p_min = p_min, p_min_exposed = p_min_exposed,
                     prob_valid = prob_valid)
  if (i %% 200 == 0) cat("  ...", i, "\n", file = stderr())
}
r <- bind_rows(out) |> mutate(
  loc = factor(case_when(n_exp == 0 & n_unexp == 0 ~ "both empty",
                         n_exp == 0 ~ "exposed empty",
                         n_unexp == 0 ~ "unexposed empty",
                         TRUE ~ "neither empty"),
               levels = c("neither empty","exposed empty","unexposed empty","both empty")),
  slack_min = pmin(slack_unexp, slack_exp))
saveRDS(r, "scratch_coefcheck.rds")
f <- r |> filter(vgam_ok)

cat(sprintf("\n=== %d reps, %d fit by VGAM ===\n", nrep, nrow(f)))
cat("\n=== the two coefficient slacks by stratum (median [min, max]) ===\n")
print(as.data.frame(f |> group_by(loc) |> summarize(n = n(),
  unexposed_face = sprintf("%.2e [%.1e, %.1e]", median(slack_unexp), min(slack_unexp), max(slack_unexp)),
  exposed_face   = sprintf("%.2e [%.1e, %.1e]", median(slack_exp), min(slack_exp), max(slack_exp)),
  .groups = "drop")))

cat("\n=== does a single threshold on the coefficient slack recover the strata? ===\n")
for (eps in c(1e-4, 1e-3, 1e-2)) {
  cat(sprintf("  slack < %.0e :\n", eps))
  print(table(stratum = f$loc, flagged = f$slack_min < eps))
}

cat("\n=== which face is active, by stratum (threshold 1e-3) ===\n")
print(as.data.frame(f |> group_by(loc) |> summarize(n = n(),
  unexp_face_active = sum(slack_unexp < 1e-3),
  exp_face_active   = sum(slack_exp   < 1e-3), .groups = "drop")))

cat("\n=== the probability check, for comparison ===\n")
print(as.data.frame(f |> group_by(loc) |> summarize(n = n(),
  prob_valid_TRUE = sum(prob_valid),
  fxn_cppo_bdry_tag = sum(p_min_exposed < 1e-6),
  coef_check_flag = sum(slack_min < 1e-3), .groups = "drop")))

cat("\n=== why the probability scale is a poor proxy: same slack, different probability ===\n")
act <- f |> filter(slack_min > 0, slack_min < 0.5) |>
  mutate(ratio = p_min / slack_min)
cat(sprintf("  over %d fits with slack in (0, 0.5): p_min / slack spans %.2e to %.2e,\n",
            nrow(act), min(act$ratio), max(act$ratio)))
cat(sprintf("  a factor of %.0f between the smallest and largest conversion.\n",
            max(act$ratio) / max(min(act$ratio), 1e-30)))
q <- quantile(act$ratio, c(.05,.25,.5,.75,.95))
cat("  quantiles of p_min / slack: ", paste(sprintf("%.2e", q), collapse = "  "), "\n")
