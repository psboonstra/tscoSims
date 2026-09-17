# Regression test for the reworked methods/cppo.R.
#
# The point of the change is to fix WHICH replicates get tagged, not what the
# test statistic is. So the acceptance criterion is: every LRT must be
# unchanged, or changed only where the old code produced no usable statistic
# at all.
suppressMessages({library(tsco); library(VGAM); library(dplyr); library(tibble)})
options(warn = -1)
levels_y <- as.character(0:5); K <- 6L
source("sim_functions/score_method.R"); source("sim_functions/true_probs.R")
source("aux_functions/safe_fit.R"); source("aux_functions/align_prob.R")
source("aux_functions/vglm_helpers.R")

set.seed(20260710 + 999); test_dat <- make_covariates(200)
p_true_test <- true_prob_scenario(test_dat, "null")
nrep <- as.integer(Sys.getenv("NREP", "800"))

run <- function(src) {
  e <- new.env(parent = globalenv())
  sys.source(src, envir = e)
  out <- vector("list", nrep)
  for (i in seq_len(nrep)) {
    set.seed(20260710 + 100000L + i)
    dat <- make_covariates(200)
    dat$Y <- draw_ordinal(true_prob_scenario(dat, "null"), levels_y = levels_y)
    res <- e$fxn_cppo(dat, test_dat, levels_y)
    s <- score_method(res, p_true_test, 0.05)
    out[[i]] <- tibble(i = i,
      n_exp = sum(dat$A == 1 & dat$Y == "4"),
      n_unexp = sum(dat$A == 0 & dat$Y == "4"),
      fit_ok = s$fit_ok, test_ok = s$test_ok,
      stat = s$stat, reject = s$reject,
      bdry = grepl("boundary", s$warnings),
      refit = grepl("direct ML", s$warnings))
    if (i %% 200 == 0) cat("   ...", i, "\n", file = stderr())
  }
  bind_rows(out)
}

cat("--- old code ---\n");  o <- run("/tmp/cppo_orig.R")
cat("--- new code ---\n");  n <- run("methods/cppo.R")
stopifnot(identical(o$i, n$i))

loc <- factor(with(n, case_when(n_exp == 0 & n_unexp == 0 ~ "both empty",
                                n_exp == 0 ~ "exposed empty",
                                n_unexp == 0 ~ "unexposed empty",
                                TRUE ~ "neither empty")),
              levels = c("neither empty","exposed empty","unexposed empty","both empty"))

cat(sprintf("\n=== 1. are the test statistics unchanged? (%d reps) ===\n", nrep))
both <- is.finite(o$stat) & is.finite(n$stat)
cat(sprintf("  both usable: %d   max |change| = %.3e\n", sum(both),
            max(abs(o$stat[both] - n$stat[both]))))
cat(sprintf("  rejections: old %d, new %d, disagreements %d\n",
            sum(o$reject), sum(n$reject), sum(o$reject != n$reject)))
cat(sprintf("  usable under old but not new: %d\n", sum(is.finite(o$stat) & !is.finite(n$stat))))
cat(sprintf("  usable under new but not old: %d   <- replicates rescued\n",
            sum(!is.finite(o$stat) & is.finite(n$stat))))
res <- n[!is.finite(o$stat) & is.finite(n$stat), ]
if (nrow(res)) print(as.data.frame(res |> select(i, n_exp, n_unexp, stat, reject)))

cat("\n=== 2. boundary tag: old vs new, by stratum ===\n")
print(as.data.frame(tibble(loc = loc, old = o$bdry, new = n$bdry) |>
  group_by(loc) |> summarize(n = n(), old_tagged = sum(old), new_tagged = sum(new),
                             .groups = "drop")))

cat("\n=== 3. refits: old vs new, by stratum ===\n")
print(as.data.frame(tibble(loc = loc, old = o$refit, new = n$refit) |>
  group_by(loc) |> summarize(n = n(), old_refit = sum(old), new_refit = sum(new),
                             .groups = "drop")))

cat("\n=== 4. size by stratum, old vs new ===\n")
print(as.data.frame(tibble(loc = loc, o = o$reject, nn = n$reject) |>
  group_by(loc) |> summarize(n = n(), old = round(mean(o), 4), new = round(mean(nn), 4),
                             .groups = "drop")))
cat(sprintf("\n  overall size: old %.4f   new %.4f\n", mean(o$reject), mean(n$reject)))

cat("\n=== 5. the both-empty replicates no longer reach VGAM ===\n")
be <- which(loc == "both empty")
if (length(be)) print(as.data.frame(n[be, ] |> select(i, n_exp, n_unexp, fit_ok, test_ok, stat, refit, bdry)))
