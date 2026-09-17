## ----message = F, warning = F----
library(dplyr)
library(tidyr)
library(tibble)

## -----------------------------
# The method every other method is compared against in the paired summary.
reference_method <- "po"

# WHY EVERY SUMMARY HERE IS UNCONDITIONAL
# ---------------------------------------
# `all_scores` carries per-replicate sparsity diagnostics (`n_empty_levels`,
# `cell_dep_exp`, `cell_dep_unexp`, `min_cell`). They are there for DIAGNOSIS --
# to explain why an unconditional defect exists and which fix to attempt -- and
# deliberately do NOT define reporting strata.
#
# Splitting a reported size or power on them would condition on a function of
# the realized outcome. For power that is fatal: under `cppo_alt` the emptiness
# of the departure cell is driven by b2, the very effect under test, so it is a
# post-treatment variable and the strata are not conditions an analyst could be
# in ex ante. For size it is merely unsound: size is P(reject | H0) over the
# joint distribution, which is what is bounded by alpha, and conditional size at
# a particular value of a data statistic has no obligation to equal alpha --
# non-constant conditional size is generic, and the reductio is conditioning on
# the p-value itself, where it is 0 or 1.
#
# Sparsity enters honestly as a DESIGN axis instead: n. See the design summary
# at the bottom, which reports empty-cell rates as a property of each
# (scenario, n) cell rather than as a partition of the replicates.

files <- list.files("out", pattern = "^job[0-9]+_scores\\.csv$", full.names = TRUE)

if (length(files) == 0L) {
  stop("No score files found in out/.")
}

all_scores <-
  bind_rows(lapply(files, read.csv, stringsAsFactors = FALSE)) %>%
  as_tibble() %>%
  # read.csv turns the empty warning string into NA; put it back so that
  # `warnings` is a character column with no missing values.
  mutate(warnings = ifelse(is.na(warnings), "", warnings))

dir.create("summaries", recursive = TRUE, showWarnings = FALSE)
saveRDS(all_scores, "summaries/all_scores.rds")


## -----------------------------
# Provenance check. Pooling score files produced by different tsco builds is a
# silent way to average across a bug fix, so this is an error, not a note.
if (all(c("tsco_commit", "tsco_dirty") %in% names(all_scores))) {

  prov <- all_scores %>% distinct(tsco_version, tsco_commit, tsco_dirty)

  if (nrow(prov) > 1L) {
    print(prov)
    stop("Score files were produced by more than one tsco build (see above).")
  }

  cat("tsco build: version ", prov$tsco_version,
      ", commit ", ifelse(is.na(prov$tsco_commit), "UNRECORDED", prov$tsco_commit),
      ifelse(isTRUE(prov$tsco_dirty), " (MODIFIED working tree)", ""), "\n", sep = "")

  if (is.na(prov$tsco_commit)) {
    warning("These results do not identify the tsco source that produced them.",
            call. = FALSE)
  }
} else {
  warning("Score files predate the provenance columns; the tsco build is unknown.",
          call. = FALSE)
}


## -----------------------------
# The population approximation error A(M) of each model class under each DGM,
# computed exactly in sim_functions/kl_projection.R. The simulation already
# reports R_n -- `kl` and `rps` are excess risks relative to the oracle -- so
# the estimation error is one join and one subtraction:
#
#     E_n^KL     = kl  - A_kl      >= 0 in population; E[E_n] ~ tr(J H^-1)/(2n)
#     excess^RPS = rps - A_rps     -> 0, and CAN dip below zero in finite samples
#
# The RPS residual can go negative because A_rps is evaluated at theta*_KL, not
# minimised over the class: minimising it would make R_n - A_rps converge to a
# positive constant instead of zero, since thetahat converges to theta*_KL. So a
# small negative value is the honest answer and is not clipped.
A_path <- "summaries/A_approximation_error.rds"

if (!file.exists(A_path)) {
  stop("Missing ", A_path, ". Build it with sim_functions/kl_projection.R ",
       "before processing results.")
}

A_tab <- readRDS(A_path) %>%
  as_tibble() %>%
  select(scenario, method, A_kl, A_rps, A_brier)

missing_A <-
  all_scores %>%
  distinct(scenario, method) %>%
  anti_join(A_tab, by = c("scenario", "method"))

if (nrow(missing_A) > 0L) {
  print(missing_A)
  stop("No approximation error A(M) for the (scenario, method) pairs above.")
}

scored <-
  all_scores %>%
  left_join(A_tab, by = c("scenario", "method")) %>%
  mutate(
    en_kl = kl - A_kl,
    excess_rps = rps - A_rps,
    excess_brier = brier - A_brier
  )

saveRDS(scored, "summaries/all_scores_decomposed.rds")


## -----------------------------
# Marginal summary. Every mean carries a Monte Carlo standard error, and the
# three success rates are reported separately, because rejection rates are
# conditional on the test having been computed: a method that fails on exactly
# the hard replicates will look better than one that struggles through them.
# KL is summarised by the median, since a single separated fit makes the mean
# infinite.
summ <-
  scored %>%
  group_by(scenario, n, method) %>%
  summarize(
    n_rep = n(),
    fit_ok_rate = mean(fit_ok),
    test_ok_rate = mean(test_ok),
    pred_ok_rate = mean(pred_ok),
    rejection = mean(reject[test_ok]),
    rejection_mcse = sqrt(rejection * (1 - rejection) / sum(test_ok)),
    mean_rps = mean(rps[pred_ok]),
    rps_mcse = sd(rps[pred_ok]) / sqrt(sum(pred_ok)),
    mean_brier = mean(brier[pred_ok]),
    brier_mcse = sd(brier[pred_ok]) / sqrt(sum(pred_ok)),
    mean_mae = mean(mae[pred_ok]),
    median_kl = median(kl[pred_ok]),
    frac_kl_infinite = mean(!is.finite(kl[pred_ok])),
    .groups = "drop"
  )

saveRDS(summ, "summaries/summary_results.rds")
write.csv(summ, "summaries/summary_results.csv", row.names = FALSE)


## -----------------------------
# The regret decomposition: R_n = A + E_n, with A known exactly.
#
# n * E_n^KL is the effective-dimension diagnostic. If n*E_n converges in
# distribution to (1/2) chi-square_d, then E[n E_n] -> d/2, and that is the
# quantity to compare against the parameter count -- 3.5 for a correctly
# specified 7-parameter PO fit.
#
# MEAN VERSUS MEDIAN, because the two are NOT interchangeable here. A single
# separated fit makes `kl` infinite, so the mean is reported over finite
# replicates only, with `frac_kl_infinite` alongside to say how much was
# dropped. The median is robust but targets a different number: the median of
# (1/2) chi-square_d is about (d/2)(1 - 2/(9d))^3, which is 3.18 against a mean
# of 3.5 at d = 7. A ~10% gap between the two columns is that, not a bug.
decomp <-
  scored %>%
  filter(pred_ok) %>%
  group_by(scenario, n, method) %>%
  summarize(
    n_rep = n(),
    A_kl = first(A_kl),
    median_en_kl = median(en_kl),
    n_median_en_kl = first(n) * median(en_kl),
    frac_kl_infinite = mean(!is.finite(en_kl)),
    mean_en_kl_finite = mean(en_kl[is.finite(en_kl)]),
    en_kl_mcse = sd(en_kl[is.finite(en_kl)]) / sqrt(sum(is.finite(en_kl))),
    n_mean_en_kl_finite = first(n) * mean(en_kl[is.finite(en_kl)]),
    A_rps = first(A_rps),
    mean_excess_rps = mean(excess_rps),
    excess_rps_mcse = sd(excess_rps) / sqrt(n()),
    # Share of total RPS regret that is approximation rather than estimation.
    # Undefined when the total is ~0, so guarded.
    rps_approx_share = ifelse(mean(rps) > 0, first(A_rps) / mean(rps), NA_real_),
    .groups = "drop"
  )

saveRDS(decomp, "summaries/decomposition_results.rds")
write.csv(decomp, "summaries/decomposition_results.csv", row.names = FALSE)


## -----------------------------
# Paired summary. Every method sees the same dataset within a replicate, so
# differencing on `data_seed` before averaging removes the replicate-level
# variation that is common to all methods. The resulting Monte Carlo error is
# typically far smaller than that of the difference of two marginal means, and
# it is what decides whether the simulation can separate the methods at this
# number of replicates.
#
# `n_paired` matters: a seed contributes only if both methods predicted
# successfully on it, so a method that fails often is compared on an easier
# subset. Read `n_paired` alongside `mean_diff`.
#
# The difference splits exactly, because A is a constant within (scenario,
# method):
#
#     mean_diff  =  A_diff            +  est_diff
#                   (approximation)      (estimation)
#
# A_diff is exact and carries no Monte Carlo error, so est_diff inherits
# diff_mcse unchanged. This is the decomposition's payoff: it says whether a
# method wins because its class fits the truth better or because it estimates
# more cheaply, and those two have different implications for a referee.
paired <-
  scored %>%
  filter(pred_ok) %>%
  select(scenario, n, data_seed, method, rps, brier, A_rps, A_brier) %>%
  pivot_longer(
    c(rps, brier, A_rps, A_brier),
    names_to = c("quantity", "metric"),
    names_pattern = "^(A_)?(rps|brier)$",
    values_to = "value"
  ) %>%
  mutate(quantity = ifelse(quantity == "A_", "A", "R")) %>%
  pivot_wider(names_from = quantity, values_from = value) %>%
  group_by(scenario, n, data_seed, metric) %>%
  filter(any(method == reference_method)) %>%
  mutate(R_ref = R[method == reference_method],
         A_ref = A[method == reference_method]) %>%
  ungroup() %>%
  filter(method != reference_method) %>%
  mutate(diff = R - R_ref, A_diff = A - A_ref) %>%
  group_by(scenario, n, method, metric) %>%
  summarize(
    n_paired = n(),
    mean_diff = mean(diff),
    diff_mcse = sd(diff) / sqrt(n_paired),
    # Negative favours `method` over the reference.
    lower = mean_diff - 1.96 * diff_mcse,
    upper = mean_diff + 1.96 * diff_mcse,
    # Exact, no Monte Carlo error.
    approx_diff = first(A_diff),
    est_diff = mean_diff - first(A_diff),
    .groups = "drop"
  )

saveRDS(paired, "summaries/paired_results.rds")
write.csv(paired, "summaries/paired_results.csv", row.names = FALSE)


## -----------------------------
# Design summary: how sparse each (scenario, n) cell is. This is a property of
# the DESIGN, reported so that the n axis can be read as the sparsity axis --
# `cppo_alt` runs 64% / 33% / 11% exposed-empty at n = 200 / 500 / 1000. It is
# emphatically NOT a partition of the replicates; see the note at the top.
design <-
  all_scores %>%
  distinct(scenario, n, data_seed, n_empty_levels, cell_dep_exp, cell_dep_unexp, min_cell) %>%
  group_by(scenario, n) %>%
  summarize(
    n_datasets = n(),
    rate_dep_exp_empty = mean(cell_dep_exp == 0),
    rate_dep_unexp_empty = mean(cell_dep_unexp == 0),
    rate_level_absent = mean(n_empty_levels > 0),
    median_min_cell = median(min_cell),
    .groups = "drop"
  )

saveRDS(design, "summaries/design_sparsity.rds")
write.csv(design, "summaries/design_sparsity.csv", row.names = FALSE)


## -----------------------------
cat("\n--- marginal ---\n")
print(summ)

cat("\n--- regret decomposition (KL for E_n, RPS for accuracy) ---\n")
print(decomp)

cat("\n--- paired vs '", reference_method, "' (negative favours the method) ---\n", sep = "")
print(paired)

cat("\n--- design sparsity (context for the n axis, NOT a stratification) ---\n")
print(design)
