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

## -----------------------------
# Record-level validation (review issue 9). Two kinds of check, treated
# differently on purpose:
#
#   HARD STOP   things that corrupt every downstream number and can never be
#               right: a (scenario, n, data_seed, method) appearing twice,
#               which happens the moment an array is re-run and the old
#               job file is left in out/. bind_rows() would double-count it
#               silently.
#   WARNING     things that describe an incomplete or exploratory run rather
#               than a broken one: a seed missing some methods, a reference
#               method absent, a cell with too few replicates. A three-method
#               single-array debug run must still process; it just says so.
dups <-
  all_scores %>%
  count(scenario, n, data_seed, method, name = "n_records") %>%
  filter(n_records != 1L)

if (nrow(dups) > 0L) {
  print(dups)
  stop("Duplicate (scenario, n, data_seed, method) records: ", nrow(dups),
       " keys. A re-run array has probably left its old job file in out/.")
}

# Every seed should carry the same set of methods. A seed that does not is
# reported, not dropped: the marginal summary is per method anyway, and the
# paired summary already restricts itself to seeds where both members exist.
methods_seen <- sort(unique(all_scores$method))
incomplete <-
  all_scores %>%
  group_by(scenario, n, data_seed) %>%
  summarize(n_methods = n_distinct(method), .groups = "drop") %>%
  filter(n_methods != length(methods_seen))

if (nrow(incomplete) > 0L) {
  warning(sprintf(
    "%d of %d seeds do not carry all %d methods (%s). Paired comparisons use only complete pairs.",
    nrow(incomplete), n_distinct(paste(all_scores$scenario, all_scores$n, all_scores$data_seed)),
    length(methods_seen), paste(methods_seen, collapse = ", ")), call. = FALSE)
}

have_reference <- reference_method %in% methods_seen
if (!have_reference) {
  warning(sprintf(
    "Reference method `%s` is not in these results (methods: %s); the paired summary is skipped.",
    reference_method, paste(methods_seen, collapse = ", ")), call. = FALSE)
}

# Small helpers so that an empty or singleton cell yields NA, never NaN or an
# error. `x` is already subset to the rows that qualify.
safe_mean <- function(x) if (length(x) == 0L) NA_real_ else mean(x)
safe_prop_mcse <- function(p, m) if (is.na(p) || m < 1L) NA_real_ else sqrt(p * (1 - p) / m)
safe_mean_mcse <- function(x) if (length(x) < 2L) NA_real_ else stats::sd(x) / sqrt(length(x))

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
    rejection = safe_mean(reject[test_ok]),
    rejection_mcse = safe_prop_mcse(rejection, sum(test_ok)),
    mean_rps = safe_mean(rps[pred_ok]),
    rps_mcse = safe_mean_mcse(rps[pred_ok]),
    mean_brier = safe_mean(brier[pred_ok]),
    brier_mcse = safe_mean_mcse(brier[pred_ok]),
    mean_mae = safe_mean(mae[pred_ok]),
    median_kl = if (any(pred_ok)) median(kl[pred_ok]) else NA_real_,
    frac_kl_infinite = safe_mean(!is.finite(kl[pred_ok])),
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
    n_kl_finite = sum(is.finite(en_kl)),
    mean_en_kl_finite = safe_mean(en_kl[is.finite(en_kl)]),
    en_kl_mcse = safe_mean_mcse(en_kl[is.finite(en_kl)]),
    n_mean_en_kl_finite = first(n) * mean_en_kl_finite,
    A_rps = first(A_rps),
    mean_excess_rps = mean(excess_rps),
    excess_rps_mcse = safe_mean_mcse(excess_rps),
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
paired <- NULL
if (have_reference) paired <-
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
    diff_mcse = safe_mean_mcse(diff),
    # Negative favours `method` over the reference.
    lower = mean_diff - 1.96 * diff_mcse,
    upper = mean_diff + 1.96 * diff_mcse,
    # Exact, no Monte Carlo error.
    approx_diff = first(A_diff),
    est_diff = mean_diff - first(A_diff),
    .groups = "drop"
  )

if (!is.null(paired)) {
  saveRDS(paired, "summaries/paired_results.rds")
  write.csv(paired, "summaries/paired_results.csv", row.names = FALSE)
}


## -----------------------------
# Boundary decomposition (policy P3, settled 2026-09-18). For any method that
# reports a `boundary` flag -- CPPO -- the HEADLINE rejection rate above is
# unconditional: the conventional chi-square test applied to every replicate,
# boundary or not, because that is what the procedure as practised does. This
# table is the diagnostic decomposition underneath it:
#
#   boundary_rate            P(constrained MLE on the boundary), a design-cell
#                            property of (scenario, n, method)
#   rejection_interior       P(reject | interior), the rate a referee who
#                            distrusts the chi-square reference at the boundary
#                            will ask for. It is CONDITIONAL on a function of
#                            the outcome and is labelled so; it is not the
#                            method's size or power.
#   rejection_boundary       P(reject | boundary), for completeness
#
# The boundary event is never used to drop replicates: under `cppo_alt` it is
# driven by b2, the effect under test.
if ("boundary" %in% names(scored)) {

  bdry <-
    scored %>%
    filter(fit_ok, !is.na(boundary)) %>%
    group_by(scenario, n, method) %>%
    summarize(
      n_fit = n(),
      boundary_rate = mean(boundary),
      boundary_mcse = sqrt(boundary_rate * (1 - boundary_rate) / n_fit),
      n_interior = sum(!boundary & test_ok),
      rejection_interior = safe_mean(reject[!boundary & test_ok]),
      rejection_interior_mcse = safe_prop_mcse(rejection_interior, n_interior),
      n_boundary = sum(boundary & test_ok),
      rejection_boundary = safe_mean(reject[boundary & test_ok]),
      # Which fitter produced the numbers.
      share_direct = mean(engine == "direct", na.rm = TRUE),
      .groups = "drop"
    )

  saveRDS(bdry, "summaries/boundary_decomposition.rds")
  write.csv(bdry, "summaries/boundary_decomposition.csv", row.names = FALSE)
}


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

if (!is.null(paired)) {
  cat("\n--- paired vs '", reference_method, "' (negative favours the method) ---\n", sep = "")
  print(paired)
}

if (exists("bdry")) {
  cat("\n--- boundary decomposition (rejection_interior is CONDITIONAL; headline is in --- marginal ---) ---\n")
  print(bdry)
}

cat("\n--- design sparsity (context for the n axis, NOT a stratification) ---\n")
print(design)
