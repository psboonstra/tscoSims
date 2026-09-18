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

# Every seed should carry the same SET of methods -- compared as a set, not a
# count, so that method substitution across files cannot slip through. A seed
# that does not is reported, not dropped: the marginal summary is per method
# anyway, and the paired summary already restricts itself to seeds where both
# members exist.
methods_seen <- sort(unique(all_scores$method))
methods_key <- paste(methods_seen, collapse = "|")
incomplete <-
  all_scores %>%
  group_by(scenario, n, data_seed) %>%
  summarize(methods_here = paste(sort(unique(method)), collapse = "|"), .groups = "drop") %>%
  filter(methods_here != methods_key)

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
#     en_kl                    = kl  - A_kl
#     rps_estimation_residual  = rps - rps_at_kl_projection
#
# NAMES (tightened 2026-09-18 after review). `A_kl` IS the class's minimum KL
# approximation regret, inf_theta KL(p, q_theta), computed exactly. The RPS
# quantity is NOT the class's minimum RPS: it is RPS evaluated at the KL
# projection theta*_KL, so it is called `rps_at_kl_projection`, and the
# difference is an ESTIMATION RESIDUAL -- how far the MLE's RPS sits from its
# own target -- not an RPS approximation/estimation split. It is evaluated at
# theta*_KL rather than minimized because thetahat converges to theta*_KL; at
# the RPS minimizer the residual would converge to a positive constant instead
# of zero. It CAN be negative in finite samples and is not clipped.
#
# NON-NEGATIVITY OF en_kl IS EXACT, NOT ASYMPTOTIC. theta*_KL was found by
# minimizing KL on the SAME quadrature grid the replicates are scored on, and
# every accepted fit is a member of the class, so kl >= A_kl holds for every
# finite replicate. A negative value therefore means a bug -- projection not at
# its optimum, a fit outside the class, or inconsistent grids -- and is a hard
# stop below, not a footnote.
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
  rename(rps_at_kl_projection = A_rps, brier_at_kl_projection = A_brier) %>%
  mutate(
    en_kl = kl - A_kl,
    rps_estimation_residual = rps - rps_at_kl_projection,
    brier_estimation_residual = brier - brier_at_kl_projection
  )

en_kl_min <- suppressWarnings(min(scored$en_kl[scored$pred_ok & is.finite(scored$en_kl)]))
if (is.finite(en_kl_min) && en_kl_min < -1e-10) {
  print(scored %>% filter(pred_ok, en_kl < -1e-10) %>%
          select(scenario, n, method, data_seed, kl, A_kl, en_kl))
  stop(sprintf("en_kl < 0 (min %.3e): a fit scored below its class's exact KL minimum. ",
               en_kl_min),
       "Check the projection, the fit's class membership, and the quadrature grids.")
}

# Each method's PRIMARY test type must be one thing. `test` is carried on every
# row ("lrt" for po, mr, tsco_*; "score" for cppo); a method with two types
# means score files from incompatible code versions were pooled.
test_types <-
  scored %>%
  group_by(method) %>%
  summarize(n_types = n_distinct(test), types = paste(sort(unique(test)), collapse = "|"), .groups = "drop")
if (any(test_types$n_types != 1L)) {
  print(test_types)
  stop("A method reports more than one primary test type; score files are from incompatible versions.")
}

saveRDS(scored, "summaries/all_scores_decomposed.rds")


## -----------------------------
# Marginal summary. Every mean carries a Monte Carlo standard error.
#
# THE TEST OUTCOME IS THREE-WAY, and nothing here is called plain "rejection":
#
#   p_reject          P(test computed AND rejects)      -- unconditional
#   p_retain          P(test computed AND does not)     -- unconditional
#   p_no_test         P(no test could be computed)      -- unconditional
#   rejection_given_test = p_reject / (1 - p_no_test)   -- CONDITIONAL on a test
#
# The three unconditional rows sum to one. `p_reject` is also the operating
# characteristic of the rule "no test -> do not reject"; whether that is the
# rule an analyst would follow is a modeling statement, so both it and the
# conditional rate are given and the reader chooses with the failure rate in
# view.
#
# WHICH TEST. `test` names each method's PRIMARY test: the LRT for po, mr and
# tsco_*, and -- decided 2026-09-18 -- the efficient Rao score test for cppo,
# because cppo's LRT has its full-model MLE on the boundary in most sparse
# draws and its chi-square_2 calibration is not justified there. That LRT is
# cppo's SENSITIVITY analysis and appears in the `*_alt` columns with its own
# three-way outcome. A table that compares p_reject across methods therefore
# compares each method's defensible test; the `test` column says which.
#
# KL is summarized by the median, since a single separated fit makes the mean
# infinite.
summ <-
  scored %>%
  group_by(scenario, n, method) %>%
  summarize(
    n_rep = n(),
    fit_ok_rate = mean(fit_ok),
    test_ok_rate = mean(test_ok),
    pred_ok_rate = mean(pred_ok),
    test = first(test),
    p_reject = mean(reject),
    p_retain = mean(test_ok & !reject),
    p_no_test = mean(!test_ok),
    rejection_given_test = safe_mean(reject[test_ok]),
    rejection_given_test_mcse = safe_prop_mcse(rejection_given_test, sum(test_ok)),
    # Alternative test, where a method supplies one (cppo: the constrained-ML
    # LRT with chi-square_2 calibration, as sensitivity analysis). Same
    # three-way outcome, summing to one. NA for methods without one.
    test_alt = if (all(is.na(test_alt))) NA_character_ else first(test_alt[!is.na(test_alt)]),
    p_reject_alt = if (all(is.na(test_alt))) NA_real_ else mean(reject_alt),
    p_retain_alt = if (all(is.na(test_alt))) NA_real_ else mean(test_ok_alt & !reject_alt),
    p_no_test_alt = if (all(is.na(test_alt))) NA_real_ else mean(!test_ok_alt),
    rejection_given_test_alt = if (any(test_ok_alt)) safe_mean(reject_alt[test_ok_alt]) else NA_real_,
    mean_rps = safe_mean(rps[pred_ok]),
    rps_mcse = safe_mean_mcse(rps[pred_ok]),
    mean_brier = safe_mean(brier[pred_ok]),
    brier_mcse = safe_mean_mcse(brier[pred_ok]),
    mean_mae = safe_mean(mae[pred_ok]),
    median_kl = if (any(pred_ok)) median(kl[pred_ok]) else NA_real_,
    frac_kl_infinite = safe_mean(!is.finite(kl[pred_ok])),
    .groups = "drop"
  )

stopifnot("primary three-way outcome does not sum to one" =
            isTRUE(all.equal(summ$p_reject + summ$p_retain + summ$p_no_test, rep(1, nrow(summ)))))
has_alt <- !is.na(summ$test_alt)
stopifnot("alternative three-way outcome does not sum to one" =
            isTRUE(all.equal((summ$p_reject_alt + summ$p_retain_alt + summ$p_no_test_alt)[has_alt],
                             rep(1, sum(has_alt)))))

saveRDS(summ, "summaries/summary_results.rds")
write.csv(summ, "summaries/summary_results.csv", row.names = FALSE)


## -----------------------------
# The regret decomposition: R_n = A + E_n, with A the exact approximation
# regret and E_n the finite-sample estimation regret.
#
# n * E_n^KL AS AN EFFECTIVE-DIMENSION DIAGNOSTIC -- with three caveats that
# the review (2026-09-18) made precise:
#
#   (i)  Under regularity (interior pseudo-true parameter, asymptotic
#        normality), n*E_n -> (1/2) Z' H Z with Z ~ N(0, H^-1 J H^-1): a
#        weighted sum of chi-square_1's whose MEAN is (1/2) tr(J H^-1), the
#        Takeuchi quantity. That is the number to compare the mean of n*E_n
#        against. Only under CORRECT specification does J = H and the limit
#        collapse to (1/2) chi-square_d with mean d/2.
#   (ii) The Wilson-Hilferty median heuristic -- median of (1/2) chi-square_d
#        ~ (d/2)(1 - 2/(9d))^3, i.e. 3.18 against a mean of 3.5 at d = 7 --
#        therefore applies ONLY where the class contains the truth. Under
#        misspecification the median of a weighted chi-square sum has no such
#        closed form, and a median-vs-mean gap there is not interpretable.
#   (iii) None of this holds in a cell where the fit is routinely on the
#        boundary, unidentified, separated or rank-deficient. CPPO under
#        `cppo_alt` at n = 200 (boundary rate ~0.6) is such a cell: n*E_n is
#        reported but is not an effective dimension there.
#
# MEAN VERSUS MEDIAN. A single separated fit makes `kl` infinite, and if that
# event has positive probability then E[E_n^KL] IS infinite -- so the
# finite-only mean below is a conditional diagnostic, not an estimate of
# E[E_n]. It is reported with `frac_kl_infinite` beside it and must not carry
# an effective-dimension claim unless that fraction is zero. The median is the
# robust summary and is always defined.
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
    mean_rps = mean(rps),
    rps_at_kl_projection = first(rps_at_kl_projection),
    mean_rps_estimation_residual = mean(rps_estimation_residual),
    rps_estimation_residual_mcse = safe_mean_mcse(rps_estimation_residual),
    # The additive components are reported; no "share", because the residual
    # can be negative and a ratio would then exceed one.
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
  select(scenario, n, data_seed, method, rps, brier,
         A_rps = rps_at_kl_projection, A_brier = brier_at_kl_projection) %>%
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
    # Negative favors `method` over the reference.
    lower = mean_diff - 1.96 * diff_mcse,
    upper = mean_diff + 1.96 * diff_mcse,
    # Exact, no Monte Carlo error. For metric == "rps" this is the difference
    # in RPS AT THE KL PROJECTIONS, not in minimum-RPS approximation regret --
    # see the naming note above the join.
    at_projection_diff = first(A_diff),
    estimation_diff = mean_diff - first(A_diff),
    .groups = "drop"
  )

if (!is.null(paired)) {
  saveRDS(paired, "summaries/paired_results.rds")
  write.csv(paired, "summaries/paired_results.csv", row.names = FALSE)
}


## -----------------------------
# Boundary decomposition of CPPO's LRT (policy P3, settled 2026-09-18). The
# boundary is a property of the FULL-model MLE, so it concerns the LRT -- now
# cppo's alternative test in the `*_alt` columns -- and not the primary score
# test, which never computes that MLE. `p_reject_alt` above is unconditional:
# the LRT with chi-square_2 calibration applied to every replicate, boundary or
# not. This table is the diagnostic decomposition underneath it:
#
#   boundary_rate            P(constrained MLE on the boundary), by the
#                            engine's own flag (exact active set on the direct
#                            path; slack < 1e-3 on the VGAM path)
#   boundary_rate_1e4 / 1e3 / 1e2
#                            the same from the recorded `slack_min` at three
#                            thresholds, applied identically to both engines,
#                            so the rate is not hostage to one hidden cutoff
#   lrt_rejection_given_test_interior
#                            P(LRT rejects | LRT computed, interior).
#                            CONDITIONAL on a function of the outcome and
#                            labeled so; it is not the method's size or power.
#   lrt_rejection_given_test_boundary
#                            P(LRT rejects | LRT computed, boundary), for
#                            completeness
#   score_p_reject_interior / _boundary
#                            the primary score test's unconditional rejection
#                            within the same split, for the contrast
#
# The boundary event is never used to drop replicates: under `cppo_alt` it is
# driven by b2, the effect under test.
if ("boundary" %in% names(scored)) {

  bdry <-
    scored %>%
    filter(fit_ok, !is.na(boundary), test_alt %in% "lrt") %>%
    group_by(scenario, n, method) %>%
    summarize(
      n_fit = n(),
      boundary_rate = mean(boundary),
      boundary_mcse = safe_prop_mcse(boundary_rate, n_fit),
      boundary_rate_1e4 = safe_mean(slack_min[is.finite(slack_min)] < 1e-4),
      boundary_rate_1e3 = safe_mean(slack_min[is.finite(slack_min)] < 1e-3),
      boundary_rate_1e2 = safe_mean(slack_min[is.finite(slack_min)] < 1e-2),
      n_interior = sum(!boundary & test_ok_alt),
      lrt_rejection_given_test_interior = safe_mean(reject_alt[!boundary & test_ok_alt]),
      lrt_rejection_given_test_interior_mcse = safe_prop_mcse(lrt_rejection_given_test_interior, n_interior),
      n_boundary = sum(boundary & test_ok_alt),
      lrt_rejection_given_test_boundary = safe_mean(reject_alt[boundary & test_ok_alt]),
      score_p_reject_interior = safe_mean(reject[!boundary]),
      score_p_reject_boundary = safe_mean(reject[boundary]),
      # Which fitter produced the numbers.
      share_direct = mean(engine == "direct", na.rm = TRUE),
      .groups = "drop"
    )

  saveRDS(bdry, "summaries/boundary_decomposition.rds")
  write.csv(bdry, "summaries/boundary_decomposition.csv", row.names = FALSE)
}


## -----------------------------
# Score-test diagnostics (review, 2026-09-18). The score test is CPPO's primary
# result, and Peterson and Harrell (1990, Section 7) warn that score tests
# misbehave with sparse cells and near-singular information. These columns say
# whether that happened: availability, the efficient information's smallest
# eigenvalue and condition number, the nuisance block's condition number, and
# the per-observation nuisance score at the reduced fit against the reduced
# optimizer's KKT tolerance of 1e-6.
if ("score_min_eig_eff" %in% names(scored)) {
  score_diag <-
    scored %>%
    filter(test == "score") %>%
    group_by(scenario, n, method) %>%
    summarize(
      n_rep = n(),
      score_available = mean(test_ok),
      n_score = sum(test_ok),
      min_eig_eff_min = if (n_score > 0) min(score_min_eig_eff, na.rm = TRUE) else NA_real_,
      min_eig_eff_q05 = if (n_score > 0) unname(quantile(score_min_eig_eff, 0.05, na.rm = TRUE)) else NA_real_,
      min_eig_eff_median = if (n_score > 0) median(score_min_eig_eff, na.rm = TRUE) else NA_real_,
      cond_eff_median = if (n_score > 0) median(score_cond_eff, na.rm = TRUE) else NA_real_,
      cond_eff_max = if (n_score > 0) max(score_cond_eff, na.rm = TRUE) else NA_real_,
      cond_nuis_median = if (n_score > 0) median(score_cond_nuis, na.rm = TRUE) else NA_real_,
      cond_nuis_max = if (n_score > 0) max(score_cond_nuis, na.rm = TRUE) else NA_real_,
      nuis_per_n_max = if (n_score > 0) max(score_nuis_per_n, na.rm = TRUE) else NA_real_,
      nuis_exceeds_kkt_tol = if (n_score > 0) mean(score_nuis_per_n > 1e-6, na.rm = TRUE) else NA_real_,
      .groups = "drop"
    )
  if (nrow(score_diag) > 0L) {
    saveRDS(score_diag, "summaries/score_diagnostics.rds")
    write.csv(score_diag, "summaries/score_diagnostics.csv", row.names = FALSE)
  }
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

cat("\n--- regret decomposition (A_kl exact; rps_at_kl_projection is NOT min-RPS) ---\n")
print(decomp)

if (!is.null(paired)) {
  cat("\n--- paired vs '", reference_method, "' (negative favors the method) ---\n", sep = "")
  print(paired)
}

if (exists("bdry")) {
  cat("\n--- boundary decomposition of cppo's LRT (alt test); lrt_rejection_given_test_* are CONDITIONAL ---\n")
  print(bdry)
}

if (exists("score_diag") && nrow(score_diag) > 0L) {
  cat("\n--- score-test diagnostics (primary test for cppo) ---\n")
  print(score_diag)
}

cat("\n--- design sparsity (context for the n axis, NOT a stratification) ---\n")
print(design)
