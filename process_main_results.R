## ----message = F, warning = F----
library(dplyr)
library(tidyr)
library(tibble)

## -----------------------------
# The method every other method is compared against in the paired summary.
reference_method <- "po"

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
# Marginal summary. Every mean carries a Monte Carlo standard error, and the
# three success rates are reported separately, because rejection rates are
# conditional on the test having been computed: a method that fails on exactly
# the hard replicates will look better than one that struggles through them.
# KL is summarised by the median, since a single separated fit makes the mean
# infinite.
summ <-
  all_scores %>%
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
paired <-
  all_scores %>%
  filter(pred_ok) %>%
  select(scenario, n, data_seed, method, rps, brier) %>%
  pivot_longer(c(rps, brier), names_to = "metric", values_to = "value") %>%
  group_by(scenario, n, data_seed, metric) %>%
  filter(any(method == reference_method)) %>%
  mutate(value_ref = value[method == reference_method]) %>%
  ungroup() %>%
  filter(method != reference_method) %>%
  mutate(diff = value - value_ref) %>%
  group_by(scenario, n, method, metric) %>%
  summarize(
    n_paired = n(),
    mean_diff = mean(diff),
    diff_mcse = sd(diff) / sqrt(n_paired),
    # Negative favours `method` over the reference.
    lower = mean_diff - 1.96 * diff_mcse,
    upper = mean_diff + 1.96 * diff_mcse,
    .groups = "drop"
  )

saveRDS(paired, "summaries/paired_results.rds")
write.csv(paired, "summaries/paired_results.csv", row.names = FALSE)


## -----------------------------
cat("\n--- marginal ---\n")
print(summ)

cat("\n--- paired vs '", reference_method, "' (negative favours the method) ---\n", sep = "")
print(paired)
