## ----message = F, warning = F----
library(dplyr)
library(tibble)

## -----------------------------
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
summ <-
  all_scores %>%
  group_by(scenario, n, method) %>%
  summarize(
    n_rep = n(),
    n_ok = sum(fit_ok),
    failure_rate = 1 - n_ok / n_rep,
    rejection = mean(reject[fit_ok]),
    rejection_mcse = sqrt(rejection * (1 - rejection) / n_ok),
    mean_rmse = mean(rmse[fit_ok]),
    sd_rmse = sd(rmse[fit_ok]),
    mean_mae = mean(mae[fit_ok]),
    mean_kl = mean(kl[fit_ok]),
    .groups = "drop"
  )

saveRDS(summ, "summaries/summary_results.rds")
write.csv(summ, "summaries/summary_results.csv", row.names = FALSE)

print(summ)
