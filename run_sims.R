## ----message = F, warning = F----
library(tsco)
library(VGAM)
library(dplyr)
library(tibble)
library(glue)


## -----------------------------
# Set my_computer = TRUE to step through a single array interactively. Set it
# to FALSE for SLURM: array_id then comes from the job array, and results are
# written to disk.
my_computer = TRUE
array_id_if_my_computer = 1


## -----------------------------
if (my_computer) {
  array_id = array_id_if_my_computer
} else {
  array_id = as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))
}

# 'sims_per_scenario' is the number of independent simulated datasets per
# scenario; 'reps_per_array' is how many of those one array_id handles.
scenario_seq = c("null_po", "null_tsco", "po_alt", "cppo_alt", "tsco_pomr_alt")
n_seq = c(200, 500, 1000)
sims_per_scenario = 1000
reps_per_array = 25

levels_y = as.character(1:5)
cutoff_level = "4"
n_test = 20000
master_seed = 20260710
alpha = 0.05

# Helpful to print warnings when they occur for debugging
options(warn = 1)


## -----------------------------
source("sim_functions/scenario_setup.R")
source("sim_functions/score_method.R")
source("sim_functions/true_probs.R")
source("aux_functions/safe_fit.R")
source("aux_functions/align_prob.R")
source("aux_functions/vglm_helpers.R")

source("methods/po.R")
source("methods/mr.R")
source("methods/cppo.R")
source("methods/tsco.R")

stopifnot(array_id >= 1, array_id <= total_num_arrays)


## -----------------------------
curr_row <-
  all_scenarios %>%
  filter(array_id >= start_array_id,
         array_id <= end_array_id)

curr_scenario_id <- curr_row %>% pull(scenario_id)
scenario <- curr_row %>% pull(scenario)
n <- curr_row %>% pull(n)


## -----------------------------
# One seed per replicate, spaced far enough apart that no two arrays collide.
rep_ids <- (curr_row %>% pull(rep_start)):(curr_row %>% pull(rep_end))
n_sim <- length(rep_ids)
data_seeds <- master_seed + array_id * 100000L + rep_ids


## -----------------------------
# The evaluation set is drawn once per array, not once per replicate.
test_seed <- master_seed + 999
source("sim_functions/draw_test_data.R")


## -----------------------------
i = 1
all_scores = NULL
array_id_stats <-
  curr_row %>%
  mutate(array_id = array_id,
         n_sim_this_array_id = n_sim,
         total_runtime_secs = NA)


## -----------------------------
begin <- Sys.time()
for (i in 1:n_sim) {

  set.seed(data_seeds[i])
  source("sim_functions/draw_single_dataset.R")
  cat(glue("scenario = {scenario}, n = {n}, array_id = {array_id}; i/n_sim = {i}/{n_sim}; seed = {data_seeds[i]}\n\n"))

  # TSCO PO|MR ----
  begin2 <- Sys.time()
  curr_fit <- fxn_tsco_pomr(dat, test_dat, levels_y, cutoff_level)
  curr_run_time <- as.numeric(difftime(Sys.time(), begin2, units = "secs"))

  all_scores <-
    bind_rows(all_scores,
              bind_cols(method = "tsco_pomr",
                        score_method(curr_fit, p_true_test, alpha),
                        run_time = curr_run_time,
                        sim_num = i,
                        data_seed = data_seeds[i]))
  rm(curr_fit, curr_run_time)

  # TSCO PO|PO ----
  begin2 <- Sys.time()
  curr_fit <- fxn_tsco_popo(dat, test_dat, levels_y, cutoff_level)
  curr_run_time <- as.numeric(difftime(Sys.time(), begin2, units = "secs"))

  all_scores <-
    bind_rows(all_scores,
              bind_cols(method = "tsco_popo",
                        score_method(curr_fit, p_true_test, alpha),
                        run_time = curr_run_time,
                        sim_num = i,
                        data_seed = data_seeds[i]))
  rm(curr_fit, curr_run_time)

  # PO ----
  begin2 <- Sys.time()
  curr_fit <- fxn_po(dat, test_dat, levels_y)
  curr_run_time <- as.numeric(difftime(Sys.time(), begin2, units = "secs"))

  all_scores <-
    bind_rows(all_scores,
              bind_cols(method = "po",
                        score_method(curr_fit, p_true_test, alpha),
                        run_time = curr_run_time,
                        sim_num = i,
                        data_seed = data_seeds[i]))
  rm(curr_fit, curr_run_time)

  # MR ----
  begin2 <- Sys.time()
  curr_fit <- fxn_mr(dat, test_dat, levels_y)
  curr_run_time <- as.numeric(difftime(Sys.time(), begin2, units = "secs"))

  all_scores <-
    bind_rows(all_scores,
              bind_cols(method = "mr",
                        score_method(curr_fit, p_true_test, alpha),
                        run_time = curr_run_time,
                        sim_num = i,
                        data_seed = data_seeds[i]))
  rm(curr_fit, curr_run_time)

  # CPPO ----
  begin2 <- Sys.time()
  curr_fit <- fxn_cppo(dat, test_dat, levels_y)
  curr_run_time <- as.numeric(difftime(Sys.time(), begin2, units = "secs"))

  all_scores <-
    bind_rows(all_scores,
              bind_cols(method = "cppo",
                        score_method(curr_fit, p_true_test, alpha),
                        run_time = curr_run_time,
                        sim_num = i,
                        data_seed = data_seeds[i]))
  rm(curr_fit, curr_run_time)
}

array_id_stats <-
  array_id_stats %>%
  mutate(total_runtime_secs = as.numeric(difftime(Sys.time(), begin, units = "secs")))


## -----------------------------
all_scores <-
  all_scores %>%
  mutate(scenario = scenario,
         n = n,
         array_id = array_id,
         scenario_id = curr_scenario_id) %>%
  select(array_id, scenario_id, scenario, n, sim_num, data_seed,
         method, everything())


## -----------------------------
if (!my_computer) {
  dir.create("out", recursive = TRUE, showWarnings = FALSE)

  write.csv(all_scores,
            file = glue("out/job{array_id}_scores.csv"),
            row.names = FALSE)

  write.csv(array_id_stats,
            file = glue("out/job{array_id}_array_stats.csv"),
            row.names = FALSE)
}


## -----------------------------
if (my_computer) {
  all_scores %>%
    group_by(method) %>%
    summarize(n_fit_ok = sum(fit_ok),
              n_pred_ok = sum(pred_ok),
              rejection = mean(reject[test_ok]),
              mean_rps = mean(rps[pred_ok]),
              mean_brier = mean(brier[pred_ok]),
              median_kl = median(kl[pred_ok]),
              mean_run_time = mean(run_time))
}
