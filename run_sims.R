## ----message = F, warning = F----
library(tsco)
library(VGAM)
library(dplyr)
library(tidyr)
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
scenario_seq = "cppo_alt" #c("null", "po_alt", "cppo_alt", "tsco_alt", "tsco_alt_stage2", "none_true")
methods_seq = c("tsco_popo", "mr", "cppo") #c("tsco_pomr", "tsco_popo", "po", "mr", "cppo")
n_seq = 200 ; #c(200, 500, 1000)
reps_per_array = 200
sims_per_scenario = reps_per_array * 1

# Outcome levels are labelled 0..5 to match the manuscript, so `cutoff_level`
# here is literally the paper's C: "5" gives the P O|5|M R of Table 3. Note
# that with C = "5" the upper partition holds a single level, so tsco_pomr and
# tsco_popo are the SAME model; use a smaller cutoff to separate them.
levels_y = as.character(0:5)
cutoff_level = "5"
# Where the accuracy metrics are evaluated. "grid" integrates the covariate
# distribution exactly by quadrature; "mc" is the old random test set, kept for
# comparison only (see sim_functions/eval_setup.R for why grid is the default).
eval_by = "grid"
n_eval_nodes = 40
n_test = 20000        # only used when eval_by = "mc"
master_seed = 20260710
alpha = 0.05

# Helpful to print warnings when they occur for debugging
options(warn = 1)


## -----------------------------
source("sim_functions/scenario_setup.R")
source("sim_functions/score_method.R")
source("sim_functions/method_setup.R")
source("sim_functions/dataset_diagnostics.R")
source("sim_functions/kl_projection.R")
source("sim_functions/true_probs.R")
source("sim_functions/check_dgm.R")
source("aux_functions/safe_fit.R")
source("aux_functions/align_prob.R")
source("aux_functions/vglm_helpers.R")
source("aux_functions/unobserved_levels.R")
source("aux_functions/tsco_provenance.R")

source("methods/po.R")
source("methods/mr.R")
source("methods/cppo.R")
source("methods/tsco.R")

# The methods to run, as a named list of closures with the common signature
# function(dat, test_dat). `methods_seq` selects from it and fixes the order.
method_list <- build_method_list(levels_y, cutoff_level)
check_methods_seq(methods_seq, method_list)

stopifnot(array_id >= 1, array_id <= total_num_arrays)

## -----------------------------
curr_row <-
  all_scenarios |>
  filter(array_id >= start_array_id,
         array_id <= end_array_id)

curr_scenario_id <- curr_row |> pull(scenario_id)
scenario <- curr_row |> pull(scenario)
n <- curr_row |> pull(n)


## -----------------------------
# One seed per replicate, spaced far enough apart that no two arrays collide.
rep_ids <- (curr_row |> pull(rep_start)):(curr_row |> pull(rep_end))
n_sim <- length(rep_ids)
data_seeds <- master_seed + array_id * 100000L + rep_ids


## -----------------------------
# The evaluation set. With eval_by = "grid" this is a deterministic quadrature
# grid, so it carries no sampling error and is identical across arrays.
test_seed <- master_seed + 999
source("sim_functions/eval_setup.R")


## -----------------------------
i = 1
all_scores = NULL
array_id_stats <-
  curr_row |>
  mutate(array_id = array_id,
         n_sim_this_array_id = n_sim,
         total_runtime_secs = NA)


## -----------------------------
begin <- Sys.time()
for (i in 1:n_sim) {
  
  set.seed(data_seeds[i])
  source("sim_functions/draw_single_dataset.R")

  # Sparsity of THIS dataset, recorded once and attached to every method's row.
  # Empty cells drive the finite-sample behaviour and the rate is scenario
  # specific -- 64% of cppo_alt draws at n = 200 have an empty exposed-arm
  # departure cell. See sim_functions/dataset_diagnostics.R.
  curr_diag <- dataset_diagnostics(dat, levels_y)
  cat(glue("scenario = {scenario}, n = {n}, array_id = {array_id}; i/n_sim = {i}/{n_sim}; seed = {data_seeds[i]}\n\n"))
  
  for (curr_method in methods_seq) {

    begin2 <- Sys.time()
    curr_fit <- method_list[[curr_method]](dat, test_dat)
    curr_run_time <- as.numeric(difftime(Sys.time(), begin2, units = "secs"))

    all_scores <-
      bind_rows(all_scores,
                bind_cols(method = curr_method,
                          score_method(curr_fit, p_true_test, alpha, wt = eval_wt),
                          curr_diag,
                          run_time = curr_run_time,
                          sim_num = i,
                          data_seed = data_seeds[i]))
    rm(curr_fit, curr_run_time)
  }
}

array_id_stats <-
  array_id_stats |>
  mutate(total_runtime_secs = as.numeric(difftime(Sys.time(), begin, units = "secs")))


## -----------------------------
# Provenance of the tsco build that produced these numbers. Read once, not per
# replicate, and attached to every row so a score file is self-describing after
# it leaves this machine. See aux_functions/tsco_provenance.R for why the
# version string alone is not enough and why the commit has to be stamped at
# install time.
tsco_prov <- tsco_provenance()

if (is.na(tsco_prov$tsco_commit)) {
  warning("tsco was installed without a recorded commit; ",
          "results will not identify the source. See install_tsco_stamped().",
          call. = FALSE)
} else if (isTRUE(tsco_prov$tsco_dirty)) {
  warning(glue("tsco was installed from a MODIFIED working tree at ",
               "{tsco_prov$tsco_commit}; the commit alone does not identify it."),
          call. = FALSE)
}


## -----------------------------
all_scores <-
  all_scores |>
  mutate(scenario = scenario,
         n = n,
         array_id = array_id,
         scenario_id = curr_scenario_id,
         tsco_version = tsco_prov$tsco_version,
         tsco_commit = tsco_prov$tsco_commit,
         tsco_dirty = tsco_prov$tsco_dirty) |>
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
  all_scores |>
    group_by(method) |>
    summarize(n_fit_ok = sum(fit_ok),
              n_pred_ok = sum(pred_ok),
              rejection = mean(reject[test_ok]),
              mean_rps = mean(rps[pred_ok]),
              mean_brier = mean(brier[pred_ok]),
              median_kl = median(kl[pred_ok]),
              mean_run_time = mean(run_time))
}
