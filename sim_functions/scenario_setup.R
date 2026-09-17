# Builds `all_scenarios`, the full grid of simulation scenarios, and assigns
# each row a contiguous block of array_ids. One array_id == one output file.
#
# Inputs (set in run_sims.R, defaults supplied here so the file can be sourced
# on its own to inspect the design):
#   scenario_seq, n_seq, sims_per_scenario, reps_per_array

if (!exists("scenario_seq")) {
  scenario_seq <- c("null", "po_alt", "cppo_alt", "tsco_alt", "tsco_alt_stage2", "none_true")
}
if (!exists("n_seq")) {n_seq <- c(200, 500, 1000)}
if (!exists("sims_per_scenario")) {sims_per_scenario <- 1000}
if (!exists("reps_per_array")) {reps_per_array <- 25}

stopifnot(sims_per_scenario %% reps_per_array == 0)

all_scenarios <-
  expand_grid(
    scenario = scenario_seq,
    n = n_seq,
    chunk = seq_len(sims_per_scenario / reps_per_array),
    stringsAsFactors = FALSE
  ) |>
  as_tibble() |>
  mutate(
    scenario_id = 1:n(),
    rep_start = (chunk - 1L) * reps_per_array + 1L,
    rep_end = chunk * reps_per_array,
    # One array per chunk. Kept in the `start_array_id`/`end_array_id` form so
    # that allocating several arrays to an expensive scenario later is a change
    # to `num_arrays` alone.
    num_arrays = 1,
    start_array_id = lag(1 + cumsum(num_arrays), 1, default = 1),
    end_array_id = cumsum(num_arrays)
  )

total_num_arrays <- max(all_scenarios$end_array_id)
