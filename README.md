# tscoSims

Simulation study for the `tsco` package.

## Layout

    run_sims.R                       top-level script: one array_id per run
    run_sims.txt                     SLURM submission script
    process_main_results.R           aggregates out/*.csv into summaries/

    sim_functions/
      scenario_setup.R               builds `all_scenarios` and the array_id map
      true_probs.R                   true category probabilities per scenario
      draw_single_dataset.R          sourced: leaves `dat` and `p_true`
      draw_test_data.R               sourced: leaves `test_dat` and `p_true_test`
      score_method.R                 every metric is defined here

    methods/                         one file per comparator, each `fxn_*`
      po.R  mr.R  cppo.R  tsco.R

    aux_functions/                   generic helpers
      safe_fit.R  align_prob.R  vglm_helpers.R

## Running

Interactively, to explore one array: open `run_sims.R`, leave
`my_computer = TRUE`, set `array_id_if_my_computer`, and run top to bottom.
Nothing is written to disk, a summary of the array prints at the end, and every
object (`dat`, `p_true`, `test_dat`, `curr_fit`, ...) stays in the workspace.

On SLURM: set `my_computer = FALSE`, then `sbatch run_sims.txt`. Each array_id
writes `out/job<id>_scores.csv`. When all arrays finish, run
`process_main_results.R`.

`sim_functions/scenario_setup.R` can be sourced on its own to inspect the
design grid without running anything.
