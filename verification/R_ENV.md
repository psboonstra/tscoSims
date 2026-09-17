# Running the R verification scripts

CRAN is blocked by the egress policy, but the **Ubuntu archive is not**, and it
ships the packages this study needs as distro debs. In an Ubuntu 24.04 container
with root:

    apt-get update
    apt-get install -y --no-install-recommends \
        r-base-core r-cran-vgam r-cran-rms r-cran-dplyr r-cran-tibble r-cran-glue r-cran-tidyr

That gives R 4.3.3, VGAM 1.1.9, rms 6.7.1. `tsco` then installs from source, with
one change: `Depends: R (>= 4.4.0)` must be relaxed (4.3.3 is what the archive
has). Do that in a *copy* of DESCRIPTION, not in the package:

    sed -i 's/R (>= 4.4.0)/R (>= 4.1.0)/' <copy>/tsco/DESCRIPTION
    R CMD INSTALL --no-docs <copy>/tsco

Nothing in the results below depends on that version relaxation; it is only the
declared minimum.

## The scripts (run from the tscoSims root)

| script | what it does | runtime |
| --- | --- | --- |
| `r_cppo_slotcheck.R` | confirms the S4 slots `cppo_vglm_valid()` relies on exist on a real `vglm` object, that the check discriminates, and that both engines agree when the constraint is slack | seconds |
| `r_vgam_boundary.R` | per-replicate VGAM vs constrained-direct comparison; writes `scratch_vgam_boundary.rds`. `NREP=600` | ~2 min / 600 reps |
| `r_vgam_boundary_analyze.R` | reads that rds and prints the VGAM outcome breakdown and the error messages | seconds |
| `r_typeI_driver.R` | the type I error run through the harness's own method functions and seeds. `ARRAY_ID`, `REPS`, `BUDGET_S`; checkpoints every 50 reps into `scratch_typeI/`, so re-running resumes | ~0.6 s / rep |
| `r_typeI_report.R` | reads every block in `scratch_typeI/` and prints sizes, paired differences against PO, and the warning decomposition | seconds |

`r_typeI_driver.R` deliberately uses a 200-row evaluation set: the LRT does not
depend on it, so the run is much cheaper, but **the accuracy metrics from this
script are not usable**. Use `run_sims.R` with its own `n_test = 20000` for those.

Seeds match `run_sims.R` exactly (`master_seed + array_id * 1e5 + rep`), so
replicate `i` of array `a` here is the same dataset as in a Slurm run.
