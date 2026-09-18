# Equivalence test for the method_setup.R refactor.
#
# The refactor must be a pure restructuring: the same replicates, the same
# fits, the same scores. So run the OLD run_sims.R and the NEW one on the same
# seeds and compare every column of `all_scores`.
#
# Two differences are expected and benign:
#   - row ORDER. The old script emitted methods in a fixed hard-coded order
#     regardless of methods_seq; the new one emits in methods_seq order. So
#     compare after sorting by (sim_num, method).
#   - `run_time`, which is wall clock and never reproducible.
#
# Usage, from the tscoSims root, with the pre-refactor script saved somewhere:
#   OLD=/tmp/run_sims_orig.R NEW=run_sims.R REPS=25 Rscript verification/r_method_setup_check.R
suppressMessages({library(dplyr); library(tibble)})

old_path <- Sys.getenv("OLD", "/tmp/run_sims_orig.R")
new_path <- Sys.getenv("NEW", "run_sims.R")
reps     <- Sys.getenv("REPS", "25")
methods  <- Sys.getenv("METHODS", 'c("tsco_pomr", "tsco_popo", "po", "mr", "cppo")')

# Run a script and return its all_scores.
#
# It has to be a SEPARATE R PROCESS, not sys.source() into a sub-environment:
# run_sims.R calls source("sim_functions/draw_single_dataset.R") with the
# default local = FALSE, which evaluates in the global environment, so the
# script only works when it is itself running globally. That is a property of
# the design (draw_single_dataset.R is sourced rather than called precisely so
# its objects stay visible for interactive debugging), not a defect.
run_one <- function(path, label) {
  src <- readLines(path)
  src <- sub("^reps_per_array = .*$", paste0("reps_per_array = ", reps), src)
  src <- sub("^methods_seq = .*$",    paste0("methods_seq = ", methods), src)
  src <- sub("^n_test = .*$",         "n_test = 2000", src)
  src <- gsub('^source\\("sim_functions/check_dgm.R"\\).*$', "", src)
  src <- gsub('^\\s*cat\\(glue\\(.*$', "", src)
  out <- tempfile(fileext = ".rds")
  src <- c(src, sprintf('saveRDS(all_scores, "%s")', out))
  tmp <- tempfile(fileext = ".R"); writeLines(src, tmp)
  cat("  running", label, "...\n")
  st <- system2("Rscript", tmp, stdout = FALSE, stderr = FALSE)
  if (st != 0 || !file.exists(out)) stop("run failed for ", label)
  readRDS(out)
}

o <- run_one(old_path, "OLD (five hard-coded blocks)")
n <- run_one(new_path, "NEW (method_setup.R + one loop)")

key <- function(d) d |> arrange(sim_num, method)
o <- key(o); n <- key(n)

cat("\n=== structure ===\n")
cat(sprintf("  rows:    old %d, new %d\n", nrow(o), nrow(n)))
cat(sprintf("  columns: old %d, new %d\n", ncol(o), ncol(n)))
cat(sprintf("  column names identical: %s\n", identical(names(o), names(n))))
cat(sprintf("  methods present: old {%s}  new {%s}\n",
            paste(sort(unique(o$method)), collapse = ","),
            paste(sort(unique(n$method)), collapse = ",")))

cat("\n=== column-by-column comparison (run_time excluded) ===\n")
cols <- setdiff(intersect(names(o), names(n)), "run_time")
all_ok <- nrow(o) == nrow(n) && identical(names(o), names(n))
for (cc in cols) {
  a <- o[[cc]]; b <- n[[cc]]
  if (is.numeric(a)) {
    d <- suppressWarnings(max(abs(a - b), na.rm = TRUE))
    same_na <- identical(is.na(a), is.na(b))
    ok <- same_na && (!is.finite(d) || d == 0)
    cat(sprintf("  [%s] %-12s max |diff| = %s   NA pattern %s\n",
                if (ok) "ok  " else "FAIL", cc,
                format(d), if (same_na) "identical" else "DIFFERS"))
  } else {
    ok <- identical(a, b)
    cat(sprintf("  [%s] %-12s %s\n", if (ok) "ok  " else "FAIL", cc,
                if (ok) "identical" else "DIFFERS"))
  }
  all_ok <- all_ok && ok
}

cat("\n=== methods_seq validation (new behaviour) ===\n")
source("sim_functions/method_setup.R")
ml <- build_method_list(as.character(0:5), "5")
t1 <- tryCatch({ check_methods_seq(c("po", "tscopopo"), ml); "no error" },
               error = function(e) conditionMessage(e))
cat(sprintf("  [%s] a typo in methods_seq now errors\n", if (grepl("Unknown", t1)) "ok  " else "FAIL"))
cat(sprintf("        %s\n", sub("\n.*", "", t1)))
t2 <- tryCatch({ check_methods_seq(c("po", "po"), ml); "no error" },
               error = function(e) conditionMessage(e))
cat(sprintf("  [%s] duplicates now error\n", if (grepl("duplicates", t2)) "ok  " else "FAIL"))
all_ok <- all_ok && grepl("Unknown", t1) && grepl("duplicates", t2)

cat("\n=== generators for the secondary studies ===\n")
gen <- tsco_cutoff_methods(as.character(0:5), cutoffs = c("3", "4"))
cat("  tsco_cutoff_methods names:", paste(names(gen), collapse = ", "), "\n")
ok_gen <- length(gen) == 4 && all(sapply(gen, is.function))
# the force() check: each closure must have captured its OWN cutoff
caps <- sapply(names(gen), function(nm) get("cc_", envir = environment(gen[[nm]])))
cat("  captured cutoffs:", paste(sprintf("%s=%s", names(caps), caps), collapse = " "), "\n")
ok_gen <- ok_gen && length(unique(caps)) == 2
cat(sprintf("  [%s] closures capture distinct cutoffs (no lazy-eval collapse)\n",
            if (ok_gen) "ok  " else "FAIL"))
all_ok <- all_ok && ok_gen

cat("\n", strrep("=", 62), "\n", sep = "")
cat(if (all_ok) "REFACTOR IS EQUIVALENT\n" else "*** DIFFERENCES FOUND ***\n")
