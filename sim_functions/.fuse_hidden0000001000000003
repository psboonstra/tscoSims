# The methods, as a named list of closures.
#
# WHY THIS FILE EXISTS
# --------------------
# Every method in this study has the SAME signature: it takes a training set
# and an evaluation set and returns the standardised list that score_method()
# consumes. The only thing that differs between methods is which fxn_* to call
# and what tuning arguments to pass it. So the per-method blocks in run_sims.R
# were five near-identical twelve-line copies whose only distinguishing content
# was a name and a function call.
#
# Three reasons to collapse them:
#
#   1. The tuning sweeps cannot be written as hard-coded blocks. The C sweep
#      needs TsCO at C in {2,3,4,5} crossed with the two stage-2 families --
#      eight entries -- and the G sweep needs one CPPO entry per shape of G.
#      Those have to be GENERATED, which means the method set has to be data.
#   2. With a `which_study` switch, each study wants a different method set.
#      Hard-coded blocks mean either three copies of the script or a thicket
#      of conditionals inside one.
#   3. Five copies of the scoring-and-timing boilerplate drift. Any change to
#      what gets recorded has to be made five times, or four.
#
# This is a deliberate departure from modelintegrate's literal-repetition
# idiom. There the methods took genuinely different arguments -- theta_tilde,
# n_hist, a long tail of Bayesian tuning parameters -- so explicit blocks
# earned their place by documenting those differences at the call site. Here
# there are no differences to document.
#
# The list stays explicit and readable, and each entry is still callable by
# hand at the console for debugging:
#
#     method_list <- build_method_list(levels_y, cutoff_level)
#     fit <- method_list$cppo(dat, test_dat)
#
# Adding a method is one line here plus its methods/*.R file. Nothing in
# run_sims.R changes.

# The main study's five methods.
#
# `cutoff_level` is the paper's C, i.e. the first level of the UPPER partition.
# Note that at cutoff_level = "5" the upper partition holds a single level, so
# tsco_pomr and tsco_popo are the SAME model; use a smaller C to separate the
# stage-2 families.
build_method_list <- function(levels_y, cutoff_level) {
  list(
    po        = function(dat, test_dat) fxn_po(dat, test_dat, levels_y),
    mr        = function(dat, test_dat) fxn_mr(dat, test_dat, levels_y),
    cppo      = function(dat, test_dat) fxn_cppo(dat, test_dat, levels_y),
    tsco_pomr = function(dat, test_dat) fxn_tsco_pomr(dat, test_dat, levels_y, cutoff_level),
    tsco_popo = function(dat, test_dat) fxn_tsco_popo(dat, test_dat, levels_y, cutoff_level)
  )
}


## ---------------------------------------------------------------------------
## Generators for the secondary studies
##
## Not used by the main study. They are here because they are the reason the
## list exists, and because writing them next to build_method_list() keeps the
## naming convention in one place: a method's name is its column value in
## `all_scores`, so it has to encode the tuning parameter.
##
## `force()` matters in both. Without it every closure would capture the same
## loop variable and, because R's arguments are lazy, they would all end up
## using its final value.
## ---------------------------------------------------------------------------

# TsCO at several cutoffs: tsco_popo_C3, tsco_pomr_C3, ...
tsco_cutoff_methods <- function(levels_y, cutoffs, stage2 = c("po", "multinomial")) {
  out <- list()
  for (s2 in stage2) for (cc in cutoffs) {
    local({
      s2_ <- force(s2); cc_ <- force(cc)
      nm <- sprintf("tsco_po%s_C%s", if (s2_ == "po") "po" else "mr", cc_)
      out[[nm]] <<- function(dat, test_dat)
        fxn_tsco(dat, test_dat, levels_y, cc_, stage1 = "po", stage2 = s2_)
    })
  }
  out
}

# CPPO under several shapes of G: cppo_<name>.
#
# Note the constraint in methods/cppo.R: the constrained direct-ML fallback is
# only valid when G is the last-cutpoint indicator, and returns null_fit with a
# tag for any other G. A G sweep needs that path written first -- the
# admissible region has a different set of faces for each G, so the
# reparameterisation does not transfer.
cppo_G_methods <- function(levels_y, G_list) {
  out <- list()
  for (nm in names(G_list)) {
    local({
      G_ <- force(G_list[[nm]])
      out[[paste0("cppo_", nm)]] <<- function(dat, test_dat)
        fxn_cppo(dat, test_dat, levels_y, G = G_)
    })
  }
  out
}


## ---------------------------------------------------------------------------
## Validation
##
## The old per-method blocks were `if (name %in% methods_seq)`, so a typo in
## methods_seq silently ran nothing and produced a scores table missing that
## method with no error anywhere. Check the names up front instead.
## ---------------------------------------------------------------------------
check_methods_seq <- function(methods_seq, method_list) {
  unknown <- setdiff(methods_seq, names(method_list))
  if (length(unknown)) {
    stop("Unknown method(s) in methods_seq: ", paste(unknown, collapse = ", "),
         "\n  available: ", paste(names(method_list), collapse = ", "))
  }
  if (anyDuplicated(methods_seq)) {
    stop("methods_seq has duplicates: ",
         paste(unique(methods_seq[duplicated(methods_seq)]), collapse = ", "))
  }
  invisible(TRUE)
}
