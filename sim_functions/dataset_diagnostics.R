# Sparsity diagnostics for one simulated dataset.
#
# WHY THESE COLUMNS EXIST
# -----------------------
# Finite-sample behaviour in this study is driven by empty cells, and the rate
# is not a footnote. Measured over 4000 draws per cell:
#
#   category 4 empty in...        exposed   unexposed   either   BOTH arms
#   null            n = 200          6.2%       5.3%    11.2%       0.3%
#   po_alt          n = 200          6.6%       5.9%    12.1%       0.4%
#   cppo_alt        n = 200         63.8%       6.2%    66.0%       4.0%
#   cppo_alt        n = 500         32.8%       0.0%    32.9%       0.0%
#   cppo_alt        n = 1000        10.6%       0.0%    10.6%       0.0%
#   tsco_alt        n = 200         13.0%       5.5%    17.9%       0.6%
#   none_true       n = 200         15.0%       5.5%    19.8%       0.7%
#
# `cppo_alt` sets b2 = -0.1, which drops the exposed arm's category-4
# probability from 0.030 to 0.005 -- half an observation per hundred. So in the
# one scenario where CPPO is correctly specified, its deviation parameter has
# NO exposed-arm data in the category it governs in about two thirds of n = 200
# datasets, and the constrained MLE sits on the boundary of the CPPO parameter
# space there (see methods/cppo.R and verification/FINDINGS.md section 3).
#
# These columns exist so that an unconditional result can be EXPLAINED --
# "CPPO's size is 0.069; the excess sits in the 11% of datasets with an empty
# departure cell; the boundary reference distribution is the mechanism" -- and
# reconstructing them afterwards from `data_seed` means re-drawing every
# dataset. Recording them at run time costs one table() per replicate.
#
# They are diagnostics, not strata. Reported size, power and regret are
# unconditional: emptiness is a function of the outcome and, under `cppo_alt`,
# of the very effect under test, so splitting operating characteristics on it
# conditions on a post-treatment variable. See process_main_results.R.
#
# The columns generalise past category 4 and past K = 6: they are counts of
# empty cells and minima, plus the two counts for the departure category, which
# is the one CPPO's G singles out.

dataset_diagnostics <- function(dat, levels_y, departure_index = length(levels_y) - 1L) {

  y <- factor(as.character(dat$Y), levels = levels_y)
  tab <- table(A = factor(dat$A, levels = c(0, 1)), Y = y)

  n_exp <- as.integer(tab["1", ])
  n_unexp <- as.integer(tab["0", ])
  n_all <- n_exp + n_unexp

  dep <- departure_index      # level K-1 in 1-based indexing = "4" when K = 6

  tibble::tibble(
    # levels unobserved in the whole dataset: VGAM drops these, which breaks a
    # constraint matrix built from levels_y and makes mr change its own df
    n_empty_levels = sum(n_all == 0L),
    # empty cells within each arm, the quantity the CPPO boundary tracks
    n_empty_exp = sum(n_exp == 0L),
    n_empty_unexp = sum(n_unexp == 0L),
    min_cell = min(n_all),
    # the departure category specifically
    cell_dep_exp = n_exp[dep],
    cell_dep_unexp = n_unexp[dep]
  )
}
