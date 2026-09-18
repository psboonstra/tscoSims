# The evaluation set: where the accuracy metrics are computed.
#
# Sourced (not called) from run_sims.R so that the objects it creates stay in
# the workspace for interactive debugging, like draw_single_dataset.R.
#
# Inputs (from the calling environment):
#   eval_by    "grid" or "mc"
#   levels_y, scenario, cutoff_level
#   n_eval_nodes  (grid)   Gauss-Hermite nodes in W
#   n_test, test_seed      (mc)
#
# Outputs (left in the calling environment):
#   test_dat     data.frame of evaluation covariate values (A, W)
#   p_true_test  matrix of true category probabilities on those rows
#   eval_wt      probability mass of each row, summing to 1
#
#
# WHY "grid" IS THE DEFAULT
# -------------------------
# Every metric in score_method.R is an expectation over the covariate
# distribution of a quantity that is already a closed-form function of
# (p_true(x), p_hat(x)). The expectation over Y has been done analytically --
# it is the sum over k inside each formula -- because p_true is known. So the
# only thing left to integrate is X, and X = (A, W) with A binary and W a
# univariate standard normal. That is a two-point sum times a one-dimensional
# Gaussian integral, which Gauss-Hermite handles to machine precision:
# R_n on the grid is converged to nine significant figures at ten nodes, for
# PO, TsCO and CPPO fits alike.
#
# Monte Carlo over a random test set estimates the same expectation with error
# O(1/sqrt(n_test)). At n_test = 20,000 that error is about 1.5% of A_kl and
# 3.1% of A_rps, which is the same order as the effects the regret
# decomposition is trying to resolve. Worse, `test_seed` was a single fixed
# value, so every array and every replicate shared ONE test set: the covariate
# sampling error was a common shift on every number in the study rather than
# noise that averages away over replicates. No number of replicates removes it.
#
# Using the grid also makes R_n and A(M) share an integration rule, so the
# subtraction R_n - A is exact rather than approximately compatible. And it is
# ~8x faster per replicate (80 rows against 20,000), which matters across a
# 6 x 3 x 5 design.
#
# `eval_by = "mc"` keeps the old behaviour, for checking the two against each
# other. It is not the route to report.
#
# Nothing here touches the hypothesis test: stat, df, p_value and reject come
# from the training data alone.

if (!exists("eval_by")) eval_by <- "grid"
if (!exists("n_eval_nodes")) n_eval_nodes <- 40L

if (identical(eval_by, "grid")) {

  if (!exists("kl_quad_grid")) {
    stop("eval_by = \"grid\" needs sim_functions/kl_projection.R sourced first.")
  }

  eval_grid <- kl_quad_grid(n_eval_nodes)
  test_dat <- eval_grid[, c("A", "W")]
  eval_wt <- eval_grid$wt
  p_true_test <- true_prob_scenario(test_dat, scenario, cutoff_level = cutoff_level)

  stopifnot(abs(sum(eval_wt) - 1) < 1e-12, nrow(test_dat) == length(eval_wt))

} else if (identical(eval_by, "mc")) {

  set.seed(test_seed)
  test_dat <- make_covariates(n_test)
  p_true_test <- true_prob_scenario(test_dat, scenario, cutoff_level = cutoff_level)
  eval_wt <- NULL          # score_method() then takes the unweighted average

} else {
  stop("eval_by must be \"grid\" or \"mc\", not ", eval_by)
}
