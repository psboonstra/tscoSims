# Draws the evaluation set used to score predicted probabilities.
#
# Drawn ONCE per array, outside the replicate loop, from a seed that depends
# only on `test_seed`. The covariates are therefore common to every replicate
# and every scenario, so differences in the accuracy metrics reflect the
# fitted models rather than the evaluation sample.
#
# input (from the calling environment):
#   n_test, scenario, test_seed
#
# output (left in the calling environment):
#   test_dat     (data.frame) columns A, W
#   p_true_test  (matrix)     n_test x K true category probabilities

set.seed(test_seed)

test_dat <- make_covariates(n_test)

p_true_test <- true_prob_scenario(test_dat, scenario)
