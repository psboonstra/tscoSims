# Draws one training dataset. Sourced (not called) from run_sims.R so that
# every object it creates stays visible in the workspace for interactive
# debugging, exactly as in modelintegrate.
#
# input (from the calling environment):
#   n         (scalar)    sample size for this replicate
#   scenario  (character) name of the data-generating mechanism
#   levels_y  (character) ordered outcome levels
#
# output (left in the calling environment):
#   dat     (data.frame) columns A, W, Y
#   p_true  (matrix)     n x K true category probabilities for these rows

dat <- make_covariates(n)

p_true <- true_prob_scenario(dat, scenario)

dat$Y <- draw_ordinal(p_true, levels_y = levels_y)
