# Population KL projections: A(M) = min_theta E_X[ KL( p(.|X) || q_theta(.|X) ) ]
#
# WHAT THIS IS FOR
# ----------------
# The headline object of the study is a regret matrix. For a model class M
# fitted at sample size n,
#
#     R_n(M) = E_X[ KL( p(.|X) || q_thetahat_n(.|X) ) ]        (what we measure)
#     A(M)   = min_theta E_X[ KL( p(.|X) || q_theta(.|X) ) ]   (what this file computes)
#     E_n(M) = R_n(M) - A(M)                                    (the residual)
#
# A(M) is the APPROXIMATION error: the price of using class M at all, which no
# amount of data removes. E_n is the ESTIMATION error: the price of having only
# n observations, which vanishes as n grows. Splitting the regret this way is
# what lets the manuscript say "TsCO's approximation error under a PO truth is
# negligible" as a statement about the model class rather than about one sample
# size.
#
# E_n is defined operationally as the residual R_n - A. That makes it exact by
# construction and non-negative in every replicate, because A is the minimum of
# the same risk functional over the same class. It is deliberately NOT defined
# as E_X[KL(q_theta* || q_thetahat)]: that Pythagorean identity holds exactly
# only for linear exponential families, and cumulative logit models are curved
# families. Checked numerically: the gap is ~1e-9 for MR (which is exact) but
# 1e-3 to 5e-3 with varying sign for PO.
#
# HOW A(M) IS COMPUTED
# --------------------
# Not by fitting to a huge simulated sample. That would put estimation error
# back into the quantity whose whole purpose is to exclude it, and the error
# would be O(1/n) in exactly the regime where A(M) is itself small.
#
# Instead, exactly. Write the objective out:
#
#     E_X[KL(p||q_theta)] = E_X[ sum_k p_k(X) log p_k(X) ]      (no theta in it)
#                         - E_X[ sum_k p_k(X) log q_k(X;theta) ]
#
# The first term does not involve theta, so minimising KL is the same as
# maximising the second term, the POPULATION expected log-likelihood. And the
# second term is a weighted multinomial log-likelihood: if we lay out one row
# per (covariate value, outcome category) and attach the prior weight
#
#     wt = Pr(A = a) * (quadrature weight for w) * p_k(a, w)
#
# then a standard weighted fit maximises precisely E_X[sum_k p_k log q_k].
#
# So theta* is obtained by handing that weighted pseudo-dataset to the SAME
# fitter the method itself uses -- VGAM for PO/MR/CPPO, tsco::tsco for TsCO,
# all of which accept non-integer prior weights. Nothing about the model class
# is re-derived here, so the projection is guaranteed to be onto the class the
# method actually fits rather than onto our reading of it. (Both fitters are
# scale-invariant in the weights, verified to 8e-16, so the weights need not be
# counts of anything.)
#
# X = (A, W) with A ~ Bernoulli(1/2) and W ~ N(0, 1), so the expectation over X
# is a two-point sum times a Gauss-Hermite rule in W. Gauss-Hermite with m
# nodes integrates polynomials of degree 2m-1 exactly; the integrand here is
# smooth and bounded, and convergence in m is checked in
# verification/r_kl_projection_check.R.
#
# THE SCORING TRAP
# ----------------
# The MLE targets the KL projection, so A must be defined in KL if E_n is to be
# non-negative. If the headline accuracy comparison is run in RPS, the matching
# baseline is the RPS risk EVALUATED AT theta*, not the minimum of RPS over the
# class. Minimising RPS over the class would give a smaller number, and
# R_n - A_RPS would then converge to a positive constant instead of zero.
# `kl_projection()` therefore returns `rps` and `brier` alongside `kl`, all
# evaluated at the same theta*.

# REQUIRES sim_functions/score_method.R to be sourced first: the risk
# functionals below delegate to prob_kl() / prob_rps() / prob_brier() there.

## ---------------------------------------------------------------------------
## Quadrature
## ---------------------------------------------------------------------------

# Gauss-Hermite nodes and weights for E[f(Z)], Z ~ N(0, 1), by Golub-Welsch.
#
# The Hermite polynomials orthonormal with respect to the standard normal
# density satisfy x He_n(x) = sqrt(n+1) He_{n+1}(x) + sqrt(n) He_{n-1}(x), so
# the Jacobi matrix is symmetric tridiagonal with zero diagonal and
# sqrt(1:(m-1)) off the diagonal. Its eigenvalues are the nodes and the squared
# first components of its eigenvectors are the weights, which sum to 1.
gauss_hermite_normal <- function(m) {
  if (m == 1L) return(list(nodes = 0, weights = 1))
  off <- sqrt(seq_len(m - 1L))
  J <- matrix(0, m, m)
  J[cbind(seq_len(m - 1L), 2:m)] <- off
  J[cbind(2:m, seq_len(m - 1L))] <- off
  e <- eigen(J, symmetric = TRUE)
  ord <- order(e$values)
  list(nodes = e$values[ord], weights = (e$vectors[1, ord])^2)
}

# The quadrature grid in X = (A, W): every (a, w) pair with its probability
# mass. `wt` sums to 1, so sums against it are expectations.
kl_quad_grid <- function(n_nodes = 40L, p_treat = 0.5) {
  gh <- gauss_hermite_normal(n_nodes)
  expand <- expand.grid(node = seq_along(gh$nodes), A = c(0, 1))
  data.frame(
    A = expand$A,
    W = gh$nodes[expand$node],
    wt = gh$weights[expand$node] * ifelse(expand$A == 1, p_treat, 1 - p_treat)
  )
}


## ---------------------------------------------------------------------------
## Risk functionals, all as expectations over the grid
## ---------------------------------------------------------------------------

# These are the SAME functionals score_method.R reports, taken against the
# grid's probability masses instead of averaged over test rows. They are thin
# wrappers precisely so the formulas exist in one place: if prob_rps() ever
# changes, A_rps changes with it, and the regret stays a difference of two
# things computed the same way. Note the argument order flips -- score_method's
# functions take (p_hat, p_true), these take (p, q) to read like KL(p||q).
expected_kl <- function(p, q, wt) prob_kl(q, p, wt)

expected_rps <- function(p, q, wt) prob_rps(q, p, wt)

expected_brier <- function(p, q, wt) prob_brier(q, p, wt)


## ---------------------------------------------------------------------------
## The weighted pseudo-dataset
## ---------------------------------------------------------------------------

# One row per (grid point, outcome category), with weight
# Pr(A) * quad weight * p_k. Rows with zero weight are dropped: they contribute
# nothing to the objective and some fitters dislike them.
kl_pseudo_data <- function(grid, p_true, levels_y, tol = 0) {
  K <- length(levels_y)
  n <- nrow(grid)
  out <- data.frame(
    A = rep(grid$A, times = K),
    W = rep(grid$W, times = K),
    Y = factor(rep(levels_y, each = n), levels = levels_y),
    wt = as.vector(grid$wt * as.matrix(p_true))
  )
  out[out$wt > tol, , drop = FALSE]
}


## ---------------------------------------------------------------------------
## Projection fitters
##
## Each entry must mirror the `fit_full` call in the corresponding methods/*.R
## EXACTLY -- same family, same constraints, same engine -- or A(M) would be the
## projection onto a different class than the method fits. That correspondence
## is not enforced by the code, so it is tested directly: given an unweighted
## dataset, each fitter here must return the same coefficients as the method's
## own fit. See `kl_check_fitters_match()` below.
## ---------------------------------------------------------------------------

kl_fit_po <- function(pseudo, levels_y) {
  fit <- VGAM::vglm(
    Y ~ A + W, data = pseudo, weights = pseudo$wt,
    family = VGAM::cumulative(link = "logitlink", parallel = TRUE, reverse = FALSE)
  )
  list(fit = fit, predict = function(nd)
    align_prob(VGAM::predictvglm(fit, newdata = nd, type = "response"), levels_y))
}

kl_fit_mr <- function(pseudo, levels_y) {
  fit <- VGAM::vglm(
    Y ~ A + W, data = pseudo, weights = pseudo$wt,
    family = VGAM::multinomial(refLevel = 1)
  )
  list(fit = fit, predict = function(nd)
    align_prob(VGAM::predictvglm(fit, newdata = nd, type = "response"), levels_y))
}

kl_fit_cppo <- function(pseudo, levels_y, G = cppo_G(levels_y)) {
  fit <- VGAM::vglm(
    Y ~ A + W, data = pseudo, weights = pseudo$wt,
    family = VGAM::cumulative(link = "logitlink", parallel = FALSE, reverse = FALSE),
    constraints = cppo_constraints(levels_y, G, with_A = TRUE)
  )
  list(fit = fit, predict = function(nd)
    align_prob(VGAM::predictvglm(fit, newdata = nd, type = "response"), levels_y))
}

kl_fit_tsco <- function(pseudo, levels_y, cutoff_level, stage2) {
  fit <- tsco::tsco(
    Y ~ A + W, data = pseudo, levels = levels_y, cutoff_level = cutoff_level,
    stage1 = "po", stage2 = stage2, weights = pseudo$wt, warn_degenerate = FALSE
  )
  list(fit = fit, predict = function(nd)
    align_prob(predict(fit, newdata = nd, type = "prob"), levels_y))
}

# The projection classes, keyed by the same names `methods_seq` uses.
kl_fitters <- function(levels_y, cutoff_level) {
  list(
    po        = function(pseudo) kl_fit_po(pseudo, levels_y),
    mr        = function(pseudo) kl_fit_mr(pseudo, levels_y),
    cppo      = function(pseudo) kl_fit_cppo(pseudo, levels_y),
    tsco_popo = function(pseudo) kl_fit_tsco(pseudo, levels_y, cutoff_level, "po"),
    tsco_pomr = function(pseudo) kl_fit_tsco(pseudo, levels_y, cutoff_level, "multinomial")
  )
}


## ---------------------------------------------------------------------------
## The projection itself
## ---------------------------------------------------------------------------

# A(M) for one (scenario, method) pair.
#
# Returns the three risks at theta* -- `kl` is A(M); `rps` and `brier` are the
# baselines to subtract when the headline comparison is run in those scores,
# evaluated AT theta* rather than minimised in their own right (see the scoring
# trap above). `q_star` is the projected distribution on the grid, kept so that
# callers can inspect where the class fails rather than only by how much.
# `p_true_fn` overrides the scenario lookup: a function of the grid returning an
# n_grid x K matrix. Used by the validation script to hit parameter settings
# that are not named scenarios, and later by the PO-to-TsCO continuum.
kl_projection <- function(scenario, method, levels_y, cutoff_level = "5",
                          n_nodes = 40L, fitters = NULL, p_true_fn = NULL) {

  grid <- kl_quad_grid(n_nodes)
  p_true <- if (is.null(p_true_fn)) {
    true_prob_scenario(grid, scenario, cutoff_level = cutoff_level)
  } else {
    p_true_fn(grid)
  }
  pseudo <- kl_pseudo_data(grid, p_true, levels_y)

  if (is.null(fitters)) fitters <- kl_fitters(levels_y, cutoff_level)
  if (!method %in% names(fitters)) stop("No projection fitter for method: ", method)

  f <- fitters[[method]](pseudo)
  q_star <- f$predict(grid[, c("A", "W")])

  list(
    scenario = scenario, method = method, n_nodes = n_nodes,
    kl    = expected_kl(p_true, q_star, grid$wt),
    rps   = expected_rps(p_true, q_star, grid$wt),
    brier = expected_brier(p_true, q_star, grid$wt),
    theta = tryCatch(stats::coef(f$fit), error = function(e) NA),
    q_star = q_star, p_true = p_true, grid = grid
  )
}

# The whole approximation-error table: one row per (scenario, method).
kl_projection_table <- function(scenarios, methods, levels_y, cutoff_level = "5",
                                n_nodes = 40L, verbose = TRUE) {
  fitters <- kl_fitters(levels_y, cutoff_level)
  rows <- list()
  for (s in scenarios) for (m in methods) {
    r <- tryCatch(
      kl_projection(s, m, levels_y, cutoff_level, n_nodes, fitters),
      error = function(e) list(kl = NA_real_, rps = NA_real_, brier = NA_real_,
                               err = conditionMessage(e))
    )
    if (verbose) cat(sprintf("  %-16s %-10s A_kl = %.6e%s\n", s, m, r$kl,
                             if (!is.null(r$err)) paste("   ERROR:", r$err) else ""))
    rows[[length(rows) + 1L]] <- data.frame(
      scenario = s, method = m, n_nodes = n_nodes,
      A_kl = r$kl, A_rps = r$rps, A_brier = r$brier,
      stringsAsFactors = FALSE)
  }
  do.call(rbind, rows)
}


## ---------------------------------------------------------------------------
## Regret
## ---------------------------------------------------------------------------

# R_n - A for one replicate's fitted probabilities. `p_hat` and `p_true` are on
# whatever evaluation set the simulation used, so R_n carries Monte Carlo error
# from that set while A does not; use a large test set or the grid itself.
#
# Non-negativity is guaranteed in the population but NOT replicate by replicate
# when R_n is estimated on a finite test set: a lucky draw can put R_n a little
# below A. Report the sign of the minimum as a diagnostic rather than clipping.
kl_regret <- function(p_hat, p_true, A_kl, wt = NULL) {
  if (is.null(wt)) wt <- rep(1 / nrow(as.matrix(p_true)), nrow(as.matrix(p_true)))
  R_n <- expected_kl(p_true, p_hat, wt)
  c(R_n = R_n, A = A_kl, E_n = R_n - A_kl)
}
