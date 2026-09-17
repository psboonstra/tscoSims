# Constrained partial proportional odds (Peterson and Harrell, 1990). The A
# effect is allowed to depart from proportionality at the highest cutpoint
# only, through the prespecified contrast G -- the manuscript's ECMO form,
#   Pr(Y >= k | Z) = expit(a_k - z b1 - z 1[k = K-1] b2).
#
# G is derived from `levels_y` via cppo_G() (defined in
# sim_functions/true_probs.R) rather than hard-coded, so that changing the
# number of outcome levels cannot leave a wrong-length G behind. The fitted
# contrast deliberately matches the one the CPPO data-generating mechanism
# uses, so that `cppo_alt` is a scenario where this method is correctly
# specified.
#
# VALIDITY OF THE FIT (added 2026-09-15)
# --------------------------------------
# A CPPO fit is only meaningful inside the CPPO parameter space, i.e. where the
# fitted cumulative probabilities stay ordered for every row. With the
# departure at the last cutpoint that is a single inequality for the exposed
# rows,
#     gamma >= alpha_{K-2} - alpha_{K-1}        (on VGAM's Pr(Y <= j) scale)
# and nothing in VGAM enforces it: the `constraints` argument fixes the LINEAR
# span of the coefficients and cannot express an inequality; the IRLS objective
# sums log mu[i, y_i] over OBSERVED cells only, so a negative fitted probability
# in an EMPTY cell never enters the likelihood; and half-stepping does not help
# because the likelihood goes UP as the fit leaves the space -- assigning
# negative mass to an unoccupied cell frees mass for the occupied ones.
#
# This is not hypothetical. The cell that matters here is the second-highest
# category, which has baseline probability 0.03 and is empty in the exposed arm
# in about 6% of n = 200 datasets. An unconstrained fitter in the Python
# verification harness ran gamma to -7000 on those datasets and reported an LRT
# of ~110 against a null exposure; the constrained answer is ~4. That doubled
# the apparent type I error (0.107 vs 0.063). See verification/FINDINGS.md.
#
# So `fxn_cppo` now (1) fits with VGAM as before, (2) checks the fitted
# probabilities on the training data for validity and the fit for convergence,
# and (3) if either fails, refits BOTH models by direct maximum likelihood with
# the parameter space enforced by construction:
#     alpha_1 = d_1,  alpha_j = alpha_{j-1} - exp(d_j)     decreasing cutpoints
#     gamma   = (alpha_{K-1} - alpha_{K-2}) + exp(t)        ordered for A = 1
# so that unconstrained optimisation over (d, beta, t) can never leave the
# space. A tag is appended to `warnings` whenever this happens, and another
# whenever the final fit sits ON the boundary (the empty-cell case), because
# at the boundary chi-square_2 is not the right reference distribution for the
# LRT and those replicates should be tabulated separately.
#
# `engine = "direct"` skips VGAM and always uses the constrained fit.

cppo_constraints <- function(levels_y, G = cppo_G(levels_y), with_A = TRUE) {
  M <- length(levels_y) - 1L

  if (with_A && length(G) != M) {
    stop("G must have length K - 1.")
  }

  if (with_A) {
    list(
      "(Intercept)" = diag(M),
      A = cbind(common = rep(1, M), deviation = G),
      W = matrix(1, nrow = M, ncol = 1)
    )
  } else {
    list(
      "(Intercept)" = diag(M),
      W = matrix(1, nrow = M, ncol = 1)
    )
  }
}


## ---------------------------------------------------------------------------
## Distance to the boundary of the CPPO parameter space
##
## Two separate questions get two separate functions, because conflating them
## is what hid the unexposed-arm face for as long as it was hidden:
##
##   cppo_vglm_slack()  how far the fit is from the boundary. A property of the
##                      model, measured on the log-odds scale.
##   cppo_vglm_valid()  whether VGAM returned something numerically usable.
##
## A fit can sit exactly ON the boundary and still be perfectly valid
## numerically -- that is the common case -- so the second question does not
## answer the first.
## ---------------------------------------------------------------------------

# Slack in the ordering constraints, from the linear predictors at W = 0.
#
# With eta_j = b0_j + A (b_common + G_j b_dev) + W b_W, both the W term and the
# common A effect are parallel across j and therefore cancel out of every
# consecutive difference. So admissibility is two conditions on the cutpoint
# spacings alone, one per arm, and W = 0 is without loss of generality:
#
#   unexposed rows:  diff(eta_j | A = 0) >= 0
#   exposed rows:    diff(eta_j | A = 1) >= 0
#
# Reading them off `predictvglm` rather than off `coef()` keeps this
# independent of VGAM's naming of constrained coefficients and correct for any
# G, not just the last-cutpoint indicator. The returned values are distances on
# the log-odds scale; negative means the fit has left the space.
cppo_vglm_slack <- function(fit, with_A = TRUE) {
  nd <- if (with_A) data.frame(A = c(0, 1), W = 0) else data.frame(W = 0)

  eta <- tryCatch(
    VGAM::predictvglm(fit, newdata = nd, type = "link"),
    error = function(e) NULL
  )
  if (is.null(eta) || !is.matrix(eta) || ncol(eta) < 2L || !all(is.finite(eta))) {
    return(c(unexposed = NA_real_, exposed = NA_real_))
  }

  gaps <- function(row) min(diff(eta[row, ]))
  if (with_A) {
    c(unexposed = gaps(1L), exposed = gaps(2L))
  } else {
    c(unexposed = gaps(1L), exposed = gaps(1L))
  }
}

# Which faces are active, as tags. `eps` is a log-odds distance; the two
# regimes are separated by nearly two orders of magnitude in practice (a
# well-behaved fit has slack above 0.03, a boundary fit below 1e-7 at n = 200),
# so the threshold is not delicate.
cppo_boundary_tags <- function(slack, eps = 1e-3) {
  tags <- character()
  if (isTRUE(slack[["unexposed"]] < eps)) {
    tags <- c(tags, "cppo: at the boundary -- cutpoint spacing (unexposed arm)")
  }
  if (isTRUE(slack[["exposed"]] < eps)) {
    tags <- c(tags, "cppo: at the boundary -- deviation parameter (exposed arm)")
  }
  tags
}


## ---------------------------------------------------------------------------
## Numerical validity of a VGAM fit
## ---------------------------------------------------------------------------

# TRUE iff every fitted category probability on the training data is finite
# and non-negative (i.e. the cumulative probabilities are ordered for every
# row) and the iterations did not hit the ceiling. Any failure means the
# reported log-likelihood is not the value of the CPPO likelihood at an
# admissible parameter and must not be used in a likelihood-ratio test.
cppo_vglm_valid <- function(fit, tol = 1e-10) {
  mu <- tryCatch(fit@fitted.values, error = function(e) NULL)
  if (is.null(mu)) {
    mu <- tryCatch(VGAM::fitted(fit), error = function(e) NULL)
  }
  if (is.null(mu) || !all(is.finite(mu)) || any(mu < -tol)) {
    return(FALSE)
  }

  converged <- tryCatch(
    is.null(fit@control$maxit) || fit@iter < fit@control$maxit,
    error = function(e) TRUE
  )

  # A fit that overshoots the boundary reports criterion["loglikelihood"] as
  # NaN, which vglm_lrt() turns into an NA statistic. Left unchecked that makes
  # the replicate silently untestable rather than refit.
  usable_loglik <- is.finite(vglm_loglik(fit))

  isTRUE(converged) && usable_loglik
}


## ---------------------------------------------------------------------------
## Direct maximum likelihood with the parameter space enforced
##
## Everything here is on the manuscript's scale, Pr(Y >= k) = expit(alpha_k -
## eta), with alpha decreasing. Only the log-likelihood and the fitted
## probabilities are used downstream, both of which are invariant to the
## choice of scale, so this need not match VGAM's coefficient convention.
## ---------------------------------------------------------------------------

cppo_direct_cutpoints <- function(d) {
  cumsum(c(d[1], -exp(d[-1])))
}

# n x K category probabilities from n x M upper cumulative probabilities.
cppo_direct_probs_from_q <- function(q) {
  M <- ncol(q)
  cbind(1 - q[, 1], q[, -M, drop = FALSE] - q[, -1, drop = FALSE], q[, M])
}

# Reduced model: PO in the non-A columns of X.
po_direct_nll <- function(v, X, Yidx, M) {
  p <- ncol(X)
  alpha <- cppo_direct_cutpoints(v[1:M])
  beta <- v[(M + 1):(M + p)]
  eta <- drop(X %*% beta)
  q <- plogis(outer(-eta, alpha, "+"))
  pr <- cppo_direct_probs_from_q(q)
  -sum(log(pmax(pr[cbind(seq_along(Yidx), Yidx)], 1e-300)))
}

# Full model. gamma is parameterised so that the exposed rows' last two
# cumulative probabilities cannot cross; with G the indicator of the last
# cutpoint this is the only inequality the model has, so the fit is admissible
# for every finite v.
cppo_direct_nll <- function(v, X, Yidx, a_col, G, M) {
  p <- ncol(X)
  alpha <- cppo_direct_cutpoints(v[1:M])
  beta <- v[(M + 1):(M + p)]
  gamma <- (alpha[M] - alpha[M - 1]) + exp(v[M + p + 1])
  eta <- drop(X %*% beta)
  lin <- outer(-eta, alpha, "+") - outer(gamma * X[, a_col], G)
  q <- plogis(lin)
  pr <- cppo_direct_probs_from_q(q)
  if (any(pr < -1e-12)) return(1e10)   # unreachable when G is the last-cutpoint indicator; a guard
  -sum(log(pmax(pr[cbind(seq_along(Yidx), Yidx)], 1e-300)))
}

cppo_direct_start <- function(Yidx, K, p) {
  tab <- tabulate(Yidx, K) + 0.5
  cum <- rev(cumsum(rev(tab)))[-1] / sum(tab)
  a0 <- qlogis(cum)
  c(a0[1], log(pmax(-diff(a0), 1e-3)), rep(0, p))
}

cppo_direct_fit <- function(X, Yidx, a_col, G, K) {
  M <- K - 1L
  p <- ncol(X)
  s0 <- cppo_direct_start(Yidx, K, p)

  best <- NULL
  for (t0 in c(-3, -1, 0.5)) {
    o <- stats::optim(
      c(s0, t0), cppo_direct_nll,
      X = X, Yidx = Yidx, a_col = a_col, G = G, M = M,
      method = "BFGS", control = list(maxit = 5000, reltol = 1e-12)
    )
    if (is.null(best) || o$value < best$value) best <- o
  }

  alpha <- cppo_direct_cutpoints(best$par[1:M])
  list(
    loglik = -best$value,
    alpha = alpha,
    beta = best$par[(M + 1):(M + p)],
    gamma = (alpha[M] - alpha[M - 1]) + exp(best$par[M + p + 1]),
    # Both faces, on the same log-odds scale as cppo_vglm_slack(). The
    # reparameterisation makes each strictly positive by construction, so these
    # are distances, never violations. exp(t) is the exposed-arm face; the
    # cutpoint spacings -diff(alpha) = exp(d_j) are the unexposed-arm face,
    # which binds when the empty cell is in the A = 0 arm.
    slack_exposed = exp(best$par[M + p + 1]),
    slack_unexposed = min(-diff(alpha)),
    converged = best$convergence == 0
  )
}

po_direct_fit <- function(X, Yidx, K) {
  M <- K - 1L
  p <- ncol(X)
  o <- stats::optim(
    cppo_direct_start(Yidx, K, p), po_direct_nll,
    X = X, Yidx = Yidx, M = M,
    method = "BFGS", control = list(maxit = 5000, reltol = 1e-12)
  )
  list(loglik = -o$value, alpha = cppo_direct_cutpoints(o$par[1:M]),
       beta = o$par[(M + 1):(M + p)], converged = o$convergence == 0)
}

cppo_direct_predict <- function(fit, X, a_col, G) {
  M <- length(fit$alpha)
  eta <- drop(X %*% fit$beta)
  lin <- outer(-eta, fit$alpha, "+") - outer(fit$gamma * X[, a_col], G)
  cppo_direct_probs_from_q(plogis(lin))
}


## ---------------------------------------------------------------------------
## The method
## ---------------------------------------------------------------------------

fxn_cppo <- function(dat, test_dat, levels_y,
                     G = cppo_G(levels_y),
                     engine = c("vgam", "direct")) {

  engine <- match.arg(engine)
  K <- length(levels_y)
  M <- K - 1L
  warnings <- character()

  # The direct fitter's reparameterisation assumes the departure is at the
  # last cutpoint. For any other G the admissible region is a different set
  # of inequalities and the fallback below would be wrong, so it is disabled.
  last_cutpoint_G <- length(G) == M && all(G[-M] == 0) && G[M] == 1

  use_direct <- engine == "direct"

  # An outcome level with zero observations in BOTH arms is a programming
  # problem, not a statistical one: VGAM drops the level, M falls to K - 2, and
  # the constraint matrices built from `levels_y` no longer conform, which
  # surfaces as "constraint matrix has too many columns". Detect it and route
  # to the direct fitter, which keeps all K levels and handles the empty cell
  # as the boundary case it is.
  lvl_counts <- tabulate(match(as.character(dat$Y), levels_y), K)
  if (any(lvl_counts == 0L)) {
    warnings <- c(warnings, sprintf(
      "cppo: outcome level(s) %s unobserved; VGAM cannot be used, fitting by direct ML",
      paste(levels_y[lvl_counts == 0L], collapse = ", ")))
    # With the departing level itself unobserved the deviation parameter has
    # nothing to attach to: the MLE puts gamma at 0, the statistic equals the
    # PO statistic, and the df = 2 reference is conservative.
    if (lvl_counts[K - 1L] == 0L) {
      warnings <- c(warnings,
        "cppo: departing level unobserved; deviation parameter not identified, df = 2 is conservative")
    }
    use_direct <- TRUE
  }

  if (!use_direct) {
    fit_full <- safe_fit(
      VGAM::vglm(
        Y ~ A + W,
        data = dat,
        family = VGAM::cumulative(link = "logitlink", parallel = FALSE, reverse = FALSE),
        constraints = cppo_constraints(levels_y, G, with_A = TRUE)
      )
    )

    fit_red <- safe_fit(
      VGAM::vglm(
        Y ~ W,
        data = dat,
        family = VGAM::cumulative(link = "logitlink", parallel = FALSE, reverse = FALSE),
        constraints = cppo_constraints(levels_y, G, with_A = FALSE)
      )
    )

    warnings <- c(fit_full$warnings, fit_red$warnings)

    vgam_ok <- fit_full$ok && fit_red$ok &&
      cppo_vglm_valid(fit_full$fit) && cppo_vglm_valid(fit_red$fit)

    # Boundary proximity is recorded whenever VGAM produced a fit at all, and
    # it does NOT trigger a refit: where VGAM returns a usable answer it agrees
    # with the constrained direct MLE to 3e-5, so there is nothing to fix. The
    # tag exists because chi-square_2 is the wrong reference there, which is a
    # question about the reference distribution and not about the estimate.
    if (fit_full$ok) {
      slack <- cppo_vglm_slack(fit_full$fit, with_A = TRUE)
      warnings <- c(warnings, cppo_boundary_tags(slack))
    }

    if (vgam_ok) {
      tst <- vglm_lrt(fit_full$fit, fit_red$fit)

      p_hat <- tryCatch(
        align_prob(
          VGAM::predictvglm(fit_full$fit, newdata = test_dat, type = "response"),
          levels_y
        ),
        error = function(e) NA
      )

      return(list(
        fit_ok = TRUE,
        p_hat = p_hat,
        stat = tst["stat"],
        df = tst["df"],
        p_value = tst["p"],
        warnings = warnings
      ))
    }

    # VGAM either errored, failed to converge, or produced a fit outside the
    # CPPO parameter space. Fall through to the constrained direct fit.
    if (!last_cutpoint_G) {
      return(null_fit(c(
        warnings,
        "cppo: VGAM fit invalid and G is not the last-cutpoint indicator, so no constrained refit is available"
      )))
    }

    warnings <- c(warnings, "cppo: VGAM fit invalid or non-convergent; refit by constrained direct ML")
    use_direct <- TRUE
  }

  if (!last_cutpoint_G) {
    return(null_fit(c(warnings, "cppo: direct engine requires G to be the last-cutpoint indicator")))
  }

  ## Constrained direct maximum likelihood for both models.
  Yidx <- match(as.character(dat$Y), levels_y)
  if (any(is.na(Yidx))) {
    return(null_fit(c(warnings, "cppo: outcome levels do not match levels_y")))
  }

  X <- cbind(A = dat$A, W = dat$W)
  X_red <- X[, "W", drop = FALSE]

  full <- tryCatch(cppo_direct_fit(X, Yidx, a_col = 1L, G = G, K = K),
                   error = function(e) NULL)
  red <- tryCatch(po_direct_fit(X_red, Yidx, K = K), error = function(e) NULL)

  if (is.null(full) || is.null(red)) {
    return(null_fit(c(warnings, "cppo: direct fit failed")))
  }
  if (!full$converged || !red$converged) {
    warnings <- c(warnings, "cppo: direct optimiser reported non-convergence")
  }
  warnings <- c(warnings, cppo_boundary_tags(
    c(unexposed = full$slack_unexposed, exposed = full$slack_exposed)))

  # (common A effect + deviation) vs none: two parameters.
  df <- 2
  stat <- 2 * (full$loglik - red$loglik)
  if (is.finite(stat) && stat < 0 && stat > -1e-7) stat <- 0
  p <- if (is.finite(stat)) stats::pchisq(stat, df = df, lower.tail = FALSE) else NA_real_

  p_hat <- tryCatch({
    X_test <- cbind(A = test_dat$A, W = test_dat$W)
    pr <- cppo_direct_predict(full, X_test, a_col = 1L, G = G)
    colnames(pr) <- levels_y
    align_prob(pr, levels_y)
  }, error = function(e) NA)

  list(
    fit_ok = TRUE,
    p_hat = p_hat,
    stat = c(stat = stat),
    df = c(df = df),
    p_value = c(p = p),
    warnings = warnings
  )
}
