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
# the parameter space enforced as box constraints on the cutpoint spacings and
# on the exposed-arm slack (see the direct-fitter section below). A tag is
# appended to `warnings` whenever this happens.
#
# `engine = "direct"` skips VGAM and always uses the constrained fit.
#
# EVALUATION POLICY FOR THE BOUNDARY (settled 2026-09-18)
# -------------------------------------------------------
# A fit whose constrained MLE sits ON the boundary -- an empty departure cell
# in the exposed arm, or an empty outcome level -- is a valid fit with valid
# predictions. What is not justified there is the chi-square_2 reference for
# the LRT. The harness records the boundary event in the `boundary` field and
# STILL returns the chi-square_2 p-value, because that is what CPPO as
# practised does; the reported headline rejection rate is therefore the
# unconditional operating characteristic of the conventional procedure, and
# the boundary rate and the rejection rate among interior fits are reported
# alongside as the diagnostic decomposition. Boundary replicates are never
# dropped: the event is a function of the outcome and, under `cppo_alt`, of the
# very effect under test, so conditioning on it would bias the comparison.
#
# A fit that did NOT converge is a different matter: it is a failure of the
# fitter, not a property of the model, and is returned as fit_ok = FALSE.

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
##
## PARAMETERISATION (rewritten 2026-09-18, after external review)
## ---------------------------------------------------------------
## The admissible region is a set of non-negativity constraints:
##
##     s_j = alpha_{j-1} - alpha_j  >= 0     j = 2..M     cutpoint spacings
##     u   = gamma + s_M            >= 0                  exposed-arm face
##
## (u >= 0 is gamma >= alpha_M - alpha_{M-1}: the exposed rows' last two
## cumulative probabilities may not cross. With G the last-cutpoint indicator
## it is the only inequality gamma introduces.)
##
## The previous version wrote s_j = exp(d_j) and u = exp(t), which keeps every
## finite parameter admissible but makes the BOUNDARY unattainable: an empty
## outcome category has its MLE at s_j = 0 exactly, and exp() can only approach
## that. The supremum then has no finite optimiser, BFGS "converges" wherever
## the objective goes numerically flat at a very negative d_j, and the reported
## optimum, likelihood and LRT depend on stopping tolerances. That is precisely
## the regime the direct fitter exists for, so it was unreliable exactly where
## it was used.
##
## Now the raw s_j and u are the parameters, with lower bounds of zero handled
## by L-BFGS-B. The boundary is attainable; "on the boundary" means "a bound is
## active", read off exactly rather than through a threshold; and a
## convergence code refers to a point that exists.
##
## Layout of the parameter vector v, shared by both models:
##
##     v[1]              alpha_1                      free
##     v[2..M]           s_2..s_M                     >= 0
##     v[M+1 .. M+p]     beta                         free
##     v[M+p+1]          u                            >= 0    (full model only)
##
## CONVERGENCE (issue 5 of the review)
## -----------------------------------
## optim's convergence flag is necessary, not sufficient. Each fit is also
## checked against the Karush-Kuhn-Tucker conditions at the returned point:
## the gradient of the negative log-likelihood must vanish in every free
## coordinate and be non-negative in every coordinate sitting at its lower
## bound (moving into the feasible region must not improve the objective).
## `converged` is TRUE only if both hold. A failed fit is reported as a
## failure, not as fit_ok = TRUE with a tag.
## ---------------------------------------------------------------------------

cppo_direct_cutpoints <- function(v, M) {
  cumsum(c(v[1], -v[2:M]))
}

# n x K category probabilities from n x M upper cumulative probabilities.
cppo_direct_probs_from_q <- function(q) {
  M <- ncol(q)
  cbind(1 - q[, 1], q[, -M, drop = FALSE] - q[, -1, drop = FALSE], q[, M])
}

# Log-likelihood contribution. An empty category at an active bound has
# probability exactly zero, which never enters because no observation sits
# there; an OBSERVED category driven to zero is penalised through the floor,
# and the optimiser moves away from it.
.cppo_direct_ll <- function(pr, Yidx) {
  sum(log(pmax(pr[cbind(seq_along(Yidx), Yidx)], 1e-300)))
}

# Reduced model: PO in the non-A columns of X.
po_direct_nll <- function(v, X, Yidx, M) {
  p <- ncol(X)
  alpha <- cppo_direct_cutpoints(v, M)
  beta <- v[(M + 1):(M + p)]
  eta <- drop(X %*% beta)
  q <- plogis(outer(-eta, alpha, "+"))
  -.cppo_direct_ll(cppo_direct_probs_from_q(q), Yidx)
}

# Full model.
cppo_direct_nll <- function(v, X, Yidx, a_col, G, M) {
  p <- ncol(X)
  alpha <- cppo_direct_cutpoints(v, M)
  beta <- v[(M + 1):(M + p)]
  gamma <- v[M + p + 1] - v[M]              # u - s_M
  eta <- drop(X %*% beta)
  lin <- outer(-eta, alpha, "+") - outer(gamma * X[, a_col], G)
  pr <- cppo_direct_probs_from_q(plogis(lin))
  if (any(pr < -1e-12)) return(1e10)        # unreachable for last-cutpoint G; a guard
  -.cppo_direct_ll(pr, Yidx)
}

# Analytic gradients. With numerical gradients L-BFGS-B stops where ITS
# estimate of the gradient vanishes, which at optim's default step of 1e-3 is
# ~1e-3 away from stationarity -- enough to fail an honest KKT check while
# reporting convergence. The gradients below are exact.
#
# With category k having probability pr_k = q_{k-1} - q_k (q_0 = 1, q_K = 0),
#   d nll / d lin_{ij} = -(1 / pr_{i,y_i}) * q_ij (1 - q_ij) * (1[j = y_i - 1] - 1[j = y_i])
# and lin_ij = alpha_j - eta_i - gamma A_i G_j. `D` below is that n x M matrix.
.cppo_direct_D <- function(q, Yidx, floor = 1e-12) {
  n <- nrow(q); M <- ncol(q)
  pr <- cppo_direct_probs_from_q(q)
  py <- pmax(pr[cbind(seq_len(n), Yidx)], floor)
  D <- matrix(0, n, M)
  # j = y - 1 contributes +1; j = y contributes -1 (only where those j exist)
  has_lo <- Yidx >= 2L
  D[cbind(which(has_lo), Yidx[has_lo] - 1L)] <- 1
  has_hi <- Yidx <= M
  D[cbind(which(has_hi), Yidx[has_hi])] <- D[cbind(which(has_hi), Yidx[has_hi])] - 1
  -D * q * (1 - q) / py
}

# Chain rule from (d nll / d alpha_j) to the spacing parameterisation:
# alpha_j = v_1 - sum_{l=2}^{j} v_l, so d/dv_1 = sum_j g_j and d/dv_l = -sum_{j>=l} g_j.
.cppo_direct_alpha_to_v <- function(g_alpha) {
  M <- length(g_alpha)
  c(sum(g_alpha), -rev(cumsum(rev(g_alpha)))[-1])
}

po_direct_grad <- function(v, X, Yidx, M) {
  p <- ncol(X)
  alpha <- cppo_direct_cutpoints(v, M)
  beta <- v[(M + 1):(M + p)]
  eta <- drop(X %*% beta)
  q <- plogis(outer(-eta, alpha, "+"))
  D <- .cppo_direct_D(q, Yidx)
  g_alpha <- colSums(D)
  g_beta <- -drop(crossprod(X, rowSums(D)))
  c(.cppo_direct_alpha_to_v(g_alpha), g_beta)
}

cppo_direct_grad <- function(v, X, Yidx, a_col, G, M) {
  p <- ncol(X)
  alpha <- cppo_direct_cutpoints(v, M)
  beta <- v[(M + 1):(M + p)]
  gamma <- v[M + p + 1] - v[M]
  eta <- drop(X %*% beta)
  lin <- outer(-eta, alpha, "+") - outer(gamma * X[, a_col], G)
  q <- plogis(lin)
  D <- .cppo_direct_D(q, Yidx)
  g_alpha <- colSums(D)
  g_beta <- -drop(crossprod(X, rowSums(D)))
  g_gamma <- -sum(D * outer(X[, a_col], G))
  g_v <- c(.cppo_direct_alpha_to_v(g_alpha), g_beta, g_gamma)
  g_v[M] <- g_v[M] - g_gamma               # gamma = u - s_M
  g_v
}

# Data-driven start: cutpoints from the marginal outcome distribution, with
# half an observation added to every cell so that an empty one starts inside
# the region rather than on its edge.
cppo_direct_start <- function(Yidx, K, p) {
  tab <- tabulate(Yidx, K) + 0.5
  cum <- rev(cumsum(rev(tab)))[-1] / sum(tab)
  a0 <- qlogis(cum)
  c(a0[1], pmax(-diff(a0), 1e-3), rep(0, p))
}

# Central-difference gradient, one-sided at an active lower bound so that the
# evaluation never leaves the feasible region.
.cppo_num_grad <- function(f, v, lower, h = 1e-6, ...) {
  g <- numeric(length(v))
  for (i in seq_along(v)) {
    at_bound <- is.finite(lower[i]) && v[i] <= lower[i] + h
    if (at_bound) {
      vp <- v; vp[i] <- v[i] + h
      g[i] <- (f(vp, ...) - f(v, ...)) / h
    } else {
      vp <- v; vp[i] <- v[i] + h
      vm <- v; vm[i] <- v[i] - h
      g[i] <- (f(vp, ...) - f(vm, ...)) / (2 * h)
    }
  }
  g
}

# KKT check at a returned optimum, using the analytic gradient. The gradient
# of a log-likelihood summed over n observations scales with n, so the
# tolerance does too: 1e-6 * n is 2e-4 at n = 200 and 1e-3 at n = 1000, against
# achieved residuals of ~2e-5 and ~2e-4 respectively (checked 2026-09-18).
.cppo_kkt <- function(gr, v, lower, n, tol = 1e-6 * n, ...) {
  g <- gr(v, ...)
  at_bound <- is.finite(lower) & v <= lower + 1e-10
  # Free coordinates: stationarity. Bound coordinates: no feasible descent.
  viol <- ifelse(at_bound, pmax(-g, 0), abs(g))
  list(ok = all(viol <= tol), max_violation = max(viol), gradient = g,
       active = at_bound)
}

.cppo_lbfgsb <- function(par, fn, gr, lower, ...) {
  stats::optim(
    par, fn, gr, ..., method = "L-BFGS-B", lower = lower,
    control = list(maxit = 5000, factr = 1e2, pgtol = 1e-9)
  )
}

# Reduced (PO) model. Returned first because the full model starts from it.
po_direct_fit <- function(X, Yidx, K) {
  M <- K - 1L
  p <- ncol(X)
  lower <- c(-Inf, rep(0, M - 1L), rep(-Inf, p))

  best <- NULL
  for (s0 in list(cppo_direct_start(Yidx, K, p))) {
    o <- tryCatch(.cppo_lbfgsb(s0, po_direct_nll, po_direct_grad, lower,
                               X = X, Yidx = Yidx, M = M),
                  error = function(e) NULL)
    if (!is.null(o) && is.finite(o$value) && (is.null(best) || o$value < best$value)) best <- o
  }
  if (is.null(best)) stop("po_direct_fit: every start failed")

  kkt <- .cppo_kkt(po_direct_grad, best$par, lower, n = length(Yidx),
                   X = X, Yidx = Yidx, M = M)
  alpha <- cppo_direct_cutpoints(best$par, M)

  list(
    par = best$par,
    loglik = -best$value,
    alpha = alpha,
    beta = best$par[(M + 1):(M + p)],
    slack_unexposed = min(best$par[2:M]),
    on_boundary = any(kkt$active),
    converged = best$convergence == 0L && kkt$ok,
    optim_code = best$convergence,
    kkt_max_violation = kkt$max_violation
  )
}

# Full (CPPO) model.
#
# Starts: the reduced-model solution with gamma = 0, i.e. u = s_M -- the full
# nll there EQUALS the reduced nll, so any improvement the optimiser finds
# makes the LRT statistic non-negative by construction -- plus a spread of u
# values from the data-driven start. `red` may be passed in to avoid refitting.
cppo_direct_fit <- function(X, Yidx, a_col, G, K, red = NULL) {
  M <- K - 1L
  p <- ncol(X)
  lower <- c(-Inf, rep(0, M - 1L), rep(-Inf, p), 0)

  if (is.null(red)) {
    red <- po_direct_fit(X[, -a_col, drop = FALSE], Yidx, K)
  }

  # Reduced solution lifted into the full parameterisation: beta_A = 0, u = s_M.
  red_par <- red$par
  beta_full <- numeric(p)
  beta_full[-a_col] <- red_par[(M + 1):(M + p - 1L)]
  start_red <- c(red_par[1:M], beta_full, red_par[M])

  s0 <- cppo_direct_start(Yidx, K, p)
  starts <- c(
    list(start_red),
    lapply(c(0, 0.5, 2), function(u0) c(s0, u0))
  )

  best <- NULL
  for (v0 in starts) {
    o <- tryCatch(
      .cppo_lbfgsb(v0, cppo_direct_nll, cppo_direct_grad, lower,
                   X = X, Yidx = Yidx, a_col = a_col, G = G, M = M),
      error = function(e) NULL
    )
    if (!is.null(o) && is.finite(o$value) && (is.null(best) || o$value < best$value)) best <- o
  }
  if (is.null(best)) stop("cppo_direct_fit: every start failed")

  kkt <- .cppo_kkt(cppo_direct_grad, best$par, lower, n = length(Yidx),
                   X = X, Yidx = Yidx, a_col = a_col, G = G, M = M)
  alpha <- cppo_direct_cutpoints(best$par, M)
  u <- best$par[M + p + 1]

  list(
    par = best$par,
    loglik = -best$value,
    alpha = alpha,
    beta = best$par[(M + 1):(M + p)],
    gamma = u - best$par[M],
    # Both faces, on the log-odds scale, same convention as cppo_vglm_slack().
    # Zero means the face is active. Never negative by construction.
    slack_exposed = u,
    slack_unexposed = min(best$par[2:M]),
    # Which bounds are active, exactly.
    active_exposed = kkt$active[M + p + 1],
    active_unexposed = any(kkt$active[2:M]),
    on_boundary = any(kkt$active),
    converged = best$convergence == 0L && kkt$ok,
    optim_code = best$convergence,
    kkt_max_violation = kkt$max_violation,
    reduced = red
  )
}

cppo_direct_predict <- function(fit, X, a_col, G) {
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
    # nothing to attach to: the constrained MLE puts gamma at exactly 0 (both
    # s_M and u active), the statistic equals the PO statistic, and referring
    # that 1-df quantity to chi-square_2 -- as the conventional procedure does
    # -- is conservative. Recorded, not corrected: see the policy note above.
    if (lvl_counts[K - 1L] == 0L) {
      warnings <- c(warnings,
        "cppo: departing level unobserved; gamma pinned at 0, statistic equals PO's, chi-square_2 reference is conservative")
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
    boundary <- NA
    if (fit_full$ok) {
      slack <- cppo_vglm_slack(fit_full$fit, with_A = TRUE)
      warnings <- c(warnings, cppo_boundary_tags(slack))
      # VGAM cannot land exactly on the boundary, so this is the thresholded
      # version; the direct path below reads it off the active constraints.
      boundary <- length(cppo_boundary_tags(slack)) > 0L
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
        warnings = warnings,
        boundary = boundary,
        engine = "vgam"
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

  # Reduced first; the full fit starts from it, which makes the LRT statistic
  # non-negative by construction.
  red <- tryCatch(po_direct_fit(X_red, Yidx, K = K), error = function(e) NULL)
  full <- if (is.null(red)) NULL else
    tryCatch(cppo_direct_fit(X, Yidx, a_col = 1L, G = G, K = K, red = red),
             error = function(e) NULL)

  if (is.null(full) || is.null(red)) {
    return(null_fit(c(warnings, "cppo: direct fit failed")))
  }

  # A non-converged optimiser has not established the constrained maximum, so
  # neither the likelihood nor the predictions can be trusted. This is a
  # failure of the fit, reported as such (review issue 5).
  if (!full$converged || !red$converged) {
    return(null_fit(c(warnings, sprintf(
      "cppo: direct optimiser did not converge (full: code %d, KKT %.2e; reduced: code %d, KKT %.2e)",
      full$optim_code, full$kkt_max_violation, red$optim_code, red$kkt_max_violation))))
  }

  # Boundary, read exactly off the active constraints of the full model.
  if (full$active_unexposed) {
    warnings <- c(warnings, "cppo: at the boundary -- cutpoint spacing (unexposed arm)")
  }
  if (full$active_exposed) {
    warnings <- c(warnings, "cppo: at the boundary -- deviation parameter (exposed arm)")
  }

  # (common A effect + deviation) vs none: two parameters. Referred to
  # chi-square_2 regardless of the boundary -- see the policy note at the top.
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
    warnings = warnings,
    boundary = full$on_boundary,
    engine = "direct"
  )
}
