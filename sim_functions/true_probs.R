# True category probabilities under each data-generating mechanism, plus the
# small helpers used to build and validate them.
#
# DESIGN PRINCIPLE (2026-09-14)
# ------------------------------
# Every DGM shares one BASELINE distribution: the category probabilities at
# A = 0, W = 0. Previously each DGM carried its own invented intercepts
# (PO used c(1.4, 0.5, -0.4, -1.3), CPPO used c(1.6, 0.7, -0.2, -1.4), and the
# TsCO DGM was parameterised separately again), so a difference between two
# scenarios confounded the structure of the treatment effect with a different
# reference distribution. Now the intercepts of every DGM are DERIVED from a
# single baseline vector, so scenarios differ only in how the treatment effect
# propagates away from a common starting point.
#
# The baseline is the manuscript's pediatric ECMO example: the six-level
# standard-of-care distribution {0.27, 0.06, 0.06, 0.11, 0.03, 0.47} for levels
# 0 to 5. Outcome levels are therefore labelled "0" .. "5", matching the paper,
# so that `cutoff_level = "5"` here is literally the paper's P O|5|M R.
#
# Parameterisations are taken verbatim from the manuscript and reproduce its
# Tables 3 and 4 to three decimal places (see check_dgm.R):
#
#   PO     Pr(Y >= k | Z)      = expit(a_k - b1 z)
#   CPPO   Pr(Y >= k | Z)      = expit(a_k - b1 z - 1[k = K-1] b2 z)
#   TsCO   Pr(Y >= k | Y < C)  = expit(a^(T)_k - b1 z)      (stage 1, PO)
#          collapsed outcome                                (stage 2, MR or PO)
#
# THE NUISANCE COVARIATE W. The manuscript's design has a single binary Z and
# no W, which is what makes MR saturated there. This simulation adds a
# continuous W, so MR is no longer saturated and the nesting lattice is not the
# paper's. W enters each DGM through that DGM's OWN linear predictors, so each
# scenario makes its own model exactly true; it therefore does not induce a
# common marginal W effect across scenarios, and the marginal (W-averaged)
# distribution is NOT the baseline. Setting BETA_W <- 0 recovers the
# manuscript's design exactly.

## ---------------------------------------------------------------------------
## Shared design constants
## ---------------------------------------------------------------------------

# Levels 0..5, as in the manuscript.
BASELINE_LEVELS <- as.character(0:5)

# Standard-of-care distribution in the pediatric ECMO trial (manuscript Tables
# 1-5). Positions correspond to levels 0..5.
BASELINE_PROBS <- c(0.27, 0.06, 0.06, 0.11, 0.03, 0.47)

# Coefficient on the continuous nuisance covariate W, used in whichever linear
# predictors the DGM has. Set to 0 to recover the manuscript's Z-only design.
BETA_W <- 0.5

stopifnot(
  length(BASELINE_PROBS) == length(BASELINE_LEVELS),
  abs(sum(BASELINE_PROBS) - 1) < 1e-12,
  all(BASELINE_PROBS > 0)
)


## ---------------------------------------------------------------------------
## Small helpers
## ---------------------------------------------------------------------------

expit <- function(x) {
  plogis(x)
}

# Pr(Y >= k) for k = 2nd level .. last level, given a probability vector.
cum_upper <- function(p) {
  rev(cumsum(rev(p)))[-1]
}

upper_cumul_to_prob <- function(q) {
  # q has columns Pr(Y >= 2nd level), ..., Pr(Y >= last level)
  q <- as.matrix(q)
  n <- nrow(q)
  K_minus_1 <- ncol(q)
  K <- K_minus_1 + 1L

  p <- matrix(NA_real_, nrow = n, ncol = K)

  p[, 1L] <- 1 - q[, 1L]

  if (K > 2L) {
    for (k in 2:(K - 1L)) {
      p[, k] <- q[, k - 1L] - q[, k]
    }
  }

  p[, K] <- q[, K_minus_1]

  p
}

# Cutpoints of a PO model whose implied distribution at a zero linear predictor
# is exactly `p`. This is the single place a baseline is turned into intercepts.
baseline_cutpoints <- function(p = BASELINE_PROBS) {
  stats::qlogis(cum_upper(p))
}

check_prob_matrix <- function(p, tol = 1e-6) {
  if (any(!is.finite(p))) {
    stop("Non-finite probabilities.")
  }

  if (any(p < -tol)) {
    stop("Negative probabilities.")
  }

  rs <- rowSums(p)

  if (any(abs(rs - 1) > tol)) {
    stop("Probabilities do not sum to 1.")
  }

  p[p < 0] <- 0
  p / rowSums(p)
}

draw_ordinal <- function(p, levels_y) {
  p <- check_prob_matrix(p)

  u <- runif(nrow(p))
  cp <- t(apply(p, 1, cumsum))

  y_idx <- max.col(u <= cp, ties.method = "first")

  ordered(levels_y[y_idx], levels = levels_y)
}

make_covariates <- function(n) {
  data.frame(
    A = stats::rbinom(n, 1, 0.5),
    W = stats::rnorm(n)
  )
}

# The CPPO departure contrast: the manuscript's ECMO model allows
# non-proportionality at the HIGHEST cutpoint only,
#   Pr(Y >= k | Z) = expit(a_k - z b1 - z 1[k = K-1] b2),
# so G is the indicator of the last cutpoint. Derived from the number of levels
# rather than hard-coded, so that changing K cannot silently leave a
# wrong-length G behind.
cppo_G <- function(levels_y = BASELINE_LEVELS) {
  M <- length(levels_y) - 1L
  G <- rep(0, M)
  G[M] <- 1
  G
}


## ---------------------------------------------------------------------------
## Data-generating mechanisms
##
## Each takes the treatment effect parameters and returns an n x K matrix of
## true probabilities. Each satisfies: at A = 0 and W = 0 the row equals
## `baseline` exactly.
## ---------------------------------------------------------------------------

true_prob_po <- function(dat,
                         b1,
                         beta_W = BETA_W,
                         baseline = BASELINE_PROBS) {

  alpha <- baseline_cutpoints(baseline)

  eta <- b1 * dat$A + beta_W * dat$W

  q <- sapply(alpha, function(a) expit(a - eta))
  q <- matrix(q, nrow = nrow(dat))

  p <- upper_cumul_to_prob(q)
  colnames(p) <- BASELINE_LEVELS

  check_prob_matrix(p)
}


true_prob_cppo <- function(dat,
                           b1,
                           b2,
                           beta_W = BETA_W,
                           baseline = BASELINE_PROBS,
                           G = cppo_G()) {

  # Same cutpoints as PO: the departure term is multiplied by A, so it
  # vanishes at baseline.
  alpha <- baseline_cutpoints(baseline)

  q <- matrix(NA_real_, nrow = nrow(dat), ncol = length(alpha))

  for (j in seq_along(alpha)) {
    eta_j <- beta_W * dat$W + dat$A * (b1 + b2 * G[j])
    q[, j] <- expit(alpha[j] - eta_j)
  }

  # The manuscript notes that b2 is bounded below (approximately -0.12 for this
  # baseline): beyond it the cumulative probabilities cross and the second-to-
  # highest category would take negative probability. Check rather than fix.
  if (any(q[, -1, drop = FALSE] > q[, -ncol(q), drop = FALSE] + 1e-10)) {
    stop("CPPO DGM produced non-monotone cumulative probabilities; ",
         "b2 = ", b2, " is likely past its lower bound for this baseline.")
  }

  p <- upper_cumul_to_prob(q)
  colnames(p) <- BASELINE_LEVELS

  check_prob_matrix(p)
}


# Two-stage conditional odds.
#
# `cutoff_level` is the FIRST level of the upper partition, matching the tsco
# package: the lower partition is the levels before it, the upper partition is
# that level and everything above.
#
# Stage 1 is a PO model for the outcome conditional on being in the lower
# partition. Stage 2 is a model for the collapsed outcome
# {lower, C, C+1, ..., K}, either baseline-category MR (reference = lower) or
# PO. Both stages' intercepts are derived from `baseline`, so the DGM sits at
# `baseline` when A = W = 0.
#
# `b2` is a vector with one entry per upper-partition level for stage2 = "mr",
# and a scalar for stage2 = "po". When the upper partition holds a single level
# -- the manuscript's P O|5|M R -- the two stage-2 families coincide and b2 is
# a scalar either way.
true_prob_tsco <- function(dat,
                           b1,
                           b2,
                           cutoff_level = "5",
                           stage2 = c("mr", "po"),
                           beta1_W = BETA_W,
                           beta2_W = BETA_W,
                           baseline = BASELINE_PROBS,
                           levels_y = BASELINE_LEVELS) {

  stage2 <- match.arg(stage2)

  cutoff_index <- match(as.character(cutoff_level), levels_y)
  if (is.na(cutoff_index)) {
    stop("`cutoff_level` must be one of ", paste(levels_y, collapse = ", "))
  }

  K <- length(levels_y)
  n_low <- cutoff_index - 1L
  n_up <- K - cutoff_index + 1L

  if (n_low < 2L) stop("Stage 1 needs at least two levels below `cutoff_level`.")

  n <- nrow(dat)

  ## Stage 1: distribution over the lower partition, conditional on being there
  low_base <- baseline[seq_len(n_low)]
  low_base <- low_base / sum(low_base)

  alpha1 <- stats::qlogis(cum_upper(low_base))

  eta1 <- b1 * dat$A + beta1_W * dat$W

  q1 <- sapply(alpha1, function(a) expit(a - eta1))
  q1 <- matrix(q1, nrow = n)
  p_low <- upper_cumul_to_prob(q1)

  ## Stage 2: distribution over {lower, C, ..., K}
  collapsed_base <- c(sum(baseline[seq_len(n_low)]),
                      baseline[(n_low + 1L):K])

  if (stage2 == "mr") {
    if (length(b2) != n_up) {
      stop("For stage2 = 'mr', `b2` needs one entry per upper-partition level (",
           n_up, ").")
    }

    # Baseline-category logits with the collapsed lower partition as reference,
    # on the manuscript's scale: log Pr(k)/Pr(ref) = alpha_k - x'b.
    alpha2 <- log(collapsed_base[-1] / collapsed_base[1])

    odds <- matrix(NA_real_, nrow = n, ncol = n_up)
    for (j in seq_len(n_up)) {
      odds[, j] <- exp(alpha2[j] - b2[j] * dat$A - beta2_W * dat$W)
    }

    denom <- 1 + rowSums(odds)
    p_up <- cbind(1 / denom, odds / denom)

  } else {
    if (length(b2) != 1L) {
      stop("For stage2 = 'po', `b2` must be a scalar.")
    }

    alpha2 <- stats::qlogis(cum_upper(collapsed_base))

    eta2 <- b2 * dat$A + beta2_W * dat$W

    q2 <- sapply(alpha2, function(a) expit(a - eta2))
    q2 <- matrix(q2, nrow = n)
    p_up <- upper_cumul_to_prob(q2)
  }

  p <- matrix(0, nrow = n, ncol = K)
  colnames(p) <- levels_y

  p[, seq_len(n_low)] <- p_low * p_up[, 1]
  p[, (n_low + 1L):K] <- p_up[, -1, drop = FALSE]

  check_prob_matrix(p)
}


# A DGM specified directly by its treatment-arm probability vector, for
# scenarios where no model is true (manuscript Table 5, label 2). W enters as a
# proportional shift of the arm-specific cumulative logits, so at W = 0 the two
# arms are exactly the supplied vectors.
true_prob_vectors <- function(dat,
                              p_control = BASELINE_PROBS,
                              p_treat,
                              beta_W = BETA_W) {

  a_ctrl <- baseline_cutpoints(p_control)
  a_trt <- baseline_cutpoints(p_treat)

  alpha <- outer(rep(1, nrow(dat)), a_ctrl)
  trt <- which(dat$A == 1)
  if (length(trt) > 0L) {
    alpha[trt, ] <- outer(rep(1, length(trt)), a_trt)
  }

  q <- expit(alpha - beta_W * dat$W)

  p <- upper_cumul_to_prob(q)
  colnames(p) <- BASELINE_LEVELS

  check_prob_matrix(p)
}


## ---------------------------------------------------------------------------
## Scenarios
##
## Effect sizes are the manuscript's Table 5 values; the label in each comment
## is that table's row. Every scenario shares BASELINE_PROBS at A = W = 0.
## ---------------------------------------------------------------------------

true_prob_scenario <- function(dat, scenario, cutoff_level = "5") {
  switch(
    scenario,

    # Table 5 row 1: no treatment effect. Every model is correctly specified
    # with respect to treatment, so this is the type I error scenario.
    null = true_prob_po(dat, b1 = 0),

    # Table 5 row 6: PO with LOR 0.5. PO true, hence CPPO and TsCO also
    # satisfied with respect to treatment.
    po_alt = true_prob_po(dat, b1 = 0.5),

    # Table 5 row 5: CPPO {b1, b2} = {0.5, -0.1}. Neither PO nor TsCO true.
    cppo_alt = true_prob_cppo(dat, b1 = 0.5, b2 = -0.1),

    # Table 5 row 3: TsCO {b1, b2} = {0.5, 0.15}. Neither PO nor CPPO true.
    tsco_alt = true_prob_tsco(dat, b1 = 0.5, b2 = 0.15,
                              cutoff_level = cutoff_level, stage2 = "po"),

    # Table 5 row 4: TsCO {b1, b2} = {0, 0.25}. A pure stage-2 effect, which a
    # PO analysis has no way to represent.
    tsco_alt_stage2 = true_prob_tsco(dat, b1 = 0, b2 = 0.25,
                                     cutoff_level = cutoff_level, stage2 = "po"),

    # Table 5 row 2: treatment distribution given directly; no model is true.
    none_true = true_prob_vectors(
      dat,
      p_treat = c(0.290, 0.110, 0.050, 0.090, 0.020, 0.440)
    ),

    stop("Unknown scenario: ", scenario)
  )
}
