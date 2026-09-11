# True category probabilities under each data-generating mechanism, plus the
# small helpers used to build and validate them.

expit <- function(x) {
  plogis(x)
}

upper_cumul_to_prob <- function(q) {
  # q has columns Pr(Y >= 2), ..., Pr(Y >= K)
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

true_prob_po <- function(dat, beta_A, beta_W = 0.5) {
  alpha <- c(1.4, 0.5, -0.4, -1.3) # for Y >= 2,3,4,5

  eta <- beta_A * dat$A + beta_W * dat$W

  q <- sapply(alpha, function(a) expit(a - eta))
  q <- as.matrix(q)

  p <- upper_cumul_to_prob(q)
  colnames(p) <- as.character(1:5)

  check_prob_matrix(p)
}

true_prob_cppo <- function(dat, beta_A, gamma_A, beta_W = 0.5) {
  alpha <- c(1.6, 0.7, -0.2, -1.4)
  G <- c(0, 0, 0, 1)

  q <- matrix(NA_real_, nrow = nrow(dat), ncol = length(alpha))

  for (j in seq_along(alpha)) {
    eta_j <- beta_W * dat$W + dat$A * (beta_A + gamma_A * G[j])
    q[, j] <- expit(alpha[j] - eta_j)
  }

  # Ensure monotonic cumulative probabilities.
  # With chosen parameters this should hold; check rather than silently fix.
  if (any(q[, -1, drop = FALSE] > q[, -ncol(q), drop = FALSE] + 1e-10)) {
    stop("CPPO DGP produced non-monotone cumulative probabilities.")
  }

  p <- upper_cumul_to_prob(q)
  colnames(p) <- as.character(1:5)

  check_prob_matrix(p)
}

true_prob_tsco_pomr <- function(
    dat,
    beta1_A,
    beta2_A_4,
    beta2_A_5,
    beta1_W = 0.4,
    beta2_W_4 = 0.2,
    beta2_W_5 = 0.6) {

  n <- nrow(dat)

  # Stage 1: conditional among 1,2,3.
  # P(Y_lower >= 2), P(Y_lower >= 3)
  alpha1 <- c(0.8, -0.6)

  eta1 <- beta1_A * dat$A + beta1_W * dat$W

  q1 <- sapply(alpha1, function(a) expit(a - eta1))
  p1 <- upper_cumul_to_prob(q1)
  colnames(p1) <- c("1", "2", "3")

  # Stage 2: MR over collapsed categories: lt4, 4, 5.
  # Manuscript convention: Pr(cat = k) ~ exp(alpha_k - x beta_k)
  alpha4 <- -0.4
  alpha5 <- -1.2

  odds4 <- exp(alpha4 - beta2_A_4 * dat$A - beta2_W_4 * dat$W)
  odds5 <- exp(alpha5 - beta2_A_5 * dat$A - beta2_W_5 * dat$W)

  denom <- 1 + odds4 + odds5

  p_lt4 <- 1 / denom
  p4 <- odds4 / denom
  p5 <- odds5 / denom

  p <- matrix(0, nrow = n, ncol = 5)
  colnames(p) <- as.character(1:5)

  p[, c("1", "2", "3")] <- p1 * as.numeric(p_lt4)
  p[, "4"] <- p4
  p[, "5"] <- p5

  check_prob_matrix(p)
}

true_prob_scenario <- function(dat, scenario) {
  switch(
    scenario,

    null_po = true_prob_po(
      dat,
      beta_A = 0,
      beta_W = 0.5
    ),

    null_tsco = true_prob_tsco_pomr(
      dat,
      beta1_A = 0,
      beta2_A_4 = 0,
      beta2_A_5 = 0
    ),

    po_alt = true_prob_po(
      dat,
      beta_A = 0.5,
      beta_W = 0.5
    ),

    cppo_alt = true_prob_cppo(
      dat,
      beta_A = 0.3,
      gamma_A = 0.7,
      beta_W = 0.5
    ),

    tsco_pomr_alt = true_prob_tsco_pomr(
      dat,
      beta1_A = 0.4,
      beta2_A_4 = 0.2,
      beta2_A_5 = 0.9
    ),

    stop("Unknown scenario: ", scenario)
  )
}
