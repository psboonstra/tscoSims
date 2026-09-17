"""
Ordinal models needed for the TsCO type I error probe, implemented directly so
that the null distribution of the LRT can be interrogated analytically as well
as by simulation.

Conventions follow the tsco package and the manuscript:

  * Outcome levels are 1..K.
  * `C` = `cutoff_level` = the FIRST level of the upper partition. The lower
    partition is {1, ..., C-1}, the upper partition is {C, ..., K}.
  * Cumulative-logit (PO) models are parameterised on the manuscript's scale,
      logit Pr(Y >= k | x) = alpha_k - x'beta,    k = 2, ..., K
    so that a POSITIVE beta shifts mass toward LOWER categories.
  * Baseline-category (MR) models are parameterised as
      log Pr(Y = k | x) / Pr(Y = 1 | x) = a_k - x'b_k
    matching the sign convention used in sim_functions/true_probs.R.

ANALYTIC GRADIENTS ARE ESSENTIAL HERE, not an optimisation nicety. The quantity
being measured -- the population KL divergence from a PO truth to the TsCO
class -- is on the order of 1e-4, and finite-difference BFGS leaves residual
slop of the same order, which would make optimiser error indistinguishable from
model misspecification. Every fitter below supplies an exact gradient, and
test_gradients.py checks each one against complex-step differentiation.

`Yoh` throughout may be either a 0/1 indicator matrix (sample log-likelihood)
or a matrix of true probabilities (population cross-entropy, whose maximiser is
the KL projection). Rows of `Yoh` must sum to 1 in both cases.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

EPS = 1e-300


# ---------------------------------------------------------------------------
# Cutpoint reparameterisation: alpha decreasing <-> unconstrained d
# ---------------------------------------------------------------------------

def d_to_alpha(d):
    return np.concatenate([[d[0]], d[0] - np.cumsum(np.exp(d[1:]))])


def alpha_to_d(alpha):
    alpha = np.asarray(alpha, float)
    gaps = -np.diff(alpha)
    if np.any(gaps <= 0):
        raise ValueError("cutpoints must be strictly decreasing")
    return np.concatenate([[alpha[0]], np.log(gaps)])


def chain_alpha_to_d(g_alpha, d):
    """Push a gradient w.r.t. alpha through to the d parameterisation."""
    g = np.empty_like(d)
    g[0] = g_alpha.sum()
    if len(d) > 1:
        # alpha_j depends on d_m (m >= 2) with derivative -exp(d_m) for j >= m
        tail = np.cumsum(g_alpha[::-1])[::-1]        # tail[m] = sum_{j >= m}
        g[1:] = -np.exp(d[1:]) * tail[1:]
    return g


# ---------------------------------------------------------------------------
# Proportional odds
# ---------------------------------------------------------------------------

def po_probs(alpha, beta, X):
    alpha = np.asarray(alpha, float)
    eta = X @ np.asarray(beta, float)
    q = expit(alpha[None, :] - eta[:, None])
    n, Km1 = q.shape
    p = np.empty((n, Km1 + 1))
    p[:, 0] = 1.0 - q[:, 0]
    if Km1 > 1:
        p[:, 1:Km1] = q[:, :-1] - q[:, 1:]
    p[:, Km1] = q[:, -1]
    return p


def po_nll_grad(v, X, Yoh, w, Km1, p_dim):
    """Negative weighted log-likelihood and its gradient, in the d/beta
    parameterisation."""
    d, beta = v[:Km1], v[Km1:]
    alpha = d_to_alpha(d)
    eta = X @ beta
    q = expit(alpha[None, :] - eta[:, None])                  # (n, K-1)
    s = q * (1.0 - q)                                         # dq/dalpha
    p = np.empty((X.shape[0], Km1 + 1))
    p[:, 0] = 1.0 - q[:, 0]
    if Km1 > 1:
        p[:, 1:Km1] = q[:, :-1] - q[:, 1:]
    p[:, Km1] = q[:, -1]
    p = np.clip(p, 1e-300, None)

    nll = -float(np.sum(w[:, None] * Yoh * np.log(p)))

    # R[:, j] = d loglik / d q_j  (unweighted, per row)
    ratio = Yoh / p                                           # (n, K)
    R = -ratio[:, :Km1] + ratio[:, 1:]                        # (n, K-1)
    sR = s * R
    g_alpha = (w[:, None] * sR).sum(0)
    g_beta = -(X * (w * sR.sum(1))[:, None]).sum(0)
    g = np.concatenate([chain_alpha_to_d(g_alpha, d), g_beta])
    return nll, -g


# ---------------------------------------------------------------------------
# Baseline-category logit (multinomial regression)
# ---------------------------------------------------------------------------

def mr_probs(a, B, X):
    a = np.asarray(a, float)
    B = np.atleast_2d(np.asarray(B, float))
    lin = a[None, :] - X @ B.T
    lin = np.concatenate([np.zeros((X.shape[0], 1)), lin], axis=1)
    return np.exp(lin - logsumexp(lin, axis=1, keepdims=True))


def mr_nll_grad(v, X, Yoh, w, Km1, p_dim):
    a = v[:Km1]
    B = v[Km1:].reshape(Km1, p_dim)
    p = mr_probs(a, B, X)
    nll = -float(np.sum(w[:, None] * Yoh * np.log(np.clip(p, 1e-300, None))))
    resid = w[:, None] * (Yoh - p)                            # (n, K)
    g_a = resid[:, 1:].sum(0)
    g_B = -(resid[:, 1:].T @ X)
    return nll, -np.concatenate([g_a, g_B.ravel()])


# ---------------------------------------------------------------------------
# Constrained partial proportional odds
# ---------------------------------------------------------------------------

def cppo_probs(alpha, beta_common, gamma, G, X, dev_col):
    alpha = np.asarray(alpha, float)
    G = np.asarray(G, float)
    eta = (X @ np.asarray(beta_common, float))[:, None] \
        + (gamma * X[:, dev_col])[:, None] * G[None, :]
    q = expit(alpha[None, :] - eta)
    n, Km1 = q.shape
    p = np.empty((n, Km1 + 1))
    p[:, 0] = 1.0 - q[:, 0]
    if Km1 > 1:
        p[:, 1:Km1] = q[:, :-1] - q[:, 1:]
    p[:, Km1] = q[:, -1]
    return p


def cppo_nll_grad(v, X, Yoh, w, Km1, p_dim, G, dev_col):
    d, beta, gamma = v[:Km1], v[Km1:Km1 + p_dim], v[Km1 + p_dim]
    alpha = d_to_alpha(d)
    G = np.asarray(G, float)
    eta = (X @ beta)[:, None] + (gamma * X[:, dev_col])[:, None] * G[None, :]
    q = expit(alpha[None, :] - eta)
    s = q * (1.0 - q)
    n = X.shape[0]
    p = np.empty((n, Km1 + 1))
    p[:, 0] = 1.0 - q[:, 0]
    if Km1 > 1:
        p[:, 1:Km1] = q[:, :-1] - q[:, 1:]
    p[:, Km1] = q[:, -1]
    p = np.clip(p, 1e-300, None)
    nll = -float(np.sum(w[:, None] * Yoh * np.log(p)))
    ratio = Yoh / p
    R = -ratio[:, :Km1] + ratio[:, 1:]
    sR = s * R
    g_alpha = (w[:, None] * sR).sum(0)
    g_beta = -(X * (w * sR.sum(1))[:, None]).sum(0)
    g_gamma = -float(np.sum(w * X[:, dev_col] * (sR @ G)))
    g = np.concatenate([chain_alpha_to_d(g_alpha, d), g_beta, [g_gamma]])
    return nll, -g


# ---------------------------------------------------------------------------
# Two-stage conditional odds
# ---------------------------------------------------------------------------

def tsco_probs(theta1, theta2, X, K, C, stage2):
    n_low = C - 1
    alpha1, beta1 = theta1
    p_low = po_probs(alpha1, beta1, X)
    if stage2 == "mr":
        p_up = mr_probs(theta2[0], theta2[1], X)
    elif stage2 == "po":
        p_up = po_probs(theta2[0], theta2[1], X)
    else:
        raise ValueError(stage2)
    out = np.empty((X.shape[0], K))
    out[:, :n_low] = p_low * p_up[:, [0]]
    out[:, n_low:] = p_up[:, 1:]
    return out


def tsco_split(Yoh, w, K, C):
    """Split the outcome into the two stages' separate estimation problems.

    The TsCO log-likelihood is exactly l1(theta1) + l2(theta2) with
    variation-independent parameters, so the two stages are fitted separately.
    Stage 1 sees the outcome CONDITIONAL on Y < C, with each row reweighted by
    its probability of landing in the lower partition; stage 2 sees the
    collapsed outcome on the original weights.
    """
    n_low = C - 1
    low = Yoh[:, :n_low]
    tot_low = low.sum(1)
    w1 = w * tot_low
    with np.errstate(invalid="ignore", divide="ignore"):
        Y1 = np.where(tot_low[:, None] > 0, low / np.where(tot_low[:, None] > 0,
                                                           tot_low[:, None], 1.0), 0.0)
    Y2 = np.concatenate([tot_low[:, None], Yoh[:, n_low:]], axis=1)
    return (Y1, w1), (Y2, w)


# ---------------------------------------------------------------------------
# Generic fitting driver
# ---------------------------------------------------------------------------

def _fit(fun, x0, args, gtol=1e-12):
    res = minimize(fun, x0, args=args, jac=True, method="BFGS",
                   options={"maxiter": 5000, "gtol": gtol})
    # One restart from the solution: BFGS occasionally stops on a stale
    # curvature estimate, and here a loose optimum would masquerade as
    # model misspecification.
    res2 = minimize(fun, res.x, args=args, jac=True, method="BFGS",
                    options={"maxiter": 5000, "gtol": gtol})
    return res2 if res2.fun <= res.fun else res


def _po_start(Yoh, w, p_dim):
    marg = np.clip((w[:, None] * Yoh).sum(0) / max(w.sum(), 1e-12), 1e-6, None)
    marg = marg / marg.sum()
    cum = np.clip(np.cumsum(marg[::-1])[::-1][1:], 1e-5, 1 - 1e-5)
    return np.concatenate([alpha_to_d(np.log(cum / (1 - cum))), np.zeros(p_dim)])


def fit_po(X, Yoh, w, K):
    p_dim = X.shape[1]
    res = _fit(po_nll_grad, _po_start(Yoh, w, p_dim), (X, Yoh, w, K - 1, p_dim))
    d, beta = res.x[:K - 1], res.x[K - 1:]
    return -res.fun, (d_to_alpha(d), beta), res


def fit_mr(X, Yoh, w, K):
    p_dim = X.shape[1]
    marg = np.clip((w[:, None] * Yoh).sum(0) / max(w.sum(), 1e-12), 1e-6, None)
    x0 = np.concatenate([np.log(marg[1:] / marg[0]), np.zeros((K - 1) * p_dim)])
    res = _fit(mr_nll_grad, x0, (X, Yoh, w, K - 1, p_dim))
    return -res.fun, (res.x[:K - 1], res.x[K - 1:].reshape(K - 1, p_dim)), res


def _cppo_nll_reparam(v, X, Yoh, w, Km1, p_dim, G, dev_col):
    """CPPO objective with the parameter space enforced BY CONSTRUCTION.

    CRITICAL, and the reason the unconstrained version below is wrong: a CPPO
    fit must keep the cumulative probabilities ordered. With the departure at
    the last cutpoint, that means gamma >= alpha_{K-1} - alpha_{K-2} for the
    exposed rows (unexposed rows are automatic). An unconstrained optimiser
    happily violates this, which gives some category a NEGATIVE fitted
    probability -- and when that category is EMPTY in the exposed arm, the
    invalid value is never evaluated against an observation, so clipping the
    log never bites. The fit then buys likelihood for free by assigning
    negative mass to an unoccupied cell. Measured consequence at n = 200 under
    the ECMO baseline: gamma ran to -7000, the LRT for a null exposure hit a
    median of 110, and the type I error read 0.107 instead of 0.063.

    Reparameterising gamma = (alpha_{K-1} - alpha_{K-2}) + exp(t) makes the
    constraint unviolatable. The MLE then sits ON the boundary when the cell is
    empty (~6% of replicates here) -- which is a genuine boundary problem, so
    the usual chi-square reference is still not right there, but the likelihood
    is at least a real one.
    """
    d, beta, t = v[:Km1], v[Km1:Km1 + p_dim], v[Km1 + p_dim]
    alpha = d_to_alpha(d)
    gamma = (alpha[-1] - alpha[-2]) + np.exp(np.clip(t, -50, 50))
    G = np.asarray(G, float)
    eta = (X @ beta)[:, None] + (gamma * X[:, dev_col])[:, None] * G[None, :]
    q = expit(alpha[None, :] - eta)
    n = X.shape[0]
    p = np.empty((n, Km1 + 1))
    p[:, 0] = 1.0 - q[:, 0]
    if Km1 > 1:
        p[:, 1:Km1] = q[:, :-1] - q[:, 1:]
    p[:, Km1] = q[:, -1]
    if np.any(p < -1e-12):
        return 1e8
    return -float(np.sum(w[:, None] * Yoh * np.log(np.clip(p, 1e-300, None))))


def fit_cppo(X, Yoh, w, K, G, dev_col, constrained=True):
    """Fit CPPO. `constrained=True` (the default, and the only correct choice)
    keeps the fit inside the CPPO parameter space; see _cppo_nll_reparam."""
    p_dim = X.shape[1]
    if not constrained:
        x0 = np.concatenate([_po_start(Yoh, w, p_dim), [0.0]])
        res = _fit(cppo_nll_grad, x0, (X, Yoh, w, K - 1, p_dim, G, dev_col))
        d, beta, gamma = res.x[:K - 1], res.x[K - 1:K - 1 + p_dim], res.x[-1]
        return -res.fun, (d_to_alpha(d), beta, gamma), res

    base = _po_start(Yoh, w, p_dim)
    best = None
    for t0 in (-3.0, -1.0, 0.5):
        r = minimize(_cppo_nll_reparam, np.concatenate([base, [t0]]),
                     args=(X, Yoh, w, K - 1, p_dim, G, dev_col),
                     method="BFGS", options={"maxiter": 2000, "gtol": 1e-8})
        if best is None or r.fun < best.fun:
            best = r
    alpha = d_to_alpha(best.x[:K - 1])
    beta = best.x[K - 1:K - 1 + p_dim]
    gamma = (alpha[-1] - alpha[-2]) + np.exp(np.clip(best.x[-1], -50, 50))
    return -best.fun, (alpha, beta, float(gamma)), best


def fit_tsco(X, Yoh, w, K, C, stage2):
    """Returns (loglik, (theta1, theta2), (res1, res2))."""
    n_low, n_up = C - 1, K - C + 1
    (Y1, w1), (Y2, w2) = tsco_split(Yoh, w, K, C)

    ll1, th1, res1 = fit_po(X, Y1, w1, n_low)
    if stage2 == "mr":
        ll2, th2, res2 = fit_mr(X, Y2, w2, n_up + 1)
    elif stage2 == "po":
        ll2, th2, res2 = fit_po(X, Y2, w2, n_up + 1)
    else:
        raise ValueError(stage2)
    return ll1 + ll2, (th1, th2), (res1, res2)


def fit_tsco_null(X, Yoh, w, K, C, stage2, z_col):
    """TsCO with every coefficient on column `z_col` constrained to zero."""
    keep = [j for j in range(X.shape[1]) if j != z_col]
    return fit_tsco(X[:, keep], Yoh, w, K, C, stage2)


def onehot(y, K):
    out = np.zeros((len(y), K))
    out[np.arange(len(y)), y - 1] = 1.0
    return out


# ---------------------------------------------------------------------------
# Combined TsCO objective over the stacked parameter vector.
#
# The likelihood separates exactly, so this is only a convenience for building
# one Hessian and one score matrix across both stages; fit_tsco still solves
# the two stages separately because they are better conditioned apart.
# ---------------------------------------------------------------------------

def tsco_dims(K, C, p_dim, stage2):
    n_low, n_up = C - 1, K - C + 1
    n1 = (n_low - 1) + p_dim
    n2 = (n_up * (p_dim + 1)) if stage2 == "mr" else (n_up + p_dim)
    return n1, n2


def tsco_nll_grad(v, X, Yoh, w, K, C, stage2):
    p_dim = X.shape[1]
    n_low, n_up = C - 1, K - C + 1
    n1, _ = tsco_dims(K, C, p_dim, stage2)
    (Y1, w1), (Y2, w2) = tsco_split(Yoh, w, K, C)
    f1, g1 = po_nll_grad(v[:n1], X, Y1, w1, n_low - 1, p_dim)
    if stage2 == "mr":
        f2, g2 = mr_nll_grad(v[n1:], X, Y2, w2, n_up, p_dim)
    else:
        f2, g2 = po_nll_grad(v[n1:], X, Y2, w2, n_up, p_dim)
    return f1 + f2, np.concatenate([g1, g2])


def tsco_pack(theta1, theta2, stage2):
    alpha1, beta1 = theta1
    v1 = np.concatenate([alpha_to_d(alpha1), beta1])
    if stage2 == "mr":
        v2 = np.concatenate([theta2[0], np.asarray(theta2[1]).ravel()])
    else:
        v2 = np.concatenate([alpha_to_d(theta2[0]), theta2[1]])
    return np.concatenate([v1, v2])


def tsco_z_index(K, C, p_dim, stage2, z_col):
    """Positions in the stacked vector of every coefficient on covariate
    `z_col` -- the parameters the joint test sets to zero."""
    n_low, n_up = C - 1, K - C + 1
    n1, _ = tsco_dims(K, C, p_dim, stage2)
    idx = [(n_low - 1) + z_col]                      # stage-1 slope on Z
    if stage2 == "mr":
        idx += [n1 + n_up + k * p_dim + z_col for k in range(n_up)]
    else:
        idx += [n1 + n_up + z_col]                   # stage-2 PO slope on Z
    return np.array(idx)
