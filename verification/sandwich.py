"""
The asymptotic null distribution of the joint LRT for Z, allowing for
misspecification of the fitted model.

THEORY. Let theta = (psi, lambda) with psi the coefficients on Z that the joint
test sets to zero, and let theta* be the population KL projection of the truth
onto the fitted class. Write

    H = -E[ d2 log q ],    J = Var( d log q ),   both at theta*.

Under misspecification the LRT of H0: psi = psi* is NOT chi-square. By White
(1982) / Foutz-Srivastava (1977),

    2( l(theta_hat) - l(theta_hat_0) )  ->  sum_j  lambda_j * chi2_1

where the lambda_j are the eigenvalues of

    H_{psi psi . lambda} * V_{psi psi},
    H_{psi psi . lambda} = H_pp - H_pl H_ll^-1 H_lp     (effective information)
    V = H^-1 J H^-1                                      (sandwich)

Under correct specification J = H, so V_pp = (H_{pp.l})^-1 and every
lambda_j = 1, recovering chi2_q. So the test is valid iff the information
equality holds AFTER PROFILING OUT the nuisance parameters, restricted to the
psi block -- not the full information equality.

TWO PREREQUISITES, both checked in this module rather than assumed:

  (a) psi* = 0, i.e. the KL projection really does put zero on Z. If it did not
      the LRT would not even be testing the right hypothesis. Under the null Z
      is independent of (Y, W) and the model class is invariant under
      Z -> 1 - Z (which maps psi -> -psi and absorbs the shift into the
      intercepts), so a unique minimiser must be a fixed point of that
      involution and hence have psi* = 0.

  (b) The eigenvalues are invariant to reparameterisation. The cutpoint
      transform used for optimisation touches only intercepts (nuisance), never
      the Z slopes, so it cannot move them.

A HAND DERIVATION worth recording, because it predicts the answer and this
module then checks it numerically. At psi = 0 the score for psi is
z * g(y, w), and for every model used here g lies in the SPAN OF THE INTERCEPT
SCORES (for stage-2 MR, g is exactly the intercept score vector; for a PO
stage, g is minus the sum of the intercept scores, since a slope shifts all
cutpoints together). Writing g = P'h and using Z independent of (Y,W) with
E[Z] = E[Z^2] = 1/2, the blocks collapse and

    H_{pp.l} = (1/4) P'C P,        V_pp = 4 (P'C P)^-1 (P'N P) (P'C P)^-1

with C the Hessian and N the outer-product information for lambda. Hence

    lambda_j = eig( (P'N P) (P'C P)^-1 ).

So the weights are 1 exactly when the information equality holds in the
INTERCEPT DIRECTIONS. Expanding N - C on those directions and using the
intercept first-order condition E_W[p(W) - pi(W)] = 0 leaves

    N - C  =  -2 * Sym{ Cov_W( p(W) - pi(W), pi(W) ) },

which vanishes only if the projection residual is uncorrelated across W with
the fitted probabilities. The slope FOC kills the correlation with W itself,
not with the nonlinear pi(W), so this is generically nonzero.
"""

import numpy as np

import models as M


def hessian(fun, x, args, h=1e-5):
    """Hessian by central differences of the ANALYTIC gradient."""
    d = len(x)
    H = np.zeros((d, d))
    for i in range(d):
        e = np.zeros(d); e[i] = h
        gp = fun(x + e, *args)[1]
        gm = fun(x - e, *args)[1]
        H[i] = (gp - gm) / (2 * h)
    return 0.5 * (H + H.T)


def score_matrix(fun, x, X, w, K, extra):
    """Per-(row, category) score of the log-likelihood, shape (n, K, d).

    s[i, k] = d/dtheta log q_theta(Y = k | x_i), evaluated one row at a time.

    These scores contain 1/q factors and so diverge wherever the FITTED model
    assigns a category essentially zero probability -- which happens in the far
    quadrature tails once the nuisance effect is large. Combined with a true
    probability that has underflowed to exactly zero this produces 0 * inf, and
    the resulting NaN silently corrupts J. The caller masks on Ptrue; here the
    scores are additionally left finite so the mask can do its job.
    """
    n, d = X.shape[0], len(x)
    S = np.zeros((n, K, d))
    one = np.ones(1)
    for i in range(n):
        Xi = X[i:i + 1]
        for k in range(K):
            Y = np.zeros((1, K)); Y[0, k] = 1.0
            S[i, k] = -fun(x, Xi, Y, one, *extra)[1]   # grad of +loglik
    return S


def sandwich_eigs(fun, x_star, X, Ptrue, wt, K, extra, psi_idx):
    """Return (eigenvalues, diagnostics) for the weighted-chi-square null."""
    d = len(x_star)
    H = hessian(fun, x_star, (X, Ptrue, wt) + extra)

    S = score_matrix(fun, x_star, X, wt, K, extra)

    # Terms where the truth puts no mass contribute nothing to either
    # information matrix, but would contribute NaN if a divergent score were
    # multiplied by a zero probability. Drop them explicitly.
    Pm = np.where(np.isfinite(S).all(axis=2), Ptrue, 0.0)
    Pm = np.where(Pm > 1e-200, Pm, 0.0)
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

    dropped = float(Ptrue.sum() - Pm.sum()) / float(Ptrue.sum())

    # J = E[ s s' ] under the TRUTH; mean score is zero at the projection.
    J = np.einsum("i,ik,ikd,ike->de", wt, Pm, S, S)
    mean_score = np.einsum("i,ik,ikd->d", wt, Pm, S)

    lam_idx = np.array([j for j in range(d) if j not in set(psi_idx)])
    pp, ll_, pl = np.ix_(psi_idx, psi_idx), np.ix_(lam_idx, lam_idx), np.ix_(psi_idx, lam_idx)

    Hpp, Hll, Hpl = H[pp], H[ll_], H[pl]
    Heff = Hpp - Hpl @ np.linalg.solve(Hll, Hpl.T)      # effective information

    Hinv = np.linalg.inv(H)
    V = Hinv @ J @ Hinv
    Vpp = V[pp]

    eigs = np.linalg.eigvals(Heff @ Vpp)
    eigs = np.sort(np.real_if_close(eigs, tol=1e6).real)[::-1]

    return eigs, {
        "max_abs_mean_score": float(np.abs(mean_score).max()),
        "info_equality_gap": float(np.abs(J - H).max() / np.abs(H).max()),
        "H_cond": float(np.linalg.cond(H)),
        "prob_mass_dropped": dropped,
    }


def weighted_chisq_cdf(t, lam, n_draw=400_000, seed=0):
    """Monte Carlo CDF of sum_j lam_j chi2_1 (exact enough for a critical
    value; the alternative, Imhof's inversion, is not worth the extra code
    here because the Monte Carlo error is reported alongside)."""
    rng = np.random.default_rng(seed)
    draws = (rng.normal(size=(n_draw, len(lam))) ** 2) @ np.asarray(lam)
    return draws, float(np.mean(draws > t))
