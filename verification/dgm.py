"""The null data-generating mechanism for the type I error probe.

Truth: proportional odds, K = 6, with the ECMO control distribution
{0.27, 0.06, 0.06, 0.11, 0.03, 0.47} holding MARGINALLY over W, binary
Z ~ Bern(0.5) with beta_Z = 0, and W ~ N(0, 1) entering proportionally.

Cutpoints are solved so that the W-marginal matches ECMO. Setting them from
the ECMO probabilities directly would make the design's stated category
frequencies hold only at W = 0.
"""

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import brentq

import models as M

K = 6
ECMO = np.array([0.27, 0.06, 0.06, 0.11, 0.03, 0.47])
BETA_W = 0.5


def quad_grid(n_gh=64, w_kind="normal", w_trunc=5.0):
    """Quadrature over (Z, W). Z is enumerated exactly; W by Gauss-Hermite.

    `w_trunc` drops Gauss-Hermite nodes beyond |w| > w_trunc and renormalises.
    Those nodes carry negligible probability but, once multiplied by a large
    nuisance coefficient, drive the fitted category probabilities to underflow
    and corrupt the information matrices. Truncation changes the target
    distribution by less than 1e-6 of its mass at the default.
    """
    if w_kind == "normal":
        wn, ww = hermegauss(n_gh)
        keep = np.abs(wn) <= w_trunc
        wn, ww = wn[keep], ww[keep]
        ww = ww / ww.sum()
    elif w_kind == "binary":
        wn, ww = np.array([-1.0, 1.0]), np.array([0.5, 0.5])
    elif w_kind == "none":
        Z = np.array([0.0, 1.0])
        return Z[:, None], np.array([0.5, 0.5])
    else:
        raise ValueError(w_kind)
    Z = np.repeat([0.0, 1.0], len(wn))
    W = np.tile(wn, 2)
    return np.column_stack([Z, W]), np.tile(ww, 2) * 0.5


def cutpoints(beta_W=BETA_W, w_kind="normal", n_gh=64):
    if w_kind == "none":
        wn, ww, bw = np.zeros(1), np.ones(1), 0.0
    elif w_kind == "binary":
        wn, ww, bw = np.array([-1.0, 1.0]), np.array([.5, .5]), beta_W
    else:
        wn, ww = hermegauss(n_gh)
        keep = np.abs(wn) <= 5.0
        wn, ww = wn[keep], ww[keep]
        ww = ww / ww.sum(); bw = beta_W
    cum = np.cumsum(ECMO[::-1])[::-1][1:]
    a = np.empty(K - 1)
    for j in range(K - 1):
        a[j] = brentq(
            lambda t: float(np.sum(ww / (1 + np.exp(-(t - bw * wn))))) - cum[j],
            -30, 30, xtol=1e-14)
    assert np.all(np.diff(a) < 0)
    return a


def true_probs(X, beta_Z=0.0, beta_W=BETA_W, w_kind="normal", n_gh=64):
    a = cutpoints(beta_W, w_kind, n_gh)
    beta = [beta_Z] if X.shape[1] == 1 else [beta_Z, beta_W]
    return M.po_probs(a, beta, X)


def simulate(n, rng, beta_Z=0.0, beta_W=BETA_W, w_kind="normal"):
    Z = rng.binomial(1, 0.5, n).astype(float)
    if w_kind == "none":
        X = Z[:, None]
    elif w_kind == "binary":
        X = np.column_stack([Z, rng.choice([-1.0, 1.0], n)])
    else:
        X = np.column_stack([Z, rng.normal(size=n)])
    P = true_probs(X, beta_Z, beta_W, w_kind)
    y = 1 + (rng.random(n)[:, None] > np.cumsum(P, 1)).sum(1)
    return X, np.clip(y, 1, K)
