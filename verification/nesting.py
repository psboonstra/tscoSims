"""
How far does "PO is nested in TsCO for discrete covariates" actually go?

The stored derivation says: under a PO truth the induced stage-1 conditional
log-odds is

    log(e^a_k - e^a_C) - [ s + log(1 + e^a_C e^-s) ],     s = x'beta

so the covariate enters through a NONLINEAR function g(s). The stored
conclusion was "with binary covariates a linear coefficient hits both points
exactly, so PO is contained in PO|C|PO for every C".

That argument counts points. With ONE binary covariate, s takes 2 values and
the stage-1 linear predictor has 2 free parameters (intercept + slope), so g
is matched exactly. With TWO binary covariates s takes 4 values while an
ADDITIVE linear predictor still has only 3, so g(s) generally cannot be
matched: the nonlinearity manufactures a Z-by-W interaction that the additive
stage-1 model has no parameter for.

This script tests that reading directly, by computing the population KL
projection on saturated and unsaturated discrete designs.
"""

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import brentq

import models as M

K = 6
ECMO = np.array([0.27, 0.06, 0.06, 0.11, 0.03, 0.47])


def solve_intercepts(target, beta_W, Wvals, Wwts):
    cum = np.cumsum(target[::-1])[::-1][1:]
    a = np.empty(K - 1)
    for j in range(K - 1):
        a[j] = brentq(lambda t: float(np.sum(Wwts / (1 + np.exp(-(t - beta_W * Wvals))))) - cum[j],
                      -30, 30, xtol=1e-13)
    return a


def kl_to(fitter, X, Ptrue, wt):
    ent = -float(np.sum(wt[:, None] * Ptrue * np.log(Ptrue)))
    ll = fitter(X, Ptrue, wt)
    return -ll - ent


BZ, BW = 0.4, 0.5

designs = {}

# (a) a single binary Z, no other covariate
Z = np.array([0.0, 1.0]); wt = np.array([0.5, 0.5])
a = solve_intercepts(ECMO, 0.0, np.zeros(2), wt)
designs["Z binary only (1 covariate)"] = (
    Z[:, None], M.po_probs(a, [BZ], Z[:, None]), wt)

# (b) binary Z and binary W, additive design matrix
Zb = np.repeat([0.0, 1.0], 2); Wb = np.tile([-1.0, 1.0], 2)
wt2 = np.full(4, 0.25)
a2 = solve_intercepts(ECMO, BW, np.array([-1.0, 1.0]), np.array([.5, .5]))
Xadd = np.column_stack([Zb, Wb])
Padd = M.po_probs(a2, [BZ, BW], Xadd)
designs["Z, W both binary (additive fit)"] = (Xadd, Padd, wt2)

# (c) same truth, but the fitted design SATURATES the 4 support points by
#     adding the Z-by-W interaction
Xsat = np.column_stack([Zb, Wb, Zb * Wb])
designs["Z, W both binary (saturated fit, + Z:W)"] = (Xsat, Padd, wt2)

# (d) binary Z, continuous W
gh, gw = hermegauss(48); gw = gw / gw.sum()
Zc = np.repeat([0.0, 1.0], len(gh)); Wc = np.tile(gh, 2)
wtc = np.tile(gw, 2) * 0.5
ac = solve_intercepts(ECMO, BW, gh, gw)
Xc = np.column_stack([Zc, Wc])
designs["Z binary, W continuous"] = (Xc, M.po_probs(ac, [BZ, BW], Xc), wtc)

print(f"{'design':<44}", "".join(f"{f'PO|{C}|{s.upper()}':>14}"
                                 for C in (3, 4, 5, 6) for s in ("mr", "po")))
print("-" * 44 + "-" * 14 * 8)
for name, (X, P, wt_) in designs.items():
    row = []
    for C in (3, 4, 5, 6):
        for s2 in ("mr", "po"):
            kl = kl_to(lambda X_, P_, w_: M.fit_tsco(X_, P_, w_, K, C, s2)[0], X, P, wt_)
            row.append(f"{kl:14.3e}")
    print(f"{name:<44}" + "".join(row))

print("\nInterpretation: a row of ~1e-16 means the PO truth lies exactly inside")
print("that TsCO class. Anything at 1e-5 or above is genuine misspecification.")
