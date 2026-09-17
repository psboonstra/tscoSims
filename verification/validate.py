"""
Validation of models.py against facts known independently of the code.

Nothing downstream should be believed until these pass, because an
independent reimplementation buys cross-checking only if it is right.
"""

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import brentq

import models as M

rng = np.random.default_rng(20260911)
K = 6
ECMO = np.array([0.27, 0.06, 0.06, 0.11, 0.03, 0.47])


# ---------------------------------------------------------------------------
# Quadrature grid over (Z, W) with Z ~ Bern(0.5), W ~ N(0,1), independent.
# ---------------------------------------------------------------------------

def grid(n_gh=48, binary_w=False):
    if binary_w:
        wn, ww = np.array([-1.0, 1.0]), np.array([0.5, 0.5])
    else:
        wn, ww = hermegauss(n_gh)
        ww = ww / ww.sum()
    Z = np.repeat([0.0, 1.0], len(wn))
    W = np.tile(wn, 2)
    wt = np.tile(ww, 2) * 0.5
    return np.column_stack([Z, W]), wt


def solve_po_intercepts(target, beta_W, n_gh=48, binary_w=False):
    """Choose PO cutpoints so the W-MARGINAL distribution equals `target`.

    Without this the ECMO distribution would only hold at W = 0, and the
    simulated data would not have the category frequencies the design claims.
    """
    X, wt = grid(n_gh, binary_w)
    cum = np.cumsum(target[::-1])[::-1][1:]          # Pr(Y >= 2..K)
    alpha = np.empty(K - 1)
    for j in range(K - 1):
        def f(a):
            q = 1.0 / (1.0 + np.exp(-(a - beta_W * X[:, 1])))
            return float(np.sum(wt * q) - cum[j])
        alpha[j] = brentq(f, -30, 30, xtol=1e-13)
    assert np.all(np.diff(alpha) < 0), "cutpoints not decreasing"
    return alpha


def check(name, cond, detail=""):
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return cond


ok = True
print("\n=== 1. probability constructors ===")
Xg, wtg = grid()
alpha = solve_po_intercepts(ECMO, 0.5)
P = M.po_probs(alpha, [0.0, 0.5], Xg)
ok &= check("po rows sum to 1", np.allclose(P.sum(1), 1))
marg = (wtg[:, None] * P).sum(0)
ok &= check("marginal control distribution == ECMO", np.allclose(marg, ECMO, atol=1e-9),
            f"max dev {np.abs(marg - ECMO).max():.2e}")

a2 = np.array([0.3, -0.4, 0.1, -1.0, 0.6])
B2 = rng.normal(size=(5, 2)) * 0.3
Pm = M.mr_probs(a2, B2, Xg)
ok &= check("mr rows sum to 1", np.allclose(Pm.sum(1), 1))

for C in (3, 4, 5, 6):
    for s2 in ("mr", "po"):
        n_low, n_up = C - 1, K - C + 1
        t1 = (np.sort(rng.normal(size=n_low - 1))[::-1], rng.normal(size=2) * 0.3)
        if s2 == "mr":
            t2 = (rng.normal(size=n_up), rng.normal(size=(n_up, 2)) * 0.3)
        else:
            t2 = (np.sort(rng.normal(size=n_up))[::-1], rng.normal(size=2) * 0.3)
        Pt = M.tsco_probs(t1, t2, Xg, K, C, s2)
        ok &= check(f"tsco C={C} stage2={s2}: rows sum to 1, all positive",
                    np.allclose(Pt.sum(1), 1) and np.all(Pt > 0))

print("\n=== 2. MLE recovers truth at large n (PO) ===")
n = 400_000
Zb = rng.binomial(1, 0.5, n).astype(float)
Wc = rng.normal(size=n)
Xs = np.column_stack([Zb, Wc])
beta_true = np.array([0.4, 0.5])
Pi = M.po_probs(alpha, beta_true, Xs)
y = 1 + (rng.random(n)[:, None] > np.cumsum(Pi, 1)).sum(1)
ll, (ah, bh), _ = M.fit_po(Xs, M.onehot(y, K), np.ones(n), K)
ok &= check("PO slopes recovered", np.max(np.abs(bh - beta_true)) < 0.02,
            f"beta_hat = {np.round(bh, 4)} vs {beta_true}")
ok &= check("PO cutpoints recovered", np.max(np.abs(ah - alpha)) < 0.03,
            f"max dev {np.abs(ah - alpha).max():.4f}")

print("\n=== 3. nesting lattice: PO truth, BINARY covariates only ===")
print("    Theory (from the stored derivation): with discrete covariates a PO")
print("    truth lies exactly inside PO|C|PO and PO|C|MR, so the population KL")
print("    projection must be exactly zero.")
Xb, wtb = grid(binary_w=True)
alpha_b = solve_po_intercepts(ECMO, 0.5, binary_w=True)
Ptrue_b = M.po_probs(alpha_b, [0.4, 0.5], Xb)
ent_b = -float(np.sum(wtb[:, None] * Ptrue_b * np.log(Ptrue_b)))
for C in (3, 4, 5, 6):
    for s2 in ("mr", "po"):
        llp, _, _ = M.fit_tsco(Xb, Ptrue_b, wtb, K, C, s2)
        kl = -llp - ent_b
        ok &= check(f"KL(PO truth -> PO|{C}|{s2.upper()}) == 0 (binary only)",
                    abs(kl) < 1e-10, f"KL = {kl:.3e}")

print("\n=== 4. same truth, CONTINUOUS W: projection must be strictly positive ===")
Ptrue_c = M.po_probs(alpha, [0.4, 0.5], Xg)
ent_c = -float(np.sum(wtg[:, None] * Ptrue_c * np.log(Ptrue_c)))
for C in (3, 4, 5, 6):
    for s2 in ("mr", "po"):
        llp, _, _ = M.fit_tsco(Xg, Ptrue_c, wtg, K, C, s2)
        kl = -llp - ent_c
        ok &= check(f"KL(PO truth -> PO|{C}|{s2.upper()}) > 0 (continuous W)",
                    kl > 1e-9, f"KL = {kl:.3e}")

print("\n=== 5. PO truth projects onto PO with KL 0 (sanity on the projector) ===")
llp, _, _ = M.fit_po(Xg, Ptrue_c, wtg, K)
ok &= check("KL(PO truth -> PO) == 0", abs(-llp - ent_c) < 1e-9, f"KL = {-llp - ent_c:.3e}")

print("\n=== 6. MR and CPPO are also misspecified under a PO truth + continuous W ===")
llp, _, _ = M.fit_mr(Xg, Ptrue_c, wtg, K)
klmr = -llp - ent_c
ok &= check("KL(PO truth -> MR) > 0", klmr > 1e-9, f"KL = {klmr:.3e}")
llp, _, _ = M.fit_cppo(Xg, Ptrue_c, wtg, K, G=np.array([0, 0, 0, 0, 1.0]), dev_col=0)
klc = -llp - ent_c
ok &= check("KL(PO truth -> CPPO) == 0 (PO is nested in CPPO at gamma=0)",
            abs(klc) < 1e-9, f"KL = {klc:.3e}")

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
