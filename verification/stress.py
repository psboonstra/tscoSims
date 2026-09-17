"""
How large can the type I error distortion get?

At beta_W = 0.5 the eigenvalues sit within 2e-4 of 1, but that may only reflect
a weak nuisance effect: the stage-1 nonlinearity is driven by the SPREAD of
s = x'beta, so a larger beta_W means a worse approximation. A referee will pick
the worst case, so this sweeps beta_W well past anything clinically plausible
and reports both the population KL and the chi-square weights.

The reported summary is
    sum_j lambda_j   -- the asymptotic MEAN of the LRT, against which the
                        software's df is the claimed value
    alpha_eff        -- the true asymptotic size of a nominal 5% chi2_q test.
"""

import numpy as np
from scipy.stats import chi2

import dgm, models as M, sandwich as SW


def one(beta_W, C, stage2, fitter="tsco", n_gh=96, n_draw=2_000_000):
    X, wt = dgm.quad_grid(n_gh, "normal")
    a = dgm.cutpoints(beta_W, "normal", n_gh)
    P = M.po_probs(a, [0.0, beta_W], X)
    K, p_dim = dgm.K, X.shape[1]
    Pc = np.clip(P, 1e-300, None)
    ent = -float(np.sum(wt[:, None] * Pc * np.log(Pc)))

    if fitter == "tsco":
        ll, (t1, t2), _ = M.fit_tsco(X, P, wt, K, C, stage2)
        v = M.tsco_pack(t1, t2, stage2)
        psi = M.tsco_z_index(K, C, p_dim, stage2, 0)
        fun, extra = M.tsco_nll_grad, (K, C, stage2)
    elif fitter == "mr":
        ll, (aa, B), _ = M.fit_mr(X, P, wt, K)
        v = np.concatenate([aa, B.ravel()])
        psi = np.array([(K - 1) + k * p_dim for k in range(K - 1)])
        fun, extra = M.mr_nll_grad, (K - 1, p_dim)
    elif fitter == "po":
        ll, (al, b), _ = M.fit_po(X, P, wt, K)
        v = np.concatenate([M.alpha_to_d(al), b])
        psi = np.array([K - 1])
        fun, extra = M.po_nll_grad, (K - 1, p_dim)
    else:
        raise ValueError(fitter)

    eigs, diag = SW.sandwich_eigs(fun, v, X, P, wt, K, extra, psi)
    q = len(psi)
    crit = chi2.ppf(0.95, q)
    draws, alpha_eff = SW.weighted_chisq_cdf(crit, eigs, n_draw=n_draw, seed=1)
    mcse = np.sqrt(alpha_eff * (1 - alpha_eff) / n_draw)
    return dict(kl=-ll - ent, q=q, eigs=eigs, sum_lam=eigs.sum(),
                alpha_eff=alpha_eff, mcse=mcse, **diag)


BETAS = [0.5, 1.0, 1.5, 2.0, 2.5]
MODELS = [("tsco", 5, "mr"), ("tsco", 5, "po"), ("tsco", 3, "mr"),
          ("tsco", 6, "mr"), ("mr", None, None), ("po", None, None)]

print("PO is a CANARY row: it is correctly specified under this truth, so every")
print("eigenvalue must come out at exactly 1. If it does not, the numerics are")
print("broken and nothing else in the table means anything.\n")
print("Nominal 5% test. alpha_eff is the TRUE asymptotic size; MC error ~1.5e-4.\n")
print(f"{'fitted model':<14}{'beta_W':>7}{'KL(truth->class)':>18}"
      f"{'q':>3}{'sum lambda':>12}{'min lam':>9}{'max lam':>9}{'alpha_eff':>11}")
print("-" * 83)
for fitter, C, s2 in MODELS:
    name = f"PO|{C}|{s2.upper()}" if fitter == "tsco" else fitter.upper()
    for bw in BETAS:
        r = one(bw, C, s2, fitter)
        flag = "" if abs(r["alpha_eff"] - 0.05) < 0.002 else "   <-- distorted"
        if fitter == "po" and abs(r["sum_lam"] - r["q"]) > 1e-6:
            flag += "  !! CANARY: PO is correctly specified, lambda must be 1"
        print(f"{name:<14}{bw:>7.1f}{r['kl']:>18.3e}{r['q']:>3}"
              f"{r['sum_lam']:>12.5f}{r['eigs'].min():>9.5f}{r['eigs'].max():>9.5f}"
              f"{r['alpha_eff']:>11.4f}{flag}")
    print()
