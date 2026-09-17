"""CPPO type I error with the parameter space enforced BY CONSTRUCTION.

The earlier result was an artefact. An unconstrained optimiser drove gamma
below its bound, giving the second-highest category a NEGATIVE fitted
probability. When no treated subject occupies that category the invalid value
is never evaluated against an observation, so the log-clip never bites and the
likelihood is inflated for free -- ll_full = -210 vs ll_red = -269, LRT ~110.

With the departure at the last cutpoint, monotonicity for treated rows requires
    a_{K-1} - b1 - gamma <= a_{K-2} - b1   <=>   gamma >= a_{K-1} - a_{K-2}
(control rows are automatic). That is a single scalar constraint, so
reparameterise gamma = (a_{K-1} - a_{K-2}) + exp(t) and optimise t freely: the
fit can never leave the CPPO parameter space.

The constrained MLE then sits ON the boundary when the cell is empty, which is
a boundary problem -- so chi2_2 is STILL not the right reference there. This
script only removes the invalid-likelihood bug; it does not fix the reference
distribution.
"""
import multiprocessing as mp, numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import chi2
import models as M

K, BW, N, NSIM = 6, 0.5, 200, 2500
ECMO = np.array([0.27,0.06,0.06,0.11,0.03,0.47])
ALPHA = logit(np.cumsum(ECMO[::-1])[::-1][1:])
G = np.zeros(K-1); G[K-2] = 1.0


def nll_reparam(v, X, Yoh, w, Km1, p_dim):
    d, beta, t = v[:Km1], v[Km1:Km1+p_dim], v[Km1+p_dim]
    alpha = M.d_to_alpha(d)
    gamma = (alpha[-1] - alpha[-2]) + np.exp(np.clip(t, -50, 50))
    eta = (X@beta)[:,None] + (gamma*X[:,0])[:,None]*G[None,:]
    q = expit(alpha[None,:]-eta)
    p = np.empty((X.shape[0], Km1+1))
    p[:,0] = 1-q[:,0]; p[:,1:Km1] = q[:,:-1]-q[:,1:]; p[:,Km1] = q[:,-1]
    if np.any(p < -1e-12):
        return 1e8
    return -float(np.sum(w[:,None]*Yoh*np.log(np.clip(p,1e-300,None))))


def fit_constrained(X, Yoh, w, K):
    p_dim = X.shape[1]
    base = M._po_start(Yoh, w, p_dim)
    best = None
    for t0 in (-3.0, -1.0, 0.5):
        x0 = np.concatenate([base, [t0]])
        r = minimize(nll_reparam, x0, args=(X,Yoh,w,K-1,p_dim), method="BFGS",
                     options={"maxiter":2000, "gtol":1e-8})
        if best is None or r.fun < best.fun:
            best = r
    a = M.d_to_alpha(best.x[:K-1])
    gam = (a[-1]-a[-2]) + np.exp(np.clip(best.x[-1], -50, 50))
    return -best.fun, float(gam), float(a[-1]-a[-2])


def rep(seed):
    rng = np.random.default_rng(seed)
    A = rng.binomial(1,.5,N).astype(float); W = rng.normal(size=N)
    X = np.column_stack([A,W])
    P = M.po_probs(ALPHA,[0.0,BW],X)
    y = np.clip(1+(rng.random(N)[:,None] > np.cumsum(P,1)).sum(1),1,K)
    Yoh, w = M.onehot(y,K), np.ones(N)
    r,_,_   = M.fit_po(X[:,1:],Yoh,w,K)
    pof,_,_ = M.fit_po(X,Yoh,w,K)
    f_un,(_,_,g_un),_ = M.fit_cppo(X,Yoh,w,K,G,0)
    f_c, g_c, bound   = fit_constrained(X,Yoh,w,K)
    return (2*(f_un-r), 2*(f_c-r), 2*(pof-r), float(g_un), g_c, bound,
            int((y[A==1]==K-1).sum()))


if __name__ == "__main__":
    with mp.Pool(2) as pool:
        out = pool.map(rep, [int(s) for s in 2718_000_000+np.arange(NSIM)], chunksize=4)
    s_un,s_c,s_po,g_un,g_c,bnd,n2 = map(np.array, zip(*out))
    s_c = np.maximum(s_c, 0.0)
    empty = n2 == 0
    print(f"n={N}, {NSIM} reps, nominal 0.05")
    print(f"Treated arm empty in the 2nd-highest category: {empty.mean():.2%}\n")
    print(f"{'test':<38}{'rejection':>11}{'SE':>8}")
    print("-"*57)
    for lab, s, df in [("PO (correctly specified control)", s_po, 1),
                       ("CPPO unconstrained  (my earlier BUG)", s_un, 2),
                       ("CPPO constrained to the CPPO space", s_c, 2)]:
        r = (s > chi2.ppf(.95, df)).astype(float)
        print(f"{lab:<38}{r.mean():>11.4f}{r.std(ddof=1)/np.sqrt(len(r)):>8.4f}")
    at_bound = (g_c - bnd) < 1e-4
    print(f"\n  constrained gamma AT the boundary: {at_bound.mean():.2%} of reps "
          f"({(at_bound & empty).sum()}/{empty.sum()} of the empty-cell reps)")
    print(f"  constrained gamma: median {np.median(g_c):.3f}, min {g_c.min():.3f}, "
          f"max {g_c.max():.3f}   (unconstrained min {g_un.min():.1f})")
    print(f"\n  LRT median: unconstrained {np.median(s_un):8.2f}   constrained {np.median(s_c):8.2f}")
    print(f"  Among the {empty.sum()} empty-cell reps:")
    print(f"    unconstrained median {np.median(s_un[empty]):8.2f}  rejects {np.mean(s_un[empty]>5.99):6.1%}")
    print(f"    constrained   median {np.median(s_c[empty]):8.2f}  rejects {np.mean(s_c[empty]>5.99):6.1%}")
