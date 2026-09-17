"""Type I error under the REWORKED DGM (2026-09-14 true_probs.R, `null`).

Truth: PO, K = 6, levels 0..5, ECMO {0.27,0.06,0.06,0.11,0.03,0.47} holding at
A = W = 0 (not marginally), binary A with b1 = 0, W ~ N(0,1) with BETA_W = 0.5.

Paper level C = 5 is 1-indexed level 6 here, so tsco_popo == PO|6|PO.
CPPO's G is the indicator of the last cutpoint, as in methods/cppo.R.
"""
import multiprocessing as mp, numpy as np
from scipy.special import logit
from scipy.stats import chi2
import models as M

K, BW, N, NSIM = 6, 0.5, 200, 6000
ECMO = np.array([0.27,0.06,0.06,0.11,0.03,0.47])
ALPHA = logit(np.cumsum(ECMO[::-1])[::-1][1:])     # cutpoints: ECMO at W = 0
G = np.zeros(K-1); G[K-2] = 1.0
C_TSCO = 6                                          # paper's C = 5 (levels 0..5)

def sim(n, rng):
    A = rng.binomial(1,.5,n).astype(float); W = rng.normal(size=n)
    X = np.column_stack([A, W])
    P = M.po_probs(ALPHA, [0.0, BW], X)
    y = 1 + (rng.random(n)[:,None] > np.cumsum(P,1)).sum(1)
    return X, np.clip(y,1,K)

def rep(seed):
    rng = np.random.default_rng(seed)
    X, y = sim(N, rng); Yoh, w = M.onehot(y,K), np.ones(N); Xr = X[:,1:]
    out = {}
    f,_,_ = M.fit_po(X,Yoh,w,K);  r,_,_ = M.fit_po(Xr,Yoh,w,K)
    out["po"] = (2*(f-r), 1)
    f,_,_ = M.fit_tsco(X,Yoh,w,K,C_TSCO,"po"); r,_,_ = M.fit_tsco(Xr,Yoh,w,K,C_TSCO,"po")
    out["tsco_popo"] = (2*(f-r), 2)
    f,(_,_,g),_ = M.fit_cppo(X,Yoh,w,K,G,0);   r,_,_ = M.fit_po(Xr,Yoh,w,K)
    out["cppo"] = (2*(f-r), 2); out["_gam"] = abs(float(g))
    out["_mincount"] = np.bincount(y, minlength=K+1)[1:].min()
    return out

if __name__ == "__main__":
    with mp.Pool(2) as pool:
        res = pool.map(rep, [int(s) for s in 31415_000_000 + np.arange(NSIM)], chunksize=8)
    names = ["po","tsco_popo","cppo"]
    lv = (0.10, 0.05, 0.01, 0.001, 1e-4)
    R, tails = {}, {}
    for nm in names:
        st = np.array([r[nm][0] for r in res], float); df = res[0][nm][1]
        R[nm] = (st > chi2.ppf(.95, df)).astype(float)
        tails[nm] = [float((st[np.isfinite(st)] > chi2.ppf(1-a, df)).mean()) for a in lv]
    print(f"REWORKED DGM, scenario 'null', n={N}, {NSIM} reps (MCSE at 5% = "
          f"{np.sqrt(.05*.95/NSIM):.4f})\n")
    print(f"{'method':<12}{'df':>4}{'rejection':>11}{'vs po (paired)':>17}{'SE':>8}{'z':>7}")
    print("-"*59)
    for nm in names:
        r=R[nm]; d=r-R["po"]; sed=d.std(ddof=1)/np.sqrt(len(d)); z=d.mean()/sed if sed>0 else 0
        print(f"{nm:<12}{res[0][nm][1]:>4}{r.mean():>11.4f}{d.mean():>17.4f}{sed:>8.4f}{z:>7.1f}"
              + ("  (control)" if nm=="po" else ""))
    print(f"\n{'method':<12}" + "".join(f"{a:>10.4f}" for a in lv) + "   <- observed rejection by nominal level")
    for nm in names:
        print(f"{nm:<12}" + "".join(f"{v:>10.4f}" for v in tails[nm]))
    gam = np.array([r["_gam"] for r in res]); mc = np.array([r["_mincount"] for r in res])
    print(f"\n  CPPO |gamma| > 20 in {(gam>20).mean():.2%} of replicates;  median |gamma| = {np.median(gam):.3f}")
    print(f"  smallest observed category count: median {np.median(mc):.0f}, {(mc==0).mean():.1%} of datasets have an empty category")
