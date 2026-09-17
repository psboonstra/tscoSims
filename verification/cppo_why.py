"""Are the CPPO 'rejections' real, or an artefact of an unconstrained optimiser?

CPPO requires ordered cumulative probabilities. With the departure at the top
cutpoint, Pr(Y>=K-1|A=1) <= Pr(Y>=K-2|A=1) means gamma >= a_{K-1} - a_{K-2}.
So gamma is bounded BELOW and free above. gamma -> +inf means "no treated
subject in the top category" -- but the top category has probability 0.47, so
that should be impossible at n=200.

That makes a large |gamma| suspicious. This records, per replicate:
  - the LRT statistic and both log-likelihoods
  - gamma
  - whether the FITTED cumulative probabilities are monotone (i.e. whether the
    reported optimum is even inside the CPPO parameter space)
  - the treated-arm count in the top two categories, the cells gamma depends on
"""
import multiprocessing as mp, numpy as np
from scipy.special import expit, logit
import models as M

K, BW, N, NSIM = 6, 0.5, 200, 3000
ECMO = np.array([0.27,0.06,0.06,0.11,0.03,0.47])
ALPHA = logit(np.cumsum(ECMO[::-1])[::-1][1:])
G = np.zeros(K-1); G[K-2] = 1.0

def rep(seed):
    rng = np.random.default_rng(seed)
    A = rng.binomial(1,.5,N).astype(float); W = rng.normal(size=N)
    X = np.column_stack([A,W])
    P = M.po_probs(ALPHA,[0.0,BW],X)
    y = np.clip(1+(rng.random(N)[:,None] > np.cumsum(P,1)).sum(1),1,K)
    Yoh, w = M.onehot(y,K), np.ones(N)
    f,(al,be,gam),_ = M.fit_cppo(X,Yoh,w,K,G,0)
    r,_,_ = M.fit_po(X[:,1:],Yoh,w,K)
    # is the reported optimum inside the CPPO parameter space?
    eta = (X@be)[:,None] + (gam*X[:,0])[:,None]*G[None,:]
    q = expit(al[None,:]-eta)
    mono = bool(np.all(q[:,1:] <= q[:,:-1] + 1e-12))
    # gamma's identifying cells, treated arm
    trt = A==1
    return dict(stat=2*(f-r), ll_full=f, ll_red=r, gam=float(gam), mono=mono,
                n_trt_top=int((y[trt]==K).sum()), n_trt_2nd=int((y[trt]==K-1).sum()),
                n_ctl_2nd=int((y[~trt]==K-1).sum()),
                gam_bound=float(al[-1]-al[-2]))

if __name__ == "__main__":
    with mp.Pool(2) as pool:
        R = pool.map(rep, [int(s) for s in 2718_000_000+np.arange(NSIM)], chunksize=8)
    stat=np.array([r["stat"] for r in R]); gam=np.array([r["gam"] for r in R])
    mono=np.array([r["mono"] for r in R]); 
    big = np.abs(gam)>20
    print(f"n={N}, {NSIM} reps\n")
    print(f"  |gamma| > 20:            {big.mean():.2%}")
    print(f"  fitted probs NON-monotone (outside the CPPO space): {(~mono).mean():.2%}")
    print(f"  overlap (|gamma|>20 AND non-monotone):              {(big & ~mono).mean():.2%}\n")
    print("  LRT statistic (df = 2; 5% crit = 5.99, 0.01% crit = 18.4):")
    for lab, m in [("all reps", np.ones(NSIM,bool)), ("|gamma|<=20", ~big), ("|gamma|>20", big)]:
        if m.sum()==0: continue
        s=stat[m]
        print(f"    {lab:<13} n={m.sum():<5} median {np.median(s):8.2f}   "
              f"q90 {np.quantile(s,.9):8.2f}   max {s.max():10.2f}   "
              f"frac>5.99 {np.mean(s>5.99):.3f}")
    print("\n  Among |gamma| > 20 replicates:")
    sub=[r for r in R if abs(r["gam"])>20]
    print(f"    monotone fitted probs: {np.mean([r['mono'] for r in sub]):.1%}")
    print(f"    treated-arm count in TOP category:        median {np.median([r['n_trt_top'] for r in sub]):.0f}")
    print(f"    treated-arm count in 2nd-highest category: median {np.median([r['n_trt_2nd'] for r in sub]):.0f}, "
          f"zero in {np.mean([r['n_trt_2nd']==0 for r in sub]):.1%}")
    print(f"    control-arm count in 2nd-highest category: median {np.median([r['n_ctl_2nd'] for r in sub]):.0f}, "
          f"zero in {np.mean([r['n_ctl_2nd']==0 for r in sub]):.1%}")
    print(f"    sign of gamma: {np.mean([r['gam']>0 for r in sub]):.1%} positive")
    print("\n  A few individual |gamma|>20 replicates:")
    print(f"    {'gamma':>12}{'LRT':>10}{'ll_full':>11}{'ll_red':>11}{'mono':>7}{'trt 2nd':>9}{'trt top':>9}")
    for r in sub[:10]:
        print(f"    {r['gam']:>12.2f}{r['stat']:>10.2f}{r['ll_full']:>11.2f}{r['ll_red']:>11.2f}"
              f"{str(r['mono']):>7}{r['n_trt_2nd']:>9}{r['n_trt_top']:>9}")
