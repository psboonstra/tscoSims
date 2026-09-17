"""Tail signature of the type I error distortion at n=200.

A distortion caused by a FIXED FRACTION of non-identified fits shows up as a
PLATEAU: the observed rejection rate stops falling as the nominal level drops,
because those replicates produce an essentially unbounded statistic and reject
at every level. A distortion caused by an ordinary imperfect chi-square
approximation instead shrinks roughly in proportion to the nominal level.
"""
import multiprocessing as mp, numpy as np
from scipy.stats import chi2
import dgm, models as M
from sim_typeI import METHODS, lrt

K, N, NSIM = dgm.K, 200, 4000

def rep(seed):
    rng = np.random.default_rng(seed)
    X, y = dgm.simulate(N, rng, beta_Z=0.0, w_kind="normal")
    Yoh, w = M.onehot(y, K), np.ones(N)
    return [lrt(spec, X, Yoh, w) for _, spec in METHODS]

if __name__ == "__main__":
    with mp.Pool(2) as pool:
        res = pool.map(rep, [int(s) for s in 5150_000_000 + np.arange(NSIM)], chunksize=8)
    names = [m[0] for m in METHODS]
    levels = (0.10, 0.05, 0.01, 0.001, 1e-4)
    print(f"n={N}, {NSIM} reps. Observed rejection rate at each nominal level.")
    print("A plateau (rate stops falling) = a fixed fraction of non-identified fits.\n")
    print(f"{'method':<10}" + "".join(f"{a:>10.4f}" for a in levels) + f"{'ratio':>10}")
    print("-"*72)
    for j, nm in enumerate(names):
        stat = np.array([r[j][0] for r in res], float)
        df = int(np.nanmax([r[j][1] for r in res]))
        g = np.isfinite(stat); s = stat[g]
        rates = [float((s > chi2.ppf(1-a, df)).mean()) for a in levels]
        # ratio of observed to nominal at the most extreme level
        print(f"{nm:<10}" + "".join(f"{r:>10.4f}" for r in rates)
              + f"{rates[-1]/levels[-1]:>10.0f}x")
