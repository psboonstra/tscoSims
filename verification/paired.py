"""
Paired type I error comparison at the sample size that matters.

Every method is fitted to the SAME replicate, so their rejection indicators are
strongly positively correlated and the raw rejection rates in a row move up and
down together (visible at n = 1000, where all seven sit near 0.056 including
correctly-specified PO). Comparing each rate to 0.05 separately therefore
attributes shared Monte Carlo drift to individual methods.

PO is correctly specified under this truth, so it is the control: what matters
is each method's rejection rate MINUS PO's on the same replicates, with a
standard error computed from the paired differences.
"""

import multiprocessing as mp
import numpy as np
from scipy.stats import chi2

import dgm, models as M
from sim_typeI import METHODS, lrt

K = dgm.K
NSIM, N, ALPHA = 6000, 200, 0.05


def rep(seed):
    rng = np.random.default_rng(seed)
    X, y = dgm.simulate(N, rng, beta_Z=0.0, w_kind="normal")
    Yoh, w = M.onehot(y, K), np.ones(N)
    return [lrt(spec, X, Yoh, w) for _, spec in METHODS]


if __name__ == "__main__":
    seeds = [int(s) for s in 777_000_000 + np.arange(NSIM)]
    with mp.Pool(2) as pool:
        res = pool.map(rep, seeds, chunksize=8)

    names = [m[0] for m in METHODS]
    R = np.full((NSIM, len(names)), np.nan)
    for j, nm in enumerate(names):
        stat = np.array([r[j][0] for r in res], float)
        df = int(np.nanmax([r[j][1] for r in res]))
        R[:, j] = (stat > chi2.ppf(1 - ALPHA, df)).astype(float)
        R[~np.isfinite(stat), j] = np.nan

    keep = np.isfinite(R).all(1)
    R = R[keep]
    print(f"n = {N}, {R.shape[0]} usable replicates of {NSIM}, nominal {ALPHA}\n")
    print(f"{'method':<10}{'reject':>9}{'SE':>8}   {'vs PO (paired)':>16}{'SE':>8}{'z':>7}")
    print("-" * 62)
    po = R[:, names.index("PO")]
    for j, nm in enumerate(names):
        r = R[:, j]
        se = r.std(ddof=1) / np.sqrt(len(r))
        d = r - po
        sed = d.std(ddof=1) / np.sqrt(len(d))
        z = d.mean() / sed if sed > 0 else 0.0
        tag = "  (control)" if nm == "PO" else ("   <-- inflated" if z > 3 else "")
        print(f"{nm:<10}{r.mean():>9.4f}{se:>8.4f}   {d.mean():>16.4f}"
              f"{sed:>8.4f}{z:>7.1f}{tag}")
