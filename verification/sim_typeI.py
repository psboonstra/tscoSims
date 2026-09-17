"""
Finite-sample type I error of the joint LRT for Z.

The asymptotic calculation (stress.py) says the weighted-chi-square correction
is negligible -- eigenvalues within 3% of 1 even at implausible nuisance
effects, asymptotic size within 0.001 of nominal. That settles the
misspecification question but says nothing about n = 200 with a category at 3%
prevalence, where the chi-square approximation can fail for ordinary
small-sample reasons: sparse cells, near-separation, and boundary fits.

This simulates the actual rejection rate. PO is included as a control: it is
correctly specified here, so any distortion it shows is small-sample noise
common to every method rather than anything to do with TsCO.
"""

import argparse
import multiprocessing as mp
import numpy as np
from scipy.stats import chi2

import dgm, models as M

K = dgm.K

METHODS = [
    ("PO",       dict(kind="po")),
    ("PO|5|MR",  dict(kind="tsco", C=5, s2="mr")),
    ("PO|5|PO",  dict(kind="tsco", C=5, s2="po")),
    ("PO|3|MR",  dict(kind="tsco", C=3, s2="mr")),
    ("PO|6|MR",  dict(kind="tsco", C=6, s2="mr")),
    ("MR",       dict(kind="mr")),
    ("CPPO",     dict(kind="cppo")),
]


def lrt(spec, X, Yoh, w):
    """Joint LRT for every coefficient on column 0 (Z)."""
    kind = spec["kind"]
    Xr = X[:, 1:]
    try:
        if kind == "po":
            f, _, _ = M.fit_po(X, Yoh, w, K)
            r, _, _ = M.fit_po(Xr, Yoh, w, K)
            df = 1
        elif kind == "mr":
            f, _, _ = M.fit_mr(X, Yoh, w, K)
            r, _, _ = M.fit_mr(Xr, Yoh, w, K)
            df = K - 1
        elif kind == "cppo":
            G = np.array([0, 0, 0, 0, 1.0])
            f, _, _ = M.fit_cppo(X, Yoh, w, K, G, dev_col=0)
            r, _, _ = M.fit_po(Xr, Yoh, w, K)
            df = 2
        elif kind == "tsco":
            C, s2 = spec["C"], spec["s2"]
            f, _, _ = M.fit_tsco(X, Yoh, w, K, C, s2)
            r, _, _ = M.fit_tsco(Xr, Yoh, w, K, C, s2)
            df = len(M.tsco_z_index(K, C, X.shape[1], s2, 0))
        else:
            raise ValueError(kind)
    except Exception:
        return np.nan, np.nan

    stat = 2 * (f - r)
    if not np.isfinite(stat):
        return np.nan, df
    if -1e-6 < stat < 0:
        stat = 0.0
    return stat, df


def one_rep(args):
    seed, n, w_kind = args
    rng = np.random.default_rng(seed)
    X, y = dgm.simulate(n, rng, beta_Z=0.0, w_kind=w_kind)
    Yoh, w = M.onehot(y, K), np.ones(n)
    out = {}
    for name, spec in METHODS:
        if w_kind == "none" and spec["kind"] == "cppo":
            pass
        out[name] = lrt(spec, X, Yoh, w)
    # minimum category count, to relate failures to sparsity
    out["_mincount"] = (np.bincount(y, minlength=K + 1)[1:].min(), 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsim", type=int, default=2000)
    ap.add_argument("--ns", type=int, nargs="+", default=[200, 500, 1000, 4000])
    ap.add_argument("--wkind", default="normal")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()

    print(f"Null: PO truth, beta_Z = 0, beta_W = {dgm.BETA_W}, W = {a.wkind}, "
          f"K = {K}, ECMO marginal.")
    print(f"{a.nsim} replicates per n; nominal alpha = {a.alpha}.")
    print(f"Monte Carlo SE at 5% with {a.nsim} reps = "
          f"{np.sqrt(.05*.95/a.nsim):.4f}\n")

    names = [m[0] for m in METHODS]
    print(f"{'n':>6}  " + "".join(f"{nm:>12}" for nm in names))
    print("-" * (8 + 12 * len(names)))

    for n in a.ns:
        seeds = 20260911_000 + np.arange(a.nsim) + n * 1_000_000
        with mp.Pool(a.procs) as pool:
            res = pool.map(one_rep, [(int(s), n, a.wkind) for s in seeds], chunksize=8)

        row_rej, row_fail = [], []
        for nm in names:
            stats = np.array([r[nm][0] for r in res], float)
            dfs = np.array([r[nm][1] for r in res], float)
            good = np.isfinite(stats)
            df = int(np.nanmax(dfs))
            rej = float(np.mean(stats[good] > chi2.ppf(1 - a.alpha, df)))
            row_rej.append(rej)
            row_fail.append(1 - good.mean())
        mc = np.array([r["_mincount"][0] for r in res])
        print(f"{n:>6}  " + "".join(f"{r:>12.4f}" for r in row_rej)
              + f"   | min cat count: med {np.median(mc):.0f}, "
                f"{(mc == 0).mean():.1%} empty")
        if max(row_fail) > 0:
            print(f"{'':>6}  " + "".join(f"{f:>12.1%}" for f in row_fail)
                  + "   | fit failures")


if __name__ == "__main__":
    main()
