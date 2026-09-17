"""Asymptotic null of the joint LRT for Z under a PO truth, by design."""

import numpy as np

import dgm, models as M, sandwich as SW

np.set_printoptions(precision=5, suppress=True)


def run(w_kind, C, stage2, n_gh=64):
    X, wt = dgm.quad_grid(n_gh, w_kind)
    P = dgm.true_probs(X, beta_Z=0.0, w_kind=w_kind, n_gh=n_gh)
    K, p_dim = dgm.K, X.shape[1]

    ll, (t1, t2), _ = M.fit_tsco(X, P, wt, K, C, stage2)
    v = M.tsco_pack(t1, t2, stage2)
    psi = M.tsco_z_index(K, C, p_dim, stage2, z_col=0)

    ent = -float(np.sum(wt[:, None] * P * np.log(P)))
    kl = -ll - ent

    eigs, diag = SW.sandwich_eigs(M.tsco_nll_grad, v, X, P, wt, K,
                                  (K, C, stage2), psi)
    return dict(kl=kl, psi_star=v[psi], eigs=eigs, q=len(psi), **diag)


print("=" * 78)
print("PREREQUISITE (a): does the KL projection put zero on Z?")
print("  If psi* != 0 the LRT is not testing the hypothesis it claims to.")
print("=" * 78)
for w_kind in ("none", "binary", "normal"):
    for C in (3, 4, 5, 6):
        for s2 in ("mr", "po"):
            r = run(w_kind, C, s2)
            print(f"  W={w_kind:7s} PO|{C}|{s2.upper():2s}  "
                  f"max|psi*| = {np.abs(r['psi_star']).max():.3e}   "
                  f"max|mean score| = {r['max_abs_mean_score']:.2e}")

print()
print("=" * 78)
print("ASYMPTOTIC NULL:  LRT -> sum_j lambda_j chi2_1")
print("  All lambda_j = 1  <=>  the usual chi2_q is valid.")
print("  E[LRT] = sum_j lambda_j   (compare with q = df the software uses)")
print("=" * 78)
hdr = f"{'W':<8}{'model':<10}{'q':>3}{'KL(truth->class)':>19}{'sum lam':>10}{'eigenvalues':>30}"
for w_kind in ("none", "binary", "normal"):
    print("-" * 78)
    print(hdr if w_kind == "none" else "")
    for C in (3, 4, 5, 6):
        for s2 in ("mr", "po"):
            r = run(w_kind, C, s2)
            e = r["eigs"]
            print(f"{w_kind:<8}PO|{C}|{s2.upper():<2}{'':<4}{r['q']:>3}"
                  f"{r['kl']:>19.3e}{e.sum():>10.4f}   "
                  + " ".join(f"{x:7.4f}" for x in e))
