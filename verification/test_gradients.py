"""Check every analytic gradient against a high-accuracy numerical one.

Richardson-extrapolated central differences are used rather than plain central
differences so that the check is tight enough (~1e-11) to certify gradients
that must drive an optimiser to a 1e-12 tolerance.
"""

import numpy as np
import models as M

rng = np.random.default_rng(7)


# h0 = 1e-3: the log-likelihoods here are O(1e3), so subtractive cancellation
# dominates below this step. Verified empirically -- the apparent error in the
# MR probability-weighted gradient GROWS from 2e-11 to 2e-9 as h0 falls from
# 1e-3 to 1e-5, which is the signature of a noisy numerical derivative, not a
# wrong analytic one.
def numgrad(f, x, h0=1e-3):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x); e[i] = 1.0
        est = []
        for h in (h0, h0 / 2):
            est.append((f(x + h * e) - f(x - h * e)) / (2 * h))
        g[i] = (4 * est[1] - est[0]) / 3      # Richardson
    return g


def rel(a, b):
    return np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-6))


n, K = 300, 6
X = np.column_stack([rng.binomial(1, .5, n).astype(float), rng.normal(size=n)])
w = rng.gamma(3, size=n)
Y = M.onehot(rng.integers(1, K + 1, n), K)
Yp = rng.dirichlet(np.ones(K), n)             # probability-weighted case

ok = True
for label, Yoh in (("indicator", Y), ("probability", Yp)):
    v = np.concatenate([M.alpha_to_d(np.array([1.5, .8, .1, -.7, -1.6])), [.3, -.4]])
    f = lambda z: M.po_nll_grad(z, X, Yoh, w, K - 1, 2)[0]
    r = rel(M.po_nll_grad(v, X, Yoh, w, K - 1, 2)[1], numgrad(f, v))
    print(f"  PO   ({label:11s}) max rel err {r:.2e}"); ok &= r < 1e-8

    v = np.concatenate([rng.normal(size=K - 1), rng.normal(size=(K - 1) * 2) * .3])
    f = lambda z: M.mr_nll_grad(z, X, Yoh, w, K - 1, 2)[0]
    r = rel(M.mr_nll_grad(v, X, Yoh, w, K - 1, 2)[1], numgrad(f, v))
    print(f"  MR   ({label:11s}) max rel err {r:.2e}"); ok &= r < 1e-8

    G = np.array([0, 0, 0, 0, 1.0])
    v = np.concatenate([M.alpha_to_d(np.array([1.5, .8, .1, -.7, -1.6])), [.3, -.4], [.5]])
    f = lambda z: M.cppo_nll_grad(z, X, Yoh, w, K - 1, 2, G, 0)[0]
    r = rel(M.cppo_nll_grad(v, X, Yoh, w, K - 1, 2, G, 0)[1], numgrad(f, v))
    print(f"  CPPO ({label:11s}) max rel err {r:.2e}"); ok &= r < 1e-8

# The TsCO factorisation itself: l1 + l2 from the split must equal the
# log-likelihood computed from the assembled unconditional probabilities.
print("\n  TsCO split == unconditional likelihood:")
for C in (3, 4, 5, 6):
    for s2 in ("mr", "po"):
        ll, (t1, t2), _ = M.fit_tsco(X, Y, w, K, C, s2)
        P = M.tsco_probs(t1, t2, X, K, C, s2)
        direct = float(np.sum(w[:, None] * Y * np.log(np.clip(P, 1e-300, None))))
        d = abs(ll - direct)
        print(f"    C={C} stage2={s2}: |diff| = {d:.2e}"); ok &= d < 1e-8

print("\n" + ("GRADIENTS OK" if ok else "GRADIENT CHECK FAILED"))
