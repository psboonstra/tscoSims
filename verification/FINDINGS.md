# Type I error of the joint LRT for Z under a misspecified TsCO model

> ## !! CORRECTION, 2026-09-15
>
> **The CPPO result in this document is retracted.** This file originally
> reported that CPPO's type I error doubles (0.10 vs 0.05) at n = 200 because
> its deviation parameter separates in ~5% of datasets. That was a bug in
> `models.py`, not a property of CPPO.
>
> A CPPO fit must keep the cumulative probabilities ordered, which with the
> departure at the last cutpoint means `gamma >= alpha_{K-1} - alpha_{K-2}`.
> `fit_cppo` was unconstrained, so the optimiser drove gamma to -7000, making
> the second-highest category's fitted probability **negative**. That category
> has probability 0.03, so it is often EMPTY in the exposed arm — and an
> invalid probability in an empty cell is never evaluated against an
> observation, so the log-clip never bites. The fit bought likelihood for free
> by assigning negative mass to an unoccupied cell.
>
> Corrected (n = 200, 2500 reps): PO control 0.0504, CPPO **0.0632**
> (unconstrained/buggy: 0.1072). Fixed by reparameterising
> `gamma = (alpha_{K-1} - alpha_{K-2}) + exp(t)`.
>
> **What survives**: in 6.1% of datasets the exposed arm has no observation in
> that category, and in 100% of those the constrained MLE sits ON the boundary
> of the parameter space, where chi2_2 is the wrong reference — those
> replicates reject at 28%. Net effect +0.013 over the PO control: the same
> order as the mild liberality of PO|C|MR and MR, not a headline result.
>
> Everything else below is unaffected: PO, MR and TsCO are parameterised so
> they cannot leave their parameter spaces. Read every CPPO number below as
> superseded.

Work-queue item 5 from the tscoSims design plan, which flagged this as
"the result that could sink the method if a referee finds it first" and asked
for a small-scale check before committing to the full design.

**Headline: the feared result does not happen. A different, smaller one does,
and it hurts CPPO — the competitor — far more than TsCO.**

---

## Setup

Truth: proportional odds, `K = 6`, ECMO control distribution
`{0.27, 0.06, 0.06, 0.11, 0.03, 0.47}` holding **marginally over W**, binary
`Z ~ Bern(0.5)` with `beta_Z = 0`, continuous `W ~ N(0,1)` with `beta_W = 0.5`
entering proportionally. Under this truth TsCO and MR are misspecified (the PO
truth induces a *nonlinear* covariate effect in the TsCO stages); **PO and CPPO
are correctly specified** and serve as controls.

All of this was done in an independent Python implementation rather than in R,
because CRAN is blocked by the egress policy on this machine — see *Caveats*.

## 1. Asymptotics: no misspecification problem

Under misspecification the LRT of `H0: psi = psi*` converges not to `chi2_q`
but to `sum_j lambda_j chi2_1`, with the `lambda_j` the eigenvalues of
`H_{psi psi . lambda} V_{psi psi}` (`V = H^-1 J H^-1`). So the usual
chi-square is valid iff the information equality holds **on the Z block after
profiling out the nuisance parameters** — not the full information equality.

Two prerequisites were checked rather than assumed:

- `psi* = 0`, i.e. the KL projection really does put zero on Z, so the LRT is
  testing the hypothesis it claims to. Verified to `1e-9`. (Argument: under the
  null Z is independent of `(Y, W)` and the model class is invariant under
  `Z -> 1 - Z`, so a unique minimiser must be a fixed point of that involution.)
- The eigenvalues are invariant to the cutpoint reparameterisation used for
  optimisation, which touches only nuisance parameters.

A hand derivation predicts the answer and the numerics confirm it. At `psi = 0`
the score for `psi` is `z * g(y, w)` with `g` lying in the **span of the
intercept scores** for every model here. Writing `g = P'h` and using
`Z` independent of `(Y, W)`, the blocks collapse to

    lambda_j = eig( (P'N P) (P'C P)^-1 )

with `N` and `C` the outer-product and Hessian informations. So the weights are
1 exactly when the information equality holds **in the intercept directions**,
and the discrepancy there reduces to
`-2 Sym{ Cov_W( p(W) - pi(W), pi(W) ) }` — the covariance between the
projection residual and the fitted probabilities. The intercept score's
first-order condition kills the *mean* of the residual and the slope's kills its
correlation with `W`, but neither kills its correlation with the nonlinear
`pi(W)`, so it is nonzero in general.

**It is, however, negligible.** Sweeping `beta_W` far past anything plausible
(`beta_W = 2.5` is an odds ratio of ~12 per SD of W):

| fitted model | beta_W | KL(truth→class) | q | sum lambda | min lam | max lam | true size of a nominal 5% test |
|---|---|---|---|---|---|---|---|
| PO&#124;5&#124;MR | 0.5 | 1.2e-04 | 3 | 3.00000 | 0.9999 | 1.0001 | 0.0501 |
| PO&#124;5&#124;MR | 2.5 | 5.2e-03 | 3 | 3.00019 | 0.9880 | 1.0168 | 0.0501 |
| PO&#124;3&#124;MR | 2.5 | 6.8e-03 | 5 | 4.99816 | 0.9672 | 1.0266 | 0.0500 |
| MR | 2.5 | 5.6e-03 | 5 | 4.97428 | 0.9628 | 1.0195 | 0.0490 |
| PO *(control)* | 2.5 | 0.0 | 1 | 1.00000 | 1.0000 | 1.0000 | 0.0499 |

The correctly-specified PO control returns eigenvalues of exactly 1.00000 at
every `beta_W`, which is what certifies the numerics — an earlier version of
this table showed PO at 0.975 and that turned out to be tail underflow
corrupting the information matrices, not a finding.

**Conclusion: asymptotically the distortion is at most ~0.001 in size, at
nuisance effects nobody would defend. This is not a threat to the paper.**

## 2. Finite sample, n = 200: a different story, and not about misspecification

6000 replicates, paired (every method sees the same datasets, so rejection
indicators are strongly correlated and raw rates move together — at n = 1000 all
seven sit near 0.056 *including* correctly-specified PO). PO is the control;
what matters is each method minus PO on the same replicates.

| method | reject | vs PO (paired) | z |
|---|---|---|---|
| PO *(control, correctly specified)* | 0.0512 | — | — |
| PO&#124;5&#124;PO | 0.0513 | +0.0002 | 0.1 |
| PO&#124;6&#124;MR | 0.0497 | −0.0015 | −0.5 |
| PO&#124;5&#124;MR | 0.0615 | +0.0103 | 2.9 |
| PO&#124;3&#124;MR | 0.0662 | +0.0150 | 3.9 |
| MR | 0.0650 | +0.0138 | 3.6 |
| **CPPO** | **0.1012** | **+0.0500** | **12.4** |

By n = 500 everything is back to nominal (CPPO 0.0546, all TsCO variants
0.049–0.052).

### Two distinct mechanisms, separated by the tail

A distortion caused by a *fixed fraction of non-identified fits* plateaus —
those replicates reject at every level. An ordinary imperfect chi-square
approximation shrinks with the nominal level. Observed rejection rates at
n = 200, 4000 replicates:

| method | 0.10 | 0.05 | 0.01 | 0.001 | 0.0001 |
|---|---|---|---|---|---|
| PO | 0.1035 | 0.0563 | 0.0143 | 0.0013 | 0.0000 |
| PO&#124;5&#124;PO | 0.1022 | 0.0512 | 0.0110 | 0.0003 | 0.0000 |
| PO&#124;6&#124;MR | 0.1123 | 0.0522 | 0.0105 | 0.0010 | 0.0000 |
| PO&#124;5&#124;MR | 0.1172 | 0.0612 | 0.0143 | 0.0008 | 0.0000 |
| PO&#124;3&#124;MR | 0.1210 | 0.0663 | 0.0132 | 0.0010 | 0.0003 |
| MR | 0.1205 | 0.0653 | 0.0132 | 0.0008 | 0.0003 |
| **CPPO** | **0.1522** | **0.1017** | **0.0610** | **0.0512** | **0.0503** |

Every method except CPPO has a well-behaved tail. Their mild liberality at the
5% point scales with the number of free Z-parameters over sparse cells
(`PO|3|MR` and `MR` have q = 5, `PO|5|MR` has q = 3, the two clean ones have
q = 2) and is the ordinary sparse-data chi-square approximation, shared with the
correctly-specified PO control at the 10% level.

**CPPO plateaus at ~0.051 all the way to nominal 1e-4 — a 500-fold
exceedance.** Diagnosis: in **5.35%** of replicates the deviation parameter
`gamma` diverges (`|gamma| > 20`; median `|gamma|` over all replicates is 0.071).
Those are separated fits, and they reject at every level. Ruled out as an
artefact of this implementation: a 3-start refit improved the CPPO
log-likelihood in **0.00%** of replicates and changed the LRT by at most 0.0000.

The mechanism makes sense. With `G = (0,0,0,0,1)`, `gamma` is identified only by
the *contrast* between the top cutpoint's Z effect and the others' — and the
other categories have probabilities 0.06, 0.06, 0.11, 0.03. The information
about `gamma` is exactly where the ECMO distribution is thinnest.

## What this means for the manuscript

**Amended by section 3 below**, which reran the n = 200 cells through the shipped
R code: the single-number size for CPPO should be replaced by the flagged /
unflagged split reported there.

1. **Item 5 can be closed.** Prespecifying C is still worth insisting on, but
   the justification is not type I error under misspecification — that cost is
   ~0.001. Do not build an argument on it.
2. **This is a result *for* TsCO, and a stronger one than expected.** The plan
   framed the asymmetry as "CPPO needs both S and G; TsCO needs one integer with
   a subject-matter anchor," and said it should be demonstrated rather than
   asserted. Here is the demonstration, in the inferential currency referees
   care about: at the manuscript's own ECMO design and n = 200, CPPO's extra
   shape parameter is non-identified in ~5% of datasets and doubles its type I
   error, while TsCO with `C` at the clinically anchored dichotomy
   (`PO|6|MR`, `PO|5|PO`) holds nominal size exactly.
3. **Prefer a PO stage 2 when the upper partition is thin.** `PO|5|PO` and
   `PO|6|MR` (q = 2) are clean; `PO|5|MR` (q = 3) and `PO|3|MR` (q = 5) are
   mildly liberal at n = 200. This is a concrete, defensible recommendation.
4. **`fit_ok` / `test_ok` / `pred_ok` are load-bearing, as the plan
   anticipated** — and for CPPO specifically. A real backend (VGAM) would hit
   `maxit` on those 5% of datasets and warn; this implementation does not stop.
   The R simulation must *record and report* those replicates rather than
   silently scoring them, and the manuscript should say how they were handled.
   How the distortion manifests in practice depends on whether the analyst
   notices the warning.
5. **Note for the full design**: at n = 1000 all seven methods sat near 0.056
   together. That is shared replicate variation, not inflation. Report type I
   error paired against a correctly-specified control, exactly as
   `process_main_results.R` already does for accuracy.

## 3. Reproduced in R (2026-09-15) -- and the aggregate number is a mixture

R 4.3.3 with VGAM 1.1.9, rms 6.7.1 and `tsco` 0.0.0.9000 were installed from the
**Ubuntu archive** (`r-cran-vgam`, `r-cran-rms`), which the egress policy allows
even though CRAN does not. So everything below is the shipped R code -- the
harness's own `methods/*.R`, `score_method()` and seeding scheme -- not the
Python harness. The only change was relaxing `Depends: R (>= 4.4.0)` to 4.1.0 in
a staged copy of `tsco/DESCRIPTION`.

### 3.0 The DGM and the CPPO fitter check out

`check_dgm.R`: **ALL DGM CHECKS PASSED**. Tables 3, 4 and 5 reproduce to
`5e-4`, i.e. exactly the rounding of the three decimals printed.

`cppo_vglm_valid()` was written against slot names that were never verified in R
because they sit inside `tryCatch`, so a wrong name would have degraded silently
to "assume valid". All three exist: `fit@fitted.values` (200 x 6, named by
level), `fit@iter`, `fit@control$maxit` (= 30). The check also discriminates:
injecting a negative fitted value or setting `iter = maxit` both return `FALSE`.
`stat`/`df`/`p_value` come back as named length-1 numerics on both paths and
`score_method()` accepts them. On a dataset where the constraint is slack the
VGAM and direct engines agree to `2.5e-10`.

### 3.1 VGAM does not run away. It gives up.

The open question was what `VGAM::cumulative(parallel = FALSE)` does on the
datasets where the unconstrained Python fitter drove gamma to -7000. Answer, over
600 null replicates at n = 200:

    VGAM returns a usable LRT        564   0.940
    log-likelihood is NaN -> NA      25    0.042
    VGAM throws an error             11    0.018

Where VGAM returns a number it **is** the constrained MLE: over those 564
replicates `|LRT_vgam - LRT_direct|` has median `7.8e-10` and maximum `2.8e-5`,
and there is no replicate where the two land on opposite sides of the critical
value. VGAM overshoots the boundary by a trivial amount when it overshoots at
all (minimum fitted probability `-1.6e-3`, median `-1.2e-4`), and when it does,
`fit@criterion["loglikelihood"]` is NaN, so `vglm_lrt()` returns NA and the
replicate would previously have been **dropped from the rejection rate**, not
counted as a spurious rejection.

So the retracted 0.107 was an artefact of the Python fitter and nothing else.
VGAM's failure mode is missing data, and it is the *opposite* direction of bias:
the 11 errored replicates reject at 6/11 under the constrained fit.

The errors are two distinct kinds, and both occur only when the exposed arm has
no observations in category 4:

- `NA/NaN/Inf in foreign function call (arg 1)` (8/11), full model only;
- `constraint matrix has too many columns` (3/11), **both** models -- these are
  datasets where category 4 is empty in *both* arms, so VGAM drops the level,
  M becomes 4, and the 5-column constraint matrix no longer fits.

### 3.2 The headline: 0.069 is a 90/10 mixture, and the harness already
### separates the two parts

2500 replicates, `null`, n = 200, the harness's own seeds (arrays 1 and 2):

    po          0.0536   (control, correctly specified)     SE 0.0045
    tsco_popo   0.0556   paired diff +0.0020   z = +0.43
    mr          0.0628   paired diff +0.0092   z = +1.55
    cppo        0.0692   paired diff +0.0156   z = +2.90

`po` sits at nominal, `tsco_popo` is indistinguishable from it, `mr` is mildly
liberal exactly as the Python harness said. `cppo` at 0.0692 is significantly
liberal against the paired control -- but the aggregate is not a property of the
CPPO LRT. Splitting on whether `fxn_cppo` flagged anything in `warnings`
(**superseded by 3.7: the correct split is on whether category 4 is empty in
either arm, which also moves the well-behaved size from 0.0373 to 0.0303**):

    unflagged   2254   0.9016   size 0.0373
    flagged      246   0.0984   size 0.3618
                                mixture: 0.9016(0.0373) + 0.0984(0.3618) = 0.0692

On the 2254 unflagged replicates CPPO is **conservative**, and significantly so:
paired against `po` on the same datasets, 0.0373 vs 0.0559, difference
**-0.0186**, paired z = **-4.22**. All of the apparent liberality lives in the
9.8% the harness already marks.

The flag is driven by sparsity, not by the fit: the boundary tag fires on exactly
the 155 replicates (6.20%) where the exposed arm has no observations in category
4 -- 155/155 and 0/2345, a perfect one-to-one match, confirming the Python
harness's 153/153 at the level of individual replicates.

### 3.3 Neither boundary diagnostic covers the bad set alone

Two independent signals exist and they overlap only partly:

                          our boundary tag
    VGAM "intersecting"   FALSE   TRUE
              FALSE        2254    104
              TRUE           91     51

The union is 246 -- exactly the flagged set, and exactly the complement of the
2254 well-behaved replicates. Neither signal alone suffices: 104 replicates are
caught only by the fitted-probability check, 91 only by VGAM's own
"nonparallelism has resulted in intersecting linear/additive predictors"
warning, and the size in those two cells is 0.41 and 0.40.

`cppo_vglm_valid()` currently inspects slots only and never looks at the warnings
`safe_fit()` has already captured. The cost of that is small but real: **17
replicates (0.68%) where VGAM warned, the validity check passed, no refit
happened, and the VGAM LRT was returned with no cppo-specific tag. They reject at
0.4118.** Folding the VGAM warning text into the invalidity criterion moves those
17 into the flagged set and takes the unflagged size from 0.0401 to 0.0373.

### 3.4 A fully empty outcome category breaks three methods three ways

Category 4 is empty in *both* arms in 9/2500 replicates (0.36%) at n = 200, and
the harness handles that inconsistently:

- `tsco_popo` returns `fit_ok = FALSE` with **zero warnings**, so `test_ok` is
  FALSE and those 9 replicates are silently dropped -- its rejection rate is over
  2491 replicates, not 2500. This is exactly the `safe_fit` blind spot noted in
  the open-thread file: a method that cannot fit scores as "not testable" and
  vanishes from the denominator.
- `mr` silently drops from df = 5 to df = 4 and rejects 2/9, with no tag marking
  the change of hypothesis.
- `cppo` hits the constraint-matrix error, refits by constrained direct ML, and
  rejects 0/9.

At n = 200 under `null` this is a 0.36% effect. It will be much larger under
`cppo_alt`, whose smallest expected cell is 0.5 per 100 per arm against 2.8 for
`null`.

### 3.5 Other nominal levels

    alpha    po       cppo     mr       tsco_popo
    0.10     0.1036   0.1340   0.1232   0.1088
    0.05     0.0536   0.0692   0.0628   0.0558
    0.01     0.0108   0.0176   0.0168   0.0108

### 3.6 What to do with this

1. **Do not report CPPO's size as a single number at n = 200.** Report the
   flagged fraction beside it. "0.069 overall; 0.037 on the 90% of replicates
   with no fitting diagnostic, 0.36 on the 10% with one" is the true statement
   and is more useful than either half.
2. The right reference distribution on the flagged subset is a mixture of
   chi-squares, not chi-square_2. That is still not derived.
3. Decide whether `cppo_vglm_valid()` should consult `safe_fit()`'s warnings
   (section 3.3). The effect is 17/2500 replicates.
4. Fix the fully-empty-category path so the three methods agree (section 3.4),
   and make `safe_fit` record non-convergence rather than only errors.

### 3.7 Correction to 3.2 and 3.3: the right cut is the empty cell, not the warning

Splitting on the `warnings` column (3.2) works but is the wrong variable. The
thing that determines whether the CPPO LRT is trustworthy is a property of the
**data**, available before any fit: whether category 4 is empty in either arm.

    where category 4 is empty      n      share   bdry tag  VGAM "intersect"  any flag   size
    neither arm                  2210    0.8840          0                 0         0   0.0303
    exposed arm (A = 1)           146    0.0584        146                51       146   0.3630
    UNexposed arm (A = 0)         135    0.0540          0                91        91   0.3926
    both arms                       9    0.0036          9                 0         9   0.0000

    mixture: 0.8840(0.0303) + 0.0584(0.3630) + 0.0540(0.3926) + 0.0036(0) = 0.0692

Three things follow.

**The problem is symmetric in the arms, and the code only checks one of them.**
An empty cell costs the same either way -- 0.363 exposed, 0.393 unexposed -- but
`fxn_cppo` looks only at `mu[dat$A == 1, K - 1L]`, so the boundary tag fires
146/146 on exposed-empty and 0/135 on unexposed-empty. That is why the tag and
VGAM's warning disagree (3.3): they are detecting two different boundaries.

**There are two boundaries, not one.** With the departure at the last cutpoint
the admissible region has two active faces:

- `gamma >= alpha_{K-1} - alpha_{K-2}` binds when the **exposed** cell is empty.
  `cppo_direct_fit` already measures the distance to it as
  `dist_to_boundary = exp(t)`.
- `alpha_{K-2} > alpha_{K-1}`, i.e. the cutpoint spacing itself, binds when the
  **unexposed** cell is empty. This face is shared with the reduced PO model and
  is *not* measured anywhere.

Verified on individual replicates (`verification/r_dossier.R`):

    i    cat-4 exp/unexp   gamma     exp(t)      min(-diff(alpha))   LRT
    1        1 / 2        -0.0439   4.58e-02     8.97e-02            3.54
    12       0 / 3        -0.1374   2.43e-07     1.37e-01            5.91
    17       0 / 4        -0.1717   5.69e-08     1.72e-01            6.11
    82       5 / 0        +0.2190   2.19e-01     2.23e-07            8.30
    56       2 / 0        +0.1049   1.05e-01     3.20e-07           14.62
    139      2 / 2        -0.0106   8.29e-02     9.35e-02            3.30

`min(-diff(fit$alpha))` is the missing diagnostic and `cppo_direct_fit` already
has `alpha` in hand, so this is two lines, exactly parallel to
`dist_to_boundary`. Checking both faces, in both arms, replaces all of the
warning-string parsing in 3.3 with an exact criterion.

**The well-behaved size is 0.0303, not 0.0373.** The unflagged set of 3.2
contains 44 unexposed-empty replicates that no diagnostic caught, and they
reject at ~0.39. On the 2210 replicates with no empty cell anywhere, CPPO
rejects at 0.0303 against a nominal 0.05 -- more conservative than 3.2 reported,
and the honest number for the regime where the asymptotics apply.

The `both arms empty` row is a curiosity worth one sentence in the manuscript:
there the CPPO statistic equals the PO statistic exactly (at i = 133 both are
0.989) because the deviation parameter has nothing to attach to, yet it is
referred to chi-square_2 instead of chi-square_1, so those replicates can never
reject.

### 3.8 Worked examples, by behaviour

All with `array_id = 1`, so `data_seed = 20360710 + i`. Run
`verification/r_dossier.R` (`IVALS=...`) for the full per-replicate report.

    i = 1     nothing wrong. VGAM converges in 5 iterations, both engines give
              LRT 3.5405, no warnings, no tag.
    i = 17    exposed cell empty. VGAM's fit is ACCEPTED by cppo_vglm_valid
              (min fitted prob 2.3e-11 > -1e-10) and emits no warning at all,
              so only the boundary tag fires. LRT 6.1133, p = 0.0470 -- a
              rejection produced entirely by the boundary.
    i = 12    same, one step milder: p = 0.0520, just misses.
    i = 82    UNexposed cell empty (5 exposed, 0 unexposed). gamma = +0.219 is
              interior; the cutpoint gap is 2.2e-7. Currently gets NO cppo tag.
              LRT 8.3014, p = 0.0158, rejects, while po p = 0.2453.
    i = 56    UNexposed cell empty AND VGAM's fitted probabilities go to
              -1.6e-3, so the log-likelihood is NaN, the LRT is NA, and the
              refit fires -- but with no boundary tag, because the check looks
              at the wrong arm.
    i = 123   exposed cell empty; VGAM ERRORS with "NA/NaN/Inf in foreign
              function call (arg 1)". Refit gives LRT 9.0471, p = 0.0109.
              Without the fallback this replicate is dropped.
    i = 133   category 4 empty in BOTH arms; VGAM errors with "constraint
              matrix has too many columns" on the full AND reduced model.
              cppo refits (LRT 0.989 = the PO statistic, p = 0.6099), mr
              silently drops to df = 4, and tsco_popo returns fit_ok = FALSE
              with zero warnings and disappears from its own denominator.
    i = 104   exposed cell empty, VGAM half-steps to a NaN log-likelihood, refit
              gives LRT 3.0797 (p = 0.2144) while tsco_popo rejects at 0.0084
              and mr at 0.0402 on the same dataset.

## Caveats

- **Sections 1 and 2 are an independent implementation, not the shipped R code.**
  CRAN is blocked by the egress policy, so `tsco`/VGAM/rms could not be
  installed from there. Section 3 lifts this caveat for the n = 200 type I error
  results: the Ubuntu archive *is* reachable and ships `r-cran-vgam` and
  `r-cran-rms`, so those cells were rerun through the shipped R code and agree.
  The theory in section 1 is still Python only. Everything above is a from-scratch Python implementation of the
  same likelihoods. That is a real strength for the *theory* questions (the
  sandwich eigenvalues are not something VGAM would hand over, and agreement
  would not have been circular) but it means **these numbers have not been
  reproduced through `tsco::tsco()` itself**. The natural next step is to rerun
  the n = 200 CPPO and `PO|5|PO` cells in R and confirm they match.
- The implementation is validated: analytic gradients checked against
  Richardson-extrapolated numerical ones (`1e-11`), the TsCO two-stage
  factorisation checked against the assembled unconditional likelihood, MLE
  recovery at n = 400,000, and population KL projections that come out at
  machine zero exactly where theory says they must.
- The `beta_W` sweep stops at 2.5; beyond that, quadrature nodes in the far
  tails underflow and corrupt the information matrices. The PO canary row
  detects this.
- One-sided in another respect: only a **PO truth** was examined. Type I error
  under a CPPO-true or TsCO-true null is a different calculation.
