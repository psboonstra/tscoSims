# Independent verification harness for the TsCO type I error probe

A from-scratch Python implementation of PO, MR, CPPO and TsCO (PO|C|MR and
PO|C|PO), written to answer work-queue item 5 of the tscoSims design plan.

It exists separately from the R simulator for two reasons. First, the
asymptotic question needs the sandwich `J H^-1` at the population KL
projection, which VGAM will not hand over. Second, agreement between this and
`tsco` is evidence about the *method* rather than about a shared convention;
CRAN being unreachable from the machine this was written on forced the issue
but did not create it.

See `FINDINGS.md` for results and caveats.

## Files

| file | what it does |
|---|---|
| `models.py` | likelihoods, analytic gradients, fitters |
| `dgm.py` | the null DGM (K=6, ECMO marginal, binary Z + continuous W) |
| `sandwich.py` | weighted-chi-square null: H, J, effective information, eigenvalues |
| `test_gradients.py` | every analytic gradient vs Richardson-extrapolated numerical |
| `validate.py` | KL projections against facts known independently of the code |
| `nesting.py` | when is PO actually nested in TsCO (saturation, not discreteness) |
| `analytic_null.py` | psi* = 0 check and eigenvalues by design |
| `stress.py` | eigenvalues vs nuisance effect size, with a PO canary row |
| `sim_typeI.py` | finite-sample rejection rates |
| `paired.py` | n=200 rejection rates paired against the PO control |
| `cppo_diag.py` | is CPPO's distortion real or an optimiser artefact |
| `tail_diag.py` | tail signature: non-identification vs chi-square approximation |

Run `test_gradients.py` and `validate.py` first; nothing else is meaningful
until they pass.
