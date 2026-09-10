# Accepted domain-rule K2.2 convergence sequence

**Stopped at user request, 2026-09-10.** No benchmark process remains running.
All three spatial runs complete through 2 s. Temporal accepted outputs reach
1.75 s (.25 timestep) and 1 s (.125 timestep); the interrupted next steps
are excluded. Partial total errors contract, but nonuniform temporal errors
do not contract uniformly. K2.2 is incomplete and K2.3 unstarted. See
`doc/reconstructed_fault/benchmarking/stage_K2_2_domain_convergence.md` for the
review and `temporal-partial-review.json` for saved-data checks. Preserve the
outputs; do not restart automatically. Resource estimates below are historical.

The user accepts the separate-global-accumulator correction and bounded
endpoint improvement as the tested baseline. Solver and quadrature development
are closed unless a new reproducible failure appears. The earlier point-rule
spatial sequence is historical evidence, not part of this convergence series.

Accepted executable SHA256:
`0559a029c7f137d9d8297c39311d18cb2080f798a48d62512da0ce42c2cd4908`.
Accepted plugin SHA256:
`40aac2f889e311f92cc6b69942b951fcafe4813dc5655460321f69f6e8d89880`.
HEAD is `a24c3623108e99240997141121a9d167fa119042` plus the saved working-tree
revision. `accepted-baseline.patch` and `accepted-source.tar.gz` preserve that
state without resetting unrelated changes. Passing evidence remains in
`../global-accumulator/` and the linked review documents.

## Inputs, reuse and prospective resources

Use the existing nonuniform 5% initial-Theta bump, fixed converged Q1 phase
field, prescribed normal stress, full I_h and unchanged support/physics,
initialization/history semantics and solver settings. The wrappers only
select the already planned mesh/timestep and a new output directory.
The original 1e-6 omitted-profile target remains documented; the approved
fixture-family allowance is 1e-4, separate from the unchanged 1e-4 actual
local/global slip-normalization requirement at every accepted time.

No restart/checkpoint files exist in the accepted new-rule 64/128 runs.
They are valid through 1 s but cannot supply an executable continuation.
Run complete new-rule 32/64/128 trajectories through 2 s and compare the
64/128 shared prefix against the accepted outputs. Do not reset histories
from saved outputs or mix the old point rule into adjacent-grid errors.

Before execution, one-rank Release estimates are 60--180 s / .5--1 GiB for
32, 350--600 s / 1.5--2 GiB for 64, and 2400--3600 s / 4--6 GiB for 128.
These follow the measured new-rule through-1-s costs of 49 / 271 / 1652 s
(the first was homogeneous). Run sequentially, with runner wall caps of
1200 / 1800 / 5400 s, not changed solver iteration budgets. Existing binaries
are hash-verified and reused; any necessary build uses -j4.

After spatial review, and only if evidence supports a resolved configuration,
run the existing .25 and .125 s timestep levels on that configuration, reusing
.5 s. Initial prospective fine-grid costs are 4000--6500 and 7000--11000 s,
respectively, with 4--6 GiB memory. Refresh these estimates from the completed
spatial costs before launching either expensive run. Do not run a full
mesh/timestep product. Stop before K2.3.

## Diagnostics and stopping conditions

Use actual pre-publication domain weak loads and mass matrices, not parent
point traction or normal-column averages as substitutes. Report exact Q1
arclength norms for total and mean-removed surface fields, endpoint/central
errors, V/Theta/C/slip, and native-FE raw stress errors. Retain the supplied
versus realized initial projection differences. Subtracting an initial error
is diagnostic accounting; it does not erase its physical influence.

Stop for a genuine solver/invariant failure, an unexplained plateau, an
allowance failure, or a required model/acceptance change. Do not repair
production merely because a discretization error remains. The bulk-history
transfer concern stays separate. A same-support reference does not establish
full-profile equivalence or Gate K2.

## Temporal launch decision

The completed spatial sequence supports the planned fine-grid time study;
see `doc/reconstructed_fault/benchmarking/stage_K2_2_domain_convergence.md`
for the numerical decision. Measured fine cost is 2571.12 s / 4488188 KiB.
Before launch, refreshed estimates are 4000--6500 s for .25 and 7500--11000 s
for .125, each 4.3--6 GiB. The two independent one-rank jobs run concurrently
in separate directories, rather than sequentially: 22 available CPUs,
approximately 20 GiB available RAM and 77 GiB free disk support this bounded
schedule. Combined memory budget is 12 GiB and elapsed estimate 2--3 hours.
Runner caps 9000/14400 s do not change solver iteration budgets. The .5 s
fine trajectory is reused; no extra mesh/timestep combinations are added.
