Frozen Maxwell and profile-history loads now remain separate from the
unknown-dependent bulk residual until both global vectors have passed through
constraints and MPI assembly. BV remains unknown-dependent. No equation,
quadrature rule, initialization/history semantics, support, full I_h,
convergence criterion, tolerance or iteration budget changed. The previously
failing Stage-I case now genuinely converges after one Newton update, and
one-/two-rank represented-increment and nonzero-slip consistency checks pass.
The twelve focused coupling/load/lifecycle checks and the fresh restart pair
pass. The short K1 and unchanged K2-64/K2-128 replays reach 1 s with positive
convergence, support and actual slip-normalization checks passing. At 1 s,
the mean-removed left/right endpoint traction discrepancies are about 64/52
times smaller. Whole-fault traction discrepancy improves by about twofold,
but the revised initial central traction discrepancy is larger and raw bulk
stress differences remain. The coarse K1 initial raw-stress limitation is
unchanged from its saved baseline. No larger campaign was run; Gate K2 and
the separate bulk-history transfer concern remain unresolved.

# Separate global accumulation: bounded verification

## Implementation and invariants

The approved correction is in `stage_K2_residual_consistency_audit.md`.
The numerical contract is recorded in `current_design.md` and
`specification.tex`. The only changed bulk arithmetic is the order in which
independent weak-load contributions are combined:

- The ASPECT Newton RHS is -R_bulk. Consequently its slip term is +BV and
  its frozen stress term is -beta*tau_old + 2*kappa*history*S; the physical
  constitutive stress still contains +beta*tau_old - 2*kappa*(chi*V+history)*S.
- `CopyData::StokesSystem::local_rhs` contains the ordinary unknown-dependent
  residual and +BV. `local_frozen_fault_rhs` contains the frozen
  -beta*tau_old + 2*kappa*history*S weak load.
- The cell assembler supplies -beta*tau_old at every bulk QP, including
  outside fault support. The profile term is present only where associated.
- `assemble_stokes_system()` constrains and MPI-compresses the two vectors
  independently, then adds the completed global frozen vector. It owns that
  scratch vector for one assembly only; there is no persistent load/history
  cache or additional simulator lifecycle state.
- Newton bases and non-committing trials use this same path. Homogeneous
  perturbation constraints, physical base lifting, pressure conventions,
  surface quadrature, B/G/K_V and history-publication ordering are unchanged.

The production interface addition is one internal CopyData vector. The
assembler's documented additive operation now explicitly names its two
destinations. Direct assembler tests check both accumulators, and the
nonuniform/constant frozen-stress test reads the frozen destination.

The opt-in diagnostic snapshots A, retains B/G/K_V, and uses owned copies of
both represented bulk vectors before subtraction. Unknown-dependent and
frozen channels are evaluated separately. An additional positive, nonuniform
V probe is discarded unconditionally and verifies nonzero B action with
bitwise unchanged frozen loads. Its assertions are regression checks, not
nonlinear stopping allowances. The test uses the existing absolute-row-sum
precision helper for nearly cancelling block actions.

## Residual-consistency evidence

Artifacts are under
`benchmarks/reconstructed_fault/uniform_shear/nonuniform/global-accumulator/`.

| Quantity at first represented Newton trial | One rank | Two ranks |
|---|---:|---:|
| Velocity affine mismatch | 6.6716594e-20 | 6.6302089e-20 |
| Scaled-continuity affine mismatch | 5.2867670e-21 | 5.6962641e-21 |
| Frozen-load change | 0 | 0 |
| Nonzero-V B-action norm | .006342515386 | .006342677732 |
| Nonzero-V residual/action mismatch | 1.0764361e-17 | 1.0467890e-17 |
| Nonzero-V frozen-load change | 0 | 0 |
| Final converged bulk norm | 2.6704664e-14 | 2.3624791e-14 |
| Final bulk target | 5.8890184e-13 | 5.8723837e-13 |
| Final relative bulk norm | 4.5346546e-8 | 4.0230326e-8 |
| Final projected surface norm | 0 | 0 |

All eight surface vertices are active in this fixture. The separate V probe
is therefore essential: it does not mistake the actual zero dV for proof of
V dependence. Both executions satisfy the explicit final convergence guard,
not just a process-exit or lifecycle-message check. The independent surface
criterion and the fresh linear residual checks remain enabled.

The one-rank initial scale changes from 5.8889378102325262e-7 to
5.8890184368694637e-7 because the same fixed-scale formula now receives the
separately accumulated initial residual. This is not a new threshold or a
fitted precision allowance. The scale remains fixed within each solve, and
its existing initial-state precision allowance remains zero here. Initial
phase/cohesive projection records match the saved audit. Partition-dependent
roundoff in this very small, nearly equilibrated load is not eliminated by
claiming exact arithmetic; the corrected residual is affine-consistent.

## Focused tests and explained harness issues

Builds use `-j4`; only focused tests run. Geometry and Stage-I unit checks pass
815 assertions in 18 cases on one rank and on each of two ranks.

- `phase_field_fault_residual_consistency`: passes, 428.16 s.
- `phase_field_fault_residual_consistency_mpi`: passes, 273.75 s after fixing
  the diagnostic-only owned/ghost vector subtraction. The first failed probe
  is preserved in `consistency-two-owned-map-failure.log` and `consistency-two.log`.
- The unchanged positive `phase_field_fault_stage_i` solves successfully in
  both its initial and clean execution. The clean numerical run took 350.27 s.
  Its expected-output comparison then passes in 6.51 s without rerunning an
  already valid solve.

The Stage-I expected output was refreshed only after the numerical checks
passed and its diff was reviewed (`stage-i-golden-review.diff`). The old
reference described the obsolete ten-iteration unconverged phase initializer;
the current unchanged fixture converges in twenty iterations. It also lacked
the current linear/nonlinear diagnostics and contained the older initial
cohesive values. Sandbox socket warnings are not incorporated into the new
reference. The required trailing blank lines follow the test normalizer's
generated output format.

Rebuilding all lifecycle plugins also exposed a registration-name collision
from the preceding audit's shared positive-convergence connector. Its macro
now has a distinct test namespace so included Stage-J fixtures can register
their own connectors. Neither its assertions nor the failure/rollback tests
were weakened. Source build logs, failed setup attempts and clean reruns are
retained separately rather than overwriting the preceding audit evidence.

The additional focused batch passes **12/12**, total wall time 740.00 s:
changed loading (one/two ranks), condensed adiabatic actions (one/two ranks),
condensed dynamic-pressure actions, surface/bulk temperature separation
(one/two ranks), linear exhaustion, nonlinear rollback (one/two ranks), and
constant/nonuniform frozen-stress weak loads (one/two ranks).
Its exact command is:

```sh
ctest --test-dir build-pf-cpdi/tests --output-on-failure -R '^phase_field_fault_(frozen_stress(_mpi)?|condensed_(adiabatic(_mpi)?|dynamic)|stage_j_temperature(_mpi)?|stage_i_rollback(_mpi)?|linear_exhaustion|changed_loading(_mpi)?)$' -j2
```

`focused-tests.log` records each result and duration. The replay comparison
helper has two analytic tests for exact Q1 moments and separating an initial
offset from subsequent evolution (`analysis-tests.log`, both pass). It does
not introduce a new reference, alter sampled data, or replace weak traction
with a column average.

The fresh sequential two-rank restart pair also passes **2/2**: create
372.15 s, resume 128.58 s (500.74 s total). The creator exercises retained
timestep-zero histories and two real evolving-phase-field/history steps;
the resume compares with that freshly generated uninterrupted reference.
Raw logs are preserved as `restart-create-raw.log` and
`restart-resume-raw.log`, in addition to `restart-tests.log`.

```sh
ctest --test-dir build-pf-cpdi/tests --output-on-failure -R '^phase_field_fault_stage_j_restart_(create|resume)$' -j1
```

The independent reference tests pass unchanged (`reference-tests.log`). The
K1 analysis now labels its old point-rule initial-C reconstruction as a
legacy discretization comparison when domain weak data are present. It uses
the actual exported domain mass for endpoint support and does not claim an
independent domain initial-projection check from point samples. Replicated
surface weak records are checked, not summed across ranks; the corresponding
analysis tests pass (`profile-analysis-tests.log`).

## Short homogeneous K1 replay

The prepared 32x128-cell, half-second-step case reaches accepted times
0, 0.5 and 1 s. All three pass the explicit final nonlinear criteria.
Release one-rank cost is **48.96 s, 478764 KiB peak RSS** (about 468 MiB).
`k1_short.resources.json` records executable/plugin/input hashes and the exact
runner command; `k1_short.log` contains dimensional and normalized residuals.

The independent scalar reference is initialized once from retained initial
histories and then advances its own histories with the accepted loading and
timestep sequence. It uses full I_h. All V/Theta/C/slip and mean-stress errors
are below the existing resolved conditional allowances; the largest
error/allowance ratio is 0.07287 (initial V). No histories are reset from later
ASPECT output. The local Theta update error is zero at all three exports.

| Time (s) | V error (m/s) | Theta error (s) | Retained C error (Pa) | Mean stress error (Pa) | Raw stress RMS / maximum error (Pa) |
|---|---:|---:|---:|---:|---:|
| 0 | 4.59544e-8 | 0 | 0 | -.0583478 | .697651 / 2.29051 |
| .5 | 1.06209e-7 | -.00698901 | .000490035 | -.0310360 | .468134 / 1.54714 |
| 1 | 2.72930e-8 | -.00766352 | .000613518 | -.00820680 | .153112 / .509375 |

**The initial raw maximum does not meet the resolved pointwise allowance on
this coarse mesh.** The valid saved same-mesh baseline
`residual-floor/convergence/space32_dt05-errors.json` has initial RMS
.6976509635107532 Pa and maximum 2.290510981079933 Pa; the new values are
.6976509635107794 Pa and 2.290510981078114 Pa. Thus the short recheck preserves
the documented resolution-dependent initialization limitation; it does not
claim a new full Gate K1 pass. Both later raw-stress checks and every velocity,
V, Theta, C and slip check pass. The analysis' six-second completion flag
remains false by design (exit 2), not suppressed to label this short run a
full convergence campaign.

Independent I_h is 108.09809507227017 m; maximum projected relative I_h error
is 8.13e-11. The omitted fraction is 5.83028e-5. Maximum actual local slip
normalization error across all measured columns/times is 5.23649e-5, with
global ratio .9999476351293. Both separate 1e-4 requirements pass. The Q1
phase field is bitwise frozen across accepted times; boundary errors are at
most 4.07e-20 m/s. Actual weak surface RMS residuals are 6.12e-8, 5.99e-9 and
1.05e-6 Pa, independently matching the production mass-consistent norms.
The exported initial C differs from the old point-rule reconstruction by
only 1.15e-11 Pa in this homogeneous case.

Artifacts: `k1-short-analysis.json`, `k1-short-errors.json`,
`k1_short-measurements/`, `k1_short-errors/` (all raw QP errors and transverse
profiles), and ParaView-readable output in `k1_short/`.

## K2-64/K2-128: completed bounded replays

Both unchanged cases reach accepted times 0, .5 and 1 s. The explicit
convergence guard passes at each state. Each log contains 17 returned linear
directions, all passing fresh residual checks. The maximum fresh/target
ratios are .914904 (64) and .974822 (128). No Armijo exhaustion, failed
nonlinear acceptance, tolerance change or larger campaign occurs.

| Mesh | Wall time | Peak RSS | Maximum omitted fraction | Maximum local actual slip-normalization error | Full independent I_h (m) |
|---|---:|---:|---:|---:|---:|
| 64x256 | 271.05 s | 1329992 KiB (1.27 GiB) | 5.91396e-5 | 6.00643e-5 | 108.144833844 |
| 128x512 | 1651.82 s | 4494064 KiB (4.29 GiB) | 5.89728e-5 | 5.77115e-5 | 108.126657883 |

These are the separately approved provisional 1e-4 omitted-fraction
allowance (original target 1e-6) and unchanged 1e-4 actual-normalization
requirement, not a tail renormalization. Both pass at all 65/129 measured
phase columns, all 192/384 bulk-QP columns and all three accepted times.
Global actual-slip ratios are .9999399357276 / .9999422884605. The support
half-width remains .3088215939070757 m. Same-support comparisons do not
establish equivalence to the full normal profile.

| Mesh, time | Bulk residual | Bulk target | Velocity / scaled-continuity residual | Surface RMS (Pa) | Fixed surface scale (Pa) |
|---|---:|---:|---:|---:|---:|
| 64, 0 | 1.29720e-11 | 2.24373e-5 | 1.29451e-11 / 8.34526e-13 | 6.98682e-8 | 952.8932 |
| 64, .5 | 9.59320e-12 | 1.41364e-7 | 9.57393e-12 / 6.07783e-13 | 6.13858e-9 | 340.8027 |
| 64, 1 | 3.31869e-12 | 1.41476e-7 | 3.29261e-12 / 4.15224e-13 | 1.09255e-6 | 793.1020 |
| 128, 0 | 2.59157e-11 | 3.17312e-5 | 2.59022e-11 / 8.36472e-13 | 6.96412e-8 | 953.1565 |
| 128, .5 | 1.91907e-11 | 1.99961e-7 | 1.91813e-11 / 5.99602e-13 | 6.13302e-9 | 340.9883 |
| 128, 1 | 6.49489e-12 | 2.00187e-7 | 6.48829e-12 / 2.92713e-13 | 1.09225e-6 | 793.4819 |

The surface target remains 1e-8 times its fixed scale. Every weak-norm
reconstruction agrees with production to roundoff. The independent local
Theta-update error is zero in these exports. The fine total particle-domain
area ranges from .2499999948168 to .2500000019759 m2; minimum domain area is
2.60277e-7 m2. Actual full admitted domain mass is checked against the weak
mass matrix, without normal clipping or an assumption of periodic domains.
The Q1 phase field remains bitwise unchanged at all accepted times.

`k2-64-residuals.json` and `k2-128-residuals.json` retain every estimated/fresh
linear residual, dimensional nonlinear block residual, fixed scale and
accepted-update count. `k2_64-measurements/` and `k2_128-measurements/` retain
actual weak loads, mass matrices, represented traction and raw parent samples.

## Actual endpoint traction and weak moments

These compare the saved **old point-rule** 64/128 runs against the **approved
domain-rule** 64/128 replays, including the accumulator correction. The
arithmetic correction is independently verified above; this trajectory
comparison must not attribute every old/new difference to that correction.
The discrete quadrature revision and its changed initial projections matter.

For every accepted state, q_Gamma is the consistent representation obtained
by solving M q_Gamma = Q, where Q is the actual pre-publication surface shear
load. No normal-column bulk average or newly published parent stress is
substituted. The old-rule loads are reconstructed from valid saved parent
samples with the old documented point-volume rule. New-rule loads and M come
from production `surface_weak` exports. The diagnostic checks these weak
loads against the saved represented traction; it does not smooth or feed a
new stress field into mechanics.

Define e(s,t)=q_64(s,t)-q_128(s,t), and the anomaly e-mean_s(e). Absolute
left-endpoint values at 1 s are 1033.501472 / 1033.247673 Pa (old 64/128),
and 1033.502752 / 1033.252682 Pa (new). Their large mean offset remains:
mean_s(e)=.25012606 / .25012642 Pa (old/new). The nonuniform endpoint
discrepancy, not that initial-history-dominated offset, improves strongly:

| Time (s) | Old left / right anomaly error (Pa) | New left / right anomaly error (Pa) | Old / new whole-fault anomaly RMS (Pa) |
|---|---:|---:|---:|
| 0 | -1.08598e-4 / -1.08598e-4 | -2.47545e-5 / -2.47546e-5 | 5.81422e-5 / 9.40301e-4 |
| .5 | 9.63108e-5 / 9.62594e-5 | -3.10881e-5 / -3.11274e-5 | 2.71228e-5 / 3.58263e-4 |
| 1 | 3.67339e-3 / 3.66451e-3 | -5.71007e-5 / -6.99507e-5 | 4.92231e-4 / 2.48890e-4 |

At 1 s the endpoint anomaly errors are approximately **64.3 / 52.4 times
smaller**. The left endpoint's anomaly change from its own initial error is
.00378199 Pa (old) versus -.0000323463 Pa (new); the corresponding right
values are .00377311 versus -.0000451961 Pa. Thus the improvement is not
obtained by resetting the initial error to zero. Subtracting an initial
error snapshot is an accounting diagnostic, not proof of identical initial
histories or removal of their subsequent physical influence.

The interior control at s=.0625 m has old/new anomaly discrepancies
-1.63688e-5 / 5.74424e-6 Pa at 1 s. The center, s=.125 m, instead has
1.16053e-4 / 8.15705e-4 Pa. The new center discrepancy was already .00280399
Pa at initialization and falls to .00108917 Pa at .5 s. The largest new
anomaly error is central, not at the tips. This bounded observation supports
the targeted endpoint correction; it does **not** imply uniformly better
initialization or a completed spatial-convergence study.

The actual left-endpoint weak quantities at 1 s are:

| Rule / mesh | Sum_j M_0j (m2) | M_00 / M_01 (m2) | Q_0 (Pa m2) | R_0 (Pa m2) |
|---|---:|---:|---:|---:|
| Point / 64 | .002407448662 | .001593230913 / .000814217749 | 2.488103881 | 2.36506e-9 |
| Domain / 64 | .002410888541 | .001607258992 / .000803629549 | 2.491660414 | 2.36424e-9 |
| Point / 128 | .001198557493 | .000792309759 / .000406247734 | 1.238410107 | 1.18294e-9 |
| Domain / 128 | .001205444272 | .000803629497 / .000401814775 | 1.245528586 | 1.18178e-9 |

Relative to t=0, the old endpoint mass-row sums fall by .00142686 / .00571311
(64/128); the integrated values change by only about -5.43e-8 / -5.31e-8.
This is not exact zero or a frozen-volume assumption. Full live domains
remain in the integration measure; initial admitted area is .154296875 m2
at both resolutions, and the observed small area variations remain in M.
First and second moments are measured separately; no product of averaged
basis weights replaces M. All endpoints and both interior controls, all
three times, raw Q/R and adjacent mass entries are in
`endpoint-comparison.json`.

## Initialization and raw-stress differences

At each resolution, the old/new initial Q1 phase arrays are **bitwise equal**.
Initial particle IDs, coordinates, volumes, H and every Maxwell-stress
component are also identical. Full I_h is unchanged. The intended supplied
Theta peak remains 210 s; its realized consistent Q1 projection changes:

| Mesh | Old / new maximum Theta0 (s) | Maximum / RMS projection change (s) | Maximum initial C change (Pa) | Maximum initial V change (m/s) |
|---|---:|---:|---:|---:|
| 64 | 210.13202882 / 210.12821889 | .00380993 / .00130389 | 9.16e-12 | 1.90649e-8 |
| 128 | 210.03268652 / 210.03177951 | .000907014 / .000318105 | 3.74e-11 | 4.59508e-9 |

The between-resolution phase/profile initialization difference is retained:
phi(0)=.5997640169 / .5997110506 and mean C0=317.5137627 / 317.2451785 Pa.
Their initial C offset is .2685841665 Pa. It is not labeled a mechanical
error, nor removed by restarting either history from later output.
The initial traction-anomaly RMS discrepancy grows from 5.81422e-5 to
9.40301e-4 Pa under the new rule. This must remain visible even though the
late endpoint deterioration is much smaller.

The full coupled 128 run, at the **same support**, is the comparison run.
No independent per-vertex K1 solves are used. New 64-minus-128 RMS differences
are:

| Time (s) | V (m/s) | Theta (s) | Retained C (Pa) | Accumulated slip (m) |
|---|---:|---:|---:|---:|
| 0 | 1.14142e-7 | .0206435 | .268584 | 0 |
| .5 | 4.05995e-7 | .0304738 | .264720 | 2.02997e-7 |
| 1 | 8.05743e-8 | .0309349 | .262816 | 2.43224e-7 |

Unsmoothed bulk stress comparisons use the native FE cell polynomials at
fine raw QPs and the full Maxwell decomposition, including frozen old
stress. Their sampled maxima are not continuous L-infinity bounds:

| Time (s) | Old / new raw stress RMS (Pa) | Old / new sampled maximum (Pa) | Old / new along-fault-anomaly RMS (Pa) |
|---|---:|---:|---:|
| 0 | .283791 / .283826 | .916887 / .919147 | .00141416 / .00464175 |
| .5 | .233258 / .233260 | .666801 / .667034 | .000562124 / .000972729 |
| 1 | .250282 / .250288 | .471250 / .469357 | .0159966 / .0160780 |

Raw bulk stress discrepancy has **not** materially improved at 1 s.
The published-FE versus mechanically constrained bulk-history transfer
concern remains separate and unresolved. No history interpolator, stress
smoothing, initialization, live-domain construction or endpoint topology was
changed to address it in this patch. A two-resolution through-1-s replay
does not identify all remaining central/raw-stress errors or establish a
convergence order. The larger temporal/spatial campaign remains held for
review; no additional production correction is proposed from these data
alone.

## Artifacts and final scope

All new artifacts are under
`benchmarks/reconstructed_fault/uniform_shear/nonuniform/global-accumulator/`:

- `endpoint-comparison.json`: old/new actual weak endpoint/interior traction,
  first/second moments, initial-input equality, realized surface histories,
  total/anomaly errors and their changes from initial differences.
- `k2-64-vs128.json`: same-support full coupled spatial comparison, including
  raw bulk errors. `k2-64-rule-change.json` and `k2-128-rule-change.json`
  separately retain the same-mesh old/new comparison.
- `k2_64/solution.pvd`, `k2_128/solution.pvd`: bulk ParaView time series.
  Their `reconstructed_faults.pvd` files provide the fault time series.
- `k2_64-measurements/` and `k2_128-measurements/`: sampled surface profiles,
  actual weak-balance/M entries and separate raw parent diagnostics.
  Native bulk/phase/particle CSVs remain in each output directory.
- `*.resources.json`: exact commands, executable/plugin/input hashes,
  process status, wall time and peak RSS. All three replays use executable
  SHA256 `0559a029c7f137d9d8297c39311d18cb2080f798a48d62512da0ce42c2cd4908`.
  These are working-tree results, not a claim of a newly created commit.

The bounded correction and requested verification/replays are complete.
The geometry implementation and prior passing evidence are preserved.
The final log audit independently checks all nine accepted benchmark states,
all 51 returned fresh linear residuals, and unchanged bulk/surface merit
scales within each solve. All pass. Source whitespace checks pass; the only
full `git diff --check` warning is the Stage-I expected output's required
generated trailing blank line, already explained with its successful test
comparison above. No numerical check was relaxed to remove that warning.
**Gate K2 is not declared passed.** No larger campaign, true-normal-stress
branch, support change, tail renormalization, criterion change or additional
production correction was made. The full ASPECT suite was not run.
