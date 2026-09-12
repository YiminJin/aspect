# K3: one additional normal-resolution level

## Preflight

This comparison changes only Box Y repetitions from 8 to 16 relative to the
completed periodic 32x256/fault32 case: 32x512 bulk cells, unchanged 32 fault
elements, common real timestep 0.375 s through 3 s. Initialization is performed
afresh from the same physical data. The verified precision-floor executable
and benchmark library are reused unchanged (hashes are in the run resource
record). No production edits, tolerance changes, support changes, or temporal
runs are part of this task.

The discriminating question is whether the remaining surface I_h feedback
error contracts under normal bulk refinement along with the phase/history
feedback, rather than exhibiting a plateau or qualitative change. Reuse the
independent continuum reference and existing 128/256 outputs, checking their
accepted time/loading sequence against this run before comparing.

Estimated one-rank Release cost: 480--650 s and about 2 GiB peak RSS, based on
the completed 256 run (238 s, 1.11 GiB). A 1200 s safety cap preserves a failed
or timed-out run without automatic retry. Opt-in timings remain enabled.
The separate omitted-profile and actual complete supported crack-strain
normalization bounds both remain 1e-4 at every accepted state.

Retain the existing profile max-error and raw quadrature stress definitions.
Cumulative feedback error is the change in the signed profile error from
timestep zero, not removal of the physical effect of initialization error.
Report error contraction as E_coarse/E_fine (>1 means improvement), for all
common times as well as the final state. Zero-error ratios are undefined.

## Result and decision

The one-rank 512 run completed initialization and eight real steps through
3 s in **542.711 s**, peak RSS **1,977,068 KiB (1.886 GiB)**. All nine state
guards and 37 fresh coupled-linear checks pass. No new solver, invariant,
support, geometry, history, or seam failure occurred. The executable, plugin,
and recorded production source hashes equal those of the verified 256 run.

The numerical result is mixed. Total phi and its accumulated feedback continue
to contract. Raw stress, H increments, and cumulative I_h feedback do not
contract from 256 to 512. Therefore this level does **not** establish ordinary
monotone spatial convergence of the remaining I_h feedback discrepancy.
There is no basis here for declaring Gate K3 passed. No additional refinement,
temporal run, production correction, or acceptance change was made.

### Final-time errors against the same independent reference

All quantities below are errors at 3 s. Profile errors are the existing
along-fault-mean transverse maximum errors, sampled at each level's own phase
nodes or stable-ID particle rows. Raw stress uses the existing volume-weighted
QP error of the accepted shear stress, reconstructed with the actual
mechanically constrained old FE history. No stress smoothing is applied.
"Increment" means step 8 minus step 7; "cumulative" means step 8 minus step 0.

| Error | 128 | 256 | 512 | E128/E256 | E256/E512 | Monotone? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Total phi, max | 2.17269e-4 | 7.69936e-5 | 4.19344e-5 | 2.822 | 1.836 | Yes |
| Raw stress, RMS (Pa) | 1.32702 | .330944 | .478560 | 4.010 | .692 | No |
| Raw stress, max (Pa) | 4.10665 | .869520 | .935234 | 4.723 | .930 | No |
| H increment, max (Pa) | 3.88036e-4 | 1.38742e-4 | 1.13666e-3 | 2.797 | .122 | No |
| phi increment, max | 1.76925e-6 | 8.74947e-7 | 4.44876e-7 | 2.022 | 1.967 | Yes |
| I_h increment (m) | 2.11134e-4 | 6.59965e-5 | 5.82342e-5 | 3.199 | 1.133 | Yes, weak final contraction |
| Cumulative H feedback, max (Pa) | 2.07514e-3 | 1.64572e-3 | 1.82201e-3 | 1.261 | .903 | No |
| Cumulative phi feedback, max | 9.80134e-6 | 5.22957e-6 | 3.32079e-6 | 1.874 | 1.575 | Yes |
| Cumulative I_h feedback (m) | 2.31488e-3 | 1.05255e-3 | 1.20014e-3 | 2.199 | .877 | No |
| Total H, max (Pa) | .0230953 | .0258859 | .0265140 | .892 | .976 | No |
| Total I_h (m) | .0345126 | .0109639 | .00706452 | 3.148 | 1.552 | Yes |

### Monotonicity over the accepted trajectory

Steps are k=0,...,8 at t=.375 k seconds. "Nonmonotone" below means the error
does not decrease through both refinements at that common time. This avoids
equating a favorable final increment with convergence of the entire trajectory.

| Error | Nonmonotone steps |
| --- | --- |
| Total phi; total I_h | None |
| Raw stress RMS/max; total H | All steps, including initialization |
| H increment | 4--8 |
| phi increment; cumulative phi feedback | 1 |
| I_h increment | 1, 5, 6, 7 |
| Cumulative H feedback | 2--8 |
| Cumulative I_h feedback | 1, 6, 7, 8 |

The full per-step numbers and contraction factors are in
`evolving/normal512-errors.csv`. For trajectory-max errors:

| Error | 128 | 256 | 512 | E128/E256 | E256/E512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Total phi | 2.17269e-4 | 7.92769e-5 | 4.31551e-5 | 2.741 | 1.837 |
| Raw stress RMS (Pa) | 1.32702 | .330944 | .486500 | 4.010 | .680 |
| H increment (Pa) | 2.09929e-3 | 1.19050e-3 | 1.40850e-3 | 1.763 | .845 |
| phi increment | 1.80696e-6 | 1.17794e-6 | 1.17865e-6 | 1.534 | .9994 |
| I_h increment (m) | 8.63808e-4 | 8.64945e-4 | 8.65665e-4 | .9987 | .9992 |
| Cumulative H (Pa) | 2.60698e-3 | 1.64572e-3 | 1.82201e-3 | 1.584 | .903 |
| Cumulative phi | 9.80134e-6 | 5.22957e-6 | 3.32079e-6 | 1.874 | 1.575 |
| Cumulative I_h (m) | 2.31488e-3 | 1.05255e-3 | 1.20014e-3 | 2.199 | .877 |

### I_h feedback interpretation and initialization

The independent I_h(3)-I_h(0) is .00330469510 m. Production gives
.00561957377, .00435724064, and .00450483469 m at 128, 256, and 512.
Relative cumulative feedback discrepancies are **70.05%, 31.85%, 36.32%**.
The fine error increases by about 14.0% from 256 to 512; it is not consistent
with a demonstrated single-power, decreasing-error spatial regime. This does
not prove that a spatially convergent limit cannot exist, nor identify a new
production defect. It means the requested three levels do not establish it.

The first real step already has a nearly resolution-independent cumulative
I_h error (.000863808, .000864945, .000865665 m). The 512 cumulative error is
smaller than 256 at .75--1.875 s, then larger at 2.25--3 s. Only the last-step
increment improves slightly (1.133 contraction), which cannot certify the
accumulated response. The retained independent-reference 2048/4096 check gives
about 8.59e-8 m difference in cumulative I_h, far below these discrepancies;
that existing evidence was reused, not recalculated or reset to production.

Initialization remains visible. Initial phi errors are 2.07468e-4, 7.83913e-5,
4.31551e-5. Initial raw-stress RMS errors are 1.08970, .277851, .446177 Pa:
the raw-stress nonmonotonicity already exists at t=0. Initial total-H errors
are .0230953, .0258859, .0265140 Pa, essentially the same maxima as at 3 s.
Initial total-I_h errors are .0368274, .00991132, .00826466 m. Each surface was
initialized normally from the same physical data; no old nodal values were
transferred. Subtracting the initial signed profile error in the cumulative
diagnostic does not remove its physical influence on subsequent mechanics.

Feedback remains present rather than disappearing: maximum cumulative H
changes are .335521, .310302, .346202 Pa and maximum cumulative phi changes
are 2.31516e-5, 1.91890e-5, 2.17273e-5. These maxima occur on level-dependent
sample sets; they are not substituted for the profile-error norms above.

### Acceptance/lifecycle evidence

The 512 maximum omitted h fraction is **5.903258e-5** and maximum actual
complete supported crack-strain normalization error is **5.875046e-5**;
both satisfy the unchanged 1e-4 requirements. Separate instantaneous, signed
history, and total supported integrals remain in every `guard_k.json`.
All final phase residuals satisfy the fixed mixed criterion; all mechanical
checks pass. Maximum fresh-linear residual/target is .986333; maximum weak
surface-balance RMS is 5.78407e-6 Pa. All 33 fault nodes remain free.

Stable-ID preceding-H handoff, noncommitting-probe restoration, irreversible H,
and exact selected Theta update checks pass (maximum Theta error 8.88e-16 s).
Fault geometry is unchanged. The run records 1400 periodic crossings; maximum
along-fault ranges of H, phi, I_h, C and V are respectively 7.728e-7 Pa,
3.898e-9, 8.305e-9 m, 2.287e-8 Pa, and 1.762e-12 m/s, well below the retained
reference feedback scales. No new qualitative seam/active-set behavior appears.

### Reproduction and changed files

From the repository root, the only simulation command was:

```sh
python3 benchmarks/reconstructed_fault/uniform_shear/evolving/periodic-domains/run_coupled.py --case spatial0375_n512_f32_periodic_floor --cap 1200 --library benchmarks/reconstructed_fault/uniform_shear/evolving/phase-floor/coupled-build/libuniform_shear.release.so --purpose 'Does normal refinement 128/256/512 at fixed fault32 and dt0375 contract the remaining Ih feedback error with the verified production implementation unchanged?'
python3 benchmarks/reconstructed_fault/uniform_shear/evolving/compare_smoke.py --case spatial0375_n512_f32_periodic_floor --expected-steps 8 --reference benchmarks/reconstructed_fault/uniform_shear/evolving/spatial0375_n128_f32-reference
python3 benchmarks/reconstructed_fault/uniform_shear/evolving/compare_normal_levels.py
```

Simulation, reference comparison and three-level accounting pass. Three cheap
accounting checks cover contracting, zero-error and nonmonotone ratios. An
initial accounting invocation preceded completion of its input-file generator
and raised FileNotFoundError; it made no scientific output or simulation
change. Accounting was then run after its prerequisite completed successfully.

New files: the 512 parameter overlay, `compare_normal_levels.py`, this report,
and saved run/analysis artifacts. `compare_smoke.py` only gains the approved
512 case name. The progress summary is updated. All preexisting production
and unrelated working-tree modifications are preserved unchanged. No rebuild,
MPI run, broad suite, extra reference trajectory, or additional ASPECT case
was performed. The accepted 128 result predates the precision-floor correction;
it is reused as authorized, with its original passing checks and hashes.

Artifacts under `benchmarks/reconstructed_fault/uniform_shear/evolving/`:
`spatial0375_n512_f32_periodic_floor.log`, its `.resources.json`, its output
directory (all raw fields/guards/profiles), its `-comparison.json`, and
`normal512-comparison.json`/`normal512-errors.csv`/`normal512-comparison.log`.
