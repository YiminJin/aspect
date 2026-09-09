# K1 bounded spatial and timestep convergence

Follow-up: the approved pressure-normalization correction and unchanged
reproducer results are recorded in
`stage_K1_pressure_normalization_review.md`. Its focused gauge tests pass,
but dt=.5 still fails at 4.5 s; Gate K1 remains unmet. The findings below describe the
pre-correction checkpoint, not a completed convergence campaign.

**Stopped for review: Gate K1 is not satisfied.** The new 0.5 s run failed
at 4.5 s, and a separate cross-build comparison confirmed that the coupled
solver does not honor `Pressure normalization = volume`. No production code
was modified, no tolerance/support/I_h change was made, and K2 was not started.

## Scope and predeclared comparisons

This follows the accepted `stage_K1_coupled_corrections_review.md` pilot.
The original 1e-6 containment target is not pursued. Support policy, full I_h,
phase/H initialization, physical coefficients, solver tolerances and search
budget are unchanged. The provisional 1e-4 containment allowance applies to
K1 only; actual slip normalization retains its separate 1e-4 requirement.
No production edits are planned or authorized by this verification step.

The five-case cross (not a parameter product) is:

| Bulk mesh | Fault spacing (m) | Real dt (s) |
|---|---:|---:|
| 16 x 64 | 1/32 | 0.5 |
| 32 x 128 | 1/64 | 0.5 |
| 64 x 256 | 1/128 | 0.5 |
| 64 x 256 | 1/128 | 1 |
| 64 x 256 | 1/128 | 2 |

All use fixed ell=0.15625 m, 3x3 particles per cell, end time 6 s and
numerical Initial time step=2 s. The accepted 16x64/dt=2 pilot is reused,
as are the previous one-/two-rank changed-loading and rollback tests and
Stage-J restart create/resume evidence. The previous refined t=0 run is
initial-stress evidence only, not a completed refined trajectory.

The existing viscosity cutoffs apply to eta=1e8 Pa s, not to the Maxwell
effective kappa. No new cutoff is applied as dt shrinks:

| dt (s) | beta | kappa (Pa s) |
|---:|---:|---:|
| 2 | 0.980198673307 | 1980132.669324 |
| 1 | 0.990049833749 | 995016.625083 |
| 0.5 | 0.995012479193 | 498752.080732 |

In particular kappa may be below the configured minimum **creep** viscosity
of 1e6 Pa s. The pointwise Maxwell formula and ordinary Stokes viscosity output
retain the same value; the time refinement does not alter parameter meaning.

Every case independently integrates its own saved Q1 profile and initializes
its scalar history once with retained stress=1500 Pa, supplied Theta0 and
that case's independently audited initial cohesive projection. Subsequent
production histories are observations only. Differences in initial phi,
I_h and cohesive traction are reported separately from conditional mechanical
errors. The scalar initial mechanical evaluation does not overwrite histories.

Resolved conditional errors use the existing 0.2% relative plus 1e-5 of
the documented dimensional scales: velocity/V=1e-4 m/s, Theta=200 s,
traction=1500 Pa and slip=6e-4 m. Velocity weighted RMS and pointwise errors,
raw (unsmoothed) stress RMS/max errors, surface ranges/endpoints and pressure
gauge are retained. A separate independently advanced scalar dt=1/256 vs
1/512 s comparison estimates temporal error at common times 2,4,6 s; this
does not replace matching the actual accepted production timestep sequence.

Case parameter files, logs, resource/provenance records and results are under
`benchmarks/reconstructed_fault/uniform_shear/convergence/`. Each run is
bounded by 7200 s. The runner records binary/plugin SHA256 hashes and the
input hash. The accepted Debug binary is retained unchanged. Initial concurrent
Debug attempts exposed a resource limit: process inspection showed the five
runs sharing roughly one CPU's execution, not the 22 CPUs reported by nproc.
Four unfinished refined attempts were deliberately terminated, with all output
and exit/resource records moved to `convergence/debug-interrupted/`; they are
not numerical failures or verification passes. The coarse Debug run continued.

To bound cost, the current source Release target is rebuilt with `-j4`, without
changing source, equations, inputs or tolerances. `release_check.prm` repeats
the accepted coarse pilot in an output-only override; its states must be
compared with the saved Debug pilot before using Release refinement results.
This additional run is a build-configuration cross-check, not another parameter
product. Remaining cases will use bounded low-concurrency execution.

## Findings that stop the campaign

### Confirmed missing pressure normalization

The current-source Release build completed successfully. Its output-only
repeat of the accepted 16x64/dt=2 pilot completed through 6 s in **28.47 s**,
peak RSS **242872 KiB**, with the same physical/numerical input. Relative to
the saved Debug result, maximum differences over all accepted times were:

| Field | Maximum difference |
|---|---:|
| Phase | 1.533e-14 |
| Surface V | 8.300e-17 m/s |
| Surface Theta | 2.078e-11 s |
| Surface C | 2.809e-11 Pa |
| Surface I_h | 1.038e-11 m |
| Particle shear stress | 4.999e-10 Pa |
| Bulk velocity x | 4.358e-18 m/s |
| Bulk pressure | **2047.668874891 Pa** |

This is **not** a complete Debug/Release equivalence pass: the pressure
discrepancy exposes an already-present gauge defect, rather than a Release
change in the constitutive equations. The difference at 6 s is constant to
`6.874e-10 Pa`; both nonconstant pressure fields agree to roundoff.

| Accepted time (s) | Debug volume mean pressure (Pa) | Release volume mean pressure (Pa) |
|---:|---:|---:|
| 0 | -2.700e-13 | 2.914e-12 |
| 2 | -1.525e-12 | 3.736e-12 |
| 4 | 7.323e-13 | 2.534e-13 |
| 6 | **-1480.929134687** | **566.739740203** |

Both actual exported parameter files specify `Pressure normalization = volume`.
That parameter requires zero domain-average pressure, not an arbitrary constant:

- `source/simulator/parameters.cc`, the `Pressure normalization` declaration,
  explicitly defines the zero-volume-mean rule.
- `Simulator::normalize_pressure` in `source/simulator/helper_functions.cc`
  computes and applies `-integral(p)/volume` for this mode.
- Conventional Newton publication in `source/simulator/solver_schemes.cc`
  normalizes `current_linearization_point` before copying to `solution`.
- The reconstructed-fault branch returns early from that conventional scheme.
  `Simulator::solve_reconstructed_fault_stokes` copies private accepted bulk
  state and publishes it through the terminal swaps without any call to
  `normalize_pressure` or corresponding adjustment bookkeeping.

The new **read-only** `convergence/audit_pressure.py` reproduces the failure
from saved native QP values and actual parameter files (exit **2**). It uses
the already documented 0.015 Pa traction allowance to demonstrate a failure
far beyond rounding error; it does not replace pressure data, change a gauge,
or define a new physical/numerical parameter. Mean subtraction in its
cross-build comparison is diagnostic only.

K1's prescribed adiabatic friction pressure makes velocity and fault
histories insensitive to this constant offset. That explains their agreement;
it does not make the ignored pressure-normalization setting correct. No
postprocessing correction of pressure was used to declare a pass.

### Separately observed 0.5 s nonlinear failure

The coarse Debug run was allowed to finish its in-progress bounded attempt.
It accepted t=0 through t=4 s, then failed at **t=4.5 s**. The configured
nonlinear tolerance remained 1e-8 and the original five-reduction search
budget was retained.

| Newton iteration at 4.5 s | Normalized bulk residual | Normalized surface residual | Rejections before next acceptance |
|---:|---:|---:|---:|
| 0 | 1.000e0 | 4.459e-3 | 0 |
| 1 | 3.702e-7 | 3.675e-6 | 0 |
| 2 | 2.035e-8 | 2.486e-12 | 1 |
| 3 | 1.829e-8 | 8.285e-13 | 3 |
| 4 | 1.775e-8 | 5.830e-13 | 4 |
| 5 | 1.754e-8 | 4.677e-13 | Exhausted; no acceptance |

Exit status was **1**. No 4.5 s accepted-state CSV exists. This is not nonlinear
iteration-budget exhaustion: the bulk residual stagnated above tolerance and
Armijo exhausted its candidates. Increasing Max nonlinear iterations would
not remove this particular stopping condition.

Its accepted pressure mean first drifted to **415.971896997 Pa at 2.5 s** and
remained there through 4 s. A connection between this unremoved pressure
null mode and subsequent residual stagnation is plausible but **not proved**.
No pressure-gauge correction, Jacobian modification, or tolerance relaxation
has been attempted. The last failed candidates' dimensional residual scales
were not instrumented in this run; normalized histories above are the actual
production log, not reconstructed dimensional residuals.

## Valid partial trajectory and reused spatial evidence

For the new dt=0.5 run, every reference history starts once from
`q0=1500 Pa, Theta0=200 s, C0=313.5117601471154 Pa` and independently integrated
`I_h=108.07223820379153 m`. Its initial phase file is byte-identical to the
accepted dt=2 Debug pilot. Thus changes between these two coarse trajectories
are not initialization/profile changes. The failed 4.5 s candidate is never
included as an accepted observation or a reference-reset point.

| Time (s) | V: ASPECT / discrete reference (m/s) | Theta: ASPECT / reference (s) | C: ASPECT / reference (Pa) |
|---:|---:|---:|---:|
| 0 | 3.1731792608e-4 / 3.1717379701e-4 | 200 / 200 | 313.51176015 / 313.51176015 |
| 0.5 | 8.4836413002e-4 / 8.4803020815e-4 | 131.26843354 / 131.29031565 | 315.86330398 / 315.86176294 |
| 2 | 1.2250966591e-4 / 1.2247667497e-4 | 100.83803953 / 100.86105650 | 313.70248986 / 313.70038208 |
| 4 | 1.1246478846e-4 / 1.1242792946e-4 | 82.48121471 / 82.50538879 | 309.53296705 / 309.53025454 |

Over accepted times 0 through 4 s, maximum surface errors are
`3.33925e-7 m/s` for V, `0.0241777 s` for Theta and `0.00271291 Pa` for C.
Maximum accumulated-slip error is `3.01789e-7 m`. These are **partial-run**
conditional comparisons, not a successful timestep-convergence result.

| Time (s) | Raw QP stress RMS error (Pa) | Raw QP stress max error (Pa) |
|---:|---:|---:|
| 0 | 2.808379 | 9.034562 |
| 0.5 | 1.889088 | 6.109587 |
| 2 | 0.272506 | 0.893758 |
| 4 | 0.250068 | 0.824698 |

Maximum velocity-profile RMS/max errors over this partial trajectory are
`8.07208e-8 / 1.31123e-7 m/s`. Stresses are native unfiltered QP values, using
the complete `kappa*(ux_y+uy_x-chi*V-history)+beta*tau_old` decomposition.
The original cell-ID/QP sidecars preserve provenance for all raw rows.

Containment remains **5.649175485e-5**, within the provisional K1-only 1e-4
allowance. The maximum actual slip-normalization error is **5.869846616e-5**,
within the unchanged 1e-4 requirement. These statements apply only to the
accepted partial trajectory and existing completed pilot, not unrun refined
trajectories. There is no support extension, truncated I_h, or renormalized V.

The saved pre-existing 32x128 **t=0-only** run remains useful evidence:

| Initial quantity | 16x64 | Saved 32x128 |
|---|---:|---:|
| Center phi | 0.600582655711 | 0.599910323280 |
| Independent I_h (m) | 108.072238203792 | 108.098095072262 |
| Initialized C0 (Pa) | 313.511760147115 | 318.747655767571 |
| Raw initial stress RMS error (Pa) | 2.808379 | 0.697651 |
| Raw initial stress max error (Pa) | 9.034562 | 2.290511 |

This shows roughly fourfold reduction of raw initial stress error, while the
initialized cohesive state itself changes by 5.23590 Pa. The latter must not
be charged to mechanics against a reference initialized on the other grid.
The saved refined run had nine fault vertices (not this cross's refined fault
spacing), and no real timesteps. It is **not** substituted for a completed
three-level spatial study. The four interrupted refined attempts have no
accepted mechanical states and provide no new trajectory-convergence evidence.

Separate scalar time-limit probes also warn against calling dt=0.5
time-continuously resolved. With the same once-initialized coarse data, the
reference's V error at t=2 relative to its dt=1/512 s trajectory decreases
from about 217.1% (dt=2) to 65.54% (dt=1) to 19.03% (dt=0.5). Those are scalar
discretization diagnostics, **not** three measured ASPECT runs. Matching a
finite-step reference does not remove this initialization transient or prove
small-dt accuracy. No histories were reset to conceal it.

## Small correction proposed for review; not implemented

1. For the current prescribed-adiabatic-pressure branch, restore the existing
   configured pressure gauge on private physical base/trial states using
   ASPECT's normalization infrastructure. Keep perturbation constraints
   homogeneous and pressure scaling unchanged; do not treat a physical
   surface-pressure offset as an inhomogeneous Newton direction.
2. Normalize and synchronize both accepted bulk publication vectors before
   failure-capable history work ends. Keep normalization-adjustment metadata
   consistent and include it in rollback. No allocation/MPI normalization may
   be inserted after the terminal history/V writes begin.
3. Add a one-/two-rank regression checking actual volume mean at **every**
   accepted timestep, including the unchanged-load tail of this fixture; also
   check gauge-equivalent starting pressures produce identical velocity/fault
   states and preserve existing exhaustion/rollback coverage.
4. Rerun the unchanged dt=0.5 failing case to determine whether this correction
   also removes bulk stagnation. If not, diagnose its dimensional residual
   floor and linear direction separately; do not increase tolerances/budget.

A blanket post-solve pressure shift is **not** proposed for true-normal-stress
friction: there pressure changes the surface equation and a shift cannot be
silently treated as gauge-only. That branch needs an explicitly consistent
pressure/gauge treatment before extending the correction to it. No new
production interface or algorithm is introduced in this verification turn.

## Verification record and artifacts

| Requirement | Current status |
|---|---|
| Accepted corrected 16x64/dt=2 pilot | Reused; velocity/history evidence remains valid; new gauge defect disclosed |
| Three completed spatial levels | **Unmet**; attempts stopped before new refined accepted mechanics |
| Three completed timestep levels on resolved mesh | **Unmet** |
| New coarse dt=0.5 trajectory | **Failed at 4.5 s**, accepted through 4 s |
| K1-only containment and actual normalization | Pass on measured accepted states only |
| Configured volume-pressure normalization | **Fails**, source cause confirmed |
| Existing MPI, rollback and restart regressions | Reused passing evidence from coupled-corrections review |
| K2.1 | **Not started**; no K1 allowance transferred |

Current-source Release build: `cmake --build build-pf-cpdi --target aspect.exe.release -j4`
passed (compiler warnings retained in `convergence/release-build.log`). Release
`aspect-release --test 'Stage-I*'` passed **31 assertions in seven cases**.
Independent analysis tests passed **14 cases**:

```sh
# From benchmarks/reconstructed_fault/uniform_shear:
python3 -m unittest -v test_convergence test_analysis test_reference test_support_resolution test_line_search_diagnosis
```

Reproduction from repository root:

```sh
python3 benchmarks/reconstructed_fault/uniform_shear/convergence/run_case.py benchmarks/reconstructed_fault/uniform_shear/convergence/space16_dt05.prm
python3 benchmarks/reconstructed_fault/uniform_shear/convergence/run_case.py benchmarks/reconstructed_fault/uniform_shear/convergence/release_check.prm --configuration Release --timeout 600
python3 benchmarks/reconstructed_fault/uniform_shear/convergence/audit_pressure.py benchmarks/reconstructed_fault/uniform_shear/diagnostics/coupled-pilot benchmarks/reconstructed_fault/uniform_shear/convergence/release_check
python3 benchmarks/reconstructed_fault/uniform_shear/analyze.py benchmarks/reconstructed_fault/uniform_shear/convergence/space16_dt05 --provisional-k1-containment
```

Expected observed statuses: first run **1** (nonlinear failure), Release repeat
**0**, pressure audit **2** (demonstrated missing normalization), partial-run
analysis **2** (incomplete trajectory, not a Python failure). Do not overwrite
saved result directories when reproducing; use a fresh output-only override.

The failed coarse run used 1341.59 s wall time including contention, peak RSS
462404 KiB. Its log/resources are `convergence/space16_dt05.log` and
`space16_dt05.resources.json`. Other artifacts in the same directory are
`pressure-audit.json`, `release-debug-comparison.json`,
`space16_dt05-analysis.json`, `space16_dt05-errors.json`, and the raw/profile
CSV folder `space16_dt05-errors/`. `partial_histories.png` and
`partial_profiles.png` show only accepted data and explicitly label the failure.

Debug executable SHA256 remains
`849326eb885d76921503b1c4d1cd0f1d2fdc97fe9853911ab858a6b585fc22eb`;
the `build-pf-cpdi/aspect` symlink still selects Debug. Current-source Release
SHA256 is `a5cb9eb10959d5d91a50d05b15e2e02b43d8344440aa0520371a03ec1fb08403`.
No production source changes, full test suite, support changes or commits
were made. New files are benchmark-only runners/analysis/tests, five cross
parameter files, the Release cross-check input, saved results and this report.
README/progress pointers were updated without deleting historical evidence.
