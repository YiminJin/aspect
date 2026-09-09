# K1 coupled-solver corrections and unchanged-pilot rerun

This review supersedes the **proposal/status**, not the saved measurements, in
`stage_K1_line_search_and_stress_diagnosis.md`. The two corrections were approved
separately. No stress smoothing, support extension, change to full `I_h`,
physical parameter, configured solver tolerance, line-search budget, or
initial/history-publication semantics is included. No K2 work is included.

## Correction 1: fixed characteristic surface normalization

At the first stabilized linearization, the surface scale is the maximum of
the initial free-surface residual norm, full-surface residual norm, and full
characteristic `K_V V_char` norm, with `V_char[i]=max(V_min,abs(V[i]))`.
It is fixed for the complete mechanical solve and shared by convergence and
Armijo merit. The characteristic surface action is no longer multiplied by
the bulk roundoff-floor factor. The bulk normalization remains unchanged.
This is the approved normalization-contract change, not an increase of a
configured accuracy tolerance or backtracking limit.

Implementation: `source/simulator/solver.cc`; authoritative rules updated in
`current_design.md` and `specification.tex`. The old generic scale-helper unit
test is correctly labeled as a **bulk** floor test; surface behavior is tested
through the actual coupled solver.

The new `phase_field_fault_changed_loading{,_mpi}` fixtures reuse K1's physical
parameters and loading law on a smaller 8x32 bulk mesh and run zero plus one
real two-second step. Their purpose is coupled normalization/lifecycle, not
profile accuracy or the K1 spatial-resolution gate. Their test-only fixed-Q1
profile snapshot includes imported periodic/shared DoFs on both MPI ranks.
Only phase DoFs are constrained; mechanics remains free except for the
production physical boundary constraints.

The tests require converged nonzero Newton updates in both solves, an
observable positive slip change under the changed loading, retained initial
Theta=200 and particle shear stress=1500, and the independently evaluated
exponential real-step Theta update. Omitting the mechanical update or the
history update fails these checks. On two ranks, the first real step has
relative residuals approximately `(1,9.509e-9)`, `(8.617e-10,9.716e-6)`, then
`(6.728e-14,2.525e-12)`: the finite curvature residual is accepted and removed
without changing the configured tolerance or search budget.

## Correction 2: frozen Maxwell stress in the entire bulk

The absolute bulk residual contains

\[
 R_{\rm hist}(w)=\int_\Omega\beta\tau_{\rm old}:\epsilon(w)\,d\Omega.
\]

Therefore `ReconstructedFaultStokes::execute` adds **minus** this term to
ASPECT's Newton RHS, once at every locally owned bulk quadrature point,
including QPs outside reconstructed-fault support. At associated QPs it also
retains the existing positive RHS counterpart of
`-2 kappa (chi V + upsilon_hist) S`. The ordinary Stokes assembler supplies
the current-strain-rate term. No other assembler adds the frozen load.

The assembler caches only the explicit mapping from independent stress
components to their particle-advected compositional FE fields. The values in
the current working vector are the frozen, currently mapped particle history;
they are not read from the older FE timestep and are never modified by the
assembly. No second stress history or transaction is introduced.

The one added public material operation is
`PhaseFieldFault::evaluate_frozen_maxwell_stress(temperature, composition, old_stress)`.
It returns only the pointwise `beta*tau_old`, using the existing bulk chemical
fraction utility, creep viscosity, shear-modulus averaging, and Maxwell time
interval. It does not need a fault association or own any weak-form work.
The slip-only residual and B action retain their existing semantics. With
temperature/composition/history frozen, this added term has no derivative in
the mechanical `(u,p,V)` Newton system, so A/B/G/K_V are unchanged.

Touched production files are the PhaseFieldFault header/source and
ReconstructedFaultStokes header/source. The stale material-model viscosity
comment was corrected: the ordinary Stokes assembler does use `kappa`.
The authoritative specification documents the sign, all-bulk support, mapped
history source, and unchanged publication lifecycle.

### Nonuniform and constant-history regression

`phase_field_fault_frozen_stress{,_mpi}` initializes particle history as
`tau_xy=1500+1000*y`, retaining the normal K1 `cell average` particle-to-FE
mapping. That mapping is **not** assumed to reproduce the analytic linear
field exactly. The independent assembly integrates the realized FE field
with QGauss(4), separate from production QGauss(3).

For identical V and other fields, the test compares production execute with
zero, constant, and initialized nonuniform FE history. It checks every cell's
RHS difference against the independently integrated negative history load,
thereby detecting wrong sign, omission, support restriction or duplication.
It additionally checks the constrained global constant-history load norm
below `1e-9` and a demonstrably nonzero nonuniform load norm above 1.

The admissible Q2 virtual velocity `w=(y^2-1/4,0)` gives:

| Quantity | One rank | Two ranks |
|---|---:|---:|
| Nonuniform frozen-history RHS work | -4.080e1 | -4.080e1 |
| Work from QPs outside fault support | -3.164e1 | -3.164e1 |
| Complete accepted-equilibrium work | 1.003e-13 | 2.274e-13 |

The independent history-work comparison retains `1e-9` absolute tolerance;
the cell-vector comparison retains `1e-11` relative tolerance. The last row
checks the complete accepted production stress, not just this assembler, so
adding a duplicate elsewhere or retaining the omitted load cannot pass.
The virtual-work field has the fixed numerical amplitude above; these work
values are test diagnostics, not traction residual RMS norms.

## Focused verification and fixture corrections

All builds used `-j4`. Only focused tests were selected.

- Changed-loading regression: one and two ranks passed.
- Nonuniform/constant frozen-stress regression: one and two ranks passed.
- Final combined CTest check: **4/4 passed** (90.22 s; valid cached stress
  results reused). Fresh stress runs had already completed their numerical
  checks in 116.30 s / 89.77 s; only output alignment required correction.
- Existing `phase_field_fault_stage_i_rollback{,_mpi}` passed with their
  original tolerances/assertions: 156.54 s and 90.78 s in the concurrent run.
  They force failure after accepted Newton updates and verify restored bulk,
  current/committed V, and all saved particle/surface histories.
- `aspect --test 'Stage-I*'`: **31 assertions passed in seven cases**, including
  forced backtracking, exhaustion without accepting the last candidate,
  release and exact fraction-to-boundary checks.
- Python reference/analysis/diagnosis tests: **11 passed**. The scalar reference
  still initializes histories once and never resets them from later output.
- `git diff --check` passed.
- Stage-J two-rank restart create/resume pair: **2/2 passed**, 344.50 s and
  122.69 s (467.22 s total). Create verifies timestep zero plus two actual
  evolving-phase/history steps; resume matches the uninterrupted accepted
  state. No initialization, H/Theta update or checkpoint assertions changed.

The setup failures encountered while constructing the new fixtures were
resolved without adjusting scientific acceptance thresholds: inherited
postprocessor names must be registered before parameter overrides are parsed;
the MPI fixed-phi snapshot must include shared/periodic imported DoFs; and
the cell-average particle mapping must not be confused with exact linear FE
reproduction. New expected output lines were aligned with ASPECT's whitespace
formatting after all numerical assertions passed. No existing verification
assertion or expected scientific result was relaxed.

Build/test logs and the recoverable working-tree checkpoint are initially in
`/tmp/aspect-k1-corrections-q7oEmm`. The saved executable there is explicitly
named `aspect-normalization-only`: it contains correction 1 but not correction 2.
The final executable SHA256 is
`849326eb885d76921503b1c4d1cd0f1d2fdc97fe9853911ab858a6b585fc22eb`.
Selected build/test logs are also retained under
`benchmarks/reconstructed_fault/uniform_shear/diagnostics/coupled-corrections/`.

## Unchanged K1 pilot and independent comparison

The pilot uses `pilot.prm` through an output-directory-only include override,
`diagnostics/corrected_coupled_pilot.prm`. It keeps the original 16x64 bulk
mesh, `ell=0.15625`, support half-width `0.3088215939070757`, full `I_h`, all
physical coefficients, tolerances and the five-reduction search budget.
The original postprocessor freezes only the converged initial phase field;
its `Evolve phase field=false` retains its existing meaning of frozen H.

Read-only GDB instrumentation records both residual norms, their fixed
scales, normalized residuals, time and Newton iteration before convergence
testing. It neither modifies program state nor retries with different
parameters. `analyze_coupled_residuals.py` converts these records into tables.
`analyze.py --provisional-k1-containment` explicitly opts into the reviewed
K1-only `1e-4` omitted-fraction allowance. It retains the `1e-4` actual slip
normalization requirement and the unchanged full-Ih scalar reference. The
measured bulk slip is `chi*V+upsilon_hist`, not just its V-dependent part.

### Completed trajectory and residuals

The unchanged one-rank pilot **completed successfully**, accepting times
`0, 2, 4, 6 s`, with the original two-second steps and loading sequence.
It required 8, 2, 3 and 4 accepted Newton updates, respectively.

| Time (s) | Bulk residual norm | Surface residual RMS (Pa) | Bulk normalized | Surface normalized |
|---:|---:|---:|---:|---:|
| 0 | 8.612596e-12 | 5.347449e-8 | 7.677072e-15 | 5.588097e-11 |
| 2 | 8.744042e-12 | 2.524514e-9 | 7.794240e-14 | 2.327659e-12 |
| 4 | 5.425114e-12 | 7.810783e-9 | 4.835823e-14 | 7.005374e-12 |
| 6 | 7.926834e-12 | 1.067325e-13 | 3.710021e-10 | 2.223200e-16 |

The bulk column is the native Stokes weak-vector norm, **not** a traction RMS
in Pa; it includes the solver's pressure-scaled continuity block. Surface
scales in Pa were `956.9356715, 1084.572069, 1114.970150, 480.0850370`;
bulk scales were `1121.859458, 112.1859458, 112.1859531, 0.02136600595`.
Each scale stayed fixed within its solve. The dimensional norms, scales and
ratios for every recorded nonlinear iterate are in
`diagnostics/coupled-corrections/nonlinear_residuals.csv`, with accepted
endpoints in `converged_residuals.json` in the same directory.

### Accepted states and independent reference

Entries below are **ASPECT / independent reference**. Surface means use
arclength weights. C denotes cohesive traction; the final column is the
volume-weighted committed particle shear stress, not the evaluated t=0
mechanical response.

| Time (s) | V (m/s) | Theta (s) | C (Pa) | Retained particle stress (Pa) |
|---:|---:|---:|---:|---:|
| 0 | 3.1731792608e-4 / 3.1717379701e-4 | 200 / 200 | 313.51176015 / 313.51176015 | 1500 / 1500 |
| 2 | 3.2642138828e-4 / 3.2627297450e-4 | 105.58152952 / 105.61263210 | 313.28460407 / 313.28188479 | 1041.97233242 / 1042.04882767 |
| 4 | 1.3628765725e-4 / 1.3626326091e-4 | 82.14206527 / 82.16971181 | 309.57825750 / 309.57514507 | 989.17893783 / 989.21146416 |
| 6 | 1.1674742859e-4 / 1.1671396278e-4 | 66.82060384 / 66.84690532 | 305.58727946 / 305.58361549 | 976.11015461 / 976.13055444 |

The independent scalar calculation still uses **full** `I_h`, initializes
histories once from the documented retained initial state, and advances on
the actual accepted dt/loading sequence. It never resets from subsequent
ASPECT histories. The initial C input is independently checked against the
Q1 projection of initialized particle q (maximum error `2.274e-13 Pa`).

At timestep zero, evaluated mean shear stress is
`1040.01656699 / 1040.26507958 Pa`; it is deliberately **not** committed over
the artificial initialization interval. Theta=200, initial H and particle
stress=1500 are retained. Relative to the saved pre-correction mechanical
initialization, `phase_0.csv` and `particles_0.csv` are byte-identical,
surface Theta/C/I_h differences are zero, and maximum V difference is
`7.0473e-19 m/s`. No initialization or stress filtering was introduced.

| Observable | Maximum absolute error over accepted times | Error / documented diagnostic scale |
|---|---:|---:|
| V | 1.484138e-7 m/s | 0.148414% |
| Theta | 0.03110258 s | 0.0155513% |
| C | 0.00366397 Pa | 0.000244265% |
| Evaluated bulk mean shear stress | 0.25593878 Pa | 0.0170626% |
| Committed particle mean shear stress, real steps | 0.07649525 Pa | 0.00509968% |
| Accumulated slip | 4.125518e-7 m | 0.0687586% |

These satisfy the existing 2% pilot trajectory target plus the documented
absolute allowances, without altering those targets. The analysis JSON
reports raw observations/reference values; the error table above is computed
from them, separately from its assumption-check exit status. Final accumulated
slip is `0.00115891294823 / 0.00115850039638 m`. The independent vertex-wise
exponential Theta-update check has maximum reported difference zero; real
reference increments are tens of seconds, so skipping the update cannot
pass by tolerance. Pointwise stresses remain unsmoothed: the raw evaluated
QP stress range is `16.7095, 17.1889, 7.23185, 6.16863 Pa` at the four times.
A small mean error does not establish pointwise stress convergence.

### Assumptions and remaining gate

- The reconstructed fault has length `0.25 m`, nine vertices, maximum
  `|y|=8.109e-18 m` and angle `1.532e-16 rad`. Its two support half-widths
  remain `0.3088215939070757 m`.
- Independent full-profile `I_h=108.07223820379 m`; maximum relative error of
  production projected I_h is `4.647e-10`.
- Omitted fraction is `5.649175485e-5`, within the **provisional K1-only**
  `1e-4` allowance, not the original `1e-6` gate. Exported phase and segment
  files are each byte-identical across all four accepted times, so this
  containment measurement applies to the complete fixed-profile trajectory.
- Actual integrated bulk slip divided by the integrated surface V is
  `0.999941301533839` to `0.999941301533846`. The maximum normalization deficit
  `5.869846616e-5` passes the unchanged `1e-4` requirement, with full I_h kept
  in both production and reference.
- Endpoint projection masses are positive (`0.006357452016 m^2` each).
  All particle volumes are positive; maximum total-volume error is
  `1.364e-10 m^2` on the `0.25 m^2` domain. Individual final volumes range
  from `0.89350` to `1.10649` times their initial uniform value. This measures
  actual geometry; it does **not** assume periodic Voronoi domains or exact
  individual-volume preservation.
- Maximum along-fault V range / `1e-4 m/s` is `2.267e-6`; maximum prescribed
  velocity error is `4.066e-20 m/s`. Transverse velocity RMS is at most
  `6.543e-13 m/s`, divergence RMS at most `2.048e-12 s^-1`.

The bounded pilot assumptions and trajectory comparison pass. **Full Gate
K1 is not declared passed:** fixed-ell spatial/refinement and timestep
convergence of the completed trajectory, including confirmation of the
provisional containment allowance under refinement, remain outstanding.
No broader solver/model change is indicated by this pilot; no K2 work began.

### Runtime and reproducibility

The instrumented debug run exited zero (inferior exited normally). ASPECT's
timer reported `314.4 s`; total wall time was `327.61 s`. Process-tree maximum
RSS was `7,271,656 KiB`, **including GDB and debug-symbol loading**; this is not
an ASPECT-only memory measurement. An uninstrumented ASPECT-only peak was not
measured in this rerun.

Commands (from repository root unless a working directory is stated):

```sh
cmake --build build-pf-cpdi --target aspect -j4
cmake -S tests -B build-pf-cpdi/tests
ctest --test-dir build-pf-cpdi/tests --output-on-failure -R '^phase_field_fault_(changed_loading|frozen_stress)(_mpi)?$' -j2
ctest --test-dir build-pf-cpdi/tests --output-on-failure -R '^phase_field_fault_stage_i_rollback(_mpi)?$' -j2
build-pf-cpdi/aspect --test 'Stage-I*'
ctest --test-dir build-pf-cpdi/tests --output-on-failure -R '^phase_field_fault_stage_j_restart_(create|resume)$' -j1
cmake --build benchmarks/reconstructed_fault/uniform_shear/diagnostics/pilot-build -j4
```

Pilot working directory:
`benchmarks/reconstructed_fault/uniform_shear/diagnostics/pilot-build`.

```sh
TIMEFMT='wall_seconds=%E user_seconds=%U system_seconds=%S max_rss_kB=%M'
{ time timeout 600 gdb -q -batch -x ../coupled_residuals.gdb --args /home/ein/repository/aspect/build-pf-cpdi/aspect ../corrected_coupled_pilot.prm; } > ../coupled-pilot.log 2>&1
```

Analysis working directory: `benchmarks/reconstructed_fault/uniform_shear`.

```sh
python3 analyze.py diagnostics/coupled-pilot --provisional-k1-containment
python3 -m unittest -v test_analysis test_reference test_support_resolution test_line_search_diagnosis
```

Full accepted particle/surface histories, bulk QPs with cell IDs, phase
profiles and actual accepted parameters are in `diagnostics/coupled-pilot/`.
The raw instrumented run is `diagnostics/coupled-pilot.log`; independent
comparison and focused verification logs are in `diagnostics/coupled-corrections/`.

## Change scope

This turn changed the two authoritative documents; the PhaseFieldFault and
ReconstructedFaultStokes headers/sources; the normalization block in
`source/simulator/solver.cc`; and the bulk-floor unit-test label in
`unit_tests/reconstructed_fault.cc`. It added the four named focused fixtures
(sources, parameter files, filter scripts and expected output). Benchmark
changes are limited to `analyze.py` (explicit provisional allowance and full
slip diagnostic), `analyze_coupled_residuals.py`, an output-only pilot parameter
override, read-only GDB instrumentation, this report and saved artifacts.

The working tree already contained physical-constraint lifting/rollback,
the initial previous-phi correction, particle-domain ownership work, earlier
fixture-budget corrections and other untracked documents/benchmarks. Those
were preserved, not reset or attributed to these two corrections. No commit
was requested or created. No full ASPECT suite or 3-D verification was run.
