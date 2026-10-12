# P1 — controlled-H phase evolution: physical qualification fails

Completed 2026-10-06. The separately authorized follow-up is recorded in
[P1_tail_report.md](P1_tail_report.md); the results below remain the original
two-run P1 evidence.

The bounded P1 implementation and checks are complete. **Both meshes converge
and show nontrivial phase evolution, but neither passes physical admissibility.**
Negative phi and local healing persist after the one permitted refinement.
P2 was not started. No production C++, governing equation, parameter default,
sequential coupling, H evolution law, or geometry-growth behavior was changed.

## Scope, baseline and changes

Worktree `/home/ein/repository/aspect-fault-propagation`, branch
`fault-propagation`, HEAD `de7beaa65846c7372711f7922abf1c5d9a745ee6`.
The preserved R7 executable at
`/home/ein/repository/aspect-pf-rsf-refactor/benchmarks/reconstructed_fault/refactoring_r7b/build/aspect-r7b-qualified`
was used read-only. Its source revision is
`6a3781277f2f594d52cd248d84f5b6fdc98fdeb1`; production source/headers match HEAD.
SHA256 remains `ed74d4d4ad1cfa5d1bd4fbade496d9748ab298c24a259a4612a94d4df375b60d`.
The CMake-only baseline difference is documented in [P0](P0_report.md).

The plugin was compiled in the task-owned `build-propagation-p1` using the
existing read-only Aspect package: GCC 12.4.0, OpenMPI 5.0.6, deal.II 9.6.2,
Trilinos 14.2/Epetra, Voro, Release, `-fno-finite-math-only -ffp-contract=off`.
No core build, other-worktree reconfiguration, commit, push or remote run occurred.
The native test build creates its generated runtime directories within this
worktree; the installed documentation paths remain unchanged.

Changes:

- [AGENTS.md](../../../AGENTS.md): restored the complete committed guidance as
  an exact prefix, retained the incoming propagation block, and appended the
  user's selected P1 scope. Refactoring-only restrictions and task-stage
  authority are distinguished; no existing instruction was discarded.
- [Test implementation](../../../tests/phase_field_controlled_history.cc),
  [parameter fixture](../../../tests/phase_field_controlled_history.prm),
  [output filter](../../../tests/phase_field_controlled_history.sh),
  [expected pass marker](../../../tests/phase_field_controlled_history/screen-output),
  and [independent numerical checker](../../../tests/phase_field_controlled_history_check.py).
- [Existing test access](../../../tests/phase_field_test_access.h): added only
  a const accessor for actual production solver parameters. No production hook.
- This report, compact evidence, STATUS and the evidence-based PLAN update.

[Identity manifest](P1_manifest.json) records executable, both plugin builds,
source/input/include hashes, resolved parameters and runtime inputs. Coarse
source is reconstructed from the recorded edit by reversing the location-only
instrumentation and assessment clarification; its snapshot and original plugin
are retained in the build directory. The [source delta](P1_refinement_instrumentation.diff)
adds compact worst-location/count/tangent-sign diagnostics and separates initial
relaxation from load-induced healing. It does not alter assembly or solves.

## Experiment and explicit checks

The native no-Stokes fixture initializes the existing finite seed from
`tests/phase_field_fault_ih.txt` in the unit box. Geometry is prescribed/fixed;
velocity and pressure are zero; temperature is static; no physical time step,
advection, RSF feedback, particle insertion or fault extension is exercised.
Both material entries use G=1e6, cohesion=1000, Gc=26.666666666666667,
ell=0.15625, AT1, Lorentz p=1; Hc=0.5 and m=128. The native initializer
supplies particle H. Reference-cell particles remain 3x3 per cell, load balancing
`none`, with native domain/CPDI generation.

The postprocessor runs five scratch-state solves: initial-H hold, identical-H
repeat, prescribed 5% and 10% H increases, then an applied-load reduction with
stored diagnostic H retained. The fixed diagnostic mask is `0.1<phi_seed<0.7`
at initialized CPDI samples. The subcritical applied candidate is 0.9 H_seed;
unloading returns the applied candidate to H_seed. The max operation here is
**prescribed diagnostic history**; no production mechanical H update is called
or qualified. The initialization solve and first hold are separate from the
subsequent loading/hold evidence.

All solves call production `PhaseFieldHandler::evolve_phase_field`; fresh
residuals/Jacobians use `PhaseFieldTestAccess::assemble`. The test does not
reimplement the governing operator or enforce phase inequalities.

Explicit numerical checks, independent of expected-output matching:

1. Fresh residual at every solved scratch state must satisfy the actual
   production target. Nonfinite results fail.
2. Both ramp changes must exceed 100 times the measured diagnostic allowance;
   prescribed H must not decrease; the mask must contain at least four particles.
3. Raw nodal/CPDI bounds and post-initialization phi decreases must fit the
   allowance. Significant violations throw, so convergence cannot yield an
   unconditional qualification marker.
4. Identical-H repeat drift must agree with the correction-based uncertainty
   estimate. If it does not, the result is inconclusive, not a pass.
5. Snapshot fingerprints compare particle IDs/positions/all properties,
   domain generation, fault geometry/version/properties, live current/old FE
   fields and system RHS exactly after normal and exceptional restoration.
6. The independent Python checker reads the CSV/log, recomputes gates and
   returns exit 2 for both actual runs. Seven offline
   [checker controls](P1_checker_controls.json) test a labeled admissible-data
   control and injected bounds, healing, residual, uncertainty and missing
   restoration failures. These synthetic data checks are not additional
   ASPECT experiments or physical validation.

### Supplied scratch-state verification

Source audit of `assemble_phase_field_system` confirms that phase values,
gradients, admissibility, g derivatives, residual, Jacobian and roundoff scale
all derive from its `current_solution` argument. H/composition remain explicit
particle inputs. In cohesive mode, the nonlinear driver passes its supplied
state and line-search trial to this assembler; the mature-mode old-solution
bypass is not active.

Dynamically, assembly at scratch phi=0.2 is repeated while **live phi=1.25**.
RHS, Jacobian entries and roundoff allowance match exactly, including the
residual-only path. Changing scratch phi to 0.25 changes RHS/Jacobian. All five
nonlinear scratch solves run while that invalid live sentinel remains in place.
An invalid scratch trial is rejected without changing accepted scratch state
or history. A separate call to the actual nonlinear entry throws for invalid
scratch phi after temporary H/live changes; restoration is checked afterward.
Both meshes pass these checks. The scratch guard restores test substitutions;
this does not demonstrate physical timestep rollback or checkpoint restart.

## Measured results

Primary evidence: [coarse CSV](P1_coarse.csv), [refined CSV](P1_refined.csv),
[coarse log](P1_coarse.log), [refined log](P1_refined.log), and independent
[coarse](P1_coarse_checks.json) / [refined](P1_refined_checks.json) assessments.

| Measurement | 16x16 | 32x32 |
|---|---:|---:|
| Cells / particles / masked particles | 256 / 2304 / 716 | 1024 / 9216 / 2792 |
| h_cell / ell | 0.4 | 0.2 |
| Native initialized profile vs prescribed profile, max difference | 0.163168 | 0.166093 |
| First hold adjustment, max | 2.243e-6 | 1.969e-7 |
| Identical-H hold-repeat drift | 0 | 0 |
| 5% ramp max phi change | 0.00630803 | 0.00596992 |
| 10% ramp max phi change | 0.00604781 | 0.00573741 |
| Maximum load-induced phi decrease | 1.77973e-5 | 4.23116e-6 |
| Lowest nodal phi across five states | -0.00718377 | -0.00603795 |
| Lowest CPDI phi | -0.00539882 | -0.00519851 |
| Highest nodal phi | 0.822389 | 0.808791 |
| Unload/identical-H refinement of solution, max | 5.501e-14 | 1.195e-13 |
| Maximum directional uncertainty estimate | 8.914e-14 | 1.492e-13 |
| Actual diagnostic allowance | 8.914e-13 | 1.492e-12 |
| Crack-density integral: equilibrated hold → unload | 0.603143 → 0.610560 | 0.593943 → 0.600890 |
| H min at hold | 0.0111620 | 0.0111145 |
| Newton iterations: hold/repeat/ramp/ramp/unload | 2 / 0 / 3 / 3 / 1 | 2 / 0 / 3 / 3 / 1 |
| Runtime / child peak RSS | 11.194 s / 199.4 MiB | 4.876 s / 210.5 MiB |
| Qualification | **FAIL: bounds and healing** | **FAIL: bounds and healing** |

Runtime includes the native CTest driver for the coarse case and direct ASPECT
launch for the refined case, so those times are not a speed comparison. Each
output directory is about 2.1 MiB, below the 10 MiB budget. Recorded resources
are in [coarse](P1_coarse_run.json) / [refined](P1_refined_run.json) run records.
Both are below the 120-s runtime and 2-GiB observation limits. Native CTest exits
8, refined ASPECT exits 1, both specifically on physical admissibility after
all convergence/restoration diagnostics. Neither failed due to timeout.

The new native test is intentionally still a **failing scientific qualification
gate** on this baseline. It was not converted to EXPECT FAILURE or given a
relaxed expected output. Its expected pass marker is not reached; explicit
filter/reference checks also reject both logs. This is a known failing fixture,
not a claim that the full regression suite passes. Existing exhaustion/frozen
regressions were retained but not rerun: no production solver was changed.

### Actual tolerances and uncertainty limits

Nonlinear relative tolerance is 1e-8, linear relative tolerance 2e-7; maximum
Newton/linear iterations 50/1000; line-search setting 3 (production loop admits
its existing final-candidate behavior). Each solve fixes
`T=max(1e-8*||r_entry||2, roundoff_entry)` with the production
`roundoff=8*epsilon_double*||S||2`. Both ramps have targets 1.545–1.595e-9
(coarse) and 8.092–8.345e-10 (refined); their final residuals are respectively
2.796–3.587e-12 and 2.114–2.681e-12. Repeat/unload targets reach the production
roundoff scale, about 2.4e-13 and 4.0e-13. All ten fresh residual checks pass.

The directional uncertainty is
`max(||J^-1 r||inf * max(1, roundoff_final/||r||2), epsilon*max(1,||phi||inf))`.
This is an estimate using the production linear solve, **not** an inverse-norm
bound. The physical diagnostic allowance is ten times the larger of that
estimate over the sequence and identical-H repeat/unload drift. Those repeats
support the estimated scale here. They do not bound discretization or model
error. First-hold relaxation is recorded separately and excluded from the
load-induced-healing gate; bounds are checked at all five solved states.
No KKT/projected-residual claim is made. The unrelated Ih undershoot guard
1e-4 is not an acceptance tolerance.

## Diagnosis and what remains uncertain

The persistent violations are not hidden by the residual criterion and are
not explained by the measured algebraic uncertainty. Raw negative phi is
already present at native initialization, before live-state poisoning or H
substitution. Identical-H equilibration removes its residual without removing
the undershoot. Refinement reduces maximum healing by about 4.2x but minimum
phi magnitude only by about 16%; CPDI undershoot improves still less. The
bounds/healing violations remain decisively resolved, not inconclusive within
the measured algebraic scale.

The refined worst nodal minimum is at `(0.125,0.21875)`, in the rounded-cap
neighborhood of the seed endpoint `(0.2,0.5)`, not on a computational boundary.
The worst CPDI sample is near `(0.130208,0.213542)` and carries H≈0.0207097,
well below bulk Hc=0.5. Native stationary-profile initialization can assign
such subcritical H inside its support; outside it the baseline remains Hc.
Thus the diagnostic ramp did not invent the low-H tail. The equality solver
allows nonsingular negative values and has no mechanism guaranteeing [0,1].
At refined hold, the assembled tangent has 2002 positive off-diagonal entries
(1978 after the second ramp). Consequently the usual nonpositive-off-diagonal
monotonicity argument cannot be assumed for this discretization. This is
supporting evidence, not proof of a unique cause.

The finite-seed distance profile is not guaranteed to be an exact two-dimensional
CPDI equilibrium around its rounded caps or near the box boundaries. The
0.16 initial profile mismatch does not improve in this pair. A source-based
cap/boundary-initialization mismatch and discrete loss of monotonicity remain
plausible contributors; these two runs do not separate them. Do not label
initial cap adjustment as new growth, all undershoot as iteration error, or
this failure as proof that every equality-solver configuration is inadmissible.
No H floor, clipping, active set, altered H law or propagation algorithm is
proposed or implemented to hide the failure.

**Next proposed work remains within P1:** isolate the finite-cap/profile/boundary
compatibility and discrete tail response using a separately selected bounded
control, before deciding whether any algorithm change is warranted. The allowed
refinement repeat has been used; no third mesh or smaller-load campaign was run.
P2 should not proceed on an unconditional P1 pass.

For eventual P2, retain these explicit questions: physical checkpoint/restart;
complete rejected-step restoration of phi/time/particles/fault histories;
transport-aware irreversibility; and whether the historical small-increment
feedback loss persists with the production mechanical H candidate/max update.
**P1 cannot answer that last question because its H increments are prescribed.**
The audited sequential `H_old -> phi -> mechanics -> H_new` order is preserved.

## Reproduction and verification record

From this worktree, use the read-only reference paths above. Only selected
plugin/test targets are built/run; the 1335 discovered native tests are not a
request to execute the full suite.

```sh
cmake -S tests -B build-propagation-p1 \
  -DAspect_DIR=/home/ein/repository/aspect-pf-rsf-refactor/build-refactor-post-r6-gcc12-unity \
  -DASPECT_BINARY=/home/ein/repository/aspect-pf-rsf-refactor/benchmarks/reconstructed_fault/refactoring_r7b/build/aspect-r7b-qualified \
  -DASPECT_RUN_ALL_TESTS=ON -DASPECT_COMPARE_TEST_RESULTS=ON -DTEST_TIME_LIMIT=120 \
  -DCMAKE_CXX_COMPILER=/opt/gcc/12.4.0/bin/g++ \
  '-DCMAKE_CXX_FLAGS=-fno-finite-math-only -ffp-contract=off'
cmake --build build-propagation-p1 --target phase_field_controlled_history -j2
# Export ASPECT_SOURCE_DIR=$PWD and one thread for OMP/DEAL_II/OPENBLAS.
ctest --test-dir build-propagation-p1 -R '^phase_field_controlled_history$' --output-on-failure -j1
python3 tests/phase_field_controlled_history_check.py \
  doc/reconstructed_fault/propagation/P1_coarse.csv \
  doc/reconstructed_fault/propagation/P1_coarse.log
```

The refined overlay changes only global refinement to 5, the plugin path and
output path; it is retained as `build-propagation-p1/refined.prm`. Exact launch
argv/environment/runtime are in the run records. Preserve existing evidence
before any reproduction; the above test remains expected to fail qualification.
A temporary Python logger imposed a 120-s subprocess limit and recorded child
RSS. The first timing-wrapper attempt could not find `/usr/bin/time` and launched
no simulation. Plugin compilation initially omitted the explicit floating-point
flags; it was rebuilt with them before any numerical run. The inherited Ih test
emits an existing unused-variable warning; no new compiler error occurred.

Final checks: committed AGENTS prefix retained; production source/include and
build configuration unchanged; test-only diff/whitespace, local documentation
links and recorded artifact hashes checked. No commit or push. Stop before P2.
