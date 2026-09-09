# K1 prerequisite correction and initial-root gate

Follow-up authorization: the user approved the initial previous-profile
correction after this report. The shared constitutive localization now uses
the converged current phi at timestep zero, without changing the old FE vector
or retained histories. The formerly blocked calculation below is historical;
the reference's default now uses the approved rule. Subsequent verification
is recorded in `stage_K1_pilot_report.md`. The sections below retain the
pre-authorization root-gate evidence and are not the current pilot status.

Working-tree report against `f4032b1824ef4892020af8c58ef981aca03dc0ec`.
This continues the accepted K0 audit. No commit has been made for this work.
The pre-existing untracked documentation collections and other unrelated
files remain untouched.

## 1. Physical boundary constraints

The original executable was tested with a no-op Stokes assembler that samples
the working iterate on its first cell assembly, before evaluating that cell's
residual. The ordinary `pre_assemble_stokes_system` signal is not emitted by
the coupled path, so that signal was not a usable observation point. The test
assembler performs no MPI collective inside its worker; the postprocessor
validates the rank-local measurements collectively after the solve.

Confirmed failures on the original executable:

| Reproducer | MPI ranks | First residual input |
| --- | ---: | --- |
| Initial prescribed x velocity 1e-4 m/s | 1 | Maximum boundary error 1e-4 m/s at step zero |
| Initially zero loading, then x velocity 1e-4 m/s | 2 | Zero error at step zero; error 1e-4 m/s at the next step |

Both fail the new assertion `First coupled residual violates current
prescribed velocities`. Artifacts are retained under
`/tmp/aspect-k1-prerequisite-ilbPnn/`: `before-initial.log`, `before-later.log`
and `before-ctest.log`. Early fixture-development failures are separate:
inherited postprocessor registration, missing particle postprocessor, and a
missing test include were corrected before these reproductions. An initially
proposed assertion of rigid interior translation was invalid: the initial
cohesive history correction can drive interior flow despite zero supplied
Maxwell stress. That assertion was removed, not loosened.

The production correction is confined to
`Simulator::solve_reconstructed_fault_stokes()` in
`source/simulator/solver.cc`:

- Obtain current **physical** constraints through the existing constraint
  computation, with the ordinary Newton-boundary-zeroing mode disabled.
- Distribute them into an owned private base iterate. The production solution
  is not mutated by this lift.
- Homogenize only the Stokes constraint inhomogeneities for all subsequent
  residual/Jacobian assembly. Hanging-node/periodic relations remain intact;
  auxiliary-field constraints remain unchanged. Thus the physical lift is
  already represented in the iterate and is not subtracted again by assembly.
- Restore the caller's constraints on both success and failure, before any
  terminal history/bulk/V publication. Existing rollback continues to discard
  the private iterate and accepted nonlinear updates on failure.

The existing matrix-rebuild policy is retained. An intermediate attempt to
skip matrix rebuilds for residual-only evaluation hit the common Stokes
assembler's debug precondition whenever prescribed boundaries are configured.
That incidental optimization was removed; no common assembler assertion or
parameter meaning was changed.

The new boundary fixtures check first-iterate physical values, homogeneous
Stokes assembly constraints, and final prescribed velocities. They use 32²
cells, ell=0.3125 m, one initial and one real step, and a discontinuous change
at positive time so the test does not rely on the CFL-selected timestep being
large. The small CFL only limits particle motion; it does not reduce the
prescribed velocity change. The existing rollback fixture now has a nonzero
prescribed velocity and retains its checks of restored bulk, V and all
particle/surface histories after accepted Newton updates.

## 2. Stale coupling fixtures

Only `Max nonlinear iterations = 50` was added to the phase solver sections
of these three inputs:

- `tests/phase_field_fault_condensed_dynamic.prm`
- `tests/phase_field_fault_condensed_adiabatic.prm`
- `tests/phase_field_fault_surface_dynamic_pressure.prm`

The last is inherited by the remaining surface coupling fixtures. The
original phase linear tolerance 2e-7 and nonlinear tolerance 1e-5, all
verification assertions, and expected outputs are unchanged. Converged
initialization takes 20 iterations and reaches 9.395e-6, consistent with the
K0 temporary-overlay diagnosis. These are now tests of the actual checked-in
inputs, not overlays.

## 3. K1 parameter and fixture contract

The K0 parameter proposal is not silently retuned in response to the new
initial-root issue:

| Quantity | Proposed value (SI) |
| --- | --- |
| Domain | (0,0.25) × (-0.5,0.5) m; x-periodic bulk |
| Bulk mesh / particles | 16 × 64 square cells; 3 × 3 particles per cell |
| Fault | Open segment (0,0) to (0.25,0); nominal surface spacing 0.03125 m |
| Phase profile | AT1, ell=0.15625 m, prescribed core 0.6, curvature 1 |
| Activation | Normal production activation 0.1, not the old 1e-6 fixture |
| Bulk/surface G and eta | 1e6 Pa and 1e8 Pa s; relaxation time 100 s |
| Creep viscosity cutoffs | Minimum 1e6, maximum 1e10 Pa s |
| Cohesion / fracture energy | 1000 Pa / 80/3 J m^-2; derived degradation m=128 |
| Temperature | Constant 293 K; static temperature; zero thermal exponent |
| Initial Maxwell stress | tau_xx=tau_yy=0; tau_xy=1500 Pa, explicitly particle-mapped |
| Initial Theta | 200 s, explicitly mapped to `phase field fault state[0]` |
| Friction | Regularized rate/state, mu0=0.6, a=0.025, b=0.013, Dc=0.001 m |
| Vref / Vmin | 1e-5 / 1e-12 m s^-1 |
| Radiation coefficient | 1e5 Pa s m^-1 |
| Normal stress | Adiabatic pressure 1000 Pa, zero gravity; dynamic-pressure gauge separate |
| Velocity boundary conditions | top/bottom ux=±U(t)/2, uy=0; U=1e-4[1+0.2 min(t/4,1)] m s^-1 |
| Artificial initial interval | 2 s; does not advance retained Theta/H/cohesive/Maxwell histories |
| Planned pilot time | End 6 s; timestep cap 2 s, actual accepted steps drive reference |
| Phase algebraic tolerances | Linear 2e-7; nonlinear 1e-8; maximum 50 iterations |
| Coupled algebraic tolerances | Nonlinear 1e-8; Stokes linear 1e-9; maximum 30 iterations |
| I_h tolerances | Quadrature and tail each 1e-10 |

`PhaseFieldFault::compute_creep_viscosity()` clips each material eta before
averaging and Maxwell reduction. `compute_maxwell_coefficients()` then uses
kappa=-eta*expm1(-dt*G/eta); neither this operation nor the ordinary/coupling
assemblers apply the eta cutoff to kappa. Thus:

| dt (s) | eta (Pa s) | kappa (Pa s) |
| ---: | ---: | ---: |
| 2 | 1e8 | 1,980,132.6693 |
| 1 | 1e8 | 995,016.6251 |
| 0.5 | 1e8 | 498,752.0807 |

In particular the last two effective viscosities are legitimately below the
minimum creep viscosity. This does not authorize an additional cutoff.

The intended benchmark-only phase freeze is precise but **not implemented**:
solve/reconstruct normally at step zero; save the converged Q1 phi values;
at k>0 use `post_constraints_creation` to prescribe every independent phi DoF
to that saved value, retaining existing periodic/hanging-node relations for
dependent DoFs. Touch no velocity, pressure, temperature or composition
constraints. With no free phi test functions and an already feasible saved
field, the ordinary phase solve has no phase correction to perform. Check
exact phi preservation in the diagnostic plugin. `Evolve phase field=false`
continues to mean frozen H; it is not repurposed to mean frozen phi.

The dimensional comparison scales and absolute allowances remain:

| Observable | Diagnostic scale | Absolute allowance |
| --- | ---: | ---: |
| V and bulk velocity | 1e-4 m/s | 1e-9 m/s |
| Theta trajectory | 200 s | 0.002 s |
| Stress/cohesive traction | 1500 Pa | 0.015 Pa |
| Accumulated slip | 6e-4 m | 6e-9 m |

The pilot trajectory target is 2% plus these allowances, not a solver
tolerance. Independent roots require traction residual <=1e-7 Pa; independent
I_h quadrature changes must be <=1e-9 m and <=1e-10 relatively. The isolated
Theta-update check uses 2e-8 s and must resolve an increment >100 times that
allowance. Geometry, containment, endpoint support, particle-domain volumes,
along-fault variation and integrated slip must be measured before using these
trajectory targets. No periodic Voronoi behavior or no-wrap behavior has been
assumed or verified for K1.

## 4. Initial-root gate: an unresolved initialization decision

The K0 initial root screen implicitly set the previous profile equal to the
converged initial profile. That assumption does not match the source:

- `set_initial_temperature_and_compositional_fields()` initializes only
  temperature/compositions in old_solution; the phase block remains zero.
- The timestep-zero phase solve updates solution, not old_solution.
- Both `ReconstructedFaultSurfaceSystem` and the bulk Stokes coupling sample
  their previous phi from old_solution, including at timestep zero.
- Cohesive initialization stores current I_h as the previous snapshot, but
  does not set the old FE phase block.

Let J_previous=integral h(phi_previous) dzeta. This is **not** the stored
previous-I_h snapshot, which is I_h at initialization. For a homogeneous
fixed current profile, the integrated history correction is
```
    integral upsilon_hist dzeta = beta C0 (I_h - J_previous) / kappa.
```
Consequently the initial scalar surface residual contains the additional
term -beta C0 (I_h - J_previous)/W. Its derivative is
\[
    F'(V) = -\kappa/W - \kappa/I_h - \sigma \mu'(V) - \eta_d < 0.
\]
The independent stationary-profile calculation gives half support
0.308821593907079 m, I_h=108.652564417493 m and ideal C0=316.227766016838 Pa.
With the proposed initial loading and 2 s numerical interval:

| Previous profile assumption | F(V_min) (Pa) | Interior root |
| --- | ---: | --- |
| Previous phi equals initial phi (K0 assumption) | +1152.2867142757 | 3.15965350117990e-4 m/s |
| Previous phi is zero (source initialization) | -32526.3180568604 | None; F is strictly decreasing |

The extra integrated history slip is 0.0170082567157612 m/s, much larger than
the imposed velocity difference 1e-4 m/s. It also invalidates the previous
no-wrap estimate based only on boundary speeds. These are **ideal-profile
preflight values**, not measured Q1/CPDI pilot data. They invalidate the
earlier reference root screen; an actual production-profile root and scalar
trajectory comparison have not been claimed.

`benchmarks/reconstructed_fault/uniform_shear/reference.py` implements this
independent scalar reduction, strict interior-root bracketing, separate
initial evaluated/retained histories, and real-step self-advancement. Its
default preflight exits 2 with the explicit no-interior-root diagnostic.
The `--previous-profile initial` option is explicitly counterfactual, not a
benchmark bypass. Three small reference tests verify rejection of the
non-interior case, initial history retention, and self-advanced real-step
ageing history with an observable increment.

Resolving this requires an explicit initialization choice: use the converged
initial profile as the previous mechanical profile at timestep zero, or
retain the current evaluation and explicitly revise K1 to admit an initial
active lower-bound solution. Neither choice has been silently implemented.
The approved Theta0/H0/Maxwell/cohesive/I_h history-retention rules remain
unchanged. No K1 mechanical pilot, convergence matrix or K2 run has begun.

## 5. Verification record

Build command: `cmake --build build-pf-cpdi --target aspect -j4` (Debug,
Voro++ enabled). New test plugins were also built with -j4. MPI runs require
socket access outside this sandbox; an initial sandbox socket failure is
infrastructure failure, not a numerical result.

The first combined focused command was:

```sh
ctest --test-dir build-pf-cpdi/tests --output-on-failure -R '^phase_field_fault_(prescribed_velocity(_later)?|stage_i_rollback|surface_(dynamic_pressure|adiabatic_pressure|rate_dependent|singular|uninitialized_adiabatic)|condensed_(dynamic|adiabatic))$' -j1
```

It reported 7/10 passes in 987.96 s. The three failures were the intermediate
matrix-rebuild debug precondition, not convergence failures. All seven
refreshed coupling fixtures passed with unchanged expected outputs:

| Test suffix (`phase_field_fault_`) | Ranks | Result / CTest wall time |
| --- | ---: | --- |
| condensed_adiabatic | 2 | PASS, 80.44 s |
| condensed_dynamic | 1 | PASS, 117.41 s |
| surface_adiabatic_pressure | 2 | PASS, 92.87 s |
| surface_dynamic_pressure | 1 | PASS, 130.82 s |
| surface_rate_dependent | 1 | PASS, 122.92 s |
| surface_singular | 1 | PASS, 164.55 s |
| surface_uninitialized_adiabatic | 1 | PASS, 179.88 s |

After retaining the matrix-rebuild policy:

```sh
ctest --test-dir build-pf-cpdi/tests --output-on-failure -R '^phase_field_fault_(prescribed_velocity(_later)?|stage_i_rollback)$' -j1
```

The rollback test passed in 67.71 s: two accepted Newton updates, then forced
failure, followed by verified restoration including the private nonzero
boundary lift. Both boundary applications passed all numerical assertions
(first residual boundary error zero at both steps, homogeneous Stokes
constraints, final boundary values), but their new output filter initially
missed the postprocessor label because its colon was missing. The label was
corrected without changing assertions or golden values. Their final command:

```sh
ctest --test-dir build-pf-cpdi/tests --output-on-failure -R '^phase_field_fault_prescribed_velocity(_later)?$' -j1
```

Pass: **2/2**, 164.73 s total. The one-rank initial/changed-loading test passes
in 92.75 s and the two-rank initially-zero/changed-loading test in 71.97 s.
Together with the successful rollback rerun and seven coupling passes, all
ten selected tests have successful final results across the recorded runs;
this is not a claim that the first combined invocation passed 10/10.

The final executable SHA-256 is
`884164bcde8958d2259a59863ef981efd40be92c5225ba9e9ca1eaec57ff23b5`.
`git diff --check` passes. The tracked patch and recoverable reference bytecode
are also saved under the prerequisite artifact directory; generated bytecode
is not part of the repository changes.

Independent reference tests:

```sh
python3 -m unittest discover -s benchmarks/reconstructed_fault/uniform_shear -p test_reference.py -v
```

Pass: 3 tests. `reference.py` exits 2 for the current previous-profile
semantics; `reference.py --previous-profile initial` exits 0 for the expressly
counterfactual comparison. `build-pf-cpdi/aspect --test 'Stage-I*'` passes
31 assertions in 7 cases. No complete ASPECT suite has been run.

K1 runtime, peak memory, reconstructed geometry, independent production FE
I_h, integrated-slip normalization, endpoint support, particle-domain volume
statistics and trajectory errors are **not measured**: the initial root gate
has not passed. The previous 2–6 minute / 0.5–2 GiB pilot estimates remain
estimates, not results.

## Changed files and handoff

- `source/simulator/solver.cc`: private physical lift, homogeneous Stokes
  constraint use, restoration; 22 added lines. No public API or parameter added.
- `doc/reconstructed_fault/current_design.md` and `specification.tex`: document
  the physical-base/homogeneous-direction and rollback contract.
- The three coupling `.prm` files listed in section 2: iteration budgets only.
- `tests/phase_field_fault_stage_i_rollback.prm`: nonzero boundary loading to
  exercise rollback of the private lift; original assertions/output unchanged.
- `tests/phase_field_fault_prescribed_velocity{,_later}.{cc,prm,sh}` and each
  matching `screen-output`: first-residual/constraint/final-boundary regressions,
  on one and two ranks.
- `benchmarks/reconstructed_fault/uniform_shear/{reference.py,test_reference.py,README.md}`:
  independent root/history preflight, its three tests and limitations.
- This report and the follow-up note in `stage_K_progress.md`: evidence and
  correction of the historical initial-root assumption.

Tracked diff summary (new files above are not staged and therefore are not
included in ordinary `git diff --stat`): **7 files, 54 insertions, 0 deletions**.
No existing test expected output has been refreshed or loosened. No Stage-J
history equation, history-retention rule, phase iteration tolerance, or
physical/numerical parameter meaning has been changed. The work remains
uncommitted, pending the initialization decision and K1 review gate.
