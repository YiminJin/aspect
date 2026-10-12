# P0 — baseline and first experiment

Completed 2026-10-05, on the existing worktree/branch as requested. **P0 passes
as an audit and bounded P1 proposal; evolving-field correctness is unverified.**
No production code, defaults, inputs, build, simulation, commit, or push changed.
All implementation conclusions below are source inferences unless explicitly
identified as executed checks. P1–P6 have not been executed by this task.

## Baseline and instructions

| Item | Evidence |
|---|---|
| Worktree / branch | `/home/ein/repository/aspect-fault-propagation`, `fault-propagation` |
| HEAD and selected baseline | `de7beaa65846c7372711f7922abf1c5d9a745ee6` (`Refactoring closeup`) |
| Other worktrees | `aspect-pf-rsf-refactor` at the same commit; `aspect` / `pf-rsf` at `3af7c2aa2`; left untouched |
| Origin | `https://github.com/YiminJin/aspect`; no fetch needed for this local baseline |
| Incoming changes | Modified root `AGENTS.md`; untracked propagation handoff directory. No tracked production patch |
| Refactor checkout | No tracked diff; untracked BP3 packages, revised input, instructions and notes remain there; not imported |
| Task build/output | None; no directories or worktrees created |
| Processes | No ASPECT/MPI/build job seen in the sandbox-visible process listing; this does not establish host/cluster idleness |

Read the supplied task instructions, PLAN, STATUS, START_HERE, the committed
root guidance (`git show HEAD:AGENTS.md`), relevant `current_design.md` and
`specification.tex` sections, `refactoring.md`, and the current-status closeout.
No ancestor or nested `AGENTS.override.md`/additional applicable `AGENTS.md`
was found. The incoming root file **replaces** the 83-line committed guidance,
contrary to its own “additive task block” instruction. Both were read; that
incoming file is preserved, not silently repaired. Original rules about the
authoritative design, no legacy core-phase architecture, and stage boundaries
still inform this audit. “M1/M2 frozen” describes refactoring scope, not a ban
on auditing M2 in the explicitly selected propagation task.

The user explicitly superseded requests for new workspaces/directories.
`START_HERE.md` also names `benchmarks/.../propagation/`, whereas the installed
PLAN/STATUS and current user request select `doc/.../propagation/`; retain the
installed location. No scientific contract was silently changed. Two roadmap
assumptions require correction: phase active-set constraints are absent, and
the approved coupling is sequential, not a joint phase/mechanical iteration.
For later P3, the established geometry rule is *committed vertices never move*
(`current_design.md` §3; specification “Append-only persistent geometry”);
bulk particle transport must not be confused with fault-vertex advection.

### Executable provenance, checked without running ASPECT

Preserved executable:
`/home/ein/repository/aspect-pf-rsf-refactor/benchmarks/reconstructed_fault/refactoring_r7b/build/aspect-r7b-qualified`.

SHA256 (freshly checked):
`ed74d4d4ad1cfa5d1bd4fbade496d9748ab298c24a259a4612a94d4df375b60d`.
Its [manifest](../../../benchmarks/reconstructed_fault/refactoring_r7b/evidence/qualified-manifest.json)
identifies source `6a3781277f2f594d52cd248d84f5b6fdc98fdeb1`, Release,
GCC 12.4.0 / OpenMPI 5.0.6 / deal.II 9.6.2 / Trilinos 14.2 Epetra,
Voro enabled, unity/PCH ON. Fresh inspection of the sibling
`build-refactor-post-r6-gcc12-unity/CMakeCache.txt` confirms the source directory,
MPI compiler wrapper, GCC launcher, deal.II prefix, Voro and Epetra choices.
Compiler output and deal.II installed headers confirm their versions; MPI's
version command prints 5.0.6 plus a sandbox socket warning. Trilinos version
comes from the manifest, not a fresh runtime probe.

`git diff 6a3781277 HEAD -- source include` is empty. The corresponding CMake
diff is removal of the 53 package-check lines, already documented in closeout.
Thus this is a **matching production-source reference binary**, not an
executable built at HEAD or a task-owned build. No other session's build was
reconfigured. Historical R7 checks and the 9.22-s model-only closeout smoke are
recorded evidence, not P0 reruns or evolving-field qualification.

## Source path and state map

Paths below are relative to the repository root. These are the relevant
production paths, not the deprecated `tmp/...phase_field_rsf` implementation.

| Operation | Real entry point and behavior |
|---|---|
| FE/material convention | `source/simulator/phase_field.cc`: `PhaseFieldHandler::edit_finite_element_variables` declares Q1 phase; `parse_parameters`, `GeometricFunction`, `DegradationFunction`. AT1 has `alpha(phi)=phi`, `c0=8/3`; `g=(1-phi)^2/[(1-phi)^2+m phi(1+p phi)]`, `m=Gc/(c0 ell Hc)`. `PhaseFieldFault::get_critical_crack_driving_forces` in `source/material_model/phase_field_fault.cc:136` returns `cohesion^2/(2G)` |
| Initial particle H | `source/particle/property/crack_driving_force.cc`: `CrackDrivingForce::initialize_one_particle_property` initializes mixture Hc. `source/reconstructed_fault/manager.cc:259` connects `post_set_initial_state` after particle initialization; `initialize_crack_driving_force` substitutes stationary-profile H where the prescribed polyline contributes; overlapping contributions fail |
| Seed and cap | Closest-point distance plus interpolated prescribed core value supplies `get_phase_field_profiles` and `stationary_crack_driving_force`. The endpoint's rounded diffuse cap is initial data. The input core value is not a transported irreversible peak field |
| Phase solve | `source/simulator/solver_schemes.cc:954,1015,1097`: supported single-advection schemes call `evolve_phase_field`; reconstruct only at timestep zero. `source/simulator/phase_field.cc:819` assembles directly from particle H/compositions and CPDI weights/gradients, not an interpolated nodal H equation |
| Phase algebra | `assemble_phase_field_system` accumulates `rhs_i=-sum_p Vp [w_i K + F grad(w_i).grad(phi)]`, with `K=sum_m f_m[H g'_m+E_m alpha']`, `F=sum_m f_m 2E_m ell^2`, `E_m=Gc_m/(c0 ell)`. `solve_phase_field_system` uses CG/AMG; `evolve_phase_field` uses Newton and residual line search |
| Particle/FE order | `solve_single_advection_iterated_newton_stokes` solves phi from incoming H, then temperature and composition. `assemble_and_solve_composition` restores particles for repeated composition iterations, advects/manages particles, and transfers mapped stress/composition fields before mechanics. CPDI domains and FE point sampling are distinct representations |
| Mechanical preparation | `source/material_model/phase_field_fault/history.cc:291`: `prepare_reconstructed_fault_mechanical_solve` prepares projected surface material/temperature, current Ih, cohesive history and Theta/V initialization. `source/simulator/solver/reconstructed_fault_stokes.cc:389` opens the coupled mechanical solve and prepares QP association |
| Mechanical trials | Manager `begin_slip_rate_nonlinear_solve`, `begin_slip_rate_trial`, accept/rollback APIs in `source/reconstructed_fault/manager_slip_rate.cc`; current/trial V are separate from timestep-committed V. Phi, geometry and input histories are fixed throughout this solve. Surface residual/Jacobian and condensed Stokes blocks are reassembled for the current mechanical linearization |
| History publication | `history.cc:539`: `commit_reconstructed_fault_mechanical_history` returns at timestep zero. Later, it samples accepted fields, constructs particle stress/H, projected cohesive traction and vertex Theta candidates, validates collectively, then publishes. `reconstructed_fault_stokes.cc:784` performs terminal commit only after convergence; no physical history accumulation per Newton/line-search iteration |
| H rule | `constitutive.cc:293`: finite-step cohesive-work candidate uses accepted traction, old traction, old h, current g and Maxwell coefficients. `history.cc:860` commits `max(old_H, candidate)` only when `evolve_phase_field` and an association are active. It is a maximum of candidates, not a sum over iterations |
| No association | `history.cc:804–878` leaves H unchanged for unassociated particles (Maxwell stress still updates). The intact limit at `g=1` in the cohesive routine requires a fault association/traction; it is **not** an implemented bulk intact-strength/direction activation law. This is a later propagation blocker |
| Restart | `source/simulator/checkpoint_restart.cc` saves/restores bulk solution vectors and manager state; native particles retain H/stress. `include/aspect/reconstructed_fault/manager.h:385` saves geometry, widths, generic properties, committed V and shear sense; load rebuilds transient caches/current V. `history.cc:235` invalidates normalization on resume and checks cohesive/mature mode identity |
| Failure | Phase exhaustion throws `ExcNonlinearSolverNoConvergence`; `core.cc:2127` applies abort/continue/cutback. Phase line-search scratch does not write H, but the supplied phi vector advances through Newton and is not transactionally restored on exhaustion. Mechanics catches exceptions and restores its entry bulk solution/current V; that entry is **after** phi solved. `core.cc:2350` restores particle backup on repeated steps, including both RNG streams (`particle/manager.cc:196`). Full evolving phi/time/history rollback is not proved by these separate mechanisms |

**State distinction:** Q1 phi is solved and retained in current/old FE vectors;
particle H is initialized, transported and max-committed; particle Maxwell stress
is transported and terminally updated; fault Theta/cohesive traction/previous Ih
are committed generic vertex properties; current Ih and projection/factorization
data are rebuilt/reused transiently. No separate `phi_peak` or particle maximum
phi field was found on this path. Activation and normal-lock thresholds are
parameters, not peak histories. Mature normals come from segment tangents, not
the gradient of the seed cap.

## Frozen-field and fixed-geometry audit

| Area | Genuine restriction or valid reuse |
|---|---|
| `Evolve phase field=false` | In **cohesive** mode it freezes H updates, not the solve. Repeated phi may still adjust with changed domains/inputs. In **mature frictional** mode parsing requires false; `evolve_phase_field` bypasses equilibrium and copies old phi after initialization |
| Mature geometry / boundary completion | `history.cc:307` enforces reference-vertex equality. `boundary_completion.cc` and `normalization.cc:620` require frozen mature prescribed profiles for completion. These cannot be reused as evolving-tip or P2 boundary treatments |
| Association/support | `manager_particle_projection.cc` keys geometry versions, particle IDs/positions/domain generation and projection metadata. Fixed geometric associations are legitimate while phi changes; cached widths originate from initial profiles (`manager.cc:524`). Growing damage outside this support does not automatically enlarge it or activate intact particles |
| Normalization | `normalization.cc:186` requires exact owned phi, geometry/version, relevant surface mixture, degradation revision and mesh validity for collective Ih reuse. Phi changes cause a miss; this is not inherently a frozen-phi optimization. Geometric remote-point maps can be reused independently of values. Current h is `1/g-1`, localization `h/Ih`; previous h uses old FE phi (current phi at timestep zero) |
| Range checks | Assembly rejects the singular degradation branch and phi greater than one at CPDI samples, but allows nonsingular negative undershoot. Ih uses `max(phi,0)` for its integrand and an empirical raw-minimum guard of `-1e-4` (`phase_field_fault.h:576`); neither is a phase bound constraint or an irreversibility tolerance |
| Mechanical algebra | B/G, surface quadrature and preconditioners assume fixed phi/geometry during a mechanical solve, consistent with the approved split. Updating either inside that solve would require a new invalidation/acceptance design |
| Geometry mutation | Initial reconstruction resets state; it is not a growth operator. Low-level `ReconstructedFault::append_*` increments geometry version, but does not align all manager V/property/history arrays or provide trial-geometry rollback. The design explicitly requires manager-controlled mutation before P3 |
| Output / checkpoint | Native `postprocess/reconstructed_faults.cc` traverses current fault data and writes committed fields; it does not demonstrate propagation. Uniform-shear's default plugin freezes nodal phi after initialization; its `Evolving profile=true` diagnostic disables that. Checkpointed widths/geometry and mature-mode markers retain their original semantics |
| Particle birth | Native `CrackDrivingForce` uses default late-property interpolation (`particle/property/interface.cc:355,498`), rather than reconstructing material history from initial Hc. BP3's special frozen-H birth rule is restricted to frozen mature cases. Keep fixture `3x3`, load balancing `none` for P1; preserve established production density controls and test births/transport in P2 before drawing material-irreversibility conclusions |

## Capability and evidence

| Capability | Implemented / tested status for this task |
|---|---|
| Real CPDI phase residual, Jacobian and nonlinear solver | Present; existing tests inspect assembly, small-residual precision and intentional exhaustion. Not executed in P0 |
| Controlled-H test access | Present: `tests/phase_field_test_access.h` exposes assembly/linear solve/vertex map; public `evolve_phase_field` needs no new production hook |
| Irreversible H | Present for associated particles at accepted mechanics; not proof of pointwise phi monotonicity |
| Phase bound/irreversibility active set or KKT solver | Absent. Available residual is the equality weak residual after affine constraint elimination |
| Fixed-geometry evolving feedback | Sequential implementation present. Historical K3 reports nontrivial evolution plus timestep/support/normalization limitations; not current qualification |
| Bulk intact activation, growth birth rule, geometry transaction | Missing on audited path; deferred beyond P1/P2 |
| Evolving restart / rejected-step equivalence | Infrastructure exists; still needs task-specific execution |

Historical [K3 evidence](../../../benchmarks/reconstructed_fault/uniform_shear/evolving/README.md)
reports a 43.412-s smoke and later nonmonotone Ih/H errors; its refined reference
also reports vanishing feedback at smaller increments under the maximum rule.
Do not label the repository as never having run evolving phi, or infer present
qualification from that history. `tests/phase_field_precision.cc` transitively
includes `uniform_shear.cc`, whose `evolving/safeguards.h` is absent in both
current and refactor checkouts. No tracked history for that header was found.
This blocks direct reuse of that plugin without dependency recovery; it does
not block the self-contained P1 approach below.

## Selected P1: one controlled-H production-solver test

**Starting point:** `tests/phase_field_frozen_history.{cc,prm}` and
`phase_field_test_access.h`; use the smaller no-Stokes setup in
`tests/phase_field_fault_ih.prm` as the parameter skeleton. The frozen-history
test's four-file quoted-include closure exists. It checks a fixed probe's AT1
weak residual/Jacobian; it does **not** already implement the proposed load cycle.
Avoid inheriting the missing uniform-shear include chain.

Proposed files, only when P1 is selected: `tests/phase_field_controlled_history.cc`,
`tests/phase_field_controlled_history.prm`, its native test expected-output
fixture, and P1 report/status. Reuse the existing test friend and plugin/test
CMake mechanism; no production hook or constitutive reimplementation is needed.
An expected-output location and task-owned build/output must be established for
that later run; none is created or selected by P0's documentation-only work.

Proposed setup is a unit box, zero prescribed velocity/pressure, static
temperature 293, no Stokes solve, no time-dependent AMR, no particle movement,
`End time=0`, and cohesive mode. Retain the finite seed
`tests/phase_field_fault_ih.txt` (0.2–0.8 at y=0.5, core 0.8) and native
stationary-H initialization; set `Fit prescribed geometry to phase field=false`
to keep reconstruction from obscuring an operator test. No tip claim is made.
Use both material entries identically with the existing uniform-shear scales
`G=1e6`, cohesion `1000`, `Gc=26.666666666666667`, reference viscosity `1e8`,
and constant rock fraction 0.5. Then Hc=0.5, `ell=0.15625`, AT1, p=1, m=128;
this avoids the very stiff m=480000 Stage-I parameterization. These are
diagnostic fixture values, not production defaults or proposed propagation physics.

Start at global refinement 4: 16x16 cells, nominal 2304 particles (`3x3` per
cell), `h_cell/ell=0.4`; keep domain/CPDI generation and load balancing `none`.
No dense particle or visualization output. A single refinement to 32x32 is
permitted only to resolve profile under-resolution or material undershoot;
report realized counts and initial profile adjustment.

At the sole postprocessor execution, copy the initialized phase state and
matrix structure; temporarily substitute particle H with an exception-safe
restoration guard (as the precision test already does). All solves use
`handler.evolve_phase_field(matrix,rhs,state)` and fresh production assembly.
Snapshot particle IDs/properties, geometry and live FE state before the probe;
require restoration on normal and exceptional paths. This is a diagnostic load
sequence on scratch FE state, not physical elapsed time or mechanical history.

1. Preserve initialized `H_seed`; solve/re-solve at that identical H to separate
   initialization relaxation from hold drift. An applied subcritical candidate
   below `H_seed` leaves committed diagnostic H unchanged; never force stored H
   down to call it unloading.
2. Use a named fixed band mask based on the initialized, partially damaged
   CPDI samples (e.g. `0.1<phi_seed<0.7`), with prescribed candidates
   `1.05 H_seed`, then `1.10 H_seed` there. Each diagnostic accepted state uses
   `H=max(previous_H, applied_candidate)`; hold elsewhere. Confirm the mask
   contains multiple resolved samples and produces nontrivial phi change.
   If not, stop and diagnose resolution/load scale rather than count a pass.
3. Lower the applied candidate to `H_seed`, retaining the maximum H; re-solve
   and measure unloading/hold drift. Never invent fault V/Theta for this test.
4. Exercise one invalid degradation-branch trial through `assemble` and verify
   no history publication. For nonlinear-exhaustion coverage, retain the existing
   `phase_field_nonlinear_exhaustion` expected-failure test. Scratch restoration
   proves only probe isolation, not simulator timestep rollback.

### Diagnostics and acceptance scales

Print one compact row per solve: raw nodal and CPDI phi extrema, H extrema and
minimum H increment, phase change and maximum decrease, residual/entry residual,
fixed roundoff allowance and absolute target, iteration counts, and CPDI integral
`sum Vp * handler.crack_surface_density(phi,grad_phi)`. Log seed-to-equilibrium
change separately. Keep geometry/version and particle-ID checks exact.

Use existing fixture tolerances: nonlinear `1e-8`, linear `2e-7`, maximum 50
Newton iterations. Production target is
`T=max(1e-8 ||r_entry||_2, 8 epsilon_double ||S_entry||_2)`, where S is the
assembled positive pre-cancellation reaction/derivative/gradient scale
(`phase_field.cc:964–1012`). Reassemble r at exit against the same H and affine
constraints; require `||r_exit||_2<=T`. Report dimensional residual and T as
well as their ratio, not only solver convergence messages.

There is **no phase active-set count or constrained KKT norm to report**;
mark these unavailable. Stokes/surface slip-rate active sets are different.
If a diagnostic box-projected residual is later added, label it separately;
do not pass the solver by replacing its equality residual with a projection.
Measure physical bound violations even for nonsingular Newton states.

For drift/phi-decrease resolution, estimate a field-scale uncertainty by solving
the final production Jacobian against the residual (a correction estimate),
including the roundoff-load scale; compare with identical-H repeat differences.
A practical diagnostic allowance is ten times the larger field uncertainty and
`epsilon_double*max(1,||phi||inf)`. Require evolution comfortably above it (e.g.
100 times), and require H monotonicity exactly for retained diagnostic IDs.
This does not prove a discrete maximum principle. Bound violations or phi healing
larger than the estimated algebraic error must be reported as a P1 finding;
one mesh comparison can distinguish discretization effects. The Ih `1e-4` guard
must not be used to declare physical phi admissibility. If the production method
needs a genuine inequality treatment, report the demonstrated blocker before
selecting a scientific algorithm change.

Native restart and full failed-step restoration are deferred explicitly to P2:
this one-postprocessor scratch sequence does not represent a checkpointable
accepted physical trajectory. A second run with half-sized prescribed-H
increments is justified only by jumps or convergence sensitivity.

### Commands and budget for P1 (proposed, not executed)

Use the established [GCC12 build recipe](../../../benchmarks/reconstructed_fault/post_r6_cleanup/README.md)
if a new core build is later selected. Do not reconfigure the sibling build.
For test-only work the hash-checked R7 executable and read-only sibling Aspect
package can supply the matching production-source reference. After a task-owned
test build/output destination is available, the existing native test driver is:

```sh
export ASPECT_SOURCE_DIR="$PWD"
export OMPI_CXX=/opt/gcc/12.4.0/bin/g++
export PATH=/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:$PATH
export OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# P1_TEST_BUILD: task-owned test build path; REFERENCE_ASPECT: binary above.
# REFERENCE_ASPECT_DIR: sibling build-refactor-post-r6-gcc12-unity, read-only.
cmake -S tests -B "$P1_TEST_BUILD" \
  -DAspect_DIR="$REFERENCE_ASPECT_DIR" -DASPECT_BINARY="$REFERENCE_ASPECT" \
  -DASPECT_RUN_ALL_TESTS=ON -DTEST_TIME_LIMIT=120
ctest --test-dir "$P1_TEST_BUILD" -R '^phase_field_controlled_history$' \
  --output-on-failure -j1
```

The proposed test name/files do not yet exist. Do not run this block before
implementation and native expected-output setup. Build only selected plugin
targets; enabling discovery is not authority to run the complete suite.
Pin compiler/flags to the reference package, record plugin/input/include hashes
and resolved parameters, and retain the original executable hash. Budget one
rank, one initialization plus five or fewer controlled solves, 120 s runtime
cap, 2 GiB memory observation ceiling, and <10 MiB diagnostic output; these are
limits, not measured performance. One additional 120-s refinement or smaller-load
run only for a concrete unresolved risk. Plugin compilation is separately timed,
at most two jobs; no full build is needed merely to finish P0.

## P2 candidate and remaining decisions

The existing `uniform_shear/pilot.prm` Box (0.25x1, periodic x), through-going
horizontal `fault.txt`, strengthening RSF (`a=.025>b=.013`) and controlled
adiabatic pressure are a better P2 basis than scaled-down production BP3.
Recover/replace its missing diagnostic dependency narrowly before using it;
do not revive the historical campaign or copy its dense outputs wholesale.
Explicitly enable both material H evolution and the plugin evolving-profile
mode if that plugin is retained. Keep cohesive mode and avoid frozen-mature
automatic boundary completion. Reuse P1 initialization only after it passes.

The actual accepted cycle is `H_(k-1) -> phi_k -> mechanics_k -> H_k`, matching
`current_design.md` §25. P2 must show phase residual at its **input** history,
mechanical residual at its frozen phi, and quantify the next-history phase
defect separately. Requiring all equations at H_k simultaneously would change
the coupling algorithm; P0 does not authorize that. Also test advection-aware
history, births if active, complete retry/restoration, restart, support/Ih
refresh and one smaller timestep before any geometry extension. The historical
small-increment feedback loss is a specific risk, not grounds to alter H here.

## Executed checks and handoff

| Fresh P0 check | Result |
|---|---|
| HEAD/worktree/status audit | Current branch verified; production diff empty |
| R7 vs HEAD production source/header diff | 0 changed files; separate CMake difference documented |
| Preserved R7 executable SHA256 | Matches committed qualification manifest |
| Frozen-history local quoted-include closure | 4 files, 0 missing |
| Historical precision/uniform-shear dependency | Missing `evolving/safeguards.h`; not runnable as-is |
| ASPECT builds / runs | 0 / 0; no numerical result claimed |
| Final documentation checks | No whitespace errors or broken local Markdown links; production/tests/benchmarks/build configuration unchanged |

Commands used were read-only `git status/worktree/rev-parse/log/diff/show`,
targeted `rg`, `sed`, `cat`, directory/cache inspection, `sha256sum`, compiler/MPI
version queries, and a Python quoted-include/hash inspection. Representative
rechecks:

```sh
git status --short --branch
git worktree list
git rev-parse HEAD
git diff 6a3781277 HEAD --stat -- source include CMakeLists.txt
sha256sum /home/ein/repository/aspect-pf-rsf-refactor/benchmarks/reconstructed_fault/refactoring_r7b/build/aspect-r7b-qualified
```

Input identities (SHA256, existing unmodified fixtures):

```text
15eae3cef491655275e96a476ddd3ba298216dc1fb680f29378d486f4ceaa83e  tests/phase_field_frozen_history.prm
ded103571ecd4b1984568e98663898a7ea3e7edb3fc7b81677e024471a1b97ad  tests/phase_field_fault_ih.prm
664c45fdbed36d17474f8caebe8480b00389194b1172d026ea4d1917959a27fe  tests/phase_field_fault_ih.txt
90830bac91753d8985a6183dbbd683dfbd4780d76aa6468b2cb13348b39b236c  tests/phase_field_test_access.h
```

Task changes: this report, STATUS, and evidence-based PLAN corrections only.
Next proposed stage: **P1 controlled-H operator test**, stopping before P2.
