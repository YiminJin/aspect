# Moving-particle BP3: ownership and update audit

2026-10-14; `fault-propagation`, `c33edaaeb145e5d3dce155bb815552ef80efc5f9`
plus incoming P2/BP3 edits and the separate deal.II 9.7 printing fix.
This is a source/equation audit, before implementation. The stationary input,
package, executables and completed results are preserved. No simulation is run.

The ownership requested by the user is compatible with the existing
particle-stress path. Two changes cannot be replaced by selecting `rk2`:
the new path needs one common 3D deviatoric convention, including its tangents;
and its rotation law must be explicitly identified. The current written model
**neglects objective rotation**, and the custom code does exactly that. An
objective rotating-material extension is a physical-model decision, not an
existing ASPECT feature that this custom path automatically inherits.

| Quantity | Owner and treatment in the moving formulation | Already present |
|---|---|---|
| Current strain rate | Direct gradient of current bulk FE velocity; apply the same plane-strain 3D deviator everywhere | Direct FE gradients at bulk QPs and particle positions; convention differs by backend |
| Bulk perturbation stress | One committed symmetric tensor per moving particle; reconstructed FE fields are disposable assembly inputs | `maxwell stress`, explicit component mappings, native migration and post-convergence candidate publication |
| H and other particle properties | Existing particle storage and CPDI access; frozen BP3 H stays attached to surviving particles | BP3 H initializer/audit, generic property transfers and CPDI |
| Composition | Existing particle storage/access; ordinary material labels advect, while the explicitly prescribed BP3 strengthening field is spatially refreshed | `InitialComposition` already refreshes only the selected strengthening field after advection |
| Fault V, theta and cumulative slip | Existing manager/surface nonlinear solve, once-only aging and accepted-step slip ledger | No particle advection of surface state |
| Phase and fault geometry | Prescribed Eulerian BP3 profile and fixed fault, including automatic endpoint completion | Mature phase bypass and prescribed geometry already present |
| Particle domains | Reconstruct Voronoi domains/CPDI from moved, managed owned/ghost particles | Native rebuild after ghost exchange; generation-based association invalidation |

## Actual ordering and reconstruction

1. `Simulator::start_timestep()` backs up particle locations/properties and RNG
   state. The selected `solve_single_advection_iterated_newton_stokes()` retains
   prescribed mature phase, solves temperature, then calls
   `assemble_and_solve_composition()` **before** coupled mechanics.
2. Composition preparation calls the particle restore/operator-split hooks and
   `Particle::Manager::advance_timestep()`. With RK2, positions move in two
   stages, with cell/rank sorting after each stage. Velocity inputs are
   `old_solution` and `current_linearization_point` available **before** the new
   mechanical solve, not the subsequently accepted new velocity. This is a
   lagged transport/mechanics split; RK2 alone does not establish second-order
   accuracy of the coupled histories. The custom Maxwell property has no hook
   that updates or rotates stress during this transport.
3. The manager applies configured particle-count controls, initializes births,
   updates automatic properties (including spatial strengthening refresh),
   exchanges ghosts, then regenerates particle domains/CPDI. The domains are
   a reconstructed integration partition, not material corners evolved by a
   deformation gradient. Association validity checks IDs, locations, volume and
   domain geometry version. Restoring particles alone does not rebuild domains;
   the normal repeated-advection path rebuilds before assembly access.
4. `Simulator::interpolate_particle_properties()` reconstructs the mapped stress
   fields using the selected particle interpolator at composition FE support
   points. For continuous fields, incident-cell proposals are MPI ADD/count
   averaged; constraints are distributed. These working FE coefficients are
   transported **old** history, even though they reside in `solution` and the
   working linearization vector. They are not another constitutive owner.
5. `prepare_reconstructed_fault_mechanical_solve()` prepares particle-to-fault
   chemical projections, surface temperature, normalization and existing state.
   Mature phase does not evolve from transported H. The existing normalization
   cache can reuse fixed phase/geometry and composition-independent degradation;
   surface friction fractions must still be refreshed from the current samples.
6. Throughout Newton and line searches, old particle tensors, their reconstructed
   FE fields and incoming surface theta remain frozen. Current strain comes
   directly from the trial FE velocity. Bulk history is sampled at every bulk
   QP by `ReconstructedFaultStokes::execute()` from mapped stress coefficients
   and enters the frozen load as beta times old stress, exactly once.
   **The maintained BP3 plugin selects `assemble_bulk_work_system()`: its fault
   residual reads those same FE coefficients at the same bulk QPs**, with direct
   FE strain/pressure, localization weights and the 20 m filter. It does not
   directly use one particle tensor per fault QP. The generic alternative
   `assemble_particle_system()` uses parent particle stress and FE gradients
   sampled at parent positions over CPDI domain quadrature; that is not this
   filtered BP3 path. Do not conflate these two surface measures.
7. On coupled convergence, `solve_reconstructed_fault_stokes()` constructs an
   accepted bulk vector and validates V. `commit_reconstructed_fault_mechanical_history()`
   samples its velocity gradients at the current particle positions, reads each
   particle's own old tensor, constructs tensor/H and nodal theta/cohesive
   candidates, and collectively validates before writing them. It then commits
   V and swaps the accepted bulk/linearization vectors. There is no second
   particle advection after this solve. Step zero skips the physical history
   update; its initial constitutive interval is kinematic, not elapsed slip time.
   A mechanics exception rolls back its bulk/V trials, not the preceding motion,
   births or spatial refresh. An in-memory retry must also restore the particle
   backup/RNG and rebuild domains/associations before their next use. The normal
   outer repeat path restores particles; this is not a complete post-publication
   surface-history rollback or automatic-cutback qualification.
8. The BP3 accepted-step output integrates cumulative slip once as dt times
   committed V (zero increment at step zero), guarded by consecutive accepted
   step numbers. It reads the captured pre-publication traction residual rather
   than recomputing Maxwell stress from newly committed histories. The mapped FE
   stress fields continue to represent incoming history until the next transfer.
   Postprocessors must not refresh them or advance constitutive state again.

Locations: `source/simulator/{core,solver_schemes,initial_conditions}.cc`,
`source/particle/{manager,particle_domain}.cc`,
`source/particle/property/{maxwell_stress,initial_composition,interface}.cc`,
`source/material_model/phase_field_fault/{history,constitutive,normalization}.cc`,
`source/reconstructed_fault/{manager_particle_projection,surface_system_bulk_work,surface_system_particle}.cc`,
`source/simulator/solver/reconstructed_fault_stokes.cc`, and maintained
`bp3/plugin/{bp3,particle_history,output}.cc`.

## One consistent moving discretization of the current non-rotational model

For this campaign use the in-plane block of the 3D plane-strain deviator,

\[
 D_3(A)=A-\tfrac13\operatorname{tr}_2(A)I_2,\quad
 D_b=D_3(\operatorname{symgrad}u_h)-\chi V_\Gamma S,\quad
 \chi=h(\phi)/I_h,\quad S=\operatorname{sym}(t\otimes n).
\]

The actual stored shear sense multiplies S; use the existing source map,
including constant endpoint continuation. For a stored deviatoric in-plane
tensor T, the missing component is Tzz=-tr2(T). Do not apply D3 again to old T
or to the trace-free slip tensor. Include Tzz in physical stress invariants and
storage diagnostics; keep physical pressure distinct from tensor trace.

Let the existing RK2 split map particle positions from Xp^n to Xp*, and let
bar(Tp^n) denote their transported old values, including explicit late-particle
initialization. In the declared non-rotational model surviving components are
unchanged by transport. Let Ph reconstruct these values for assembly. With
beta=exp(-dt G/eta), kappa=-eta expm1(-dt G/eta), the single-owner update is

\[
 T_h^{\rm trial}(x)=\beta P_h[\bar T_p^n](x)+2\kappa D_b[u_h,V](x),
 \qquad
 T_p^{n+1}=\beta\bar T_p^n+2\kappa D_b[u_h^{n+1},V^{n+1}](X_p^*).
\]

This is the split discretization of DT/Dt+(G/eta)T=2G Db **without spin**.
G/eta and the rate source are held fixed in its constitutive interval. The
particle update does not read Ph back as its old tensor. Ph is generally not
exact at arbitrary particle points: particle/FE reconstruction error remains,
but repeated FE-to-particle replacement of the old tensor is not introduced.
There is no authoritative d_R or FE Z in this moving branch. Past mature slip
subtraction is already contained in the particle tensor; do not add the FE
branch's separate slip-memory reconstruction and count it twice. Mature
cohesive traction stays zero while surface theta and slip continue to evolve.
Fresh bulk perturbation stress stays zero; existing background shear/normal
tractions remain surface data.

With incoming history frozen, the common constitutive tangent must be

\[
 \delta T=2\kappa[D_3(\operatorname{symgrad}\delta u)
                  -\chi\delta V S],\qquad
 \delta\sigma_n=\delta p-\delta T:N,\quad N=n\otimes n.
\]

This determines the bulk trace term, B/G actions, friction tangent and filtered
normal derivative together; no pressure reinterpretation is needed. The current
particle branch uses full symgrad in both point responses and particle updates;
the FE branch uses D3. Bulk Newton/Picard trace-term admission is also currently
tied to `uses_fe_strain_history()`. Those conditions must be separated from
history ownership for the selected consistent particle formulation.

**Additional source-level tangent gap:** `surface_system.cc` currently uses raw
symgrad in matrix-free G, filtered normal G, and explicit G construction even
when the point response uses D3. With delta D=I2, delta p=0, n=ey, kappa=2 and
mu=.6, the deviatoric response derivative is .8 while the existing raw-gradient
G expression gives 2.4. For trace-free directions this discrepancy vanishes;
weak incompressibility does not make every FE direction pointwise trace-free.
This is an algebra/source finding, not a newly executed native residual test.
Preserve the previous passing recurrence/trajectory evidence; it does not
qualify this off-constraint tangent. Repair and check it before new mechanics.

## Rotation finding and precise model boundary

`current_design.md` section 19 explicitly specifies a non-rotational law;
`specification.tex` Stage B agrees; `pf_rsf.tex` near equation `maxwell` says
rotation terms in the objective rate are neglected. `MaxwellStress` performs no
automatic update, and `compute_maxwell_stress()`, fault assembly and terminal
particle publication contain no W, stress commutator or corotation. RK2 moves
positions only. The custom implementation therefore matches the written
no-rotation assumption, **not** an objective rotating-stress law.

ASPECT's separate `Rheology::Elasticity::fill_reaction_outputs()` uses
W=(grad u-grad u^T)/2 and the increment dt sym(W T-T W). Its `elastic stress`
particle plugin connects a pre-advection update to the restore hook, requires
the viscoelastic/viscoplastic material and a different current/old property
layout. None of this is called by `maxwell stress`/`PhaseFieldFault`. Enabling
the standard plugin would not preserve this model's coefficients, ownership
or once-only terminal publication.

If objective material rotation is intended, an example governing replacement is
the Jaumann equation DT/Dt-W T+T W+(G/eta)T=2G Db. It is **not selected here**:
the authoritative model currently excludes that term. For rigid rotation and
Db=0 it requires beta Q Tn Q^T, Q=exp(dt W); the current code returns beta Tn.
For nonzero spin and rate, simply rotating old T then adding 2 kappa Db is a
split approximation, not the general exact rotating Maxwell solution. The
rotation's time level must also be specified: lagged transport spin can be a
frozen input, whereas current-trial spin requires its derivative in bulk/G.

There is a second scope question for such an extension: the particle stores
**perturbation** stress, and BP3 supplies only surface background tractions.
Objectively rotating total prestress would require a defined bulk background
tensor and its evolution. For Ttotal=Tbg+Tperturb, its commutator contains
[W,Tbg] as well as [W,Tperturb]; the surface scalars do not determine the former.
Do not silently invent that tensor, rotate the surface loads as material
history, or claim that rotation of perturbations alone rotates full prestress.
Until this choice is resolved, only the explicitly non-rotational split above
is determined by the current scientific contract.

## Necessary changes before a moving BP3 test

| Change | Smallest affected locations / existing functionality to retain |
|---|---|
| Select particle ownership and ordinary RK2 in a **new** fresh input | Use existing `Bulk history representation = particle stress`; do not relax FE-history stationary guards or change the retained stationary package. No d_R allocation/publication or FE-history conversion. |
| Make the chosen D3 convention independent of FE ownership | Point response and terminal update in material `constitutive.cc`/`history.cc`; admit existing trace-subtraction assemblers in simulator `assembly.cc`/`newton.cc`, including the applicable preconditioner. Preserve legacy results/defaults through an explicit scoped selection. |
| Correct every derivative of the selected response | All G paths in `surface_system.cc`, including filter and explicit construction; retain existing B/slip and pressure signs. Add spin derivatives only if a current-spin model is explicitly selected. |
| Decide rotation law and stress/background scope before implementing rotation | Update authoritative equations first if objective rotation is selected; use one time-discrete update for assembly and terminal publication. Never add an unrelated automatic elastic-stress update. |
| Preserve current property/geometry treatment | Existing prescribed mature phase, automatic completion, consistent normalization, spatial strengthening refresh, transported survivor H/composition and surface theta/slip. New particle H already has the named frozen-BP3 spatial initializer; Maxwell births interpolate transported old stress once before the new interval update. |
| Exercise domains, population and rejection in the selected moving path | Reuse native RK2, migration, RNG backup, late initialization, count bounds and post-ghost domain rebuild. Verify rebuild before any access after restore, valid support after motion/insertion/deletion, synchronized ghosts and one candidate/commit per surviving or born particle. Existing mechanics rollback covers V/trials; complete-step post-commit cutback is not established. |
| Update affected observations and add focused gates | Keep observer-only traction snapshots/slip publication. Check nonzero-divergence/3D trace and all G finite differences; translation of nonuniform stress; rigid rotation according to the **selected** law; coupled Maxwell/aging/slip once; cell/rank crossing, births/deletion/domain coverage, and prepublication failure/retry. Retain physical inputs and tolerances. |

The current effective q6 option applies only to accepted-FE mechanics; the
particle branch falls back to the native q3 rule. A new matched comparison must
check/select one actual common rule across all coupled consumers rather than
assuming the FE parameter sets particle quadrature. No quadrature or projector
redesign is authorized by this audit. The current package has population changes
disabled; ordinary count/birth infrastructure already exists, but its final
moving-campaign policy and genuine inflow history must be recorded and tested.
Interpolation is not an independently prescribed inflow stress condition.

## Checks and disposition

Source audit and two algebraic illustrations pass: the derivative values above,
and beta=.8, rotation angle=.1 rad applied to diag(1,-1), which gives objective
shear .158935464636049 versus the custom law's zero. These arithmetic checks
illustrate the equations; they do not execute or qualify ASPECT mechanics.
No build, mechanical trajectory, transport/rotation regression, server job,
restart or automatic cutback was run for this documentation task.

Evidence: `build-bp3-native/moving-particle-audit/{incoming.json,incoming.diff,
incoming-status.txt,algebra.json,preservation.json}`. Source/configuration
hashes and incoming paths verify preservation. Only this note and the two
recovery/status documents change. The next implementation scope should resolve
the no-rotation versus objective-law decision and stress/background definition,
then make the scoped convention/tangent corrections and bounded moving-history
gates. The first-event server input remains stationary and must not be relabeled
as a moving-particle qualification.
