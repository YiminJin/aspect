# Current formulation, variables, timestep and energy

Audited at `27178813cebd471b93a7a9308f41215b71b4a20b` plus the explicitly selected
objective-stress changes. Source APIs, not historical plan names, determine the
entries below. Units use metres, seconds, pascals and energy per out-of-plane
metre. The [spin report](OBJECTIVE_STRESS_REPORT.md) defines the new option's
actual scope; the [BP3 review](BP3_REVIEW.md) describes the production evidence.
The [coupled Stage C report](TRANSPORTED_OBJECTIVE_STAGE_C_COUPLED_REPORT.md)
adds a restricted complete mature material step; cohesive material motion and
production readiness remain unqualified.

## One formulation table

| Route | Physical law and independent history | Geometry/fracture/pressure | Spin selection |
|---|---|---|---|
| Production mature BP3 | Exponential Maxwell response to D3−chi V S. Independent `DGHistory::trial_stress` T and Q1 `FEStrainState::Z`; physical tau=T−2chi Z S. Z/G is auxiliary slip memory, not cohesive elastic slip. | Frozen physical phase and prescribed straight geometry; no new fracture. Incompressible mixed pressure perturbation plus separate background fault traction; work-measure normal loads, optionally Helmholtz filtered for friction. | Legacy remains the production selection. The separate mature DG reaction control below has complete residual history; it is not a resumed production trajectory. |
| Mature DG reaction control | Independent T and R DGQ2 plus Q1 auxiliary Z; tau=T-R-2chi Z S. Finite complete B commutator and exponential source/relaxation. | Matched constant G/eta, frozen physical phase/geometry; accepted native pressure/work convention; restricted fixed-geometry AMR and schema6. | Experimental `Jaumann DG Lie`: accepted-gradient total spin, stationary-history reaction control. Stationary control; the separate material selector below adds transport. |
| Mature moving material experiment | Independent T/R DGQ2, labelled Q1 auxiliary Z and age, transported physical profile/obstacle; tau=T-R-2chi Z S. SSPRK3 material transport, total-spin finite rotation and exponential reaction. | Constant G, fixed bulk mesh, growth off, same interior labels; positive prescribed spatial eta permitted through converged projected-reaction correction. All maps/state commit together; schema7 rejects law/material changes. | Experimental `Jaumann DG material Lie`; matched and heterogeneous bounded translation/rejection/restart evidence. No cohesive energy, moving AMR, boundary completion or production claim. |
| Mature particle control | Each particle owns physical Maxwell **branch** tensor; native FE mapped stress supplies mechanics. No independent auxiliary tensor or slip memory. Same mature RSF/normal/loading conventions as its explicitly chosen fixture. | Frozen phase/fault, existing CPDI/particle dependencies retained. | Default legacy; `Jaumann lagged Lie` is opt-in, fixed mesh/stationary particles/full3D deviator only. |
| New cohesive-interface/native RSF | `CohesiveInterface::BulkHistory` independent DG coefficients and complete native-gradient remainder; physical bulk stress adds analytic jump-lifting memory. Gauss records own cohesive jump/damage/age/reference/relaxed jump. Backward-Euler Maxwell. | Persistent formed material and replaceable zero-history completion. One fracture inventory Gc integral a ds. Mean-zero solved p′ and background p0; variational normal coupling and separate sided contact checks. | Legacy component rate default. Experimental `Jaumann cut DGQ2 lagged Lie BE` rotates complete native-plus-lifting stress in formed piece/side history. Uncut DGQ2 is rejected; material transport/deformed-geometry coupling remains unsupported. |
| Earlier P2/PF6a evolving bulk phase | Accepted native d_R or DG trial stress plus **physical** elastic-slip Z/G. Reconstruct tau=T−2chi Z S; evolving profile produces Qgeo. | Recoverable bulk+cohesive+diffuse fracture energy, physical phase obstacle. P2 CPDI and PF6a FE quadrature remain distinct. | Restricted no-spin reference; proposed corotational equations do not automatically change it. |

These routes share Maxwell tensor algebra, full3D plane strain, native FE
mechanics and accepted-state discipline. They do not share their Maxwell time
discretization, fracture inventory or surface history meaning.

## Variables and ownership

In plane strain the stored in-plane deviatoric block has zz=−xx−yy;
`|A|3²=A:A+trace2(A)²`. FE symmetric strain is deviated with trace/3, never/2.
`Lij=partial_j v_i`; a local segment-frame projection changes components, not
the spatial tensor's physical evolution.

| Variable | Meaning/units/frame/status | Update, owner, persistence |
|---|---|---|
| `DGHistory::trial_stress[c]` | T, Pa; spatial tensor coefficients in DGQ2; independent, not physical reconstructed stress. Three in-plane components × nine coefficients/cell. | `make_dg_history_trial`: T+=beta T−+2eta_ve Pi3[D3(v)]. `PhaseFieldFault` owns the accepted pointer. Mature accepted-DG checkpoint v4 stores CellId/geometry/coefficients, phase, I/Z, clocks and publication count. [DG source](../../../source/material_model/phase_field_fault/dg_history.cc), [checkpoint](../../../source/material_model/phase_field_fault/dg_accepted.cc). |
| `DGHistory::residual` and attempt anchor | Independent R, Pa, global DGQ2 deviator in the new mature control; optional owner attached to T. | `dg_objective.cc` constructs complete finite-rotation T/R anchors from accepted FE gradients; `make_dg_history_trial` applies matched reaction. Parent/step/actual-dt keys prevent retry reuse; published observation bypasses the incoming anchor. Schema6 persists T/R; derived caches rebuild after restart/transfer. |
| Mature `FEStrainState::Z`, `I` | Z=G s_mem, Pa m; I, m; Q1 surface scalars. Auxiliary memory has no separate energy/resistance. | Z+=beta Z−+eta_ve V; same immutable accepted anchor as T. tau=T−2(h/I)Z S. `retained_dg_qp_stress` and `evaluate_dg_history_stress` reconstruct the same field. Cohesive traction remains zero. |
| P2/cohesive `Z` | G s_elas=I q_coh, Pa m; physical recoverable elastic slip, not mature s_mem. | Same matched-parameter scalar recurrence, different constitutive interpretation/energy. `prepare_fe_strain_commit`; checkpoint restrictions depend on selected route. [FE history](../../../source/material_model/phase_field_fault/fe_strain.cc). |
| `displacement_history` | Accepted d_R, m, for FE history; gradient reconstructs a compatible tensor. In mature DG it stores only the fixed phase/geometry anchor; its velocity block is zero. | Recurrence `update_fe_strain_trial`; not a general independently rotating/advected tensor. Shared vector representation does not prove kinematic compatibility under spin. |
| Particle `maxwell stress` | Physical branch stress, Pa, spatial symmetric tensor. In BP3 this is a perturbation about background tractions. Independent on particle route; derived diagnostic samples on DG route. | `build_history_candidates` → collective validation → `publish_history_candidates`. Native particle backup/migration/checkpoint. New option rotates incoming physical tensor once before exponential Maxwell/source. [history](../../../source/material_model/phase_field_fault/history.cc), [property](../../../source/particle/property/maxwell_stress.cc). |
| `BulkHistory::tensor`, `native`, `projection` | New cohesive T=T_DG+2G D3(d_R)−Pi3[2G D3(d_R)], Pa. Native remainder is physical independent history information, not a disposable interpolation error. | `BulkHistory::advance`: beta_BE times old plus native displacement source; complete native vector retained. `pack/restore` persist CellId vertices, 27 coefficients and native velocity-component coefficients. [owner](../../../source/reconstructed_fault/cohesive_bulk_history.cc). |
| `BulkHistory::cut_tensor`, `cut_geometry` | New cohesive independent tensor polynomials on formed lifting pieces/sides; native remainder stays exact. The ordinary DG `tensor` becomes a derived projection cache in this mode. | `cohesive_cut_history.cc`: finite complete physical rotation; BE source is restricted into each local polynomial. Cut geometry/basis/coefficient records extend physical CellId checkpoint data, with a distinct law signature. Temporary candidate tails are not partitions. Side-aware tensor and lifting observations match. |
| Cohesive `State::memory` | Q1 relaxed jump/lifting memory, m. Physical tensor contribution is `+2G sum memory_j dev3(gradient lifting_j)` with the source's sign convention. | memory+=beta_BE(memory−+increment); physical stress in `System::Implementation::physical_stress`. The new cut reaction anchor rotates the complete A+H tensor; b itself has no scalar commutator. [system](../../../source/reconstructed_fault/cohesive_system.cc), [advance](../../../source/reconstructed_fault/cohesive_advance.cc). |
| Gauss `PointState` | Signed `jump`, `reference`, `relaxed_jump`, `absolute_slip`, m; damage a dimensionless; theta normalized age V0 Theta/Dc. Physical histories, attached to material identities. | `CohesiveEvolution::solve`, surface return and aging; accepted owner/manager persist every point and identity. Q1 jump plus Gauss offset preserves nonuniform history. [point law](../../../include/aspect/reconstructed_fault/cohesive_evolution.h). |
| Mature `V`, `Theta`, accumulated slip | V≥Vmin, m/s, oriented scalar; age seconds; slip m. Scalar surface histories, no tensor rotation. | Manager separates committed/current/trial V. `FaultFriction::update_state` uses exact frozen-V aging. BP3 output owns accumulated slip/monitor clocks in its checkpoint status. [manager](../../../source/reconstructed_fault/manager.cc), [friction](../../../source/material_model/rheology/fault_friction.cc). |
| Bulk phi versus surface a | phi dimensionless diffuse geometry/damage on older routes; new geometric phi is a visualization/localization field, distinct from physical Gauss damage a. | Physical phase irreversibility is an accepted lower bound; numerical remapping is a separate operation. New cohesive route charges Gc integral a ds once and does not add a second phi fracture charge. |
| p′, p0 and normal traction | Pa; p′ solved mixed multiplier, p0/background prescribed input. sigma_n=p_background+p′−n.tau.n, compression positive, where that route selects actual-pressure friction. | Pressure receives no commutator. New cohesive p0 is an isotropic bulk confinement; BP3 background normal/shear traction scalars are prescribed surface loads, not a known full prestress tensor. Variational and sided tractions are separate diagnostics. |

Particle removal, ordinary FE fracture quadrature and removal of particle-derived
geometry are separate changes. The new particle spin option removes none of
these dependencies. The production DG route still retains stationary diagnostic
particles and material/geometry infrastructure as documented in its reports.

## Timestep map and one actual step

For production BP3 step816, the accepted incoming state is815 at
5172072493.8982706 s, dt=.0074540672088498705 s. The resulting clock is
5172072493.9057245 s. The new supplied resumed log shows this step accepted
after the observer repair.

1. Native timestep entry restores accepted solution/history and prepares
   reconstructed geometry/normalization. `prepare_reconstructed_fault_mechanical_solve`
   checks frozen profile/geometry/material, and begins from committed surface
   state. Stationary particles do not advect; no transport, birth, phase damage
   or fault admission occurs on this production route.
2. `solve_reconstructed_fault_stokes` opens manager current/trial slip fields.
   Native FE velocity gradients provide D3; `compute_maxwell_coefficients`
   supplies beta=exp(−dt G/eta), eta_ve=−eta expm1(−dt G/eta). Retained history
   reconstructs tau− from the immutable T−/Z−/I. Production has no rotation.
3. The [Stokes assembler](../../../source/simulator/assemblers/reconstructed_fault_stokes.cc)
   adds2eta_ve D3 and frozen beta tau−, subtracting2eta_ve chi V S.
   The [surface bulk-work assembly](../../../source/reconstructed_fault/surface_system_bulk_work.cc)
   uses the matching work adjoint, actual pressure and current trial rates.
   Normal filtering, friction derivatives and active sets enter the reduced
   nonlinear solve. Newton/line-search trials do not change accepted histories.
4. After convergence, T+/Z+ candidates and particle diagnostic samples are
   prepared and independently checked. The outer timestep manager can still
   reject. The existing deferred-publication callback now also covers the new
   particle-spin option; reject restores entry vectors/bookkeeping without
   rotating or committing accepted history.
5. Outer acceptance publishes once. `prepare_fe_strain_commit` and
   `publish_fe_strain_commit` exchange DG/state pointers only after validation;
   the manager commits V and age. BP3's matching-frame observer checks mechanical
   loads against an independent evaluation. It does not advance stress again.
6. Output and checkpoint occur at accepted endpoints. `serialize_dg_accepted_history`
   records the physical endpoint and native clock relation explicitly. Particle
   spin checkpoints add a versioned registry marker; absence means legacy.

For the **new particle spin** route, step2 additionally samples the accepted
`get_old_solution()` velocity gradient and freezes it throughout the attempt.
Apply R=exp(dt skew L−) to each incoming physical tensor, then use the same
Maxwell/source formula. The frozen load, surface residual, work observer and
terminal particle update use the same convention. No current-velocity spin
Jacobian is claimed: W− is an explicit time lag, and the existing tangent is
the derivative of that declared discrete residual.

For the **new cohesive-interface** route, `System::Implementation::prepare`
builds geometry/lifting/quadrature, incoming Gauss history and native mechanics
responses from one accepted `State`. `trial` solves the pressure/contact/surface
return, forms smooth native du and Q1 jump increment, then
`BulkHistory::advance/check_advance` retains full native history. All physical
storage/work/source/contact/tangent/geometry tests precede publication. Damage,
formed identities and temporary completion belong to the candidate; rejection
discards them. Restart restores CellId bulk records and all material Gauss
histories. That state machine is not validated by BP3's successful observer.

## Energy map

Positive dissipation and positive input work use the source's sign. All bulk
tensor contractions retain zz. Surface weights are physical arclength weights;
new cohesive bulk quadrature is geometry-aware and shared by storage, work and
residual. A numerical history transfer is not a new physical work term.

| Inventory/term | Definition and time/measure convention | Source and claim |
|---|---|---|
| Mature branch storage | integral_B |tau|3²/(4G) dA, with tau the **reconstructed physical** tensor, not T_DG alone. No G s_mem²/(2I) term. | Mathematical Maxwell branch storage. BP3 supplied profiles/observer do not constitute a complete measured trajectory energy ledger. |
| Earlier cohesive recoverable energy | integral_B |T−2chi Z S|3²/(4G) dA + integral_Gamma Z²/(2G I) ds. | `evaluate_fe_phase_energy_at`, normalization derivatives and physical obstacle. P2 CPDI versus PF6a ordinary FE quadrature are documented discrete replacements. |
| Fracture | Older bulk phase Gc times its documented diffuse functional; new interface Gc integral_Gamma a ds. | `fe_phase_energy.cc` / `cohesive_advance.cc`. Inventories are alternatives, not additive charges for the same fracture. |
| New cohesive bulk storage | integral_B |physical_stress(State)|3²/(4G) dA. | `cohesive_advance.cc`: before/after stress from DG+native remainder+lifting memory. Increment uses (tau+ +tau−):(tau+−tau−)/(4G), avoiding subtraction of large storages. |
| New cohesive bond storage | (1−a)(j−j_ref)²/(2 c a), c=2Gc/peak², on0<a<1; documented intact/decohered limits. | `bond` and Gauss point law; actual jump/reference, not mature slip memory. |
| Friction and radiation | Positive point-return friction work and eta_rad V² dt, counted once, integrated with surface weights. | `r.point[q].friction/radiation`; mature source separates friction from radiation resistance as well. Contact normal work follows each route's no-opening/variational constraint. |
| Maxwell loss | Continuous branch integral dt integral_B |tau|3²/(2eta) dA. New cohesive BE ledger uses2G dt Ebulk+/eta. | The exponential BP3 step and backward-Euler cohesive step have different exact discrete balances; do not substitute coefficients between them. |
| Endpoint loading work | New cohesive integral_B [tau+:D3(particular du)−(p0+p′+) div(particular du)] dA. Trapezoid counterpart recorded separately. | `cohesive_advance.cc` `work`/`trap`; mixed pressure and lifting determine the actual adjoint. BP3's weak traction check is a different observable. |
| BE numerical terms | bulk norm increment²/(4G), plus Gauss `surface_rem`=q dj−Delta bond−Gc Delta a. | New cohesive ledger: work−(Delta bulk+Delta bond+fracture+friction+radiation+Maxwell+BE+surface_rem). Physical remainder removes numerical terms separately. Unchanged acceptance bound1e−10 max(1,|work|). |
| Rotation | tau:[W,tau]=0; finite R congruence preserves quadratic storage in the rotation substep. Forcing/source splitting error remains. | New opt-in kernel and independent local ledger. No artificial spin loss or change to old energy gates. |
| Boundary/history flux | Required for advected/material histories and changing physical surface measure. | The restricted mature material selector transports complete history with explicit zero or accepted-trace inflow. The bounded-profile comparison measures coupled material-energy flux and weak mechanical/interface work. Stationary BP3 and the earlier local kernel do not qualify this extension; general moving AMR/cohesive shape work remain open. |

## Unproved reductions and model choices

At constant matched parameters and fixed localization without transport/spin,
the mature separated representation reproduces the physical Maxwell recurrence.
With spin, `C[s_mem]=−chi s_mem [W,S]` even for fixed geometry. Therefore a
physical mature law with Qgeo=0 requires the additional residual equation in
`particle_free_history.tex` eq.(residual). Setting it to zero would instead
select the cohesive-like geometric forcing Qgeo=[W,2chi Z S]. This is not an
integration approximation and is explicitly rejected by the new selector.

The newer interface uses smooth native strain plus displacement-jump lifting;
its bulk forcing is not automatically D−chi V S. A derivation must decide how
the physical/lifting structural tensors transform, preserve the native remainder
under spatially varying rotation, and rederive its BE energy/pressure adjoint.
No equivalence to production mature BP3, nor objective finite-strain
hyperelasticity, is claimed.

## Transported extension state contract (Stage A)

2 November 2026. [TRANSPORTED_OBJECTIVE_SPEC.md](TRANSPORTED_OBJECTIVE_SPEC.md)
closes the extension's equations and kinematics; its [report](TRANSPORTED_OBJECTIVE_REPORT.md)
records staged checks. The formulation table above describes **current source**.
This section keeps the full material-transport contract separate from the
broader Stage C contract. The restricted mature material selector implements
the same-label growth-disabled subset; general ALE/birth/cohesive motion remains
proposed.

The researcher-facing distinction is: bulk stress follows bulk material; age,
slip and damage follow specified interface labels; numerical nodes may move
along that interface independently. A step moves those fields first, changes
physical damage only in its own solve, and publishes the entire result once.
Moving a node or recomputing a profile must not create fresh age, erase slip,
rotate pressure, leave a damage trail, or invent elastic prestress.

All fields below have accepted A and private candidate C. Rejection destroys C;
observations read A/C without mutation. Checkpoint means accepted state only.
Legacy schemas remain unchanged. Stage C reaction controls add mature schema6
and cohesive cut partition records. Mature material schema7 now persists the
complete restricted moving state and material/law identity.
Names prefixed “new” below remain proposed unless explicitly marked implemented.

| State, meaning and frame | Storage / current source owner | Transport and reaction | Geometry, transfer and new-schema persistence |
|---|---|---|---|
| Physical tau, Pa, global full3D deviator | Mature reconstruction in `dg_history.cc`; particle physical tensor is a control. New direct tensor is a Stage B candidate. | Bulk v; J_v tau+r_b tau=2G(D−Lcal[V]) for mature. | Project/migrate all xx,yy,xy with implied zz, validate stress/storage; persist CellId/basis and law/clock metadata. Not a transported displacement gradient. |
| T, R, Pa, global independent deviators | Current `DGHistory::trial_stress` T and Stage C `DGHistory::residual` R for the selected mature reaction control. | Both follow bulk v; SPEC(10) supplies complete R source. | T−2G Lcal[s]−R is physical. Same transfer and reconstruction quadrature; both persist, even zero incoming R. Stationary schema6 persists R; material schema7 adds same-parent DG transport and complete geometry/profile/surface transaction. |
| s_mem=Z/G, m, auxiliary mature scalar | `FEStrainState::Z` Q1 in `fe_strain.cc`, accepted pointer owned by material. | Interface u_Gamma; D_Gamma s+r_Gamma s=V. | Material selector preserves same ordered labels following accepted bulk velocity, rebuilds extension/normalization, and commits reaction with tensors. Schema7 persists state/identity; resampling/ALE redistribution remains unsupported. No separate energy/resistance. |
| Physical elastic e=Z/G, m, older P2 route | Same legacy Q1 slot, different constitutive meaning. | Interface labels; source V−r_Gamma e; physical Qgeo from its energy. | Transfer obstacle and physical e together; preserve E_coh and actual I derivatives. Do not reinterpret an old mature checkpoint as elastic slip. |
| Native cohesive A, Pa, global tensor | `BulkHistory::{tensor,native,projection}` in `cohesive_bulk_history.cc`. | Complete physical/native remainder obeys bulk transport/reaction; scalar d_R transport alone is inadequate under general W. | Stage C cut T_ind plus exact native remainder is qualified for fixed-geometry reaction. Cut partition/bounds/coefficient/native records persist; bulk material motion is still pending. |
| Cohesive b, m, relaxed lifting scalar | `State::memory` Q1; `PointState::relaxed_jump` its Gauss trace. `cohesive_advance.cc`, `cohesive_state.cc`. | D_Gamma b=V−r_Gamma b, single-state interface labels. | Stage C reaction compensates complete lifting spin in cut T_ind, retaining b and trace equality. General K transport/rate/shape terms and numerical remap remain required. Persist b, full bulk history and physical trace IDs. |
| Jump j and reference j_ref, m; signed and absolute slip, m | `State::jump/offset/rate`, `PointState::{jump,reference,absolute_slip}`; manager's left-vertex Gauss3 registry. | D_Gamma j=V, D_Gamma j_ref=0, absolute source abs(V); instantaneous V is solved, not advected as an independent physical law. | Transfer Q1 plus nonuniform Gauss offsets, preserve Gauss consistency/material IDs and signed orientation. No nearest-node reset. Persist all accepted traces and offsets. |
| RSF Theta (s) / normalized theta | Mature manager vertex properties; cohesive Gauss PointState; `fault_friction.cc` versus `cohesive_interface.cc`. | Interface labels; respective exact frozen-V mature or implicit cohesive aging; SPEC §2 fixes units. | Positive intensive state via ALE transfer, not density conservation or tensor rotation. Persist accepted state and normalization, virgin clock/birth inputs. |
| Physical a (dimensionless) and formed identities | Cohesive PointState damage, `State::{formed,ids,identity_clock,metadata}`. | Interface labels; existing point-return irreversible source; physical admission separate from motion. | Transport old material lower bound; bound-preserving identity transfer. Persist damage, formed prefix/IDs and admission metadata. Temporary completion has no physical history or fracture inventory. |
| Physical bulk phi and obstacle; old diagnostic H | Material/FE phase field and accepted obstacle, `fe_phase_energy.cc`, `fe_strain.cc`. H retains its documented diagnostic/initialization role. | D_v phi=R_phi; R_phi=0 if no growth. Material lower bound via pullback. | Bulk scalar material transfer and gradient reconstruction. Do not Eulerian-max old/new fields. Persist accepted phi and obstacle/initialization history where it is independent. H is not new fracture storage. |
| New geometric phi on native cohesive path | Derived from curve/support for visualization/localization. | Rebuild from physical curve/profile under declared map. | No independent phase obstacle or second fracture charge. Rebuild/cache derived fields; persist physical profile inputs instead. |
| Curve X, orientation, material parameter/reference metric, width/profile | Mature manager/Fault; cohesive `Geometry::{Piece,nodes,support}`. Current graph requires increasing x. | Physical normal speed common v±·n; labels symmetric mean; nodes w=u_Gamma+alpha t. Reactions only physical birth/growth. | New carried material coordinates/profile representation, reference/current measures, IDs and admissible tube bounds. Rebuild t/n from geometry, not stress spin. Existing graph cannot represent overturned/vertical folds; retain diagnostic rather than silently change topology. |
| chi, I_h, S, pi and lifting K | `normalization.cc`, `fe_normalization.cc`, `dg_geometry.cc`, cohesive `Geometry::lifting`. Derived maps and projected scalar coefficients. | Differentiate with SPEC(4–9), K product/transport defect. | Rebuild from accepted/candidate geometry/profile/mesh; transfer physical inputs, not stale caches. Persist projection/geometry definitions and generations, recompute derived matrices on restore. |
| Bulk eta/material fractions; surface mixtures and r_Gamma | `constitutive.cc` property evaluation; surface property projections. | Advected composition follows material v if selected; imposed depth law samples spatial depth. Rates from actual transported point. | New coefficient-provider identity/configuration and field state in checkpoint. Constant G; no hidden switch between fixed spatial layer and advected material layer. |
| Native v,p', accepted-gradient anchor | Simulator FE solution/old solution and pressure scaling; reconstructed-fault solver/assembler. | v solved quasi-statically, p' multiplier; accepted v−/W− frozen per attempt. | Standard FE transfer plus explicit accepted-anchor semantics; preserve pressure gauge/confinement. Persist accepted solution and law/lag, no extra rotation at restart. |
| Background q0,pn0 / isotropic p0, Pa | Benchmark loading plugin / cohesive Parameters, separate from branch history. | Declared follower/material or spatial input, no inferred bulk Maxwell tensor. | Persist exact loading/configuration/inflow identity; work is external and counted once. Optional full initial tensor must be explicitly supplied. |
| Fracture/energy/work observers, publication counters | Current `cohesive_advance.cc`, `cohesive_state.cc`, BP3 plugin and simulator accept transaction. | Derived balances from accepted endpoints; counters once per accepted physical step. | Reconstruct with current weights, flux/shape work and actual stress. Persist physical accumulated inventories, schedules and counters; derived caches are disposable. No constitutive state advance in output. |

Proposed material-point bulk transport owners should reuse `dg_transport.cc`/
`dg_transfer.cc` where their restrictions permit, not move scientific ownership
into a benchmark plugin. `dg_accepted.cc` and `cohesive_state.cc` are the schema
owners to extend after representation qualification. Simulator/core and native
solver own the outer acceptance boundary; manager owns surface registry/state;
geometry owners own topology/maps. No second stress-selector or checkpoint
converter is introduced in this design.

### One proposed accepted transported timestep

For a fresh bounded fixture, let accepted time be t_n and attempt dt. From that
single parent: transport complete T/R (or qualified direct tau) with bulk v−,
move the material curve using its common normal speed and selected labels,
ALE-transfer surface histories, and pull back the old damage obstacle. Rebuild
chi/I/S/lifting and their adjoint with current physical weights. Apply accepted-
spin finite congruence and declared endpoint-rate relaxation, solve current
v,p',V and RSF, then any selected physical phase/admission variation with its
actual energy derivative. Check complete reconstructed stress, tangent,
physical work/flux, bounds and timestep error. On acceptance atomically publish
all fields, geometry, counters and anchors; checkpoint that endpoint. Rejection
returns to the same parent and recomputes with the retry dt. The mature
growth-disabled same-label subset now implements this lifecycle and schema7
restart; physical phase/admission variation and general ALE/shape work remain
design requirements beyond that selected subset.

### Stage B first-part implementation and evidence

The [fixed-profile comparison](TRANSPORTED_OBJECTIVE_STAGE_B_FIXED_REPORT.md)
implements the proposed independent T/R and direct-tensor owners in a bounded
Python Cartesian DGQ2 fixture. These are coefficient arrays in the experiment,
not new `PhaseFieldFault` state or a checkpoint schema. `stage_b_fixed.py` maps
nine Q2 modes per scalar/cell to full3D plane-strain tensors, evolves R from
its commutator source and reconstructs T−B−R with analytic B. Its component
reference is independent of those owners; strain/source projection and residual
projection are separately measured. Four mesh levels establish spatial capacity
only. The time study demonstrates finite-step Lie closure error rather than
hiding it with an assignment from direct stress. Source/geometry/MPI ownership
in the tables above remains unchanged. Remaining Stage B transport/coefficient
and then native lifecycle/work qualification are still required.

### Stage B completed representation decision

The [moving/layered Stage B report](TRANSPORTED_OBJECTIVE_STAGE_B_MOVING_REPORT.md)
selects a complete DGQ2 R beside existing DGQ2 T and surface s for the first
native mature integration. Both tensor owners follow bulk material v; s follows
the interface labels. R's full source includes geometry/profile/orientation,
association-gradient and local/projected-rate mismatch, not just [W,B]. Stage C
now stores R in the mature stationary owner/schema6 and moving owner/schema7.
The matched and spatial-eta material transaction is locally verified; broader
advected composition and variable-G source laws still require separate admission.
Use the existing DG space/projection/transfer infrastructure and actual surface
coefficients; checkpoint both independent tensors, scalar memory, geometry/
profile/obstacle and accepted anchors with new compatibility metadata. No old
checkpoint is reinterpreted as containing R=0. Direct physical stress remains
the independent regression and its same-mesh memory advantage is recorded.

`moving_model.py` supplies prescribed reference geometry/surface memory and
separate characteristic integrals; `stage_b_moving.py` evolves independent tau,
T and R on Eulerian Q2 using volume/upwind material transport. Exact-characteristic
Lie updates verify the selected first-order convention separately from the RK4
spatial-reference fixture. A best-projection P1 residual is screened, not evolved.
Twenty numerical and8 selection/profile checks pass. These owners are Python
coefficient arrays, not serialized native state. Native pressure/work coupling,
RSF, shape variation, publication/rejection/restart and distributed transfer
remain Stage C/D obligations beyond the reaction controls described above.
`ReconstructedFaultManager::stage_material_motion` now stages coordinates and
support widths with unchanged material label indices/properties/rates. It
invalidates spatial associations and supports archive/swap rollback. It does
not resample labels, evolve a profile, or publish a complete material step.
The newer cohesive native remainder/lifting and
its physical damage histories in the single table above remain distinct.

### Boundary/profile accuracy limits measured after the coupled experiment

[The isolation report](BOUNDARY_PROFILE_ISOLATION_REPORT.md) verifies the native
pure-transport volume/face energy identity and distinguishes prescribed incoming
history from its zero-inflow control. The isolation-only material selector used
zero inflow; that study adopted no new algorithm. The selected bounded-profile
extension below now adds an explicit incoming-trace option. Its Q1 physical
profile/obstacle pullback repeatedly interpolates. At fixed h, decreasing dt can
increase interpolation diffusion; compare joint h/dt refinement at controlled
CFL and separate representation/remapping cross terms. Independent localization
projection delta_B=P B-B contributes to reconstructed-stress error but has no
separate auxiliary energy. Pure DG energy-flux verification does not close the
full coupled work budget or the separate cohesive shape-work gate.

### Selected bounded-profile implementation

The moving material row now has `Material profile transport = Q1 remap`
(default) or `bounded MacCormack`. `dg_material_profile.cc` owns the accepted
Q1 forward/return correction and donor bounds, and returns the single physical
profile/transported obstacle to `DGMaterialTrial`. It adds no independent tensor
or surface history. `dg_material.cc` optionally freezes accepted incoming T and
R+P B, with the old zero-inflow control retained. Both selectors enter schema7
identity when nondefault. Accepted velocity, normals, reaction, work adjoints,
publication and rollback retain their existing owners and conventions.
See [specification §10.3](TRANSPORTED_OBJECTIVE_SPEC.md#103-opt-in-bounded-q1-material-profile-and-accepted-trace-tensor-inflow)
and [coupled comparison](BOUNDED_PROFILE_TRANSPORT_REPORT.md). The reference
uses identical pure-translation mechanical and incoming-history boundary data;
it is not a comparison against the older sheared/zero-inflow trajectory.


### Bounded-profile coverage condition (current implementation)

The nondefault bounded Q1 profile and accepted-trace inflow selections require
`Association support = full normal cross-section`. The transported mature
selector admits this existing policy for finite interior fault endpoints;
there is no tangential continuation or cohesive-route admission. This corrects
the support/I mismatch exposed by the retained truncated-support comparisons.
The existing inverse-transpose material support-width map is unchanged.
Schema7 material identity additionally records full-normal support, so a
checkpoint from the truncated or early unidentified-support path is rejected.
The default Q1/zero-inflow/stationary-support identity is unchanged. See
[the measured comparison](BOUNDED_PROFILE_TRANSPORT_REPORT.md) for coverage,
normalization and the remaining spatial/time errors.
