# PF0 source audit — 15 October 2026

## PF5 accepted-history/persistence addendum — 16 October 2026

The user selected a bounded fixed-mesh/fixed-geometry PF5 subset after accepting
the PF4 method recommendation. Unlimited upwind DGQ2 is retained; no limiter
setting or production default changes. The original audit and prior evidence
below remain historical records. See [PF5_REPORT.md](PF5_REPORT.md) for measurements.

| Operation | Actual owner and implementation | Restricted contract / remaining dependency |
|---|---|---|
| Composed trial | `phase_field_fault/dg_accepted.cc::prepare_dg_accepted_step`; `dg_history.cc::make_dg_history_trial` overload | Native exact Maxwell/source half steps around existing pure SSPRK3 transport, from one immutable accepted tensor. No reaction or source in the transport substep; no limiter. Uniform prescribed surface V/Z only |
| Surface update | Existing `constitutive.cc::compute_surface_cohesive_candidates`, `fe_strain.cc::prepare_fe_strain_commit` | Existing full-step Z/q law once. Fixed I/phase, physical slip Z/G. Supplying a transported tensor bypasses a second native bulk update |
| Terminal publication | `history.cc::commit_dg_accepted_step`; existing `HistoryCandidates` validator/publisher and `publish_fe_strain_commit` | All validation/sampling/allocation before writes. Parent, step, clock and rates reject stale/duplicate/changed trials. Derived particle stress and frozen H still published; particles never supply incoming T or mechanical/weak loads |
| Persistent tensor | Existing material-owned `FEStrainState::dg_history` | Three authoritative DGQ2 fields; shared velocity-history block remains zero and fixed phase anchor remains native FE data. Publication counter and committed step identify accepted endpoints |
| Native archive | `simulator/checkpoint_restart.cc` delegates after manager serialization to material `serialize_dg_accepted_history` | Selected registry marker gates version-1 CellId/local-Q2 payload plus Z/I, phase anchor, material and geometry checks. Unselected path writes no new bytes. Same-mode fixed serial restart only; no checkpoint conversion |
| Restore | Existing triangulation/solution, postprocessor, fault-manager and particle/RNG loading; material `restore_dg_accepted_history` after all userdata | Rebuild native spaces, fixed phase/normalization/associations, then restore T/Z/I. Surface-temperature cache must also be rebuilt before native constitutive trials. A material callback alone runs before particle userdata is ready |
| Frozen profile | Existing FE phase and consistent I/chi evaluation | Normal bulk motion at nonzero Z retains `Q_geo=-2 Z (v.grad chi) S` through physical reconstruction. No added spin or mature Qgeo=0 substitution |

The manager's native load makes an omitted default shear sense explicitly +1.
Raw manager archive bytes therefore differ by this representation change; all
history values and complete manager archives after the same native load are
compared separately. Registry-only unused values contain intentional NaNs;
bitwise comparison preserves them rather than treating `NaN != NaN` as mutation.

Guards retain serial/fixed grid, unchanged geometry/phase/I/particles, no initial
or later AMR, no periodic/hanging/open-boundary splitting, no mesh motion, no
surface transport, no state-dependent friction and no coupled outer update.
Remaining particle dependencies include CPDI/domain/H startup, surface mixture
projection, support/association and fixed-particle validity checks, native
checkpoint/RNG services, and derived samples. This is independent bulk-history
advancement, not an end-to-end particle-free model.

---

The PF0 audit below is preserved at its stated baseline. The implemented PF2
addendum at the end and [PF2_REPORT.md](PF2_REPORT.md) describe subsequent
authorized source changes and fresh-source verification.

PF0/PF1 only. No production constitutive, discretization, parameter, filter,
checkpoint or solver change is selected or made. This audit resolves existing
APIs from source and established laws from `current_design.md` and
`specification.tex`; the new derivation is a proposal, not a replacement spec.

## Repository and instruction identity

| Item | Observed identity |
|---|---|
| Worktree | `/home/ein/repository/aspect-pf-rsf-particle-free` |
| Branch / HEAD | `pf-rsf-particle-free` / `3232056cde12dbae17ef4265ecb804f9098b88f1` |
| Actual propagation base | Local `fault-propagation` worktree/branch has that same HEAD and a clean status at audit time; no remote-tip assumption |
| Incoming edits | Unstaged `AGENTS.md`; untracked `particle_free/`, `particle_free_history.tex`, `particle_free_plan.md` under `doc/reconstructed_fault/`; nothing staged |
| Effective instruction file | Repository-root `AGENTS.md`; no ancestor, nested or override instruction file found in the work paths. The instructions supplied in chat also apply; the latest explicit PF0/PF1 request controls scope |
| Root guidance | Already contains authority, particle-free science and handoff requirements; left byte-identical. `ROOT_AGENTS_APPEND.md` is absent, so no missing fragment is invented or merged |
| Requested plan path | `doc/reconstructed/particle_free_plan.md` is absent. Installed `doc/reconstructed_fault/particle_free_plan.md` and `particle_free/PLAN.md` are byte-identical: SHA256 `87c81192383bfae38a8e0a97940cd982a5de32285d1722e9bd99333968336f5e` |
| Actual derivation | `../particle_free_history.tex` (not a file inside this directory), SHA256 `bb05702d5c712bc51a87f0b2a75d881d56601d86106bd600d305815b00927539` |

Read: the package README/PLAN/PROMPTS/VERIFICATION/STATUS, actual derivation,
`CURRENT_STATUS.md`, propagation STATUS/PLAN/closeout/cleanup, refactoring
guidelines and relevant established design/specification sections (common
cohesion, stationary mature FE, total energy, bounds, Stage B and lifecycle).
Historical stage selections are context. No other worktree/build/job is changed.

## State ownership and dependency inventory

Locations below are repository-relative and refer to this HEAD. No historical
PhaseFieldRSF or core-phase-field implementation supplies the design.

| State/operator | Owner / source location | Initialization, accepted update and use | Particle dependency |
|---|---|---|---|
| Legacy bulk stress | `source/particle/property/maxwell_stress.cc:25`; `source/material_model/phase_field_fault/history.cc:588`, `:892`, `:997` | Explicit compositional mapping initializes every symmetric component. The property itself has **no automatic update or rotation** (`maxwell_stress.cc:117`). Material computes candidates after mechanical convergence, then publishes. Bulk/fault loads read the mapped FE representation, history commit reads parent stress | Authoritative particle state; advection, sampling, mapping and population interpolation matter |
| Accepted FE bulk history | `include/aspect/material_model/phase_field_fault/fe_strain_internal.h:45`; `source/material_model/phase_field_fault/fe_strain.cc:201`, `:386`, `:455`, `:474`, `:524` | Authoritative velocity block `d_R`, stored accepted phase, surface `Z,I`. Fresh unloaded perturbation state. `d_R+=beta d_R-+(kappa/G)u+`; stress `2G D3(symgrad d_R)-2(h/I)Z S`. Trials are scratch; swap only on terminal publication | Independent of particle stress projection, **not particle-free**: preparation validates particle properties/positions/population; accepted stress is still sampled onto particles |
| Cohesive history | `history.cc:1231`; `constitutive.cc:378`, `:474`, `:506`; `history.cc:1150`, `:1185` | Manager generic nodal properties store committed `q,I-`; optional surface-elastic-slip semantics make `Z_j=I_j q_j` the Q1 variable. Interpolate nodal products before dividing by interpolated I. Fresh seeded `q_p=g sqrt(2GH_p)` is projected; unloaded initialization sets nodal q=0 directly after normalization/association validation | Surface option avoids repeated q projection. Legacy q update, nonzero initial q, mixtures and support still use generic particle projection |
| Mature slip memory | `fe_strain.cc:251`, `:402`, `:488`; `constitutive.cc:532` | Mature cohesive q remains exactly zero; FE Z advances independently as `beta Z-+kappa V+`. This Z/G is auxiliary memory, not physical cohesive elastic slip or resistance. Frozen localization makes compact reconstruction exact within its supported restrictions | FE memory itself is surface/FE-owned; geometry, material and validation services remain dependent |
| RSF Theta and V | `history.cc:388`, `:830`, `:1065`; `source/material_model/rheology/fault_friction.cc:33`; `source/reconstructed_fault/manager_slip_rate.cc:109` | Initialize positive Theta from exactly one mapped particle field by generic projection. Mechanics uses committed Q1 Theta. Exact frozen-V aging candidate is committed once. Manager distinguishes timestep-committed, Newton and line-search V; cumulative slip is manager-owned | Initial Theta/material composition project from particles; no general surface ALE/material-transport equation is implemented here |
| Phase / physical lower bound | `source/simulator/phase_field.cc:864`, `:1536`; `source/simulator/solver/reconstructed_fault_endpoint_phase.cc:32` | Q1 phase in simulator vectors; initial seed/H relaxation (or explicit shared seed), then immutable accepted physical lower bound. FE endpoint anchors its accepted phase, d_R and Z across all trials. H is not the total-energy phase driver | Fracture energy/residual/tangent still assemble over CPDI particle domains, even with FE bulk history |
| H | `source/reconstructed_fault/manager.cc:300`; `history.cc:963`; `constitutive.cc:325`, `:363` | Native prescribed profile initializes particle H, leaving noncontributing baseline H alone. Optional recoverable update `max(H-,q^2/(2Gg^2))`; default incremental-work candidate differs. FE initialized phase energy sets particle H contribution to zero; retained H is diagnostic thereafter, not regenerated from accepted phi | H remains a particle property; initial relaxation still needs it. Default legacy phase solve consumes it |
| Geometry, labels and extension | `source/reconstructed_fault/manager.cc:437`, `:554`, `:962`; `source/reconstructed_fault/boundary_contact_manager.cc`; `source/material_model/phase_field_fault/boundary_completion.cc` | Lightweight replicated fault geometry/properties; prescribed resampling or FE ridge fit, fixed support metadata, stored shear sense and closest-segment xi. Fitting uses FE phase quadrature, not a cloud of stress particles. Paired completion supplies computational boundary profiles/loads | Geometry coordinates can be prescribed independently. Startup H, material projection, associations and domain coverage still require particles. No selected birth/propagating-tip/overlap history law |
| I_h and localization | `source/material_model/phase_field_fault/normalization.cc:259`, `:1467`; `fe_normalization.cc:205`, `:228`, `:327` | `h=1/g-1`; normal integration and consistent Q1 projection `I=M_F^-1 b`; current I transient, previous I committed. P2 uses exact rational integrals on linear normal rays and their pullbacks. Mature dipping FE retains general oblique cell intervals and paired exterior completion | Numerical normal integration reads Q1 phase directly. Surface material mixtures are still particle-projected; FE P2 initialization/services also require particle infrastructure |
| CPDI / shared services | `source/simulator/phase_field.cc:697`, `:913`; `source/particle/particle_domain.cc`; `source/reconstructed_fault/manager_particle_projection.cc:140`, `:326` | Phase handler requires a particle manager, H, generated domains and CPDI data. Domain-averaged weights/gradients and volumes define the phase functional. Associations/domain quadrature are cached independently from lazy projection factors | Structural dependency remains even when all visible stress reads are bypassed. Generic property projection, domain partition, chemical data and support validation are live capabilities |

### Actual constitutive laws and conventions

Use `kappa=eta_ve` (Pa s), `beta=exp(-dt G/eta)` and `lambda=eta/G`.
`constitutive.cc:75` uses `-eta*expm1(-dt G/eta)` stably. At a bulk point:

```
q+ = (kappa_s V+ + beta_s I- q-) / I+
v_frac = chi+ V+ + beta_s q-/kappa_s * (h+ I-/I+ - h-)
tau+ = beta_b tau- + 2 kappa_b (E+ - v_frac S)
```

At matched constant coefficients, fixed coordinates/association, and compatible
incoming data `Z-=I- q-`, this becomes exactly

```
Z+ = beta Z- + kappa V+
tau+ = beta tau- + 2 kappa (E+ - chi+ V+ S)
       - 2 beta Z- (chi+ - chi-) S.
tau_tr+ = beta tau_tr- + 2 kappa E+
tau+ = tau_tr+ - 2 chi+ Z+ S.
```

Thus **the existing cohesive law already contains redistribution**. The last
term is not an additional PF2 constitutive proposal. Exact nodal-Q1 Z semantics
belong to the optional surface history path; default repeated particle q
projection is an additional discrete operation and cannot be called exact
compact-history evolution.

Mature mode instead forces q-=q+=0, unchanged h and I and zero history correction
(`constitutive.cc:537`). Its point recurrence is unmodified Maxwell slip
subtraction. Fixed-coordinate FE Z is auxiliary; moving particle stress carries
the old-profile history directly. Rejecting particle motion in the FE option
is essential, not an incidental implementation restriction.

There is **no Jaumann rotation** in these fault recurrences. Particles transport
stored components along numerical trajectories; the stationary FE option has
neither advection nor spin. `E` is the full symmetric gradient in the legacy
default. Accepted FE history and the explicit mature particle-deviator option
use `D3(symgrad_2 u)`: embed raw plane strain in 3D, subtract trace/3 once,
and reconstruct `tau_zz=-(tau_xx+tau_yy)`. Do not deviate an old deviator again.
Stored tensor norms include `zz^2` and two off-diagonal terms. The trace
distinction is explicitly recorded in the established spec, not a conflict
to silently fix in PF0/PF1.

Bulk G/eta use local bulk composition and temperature; surface G/eta use Q1
projected surface composition and surface temperature. They can differ in
legacy paths. `phase_field_fault.cc:660` rejects heterogeneous/mismatched,
temperature-dependent FE coefficients and viscosity clipping. Surface history
has its own homogeneous restrictions. The compact constant-coefficient algebra
does not authorize extending these options to heterogeneous material.

For completeness, the non-FE particle phase history is a **maximum**, not an
accumulated work integral. With q+ the accepted local surface-history traction
or legacy projected traction, `constitutive.cc:325` implements

```
C_inc = dt/(2 kappa_s) [(q+/g+)^2 - (beta_s h- q-/(1-g+))^2], 0<g+<1
C_inc = dt (q+)^2/(2 kappa_s), g+=1 and h-=0
Y_rec = (q+/g+)^2/(2 G_s)
H+ = max(H-, C_inc)              [default incremental-work selection]
H+ = max(H-, Y_rec)              [optional recoverable-energy selection].
```

Only associated particles in the evolving-history update enter these laws;
unassociated particles and frozen-history paths retain H. Initialized FE
total-energy phase evolution does not use H as its driver. No positive-increment
accumulation or rate conversion is inferred from the units. For RSF the exact
frozen-positive-V aging update is
`Theta+=Theta- exp(-V dt/Dc)+(Dc/V)[1-exp(-V dt/Dc)]`, with stable `expm1`.
Mechanics freezes incoming Theta, and terminal publication evaluates aging once
at the accepted V (`history.cc:830`, `fault_friction.cc:33`). This is a temporal
surface update at the selected labels; it does not establish a general
surface material/ALE transport discretization.

### Forces, pressure, phase functional and measures

`constitutive.cc:204` treats background shear/normal tractions as surface data,
not extra bulk prestress. With dynamic pressure, compression-positive normal
input is `sigma_bg + p - tau:(n tensor n)`. With the selected adiabatic-pressure
option it is `sigma_bg + p_adiabatic` **without** subtracting deviatoric normal
stress. Preserve that selected pressure convention. Friction is mu(V,Theta)
times this input; radiation damping is added once. Filtered mature BP3 normal
input is a separate numerical option; it remains unchanged. Native FE P2
requires fixed homogeneous Coulomb friction, no RSF, prescribed adiabatic
pressure and zero radiation damping for its endpoint potential.

The optional cohesive bulk-conjugate assembly is in
`source/reconstructed_fault/surface_system_bulk_work.cc:36`: shear exchange
uses `JxW*chi*(tau:S)` on the shared bulk rule, resistance uses the surface
measure. The generic particle-volume surface assembly is different. The bulk
source and B action remain in
`source/simulator/assemblers/reconstructed_fault_stokes.cc:59`; old history
is obtained through `retained_history_stress` (`:540`). Legacy FE stress mapping
is not an interchangeable exact reference for native FE strain history.

For initialized FE P2, `fe_phase_energy.cc:94` evaluates

```
E_bulk = sum_bulk w |2G D3(symgrad d_R) - 2 Z h(phi)/I(phi) S|_3^2/(4G)
E_coh  = sum_surface w Z^2/(2G I(phi))
E_frac = sum_particles Vp Gc/(c0 ell) [alpha(phi_p)+ell^2 |grad(phi)_p|^2].
```

`phi_p=sum_i w_ip phi_i` and `grad(phi)_p=sum_i grad(w_ip) phi_i` are CPDI
averages, not ordinary FE Gauss point values. In the non-FE path add
`sum_p Vp H_p g(phi_p)`. In the initialized total-energy path that H term is
zero (`phase_field.cc:957`); FE bulk/cohesive derivatives are added at `:1068`.
`fe_normalization.cc:327` pulls I derivatives back using the projection
transpose; second variations include both I Hessians and squared first
variations. Replacing CPDI with conventional FE fracture integration changes
the discrete functional, even if quadrature is highly accurate. That is PF3.

`evaluate_fe_phase_energy` freezes incoming d_R,Z. The consistent-endpoint
variant instead builds **trial endpoint** d_R,Z from the same immutable
incoming state (`fe_phase_energy.cc:60`) and freezes those for partial phase
derivatives, then solves the mechanically reduced problem. The derivation's
phase-first split is not automatically the existing endpoint algorithm. PF2
should freeze phase and preserve this distinction rather than adopt a new
coupling order.

## Comparison to derivation labels

| Label | Source interpretation | Classification |
|---|---|---|
| `eq:C` | No general commutator operator exists. Frozen FE/no transport gives C=0; cohesive changing phase supplies the finite profile difference; mature particle transport stores memory directly | Exact restricted identity; general extension/label/spin terms unsupported |
| `eq:masterT` | Matched-coefficient algebra above transforms the point recurrence; legacy q projection and particle-to-FE mapping are additional operations | Exact transformation under stated assumptions; discrete replacements otherwise |
| `eq:p2stress` | FE `tau=2G R-2chi Z S`, `Z=G s_elas`; legacy cohesive history correction gives the same fixed-coordinate redistribution | Existing selected cohesive physics; adding spin or general moving-profile closure would be a physical extension |
| `eq:residual` | Fixed mature FE is the C=0, initially zero residual special case. No mature residual tensor field is stored; moving particles retain physical stress instead | Exact within enforced frozen restrictions; omitting residual for general DG compact history is falsified by PF1 |
| `eq:phasevar` | FE bulk/cohesive energy and actual projected-normalization pullbacks agree structurally; CPDI fracture remains. Endpoint histories and incoming phase-bound semantics must be distinguished | Matching restricted continuum variation with explicitly different quadratures/discrete functional; PF1 does not qualify native derivatives |
| `eq:variableG` | Legacy material mixtures allow unequal rates; optional FE rejects them. Proposed extra terms include `2(D_t G)L[s]` and `2G(r_b L[s]-L[r_F s])` | Constant-coefficient restriction is substantive. A varying-spring stress law with `(D_t G/G)tau` would be a different constitutive choice |
| `eq:DG` | No reconstructed-fault tensor DG material-transport solve is implemented. Particles provide transport in the legacy path | Proposed discretization; must retain material/divergence correction and the audited no-spin convention. No limiter is ported |

No established-spec/source conflict requiring a production decision was found
in this bounded audit. The derivation's objective-rate/general moving-surface
extension and first-order split are **not established source behavior**. The
supplied historical limiter and Zhu PDF are not installed here; the unrelated
`source/material_model/rheology/elasticity.cc` contains none of the named
historical DG/WENO routines. No limiter-source claims were reverified.

## Acceptance, failure, MPI and persistence

`history.cc:588` computes complete scratch samples/candidates, validates
cohesion/Theta/stress/H and then prepares FE candidates. All ranks exchange a
local sample failure before terminal writes. Particle FE samples are checked
against the separate native Maxwell recurrence before they become derived
publication values. `publish_history_candidates` writes q/I, Theta and particle
stress/H; `publish_fe_strain_commit` swaps d_R/Z/I and clears samples. Duplicate
FE publication at one physical step is rejected. Timestep zero publishes no
physical Maxwell increment.

The endpoint driver saves solution/linearization/pressure and checks immutable
incoming FE/surface/H anchors. It refreshes trial normalization, validates the
slip commit and publishes once (`reconstructed_fault_endpoint_phase.cc:646`).
Exception paths restore solver vectors/caches and rollback V. The ordinary
mechanical driver also snapshots/restores bulk state and V
(`reconstructed_fault_stokes.cc:642`). Do not generalize these prepublication
checks into a claim of arbitrary rollback after terminal writes or of complete
physical-step/restart qualification.

Bulk cells/particles are distributed and contribute once; fault geometry and
surface vectors are replicated with collective reduction. FE d_R is distributed,
surface Z/I replicated. Manager serialization (`manager.h:397`) includes faults,
generic properties, support widths, shear senses and committed V; transient
projection caches rebuild after loading. Particle manager owns its native
serialization/AMR/backup data and population RNG. **FE history is fresh-only**:
`history.cc:247`, `phase_field_fault.cc:660`, and `fe_strain.cc:167` reject resume,
mesh/geometry changes, particle transport and birth/deletion. There is no native
FE-to-DG checkpoint conversion or defined transported phase obstacle here.
Neither PF0 nor PF1 changes these contracts.

## Bounded retained baseline and provenance

Existing native fixture: `benchmarks/reconstructed_fault/bp3_native/history-uniform-q6.prm`.
No build tree existed in the new worktree. Reused **copies** of the retained
`build-bp3-native` core, maintained BP3 plugin and native-history observer in
`build-particle-free/pf0/`; outputs stay there. The exact command is in
[PF1_REPORT.md](PF1_REPORT.md). Hard budget 180 s, serial/one thread; actual
60.274 s, exit 0. No reconfiguration or source build was performed.

The core identifies itself as `fault-propagation, c33edaaeb`, SHA256
`1ba2f7f43b37b75b1462d9cd5e48998fc8cc58b891e943829be8240f70776d12`.
Retained CMake records GCC12.4 `/opt/gcc/12.4.0/bin/g++`, Release,
deal.II9.6.2 `/opt/dealii/9.6-local`, Trilinos14.2, p4est2.8.7, Voro enabled;
binary comments also contain GCC15/16 dependency objects. This is a historical
dirty-build identity: current HEAD includes core changes beyond the embedded
label. Exact compiled-source correspondence is **not certified**. Artifact
and input hashes, original cache and complete log are retained; the run is a
fresh retained-binary baseline, not a fresh build of all audited source.
Nonfatal socket-discovery warnings appeared in the sandbox; the serial run
completed all assertions, so no networking workaround or rerun was needed.

The fixture uses an affine Cartesian Box, 64x32 Q2/Q1 cells, origin
(-1234.5,137.25)m, extents 2000x1000m, a 60-degree straight prescribed mature
fault and paired boundary completion. Shared mechanical quadrature is 6x6,
surface spacing20m, 4x4 fixed samples/cell (32768 particles), phi prescribed
and frozen, Lorentz p=1, ell20m, G=32038120320Pa, eta=1e26Pa s, q=0.
No mechanical boundary loading or coupled Stokes solve is used: observer
injects nonuniform Q2 velocities, six dt/lambda values and nonconstant V,
checks native commits/rejection/duplicates, then restores its disposable
histories. BP3 normal filter20 is retained but no filtered mechanical solve is
qualified. Measured baseline results and its very large synthetic scales are
reported separately from PF1 unit-scale identities.

Evidence: [baseline manifest](../../../benchmarks/reconstructed_fault/particle_free/evidence/baseline.json),
[normalization](../../../benchmarks/reconstructed_fault/particle_free/evidence/native-normalization.csv),
[history paths](../../../benchmarks/reconstructed_fault/particle_free/evidence/native-history.csv),
[source/reference manifest](../../../benchmarks/reconstructed_fault/particle_free/evidence/source-manifest.json).
The PF2 proposal is in [PF1_REPORT.md](PF1_REPORT.md); no production work follows
without selection of that scope.

## PF2 implementation addendum — 15 October 2026

The user selected PF2 with fresh-source baseline, same-build FE/DG comparison,
authoritative DG history without particle-stress feedback, and nonzero physical
cohesive slip/traction. This supersedes the audit's earlier test-only boundary
for that restricted implementation. The law remains the matched-coefficient,
fixed-profile, nonrotational cohesive law derived above.

| Implemented state/operator | Source and ownership | Verified semantics |
|---|---|---|
| Independent DG trial tensor | `source/material_model/phase_field_fault/dg_history.cc:22` | Material-owned scalar DGQ2 space, three distributed component vectors. No particle-derived input; zz is reconstructed from the in-plane trace |
| Trial recurrence | `dg_history.cc:69` | Project native Q2 symmetric gradient after consistent3D deviation with Gauss3 mass/source integration; form a new tensor candidate from immutable incoming coefficients |
| Physical stress | `dg_history.cc:142`, `fe_strain.cc:369` | Evaluate DG trial tensor and subtract current fixed h/I times physical surface Z. The shared FE vector contributes its phase block only; its velocity block remains zero |
| Shared surface/localization state | `fe_strain_internal.h`, `fe_strain.cc:201`, `normalization.cc:296` | Existing fixed geometry, phase, consistent normalization and physical Z/I remain shared native data; no DG displacement-history evolution or new normalization |
| Mechanical inputs | `source/simulator/assemblers/reconstructed_fault_stokes.cc:78`, `:536`; `source/reconstructed_fault/surface_system_bulk_work.cc:138`, `:200` | Both native owners bypass mapped particle stress and use authoritative retained stress. Poisoned particle and mapped FE stresses give exactly unchanged actual loads/solutions |
| Terminal publication | `history.cc:644`, `fe_strain.cc:478`, `:544` | Compare against independent native Maxwell samples, validate complete candidates, then swap DG/Z/I and publish derived particle samples once. Rejected trial/step and duplicate checks retain accepted state |
| Restrictions / initial graph | `phase_field_fault.cc:436`, `:664`; `source/simulator/phase_field.cc:1536`, `:1727`; `fe_phase_energy.cc:102` | Explicit default-off selector, fixed serial homogeneous cohesive scope, rejected phase/mesh/transport/restart operations. Initial CPDI graph matches FE to retain identical accepted initialization |

The variable transformation and fixed-profile Maxwell/cohesive law are exact
under the selected assumptions. DGQ2 storage/source projection is an explicit
discrete replacement, verified on the actual affine mapping. The initial-graph
compatibility correction is numerical infrastructure for the new option, not
a change to the CPDI functional or physical seed. Generic particle transfer,
parameter defaults, spin, H and constitutive formulas are unchanged.

A fresh pre-edit build is now certified against 875 HEAD file hashes. Its
baseline is reproduced byte-for-byte in all bulk/surface field CSVs by the final
FE control. Final FE and DG share one executable/plugin and byte-identical
saved accepted phase coefficients; the measured PF2 gates pass. The earlier
exact-initialization failure and all PF0/PF1 evidence remain preserved.
See [PF2 report](PF2_REPORT.md) and
[new source/artifact manifest](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf2/source-manifest.json).

Remaining dependencies are substantive: phase-handler startup H/CPDI/domains,
surface chemical projection, particle/fault support services, particle material
data and validation during history publication, and derived stress/H samples.
Frozen H is still checked even though it cannot feed the fixed-phase mechanics.
This is not an end-to-end particle-free configuration. No PF3/transport/limiter,
MPI/AMR/restart or broader physical qualification follows from PF2.

## PF3 addendum — fixed-surface phase operator, 15 October 2026

PF0/PF1 and PF2 entries above remain historical. PF3 adds the selected explicit
phase operator; it does not enable outer physical DG phase stepping.

| Quantity / operator | Actual owner and source | Trial / accepted semantics | Particle dependency |
|---|---|---|---|
| Independent trial tensor T | `dg_history.cc::evaluate_dg_trial_tensor` | Batch values of accepted DGQ2 coefficients; no phase or displacement-velocity reads | None in tensor evaluation |
| Physical trial stress | `dg_history.cc::evaluate_dg_history_stress` | Optional explicit trial phase; accepted T and physical Z held fixed, `tau=T-2Z(h/I)S` | No particle stress input |
| Bulk/cohesive phase energy | `fe_phase_energy.cc::evaluate_fe_phase_energy_at` | DG owner bypasses raw FE-strain evaluation; actual projected-I gradient/Hessian pullbacks reused | Fixed geometry/particle restrictions still validated; no property/CPDI values used |
| Ordinary FE fracture | `PhaseFieldHandler::evaluate_cell_fracture`, `source/simulator/phase_field.cc` | Existing AT1/Gc/c0/ell, Q1 values and gradients, one shared mechanical rule for E/g/H | No H, composition or domain input after DG initialization; homogeneous material required |
| Full DG phase assembly | `PhaseFieldHandler::assemble_phase_field_system` | Initialized DG branch returns before accessing CPDI/domain/property managers; H is absent | Restriction validation still needs particle count/positions |
| Retained fracture comparator | `evaluate_fe_fracture` / fracture-only assembly | Still CPDI domain-average functional, H=0 for initialized independent history | Actual native domains; named separately from ordinary FE |
| Normalization | `fe_normalization.cc` | Fixed analytic rays and `P=M_F^-1 B_F`; DI and D2I included. Symmetric jet-product grouping changes arithmetic only | No new projection or normalization law |
| Bounds / lower bound | Existing native `solve_bound_constrained_phase` and test fixture | Immutable accepted phase in this fixed material configuration. Noncommitting trial solution; no per-iteration history update | No H/property values in selected assembly; native box/linear solver infrastructure reused |
| Publication / coupled time step | Existing `evolve_phase_field`, `validate_fe_strain_restrictions`, native history publication | Published phase and I remain frozen for DG; physical outer phase stepping, FE endpoint API use, transport and conversion still rejected | Existing startup/support/material validation and derived output remain |

The selected energy is ordinary shared-quadrature FE fracture plus the unchanged
bulk/cohesive functional from derivation `eq:energy`/`eq:phasevar`. The discrete
CPDI replacement is explicit in both authoritative design files and
[PF3_REPORT.md](PF3_REPORT.md). It is a different discrete functional, not a new
continuum constitutive law. Physical trial redistribution agrees with
`eq:phasejump`; it is not a stress-preserving numerical remap.

Measured evidence:67 final gates pass, native bounded KKT4.774e-15 with zero
healing, exact repeat/accepted-history preservation, and particle-property NaN
poisoning leaves E/g/H unchanged. Sharp-profile quadrature failure at6 points
and coarse phase-space nonmonotonicity remain explicit. The FE/DG frozen native
comparison still passes against the source-proven fresh baseline. See the report
for raw versus resolved errors, failed controls and qualification boundaries.

## PF4 addendum — prescribed DG material transport, 15 October 2026

The user selected PF4 as proposed. The new fixed-velocity transport operator is
noncommitting; historical PF2/PF3 source and qualification boundaries above
remain except for this explicit operator extension.

| State / operator | Actual owner and source | Semantics / particle dependency |
|---|---|---|
| DG space and tensor coefficients | `include/aspect/material_model/phase_field_fault/dg_history_internal.h` | Existing `PhaseFieldFault::DGHistory` definition moved intact from `dg_history.cc`; shared by history and transport, with the material still sole owner |
| Material advection matrix | `dg_transport.cc::make_dg_transport` | Native continuous Q2 velocity/divergence, shared-point upwind neighbor traces and prescribed boundary inflow; Gauss3 mass and Gauss4 volume/faces. Construction checks fixed native-history restrictions including particle count/positions |
| Divergence correction | Same method, `DGTransport::divergence_correction` | `+div(v) A` distinguishes material advection from density transport. The matrix is retained separately for a named negative control; production construction always includes it |
| Reaction/source | Same method and `advance_dg_transport` | Optional diagnostic r=0 or existing G/eta; explicit tensor source and inflow at SSPRK3 stage times. No native spin, material parameter or constitutive-source replacement |
| Analytic initial fixtures | `project_dg_transport_field` | L2 project prescribed scratch tensor into the existing DG space; no conversion of particle stress and no accepted initialization/publication |
| Trial time stepping | `advance_dg_transport` | Immutable input, temporary stage vectors, separate returned candidate; finite source/inflow and bounded diagnostic CFL checks. Advancement reads no particle property/domain or mapped stress |
| Geometry and surface labels | Benchmark `transport_checks.cc` only | Fixed native geometry and prescribed characteristic surface/localization data. No native surface Z/theta/phase-bound transport, birth or geometry motion |
| Cohesive / mature commutator checks | Same observer plus independent `assess_transport.py` | Cohesive redistribution and mature direct/residual equations evaluated in actual DG space. Residual is a scratch tensor, not a new mature production history or cohesive resistance |
| Published material state | Existing `fe_strain.cc`, `history.cc`, outer phase guard | Unchanged. Native material state remains unloaded/fixed; physical outer DG phase/transport integration and checkpoint conversion are not enabled |

The selected discrete equation is derivation `eq:DG` with the audited
non-objective derivative, including its divergence correction and no spin.
The analytic source controls separate that material law from the externally
forced objective identity. Full3D plane-strain norms reconstruct zz=-(xx+yy).
Consistent-I phase derivatives, CPDI startup, generic particle projection and
mechanical constitutive updates are untouched.

All284 final transport gates pass. Native/direct characteristic errors converge
at about third order in space and time; matrix/polynomial checks are near
roundoff. Omitting normal/tangential commutators leaves40.78%/5.45% errors;
retaining the mature residual gives converging, nonzero finite-discretization
closure defects. Signed jumps retain16.50%/15.82% sampled overshoot/undershoot
on32x64; no limiter or general nonsmooth qualification is claimed. The final
PF3 assessment is byte-identical to its retained result, and frozen baseline/
FE/DG authority, source, work and publication checks still pass. See
[PF4_REPORT.md](PF4_REPORT.md) for equations, measured quadrature/representation
errors, costs, provenance, remaining dependencies and the next proposed task.

## Selected bounded PF4 limiter comparison — 16 October 2026 UTC

The [user handoff](PF4_LIMITER_HANDOFF.md) supersedes only the earlier selected
unlimited experiment scope. [PF4_LIMITER_REPORT.md](PF4_LIMITER_REPORT.md) records
formulas, thresholds, exact source changes and measured rejection of adoption.
Existing laws/defaults, PF0–PF4 reports/evidence and incoming manual derivation
are preserved. This is optional scratch limiting, not constitutive qualification.

- Read the actual `../aspect-weno/source/simulator/limiters.cc`, SHA256
  `348ac0ad13904cf0c9bd6a25fd225b71bd90cef0be3839cc77d17f781c5a69fc`.
  It already extrapolates neighboring polynomials over the target, computes their
  **target** averages, restores the target mean, and writes separately. These
  are reused. No defect from historical `elasticity.cc` is imputed to it.
- The actual configured reconstruction rule in neighboring
  `source/simulator/introspection.cc:148` is degree+1: degree2 uses Gauss3,
  with a full-rank9×9 Q2 mass. The fixed Gauss2 rules belong to KXRCF. New
  reconstruction uses Gauss3 directly and retains every mixed Q2 mode.
- Correct the all-zero-smoothness division, use common full3D plane-strain
  tensor norms/weights and dimensionless reference-coordinate first/second
  derivatives. Pointwise inflow replaces whole-face integrated-velocity sign;
  prescribed physical inflow enters KXRCF, while reconstruction uses genuine
  interior neighbors only. Serial cell traversal imports no face-once MPI or
  coarse/fine qualification; all earlier unsupported-path guards remain.
- New `dg_limiter.cc` and shared private stencil/statistics data in
  `dg_history_internal.h` implement the handoff. `dg_transport.cc` optionally
  limits three SSPRK3 candidates, with immutable stage reads and separate writes.
  A null pointer preserves the existing unlimited default. No runtime physical
  parameter, accepted owner/layout, constitutive source, chi filter or stress
  clipping is added. Design/spec addenda explicitly describe the selected scope.
- The main detector uses norms of tensor differences before candidate mean
  shifting, patch **own-cell RMS**, an unweighted mean-mismatch neighbor sum,
  and a complete-polynomial RMS safeguard. RMS normalization and safeguard
  are experimental Fu–Shu adaptations. C=.1 was frozen from .05/.1/.2 saved
  snapshot screening before validation. KXRCF keeps threshold1 and
  `(cell diameter/domain diagonal)^1.5` scaling. Both use the same WENO.
- Local independent checks cover constants/piecewise constants, complete Q2
  including x²y², means, order, tensor covariance, zero crossings, direction
  changes, amplitudes1e±250, boundary/mixed inflow and normalization sensitivity.
  The mean-only mixed-xy indicator misses a discontinuity that Ipoly=.287678
  detects, but forced reconstruction changes it only.03749%. A constant
  background suppresses the RMS-normalized decision; no unrequested redesign.
- 77 local/snapshot and66 transport gates pass; operator/source/publication
  control is byte-identical to retained PF4. 628 native assertions and94 PF1
  checks pass; standalone DG limiter/transport compilation passes. No new PF2/
  PF3 native campaign was necessary or claimed. The first independent checker
  probe exposed integer CSV dtype truncation in its output buffer; float
  conversion corrected the checker, with the failed probe preserved.
- At32×64 both limiters reduce jump oscillation to below.05%, but increase
  relative tensor L2 from.17755 to approximately.3765, first-front width from
  .02353 to approximately.098m and quadratic-norm loss from3.574% to28.9%.
  Finest smooth rates pass2.5, with larger errors and greater cost. Neither
  earns adoption; defaults stay unlimited. Norm loss is not mechanical energy.
- Advancement/limiting do not read particles. Fixed-particle validation,
  initialization H/CPDI/domains, support/association, surface material projection
  and derived sampling remain particle-dependent. No accepted surface/phase-bound
  transport, persistence, MPI/AMR, moving geometry, coupled BP3 or PF5 is qualified.

Portable evidence: [pf4_limiter](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf4_limiter/README.md).
No commit/push or subsequent stage is selected or performed.

## Selected PF4 timestep/scaling/finite-width follow-up — 16 October 2026 UTC

The user selected four bounded follow-ups and confirmed the benchmark-only scalar
bounds, largest common scaling factor, and delta=ell/ln9 with actual analytic
transition overlap. [PF4_STABILIZATION_REPORT.md](PF4_STABILIZATION_REPORT.md)
records the complete measured comparison and exact formulas. Prior reports,
physical laws/defaults and evidence are preserved.

- `dg_limiter.cc` adds an optional common-factor scaling branch about the tensor
  cell mean, using full embedded3D projection onto the benchmark tensor and the
  specified GL9x9 control points. It retains all Q2 modes and clips no component
  or mean. A mean outside the envelope beyond128eps is explicitly rejected;
  all actual scaling stage means have zero excess. Constant-state activation
  from roundoff is avoided with a numerical comparison allowance, not a stress
  floor or tuned physical bound. No production selector/default changed.
- `dg_history_internal.h` adds scratch quadrature/stencil information and optional
  per-call norm/mean/capture statistics. The exact Gauss3 quadratic norm change
  uses polarization to avoid subtracting large norms. Diagnostics default off;
  accepted state/layout/checkpoints, native transport/SSPRK3 and material-law
  updates are untouched. WENO formulas/thresholds/weights remain unchanged.
- `stabilization_checks.cc` reuses native projection, transport and observer
  infrastructure. It compares16x32 and fixed32x64 dt/dt2/dt4, with common physical
  output times; errors use the fixed actual tanh pulse with ell=.15625m read
  from the phase handler. No phi/chi/history smoothing or new physical parameter.
  Held-field12-call diagnostics at fixed time isolate non-idempotent map effects
  without advancing physical state or changing a trajectory.
- `assess_stabilization.py` independently checks the common factor, tensor mean,
  all Q2 coefficients, norm contraction, captured WENO/scaling calls, common-time
  fields, analytic peak/overlap/fronts and SSPRK3 norm identity. Direct stage-call
  sums and RK-weighted terms (1/6,2/3,1) are distinguished from the remaining
  Euler/SSP norm change and analytic boundary flux. None is a mechanical-energy
  qualification. `plot_stabilization.py` plots the measured prescribed fields.
- All149 transport/ledger,45 local/algebra and77 retained WENO gates pass.
 628 assertions/8 native cohesive-Maxwell cases,94 PF1 checks and standalone
  limiter/transport compilation pass. Fresh original operator/publication
  results and prior base-step unlimited/WENO fields are byte-identical.
  Final mean-excess logging leaves all36 final comparison fields byte-identical.
- At32x64 original dt, scaling reduces sampled signed-pulse overshoot to
  roundoff with L2 error.22675 and Q loss12.549%, versus WENO approximately.3765
  and28.9%. Scaling's direct RK-weighted Q change is-9.595%, versus WENO-6.30%:
  larger total WENO loss is not a larger direct contraction. Finite-width KXRCF
  increases Q in257/309 calls. At dt/4, sharp WENO fields change approximately13%
  from dt/2, so this bounded family is not temporally resolved.
- For ell-width resolved by5 cells, unlimited finite-width L2 error is.04276%,
  positive peak attenuation.003715%, first-front broadening.02889%; scaling
  remains idle. KXRCF dt/4 gives5.1565% error and15.0868% first-front broadening.
  These data justify keeping unlimited for the selected smooth control and
  retaining stabilization as experiments, not designing/adopting a more elaborate
  method here. GL9 bounds are not continuous bounds; polynomial trace extrema
  reveal a small between-point scaling overshoot, explicitly reported.
- Particle startup H/CPDI/domains/support/surface projection dependencies remain.
  Limiting/advancement read no particle properties and publish no accepted state.
  MPI/AMR/periodic/moving/coupled transport, surface-history/phase-bound transport,
  restart, PF5 and propagation remain outside this task. No general physical
  stress envelope or coupled stabilization has been qualified.

Portable evidence and commands:
[pf4_stabilization](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf4_stabilization/README.md).
All final native runs finish within180s, at most two simultaneous, one rank/thread;
maximum164.762s. No commit/push or subsequent stage performed.

## PF4 selected transport-method comparison — 16 October 2026 UTC

This addendum audits the separately selected bounded comparison. Production
source, defaults, physical laws and previous evidence are unchanged. Worktree
and branch remain `/home/ein/repository/aspect-pf-rsf-particle-free` /
`pf-rsf-particle-free`, HEAD `3232056cde12dbae17ef4265ecb804f9098b88f1` plus the
preserved working tree. The new `PF4_METHOD_COMPARISON_REPORT.md` and
`evidence/pf4_methods/` distinguish fresh scalar/native controls from reanalysis
of retained DG/scaling controls. There were no WENO parameter or timestep sweeps.

| Actual primary source | Audited behavior |
|---|---|
| `source/simulator/parameters.cc`, stabilization declarations | Native EV is default, SUPG experimental; alpha=2, cR=.11, beta=.052, gamma=0, smoothing off. The benchmark selects each path without changing these parameters |
| `source/simulator/entropy_viscosity.cc`, `compute_viscosity`, `get_artificial_viscosity` | Scalar old/old-old midpoint entropy and residual, global range/entropy normalization, cell-diameter cap/startup fallback. SUPG tau is h/(2 p norm(v)) for zero physical diffusion. The independent-history DG path does not call these routines |
| `source/simulator/assemblers/advection.cc`, `AdvectionSystem` | Native continuous-Q2 composition mass/advection plus isotropic EV diffusion, or SUPG test including transient BDF residual. First physical step BE; BDF2 thereafter. No composition clipping |
| `source/simulator/assembly.cc`, `source/simulator/introspection.cc` | Actual scalar assembly and stabilization quadratures are Gauss3 for these Q2 inputs; existing DG Gauss3 mass/Gauss4 transport remain unchanged |
| `source/simulator/helper_functions.cc`, extrapolated range | Q2 support-point range, old-field extrapolation after startup; separate Gauss3 entropy integration. Tensor components are treated as independent scalar fields |
| `source/simulator/initial_conditions.cc` | Native composition initialization is support-point interpolation, not an L2 projection; the sharp pulse therefore starts with appreciable representation/interpolation error |
| `source/simulator/solver_schemes.cc`, prescribed Stokes plugin | `single Advection, no Stokes` invokes actual native scalar solves and prescribed constant velocity without a mechanical solve. No production history backend is replaced |
| `source/material_model/phase_field_fault/dg_history.cc` | The representative initial tensor comes from native FE strain-gradient projection with the audited full 3D deviator and matched Maxwell kappa. Independent differentiation checks the actual Q2 source; no particle stress is substituted |
| `source/reconstructed_fault/surface_system_bulk_work.cc` and cached manager QP associations | Frozen weak maps use actual JxW chi, Q1 surface tests, shear tensor S and normal tensor n⊗n. Scratch native stress reconstruction and an independent evaluator agree; no scratch state is published into mechanics or the surface system |

New benchmark-only files are `scalar_method_checks.cc`, `native_method_checks.cc`,
`transport_method_fields.h`, eight explicit parameter files, independent
`assess_methods.py`, and `plot_methods.py`. CMake registers two separate diagnostic
DSOs. The existing runner gains an opt-in `--no-particle-observer`; its default
expands four old fixtures byte-identically. No production parameter/interface,
DG operator, limiter, source/transport quadrature or accepted-state rule changes.

The scalar pulse bound [-1,1] appears only in the pre-existing benchmark scaling
reference. It is not a physical bound on native independent history. The
native history fixture has non-proportional tensor components and nonzero shear
and normal weak loads; its native G/eta source coefficient is retained. Its
subsequent zero-source/reaction transport is explicitly a prescribed subproblem,
not a new constitutive evolution law. Full tensor errors include zz and both xy
contributions. Source normalization, chi and Z are never limited.

The independent reference assembles DG/CG mass systems from Q2 polynomials.
Sharp/native polynomial integration splits at actual transported discontinuities;
Gauss3 is exact on each piece. Finite-width Gauss16 versus Gauss24 checks the
smooth reference. A separate Legendre-moment calculation verifies the sharp
32×64 projection error 0.149862170276557. Orthogonal representation/evolution
errors and actual initialization error are distinguished, including exact
characteristics of the initialized polynomial. Continuous projections are
unconstrained-space lower bounds, not claims about strong boundary admissibility.

The initial benchmark plugin incorrectly queried a not-yet-built mesh during
adiabatic initialization; the corrected fixed-grid plugin reads its mesh
resolution from the existing refinement parameter. Two failed launches remain
preserved. A preliminary symmetric native source cancelled the normal weak
load; the final prescribed source adds an asymmetric y term to exercise both
loads. Neither correction affects a production equation. All successful pulse
fields are unchanged by the stronger native source. Native polynomial numbering
is exported and used explicitly rather than assuming CG and DG local numbering
coincide.

Remaining particle dependencies are unchanged: H/CPDI/domain initialization,
fixed-particle admission, support/associations, surface material projection and
derived sampling. The experimental transport advance reads independent tensors
and prescribed FE velocity only. Accepted publication, phase/surface transport,
restart, AMR/MPI, periodic/moving geometry and coupled paths are not enabled.
