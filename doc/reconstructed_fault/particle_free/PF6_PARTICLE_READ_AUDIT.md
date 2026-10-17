# Stationary mature DG/FE configuration — remaining particle reads

16 October 2026. [Mature report](PF6_MATURE_DG_REPORT.md),
[scope](PF6_MATURE_DG_FORMULATION.md). Historical audits remain below.

The homogeneous mature fixture retains 66,176 fixed reference-cell samples,
not Voronoi domains. `Integration scheme = BP3 stationary samples` is the
explicit no-motion integrator. `Evolve phase field=false` also freezes H;
that flag alone would not freeze particle coordinates. The observer compares
all IDs, coordinates and inert H before and after every accepted step and
requires the same population. No insertion, removal or advection is selected.

| Read/operation | Domain-free mature DG/FE role | Replacement still needed |
|---|---|---|
| Generator, native ID/CellId/owner, manager/property metadata | Fixed independent test-point ownership and trace selection | Mesh-owned deterministic points with equally strong trace/source checks |
| Initial H, Theta, strengthening and zero stress properties | Diagnostic initialization; no phase, surface state or material projection | Optional diagnostic fields on mesh points; directly prescribed surface Theta already replaces its particle projection |
| Phase-handler manager/H lookup | Finds the associated diagnostic owner; every physical phase DoF is prescribed from the native stationary profile | Decouple phase setup from diagnostic-manager registration |
| FE-history initialization/restrictions | Verify unloaded point samples and immutable positions/population | Equivalent independent mesh-point registry |
| Geometry-only point/fault associations | Point position and fault coordinates only; empty quadrature and NaN domain measure | Mesh-point geometric association; keep projection-factor rejection until then |
| Native terminal publication | Read old diagnostic stress/H, matching surrounding CellId and coordinates; independently form the direct native Maxwell update and compare with reconstructed DG/FE stress before writing derived samples | Retain until a separately verified native-gradient versus independent-history validator replaces it |
| Particle integrator/property updates, backup/ghost infrastructure | Stationary lifecycle and derived accepted diagnostics; no mechanical history feedback | Dedicated diagnostic owner; mature DG restart remains explicitly unsupported |
| Work, monitor and visualization observers | Diagnostics, fixed-population checks and output scheduling | Mesh/surface authoritative exports already added; remove redundant point outputs separately |

Bulk coefficients come directly from configured homogeneous data. Initial physical
phase comes from fully prescribed mesh constraints. Geometric support uses bulk
mesh margins, native stationary-profile support and fixed prescribed fault geometry.
Nodal Theta is prescribed before preparation. Accepted mechanics and fault weak
loads read independent T/Z/I (or native FE d_R/Z/I), never diagnostic particle stress
or particle material labels. The independent consistency check can reject a bad
publication; this acceptance dependency deliberately remains.

The matched particle control is a separate legacy backend: native domains/CPDI,
particle material/state transfer and stress-to-FE LLS interpolation remain enabled.
It uses the same physical inputs and selected 3D-deviator convention. Its exported
Z=0 denotes absent independent storage, not zero physical memory. Its exported
stress is the accepted native mechanical evaluation from actual frozen incoming
transferred FE history and current velocity; it is not a newly remapped point
cloud or a second Maxwell increment after publication. Original legacy
symmetric-gradient results and all domain failures remain preserved separately.

---

# Homogeneous domain-free configuration: remaining particle reads

16 October 2026. [Measured report](PF6_DOMAIN_FREE_REPORT.md). The following applies
only to default-off `Use homogeneous mesh data`. The historical CPDI inventory
below remains applicable to the separate legacy configuration.

| Remaining read/operation | Selected effect and future replacement |
|---|---|
| Reference-cell point generator and native ID/cell/rank ownership | Same 144 diagnostic points; later use deterministic mesh-owned test points if the independent check retains equivalent trace/error coverage |
| `PhaseFieldHandler::initialize` associated manager/H property lookup | Finds the diagnostic point owner; it no longer requires domains or CPDI. A dedicated cell-point validation owner can remove this lookup later |
| `crack_driving_force` initialization and manager's analytic prescribed-core H setup | Configured threshold and native analytic diagnostic H. No particle chemistry/domain is read, and H does not initialize or drive the physical phase in this mode |
| `maxwell_stress` initialization | Direct zero diagnostic samples of unloaded independent history; no FE stress mapping or particle-to-bulk transfer |
| `fe_strain.cc` ID/position snapshot and unloaded diagnostic stress check | Enforces fixed check-point population/positions; authoritative T is initialized independently to zero |
| `history.cc` ID/position/CellId, diagnostic incoming H, publication stress/H writes | Matching-cell native FE source and independent DG Maxwell comparison, followed by derived samples only after acceptance. Keep this guard until a separately checked mesh-point replacement exists |
| Geometry-only fault associations | ID/physical point/normal-profile fault coordinate; no volume, domain vertices or quadrature. Support comes from bulk mesh/fixed fault. Particle projection operators reject this mode |
| Native RK2, ghost ownership, backup/restore, checkpoint/RNG | Effective advection dt=0, unchanged constitutive dt. Same-count replay exact; changed-count native RNG replay unavailable and population changes disabled |
| Benchmark/visualization exports | Diagnostic point properties and fixed positions only. Non-authoritative static FE stress slots are labeled separately from reconstructed DG physical stress |

There is **no selected particle material/domain projection or CPDI initialization**.
Configured homogeneous background data supply physical materials directly; the
supplied physical phase seed initializes Q1 phase; native bulk/fixed-surface
quadrature and existing normalization/derivatives remain. The input removes the
unused chemical particle property and all particle-to-FE mappings. Every exported
selected state has a null domain-handler pointer and unweighted associations;
a domain read would throw. No particles are added or moved.

Legacy CPDI/material projection retains the domains, domain fixes and coverage
failure documented below. Its genuine sparse support limitation no longer blocks
this homogeneous DG configuration. Heterogeneous materials, diagnostic-point
removal, surface transfers and moving/born faults require separately selected work.

---

# Local-transfer particle-dependency update — 16 October 2026

[Measured local report](PF6_LOCAL_REPORT.md). All 144 IDs/physical coordinates
remain identical in the supported serial/1/2/4-rank transfers and restarts.
Refinement changes four surrounding CellIds, and repartition changes three
owner ranks in each multi-rank local fixture. Local slots can change for many
more particles; no physical movement follows from those index changes.

The former refinement lifecycle left domain geometry indexed by old local slots.
Selected endpoint domains now rebuild after final unpack/ghost exchange. A separate
closed-upper-bin defect in the Voro adapter dropped boundary candidates; a derived
outward artificial-wall bound and insertion-count check repair that omission.
The original volume and physical-coverage guards remain. All exported supported
domains match independent global Cartesian Voronoi rectangles. A genuine one-ring
support failure remains at 256 total cells for this sparse fixed population;
neither moving particles nor adding points/volumes is used to hide it.

These reads still affect initialization, material projection and whether remapping
can proceed, despite no particle-stress feedback into accepted mechanics or loads.
Prior replacement proposals below remain: independent FE/analytic materials and
cell/surface quadrature, explicit native phase initialization, geometry-only
associations and independent cell-point publication checks. Do not remove the
existing consistency guard while replacing those consumers. Same-rank RNG restores
exactly; changed-rank native RNG replay is unavailable and population changes are
rejected unless replay is compatible. No stochastic birth/replenishment claim follows.

---

# Particle reads in the bounded fixed-reference DG fixture

Source audit, 16 October 2026. This inventory distinguishes an active read from
an authoritative physical history. Positions are frozen, but the particle
manager still runs RK2 stages, property updates, ghost exchange, domain rebuild,
backup/restore, and checkpointing. Fixed particles do not mean those operations
are skipped. The effective freeze switch is presence-based, not Boolean:
`source/particle/integrator/rk_2.cc::local_integrate_step` replaces its requested
advection dt by zero when `ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION` exists.
The constitutive timestep is unchanged. The selected input explicitly uses rk2.
All 144 IDs and physical coordinate strings are identical at initialization,
before/after both physical steps, every exported remap and restart in the
1/2/4-rank fixture. Eighty particles are on child faces. Cell/rank ownership and
reference coordinates change with remeshing, while physical positions do not.
See [measured results](PF6_TRANSFER_REPORT.md) and the archived `particle-freeze.txt`,
particle CSVs and independent comparisons.

| Read / owner | Selected fixture effect | Replacement to qualify later |
|---|---|---|
| Particle generator, initializer, mapped composition fields; `source/particle/manager.cc`, `source/particle/property/{initial_composition,crack_driving_force,maxwell_stress}.cc` | Initial locations/IDs, chemical labels, H, unloaded stress, integrator storage; material metadata and native initial FE composition | Analytic/material FE initialization and explicit independent histories; compare every coefficient and initial seed before removing |
| `reconstructed_fault/manager.cc` initial reconstruction/H initialization and prescribed geometry fitting helpers | Startup H/CPDI seed and prescribed full-normal associations; fixed reference surface is retained, no later particle-derived surface motion selected | Direct documented physical phase seed, fixed surface geometry/connectivity, analytic initialization of material and surface fields; initial seed equivalence is a separate gate |
| `simulator/phase_field.cc::make_sparsity_pattern`, initialization assembly/`assemble_phase_field_system` fallback | CPDI/domain infrastructure is still required and its graph is built. The supplied FE seed replaces the initial phase after native initialization. Accepted DG endpoint fracture uses ordinary cell FE quadrature, bypassing particle H/composition/CPDI integrals | FE phase setup/initialization without CPDI and a phase graph built from the actual selected functional; preserve the existing seed and verify energy/KKT independently |
| `particle/particle_domain.cc`; `reconstructed_fault/manager_particle_projection.cc` cache, domain volume, associated ID/position, chemical projection | Active fixed-support ownership and projection weights; chemical material fields on the surface. Domains are regenerated from fixed positions, including after transfers/restart | Cell/surface quadrature with explicit geometry-only association and FE material interpolation; verify constant and heterogeneous coefficients separately |
| `material_model/phase_field_fault/normalization.cc::compute_fault_surface_compositions` | Projects particle chemical composition to fixed surface material properties. Homogeneous matched G/eta/Gc/c make it physically redundant here, but the read remains | FE/analytic material evaluation at surface quadrature, with a defined surface projection and independent heterogeneous-material check |
| `simulator/phase_field.cc::PhaseFieldInterpolator::properties_at_points`, `is_in_crack_zone` | Reads particle properties/H and domain interpolation weights to select and populate mapped FE fields. Mapped stress is overridden by independent DG history; chemical fields still enter material inputs. Homogeneous matched coefficients remove that sensitivity only in this fixture | Independent FE/analytic material fields and direct DG diagnostics, including replacement of the H-dependent interpolation selection |
| `simulator` particle-to-composition mapping (`interpolate_particle_properties`); `phase_field_fault.cc::evaluate`, `surface_system_bulk_work.cc` | Bulk material composition originates in particle fields. Independent DG stress overrides mapped stress in accepted mechanics and weak loads; non-stress chemical coefficients retain this dependency | Preserve material composition as a native FE field or prescribed material function; avoid interpreting removal of stress feedback as removal of material dependence |
| `phase_field_fault/fe_strain.cc::prepare_fe_strain_history`, `validate_fe_strain_restrictions` | Reads initial particle stress to require unloaded initialization; owns ID/position snapshot and rejects motion/birth/deletion. New transfer rebuild migrates this ownership without moving coordinates | Independent initial-history contract and mesh/geometry epochs; retain motion/population rejection until transport/birth are selected |
| `phase_field_fault/history.cc::sample_accepted_history`, `compute_history_candidates`, `commit_reconstructed_fault_mechanical_history` | Reads ID, cell, position, chemical properties, H and surface association. Native Maxwell candidate is evaluated on the matching CellId for DG. Incoming physical stress comes from accepted DG, not stored particle stress. Stored H is validated/updated as derived output; the actual consistency guard can reject publication | A nonparticle independent source/recurrence evaluation at owned cell test points. Keep the particle check until matched-cell equivalence and its failure modes are covered; do not delete it to make a run pass |
| `phase_field_fault/history.cc::evaluate_initial_cohesive_particle_values`, legacy projected-cohesive branches, RSF state projection | Available conditional reads of particle H/composition/stress/Theta. **Not active** for unloaded surface-elastic-slip initialization and fixed-friction PF6a. These are not qualified replacements | Explicit initial surface histories and future state/birth rules, when those paths are selected |
| `reconstructed_fault/surface_system_particle.cc` | Alternative particle/CPDI mechanical work path; **not the accepted bulk-conjugate DG weak-load path**. Its existence must not be counted as active particle stress feedback | Separate FE-versus-CPDI comparison; no equation/path replacement in this task |
| `surface_system_bulk_work.cc`, `surface_system_diagnostics.cc`, visualization/particle output and benchmark observers | Optional normal-stress/line/particle comparisons read particle stress and interpolated samples; metadata and output only in the selected path. Mapped FE stress may still be sampled before the independent-history override | Diagnostics directly from owned DG polynomials, labeling any actual particle-transfer comparison separately |
| `particle/manager.cc`, `particle_domain.cc`, simulator checkpoint/retry signals, reconstructed-fault association persistence | IDs/positions/properties, ownership, backup/restore, RNG, ghost/domain reconstruction, archival. These can affect whether a trial succeeds even with zero particle stress feedback | Remove persistence only after all remaining physical/setup consumers have independent owners; preserve restart conversion guards and demonstrate equivalent rollback first |

The accepted bulk mechanics uses native FE velocity gradients and independent DG
T, reconstructing physical stress as T-2 chi Z S. The surface bulk-conjugate
functional reads that same independent physical history. The endpoint phase
energy reads T/Z/phase and ordinary FE fracture. No claim of a particle-free
initialization, material/geometry pipeline, or persistence system is made.
