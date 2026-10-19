Storage maintenance, 19 October 2026: runtime/source/equation bytes and live
build/checkpoint files verified unchanged. Only duplicate immutable executable
storage, lossless historical-field archiving and redundant expanded package
copies changed; original errors and data remain recoverable.
[Measured cleanup and restoration](DISK_CLEANUP_20261019.md).

# BP3 work-observer repair audit — 19 October 2026

[Source/verification report](BP3_DG_WORK_OBSERVER_FIX.md).
The supplied signed weak loads establish the failed projection at node 4918;
high-precision reevaluation confirms the normalized errors from their actual
binary operands. Matching the native frame makes the step-816 normal error
at that node 5.960462e-9 Pa, without changing the 1e-5 Pa bound. Direct frame
differences and separately rounded integrated-load differences are reported
separately. Archive/raw input hashes and the original failures remain.

The sole runtime edit is `bp3_native/server/plugin/work_audit.cc`. Its evaluator
now receives explicit S/N tensors; QPs use the actual cached Stokes association
frames. Point/domain diagnostic frames use their owned segment, matching native
DG stress reconstruction. Detailed exports, first Maxwell diagnostic, and the
guarded native work loads all use those frames. The first-segment contraction
is retained separately as `original_*` trace data; `matching_*` are checked and
`guard_uses_matching_frame=1` identifies the repair. Failure CSV field meaning
and collective failure decision are retained. The literal assertion/message,
raw-normal rule, weights, pressure/background signs and independent publication
checks are unchanged. Core source, constitutive equations, histories, checkpoint
format, physical settings and defaults remain unchanged.

Two-rank fresh/rejection/restart controls and an intentionally inconsistent
corrected-load negative test verify the actual changed plugin. Physical output
comparisons are byte-for-byte: nine against the prior build, eleven for retry,
fourteen after restart. Once-only publication passes. Ordinary plugins compile
and both new native server inputs validate. The negative compile macro remains
absent in ordinary builds. Full-grid repaired advancement is still a server
confirmation; no job was launched. Strict matching remains false.

---

# Server replay 3613912 log audit — 19 October 2026

The supplied native replay measures the original normal-load guard violation
at 816/node 4918 (1.0013576e-5 Pa), the direct frame contribution at the same
node (9.9316234e-6 Pa), and passing matching-frame maxima (normal 2.0868e-7 Pa,
shear 5.0679e-7 Pa). The unchanged-threshold shadow passes all seven steps.
Mechanical/publication log records match the original run, as do accepted
810–815 time/dt. [Reduction and hashes](../../../benchmarks/reconstructed_fault/particle_free/evidence/server_3613912/log-analysis.json).

This confirms that replacing the observer's first-segment projection frame by
the actual mechanical QP association frame removes this observed violation.
Signed nodal loads are still requested before completing repair verification.
No runtime/core source edits in this follow-up; independent publication,
constitutive laws, numerical settings and original guards remain unchanged.
The prior report/evidence and strict-match=false are preserved. The next source
change should correct observer frames rather than enlarge an allowance.

---

# Server 3610007 observer diagnosis audit

[Report](BP3_SERVER_3610007_DIAGNOSIS.md). The newly supplied state/load exports
end at 815. Original logs identify the unchanged accepted-work guard at 816,
after converged mechanics and passing independent publication. Checkpoint
metadata identifies 809/slot 03. Original raw data and errors are preserved.

`surface_system_bulk_work.cc` projects stress with each cached association's
tangent/normal. `server/plugin/work_audit.cc` projects with the first segment's
frame. Independent per-anchor native resampling reproduces 5776 nodes and
demonstrates a 1.7796e-5 Pa point-traction mismatch from this choice at 15 MPa
shear. This does not reproduce the unavailable failing native weak integral.
The 9.9101e-6 Pa estimate from step-815 exports is explicitly a surrogate.

Only `work_audit.cc` is edited for this diagnosis. Original load arithmetic and
the literal `consistency<1e-5` assertion/message remain. Separate arrays sample
the same accepted physical stress with native QP frames and directly integrate
the frame-difference contractions. A root trace precedes the assertion; an MPI
maximum of failure flags precedes collective root failure export. This observer
does not alter accepted history, velocity, phase, coefficients or source cache.
A compile-only local negative-test macro perturbs a diagnostic load, never
mechanics/history, and is absent in ordinary builds. No new production parameter.

Two-rank adaptive control: twelve output files are byte-identical to the earlier
control. A deliberate 2e-5 Pa diagnostic inconsistency still fails the original
guard, writes 117 finite rows and aborts within its timeout. Ordinary plugin
rebuild/replay parser check passes. All five changed replay parameters concern
output/restart/termination; no physical or adaptive-control changes. The first
failed parser attempt (resolved-print inline comments in library names) remains
in evidence; removing comments preserves the parsed parameter dictionary.

The actual server failure mechanism remains unconfirmed pending a bounded
809→816 replay. No threshold relaxation, observer-frame repair adoption or
full-event claim is made. Source/artifact hashes and original input hashes are
in the linked evidence. Existing scientific failures and strict-match=false stay.

---

# Homogeneous-coefficient startup repair audit

[Report](BP3_DG_HOMOGENEOUS_COEFFICIENT_FIX.md). The original server log and a
fresh bounded mixed-label run identify the failure at the unchanged exact
Maxwell-coefficient guard. The source defect is inconsistent material arithmetic:
ordinary bulk assembly and independent history use configured homogeneous G,
whereas constitutive bulk/fault responses still average equal G values using
spatially varying chemical fractions. Harmonic rounding makes kappa nonconstant.

`compute_elastic_shear_modulus()` in the material returns configured G when
homogeneous mesh data is selected, and the original compositional average otherwise.
`phase_field_fault.cc`, `constitutive.cc` and `history.cc` share this selection
for native viscosity, frozen and current point stress, surface candidates and
history/initialization coefficients. Test-only surface coefficient accessors use
the same production accessor. Native constitutive inputs, recurrence, rate/spin,
full plane-strain convention and Maxwell exp/expm1 expressions are unchanged.

`surface_system.cc` is byte unchanged from task entry: the exact homogeneity
assertion, filtered factorization, cache dependencies and collective decisions
are preserved. Normalization/degradation, derivatives, quadrature, tolerances,
publication and checkpoint formats are untouched. No parameter change is needed.
The regression covers 1001 mixture fractions/four averaging selections plus
independent heterogeneous-mixture controls. 127 native cases / 165094 assertions
pass. Bounded native operator/MPI/restart/rejection evidence is linked in the report.

The local runner adds an explicit mixed-label fixture option; server inputs
remain unchanged. The packager excludes the copied `log.txt` from runtime inputs
while retaining it as qualification evidence. Earlier source archives and
all incoming manual work/scientific failures remain. Strict reference matching
remains false; no further propagation or server job is launched locally.

---

# Stationary mature DG optimization audit — 17 October 2026

Reviewed base `b2dfdebe43437d6028b06d4f9f862cc80a00cc71`; no tracked subsequent
changes at entry. [Measured report](BP3_DG_OPTIMIZATION_REPORT.md) and
[exact factorization/cache dependencies](BP3_DG_OPTIMIZATION_FORMULATION.md).
No physical law, rate/spin convention, quadrature, limiter, native adaptive
control, pressure scale, acceptance or publication allowance changes.

- `phase_field_fault.{h,cc}` adds an independent optimization eligibility predicate
  and reference selector for the homogeneous fixed-phase mature accepted-DG route.
  It does not enable consistent endpoint phase or change the default history backend.
- `utilities.{h,cc}` adds conservative indexed segment admission with exact native
  leaf arithmetic/order and reference fallback. `manager.cc` owns/invalidate its
  geometry lifetime; knot ownership, overlap rejection and endpoint completion stay.
- `dg_history.cc`, `fe_strain.cc`, `history.cc` separate geometry/QP/basis/fixed-phase
  layouts from accepted stress values and current Maxwell/source operands. Reuse
  Stokes associations; batch cell DG/native FE values and independent publication
  gradients. Cache Gauss3 mass inverse/invariant roundoff geometry, retaining nine
  modes and the original operand-derived two-path allowance. Changed trial phase
  still bypasses the fixed-phase cache outside the selected mature route.
- `reconstructed_fault_stokes.cc` assembler and `surface_system_bulk_work.cc`
  consume the same batched authoritative reconstruction at actual physical QPs.
  Sparse B expands closed constraints using the retained native implementation.
- `surface_system.cc` factors the actual filtered derivative as
  `S_Gamma-W_mu H^-1 N_Gamma`. S/N retain native work weights and RPE trace average;
  consistent W_mu, frictional tangent and active-set solve refresh per required
  linearization. No commutation of variable friction, nodal product or dense map.
- Simulator solver/assembly extend homogeneous **Stokes-direction** eligibility
  for within-attempt A/preconditioner reuse and residual-only line searches.
  Nonzero thermal constraints previously triggered an A overwrite; the observed
  failure and correction are retained. Every new attempt/retry rebuilds bulk
  operators. Condensed wrappers preserve constraints/pressure scaling and reuse
  owned/ghosted work vectors without changing `C=A-B K_V^-1 G`.
- `bp3_native/coupled_checks.cc` adds opt-in source/QP/operator probes and batches
  native operand sums/dense-layout preparation; `server/plugin/work_audit.cc`
  batches its independent accepted-stress QPs. Coverage/weak integration unchanged.
  Rank-local relaxed counters and existing timers distinguish searches, misses,
  FEValues constructions, setup/application and history/diagnostic costs.
- New native units cover indexed search, exact admission/fallback/overlap and a
  5776-node varying-friction kernel. Native fixture probes cover full operators,
  hanging constraints, pressure scaling, restricted K and non-proportional history.
  `ASPECT_FAULT_COMPARE_COUPLING` is an expensive verification-only selector; its
  2e-11 relative action guard does not alter production tolerances.
- Server template retains physical/adaptive inputs with an explicit optimization
  selection. Packager includes the bounded verification helper closure and native
  unit profile fixture; output control/checkpoint formats remain unchanged.

126 cases / 149074 assertions pass; B/G/C relative differences <=1.88e-15,
1.77e-14, 9.46e-16. Source/allowance arrays match bitwise; independent stress
sampling <=3.09e-12 Pa. Full timing-pair stress RMS <=2.10e-10 Pa. Rejection,
cutback, 1/2/4-rank ownership, restart, publication and collective stop pass.
Measured time/memory tradeoffs and retained engineering failures are in the report.
Stationary diagnostic point identity/trace validation remains an acceptance
particle dependency; domains/particle stresses do not feed selected mechanics.
Legacy CPDI and all previous evidence remain. `strict_reference_match_passed=false`.

---

# BP3 server startup review audit — 17 October 2026

[Report](BP3_DG_STARTUP_REVIEW.md). Task entry: base
`a8707307008401ac9cff702a369e2e2a611fa0f7` plus preserved worktree. Scientific
source, independent publication allowance, defaults and prior failures remain.

- `bp3_native/server/plugin/output.cc` reduces the complete local stop request
  with integer MPI maximum before scheduling any output. Heavy/profile flags
  are also collective before native output gates and root-write collectives.
  Output completion and termination consume the same shared decision.
- `bp3_native/coupled_checks.cc` retains its native-output dependency and makes
  the scheduled field flag collective before exports/index writing. Two native
  timer scopes label dense stress/coefficient costs; equations/checks unchanged.
- `source/simulator/checkpoint_restart.cc` adds only a native `Resume snapshot`
  timer covering read/restoration. Storage, payload, rotation and state unchanged.
- `bp3_native/tests/collective_stop/` is a separate bounded test build. A compile
  definition injects a rank-1-only wall request at accepted step 2 and exports
  request/decision traces. The ordinary server adapter defines no test macro.
- `startup.prm`/`startup-resume.prm` change only accepted-step bounds (5/8),
  checkpoint cadence (1), graceful budget (7200 s) and resume mode relative to
  first-event inputs. Native adaptive timestep/physical inputs are identical.
- DG checkpoint serialization is not redesigned: all_gather still replicates
  39 doubles/cell plus containers, buffers and packed histories on every rank.
  Resource collection is an explicit server startup gate, not a memory claim.

Current two-rank wall-stop/restart fields match the uninterrupted control exactly;
all four parameter files validate and the rebuilt unit suite passes 124/118034.
Prepared server startup and unqualified first-event evolution remain distinct.
`strict_reference_match_passed=false`; detailed commands/results and retained
accuracy limitations are in the report and package qualification records.

---

# First-event source-distribution audit — 17 October 2026

[Package report](BP3_DG_FIRST_EVENT_PACKAGE.md). ASPECT core, headers, unit tests
and build inputs match task entry exactly. All earlier science failures and
particle controls remain. No publication allowance or physical check was removed.

- `bp3_native/coupled_checks.cc` adds default-off `Scheduled field output`, using
  existing BP3 OutputSchedule/FirstEvent helpers and Heavy-output parameters.
  Dense tensor/coefficients/cut exports observe accepted states only; scalar
  checks, profile/weak loads and native history publication still occur every
  accepted step. Root writes the replicated weak loads in this selected mode.
- The selected output schedule/event state uses its own optional checkpoint key;
  inactive/default diagnostics retain their original serialization. Reuse of an
  older unscheduled checkpoint with the selected schedule is explicitly rejected.
- Per-step CSVs use existing `BP3::needs_header` for fresh restart branches. This
  repairs missing observer headers without changing history/state or old data.
- The new first-event CMake adapter compiles maintained BP3/observer sources from
  the supplied ASPECT tree; no local test/loading macros are defined. It exposes
  runtime libraries/executable in `plugin/`, leaving objects in `plugin/build`.
- Standalone fresh/resume inputs select the previously qualified homogeneous
  stationary mature DG route, native adaptive BP3 time controls and native
  first-event termination. Phase, transport, mesh and geometry remain fixed.
  Environment setup changes variables only; build and `ibrun` are manual.

Default two-rank fields reproduce 36 retained files byte for byte; scheduled
fields match 20 corresponding files. Split restart to step five is bit identical
in all compared histories/tensors/loads and does not emit a spurious step-four
snapshot. Full first-event accuracy, cost and larger-rank behavior remain server
questions. `strict_reference_match_passed=false` remains in the distribution.

---

# Mature native acceptance / pilot audit — 17 October 2026

[Scope](PF6_MATURE_PILOT_FORMULATION.md), [measured report](PF6_MATURE_PILOT_REPORT.md).
Entry is `a8707307008401ac9cff702a369e2e2a611fa0f7` plus the preserved incoming
worktree (12597 entry file hashes). Earlier audits, failed evidence and
`strict_reference_match_passed=false` remain below and in their original files.

## Equations and initialization

This extends the **fixed-chi**, fixed-geometry native mature experiment; it does
not extend its equations to evolving/transported localization. Retain
`T_new=beta*T_old+2*kappa*D3(symgrad u_new)`,
`Z_new=beta*Z_old+kappa*V_new`, `tau=T-2*chi*Z*S`, with
`beta=exp(-G*dt/eta)`, `kappa=-eta*expm1(-G*dt/eta)`. Z=G*s_mem has no cohesive
energy/resistance. The native source, full 3D plane-strain convention, RSF aging,
physical initial state, normalization/projection/derivatives, active association
selector and paired/constant endpoint completion are unchanged.

`normalization.cc::project_surface_chemical_compositions()` previously assigned
zero/background labels throughout homogeneous mesh-data mode. That was harmless
for the local uniform-a fixture but would erase the intended BP3 transition.
The mature branch now evaluates the **configured InitialComposition manager**
directly at fixed surface nodes during fresh initialization and retains the
manager-owned values on subsequent steps/restart. Bulk G/eta/density/thermal and
profile coefficients remain configured homogeneous values; surface a may vary.
No particle projection, new material parameter, alternative friction law or
prestress compensation is introduced. Cohesive homogeneous initialization keeps
its previous branch. The full ordinary BP3 initializer/independent regularized
law checks both actual surface nodes and interpolated interior values.

## Source and ownership changes

| Source | Change / retained contract |
|---|---|
| `phase_field_fault.cc` | Existing accepted-DG flag admits fixed frozen homogeneous mature MPI/restart. FE and particle conversion restrictions remain. The D3 diagnostic-sample convention can resume only through accepted DG; particle-authoritative D3 restart remains guarded. All 34 material defaults match entry. |
| `fe_strain_internal.h` | Initialize the mature branch's unused cohesive m to zero, making checkpoint metadata deterministic; the actual phase-handler degradation law is unchanged. |
| `reconstructed_fault_stokes.cc`, `core.cc` | Reuse the existing deferred outer-publication callback for mature DG. A converged mechanical trial returns staged u/V with manager trials rolled back. The actual timestep decision drops it on repeat or performs the validated native commit once on acceptance. No Newton/tangent, residual scaling, FGMRES or physical phase ordering changes. |
| `history.cc` | Permit the mature native commit owner under the accepted-DG flag. The separate prescribed PF5 split-commit API explicitly rejects mature mode. Every source contribution still reads the same accepted T/Z/Theta and is applied once. |
| `dg_accepted.cc` | Version 4 reuses version-3 global owned CellIds, physical corners/phase anchors and all 27 coefficients/cell; retains Z/I, geometry/material metadata, accepted counter and separate native next-step clock. Mature restore requires exactly zero cohesive traction, independently of nonzero Z. Versions 1–3 retain their old routing. The prescribed transport API rejects mature mode. |
| `phase_field.cc` | Same-mode mature restart may retain fully prescribed phase. Owned candidate versus ghosted-anchor vector equality gave a false MPI failure: the guarded reproduction measured zero changed owned values / zero error. The replacement compares **every owned physical value exactly**, then MPI-mins the result. No tolerance/floor was added. |
| `dg_history.cc` | Initialization log names the accepted lifecycle rather than implying the PF5 prescribed-transport path. Projection/reconstruction and source-roundoff arithmetic are unchanged. |
| `coupled_checks.cc` | Default-off lifecycle diagnostics admit restart/repeated trials, prepare transient boundary associations before diagnostic point reads, deep-compare all DG coefficients, and compare all property bytes (NaN registry sentinels included). Preserve step-zero slip semantics. Existing one-owner guard injection is supplemented by actual outer rejection and native nonlinear failure/cutback controls. Save diagnostic progress through native postprocessor checkpoints. |

Native checkpoints follow `advance_time`: accepted constitutive step n coexists
with next native clock n+1. Restore does not advance T/Z/state. Existing frozen
normalization restart validation compares a cold solve with stored I, then retains
the checkpointed I; no new normalization allowance or derivative rule is added.
The serial phase-reference import remains fresh-only, fixed-grid and MPI1.
No AMR, moving geometry, phase evolution, mature transport or conversion was enabled.

The independent publication comparison remains mandatory on every rank before
terminal writes. `fe_strain.cc`, including its validator and operand-derived
allowance, is byte unchanged from entry. No failed comparison was removed or
empirically rescaled. `dg_limiter.cc`, `dg_transport.cc`, particle manager/domain
sources and native fault-manager source also match entry. Old CPDI/domain and
quadrature failures remain evidence for their original branches.

## Remaining particle reads in the selected path

- Native initial-composition/H properties are still created for the fixed points;
  they include inert initial theta/strengthening as well as H. Prescribed mesh
  phase, configured bulk coefficients and surface initialization do not read
  these point fields. Generic initialization/checkpoint compatibility still
  allocates them.
- `fe_strain` restrictions read IDs, coordinates, local ownership and population
  counts to enforce stationary sampling; stress evaluation uses matching native
  CellIds. Mesh/fault data provide physical geometric support, not Voronoi domains.
- Native history candidate/publication code reads old point stress and H for the
  independent Maxwell consistency calculation and frozen-H bookkeeping. Accepted
  mechanical assembly and fault weak loads read authoritative DG T/Z and native
  FE forcing/localization. Point stress is overwritten only with accepted derived
  samples, never transferred back to mechanics.
- Benchmark/native audit and checkpoint writers read point properties, IDs and
  next-ID bookkeeping. The existing native particle checkpoint owner remains.
  Domains, CPDI and face-data generation are disabled; no selected material or
  mechanical branch calls a particle-domain accessor.

Effective motion is disabled by `BP3 stationary samples`; the RK2 freeze variable
is unset. Full point audits compare IDs/coordinates before/after, including
rejection and restart: motion, births and deletions are zero. Fresh MPI generators
may number identical physical points differently; restart retains IDs exactly.
A subsequent removal should first implement deterministic mesh-cell validation
points using independent native FE gradients and preserve negative consistency
controls. Only then remove diagnostic point history/initial-property/checkpoint
requirements. Keep the separate legacy CPDI backend and its known domain defects.

---

# Mature resolution-separation audit — 17 October 2026

[Formulation](PF6_MATURE_RESOLUTION_FORMULATION.md),
[measurements](PF6_MATURE_RESOLUTION_REPORT.md). All earlier audits/failures remain.

- `phase_field_fault.h` adds an empty-by-default immutable benchmark phase
  reference (physical evaluator plus explicit exterior lattice spacing), with
  no constitutive history or imported I. `boundary_completion.cc` restricts it
  to fresh serial fixed-mesh frozen mature mechanics, verifies every phase
  constraint/physical vertex and the refined boundary trace, and retains the
  coarse exterior Q1 spacing. The original no-reference admission, integrand,
  consistent normalization projection/derivatives and paired completion remain.
- `fe_strain.cc` applies an explicit existing FE quadrature value above three
  to mature particle comparisons as well. Default three retains the native
  particle rule exactly; independent history is unchanged. Description only
  in `phase_field_fault.cc`; no new material parameter or changed default.
  All actual points/weights are compared; selected controls use 36 bulk points.
- Maintained BP3 `phase_reference.{h,cc}` reads an immutable admissible, aligned,
  nonoverlapping dyadic Q1 snapshot. `bp3.cc` prescribes its complete polynomial
  on fine vertices and attaches its exterior data before preparation. Default
  empty input retains the original physical seed. Configuration identity records
  an explicitly selected reference; CMake adds only this maintained source.
  Boundary loading, profile law, physical initial state and geometry are unchanged.
- `coupled_checks.cc` exports actual native surface weak loads/mass, optional
  common physical-point stress and DG support nodes. It reads accepted DG stress
  and immutable incoming particle transfer plus native accepted forcing, with
  explicit cell trace ownership; no second Maxwell update. Original exports
  remain byte-identical in the reproduced CC/FF controls.
- Drivers pin the same core/plugins and 1031 native source hashes, actual inputs,
  logs/cost/output hashes. The reduction uses the full 3D tensor metric and
  Cauchy pressure, common physical-profile weights, fixed endpoint windows and
  neutral backend differences. The bulk-work mass is measured, not assumed
  geometric: subtract each native M*1 background and separately evaluate a
  shared geometric surface functional. No native mass/operator is replaced.
- Independent all-vertex/nonnodal Q1 checks and saved all-nine-mode DG evaluation
  pass. Native fixed phase/I/point population, state/Maxwell/work and once-only
  publication checks pass; old CC/FF fields and default particle fields reproduce.
  No acceptance allowance or failing validator was removed or relaxed.

DG constitutive/source/transport/limiter/accepted-transfer/checkpoint kernels,
normalization integration/projection/derivatives, phase-energy, particle manager
and particle domains are byte-identical to task entry. Mature restart/MPI and
phase evolution remain unsupported or unqualified. The selected physical seed
representation and mechanics still show large differences; temporal-load failure,
`strict_reference_match_passed=false` and all prior evidence remain unchanged.

---

# Stationary mature DG extension audit — 16 October 2026

[Declared scope](PF6_MATURE_DG_FORMULATION.md), [measured report](PF6_MATURE_DG_REPORT.md).
The earlier audits and all failed evidence remain below.

- `phase_field_fault.cc` extends the existing default-off homogeneous mesh-data
  selection to fresh serial frozen mature independent DG/FE histories. Require
  static fields, configured homogeneous materials, no domains/CPDI, fixed geometry,
  fixed mesh, and no restart/conversion. Mature DG does not enable the separate
  PF5 prescribed-transport/checkpoint flag. Existing cohesive selections stay intact.
- `fe_strain.cc` reuses the mature history owner, phase/I/geometry and auxiliary Z,
  adds owned Cartesian source-cell lookup and initializes all nine DGQ2 modes.
  DG owns T; its displacement-history velocity block remains zero. The mature
  DG publication count advances through the existing native commit owner.
- `dg_history.cc` reconstructs mature stress with the actual oblique segment,
  shear sense and native source map, including constant endpoint continuation:
  tau=T-2(h/I) Z S. General native localization replaces only the old horizontal
  assumption for this selected mode. The cohesive branch is retained. The
  Gauss3 native-source projection and its operand-derived allowance are unchanged.
- `constitutive.cc`/`history.cc` generalize the existing mature FE zero-cohesion
  path to independent history. C=0 and Q_geo=0; Z evolves independently by
  beta*Z+kappa*V and is never reconstructed from I*C. Native stress, RSF, loading,
  normalization and rate/spin conventions are preserved. Complete validation
  still precedes publication; the stress validator and its allowance are unchanged.
- The domain-free mature phase route in `simulator/phase_field.cc` requires every
  physical phase DoF prescribed and admissible. At later callbacks it requires
  exact equality with the accepted physical phase. It performs no CPDI projection
  or phase solve. The first positive-step gate caught and repaired an erroneous
  initialization-only callback guard; its original failure is retained.
- `manager.cc` reuses mature support from bulk-cell margins and native stationary
  profile widths, plus fixed prescribed geometry and paired boundary completion.
  General normalization, its projection and derivatives are unchanged.
- The maintained BP3 callback provides the existing positive nodal Theta function
  before domain-free preparation. Missing state rejects rather than falling back
  to particle projection. `initial_composition.cc` permits diagnostic static-field
  initialization only for this explicit homogeneous mature selection. Legacy
  particle field requirements are retained.
- Benchmark work/trajectory observers recognize both independent backends and
  the selected common D3 particle convention. They export accepted reconstructed
  stress, complete DG coefficients, frozen phase, raw/filtered traction and native
  recurrence/publication checks. For particle control the stress export evaluates
  the actual accepted native operator from its frozen incoming transferred FE
  fields, not a second update of newly published particle samples.
- The stage-only operator observer compares parallel native-FE and DG histories,
  actual surface weak loads, endpoint continuation and normalization. Its old
  FE-only test allowance omitted DG source-projection roundoff; the diagnostic
  now adds the existing derived DG allowance. Production tolerances were not
  multiplied or otherwise changed. Original failed run is retained.

No transport, limiter, phase-energy, checkpoint or mesh-transfer algorithm is
changed in this task. Remaining particle dependencies and measured limits are
reported separately; this is not propagation, mature restart or PF5 completion.

---

# Homogeneous mesh-data replacement audit — 16 October 2026

[Measured results](PF6_DOMAIN_FREE_REPORT.md), [rules](PF6_DOMAIN_FREE_FORMULATION.md).
All incoming audit text and failed evidence are retained below.

- `phase_field_fault.{h,cc}` adds default-off `Use homogeneous mesh data`, restricted
  to accepted DG consistent endpoint, prescribed physical seed, static fields,
  homogeneous configured coefficients and fixed prescribed geometry. Reject
  domains/CPDI/face-data requests. Bulk material uses the configured background
  fractions; viscosity returns the validated constant without material projection.
- `simulator/phase_field.cc` permits domain-free diagnostic points on this selected
  configuration, uses the cell FE graph, and prohibits CPDI assembly fallback.
  Existing supplied-profile initialization bypasses H-driven phase relaxation.
  `fe_strain.cc` changes only its initialization message; its publication validator
  is identical to the accepted preceding implementation.
- `normalization.cc::project_surface_chemical_compositions` sets background labels
  directly for this homogeneous selection. Normal integration, consistent mass
  projection and every normalization derivative/pullback are unchanged. Independent
  Gauss24/32 I/DI/D²I and FD action checks pass. `fe_normalization.cc` and
  `fe_phase_energy.cc` are byte-identical to task entry.
- `reconstructed_fault/manager.cc` obtains selected support from bulk vertices and
  fixed full-normal fault geometry. H initialization retains native analytic
  diagnostic values using configured background material; it no longer reads
  particle chemistry and does not initialize the selected physical phase.
- `manager_particle_projection.cc` caches geometry-only point associations with
  NaN volume and empty quadrature; cache validation uses ID/position/fault metadata.
  Particle projection-factor construction explicitly rejects this mode. Matching
  CellId traces still come from the native particle owner in `history.cc`.
- `history.cc` uses the configured background material for the independent native
  Maxwell check, skips chemical particle mappings, and records/rejects changes of
  mode through the existing surface property registry. No physical law, source
  allowance, stress validator, publication ordering or time-level rule is changed.
- `particle/property/maxwell_stress.cc` initializes selected diagnostic samples to
  the same unloaded zero as independent DG; `crack_driving_force.cc` reads the
  configured homogeneous threshold. Static FE fields need no particle mapping.
  The selected fixture removes its unused particle chemical property.
- `particle_domain.cc`, `particle/manager.cc`, DG projection/transport/limiter,
  accepted-history checkpoint packing and native transfer operators are unchanged
  in this task. Legacy domain coverage still fails on the fresh deep negative
  control; all 21 frozen FE/CPDI CSVs reproduce exactly.

Runtime private-pointer checks establish no domain object at every exported selected
point state. Since construction is solely request-driven and access throws without
an object, no successful domain read occurs; the evidence does not invent access
counters. Every point ID/coordinate remains fixed. The 1/2/4-rank full cases retain
the original source/averaged-trace negative check, outer rejection/retry and single
publication. Same-count and continued restart match exactly; serial-to-four differs
only by native functional summation roundoff, with the retained RNG restriction.

---

# Gauss12/local-transfer followup audit — 16 October 2026

[Measured report](PF6_LOCAL_REPORT.md) and [formulation/amendments](PF6_LOCAL_FORMULATION.md).
Preserve all incoming audit text and evidence below.

- Physical Gauss12 is selected in fixture inputs through the existing parameter;
  the production default is unchanged. Analytic ray normalization, its surface
  mass projection, DG Gauss3 source/mass, constitutive laws and limiter are unchanged.
- `fe_strain.cc` accepts the selected affine Cartesian local geometry and convex
  homogeneous Q1 hanging constraints. Local physical rules conservatively split
  fixed-surface knots; every physical consumer uses the same rule, without MPI
  collectives inside cell assembly. Uniform rules retain their existing layout.
- `dg_transfer.cc` checks complete flagged coarsening families and the same
  256-epsilon physical phase/obstacle representability condition. Native distributed
  SolutionTransfer retains all nine modes of each tensor component. Source/point
  caches rebuild; physical clocks, Z/I, rates and publication counts stay fixed.
- `core.cc` finalizes selected refinement flags before preparing histories or
  particle data, preventing stale ownership after canceled/no-op flags. Phase
  rejection remains before userdata preparation and mutation; legacy hook order stays.
  Restored accepted DG endpoints check the existing termination manager before the
  do-while loop, avoiding an extra physical iteration after a completed checkpoint.
- `particle/manager.cc` rebuilds selected endpoint domains after unpacking, bounds
  handling and final ghost exchange. Fixed coordinates do not imply stable indices.
- `particle/particle_domain.cc` retains nonperiodic 2D closed-patch candidates at
  artificial upper container walls using a derived floating-point bound and outward
  rounding, keeps physical walls/particle coordinates fixed, and verifies insertion
  count. Existing positive-volume/coverage guards remain. Failure context now names
  particle/index/CellId/box/candidate count. Genuine deep-neighborhood insufficiency
  remains guarded; no new neighbors or fabricated volumes are supplied.
- `reconstructed_fault_endpoint_phase.cc` explicitly rejects hanging-node phase
  evolution. The current minimizer's independent nodal variations are not silently
  reused as constrained variations. Frozen local evaluation/restart is the qualified
  subset; no phase-first composition or altered physical KKT/energy acceptance.

All 13 final cases use one source/build/observer set. Native 1/2/4-rank transfers,
restore-only and continued restart, independent geometry/quadrature references,
and the frozen FE control pass; all 21 control CSVs are byte-identical. Cross-rank RNG replay remains
unavailable with population changes disabled. The original strict-reference false
flag and Gauss6 failure remain unchanged; numerical remapping does not resolve them.

---

# Fixed-reference transfer and matching-cell ownership audit

16 October 2026; base `a8707307008401ac9cff702a369e2e2a611fa0f7` plus preserved
incoming work. [Rules](PF6_TRANSFER_FORMULATION.md), [measured report](PF6_TRANSFER_REPORT.md),
[remaining particle reads and replacements](PF6_PARTICLE_READ_AUDIT.md).

- `dg_transfer.cc` adds accepted-state-only uniform Cartesian remapping with
  native distributed SolutionTransfer. Every scalar DG component retains all
  nine coefficients. Coarsening is L2 projection; phase/lower-obstacle coarsening
  is admitted only when the parent Q1 interpolant reproduces the child field.
  The collective precheck runs before particle/userdata preparation and mesh
  mutation. New history clears source-roundoff and derived point caches.
- `simulator/core.cc` routes those pre/post operations through native refinement,
  transfers the older native solution buffer on this DG path, rebuilds constraints
  before accepted-cache construction, and rejects pending endpoints before remap.
  The existing physical outer acceptance/publication mechanism is retained.
- `history.cc` selects the particle CellId for native DG consistency gradients,
  checks association ID/order/position, and propagates local sampling failures
  before later collective evaluations. Continuous temperature/phase sampling and
  the legacy gradient path retain their existing conventions. No unused averaged
  gradient is computed for DG. The historical failure CSV key `remote_gradient`
  remains replay-compatible; new `matching_cell_gradient_trace` identifies the
  changed selected trace convention. Original failure CSVs/errors are untouched.
  Publication allowance and validator are unchanged; statistics now reduce
  error/bound ratios and old-allowance failure counts over ranks.
- `dg_accepted.cc` adds version-3 endpoint payload packing by uniquely owned
  CellId with physical corners/phase, complete coefficients and clocks. Restore
  checks global ownership and physical phase/geometry collectively, creates owned
  vectors, then refreshes ghosts. Old serial v1/v2 branches remain readable;
  new backend conversion is not admitted. Native particle/RNG/manager persistence
  stays in its existing owner.
- `simulator/phase_field.cc` reduces ordinary FE fracture's owned-cell energy,
  gradient, Hessian and arithmetic scales for the selected MPI endpoint. The
  residual arithmetic norm uses the sum of squared uniquely owned global rows.
  The cell integrand, quadrature, phase variations and native KKT/energy tests
  are unchanged. `phase_field_fault.cc` admits MPI only through the selected
  accepted DG consistent-endpoint mode; other DG MPI/transport/automatic-AMR
  restrictions remain.

Measured uniform serial/1/2/4 transfers and restart pass independent common-rule
checks. The unchanged native Gauss6 physical quadrature misses the declared 1e-9
functional screen on the coarse evolved fixture; this is reported separately as
`native_rule_vs_high_rule_screen_passed=false`. No quadrature or physical
acceptance rule is changed. See the measured report and portable replay.

Native laws remain T_e=beta*T_n+2*kappa*D3(symgrad u), Z_e=beta*Z_n+kappa*V,
tau=T-2*chi*Z*S and q=Z/I, with the full 3D plane-strain metric and the audited
rate convention. Transfer applies none of these updates. `constitutive.cc`,
`dg_history.cc`'s projection/recurrence, transport, limiter, physical inputs and
all preceding evidence are preserved. The source audit below is historical
provenance, including every original strict error and false flag.

---

# PF6a publication-consistency repair — 16 October 2026

[Detailed derivation and measured evidence](PF6A_PUBLICATION_REPORT.md).
`history.cc` captures both prepublication paths on failure: actual native
remote-point velocity gradient/Maxwell operands, full local FE coefficients,
DG incoming/trial coefficients, phase/I/Z, geometry/point coordinates and error
budget. The reproduced original guard fails by 1.24971. Independent 80-digit
polynomial differentiation identifies native point-gradient cancellation; state,
Maxwell recurrence and localization agree. The original stress-evaluation budget
omits differentiation/source-projection conditioning.

`dg_history.cc` preserves its Gauss3 mass/source projection and recurrence, adding
scratch forward-error data from velocity coefficients and widths, explicit Q2
value/derivative operation bounds, and the computed inverse defect against the
analytic tensor-product Q2 mass. `dg_source_publication_allowance` combines both
evaluation paths in the embedded 3D plane-strain metric. `fe_strain.cc` requests
this data only for a publication candidate; ordinary Newton trial coefficients
are bit-identical without it. Stable positive error-factor products avoid
near-unit subtraction. `history.cc` adds this derived source budget to the old
history/localization budget and retains the throwing comparison before terminal
writes. Default FE/particle paths retain the old allowance; no physical law,
constitutive rate, limiter, source quadrature or production selection changes.
Scratch bounds are never read as accepted constitutive history or serialized.

`pf6a_checks.cc` adds default-off postpublication batch stress output, using the
same complete DG coefficients, FE phase and surface Z/I on the unchanged rule.
Full pointwise comparisons at startup/onset/end and byte-equal accepted states
verify this diagnostic-only cost correction. The production particle-position
guard still uses the original independent point paths and rejects inconsistent
incoming/staged histories. Actual outer rejection, rollback/RNG, cutback, retry
and once-only publication are verified. Fresh full run: 72 accepted steps,
879.915 s under the 900 s cap; original failed/capped artifacts remain intact.

The original selected 1D reference is retained. Final native/reference phase
profiles choose opposite mirror branches; primary unregistered strict errors
remain failed. Native compares departure candidates with an allowance relative
to the current best; the independent reference's between-candidate comparison
is strict. Both pass their original KKT and tested translation-curvature checks;
energies differ below the native arithmetic screen. No algorithm is retuned or
reference silently replaced. This limits trajectory equivalence, not the
verified Maxwell/source/publication identities.

Particle dependencies remain startup H/CPDI/domains, material projection,
association/support, derived samples/publication validation and persistence.
Independent DG stress still bypasses particle-stress feedback. The proof assumes
unique interior samples on fixed affine Cartesian cells. `point_gradients(avg)`
may average adjacent cells at faces, whereas DG evaluation selects a CellId
trace; AMR/MPI must establish matching evaluation/ownership separately. Current
unsupported-path guards and prior general-PF5/propagation limitations remain.

---

# Matched evolving spatial-reference addendum — 16 October 2026

Scope/revision: `a8707307008401ac9cff702a369e2e2a611fa0f7` plus preserved incoming
work; serial fixed geometry, no transport. [Report](PF6A_REFERENCE_REPORT.md)
and [formulation](PF6A_REFERENCE_FORMULATION.md) retain unlimited DGQ2 and all
physical/default/acceptance settings. **No production source change.**

- `pf6a_spatial_reference.py` reuses retained `Reference.direct/refine`, the
  existing `matched_initialization` continuum BVP, material constants and the
  consistent-endpoint phase partial. It algebraically eliminates the retained
  1D Q2 mechanical equations using P1 projection; zero-start compatible history
  makes its mean authoritative in this reduction. It does not change native
  DG ownership, drop native Q2 modes, or call the historical split time driver.
  Independent full Q2 matrix and directional derivative checks pass.
- Native `initialize_fe_shared_phase` interpolates the prescribed seed file at
  native nodes. Source-origin audit proves exact table/BVP nodal identity for
  the selected 512 mesh. Main finer phase references evaluate that same
  documented continuum definition at additional nodes. Literal table-interpolant
  refinement is preserved separately: identical-field representation at 1024
  does not imply unchanged obstacle-space stability. No seed fitting, loaded
  import or mesh-dependent H relaxation is used as the primary reference.
- `run_pf6a_reference.py` reuses the existing PF6a observer and exact source-
  pinned core, one fresh 512-cell trajectory with fixed quadrature 12 and dt.
  The serial DG guard in `phase_field_fault.cc` remains; permission for eight
  processes does not supply an MPI implementation. No production API/hook added.
- The actual first positive native endpoint fails in
  `history.cc::commit_reconstructed_fault_mechanical_history` at the particle-
  position consistency test `error <= 128 eps max(1,evaluation_scale)`. Candidate
  DG stress is compared with independently computed Maxwell stress; both read
  independent accepted history, not particle stress feedback. Terminal
  `publish_history_candidates` / accepted FE/DG swaps follow this check and are
  not reached. Initial exports have zero publications; rollback byte equality
  after this process-level failure was not measured. The error text alone does
  not establish the numerical cause or justify accepting the candidate.
- Existing `prepare_fe_strain_commit`, `candidate_fe_strain_stress`, source
  projection and sampling retain their audited equations. The exception does
  not export failing components/error/bound/position; capturing those is the
  concrete next diagnosis. No constitutive compensation or tolerance change.

Particle dependencies include this publication validation as well as startup
H/CPDI/domains, material projection, association/support, derived samples and
persistence. No particle-authoritative comparison, coupled transport, AMR/MPI,
surface transfer, moving geometry, birth or propagation is qualified. Main
1D targets pass at 512; the native confirmation fails and is preserved as such.

---

The previous audits retain their original scope and evidence.

# PF6a spatial diagnostic addendum — 16 October 2026

Revision `a8707307008401ac9cff702a369e2e2a611fa0f7` plus preserved incoming
PF6a/WENO/manual work. [Spatial formulation](PF6A_SPATIAL_FORMULATION.md) and
[measured report](PF6A_SPATIAL_REPORT.md) separate identical-field evaluation,
frozen native mechanical resolution, compatible-history storage and fracture
quadrature. No production implementation or constitutive/default change.

- `pf6a_spatial_checks.cc` is a separate benchmark DSO using the existing test
  friendship and endpoint diagnostic clock/trial APIs. Frozen trials use
  immutable imported physical phase/T/Z, actually solve native mechanics,
  export the explicit endpoint and restore clocks/state/manager fields. Gates
  assert unchanged histories and zero publication. Native weak residuals,
  actual reconstructed stress, complete cell moments and traces are exported.
- `prepare_pf6a_spatial.py` uses all saved Q2 velocities and the native audited
  Maxwell coefficients to recover the compatible accepted FE displacement.
  Nine unisolvent points check all complete tensor modes before comparing DG
  and FE on common fields/ordinary-FE energy. This is a fixture compatibility
  construction, not a general history or checkpoint conversion.
- The same observer calls the actual retained CPDI fracture assembler on FE
  separately. Native FE+CPDI equations remain distinct from the common FE
  diagnostic. No unmatched particle-authoritative stress or transferred
  particle values are substituted as a reference.
- Independent `localization_1d.py::Reference` and `normal_resolution.py` are
  reused with imported physical fields and their mechanical equations; no
  historical initialization/evolution is run. Independent local moments verify
  tau = mean(tau) + (1-P)(beta Tn) - Ze(1-P)chi, including the mean/load changes.
  The full plane-strain metric, native normalization and rate are retained.
- The conditional event imports a common physical entry once at native step 0,
  then observes the existing actual coupled outer loop. A benchmark-only
  boundary plugin supplies the same linear ramp's exact interval mean at each
  selected dt. It does not integrate phase or histories. Initial enriched-space
  KKT defects and omitted matching accumulated-slip/H imports are explicit;
  no restart/AMR or general RSF transfer qualification follows from this setup.
- The source/transport rules, consistent phase partial/reduced tangent,
  irreversibility, accepted-history ownership and final native publication
  remain owned by the prior PF6a implementation below. New independent checks
  differentiate all Q2 velocity modes and verify T/Z advancement once per step.

All normal meshes are fixed and serial; only 128 is a new normal level after
32/64 diagnosis. Remaining particle dependencies and unsupported stages are
unchanged. The evidence records exact native/observer hashes, inputs, common
physical points, costs, mathematical checks and preservation of incoming work.

---

The prior audits retain their original implementation/evidence scope.

# PF6a source addendum — 16 October 2026

Selected source root/branch/HEAD: `aspect-pf-rsf-particle-free`,
`pf-rsf-particle-free`, `a8707307008401ac9cff702a369e2e2a611fa0f7` plus
preserved incoming work. [Formulation](PF6A_FORMULATION.md) and
[measured report](PF6A_REPORT.md) describe the bounded serial fixed-grid,
fixed-reference-surface, no-advection coupled experiment. The existing
consistent-endpoint/reduced-Newton solver governs; the derivation's split
proposal is not adopted. PF3 FE fracture replaces CPDI only in this selected
DG endpoint functional. No constitutive, limiter or production-default change.

| Source owner | Audited implementation / contract |
|---|---|
| `fe_phase_energy.cc::make_fe_endpoint_trial` | Reconstruct T_e from accepted DGQ2 T_n and native exact Maxwell/Q2 source, Z_e from accepted Z_n and solved V; explicit phase carried in the existing shared vector. No trial updates its parent |
| `evaluate_fe_endpoint_energy/action` | Fixed-endpoint phase partial and first/second consistent I/chi pullbacks; explicit DG tensor supplied, full plane-strain metric and native rate preserved |
| `reconstructed_fault_stokes.cc::make_reconstructed_fault_endpoint_tangent` | Preserve the DG endpoint while solving native mechanical response; dT_e and dZ_e enter the reduced tangent, no frozen-physical-stress substitution |
| `phase_field.cc` | Ordinary FE fracture-only gradient/Hessian used with explicit endpoint elastic/cohesive energy; no double accepted-history elastic term |
| `fe_strain.cc`, `history.cc` | Selected phase/I evolution and new phase anchor allowed; fixed mesh/surface/material/particle guards retained. Existing candidate validation/sampling/terminal swaps still own publication |
| `reconstructed_fault_endpoint_phase.cc`, `core.cc` | DG candidate staged through native timestep-manager reaction. Accept once before postprocessing; discard/restore trial bookkeeping on exception or repeat. Old FE publication branch unchanged |
| `dg_accepted.cc` | Reuse native checkpoint marker/ownership. Version 2 records accepted step and native next-step clock separately, checks canonical accepted vectors/anchor/publications, and restores without update. PF5 manual v1 remains unchanged; conversion rejected |
| `dg_transport.cc`, `prepare_dg_accepted_step` | Explicitly reject this coupled mode, preventing the prescribed PF5 transport composition from being mistaken for PF6a |
| `pf6a_checks.cc` | Observe real native advancement and expose immutable diagnostics; actual solve exception, actual manager cutback, automatic filesystem restart and canonical full-manager comparisons. It does not integrate coupled physics |

The late audit found publication inside `solve_timestep` preceded a second,
real outer rejection point in `TimeStepping::Manager::update`. PF6a now delays
DG publication through that decision. A native `repeat on cutback` fixture
verifies unchanged anchor bytes/phase and identical accepted-schedule results.
A final error-path audit also found that a failed deferred context check needed
entry-bookkeeping restoration. The narrow catch/restore correction is verified
by a stale-context regression that fails on the pinned previous core, followed
by unchanged accepted coarse/cutback/restart exports.
Another correction distinguishes automatic post-`advance_time` checkpoints from
PF5's observer-prescribed endpoint clock. Both old failures are retained.

Native source projection uses Gauss3; existing split mechanical six-normal-point
quadrature is compared to twelve and independent 24/32-point integration.
Transport quadrature/operator and all WENO code/settings are unchanged. Fresh
final-source dt/dt2 controls reproduce all primary physical exports; historical
build groups and observer changes are identified separately in evidence.

Particle dependencies remain startup H/CPDI/domain initialization, surface
material mixture projection, support/associations, fixed-particle guards,
derived stress/H samples and particle/RNG persistence. Particle stress is never
incoming DG tensor or mechanical/fault-load feedback. Physical phase p_n, not H,
is the accepted obstacle/driver convention in the endpoint solve. No bulk or
surface advection, AMR/MPI, moving geometry, surface transfer, birth or general
propagation is enabled or qualified. Large normal spatial stress sensitivity
remains despite verified operators and native KKT; see the report's next task.

---

The original audits below retain their historical scope and findings.

# PF0 source audit — 15 October 2026

## Pre-PF7 candidate-construction audit — 16 October 2026

The newly selected comparison is documented in
[WENO_CLS_FORMULATION.md](WENO_CLS_FORMULATION.md) and
[WENO_CLS_REPORT.md](WENO_CLS_REPORT.md). Earlier phase reports and original
evidence remain historical. `dg_limiter.cc` originally implements complete
neighbor-polynomial extrapolation plus a target-cell mean shift; this is the
compact whole-polynomial construction, not the 2016 constrained-neighbor fit.
The new private `make_dg_limiter(..., constrained_candidates)` option builds
neighbor-own-cell Q2 mass/patch-average rows and constrained correction maps.
The common full-tensor useful set and all nine coefficients are retained.
Uniform translation-equivalent maps are cached within construction; this changes
no physical model or transport equation.

Detection still precedes fitting and uses unmodified polynomials. Thresholds,
patch RMS normalization, epsilon/zero-beta branch, derivative smoothness,
nonlinear weight formula, Gauss3 mass/reconstruction and Gauss4 transport remain
unchanged. Physical means are now optionally measured directly in Pa, not only
as the previous normalized reconstruction residual. A post-stage observational
callback runs after the complete separate output is available; benchmark code
checks interface flux changes against the existing material matrix and traces
SSPRK3 mean balances. The callback has no accepted-state access or publication.

The `stabilization_checks` observer gains candidate/case/path/trace selection
with its previous defaults. `run_candidate_case.py` creates bounded unique
fixtures; `assess_candidates.py` independently solves Legendre-coordinate KKT
systems and evaluates errors, widths, norm changes, mean/flux traces and held
limiting. This is a controlled Q2/tensor adaptation motivated by the primary
2016 manuscript, not a full published-method implementation. No detector tuning,
stress envelope, chi limiting, history backend, restart or constitutive change
is introduced. Existing serial/fixed-grid and AMR/MPI/periodic/moving/coupled
guards remain. Particle dependencies are unchanged from PF5.

All 30 final same-build transport records pass execution/nonpublication and the
independent local KKT checks pass. The constrained construction improves sharp
pulse error relative to mean shifting, but all WENO temporal screens fail and
x-only scalar fields acquire transverse variation. The cached/uncached strict
final-field equivalence failure is retained. This is diagnostic evidence, not
adoption or authorization to begin PF7; unlimited DGQ2 remains the baseline.

---

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
