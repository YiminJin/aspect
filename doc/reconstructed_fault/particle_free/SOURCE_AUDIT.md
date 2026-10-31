# Cleanup and research handoff — 30 October 2026

[Cleanup report](CLEANUP_REPORT.md) · [Research handoff](RESEARCH_HANDOFF.md).
Generated artifacts were reduced; unique source, original BP3 data, native runtimes,
current RSF step748 and complete server package are preserved. No scientific
evolution or conclusion changed. The next research-review work is BP3 inspection
and variable/timestep/energy maps, then the controlled-crust/propagation paper
sequence. No realistic subduction application or new run is selected.
The RSF episode remains incomplete and all recorded accuracy failures remain.
Some historical bulk-animation frames were pruned in error; the cleanup report
records the missing frames, exact recoveries and limits on re-rendering. Current
RSF surface/event data and original BP3 server results are unaffected.

---

# RSF episode estimator repair and server handoff (2026-10-30)

`cohesive_timestep.h` uses the complete assembled residual, unchanged certificate
resolution and current admissible symmetric tangent bound to estimate linearized
displacement resolution. Rate intervals are diagnostics; the new selected mode
controls continuous jump/aging/damage defects and numerical history resolution.
It never tests damage against an exemption threshold. `weak_tolerance` names the
existing1e-8 Pa value without changing it. `cohesive_advance.cc` computes diagnostics
from immutable parent and candidate; only the ordinary accepted owner persists them.
`time_stepping/cohesive_rsf.cc` preserves legacy default behavior and all native
acceptance checks. See `RSF_EPISODE_ESTIMATOR.md` before interpreting estimates as
nonlinear or spatial error bounds. The independent unit tests, real failed-state retry, rejection/restart and
MPI ownership checks pass. The actual four-rank continuation advances normally;
its native wall stop preserves a complete checkpoint for server continuation.
The episode itself remains incomplete. See `RSF_EPISODE_REPORT.md` for scope,
measured errors/cost and retained failures. Packaging is read-only: no native
history conversion or checkpoint reset. Server startup remains prepared,
not remotely qualified.

---

# RSF instability / adaptive native stepping audit (2026-10-29)

Current work: `COHESIVE_RSF_INSTABILITY_REPORT.md`. Physical law unchanged;
new specimen parameters explicitly supplied in separate inputs.

- `cohesive_advance.cc`: immutable-parent Gauss maxima of |dj|/Dc,
  |log(theta+/theta)|, |log((|V+|+V0)/(|Vold|+V0))| and damage increment.
  First physical step omits speed-change indicator because initialization has
  no accepted physical rate; slip/state/damage checks still apply.
- `time_stepping/cohesive_rsf.cc`: optional native plugin reads the pending
  metrics, proposes safety*dt/max(normalized indicators), capped by growth;
  rejects ratio>1 through the actual native repeat lifecycle. Rejection does
  not publish a trial. Read-only SimulatorAccess accessor exposes the owner.
  The indicator's V0 is not a change to the friction law. No persistent plugin
  clock/history is needed: controller state derives from checkpointed owner.
- `cohesive_evolution.h` / simulator adapter: typed recoverable exceptions only
  for semismooth iteration/line-search failure and negative endpoint symmetric
  tangent. Native failure strategy cuts timestep; continue-after-failure is
  prohibited. All contact/source/physical-work/geometry/history guards remain
  fatal. A positive step tangent does not certify continuous-time RSF stability.
- `cohesive_mechanics.cc`: opt-in small K/Q/N export, with actual physical
  quadrature weights, for independent current-pressure/state/radiation spectral
  analysis. No dense global bulk operator or response-span history replaces
  the general native owner.
- Slow loading exposed cancellation in separately summed O(174 J/m) storages.
  Use (tau+ + tau):(tau+ − tau)/(4G) for the increment, with extended local sums
  and ordinary MPI reduction. Export legacy subtraction/ledger independently.
  Same operands, tensor metric, quadrature, law and1e−10 J/m bound. The old
  first-step residual−2.519e−10 becomes−2.895e−14 J/m; source/weak checks unchanged.

The bounded comparisons are closed; see the report and `rate-floor.json`.
The tighter replay reproduces a pre-publication computational floor at a nearly
zero-damage Gauss point. `StepMetrics` now carries its index/coordinates and
incoming/candidate slip, rate and damage. `cohesive-rate-failure.csv` is written
before the unchanged floor assertion; the existing adaptation CSV schema stays
unchanged. This instrumentation does not alter the solve or controller formula.
The failure is not classified as a physical instability; exact onset-rate jumps
and algebraic active-set noise have not been independently separated.

The launch helper requests accepted-state termination before its hard wall limit.
Independent analysis matches physical points across temporary-reserve layouts,
uses common physical times, includes the full 3D stress norm and reports mature-
patch state separately to avoid dilution by unchanged intact history. All five
retained portable states reproduce physical exports bitwise after instrumentation.
No new constitutive law, tolerance change, damage floor or birth exemption.

Preserve prior audits and all failed evidence below.

---

# Portable setup and complete-support source audit

[Measured report](COHESIVE_RSF_PORTABLE_REPORT.md); baseline f2f730529 plus local
cleanup. Constitutive source and all prior qualification evidence remain intact.

- `cohesive_system*` declares explicit seed/prescribed-history/native-memory and
  completion settings; fingerprints new setup/Box parameters while retaining
  the unchanged old v2 signature. Actual owned-cell Box bounds are reduced once.
- `cohesive_initialization.cc` samples fresh physical inputs, retaining Gauss jump
  offsets and the native FE remainder. Relaxed jump must match the Q1 trace within
  the existing identity allowance. No imported kernel or history reset is used.
- `cohesive_geometry.{h,cc}` encloses the full radius-R disk centered R ahead and
  the vertical graph strip. Outward interval polynomial envelopes, not sampled
  quadrature points, protect interior extrema. No integration rule changes.
- `cohesive_state/advance/mechanics` check initial/accepted/trial/restarted support
  before sampling, assembly or publication; log clearance/approach and preserve
  accepted histories on failure. Formed-length and domain guards are independent.
- Benchmark `pilot.cc` keeps the old loading arithmetic and adds explicit strain
  functions, interval-average velocities and restart schedule validation. Fresh
  inputs reuse ASPECT's standard Box origin/extents instead of new geometry knobs.
- `qualify.py` supports binary/plugin/work/MPI overrides, actual-rank checks and
  a fixed portable verification suite. Active analysis reads only local native
  outputs, including relocated/direct launches. Archived replay remains separate.

Loaded pilot surfaces, geometry and physical bulk exports are bitwise preserved.
Custom history restart is exact; incompatible configuration/history and missing
support fail before publication. Point/normal accuracy and broader physical
propagation remain outside this cleanup. Earlier audit content follows unchanged.

---

# Default-off native ASPECT cohesive–RSF source audit

Entry f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved local work.
[Completed report](COHESIVE_RSF_NATIVE_REPORT.md), [formulation](COHESIVE_RSF_NATIVE_FORMULATION.md),
[changed files](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_rsf_native/CHANGED_FILES.md).
The earlier accepted prototype is retained separately below.

- `cohesive_bulk_history.{h,cc}` owns all 27 DGQ2 coefficients/cell and native FE
  gradient remainder. Gauss3 source projection retains nonrepresentable native
  strain; independent Gauss6 physical recurrence checks full plane strain.
  MPI updates use owned vectors before ghost refresh; CellId records are native
  checkpoint state, not affine/lifting response coefficients.
- `cohesive_geometry.{h,cc}` owns the retained C1 graph, cut/support quadrature,
  compact signed lifting and accepted-stress causal append. R and numerical
  support remain distinct; manager topology contains formed material only.
- `cohesive_system*`, `cohesive_mechanics.cc`, `cohesive_initialization.cc`,
  `cohesive_advance.cc`, `cohesive_state.cc`, `cohesive_output.cc` separate owner,
  native mixed inverse/normal work, checked conversion, immutable trials,
  typed manager Gauss rows, checkpoints and accepted-only diagnostics. The unused
  legacy nonnegative rate container is zero; physical signed Q1 rates are native.
- `cohesive_evolution.h` shares the retained signed admission solver with the
  reference. `cohesive_continuation.cc` proposes accepted-active-set Newton
  updates; the full signed weak/current-pressure certificate still decides.
  No global RSF potential or general stability/uniqueness certificate is claimed.
- `solver/cohesive_rsf.cc` uses ASPECT's actual constrained distributed Stokes
  inverse, native pressure scaling and AMG. Complete nonsymmetric pressure
  derivatives remain. Particular displacement interpolation masks velocity;
  pressure is solved natively. Finite checks use owned vectors. Initialization
  stress/pressure/storage and K/N are checked against the explicit source state.
- Conditional simulator hooks defer publication to the actual outer decision;
  native v2 checkpoints reject live trials and save geometry, identities, clocks,
  every surface history and general bulk memory. Derived caches rebuild on
  restore. The clock closure bound counts accepted additions and outer retries;
  it changes no physical/numerical allowance. The payload is absent when off.
- The false-by-default parameter and scoped guards protect all legacy routes.
  No BP3 source/input changed relative to the 1456-file entry hash inventory.
  No new particle read or particle mechanics dependency exists in this pilot.

Final native two-rank pilot, general 1/2/4-rank history, actual rejection and
same-rank restart pass within the report's scope. Final four-step restart and
completed checkpoint readback preserve every physical record and exported field
bitwise. Source <=1.923e−13 scaled, weak <=9.977e−9 Pa; no tolerance relaxation.
Legacy unit controls pass. Original owned/ghost, scaling, first-append,
nonfinite initialization, clock and formed-budget failures remain preserved.
Normal/point-traction and broader spatial/branch findings are not repaired by
this integration. All earlier audit evidence follows unchanged.

---

# Integrated propagation / parallel native implementation source audit

[Report](COHESIVE_INTERFACE_PROPAGATION_REPORT.md) · [selection](COHESIVE_INTERFACE_PROPAGATION_SELECTION.md) · [evidence/index](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_propagation/INDEX.md).
Baseline f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved work.

`cohesive_interface_propagation.py` is the maintained experimental entry point.
It reuses the candidate-admission owner, load-turn kernel and native signed
return; every accepted state is solved, not replayed. Root owns one atomic
state/material-ledger checkpoint and immutable per-step journal. All 160 main
physical publications and 27 formed transactions are counted separately.
Every persistent Gauss ID, old arc, jump/reference, damage, normalized age,
absolute slip and Maxwell memory is preserved. New metadata adds zero work
and fracture; physical activation pays its actual cohesive storage and Gc da.
Virgin candidates may be replaced only with zero physical histories, while
formed history is immutable. Stress/storage transfer defects are exactly zero.

No constitutive source changes. With e=jump-reference and C0=2Gc/peak²,
cohesive storage is (1-a)e²/(2 C0 a), q=(1-a)e/(C0 a), and fracture inventory
is Gc integral(a dGamma). Intact a=0 uses the retained weak tied constraints;
a=1 has q=storage=0. Current-pressure regularized RSF/radiation contributes
a[sign(V) pn mu(V,theta)+RAD V], pn=p0+N c. Age is normalized:
theta_new=(theta_old+V0 dt/Dc)/(1+abs(delta_jump)/Dc). For homogeneous Maxwell,
beta=1/(1+G dt/eta), affine memory_new=beta*(old+boundary increment), and
jump memory_new=beta*(old+signed nodal increment). Physical stress is the full
3D plane-strain native basis reconstructed with those coefficients, not an
auxiliary coefficient. No hand rotation, history reset or phase floor.
Independent recurrence/trace/RSF checks cover every accepted step. General
arbitrary bulk tensor histories outside this response span remain unintegrated.

`cohesive_interface_path.cc` adds explicit trailing `column-mpi`; its helper
`cohesive_interface_path_parallel.h` uses deal.II/MPI and preserves the serial
path. All ranks construct identical replicated mesh/master layouts. Root
assembles A/B/D and normal-work geometry once and broadcasts them. Ranks solve
owned response columns and integrate owned whole cells; complete columns and
basis values are copied, not summed. Cell-ordered folding makes R-disk samples
rank-independent (legacy regrouping <=4.04e-15 relative). Independent energy
sums differ <=2.34e-14. Full Gauss rules, gauge/mean pressure, normal convention,
constraints, work adjoint and pressure tangent are unchanged. Hoisting native
shape/strain lookup out of coefficient loops preserves addition order. Default
serial 19 tables and actual grown endpoint operators/bases match byte-for-byte.
Production source/CMake and BP3/defaults are unchanged by this task.

Qualify MPI before evolution: 1/2/4-rank exact columns, short advanced histories,
same-rank JSON restart and single publication; injected nonroot failure aborts
before a completed export, leaves accepted input unchanged, and retries exactly.
Incomplete exports are preserved and never treated as cache hits. New source
versions/binaries and commands are hashed; the main ran the qualified second
parallel binary, later shape-cached binary verified on its grown state. Timings
are not overlapping. No robust local MPI speedup is claimed. Root-coordinated
JSON ownership and replicated mesh are explicit prototype restrictions.

Focused checks: temporal windows .45375-.45625 and .45875-.460 at half dt;
finer own-state continuation .440-.457375; independent fitted same-field
normal/shear/tensor reconstruction at first maturation. Full pressure FD
relative discrepancy 7.96e-10. The normal-only response probe preserves actual
K/Q/history and changes only measured N, retaining its derivative. Its initial
old-operator predictor failed the strict certificate (651.57 Pa); recomputing
the existing admission predictor for N succeeds (4.27e-9 Pa). Failed input/log
are retained, no tolerance changed, no probe published. Restore the main cell
spacing after a fine diagnostic; never leak process-local settings. A NumPy
boolean serialization fix resumes the unchanged atomic accepted checkpoint.

Max main weak residual 9.98e-9 Pa; minimum normal 9734 Pa; BE ledger 1.27e-12
J/m. Separate physical trapezoidal-work screens pass in loading/hold, using
sum of absolute step remainders against sliding and added fracture separately.
Normal weak targets still fail (2.545/3.255% bend/late-turn perturbations), and
raw traction/old ahead-region failures remain. Their measured response impact
is small here, not a point-stress accuracy proof. Fine spacing covers first
maturation, not the complete later hold or general bulk convergence. Report
contains the concrete next ASPECT integration task; no new campaign authorized.

The original audit and its failed evidence follow unchanged.

---

# Short loading-turn / future-material refinement source audit

See [report](COHESIVE_INTERFACE_FORMED_TURN_REPORT.md) and [evidence/index](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_formed_turn/INDEX.md).
Baseline f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved work.

No existing implementation or native binary is changed. Four focused scripts
`cohesive_interface_formed_turn.py`, `check_...`, `analyze_...`, `verify_...`
reuse the accepted candidate owner and structural-Q1 native solver. Configuration
is process-local; physical source, production paths, BP3 and defaults are unchanged.
Only future material cells are .09375/.046875 m; old registry/quadrature/history
are immutable. Keep the same physical R, support, Gauss rules, Maxwell recurrence,
current-pressure tangent, RSF, radiation and single fracture inventory.

The source is the own verified .425 s half-step checkpoint, identical in both
runs. Complete .440 s with 120 physical publications each, 6/12 registry commits,
zero transfer stress/storage and zero metadata energy. Actual loaded native
contact rejection/retry and JSON rebuilt-context restart pass. FD errors
4.47e-10/5.39e-10. Quarter-step own-state .4375–.440 suffix adds 40 publications,
changes growth -.2128%, and retains front brackets. Engineering invariants and
scientific accuracy screens are stored separately, with unchanged thresholds.

Independent frozen bulk reconstructions share physical geometry, jump, Maxwell
memory, Gauss histories, normalization and virtual-slip tests; no accepted
history is converted. R-disk tensor/shear checks pass within declared accuracy.
Normal-work error is 1.937%/2.006% depending on fitted reference: no robust 2%
qualification. Raw sided tractions remain poor; earlier normal/point failures
are not overwritten. Weak incompressibility is distinct from strong divergence.

Checkpointed raw data are separate from compact PNG/GIF/CSV evidence. Only new
unused cell-mean diagnostic bases are omitted with hashes/reproduction inputs;
no old archive, field or physical history is removed. Entry preservation hashes,
publication accounting and final manifest cover the completed bounded scope.

---

# Temporary-candidate / atomic-formation source audit

[Report](COHESIVE_INTERFACE_CANDIDATE_ADMISSION_REPORT.md) · [formulation](COHESIVE_INTERFACE_CANDIDATE_ADMISSION_FORMULATION.md) · [evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_candidate_admission/INDEX.md).
Baseline f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved work.

- User selects the temporary-candidate distinction. It follows the existing
  append-only persistent-geometry contract: all correction happens before
  commitment. No already formed curve, Gauss owner, history or direction moves.
  The selected old .410 s input has no wholly virgin piece reinterpreted as
  temporary. Legacy checkpoint conversion and removal of old SVD shadows are
  expressly unsupported, not silently performed.
- Experimental `cohesive_interface_evolution.h` adds opt-in independent Q1
  masters for nodes fixed by three tied Gauss values. The remaining constraints
  still use SVD; C^T R and C^T J C enter once with the complete nonsymmetric
  pressure-dependent derivative. This is analytical constraint elimination,
  not post-solve clipping, a new energy, penalty, threshold or damping law.
  `cohesive_interface_turn_solver.cc` exposes an explicit per-context switch;
  default off. Constitutive `source/reconstructed_fault/cohesive_interface.cc`
  and all production sources remain unchanged.
- Experimental `cohesive_interface_path.cc` optionally exports R-disk samplers
  for immutable arc endpoints on candidate-independent Gauss8 bulk cells.
  Gauss12 verifies that numerical measure. Mechanics/energy/normal-work Gauss6,
  Gauss3 surface histories, full plane strain and .75 m numerical support stay
  unchanged. The old default sampler and eight default tables match bitwise.
- `cohesive_interface_load_turn.py` only gains an optional native-library path
  for an isolated new build. The legacy tied-birth rejection is unchanged and
  actually re-exercised on a PRIVATE copy, preserving original failure files.
- New `cohesive_interface_candidate_admission.py` owns view versus formed-prefix
  versions, IDs, exact-zero checks, immutable trials and one atomic commit.
  Retained nonzero native stress/pressure response columns and Gauss coordinates/
  weights are bitwise identical. New/retired lifting coefficients are exactly
  zero. The generally nonzero bulk tensor history is preserved everywhere in
  the retained approximation, not reset in the footprint of a new plane.
- Direction is sampled from accepted stress at the formed endpoint (an explicit
  update to the propagation hypothesis), using the unchanged conjugates,
  history preference and physical R. Cached geometry is numerical predictor
  state. Any forming trial is rebuilt at its actual incoming clock; all retries
  use the same accepted Maxwell/slip/age/irreversibility history. Fixed refresh
  cadence is measured independently of physical timestep and reserve length.
- New geometry is allowed only as an unpublished trial view before native joint
  admission. A virtual multiplier above peak is NOT an admissible tied birth.
  The coupled positive-time update accounts for actual cohesive storage,
  Gc*Delta(a), friction, radiation and Maxwell losses. Promotion changes metadata
  only; full-view energy equals persistent inventory because remaining candidate
  history is zero. No second Gc*length charge or zero-time physical publication.
- `qualify/check/analyze/verify_cohesive_interface_candidate_admission.py`
  provide actual rejection/retry, nonzero-history restart, changed-cache matched
  controls, candidate completion and energy/field checks. Source defaults and
  BP3 are hash-protected. Native weak 1e-8 Pa, FD 1e-6, transfer 1e-7 Pa and all
  original acceptance tolerances remain. Coarse/suffix physical-energy failures
  are retained; fine timestep meets the unchanged 1% screen.
- This is the isolated serial exact-jump native fixture and JSON owner. It does
  not replace DG/FE/particle production history, qualify full Simulator restart,
  MPI/AMR, arbitrary evolved-field remapping, secondary fracture or a global
  RSF stability certificate. Historical point-traction and normal-reaction
  failures and older spatial/branch errors remain in their original evidence.

---

# Boundary-driven causal extension source audit

[Report](COHESIVE_INTERFACE_LOAD_TURN_REPORT.md) · [data/index](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_load_turn/INDEX.md).
Baseline f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved work.

- Experimental `cohesive_interface_path.cc` optionally assembles two fixed
  laboratory affine strain/pressure response columns. No affine history is
  rotated. Q2/P1disc, Gauss6, resolved .75 m numerical support, physical .75 m R,
  normal bulk-work measure and surface Gauss3 ownership are retained. The
  original one-affine option remains default; seven tables match bitwise.
- `cohesive_interface_evolution.h` generalizes the prescribed affine block from
  one to two modes and shifts jump-column offsets consistently in force,
  current normal pressure and tangent. Both memories use the existing Maxwell
  beta recurrence. Unknown jump increments, signed local law, constraints,
  complete pressure derivative, residual/admission/curvature tolerances and
  publication remain unchanged. `cohesive_interface_turn_solver.cc` adds an
  explicit multi-affine C API, preserving the existing API and default layout.
- `cohesive_interface_load_turn.py` owns the isolated physical loading,
  immutable accepted trials, append-only geometry transactions and JSON state.
  Known affine values are algebraically eliminated only in the reused search
  predictor; native acceptance certifies the full original two-mode residual.
  Boundary work sums both conjugate reactions. No production checkpoint,
  production material law, Simulator path or BP3 file changes.
- Native history consists of both affine memories, all signed surface Maxwell
  modes and six owned Gauss histories (jump, damage, theta, reference,
  relaxed_jump, absolute_slip), plus total displacement/jump clocks. Appends
  zero-extend modes/new tied histories, retain accepted virgin age, preserve
  old response/physical stress-pressure columns and all old records bitwise.
  Transfer storage/stress defect and new fracture inventory are zero. Failed
  birth, duplicate and stale publication leave the accepted state unchanged.
- `qualify/check/plot/verify_cohesive_interface_load_turn.py` provide independent
  homogeneous Maxwell/old-layout controls, coupled FD, actual contact-loss
  rejection, rebuilt native-context JSON restart, common-field fitted
  reconstruction and preservation checks. No new physics or tolerance policy.
- Fronts now use actual irreversible Gauss damage brackets rather than current
  slip, which can unload. The previous slip diagnostic remains explicitly
  non-authoritative. Common-time shape replays use their own accepted history,
  identical timesteps and unchanged geometry; not a history conversion.
- The retained persistence/maximum-shear rule is a hypothesis. Physical R,
  numerical support w and reserve S are separately fixed/named. Both target
  reserve and append scheduling change the outcome; permanent virgin-direction
  commitment remains unmodified under the append-only specification.
- Point-normal jumps (~520 Pa current Cartesian/16.6 Pa fitted), historical
  ahead-region 2.85%/2.29 Pa normal failure and new small old-region relative
  normal failure (3.56%, .118 Pa) remain. Selection-disk accuracy does not qualify
  tip point stress. Half-quantum physical fracture-work screen 1.000894% fails.
  Engineering verification is not a claim that all scientific gates passed.

---

# Fixed-path bulk/contact source audit

[Report](COHESIVE_INTERFACE_BULK_CONTACT_REPORT.md) · [selection](COHESIVE_INTERFACE_BULK_CONTACT_SELECTION.md).
Baseline f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved work.

- Only existing isolated `cohesive_interface_path.cc` is extended: optional
  independent vertical lifting/normal-test half-widths, fitted-grid diagnostic
  mapping (Q4 exact saved graph), aligned surface/support knots, physical
  energy/divergence/pressure diagnostics. Cartesian mesh-scaled defaults retain
  seven original operator/trace/column tables bitwise. No production source,
  material law, actual residual/tangent or checkpoint backend is changed.
- Frozen controls use the same actual Γ and physical surface Maxwell-memory
  field, solve each mixed bulk response independently, and compare common Q1
  tangential tests/common normal bubbles. Full 3D plane strain and physical
  mean pressure convention are retained. Raw sided normal and sided-mean shear
  diagnostics remain distinct from variational contact reactions.
- Independent 1D assembly/projection exposes artificial stiffness from an
  unresolved counter-lifting gradient kink. Fixed numerical support improves
  shear/work/growth errors, while raw cut-cell contact/incompressibility error
  persists. Higher pressure degree cannot create a sided trace inside a cell;
  Q3 alone is not used as an accuracy reference. Fitted Q2 refinement and a
  fitted Q3 alternative qualify the reference within the stated scope.
- New investigation/assessment/evolution/fresh/repeat/plot/verification scripts
  reuse the unchanged native signed C++ solver and point return. No trajectory
  replay, pressure lag, constitutive calibration, tolerance relaxation, stress
  clipping or extra fracture inventory. R/persistence are unchanged.
- Imported common-history controls are new diagnostic initializations with
  recorded reconstruction energy/free-force defects, never silent conversions.
  Fresh fixed-path controls and fresh causal runs verify actual evolution
  independently of imported-state relaxation. Fitted geometry remains a fixed-
  path diagnostic; it is not claimed to preserve history on geometry extension.
  Corrected causal extensions retain the fixed Cartesian nested representation
  and its full old-field/history preservation checks.
- Bounded native rejection/retry/restart and once-only publication pass. The
  raw point-traction defect and small ahead-region normal target failure remain
  explicit. No general physical propagation, objective finite sliding, MPI or
  production integration qualification is inferred.

Full earlier audit follows verbatim.

---

# Causal path experiment source audit

Task baseline f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved local work.
[Detailed report](COHESIVE_INTERFACE_PATH_REPORT.md) and
[staged formulation](COHESIVE_INTERFACE_PATH_SELECTION.md) distinguish verified
mechanics from the uncalibrated propagation hypothesis and failed spatial gates.

- New isolated `cohesive_interface_path.cc`: fixed native Q2/P1disc or Q3/P2disc
  background matrix, cut integration, complete tangent/normal gradients, nested
  tangential modes and constrained normal-bubble work reaction. Exact full 3D
  plane-strain deviator, mean-zero p′ plus physical p0, current-pressure tangent.
  Gauss6 assembly/Gauss8 independent work; lossless compressed F64 basis tables.
- New `cohesive_interface_path.py`: accepted tensor sampling, analytic conjugate
  maximum-shear directions, structural tie preference, fixed physical R and
  derivative-consistent persistence interpolation. Existing old basis functions
  and geometry never change. New Gauss records are tied and virgin; six old
  histories and Maxwell coefficients are retained, with zero extension only.
  Independent column/physical-state checks cover nonuniform evolved memory.
- The unchanged `cohesive_interface_turn_solver.cc`, native signed evolution
  header and `CohesiveInterface` point return compute physical updates. Python
  owns only this experimental controller/archive. No production checkpoint,
  geometry, material, history or BP3 source is changed. CMake adds one target.
- Analysis/qualification/plot/package scripts record actual fronts, physical
  work, normal contact versus raw traction mismatch, finite-difference checks,
  actual failed native contact trial, atomic retry/publication/restart, and
  entry protection. The original interpolation and point-pressure data remain.
- Current work-reaction mode is selectable and used only for fresh labelled
  controls. Legacy experimental point pressure remains default. No hidden
  pressure shift/lag, normal opening, dilation, extra damping, strength change,
  second fracture inventory, history reset or angular sector bound is introduced.
- Qualification is limited: Q2/Q3 growth differs 17.35%, raw point traction
  continuity remains unqualified, physical R/tied reserve are assumptions, and
  no unique global path or general RSF potential/stability theorem is inferred.

Full earlier source audit follows verbatim.

---

# Curved native mixed mechanics and conditional path-selection audit

[Report](COHESIVE_INTERFACE_TURN_REPORT.md) · [formulation](COHESIVE_INTERFACE_TURN_SELECTION.md).
Only the isolated benchmark generator/CMake target and new reference/analysis/
plot/qualification/package files change in this task. The protected production
point law, DG history, simulator, BP3 and old archive hashes retain their entry
values. The optional smooth generator leaves the sharp-corner model intact.

The smooth lifting is symmetric on both sides and includes the complete
gradient of t(x)φ(x), including t′φ. Native vector displacement/P1disc pressure,
Q4 geometry, actual sided traces, consistent arclength, pressure reference,
current friction pressure and full 3D deviatoric work are retained. Independent
Gauss8 bulk integration agrees with Gauss6 to <1e−13 relative. Independent
Gauss12 geometry/mass/known-stress weak patch checks support the Gauss3 measure;
they do not re-quadrature arbitrary evolved Gauss history. No normal constraint
is removed and no opening, dilation or new damping is introduced.

The new stateless C wrapper reuses CohesiveEvolution::solve and the unchanged
CohesiveInterface return. Native states are computed, not trajectory replayed.
An accepted-active-set predictor only proposes an increment; complete native
signed admission/residual/tangent checks gate every accepted result. The
full-Q2 first-admission failure remains, despite bounded predictor/step probes.
Qualified Q1 increments are an explicit Galerkin restriction. Existing full-Q2
permanent slip is a static affine offset; all Gauss histories remain intact.
The prepared nonzero Cartesian Maxwell history lies in the retained affine
mode. Arbitrary history remapping to a smaller trace is not established.

The 12×12 finite specimen uses exact homogeneous bulk similarity; Gc, Dc,
material inputs and physical histories are not scaled. Separate provisional
branches start from identical old material. A finite-horizon mean-new-damage
score selects one primary candidate without invoking a global incremental
potential. Its turn-width dependence is a measured constitutive/path-design
limitation. Losing previews contribute no accepted fracture inventory.

Geometry discard/retry, inconsistent-geometry rejection, duplicate publication,
JSON restart with native context rebuilding and fresh selected evolution pass.
Ownership is confined to the experimental archive; no production restart or
general geometry-cache/Maxwell transfer qualification is inferred. Bitwise
archived-sharp operator expectation failed at roundoff (2.17e−16 relative K);
the physical stress/pressure probe differs 2.59e−12 Pa at the unchanged 1e−7 Pa
gate. Rebuilt smooth outputs are bitwise. Raw failures and prior science remain.
Full earlier audit follows verbatim.

---

# Focused Q1 surface-resolution reference audit

[Report](COHESIVE_INTERFACE_SURFACE_REPORT.md) · [selected scope](COHESIVE_INTERFACE_SURFACE_SELECTION.md).
Only the independent `cohesive_interface_front.py` gains a cached physical
layout and optional initial increment. Existing unit-domain history is bitwise
over its 20-step loaded identity control. No C++/production/BP3 source changes.
New reference/analysis/plot/package scripts use the existing point law and exact
layer DtN; no new constitutive update, penalty, phase floor or pressure fit.

With dx=L/n, R=Qᵀ diag(w)/dx, beta=(1+G dt/eta)^−1, z=beta(m_old+increment):
  f=G/H wall−Kz−R[q+a sign(V)(pn mu+eta_rad |V|)],
  pn=p0+P N z, V=Q increment/dt.
The derivative includes beta a sign(V) mu P N as well as point return/state
and cohesive derivatives. Complete signed Gauss box feasibility certifies tied
reactions; for positive interior Q1 Gauss shapes a tied cell fixes both adjacent
nodal increments. Curvature is checked on that exact admissible nullspace, not
artificial constrained rows or the positive search cone. It is not a general
non-associated RSF stability theorem. Fixed endpoint reactions are not free rows.

Pressure/reference mean modes retain K(0)=G/H,N(0)=0; physical p0=10 kPa remains.
Alias300→600 actual final shear-load changes <2e−5 Pa. Gauss3 per surface cell
is fixed, with independent damage, age, jump/reference, relaxed memory and
absolute slip. Refinement is fresh initialization, not Gauss-history remapping.
Common .125 m box averages integrate actual histories; Q1 diagnostic contours
are used only after checking equality with stored Gauss damage. The best-L2
linear jump projection is a separate diagnostic, never an evolution transfer.

The fine first-hold previous-rate predictor stop is preserved and reproduced.
Only the initial increment changes to zero; all accepted arrays/clock/age stay
fixed. It returns nonzero growth and passes the same signed residual/curvature
checks. Its restored trajectory keeps publication counters and all old history.
A preliminary absolute-random FD probe crossed near-zero RSF rate scales
(1.10e−6 discrepancy); increment-scaled admissible probes give ≤5.14e−9 under
the unchanged 1e−6 gate. Original logs remain; no tolerance is enlarged.
Actual NPZ history serialization/retry is bitwise on both resolutions, separate
from the run-time memory-copy repetition diagnostic. No native restart claim
is inferred from this independent reference.

Native FE versus exact bulk differences at fixed 256 surface resolution are
reported separately from 256→512 surface changes. Late practical targets pass;
first admission shifts 72.5 ms despite identical initialization. Two meshes,
late temporal control and a common quadrature rule do not qualify exact onset,
convergence order or nonlinear surface quadrature independently. Prior 65.4%
early failure, finite-wall sensitivity, absent bulk inertia and production
ownership/qualification limits remain. Full earlier source audit follows verbatim.

---

# Finite-wall pressure mechanism and longer native continuation audit

[Report](COHESIVE_INTERFACE_BOUNDARY_REPORT.md) · [selected scope](COHESIVE_INTERFACE_BOUNDARY_SELECTION.md).
Only the experimental native fixture adds `CI_FAULT_FRACTION` (default .4,
optional string recorded in geometry identity) and guarded `CI_EXTEND_LOAD_END`.
Existing checkpoint strings are unchanged when the geometry option is absent.
A future load plan may extend only if the accepted clock precedes its old end;
an actual accepted-hold checkpoint rejects reinterpretation before publication.
No source/include production API, point law, physical parameter, limiter or
numerical tolerance changes. All protected files/archives retain entry hashes.

For layer distances b,t and H=b+t,d=t−b, incompressibility fixes the nonzero-mode
cross-section flux. Slip response alone gives dm/2, and pressure-driven flow
adds −H³p_x/(12G). Thus C=6Gd/H³, D=3Gd²/H³, N(k)≈−iC/k and
K(0+)=G/H+D, while the separately admissible mean mode has K(0)=G/H,N(0)=0.
The zero mode is not replaced by a nonzero-mode limit. The old algebraic pin
plus global pressure-mean removal remains; physical p0 anchors mean confinement.
Normal traction is p0+p′−tau_yy at current trial coefficients, with complete
pressure derivatives. Pressure has no recoverable energy norm that would bound
these normal-load changes from the much smaller shear-stiffness change.

`analyze_cohesive_interface_boundary.py` reuses the exact layer DtN and sharp-Q1
Galerkin reference. It validates mean/long modes, 50-digit small-k limits,
centred/reflected/wall-distance responses and alias tails. For identical compact
memory A=∫m,J=∫(x−4)m, the 8→16 m change is approximately
Δpn=C[A(x−4)−J]/16 and Δtau=−DA/16. Residual .0195 Pa versus 635 Pa RMS normal
change identifies a physical finite-wall/period effect. The operator/evolution
decomposition closes to 3.74e−11 Pa. Q1 patch tests are full weak-load functionals
including tied reactions, not free reduced directions across intact constraints.

The reflected native run is a fresh proper-rotation diagnostic; the longer run
retains y=.4 on the unchanged 16×1 m periodic mesh. ASPECT schedules/checkpoints;
the existing C++ signed coupled solver computes new states, not prescribed
trajectory replay. Restore keeps all Gauss/nodal histories and physical clocks.
The pre-extension past is bitwise through 2.44875 s; the old endpoint's changed
snapping landmark has a reported 1.47e−14 s clock difference, without rewriting it.
A resource-bounded completion replays three states bitwise. Logical CSV histories
select the restored branch and count each accepted publication once; all raw
records/checkpoints remain. The half-step control keeps its already-pending
first interval, then halves subsequent increments.

Gauss3 mechanics/source/history and independent actual Gauss6 full-plane-strain
stress/work checks are unchanged. The exact Q1 continuum strip is re-solved at
the refined actual memory to assess mechanical FE coupling error (~1.10 Pa
work-adjoint shear load/.00210 Pa projected normal at the tip), not merely
re-evaluated on another mesh. It does not measure raw point-trace accuracy or
surface-damage spatial convergence. The new fracture reservoir is solely Gc∫a;
initial seed inventory 3 J/m remains. Bulk/bond storage, friction, radiation and
Maxwell losses are separate. Raw physical work error is never corrected with
background subtraction or equated to exact BE ledger closure.

The finite trajectory reaches 3.2135 m full left advance with bounded temporal
uncertainty (1.71 mm); new-window and hold energy screens pass. Finite-period
boundary effects, old early spatial/branch failures and absent bulk inertia
remain limitations. A quasi-static front secant is not a physical rupture-speed
qualification. No free path, bend, transport, MPI or broader propagation claim.

---

# Horizontal-domain control audit (2026-10-26)

[Report](COHESIVE_INTERFACE_DOMAIN_REPORT.md) and
[selection](COHESIVE_INTERFACE_DOMAIN_SELECTION.md).
The only native benchmark changes are optional pre-publication observation/
detached-stop hooks and a diagnostic-only independent extended-precision FD
reference. `source/`, `include/`, BP3, the physical point law and Newton equations
retain entry hashes. Default behavior outside the opt-in fixture is unchanged.

`cohesive_interface_domain_diagnostics.inc` observes the immutable trial before
`publish()`, exports actual full Gauss6 bulk fields, and rejects detached a≥.01
growth without publication. `CI_DERIVATIVE_ROUNDOFF_REFERENCE` permits a second
independent residual evaluation only after the double diagnostic fails; it
retains exactly the original perturbations, active columns and 1e−6 gate.
The independent long-double equations reproduce the retained point law; they
never supply a physical state, force, tangent, energy or Newton update. A 1%
biased tangent still fails. Original double errors remain visible. MP50 checks
use saved coefficients and native-active directions; a constrained coordinate
crossing the intact cusp is not a classical admissible derivative direction.

For a frozen native relaxed-jump field m, shear and normal actions are
`tau = G w/H - K m` and `pn = p0 + QP N m`, using the retained common surface
projection and globally mean-free solved p′. Added tied Q1 cells have zero
Gauss and nodal jump at every accepted step, hence make no additional physical
activation/work. The 16 m experiment is a change to horizontal period/bulk
domain, not a remapping or constitutive change. Geometry is translated only in
analysis. Pressure scaling, sided contact and full 3D tensor convention remain.

Independent actual Gauss6 bulk integrals use
`||tau||² = xx²+yy²+zz²+2xy²` and elastic storage `integral ||tau||²/(4G)`.
No scalar-magnitude subtraction replaces the tensor difference. Common fields
are paired by physical quadrature coordinates and weights; cell means are used
only for plots. Normal-pressure differences are not removed as gauge offsets:
both full-domain mean p′ values are already ~1e−13 Pa. Raw work acceptance uses
the sum of absolute step remainders; independently integrated homogeneous-strip
excess quantities only compare different extensive domains.

Fresh reference/archived trajectories are bitwise, as are restarted enlarged
overlap histories and physical columns. All publication counters advance once.
Domain-sensitivity targets for growth, stress and normal pressure fail despite
passing energy/residual/lifecycle checks. Normal maps are particularly sensitive
at identical memory. No distant-boundary, full propagation or new stability
certificate follows; retain prior spatial/branch failures and fixed wall limits.

---

# Native signed evolution / straight propagation audit (2026-10-26)

[Report](COHESIVE_INTERFACE_EVOLUTION_REPORT.md) and
[selected gates](COHESIVE_INTERFACE_EVOLUTION_SELECTION.md).
`cohesive_interface_evolution.h` owns the experimental stateless signed solve;
the accepted C++ `CohesiveInterface::evaluate` remains the single point law.
For fixed homogeneous mixed Schur energy K, trace Q, weights W and physical
normal maps N_minus/N_plus:

    c = beta (c_accepted + Delta x), beta = (1+G dt/eta)^−1,
    pn = p0 + (N_minus+N_plus)c/2,
    F = (K c)_jump + Q^T W r(Delta jump,pn;accepted a,Theta,history),
    J = beta K_jj + Q^T W [r_delta Q + r_p beta N_j].

Both pressure traces must stay compressive. Intact Gauss reactions satisfy
|lambda|<=peak; signed semismooth complementarity selects the tied set, then
C=ker(Q_tied) gives C^T F/C^T J C and a complete weak bounded-reaction certificate.
There are no artificial constrained-row diagonals or twice-reduced bulk terms.
Absolute row/column sums bound box-iteration norms for sign-changing Q2 shapes.
SVD rank cutoffs use machine epsilon. Algebraic equilibration changes neither
F nor its acceptance tolerance. The retained local mass-scaled symmetric-J test
is not promoted to a global RSF potential-Hessian stability theorem.

The fitted owner (`CI_NATIVE_EVOLUTION`) consumes fixed native maps plus one
initial snapshot and computes subsequent states. Its native independent source,
point return, field reconstruction and BE checks precede one owner publication.
Existing ASPECT serialization stores physical clocks, histories and fixed maps.
`cohesive_interface_signed_front.inc` maps the actual straight native mixed
operators into this same solve, caches only geometry/layout factors, and reuses
the existing front predictor. Predictor positivity only proposes a root; full
signed weak acceptance and unconstrained admissible polish remain authoritative.
Accepted-rate fallback and contact-aware line search handle failed proposals.
An optional physical cutback wrapper retries bounded numerical failures from
unchanged accepted data; negative curvature, buffer/contact acceptance and the
original weak/ledger/derivative gates remain. Test-only cutback/state-audit
hooks qualify exact publication/restart, without production parameters.

The bulk is solved by exact fixed-geometry mixed condensation, including solved
pressure and the full 3D plane-strain deviator. The component uses ASPECT's FE
mesh, native assembly and postprocessor checkpoint; the host Stokes/advection
scheme is disabled. Its accepted bulk history is in native relaxed FE-response
coefficients, distinct from the host dummy solution and protected production
DGQ2 backend. No particle state enters this isolated component. This does not
claim replacement of the production propagation/history backend.

The straight law remains Gc integral(a ds) on Gamma, with phi provisional.
New virgin records preserve the material aging clock; no extra bulk fracture
reservoir, phase floor, reference reset or corner law is added. Actual Q1 jump
threshold fronts are checked only where all new-material nodal increments are
monotonic; no interpolated age or off-Gauss constitutive recomputation is used.
Native Gauss6 bulk fields independently verify work/stress/storage; compact
physical means and unaveraged extrema are labeled separately.

Native reference histories differ <=1.11e−15; loaded rejection/cutback and
matching-step restart are bitwise. Late straight full advance is about .75 m,
with a resolved cohesive zone. The original 65.4% early spatial failure remains.
Whole-run energy passes at dt=.0025; hold-only .0025 screens fail, while the
focused .00125 hold passes. Candidate-buffer stops are preserved. Source,
include, BP3 and previous evidence archives are protected; only experimental
benchmark code and documentation changed in this task.

---

# Activated-append work/Maxwell audit (2026-10-26)

[Report](COHESIVE_INTERFACE_ENERGY_REPORT.md) and
[selection/identity](COHESIVE_INTERFACE_ENERGY_SELECTION.md).
New `cohesive_interface_energy.py` reuses `Admission.solve/publish` unchanged,
with one selected full timestep and one checkpoint-based half-step replay.
It independently partitions the raw trapezoidal-work remainder using the
native K matrix and relaxed coefficients in long double. The independent
homogeneous Maxwell control integrates exact exponentials over the ramp/hold;
it tests individual response/work/loss, not only ledger cancellation.
`analyze_cohesive_interface_energy.py` exports common-time response and Gauss
profiles, separate loading/hold errors and two PNGs. No physical equation,
acceptance tolerance, production source or BP3 setting changes. Raw energy
screens pass at the selected timestep; historical failures remain below.
All post-event derivative checks and publication checks are retained. The
new runs are independent reference evolution, distinct from the prior native
publication/checkpoint adapter and from a future native outer nonlinear solve.

---

# Event-resolved oblique admission audit (2026-10-26)

See [report](COHESIVE_INTERFACE_ADMISSION_REPORT.md) and
[equations/selection](COHESIVE_INTERFACE_ADMISSION_SELECTION.md).
`cohesive_interface_admission.py` reuses the native fitted Q2/P1disc mixed K/N/Q
operators. The signed semismooth predictor represents peak*partial|jump|, then
polishes in null(Q_tied) with C^T F and C^T J C. Final acceptance always checks
the unrestricted bounded Gauss multiplier equation; Q2 basis functions are not
assumed positive. Full pressure, implicit age and damage derivatives are retained.
A partial-Gauss algebraic reference independently checks the constrained tangent.
The tied event is bracketed from unchanged incoming histories; the bracket is
not a physical-onset accuracy claim. New damage is solved after tied insertion,
not invented during geometry birth. Every physical update ages/relaxes once.

The native owner gains only an opt-in `CI_ADMISSION_INPUT` replay verifier. It
uses the unchanged compiled point return, actual mixed-response coefficients,
sided pressure checks, Maxwell recurrence, weak force and BE work before native
Gauss-history publication. Invalid/stale candidates cannot publish. Actual ASPECT
restart passes bitwise across activation. Native replay verifies an independently
solved sequence; it does not solve production outer Newton or general topology.
With replay disabled, the prior native owner archive is bitwise unchanged.
Production source/include/BP3 files are unchanged.

Physical energy is reported separately from the exact BE ledger. The fine global
remainder passes the 1% fracture screen but fails the 1% sliding-loss screen.
Matched homogeneous-Maxwell and alternative dissipation-quadrature diagnostics
are retained without subtracting or redefining that error. The response test
activates an entire prescribed append at small damage; no fully decohered region
or selected free direction is demonstrated. The earlier spatial failures and
closed-corner limitations remain.

---

# Non-collinear exact-jump reference and owner audit (2026-10-26)

See [formulation](COHESIVE_INTERFACE_KINK_FORMULATION.md) and
[report](COHESIVE_INTERFACE_KINK_REPORT.md). No production source is changed.
The new `cohesive_interface_kink.cc` executable assembles actual native deal.II
Q2/P1disc mixed mechanics on a fitted piecewise-straight candidate. Duplicated
upper trace nodes carry t*delta; both displacement components, divergence,
pressure and full plane-strain deviators enter A/B/D. Condensation retains
K=D−B^T A^-1 B and both normal-pressure responses. Pressure space stays fixed;
installing the new segment changes the trace constraint/admission subspace.
The independent Gauss6 field/energy evaluator uses the same physical FE response
with higher quadrature, rather than inferring unchanged stress from jump alone.

The reference harness preserves complete physical coefficients and Gauss records,
checks the actual compiled unchanged point return, and includes the pressure
partial in the nonsymmetric tangent. New records have zero damage/jump/storage
and inherited virgin age; old normals, bond reference, age and Maxwell memory
are immutable. Native `cohesive_interface_kink_owner.cc` uses actual manager
swaps, native archive ownership and ASPECT checkpoints for this imported verified
state. The adapter explicitly sets the existing +1 shear convention so fresh and
restarted metadata agree; production serialization is unchanged. Populated
production projection caches, general pressure-space changes and production
outer cutback remain unqualified.

The kernel's radius-.1 m stress average is diagnostic, not a changed strength
law. The prescribed kink does not follow the nearly horizontal/vertical
maximum-shear directions of the tested loaded snapshot. Corner zero jump follows
from continuous traces and zero opening on two different normals; relaxing it
would require a stated physical/contact choice. Only one forward graph append
is admitted. Overloaded insertion is rejected, not converted to zero-cost damage.
The preliminary invalid overload and fresh/restart metadata mismatch are recorded
in the evidence. The older 65.4% early spatial failure is retained explicitly.

---

# Finite exact-jump insertion audit

The optional benchmark `cohesive_interface_native.cc` and focused
`cohesive_interface_insertion.inc` now use a finite installed jump subspace;
`movable_node` excludes absent/end-point variations, while installed intact
Gauss records retain bounded weak admission in `cohesive_interface_front.inc`.
Bulk FE/pressure and geometric lift operators remain fixed. The native manager
is rebuilt with actual finite geometry; old records map by immutable material
lattice indices, and virgin records inherit the accepted dimensionless aging
clock. Geometry swaps invalidate native associations. Generation-tagged trials,
rollback on failed validation, deterministic retry and strict restart configuration
checks protect ownership. No source/ or include/ production file changed.

The identical QP*N sum is cached before Newton evaluations; prior-front and
cached trajectories remain bitwise identical. The negative unrestricted release
mode has 93.66% support on tied nodes and is dominated by the retained bond
softening term; the admitted reduced tangent is positive. Independent LP, tangent,
full-stress/work, Gauss6 and monotonic Q1 threshold/inventory checks are linked in
[the report](COHESIVE_INTERFACE_INSERTION_REPORT.md). No particle dependency is
introduced: this isolated constant-coefficient native FE history family uses no
particle initialization, coefficients or mechanical feedback. General imported
history and pressure-space transfer remain outside scope. Prior audits follow.

---

# Weak-admission front audit

[Equations](COHESIVE_INTERFACE_FRONT_FORMULATION.md),
[results](COHESIVE_INTERFACE_FRONT_REPORT.md),
[future architecture](COHESIVE_INTERFACE_ARCHITECTURE_PROPOSAL.md).
Modified only the optional native fixture and added its focused front solver
include plus independent/run/check/analysis/plot scripts. Production source,
headers, CMake targets, calibration and BP3 are unchanged.

Intact equalities Q_T delta=b_T and bounded Gauss multipliers certify the complete
weak equation. A positive-formation predictor is accepted only with the full box
certificate, including virtual negative/alternating directions. Mature rates stay
free. The code does not clip physical stress or damage; projecting dual multipliers
is the actual intact strength constraint. Tied histories have zero jump/a exactly;
physical damage/age remain Gauss-owned. Current normal pressure and all derivative
terms are retained. Partial sign-changing Gauss ties require a general nullspace
and are explicitly outside this monotone fixture.

Stability excludes artificial tied rows and separately records release curvature;
no negative value occurred. The activating step exercises native owner rollback,
retry and one commit; actual simulator restart matches bitwise. This remains an
isolated component driver, not the production outer cutback/phase driver.
Original failed semismooth drafts/logs are preserved, not interpreted as physical
instability. The broad-zone/maturity result and early spatial limitations remain
explicit. The proposed geometric role for phi is not a production model change.

---

# Exact-jump native component audit

[Derivation](COHESIVE_INTERFACE_EXACT_FORMULATION.md) and
[results](COHESIVE_INTERFACE_EXACT_REPORT.md). Only the existing optional native
fixture is changed: CI_EXACT_JUMP defaults off, adds (H−Fχ)δ_h e_x with complete
regular strain/divergence and pressure coupling, duplicates aligned Q1 pressure
traces using native periodic constraints, and evaluates both physical tractions.
Native mixed residual, exact integrated displacement jump, generalized pressure
work and independent higher-order energy/trace integrals are checked. Source-only
operators agree with archived values to <9e−16 relative; no material/solver/source
production path or default changes. All protected prior source/evidence hashes
are checked, except the explicitly edited fixture whose original is archived.

Signed Δδ/Δt uses the existing experimental signed property. Restart saves the
representation flag; a mismatch rejects before accepted output changes. Actual
manager histories and bulk reduced memories restart bitwise. Added run/analysis/
plot scripts and compact evidence retain predeclared criteria and original failures.
The first draft's unsigned negative basis derivative was caught before publication,
corrected without loosening tolerance, and its source/log archived.

This is an aligned fixed-surface FE realization, not generic cut-cell enrichment,
MPI qualification, general DG-memory conversion or production Stokes integration.
The independently measured finite-width bias is corrected; short-wavelength FE
error and nonuniform intact admission remain distinct open issues. Earlier audit
entries follow unchanged.

---

# Native cohesive-interface component audit

[Equations/ownership](COHESIVE_INTERFACE_NATIVE_FORMULATION.md) and
[measured evidence](COHESIVE_INTERFACE_NATIVE_REPORT.md). Added the isolated
`reconstructed_fault/cohesive_interface.h/.cc` return and benchmark fixture,
parameter file, run/analysis/plot scripts, plus one opt-in CMake target. No old
solver, material, manager, checkpoint schema or default was changed.

The fixture uses actual native FE/constraints and a dedicated native manager,
with physical damage/age at Gauss3 surface points and separate signed velocity.
The old manager slip-rate API requires nonnegative velocity and is not used to
encode reversal. Actual ASPECT postprocessor save/load owns these new records;
geometry/operator caches rebuild. Exact bulk memory is restricted to the fixed
homogeneous geometry/load family. No particles are read.

Surface Gauss3 corrects the demonstrated nodal checkerboard without changing the
law or tolerances. Direct energy assembly avoids Schur-subtraction cancellation.
Independent native-QP stress/work and complete pressure-dependent derivatives
are verified. Local positive symmetric-part curvature is checked before commit;
it does not certify arbitrary nonsymmetric dynamics or nonuniform intact fronts.
The finite-width source has no literal displacement jump; its measured residual
7.42% sharp-interface discrepancy is an explicit remaining kinematic issue.
Protected entry hashes and exact source/binary provenance accompany the evidence.

---

# Cohesive-compliance / geometric-contact independent reference audit

See [derivation](COHESIVE_INTERFACE_FORMULATION.md) and
[measured report](COHESIVE_INTERFACE_REPORT.md). Five new Python files implement
the stateless bond/contact return, point/lifecycle checks, direction/mapping/symbol
checks, mixed-layer seeded reference and plots. They reuse retained RSF parameters
and state law, but change the fracture internal variable and energy deliberately.
No native API, owner, assembly, production parameter or BP3 source is changed.

The layer Schur operator retains both displacement components, incompressibility
and actual pressure response. Independent strain-energy and normal-traction checks
precede condensation. The complete nonsymmetric return tangent includes damage,
state and pressure derivatives. Increment-based Newton arithmetic fixes a measured
cancellation failure without loosening force tolerance. Exact even/odd kernel
construction removes FFT reciprocity roundoff without changing the operator.
The ledger retains signed cohesive endpoint error separately from physical loss.

Owner tests discard a trial, restore persistent serialized state, retry bitwise
and publish once. Loaded full-tensor mapping retains old records and rejects
unaccounted switch work. This does not qualify arbitrary old cross-band-age or
native checkpoint conversion. Gamma lifting is tested for normalization/work
adjoint only: native diffuse and sharp interface mechanics are not assumed equal.
Previous audit and original failed evidence follow unchanged.

---

# Structural-plane substantial-sliding / mixed / birth audit

See [equations](STRUCTURAL_SLIDING_FORMULATION.md) and
[results](STRUCTURAL_SLIDING_REPORT.md). Seven new independent Python scripts
reuse the protected structural return and equilibrated strip. Developed and
analytic-mature profiles keep all T/A/K histories through velocity changes.
The full mixed pencil retains actual normal pressure, three-dimensional
plane-strain tensors and implicit frictional state. An independent nonlinear
oblique laminate solves its nonsymmetric pressure/flow Jacobian jointly.
No native constitutive or checkpoint code is modified; the mature endpoint
reference does not remove the native phi<1 guard.

The existing scalar native intact multiplier was inspected. Its rate-independent
return cannot be copied into the new kinetic model without a new formation law.
The one-sided audit includes delta I, resolves phase down to 1e-24, and exposes
the missing exact-intact traction/admission rule. Phase-energy and normalization
variations are derived, but no old symmetric-Hessian certificate, latent history,
normalization floor, pressure lag, or added damping is silently adopted.
The report separates bounded continuum instability, numerical error and the
phase-admission constitutive gap. Prior audit content is preserved below.

---

# Structural-plane RSF continuation audit

Bounded independent task: [formulation](DIRECTIONAL_CONTINUATION_FORMULATION.md)
and [measured report](DIRECTIONAL_CONTINUATION_REPORT.md). Four new benchmark
Python files implement recorded-plane signed kinetics, actual normal pressure,
full Maxwell tensors, complete pressure/state tangent, conditional first-record
admission and a joint 1-D equilibrated band. No production source is edited.
The inherited angular maximizer is used only for new-plane admission/competing
plane diagnostics, independently checked with actual-plane angular optimization.
It no longer imposes other-plane inequalities during existing-plane sliding.

Geometry identity is explicit at fixed small strain; A/K are never projected
or reset. New records require unassigned zero A/K, retain T/Theta/phase/clock,
and have no standalone energy jump. Actual phase creation/second-plane birth
is not implemented. All tensor contractions include full plane-strain zz/xy.
The band reuses the existing exact 1-D mixed equilibrium reduction and lumped
P1 measure; constant pressure follows its symmetric solution, while point
checks exercise the nonzero pressure tangent. It does not replace a full 2-D
oblique stability analysis. Old symmetric Hessian certificates are not reused.

Source hashes, commands, test errors, unmatched original temporal criterion
and unresolved tiny work-trend result are retained in the evidence package.
95 protected baseline files remain unchanged; original audit content follows.

---

# Exact Feng endpoints and single-direction RSF/radiation audit

CONSTITUTIVE_RECONCILIATION_FORMULATION.md transcribes closed-contact Feng
71–81 and checks Appendix E's complete stationary equation independently.
The principal-asin shortcut misses a reversed-memory branch and is undefined
at a C1=0 mature limit; printed physical-shear versus relative-shear signs are
separate. Complete roots still require final global maximization. Fresh
pre-damage obstacles and refined Maxwell loading show an inadmissible
single-ray tangent at first symmetric activation while Y=0 and phase force
is +8 Pa. This is not a permission to average flows or erase tensor memory.

New independent reconcile_feng_rsf.py / check_feng_rsf.py / plot_feng_rsf.py
retain T/A/K conventions, full plane strain and the current fixed-phase
energy. The new kinetic candidate adds explicitly uncalibrated RSF/radiation,
with chi from normalized AT1 density and a stated coherent-profile assumption.
No arbitrary reference width, K hardening or spatial state operator is added.
This is not a replacement of native h/I normalization or intact initialization.

Fei DOI 10.1002/nag.70416 is the IN-PLANE paper: A14 uses previous-step normal
traction. Our old-stress/current-normal extension to free maximizing directions
is labeled an adaptation. Full mixed stress/history/state/velocity/pressure/
rate/angle pencils and coupled/lagged amplification maps are independently
verified. Prior pressure perturbations remain in the lagged right-hand side.
Positive kinetic slope regularizes smooth-branch evolution, not switching
across ties. Free-direction lagged endpoints fail even their lagged objective;
80-digit roots confirm this. Fixed-plane control success is reported separately.

Numerical reference corrections are invertible row/column equilibration of
both QZ matrices and complete-Jacobian relative FD measurement (including
zero columns). Original failed logs are retained; physical tolerances are
unchanged. See the report for measured errors, dt sensitivity and limitations.
No native source, production checkpoint, BP3 or previous evidence is changed.

# Feng actual-direction and dissipation reconciliation

FENG_DIRECTION_FORMULATION.md distinguishes the actual published trace-free
unit increment N, min-positive-pressure friction and global maximizing F from
the current sum of two conjugate systems. The explicitly mapped Maxwell law
and current h/phase energy are not claimed identical to Feng's elastic/phase
model. Production files are not edited. New independent feng_direction.py,
check_feng_direction.py and plot_feng_direction.py produce compact evidence.

The legacy pf-cpdi phase_field_rsf.cc selector was recovered at e3ef3ae89.
Its intended L21-L12 spin sign is observer-dependent; the literal 2-D helper
leaves w[0] NaN and its caller tests that component. Source excerpt, caller
angle and hashes are retained. The defect is not transferred to the new
reference, and no legacy backend is changed.

Angular piecewise global maximization includes pressure-order/sign boundaries.
Tie continuity acts only among true maxima in fixed small-strain coordinates;
passive basis changes rotate all tensors and the preferred director together.
The point corrector differentiates xi and angle, including physical-pressure
dependence. Seventeen identical archived inputs, independent algebraic roots,
coupled finite differences, BE power ledgers and full mixed pressure symbols
are checked. The old mature null is removed, but strict single-direction
localization persists; phase=.8 mixed coefficient -0.00195453 G, rate threshold
.76761128. Final global-max failures and multiple-return symmetry are explicit
rejections. This is not a native integration or propagation qualification.
See FENG_DIRECTION_REPORT.md for numbers, tolerances, scope and evidence.

# One explicit Pranger-inspired scalar-state feasibility audit

PRANGER_FEASIBILITY_FORMULATION.md records the exact T/A/K Maxwell mapping,
physical pn=p0+p'-n.tau.n, full plane-strain norm, both changing Coulomb
systems, asinh creep and independent scalar state. M_ex(q)=q+ell_r² Delta q
enters only the state row. The isotropic 2-D extension and uniform c1=c2=1 are
explicit assumptions, not the paper's localized calibration. The implicit
averaging inverse is never substituted. No K-gradient energy or extra damping
is added. The accepted total-K experiment is retained but retired for mature RSF.

New independent benchmark pranger_feasibility.py constructs the complete mixed
plane-wave pencil and its exact constrained elimination. check_pranger_feasibility.py
checks nonlinear coupled directional derivatives, full generalized QZ,
admissibility/contact, independent 80-digit angular roots and the constant-friction
rank limit. Pressure is eliminated through incompressible momentum, not fixed.
Passive accumulated histories do not alter the symbol. The mature null includes
recovered T/A memories and verifies their exact cancellation in physical tau.
plot_pranger_feasibility.py creates analytic/spectral figures, not a trajectory.
The archived one-system input is retained as a compact row with provenance;
the altered rate magnitude and all synthetic two-system states are labeled.

The scalar state/direct effect cannot see the conjugate-redistribution mode.
Mature oblique evolution is singular; finite-h reversed memory independently
has the denominator 1/h+(Qxy/tauxy)cos²(2alpha), with an angular pole also for
b=0. A benign steady control does not rescue uniform directional coercivity.
The candidate fails before the conditional strip/velocity-step implementation.
This is a failure of our two-conjugate adaptation, not a claim that the paper's
distinct flow law has the same defect. Existing constant-friction update still
rejects its separate three-root reversal; no native check is bypassed.

No production header/source, CMake, default, native checkpoint or BP3 artifact
is edited in this task. PRANGER_FEASIBILITY_REPORT.md and the compact archive
record measured bounds/timings, the missing directional closure and formal
phase/birth/intact-limit dependencies. Earlier source-audit entries follow unchanged.

# Mature K-gradient cross-band reduction and compatibility

MATURE_GRADIENT_FORMULATION.md derives interface moments, normalized profiles,
units, energetic microforce and its linear accumulated-slip resistance. The
current native mature surface specialization has C=Ecoh=0 and auxiliary memory;
constant total-K gradient storage does not disappear with Q in that limit.
References to Gurtin2003, Miehe2016, Pranger2022 and Fei phase-field RSF distinguish
energetic hardening from diffuse kinematics and spatial rate/state interaction.

No production code is changed. New benchmark mature_gradient_strip.py reuses
the qualified C++ projection/action and P1 mass/gradient structure, solving both
K components with shear traction and pressure reference jointly. It exactly
eliminates 1-D mixed mechanical equilibrium and compatibility, rather than
prescribing point strains. Physical p=p0+tau_yy-mean(tau_yy), changing plane
normals, both conjugates and pressure derivatives are retained. Source recurrence
and separate BE work/storage/dissipation checks use full plane-strain metrics.
The multivalued prescribed-X return is never called.

Fresh cases use the same continuum fixed phase and zero T/A/K/Q/L. The phi<1
API is respected: .999 core is near-mature, .02 surroundings are stronger
nearly intact material, not exact intact material or a new birth rule. Natural
zero external microtraction is enforced variationally; no internal K skin,
phase mask, direction lock, coefficient degradation or K reset is inserted.
Cohesive core/shoulder/exterior forces, gradient power and friction are separate.
The independent analytic local mechanical elimination, complete joint action,
limited mesh/dt comparisons and immutable retry checks pass; this is not native
publication/MPI/restart requalification. Prior lifecycle evidence remains intact.

MATURE_GRADIENT_REPORT.md records the selected 6 mm finite-slip failure and the
narrow approximate 1 mm window without relabeling it a mature limit. All 67
protected entry artifacts/sources are unchanged; status/audit history tails are
preserved. The old fixed-phase qualification remains valid within its scope.
Coupled phase–microforce progression is blocked by the demonstrated mature-law
compatibility issue. A rate/state spatial interaction is proposed for review,
not implemented or certified by the previous energetic-gradient tests.
Earlier audit entries follow unchanged.

# Experimental fixed-phase K-gradient microforce path

COULOMB_GRADIENT_LITERATURE.md records the published antecedent, energy/variable
mapping and ten additional assumptions before implementation. The exact weak
form, boundary power, BE remainder and complete nonsymmetric tangent are in
COULOMB_GRADIENT_FORMULATION.md; measured results are in COULOMB_GRADIENT_REPORT.md.

bulk_cohesive_gradient.{h,cc} owns the convex lens projection, complete action,
joint given-K point evaluation and fixed Cartesian continuous bilinear K space.
K nodes are native Gauss6 history points; exact gradient stiffness, native JxW
mass and linear boundary extrapolation preserve constants/affine/xy fields.
No local prescribed-X root is chosen: that inverse has three saved admissible
roots. Physical normal traction always uses tau, never tau-Q-X. Microfree
boundaries are explicit. No dilation, pressure clipping or new viscosity is added.

BulkCohesiveState retains existing DGQ2 T/native FE remainder and QP A/K/Q/W/L
owners. The benchmark-only positive length is captured once, included in the
checkpoint material tuple and paired with a derived immutable geometry cache.
The history serializer and all existing constitutive source bodies remain
unchanged. Restart rebuilds the cache. Old/new energetic states are not converted;
zero length retains the exact old tuple, law and outputs.

bulk_cohesive_phase.cc assembles the actual joint u/p/K normal-map system with
full pressure/direction dependence and native mean-pressure constraints. It
keeps Gauss6 mechanics, source3, full plane strain, all nine DG modes and the
independent source-publication check. Gradient storage/BE loss are added once;
physical friction is (tau-Q-X):W. Exact sticking roundoff cleanup occurs only
after complete convergence, followed by all residual checks again. Fresh first
iteration may use a fixed-K elastic lifting solely as a Newton guess. Final
mechanical/source/energy/contact/dissipation checks and immutable-parent pending
publication remain. Fixed-phase serial Cartesian eligibility is explicit;
phase-active use and the old symmetric Hessian remain disallowed.

Benchmark controls independently verify spatial energy/action, both projection
branches/conjugates, actual mixed wavelength symbols, forward strip response,
three native meshes, timestep sensitivity, homogeneous identity, rollback,
publication/restart and defaults. Actual Schur conditioning is measured after
sparse K elimination, not a frozen-pressure surrogate or a stability certificate.
Two original native failures, the nonunique point inverse and all earlier
scientific limitations remain. No new particle/domain dependency is introduced;
fixed diagnostic particles and existing native ownership checks are retained.
Earlier audit entries follow unchanged.

# Coulomb mixed mechanical localization audit

COULOMB_LOCALIZATION_REPORT.md and COULOMB_LOCALIZATION_FORMULATION.md audit
the actual nonsymmetric mixed tangent before any phase reduction. Native
CR::derivative strain/pressure/phase columns include changing plane directions;
the independent active-plane continuous-rate construction retains full 3D
plane strain. Rank loss of [Q,b;n^T,0], with b=(tau_p-Id)n, occurs in admissible
developed-damage states even when fixed-pressure stiffness is positive.
An independent closed single-system formula gives hcrit=100.4981374 and
phi_crit=.7676112793 at the retained m=4,mu=.1. Double-system sampled states
also lose at deeper damage. Real 3x3 null vectors are checked, not inferred
from the symmetric part. The phase-gradient correction vanishes at high k;
delta phi=0 is permitted by the strict physical obstacle force. Local Maxwell
relaxation supplies no slip-gradient control. No nonsymmetric phase-active
acceptance rule is claimed to cure this constitutive prerequisite failure.

bulk_cohesive_phase.cc adds only ASPECT_COULOMB_LOCALIZATION-gated read-only
exports on the already selected fixed_coulomb route: actual condensed matrix,
free-coordinate indices and cached native point derivatives. The exporter
reconstructs immutable incoming history and reuses the actual response;
converged trial labels require a matching accepted ledger row before analysis.
No alternate physical return, tangent assembly, residual or publication is
introduced. Existing pressure AffineConstraints are applied once. Independent
native FE matrix integration matches the exported operator to 1.06e-15.

The existing Coulomb block fixture alone admits a phase override, local basis
map and pending-step clamp on resume, without adding production parameters.
Spatial controls initialize uniform physical phase/zero histories freshly.
Full/half steps, actual Gauss weights, Riesz-scaled mixed singular values and
slip participation widths are compared; conditioning deteriorates but no
collapsing band is observed during the bounded window. Both fine full-step
initial-lift contact failures and unchanged-parent cutback controls are kept
separate from continuum localization. The strict checks remain intact.

All 60 protected hashes, including material bodies, prior archives, BP3 and
old executables, are unchanged. A retained native trajectory matches 33 state,
mean and phase files bitwise. History ownership, once-only publication,
stationary particle fingerprints and source checks remain active. The phase
guard is retained. Remedy derivations compare a physical energetic K-gradient
with a new dilatant volume/history/continuity law; neither is implemented.
Entry deltas/provenance and compact evidence are archived separately from full
native runs/checkpoints. Older audit entries below retain their scope.

# Confined Coulomb native fixed-phase integration

COULOMB_NATIVE_SELECTION.md fixes the physical experiment; COULOMB_NATIVE_REPORT.md records measured qualification and limitations. bulk_cohesive_coulomb.{h,cc} promotes the identical qualified point body; benchmark compatibility includes reuse it rather than maintaining another law. The new explicit BulkFlowRule is closed Coulomb fixed phase, default off, restricted to bulk-only fixed phase. G/eta/Gc/ell/m and strength-drop input are unchanged; c_r=0 is distinct. No mature/reconstructed-surface/BP3 route is enabled by this selection.

bulk_cohesive_phase.cc assembles full nonsymmetric strain and pressure actions with p'/G scaling, including all normal-traction and changing-direction derivatives. A physical zero-mean perturbation-pressure AffineConstraint is applied once with C^T residual and C^T J C. Total weak stress/reactions use p0+p'; initialization has tau=0 and sigma=-p0Id. The pressure-independent branches retain their original arithmetic and pass a bitwise native comparison. Point contact errors are communicated collectively with CellId, quadrature index, reference/physical position and total pressure; rejected line-search candidates cannot publish. Cohesive point caches allocate only in the new mode.

The actual DG/native source and quadrature history owners are reused. The distinct checkpoint tag/material record includes p0/mu/c_r; existing signatures/layouts stay unchanged. Core checkpoint critical-parameter validation independently refuses changed Surface pressure before restore. Native rejection leaves accepted solution/histories intact; same-rank 1/4-rank restarts reproduce 48 files bitwise per case. No archived P5c history is converted. All phase values remain exactly .2. Because no phase directions exist in this mode, the old symmetric phase Hessian is not applied; it remains mandatory on its original phase-active modes. Enabling new-law phase evolution is explicitly rejected, rather than bypassing that safeguard.

The native fixture reuses the distributed MomentCycle observer under a test-only initialization/output selection. It exports actual Gauss6 pressures/stress/history, decomposes accepted W into conjugate plane speeds and uses total-stress nodal weak reactions, without interpolating histories into face constitutive states. Isotropic undefined directions carry flags/NaN angles. Stationary particle ID/coordinate fingerprints, unique cell ownership and once-only publication remain checked. The 576 actual mechanical points satisfy trial/returned compression; native source discrepancy remains below 1e-12.

The original FD failure is preserved: a nonzero-mean pressure direction, after applying the mean constraint, crossed yield sets during the hold. Corrected physical mean-zero probes check active-system consistency without changing error bounds. Original filename-glob and overstrict pointwise comparison failures are preserved too. Homogeneous physical means/storage meet 1e-12; local field spread 1.04e-9 in the reported resistance scaling is separate, not relabeled as pointwise identity or used to loosen native/source checks. Work/dissipation identities and numerical remainders are separate; a BE ledger pass is not physical-energy convergence. All 51 protected entry source/artifact hashes remain unchanged. The proposed nonsymmetric reduced-VI acceptance needs separate phase-active verification; no such task is begun.

# Coulomb direction-selection qualification

Actual bulk caller source, not legacy parameter names, governs the audit. source/simulator/solver/bulk_cohesive_phase.cc passes S0 only for fixed_tensor and nullptr for radial_tensor; source/material_model/phase_field_fault/bulk_cohesive_tensor.cc then projects into the complete in-plane trace-free space. Scalar xy projections are diagnostic. Bulk eligibility in phase_field_fault.cc excludes mature/reconstructed-surface paths. Normal-lock thresholds are not consulted in the bulk return, and all physical phase masters remain available above their accepted obstacle. Out-of-plane flow is excluded by the established constitutive space, not a particular in-plane direction lock.

bulk_cohesive_history.cc initializes R from mu*surface_pressure, whereas the actual mechanical weak stress is tau-p_solution Id. No background surface pressure term is added to bulk weak loads. The new public SimulatorAccess/FEValues restore observer reads pressure at native Gauss8 points after full owner restoration, before the existing termination manager exits. It changes no requested timestep, state, particle, constraint, cache or checkpoint. Four 4-rank reads terminate without a physical step; source/copy hashes remain identical.

Archived directions use actual physical cell means with pressure averaged by the same quadrature. Principal axes are in-plane axes of mean C=-sigma, sigma=tau-p_mech Id. Coulomb glyphs are bare-stress candidates (Q=0 diagnostic), not a constitutive replay on averaged histories. Cyan net-flow labels are an equivalent signed shear pair, omitted when mean W is zero. Hypothetical confinement is labeled and not applied or reequilibrated. The existing mechanical field is not silently identified with the prescribed friction pressure.

The new implementation lives under benchmarks/.../bulk_cohesive/coulomb_point.{h,cc}; no native source/interface/parameter/checkpoint changes. It reuses the retained tensor state, compliance, rate/spin convention and work accounting. Normal pressure uses physical traction. Non-associated tangential systems enforce all orientation inequalities, including two active conjugates, without rotating old A/K or adding volume/opening. Closed-contact domain violations and nonunique/absent admissible roots fail explicitly. A separate backstress-release control also requires closed returned stress when the trial is closed, including the zero-friction identity path. The original net L meaning is retained; plane-slip sum and new friction power are separate. The constant-R radial control is unchanged and recovered exactly at mu0,c_r=R.

The derivation in COULOMB_DIRECTION_FORMULATION.md precedes implementation. Complete derivatives include strain, phase, pressure, plane directions, multipliers, tractions, yield margins and system slip. Independent trigonometric equations use 70-digit arithmetic, separate from the quadratic candidate code. Existing scaled identity/FD bounds remain 1e-12/1e-6. Point history discard/retry and file restart preserve every tensor; no production lifecycle adoption is claimed. Zero-dilation Coulomb response has measured nonsymmetric tangent, so the old potential/Hessian certificate must remain restricted to its retained law.

Point and archive checks pass; no constitutive failure or tolerance relaxation occurred. An initial analysis JSON NumPy-scalar serialization issue and an old-control invocation missing its output environment variable were setup issues, corrected without physics changes. Original P4/P5c/BP3 evidence and 52 entry source/artifact hashes remain. Native integration requires explicit stress/traction-reference, calibration and stability decisions; this task does not select a new confined specimen or opening law.

# P5c extended continuation closeout

[P5C_CONTINUATION_REPORT.md](P5C_CONTINUATION_REPORT.md) records measured connected extension, local timestep checks, costs and limitations. Native equations/source/binary remain byte-identical to the selected P5c baseline. The benchmark-only plugin reuses the existing native observer and extends only the end of the same.018/s grip motion, retaining histories/clocks. It does not integrate physics in Python, reset phase, alter viscosity, mask growth or change a production parameter.

The read-only event observer uses actual accepted Q1 phase. Initial support is the union of cells with any nonzero initial vertex; newly activated cells have four exactly-zero initial vertices. Four-neighbor connectivity and horizontal Q1 crossings are fixed diagnostics. Native collective user-request termination handles event stopping. The selected radial endpoint is177/.170 s; fixed stops176/.169375 s, one step after its own target request. No remote/boundary event occurs.

An interactive interruption killed the first observer while native MPI continued to185/.175 s. Extra states and their positive1014-master certificate are preserved as diagnostics. Recovery from immutable153/.155 s reproduces96 phase/physical-history-mean/weak-force/reaction exports bitwise and restores the pre-rapid/event checkpoint. Subsequent controller lifetime is independent of the interactive turn. No accepted state is synthesized from CSV. All protected checkpoint file hashes are retained for verification.

The main tensor/phase histories advance once per accepted native interval, with all existing source, mechanical/KKT, dissipation, BE and mechanically relaxed stability checks. The campaign needs no stability retry; inherited rejection qualification is not counted as a new test. Four local replays use the same incoming native histories, halve only the pending interval, and retain their physical load/hold endpoints. Common comparisons use physical time, not publication index; no mirror/branch alignment is performed.

Postprocessing uses actual Gauss8 physical cell integrals and independent reaction-displacement work. Full-plane-strain tensor differences use xx²+yy²+(xx+yy)²+2xy²; cell-mean stress comparisons are labeled as such. Exact Q1 norms use Gauss2, normalized by growth since common incoming state. Native mechanical/fracture weak gradients share their lumped Q1 test mass and are distinct from box-averaged Y. The inherited outside-support growth label is corrected to avoid calling connected extension remote nucleation. CSV export admits missing initialization energy terms.

Replay target suppression is corrected to retain simultaneous remote/boundary safety flags, with explicit synthetic priority controls. All48 replay states have zero such flags, so the correction changes none of these trajectories. The supplemental safety observer launched after the runs and exited; no live MPI safety-trigger qualification is claimed. Primary event stopping itself is exercised. All native commands have finished.

Original P4 temporal/spatial/branch/boundary failures and strict_reference_match_passed=false remain. Unlimited DGQ2, all history modes, Gauss8/source3, constitutive conventions, tolerances,1024 capacity,41 protected/BP3 hashes and earlier evidence remain unchanged. Full native archives/checkpoints are separate from the compact PNG/GIF/CSV/JSON review package. No continuation beyond the selected experiment or new spatial campaign is authorized by this closeout.

# P5c targeted axial refinement — 22 October 2026

[P5C_AXIAL_REPORT.md](P5C_AXIAL_REPORT.md) records the completed selected comparison.
No native source, plugin, law, history layout, limiter, default or acceptance
change. The same P5c executable/plugin is reused, with exact hashes verified.
Resolved ParameterHandler values differ only in X repetitions, output path and
the prefix termination step. Fresh continuum initialization/zero histories are
used on each mesh; no accepted field or quadrature history is interpolated.
Native same-mesh restart from publication95 switches the pending timestep only.
Actual endpoint archives contain publications95 and137 at the expected times.

New benchmark-only helpers prepare inputs/checkpoint copies, compute the
comparison and package compact evidence. Consistent Q1 growth norms use exact
Gauss2 integration; growth fields are compared on common physical points after
subtracting each mesh's own initial field. Stress/Y/W exports are actual native
Gauss8 physical means, not laws evaluated on interpolated/averaged histories.
Common force kernels are physical coarse-Q1 hats, exactly representable on both
grids. They pull back complete native mechanical/fracture gradients once;
independent AT1 integration agrees within4.44e-16 relative to Kf. Native nodal
KKT tests remain distinct because their supports change under refinement.

Refined fixed/radial source/mechanics/KKT/BE and relaxed-stability gates pass,
with once-only publication and fixed particle ID/coordinate fingerprints. All41
protected hashes and the original P5c ZIP remain unchanged. No new rejection or
cross-rank restart qualification is claimed. The accepted-state restart is
actually exercised at the scheduled split. One unloaded launch with a missing
freeze environment flag was stopped, preserved and excluded before a fresh run.

Radial driving/growth reduction survives with changed magnitude; earlier onsets
and28.68%/37.97% growth-field differences leave spatial accuracy unresolved.
Small extra radial flank growth is reported, not hidden as uniform suppression.
A short loading extension is recommended only. P4 failures/BP3 are preserved.

# P5c closeout

[Measured report](P5C_REPORT.md): all staged checks and both fresh seeded trajectories plus focused halving are complete. Radial cap growth is35.83% lower at the coarse endpoint; growth-relative temporal differences are at most3.19% and0.231% at the radial endpoint. Neither case activates initially intact material or advances developed-damage fronts. Minimum nonempty mechanically relaxed curvature41.7597; at most7 masters versus the unchanged1024 cap. Every native acceptance/publication check passes and final checkpoints preserve all four endpoint states. The41 audited protected files remain unchanged. This qualifies an early fixed-grid finite-body comparison, not P4 spatial/propagation accuracy. Earlier entries below retain historical scope.

# P5c native tensor uniform-MPI extension — 22 October 2026

The selected P4 checkpoint itself (step127, q8, pub127, t=.15875) serializes
G=1e6, eta=1e8, m=4, R=1000. Resolved parameters and fracture-parameters.csv
agree: fracture.prm overrides native.prm with cohesion2000/Gc1/ell.046875;
Kf=Gc/((8/3)ell)=8 and m=Kf/(cohesion²/(2G))=4. P5b's m16 attribution
was an audit error, not a different selected scientific baseline. The original
P5b archive and separate mismatched preliminary run remain untouched.

Only the tensor parser's serial restriction is removed. The native distributed
bulk solver retains its phase-coordinate identity guard (no distributed hanging
phase), homogeneous matched-coefficient, fixed mesh/geometry/no transport rules.
No constitutive, residual/tangent, projection, pressure, tolerance, stability
limit or checkpoint-record implementation changes. All 27 DG modes, native FE
source complement and A/K/Q/W/L stay on existing uniquely owned CellIds.

The reused benchmark observer now serializes all 25 tensor point-history fields
for independent comparisons and rejection fingerprints, and evaluates physical
tau=T-2G A and Y=h' Q:Q/(4G), full plane-strain energy and weak boundary reactions.
Its scalar arithmetic remains selected as before. Tensor cell averages are
authoritative; the old scalar dense-QP table is suppressed in tensor mode.
A virtual initialization-value hook permits the isolated exact-zero/nonuniform
small controls; specimen initialization still uses the existing continuum seed.
The new P5c plugin supplies only test loading/initialization and observation.

One/four-rank tests compare 65 physical states including every DG coefficient,
native FE memory coefficient and every point tensor component. Existing native
publication checks cover owned quadrature points with the derived ghosted FE
source; checkpoint reload rebuilds that cache. Ownership totals and duplicate
CellId checks pass. Actual rejected trials/retries leave all accepted owners and
solution unchanged. Same-rank restart from publication63 reproduces publication64
bitwise in fixed/radial modes on both rank counts. Cross-rank restart, hanging
MPI, AMR, transport and BP3 tensor mode are not qualified by these checks.

See P5C_SPECIMEN_SELECTION.md and evidence/p5c_native for stage results.

# P5b native tensor return integration — 22 October 2026

`Bulk cohesive flow rule` is a bulk-only experimental selection with unchanged
`fixed scalar` default and explicit `fixed tensor`/`in-plane radial tensor`
variants. Native tensor advancement is guarded to serial. Existing fixed-mesh,
constant/matched-coefficient, no-RSF/no-transport/no-moving-mesh guards remain.
The stateless P5a law is unchanged. T remains DGQ2 plus native FE source remainder;
only CellId/mechanical-QP cohesive A/K/Q/W/L are added to the existing owner.
There is no surface Z or quadrature-owned physical phase in this bulk model.

Actual residual/tangent assembly uses full tensor stress/Y and all P5a current
strain/phase derivatives, including dE. The scalar branch retains its original
arithmetic. Full tensor Q:Q/(4G), rather than one xy component, enters phase work
and energy remainders. Source projection, native-gradient calculation, nine DG
modes and 1e-12 independent publication check remain. Every trial anchors one
accepted parent; the existing outer callback accepts once or restores solution
and all histories. Native projection is a derived cache rebuilt after restart.

The potential is the minimized point F plus FE fracture minus beta*p div(du).
Row factors beta*G and Kf recover its actual mixed Hessian. AffineConstraints
apply C^T residual/Hessian once; only independent masters enter KKT/stability.
The old FD diagnostic mistakenly included artificial hanging-row diagonals;
its row restriction now matches the mathematical action. This does not change
ordinary acceptance or Newton equations. Raw action/constraint exports are
test-only. Independent C reconstructions, complete native FD and the actual
mechanics/pressure Schur complement pass; the latter has a nonzero correction
on nonuniform admissible directions even though the smallest mode is uniform.
The existing bounded local stability gate is reused, not a global theorem.

Experimental fixed/radial tensor cell-history v1 signatures preserve all six
components of A/K/Q/W plus L, physical cell/phase anchors, DG/native coefficients,
clock, publication count, coefficients and quadrature. Observers that include
bulk_cohesive_internal.h must be rebuilt against this executable; old scalar
observers/generic particle stress output are not authoritative for tensor modes.
Scalar v2/v3 byte layout
and mature production checkpoint paths are unchanged. Same-mode replay passes;
mode/layout mismatches fail before publication. No checkpoint conversion exists.

Input audit: native m=Kf/(cohesion^2/(2G)). The P5a m=4 integration control needs
cohesion=2000 Pa at Kf=8 Pa/G=1e6 Pa. Correction at P5c entry: retained P4 fracture.prm also overrides cohesion=2000 Pa and has m=4. The former m=16 attribution missed that override; the separate excluded preliminary m=16 run is retained. The uniform
obstacle/control is not a resolved diffuse seed. Radial dt=.1 globalization
failure, unmatched m=16 input, old artificial-row failure and end-cap probe
assumption remain in evidence. Only dt=.0125 is qualified; no tolerances change.
See P5B_NATIVE_REPORT/SELECTION and evidence/p5b_native. Protected P4/P5a/BP3
artifacts retain entry hashes; no physical directional specimen is run.

# P5a selected tensor point law — 22 October 2026

Only the new `bulk_cohesive_tensor.h/.cc` stateless API and standalone point
fixture select the approved tensor return. The old `BulkCohesive::update` and
all native solver/history/checkpoint sources remain unchanged. No native model
parameter or production checkpoint format is introduced. The point CMake target
reuses the existing scalar Parameters/compliance and deal.II/Boost infrastructure.

T is the full plane-strain deviator. Cohesive A, accumulated K and multiplier Q
are physical in-plane trace-free tensors; L is accumulated engineering strain.
They are not a scalar times a newly chosen direction. With X=P(Tplus),
V=hX-2G(1+h)beta An and v=||V||_3D/sqrt(2), stick at v<=Rh. Otherwise use
E=V/v, Q=(X-RE)/(1+h), A=hQ/(2G) and Delta A=(v-Rh)E/[2G(1+h)].
W=Delta A/(beta dt), K+=dt W, L+=dt sqrt(2)||W||_3D; tau=Tplus-2GA.
This is the selected new isotropic shear-flow law, not a variable transformation
of unrestricted fixed-xy flow. Restricting all memory/increments to span(S0)
recovers the scalar law; the full tensor metric includes both xy contributions.

The derivative uses dE=[dV-E(E:dV)/2]/v and all h/phase terms; both current strain
and phase actions include constitutive direction response. At h=0 A/W vanish,
Q is the constrained multiplier and the right phase derivative supplies finite
Y=h'Q:Q/(4G). Yield equality selects the sticking generalized Jacobian. No
phase floor or persistent director exists. The control S may be sign-relabelled
or transformed with a passive basis; no objective-rate/finite-rotation claim.

Point BE accounting uses full tau/Q tensor norms, with the same stress,
traction and compliance remainders as the scalar law. Energy is per bulk
volume, not surface or specimen energy. Accepted incoming phase is immutable
under differentiation; no fictitious interpolated history is used.

The fixture owns T/A/K/Q/phi/L, counters and work sums only. Trials read the same
accepted state, rejection leaves it unchanged, stale/duplicate commits fail,
and file restart compares actual components independently of the serialization
list. A populated record tests full Tzz and both independent A/K/Q components.
The fixture Boost tag is separate from every native format. Two 800-step paths
and independent point minimization pass; see P5A_POINT_REPORT/SELECTION and
particle_free/evidence/p5a_point/verification.json for numbers/commands/hashes.

Preserved development failures: initial fixed-floor comparison omitted the
predeclared max(reference,scale), then was corrected; the independent energy
probe at epsilon=.001 failed 1e-7 and one halving gives second-order convergence;
a Python analysis initially used ||W|| instead of sqrt(2)||W|| for engineering
speed. These were harness/reference corrections. A final zero-load test at
phi=1e-303 exposed 2G/h overflow before multiplying zero. Multiplication now
precedes division; hypot and a normalized direction retain representable tiny
flow/derivatives, and an exact component-zero test prevents erasing a tiny
forbidden intact memory. Zero and phi=1e-200 flow controls pass. These source
arithmetic corrections add no phase floor, clipping or constitutive change;
all verification tolerances remain unchanged.

P5a completes material-point qualification only. P5b must reuse the independent
DG/native-FE T owner while separately adding cohesive tensors on the current
quadrature layout, with versioned experimental restart, complete coupled
residual/tangent and relaxed stability checks. Do not reuse native displacement
memory as cohesive A, duplicate the physical phase owner, project nonlinear
stress into T, or silently convert loaded scalar checkpoints. P4 failures and
protected BP3 remain; no full directional specimen is run.

# Bulk continuation/publication stability remedy — 21 October 2026

`solve_bulk_cohesive_phase()` now selects a loading-aware accepted-state guess
for changed/zero prescribed velocities, leaving all coupled unknowns free. The
unchanged nonzero-loading predictor is retained. Test-only
ASPECT_BULK_TEST_PREDICTOR reproduces the former guess but cannot disable the
new gate. The old diagnostic ASPECT_BULK_RELAXATION_GUESS remains reproducible.
No new production parameter or mature/legacy default is introduced.

`bulk_stability.h` uses the actual already-condensed tangent with beta*G and
Kf physical row factors. It eliminates the complete mixed mechanics/pressure
block with native UMFPACK, assembles a bounded dense critical phase Schur
operator, and uses deal.II LAPACK for its full spectrum. Critical masters are
strictly interior or unresolved at the existing 1e-7 phase multiplier tolerance;
Kf*C^T mass is the metric. No artificial row, pressure saddle eigenvalue or second
C^T reduction enters. Expanded physical directions establish feasible amplitude
before negative curvature is labelled an instability. Strict active obstacles
exclude phase directions; fixed-phase incompressible mechanics is positive.

Both actual cross actions are retained. The first entrywise symmetry check
falsely rejected cancelled entries, safely recovering before publication; it
and its source/binary are preserved. The final norm-relative cross check is the
pre-existing independent-analysis 1e-11 bound, measured 2.03e-15/2.52e-15. The
Schur asymmetry, minimum-mode residual, mixed solve residual and floating-point
margin are exported. This is a numerical local test, not a rigorous enclosure
of an arbitrary ill-conditioned inverse or a global/nonsmooth stability theorem.
An unresolved check or >1024 critical masters prevents publication. There is
one unchanged-parent accepted-state retry, then a useful stop if the gate fails.

The gate precedes construction of `next`, source projection and the existing
outer pending-publication callback. It mutates no accepted history. Four-rank
forced-predictor rejection recovers with bitwise identical complete T/native-FE/
a/k/L/q/rate/phase, stress means, weak forces and reactions. The existing outer
rejection fixture is replayable after a checkpoint via test-only
ASPECT_BULK_TEST_RECOVERY; two discarded trials leave owners unchanged, retry is
deterministic, publication is once, and split/restart is bitwise identical.
The preceding loaded state remains bitwise identical and grows genuinely.

See BULK_STABILITY_REPORT/SELECTION and evidence/bulk_stability/verification.json
for hashes, commands, checks and costs. Native gates remain 1e-9 mechanics,
1e-7 KKT, 1e-12 independent source and scaled 1e-10 BE ledger. Point-law and
history-owner sources, P4 archive and protected BP3 hashes are unchanged.
Curvatures match the independent P4 reference to <4e-13. Full checks cost 85–88 s;
strict-obstacle checks <1 ms. No scalable or new hanging-node claim follows.
P4 temporal/spatial/boundary failures remain; P5 is a proposal for scientific review.

# Physical fracture scales / propagation specimen — 21 October 2026

The current task audits and selects benchmark inputs, not a constitutive rewrite.
`bulk_cohesive_history.cc::initialize_bulk_cohesive_history` sets
Kf=Gc/[(8/3)ell], m=2G Kf/C² and R=mu p_N. Thus the bulk option's configured
cohesion C is the peak-to-residual drop: tau_p=R+C. Parameter declarations give
G/C in Pa, eta in Pa s and Gc in J/m²; phase ell is m. Bulk a/k/phi/h are
dimensionless, w is engineering shear rate, I=int h dy is m and profile Z is
Pa m. This I is a compliance integral, not the diffuse width. For p=1 the
phase-handler guard g''(0)>0 requires m>3. The first m=3 startup was rejected
before initialization; selecting ell=.046875 gives m=4 at the same Gc/strength.
No guard or tolerance is weakened. Original rejected inputs/logs are retained.

The retained Lc=26.6667 m and slip scale=.0266667 m dwarf its 1.5x1 m specimen
and .00225 m grip motion. The new selected values are .25 m and .0005 m.
The 1-D full-band AT1 normalization integrates to (8/3)ell, so Gc normalizes
fracture storage, not total frictional/viscous traction-slip work. The .6
pre-existing profile already stores .5467498155 Gc per straight unit length.
Stationary-profile excess work tends to Gc but need not be a realizable
irreversible Coulomb history. The independent constrained strip is checked
against the actual unchanged C++ point update and includes all BE remainder
terms; its extended-load nonconvex continuation stop remains evidence.

`propagation.cc` reuses the qualified `parallel.cc` owner/diagnostic path via a
fixture-only seed-parameter hook, whose original constants remain the default.
It reads actual homogeneous coefficients, constructs the native profile afresh
and uses endpoint x=1 m. No constitutive/source/acceptance/core binary changes
are needed. Physical tau is the complete DG/native-FE T minus 2GaS; full
plane-strain storage includes zz and both xy contributions. No stress-magnitude
surrogate, projected local history, transport or spatial activation gate is
introduced. See BULK_FRACTURE_SELECTION.md and BULK_FRACTURE_REPORT.md for the
single bounded trajectory, physical averaging definitions and limits.

The new trajectory completes .185 s with all native guards. Its first growth
is centerline-led; low/mid-phase contours extend as a connected softening zone,
while the .5 core advances only 1.005 mm. The exact BE/work ledger is retained
separately from unqualified temporal response. An independent first-hold check
uses original QP histories and the native return with zero increment: full
stress/pressure scale by beta, positive-phase a/q scale by beta, and intact q
uses the actual relaxed return (it is not blindly scaled). The resulting
physical phase gaps are positive, minimum 2.86881e-6 scaled, so a no-growth KKT
candidate coexists with the native growing hold result. No history is published
by this diagnostic and no branch/jump/stability rule is adopted. This newly
measured limitation is preserved alongside the earlier branch/spatial findings.

# Bulk MPI / horizontal-side-release audit complete — 21 October 2026

Entry and old-source identities: build-particle-free/bulk-parallel/entry.json.
The stateless bulk_cohesive.cc constitutive law is byte-identical. Only the
selected bulk owner/solver and read-only fixture paths are extended. Owned cells
assemble raw R/H once; a collective sum precedes complete native constraint
condensation. Phase duals remain C^T g, tangents C^T H C; already-reduced fracture
entries are not reduced again. Newton coefficients remain replicated, direct
UMFPACK is on rank zero, and the correction/error decision is broadcast. This
is bounded MPI parallel assembly, not a scalable distributed factorization.

The actual normal-velocity component constraints determine whether pressure has
a constant null mode. Closed controls retain a gauge; lateral ux release removes
it. Mechanical p is distinct from fixed friction p_N=10000 Pa. Natural horizontal
traction is zero, lateral uy is restrained, and only prescribed components enter
reaction work. Initial ASPECT adiabatic p is a guess, not an imposed side traction;
the first unloaded accepted solve equilibrates it. Bulk history publication is
still deferred to the existing outer callback. All fixed-mesh owners are saved
as unique physical CellId records in v3; native FE ghost values and the derived
projection rebuild after restore. The old v2 serial archive remains readable.

The first MPI setup failure exposed an inherited, unused dense surface-
normalization sparsity extension. Excluding bulk mode fixes the nonlocal-row
insertion without changing its residual. Ordinary FE fracture optionally emits
sparse raw physical entries, retaining the old return type/API. The 1/2/4-rank
fresh and restart/rejection checks pass. The sole four-rank 96x128 boundary
control reaches .185 s with 148 publications and all original gates. See
BULK_PARALLEL_REPORT.md, BULK_PARALLEL_SELECTION.md and their evidence for commands,
operator/owner errors, costs and preserved failures. The final parameter-help correction changes no equations and reproduces a
growing endpoint bitwise. Numerical-run source/binary identities remain archived
separately. Physical stress agrees across 1/2/4 ranks to 4.07e-13 relative;
independent saved-state reconstruction agrees with native dense stress to
1.55e-11 Pa. Native weak forces and full-tensor storage are independently
reassembled from original Gauss8 values. No mature BP3 operator, physical-law/
default, limiter, AMR, transport or geometry change is selected.

---

# One exploratory continued-loading schedule — 21 October 2026

`bulk_cohesive/continued.cc` builds a separate diagnostic DSO, reusing the native
energy observer and all actual solver/history owners. Its compile-time fixture
selection changes only the prescribed shear stop from .15 to .175 s; old
fixtures still select .15. The new parameter input inherits `spatial_refined`
and changes the DSO/output and end time .185. Fresh initialization is necessary:
retained native v2 checkpoints contain steps 123/127/128 after the old hold
began. No histories are reconstructed into a new native initial state.

Extra exports are original-QP local slip strain rate from .12 s and complete
DG/native-FE/local/phase owners at each accepted endpoint from .15 s. The
recurrence is w=((1+Delta t G/eta)a_n-a_(n-1))/Delta t in sliding; sticking
sets w exactly zero. Signed and absolute shear memories remain independent
persistent owners. A read-only bounded helper archives completed native rolling
checkpoints via their last-good pointer. It neither mutates state nor changes
checkpoint storage/restart. Constitutive/source/check/publication code and the
native core are unchanged. See [pre-run scope](BULK_CONTINUED_SELECTION.md).

The trajectory closes at .185 s with 148 publications; the whole prefix through
.15 is identical to the retained 96x128 case, including full physical-stress
exports. `assess_continued.py` reuses the actual cell/Gauss8 layout and physical
Q1 evaluation. It computes phase superlevel connectivity on 8 subintervals/cell,
checks 16 at appearances/joins/final state, and solves first outward 45-degree
crossings against the Q1 polynomial with bracketing. These are diagnostic field
evaluations, not native refinement or constitutive-history interpolation. The
independent native-node activation graph is explicitly a different diagnostic
from finite-level contour connectivity. All stress norms retain
xx^2+yy^2+(xx+yy)^2+2xy^2.

Independent work uses actual unconstrained boundary forces and displacement
increments: W_BE=F_n.du_n, W_tr=(F_n+F_(n-1)).du_n/2. All native BE numerical
terms, phase/mechanical/pressure residual work, physical storage/dissipation,
and signed/absolute external remainders remain separate. For fixed phi and
homogeneous Maxwell coefficients, the hold admits tau_j=(product beta)tau_140,
beta=1/(1+Delta t G/eta), w=0 and unchanged signed/absolute shear memory. The
measured hold matches this full-tensor stress to 1.50e-15 relative L2 and zero
phase/slip change. Native source tolerance 1e-12, mechanics 1e-9 and phase KKT
1e-7 remain unchanged. The largest measured source error is 7.32e-15.

The measured low-phase seed-component jump to x=1.5 at .17375 is joining of
distributed activation, not surface extension or validated speed. The new
temporal uncertainty and earlier failed spatial target remain unresolved.
No constitutive, tip/birth, acceptance, native checkpoint, transport or limiter
code changes. [Measured report](BULK_CONTINUED_REPORT.md).

# One aligned-wide spatial comparison — 21 October 2026

[BULK_SPATIAL_REPORT.md](BULK_SPATIAL_REPORT.md) reuses the 48x64 half-step
trajectory and adds one fresh 96x128 run at the same Delta t=.00125 s. The
native core SHA remains 31ad1e59820079e712bdae23732697394cbfe36e8377001fa50cdac2a05b8d5a;
no constitutive, solver, history-owner, projection, acceptance or checkpoint
implementation changes. Parsed inputs differ only by mesh repetitions,
diagnostic DSO and output location. The two actual timestep sequences match.

`bulk_cohesive/spatial.cc` reuses the energy observer in a separate CMake target.
Only its inherited read-only area accumulator changes from double to long
double: summing 786432 positive weights in double gave 1.4999999999983302 m2,
failing the existing 1e-12 guard. Independent accurate summation gives 1.5.
The guard, actual JxW, normalization and constitutive quadrature are unchanged;
the failed initialization has no positive physical step and remains evidence.
Old binaries/DSOs and the first seed export are preserved; the fresh retry seed
is bitwise identical. No particle/domain/coverage workaround is introduced.

`assess_spatial.py` reconstructs each actual affine cell/QP layout from native
exports and checks positive physical coverage. It verifies zero initial DG,
local and native-FE memories, direct continuum nodal sampling, irreversibility,
unloaded stationarity, native source/residual/KKT/publication gates, energy
components and dense physical-stress snapshots with the full plane-strain norm.
Named-header guards and direct dense-versus-compact q comparisons protect the
point/column ordering. Phase fields/increments are evaluated in common physical
coordinates; local q histories stay on their original QPs. Y is formed there
before positive-JxW physical-kernel averaging. No constitutive state is remapped.
`plot_spatial.py` uses equal scales; native Y values occupy midpoint display
bins, whose areas are explicitly not quadrature weights. This removes marker
gaps without changing samples, quantitative averages or physical history.

Both meshes remain serial and fixed, with no runtime AMR/transport. All nine
DG modes, the native FE source complement, ordinary FE phase and existing
material identities are retained. Quantitative spatial growth agreement fails;
neither a convergence order nor a fine-mesh temporal/restart/MPI qualification
is claimed. The earlier lifecycle and BP3 evidence retains its original scope.

# Boundary work and temporal diagnostics — 20 October 2026

[BULK_ENERGY_REPORT.md](BULK_ENERGY_REPORT.md) adds only the test DSO
`bulk_cohesive/energy.cc`, a compile-time diagnostic option in the existing
fixture, analysis/plots and two timestep inputs. No ASPECT core or constitutive
source is edited/rebuilt. The qualified core SHA remains 31ad1e59820079e712bdae23732697394cbfe36e8377001fa50cdac2a05b8d5a.

The new DSO reconstructs accepted **unconstrained** boundary forces with the
same mechanical Gauss8 rule, full reconstructed plane-strain stress, actual
native pressure, native velocity FE shape gradients and the physical Dirichlet
rows. It records F and dt*accepted_velocity at every accepted endpoint, including
zero-motion states. Physical pressure scaling and both symmetric xy contributions
are retained. The direct force-dot-increment agrees with native W_BE to <=1.74e-17
J/m. Averaging old/new boundary forces is a work quadrature only: no constitutive
history or tangent is modified. All other fixture builds retain the old export
behavior; only the energy DSO reduces dense exports and saves QP q every step.

Independent analysis uses every original QP q/phase, not interpolated material
states. Baseline DG vector blocks follow active-cell/Morton order, while local
history records follow numeric CellId order. Independent actual QP exports catch
and resolve the initial decoder ordering mistake. Stress, energy, rate/phase/
fracture identities and native source/ledger tolerances are checked separately.
New-run full bulk stress/storage checks are at the declared snapshots; per-step
stress BE losses are retained native data, not falsely labeled independently
reconstructed. The original baseline endpoint generalized force is recovered
from saved W_BE/dgamma during loading and cross-checked against stress moments.
Zero work on hold is not zero reaction.

Only h/2 and h/4 fresh trajectories are added. The native quarter end-time
closure produces a final dt=6.66e-16 s step; its accepted publication/dissipation
and zero phase/work increments remain in the analysis. Common-time cumulative
budgets include the last matching state instead of dropping this step. Existing
acceptance, checkpoint implementation and source-history ownership are unchanged.
The original 1% endpoint screen is retained alongside the newly recommended
physical-work/temporal **reporting** criterion. No BP3/default change is made.

# Native FE complement for the bulk source — 20 October 2026

The default-off fixed-mesh homogeneous **bulk** experiment now retains the
native FE source component outside DGQ2. See
[BULK_SOURCE_HISTORY_REPORT.md](BULK_SOURCE_HISTORY_REPORT.md). The original
pure-DG failure and the first complement's cancellation failure remain evidence.
No BP3/default path or physical constitutive equation changes.

`BulkCohesiveState` owns independent nine-mode DG T, native FE velocity memory A,
local mechanical-QP histories and the accepted physical phase. A+=beta(A_n+du).
`rebuild_native_projection()` uses the same matching-cell Gauss3 mass projection;
`evaluate_tensor()` returns T_DG+2G dev3 symgrad A-P3(2G dev3 symgrad A) at arbitrary
matching-cell/reference points, including faces. Repeated cell traces are not
averaged. Local a/q/k/L are retained on original QPs, not interpolated to a new
constitutive layout. The constant-coefficient, fixed-mesh identity retains the
complete native source without dropping independent DG modes.

`solve_bulk_cohesive_phase()` caches full accepted tensors and uses actual
native incremental gradients in its unchanged coupled return/residual/tangent/
energy. It advances DG and A only after convergence into a separate next state.
The existing pending callback publishes them together once or discards them.
Bulk storage uses the full plane-strain physical stress, including cross terms,
zz and both xy contributions; there is no additive artificial remainder energy.
The ledger column `projection_error` now checks the **full reconstructed**
recurrence with exactly the old 1e-12 metric; raw DG discrepancy is exported
separately as R. Independent analysis also verifies P3(A) and all DG coefficients.

A replayed step-47 failure exports actual/expected tensors and both paths'
coefficients before publication. A 70-digit reconstruction isolates native-memory
gradient cancellation. The corrected evaluation removes one cellwise constant
translation before summing, using long double; partition of unity leaves the
mathematical FE gradient unchanged. Source projection, incremental native
calculation and the consistency tolerance are unchanged. Deliberately inconsistent
histories still fail by ratio 1.4142e7. Full failure location/components are in
the report and exported CSVs.

The existing checkpoint mechanism gains velocity-memory coefficients and an
explicit bulk v2 signature; the derived projection/cell associations rebuild on
restore. V1 bulk checkpoints explicitly fail conversion. BP3 serialization is
unchanged. The small final affine/distorted pair passes independent tensor,
source, storage and tangent checks, actual rejection/retry and once-only commit.
Restart reproduces 404 field/history exports bitwise. The new private header is
consumed by two native unity translation units; both were rebuilt into an
isolated executable, together with matching new fixture DSOs. The public model
header, original native builds/DSOs and all 34 protected BP3 files remain pinned.

The following dated entries describe retained earlier attempts; they are not
claims that the subsequently qualified complement is still absent.

# Bounded rotated-mesh audit — 20 October 2026

[Control report](BULK_ORIENTATION_REPORT.md) records a benchmark-only extension;
no production C++, source rule, tolerance or accepted-history path changes.
`bulk_cohesive/orientation.cc` includes the original fixture under a test-only
macro and registers a Box-derived imported fixed geometry plus mesh/projector
audit. `make_orientation_mesh.py` produces a conforming rigid 45° Q1 core and
annulus, checks determinant corners and preserves exact outer face coordinates.
The existing zero-function topography selects MappingQ1. Default zero-topography
would select MappingCartesian in `core.cc::construct_mapping`, which cannot
be used for this control. The identity control quantifies the mapping change.

Only the isolated DSO replaces Cartesian normal-column diagnostics with the
same positive-weight moment identity integrated with physical JxW over the
area; no line normalization or constitutive history is inferred. The original
column coverage code remains compiled unchanged in the original fixture.

`bulk_cohesive_phase.cc` source3 projection and <=1e-12 independent Gauss8
check remain byte-identical. All nine DGQ2 modes are present. Native Q2 gradient
is exactly in DGQ2 on affine rotated cells, but generally rational on bilinear
transition cells (J^-T times reference gradient). All-basis native and independent
NumPy projectors confirm the capacity failure, with source3/source8 agreement.
At .1075 s the actual native guard rejects before phase/history/pending-commit
publication, after 42 valid accepted states. It is not an MPI, observer-coverage
or quadrature-order failure. The unchanged exception does not export the actual
native error value; synthetic gradient errors are labeled separately.

The physical-coordinate assessor decodes only accepted original-QP histories,
checks geometry/phase clocks and independently reconstructs native weak gradients
under the established normalized 1e-10 screen. Common compact kernels average
q/Y after native evaluation; their fracture action uses the same physical test.
Extra quadrature integrates Q1 fields only. The native KKT condition is reported
separately. No post-onset orientation or energy qualification is claimed.

# Saved-state angular audit — 20 October 2026

[Angular report](BULK_ANGULAR_REPORT.md) adds only read-only analysis and figures.
No C++/input/binary/native run changes. `PhaseFieldProfile` defaults to N=5000;
the fixture uses peak .6, ell=.15625, AT1 and Lorentz p=1/m=128, followed by
vertex sampling and Q1 interpolation. The ideal profile satisfies
ell² P'²=P(.6−P)(1.5−P)/(1−P)². Its smooth straight-band fracture force is
Kf h'/1280, with the positive −2 Kf ell² P'/r cap curvature contribution.
Neither the tabulated function nor its Q1 representation has that classical
pointwise Laplacian; actual native g_f includes cell-interface gradient jumps.

`assess_angular.py` reads the existing persistent vector exactly in the fixture's
native ordering: DG fields, std::map<CellId> local a/k/L/q/rate, phase and clocks.
CellId uses numeric coarse-id ordering; active or lexical order would be wrong.
Fixed geometry from the complete step-52 export is reused, independently of its
stress values. All a/k/L/q records reproduce that export exactly. Other-step
phase/clocks and reassembled weak driving validate the missing dense step-54
recovery; normalized mechanical discrepancies stay below 1.40e-14 across cases. No state
is passed through a return map or interpolated into another layout.

Arc q/Y and F/R use one positive unit-mass test, sum_i N_i(z) N_i/m_i.
Y is evaluated from each saved local q/phase first, then averaged separately;
the mean q is not treated as a constitutive state. Independent smooth weak
fracture actions use edge integration by parts and explicit compact-support/
band-cap splits. Native and smooth forces are compared with the same test;
strong point predictions are labeled separately. Phase contours distinguish
seed geometry from driving lobes. Current data support the combined weak-balance
mechanism but do not remove grid-orientation uncertainty; a proposed rotated
fixture must qualify non-Cartesian diagnostics/source projection before evolution.

# Bounded cap-flank/boundary audit — 20 October 2026

[Diagnosis](BULK_SENSITIVITY_REPORT.md) adds only three input overrides and
read-only analysis/plotting. No production C++, material/seed/loading function,
solver tolerance, default or accepted-history code is changed. The pinned
core/fixture binaries and original bulk evidence remain identical.

- `bulk_cohesive/native.cc::seed` samples the same continuum at actual vertices;
  it does not prolong the coarse obstacle. `BulkCohesiveLoading` depends only on
  y and time, so widening x retains identical affine strain history on every
  boundary, including the additional top/bottom portions. The new runs are fresh
  (`Resume computation=false`) with zero histories, never a QP-layout conversion.
- The new `assess_sensitivity.py` independently assembles Q1 g_f with exact
  Gauss2 integration and g_m from actual exported Y/JxW. It checks each native
  g_f/g_m/total contribution and actual area (1 or 1.5 m²), not point counts.
  The unit-domain-only assumptions in the older assessor are left intact and
  are not applied to the widened case. New g/m densities are explicitly weak
  diagnostics, not pointwise phase equations or interpolated local histories.
- Native KKT is interpreted against the current step's physical obstacle.
  Previously grown nodes may be at the obstacle during a hold with positive
  residual. Shared-node fracture density is Kf[1+2 ell²(K phi)_i/m_i]; changing
  the representation/basis changes this curvature contribution even where
  initial nodal phi agrees exactly. The square control demonstrates both
  directional discretization effects without labeling either backend a truth.
- Fine G6/G8 reproduces flank activation (phase difference 8.41e-11), with free
  ramp-end flank residuals below 3.1e-14 Pa. Widening changes the flank amplitude
  by 28.66% and suppresses right-boundary damage. Near-field basis, seed and
  parameters are identical for this latter comparison. At x=1 the old boundary
  becomes an interior point; raw weak test support there differs, so comparisons
  use the actual boundary statistics separately from near-field functionals.
- Five centerline contours remain fixed; low cap-flank cuts advance slightly,
  including some initially intact edge nodes outside the older narrower mask.
  Those differences do not contradict the retained old-window result, nor qualify
  connected centerline propagation. Numerical gates pass, original physical
  energy failures remain, and no additional boundary or mesh sweep is selected.

# Selected bulk-cohesive model — 20 October 2026

Entry `f1f9f29cf40c32d1ddcea000b724375b64853532`, plus preserved incoming
P4a/manual changes. The user-supplied bulk derivation/brief explicitly select a
new law; this is not a claim that the old P2/P3/P4a equations have been repaired
by an exact variable transformation. See [formulation](BULK_COHESIVE_FORMULATION.md),
[pre-run screens](BULK_COHESIVE_NATIVE_SELECTION.md) and
[bounded report](BULK_COHESIVE_REPORT.md). All previous audit sections below
retain their original historical scope.

- `bulk_cohesive.cc` is a surface-independent stateless 3D constitutive module:
  backward-Euler T/a return, intact mixed multiplier, physical stress, partial
  phase driving, eliminated-state consistent tangent, energy and dissipation.
  Signed/absolute engineering shear increments use dt*w; w is not m/s surface
  slip. It uses fixed S:S=1/2 and lambda=eta/G for both Maxwell mechanisms.
- `bulk_cohesive_history.cc` owns accepted/trial separation and conditional native
  checkpoint records. DGQ2 T has all nine modes per component; local a/k/L/q/rate
  are stored at the actual fixed mechanical quadrature. Accepted physical phase
  is the obstacle. Fresh initialization is explicitly unloaded; same-mode restart
  checks material, layout, cell identity, mesh, phase and accepted clock.
- `bulk_cohesive_phase.cc` solves actual coupled mechanics/phase from the same
  immutable accepted owner. It assembles raw physical terms, condenses once using
  native AffineConstraints, and evaluates master-space KKT via C^T g. Free actions
  are C^T H C with no artificial constrained diagonals. It reuses ordinary FE AT1
  fracture, native UMFPACK, native FE and existing outer publication semantics;
  it neither invokes the old surface energy nor differentiates viscous metrics
  as stored energy. The generally nonsymmetric residual is not the old endpoint
  potential.
- Gauss3 projects native consistent-deviator FE source into DGQ2; an independent
  unprojected mechanics-QP path checks its discrepancy before publication at the
  unchanged selected 1e-12 scale. No nonlinear localization correction is projected
  away. Physical stress is reconstructed from T and local a at those same QPs.
- `solver_schemes.cc` dispatches only the new enum to this solver before old
  observers; `core.cc` extends the existing outer backup/pending callback to it.
  `checkpoint_restart.cc` includes the new owner only for the new mode, leaving
  default archive bytes unchanged. Generic bulk-mode viscosity output uses its
  BE coefficient; existing surface/default Maxwell coefficients stay unchanged.
- `phase_field_fault.cc` retains the default `cohesive` selector and admits the
  explicit `bulk cohesive` enum with serial/fixed-mesh/homogeneous/constant-friction/
  static-field/AT1 guards. Old consistent-endpoint/surface options must be off.
  `fe_strain.cc` selects its actual mechanical Gauss rule independently of old
  reconstructed strips; `phase_field.cc` admits the new mode to ordinary FE
  fracture evaluation without requiring an old surface strain history.
- The original hold globalization failure had 27 free-master directions pointing
  below their exact obstacle at a mechanically unequilibrated predictor. A
  feasible projected Newton search in independent masters, followed by native
  constraint expansion, corrects this demonstrated implementation issue. It does
  not cap phase below 1, alter the return map or relax KKT/residual/energy gates.
  Original failure, saved directions and checkpoint replay remain retained.
- `observe_profile` accepts only parameters, positive raw normal weights and
  copied local responses. It computes I/se/Z/rho and storage/viscous/friction
  residual identities without mutable access to the simulator. The native fixture
  rebuilds straight-column diagnostic moments/masks and checks persistent state
  exactly. This does not implement automatic M3 ridge reconstruction or physical
  surface birth. I=0 has no normalized observer row; it does not suppress bulk q.
- New mechanics, coefficients, phase driving and publication read no particle
  values. Generic phase initialization still requires diagnostic particle/H
  infrastructure, and checkpoints retain it. Domains/CPDI are disabled; the
  explicit existing RK2 freeze gives dt=0 for particle motion. Legacy particle
  and CPDI equations and their earlier failures are unchanged.
- Independent material/profile checks, native source/action checks, actual outer
  rejection/retry/once-only publication and new-history restart pass. The complete
  native unit suite passes 136 cases/165676 assertions; the retained P3 control
  reproduces 16 exported physical/history/load/constraint files bitwise. All 34
  protected BP3 identities and incoming manual AGENTS/P4a source remain unchanged.
  The physical response is **not propagation-qualified**: no tracked centerline
  tip advance; spatial growth-area and endpoint external-work screens fail.

# P4a finite-tip support and driving audit — 19 October 2026

Entry `f1f9f29cf40c32d1ddcea000b724375b64853532` closes P3. See
[P4A_TIP_REPORT.md](P4A_TIP_REPORT.md) for the equations, actual native figures,
pre-run screens, measured integration results and unresolved scientific choice.
No production source/header/CMake file is changed in P4a.

- `fe_strain.cc::prepare_fe_strain_history` obtains actual native bulk-point
  associations from the manager. `utilities.cc` admits segment-normal traces
  only for xi in [0,1] and within normal width; it does not continue a physical
  tip. The horizontal fixture has zero left boundary-completion wedge. Its
  rounded right cap is phase/obstacle support, not a continued surface.
- `fe_phase_energy.cc::evaluate_fe_phase_energy_at` explicitly treats an
  invalid-segment native point as T-only bulk energy and skips all mechanical
  phase variation there. Within the strip it reconstructs tau=T-2Z h/I S and
  assembles direct phase and complete consistent-I adjoint contributions.
  The surface term is Z²/(2GI). Ordinary FE fracture is integrated everywhere
  by `PhaseFieldHandler::evaluate_cell_fracture`. Thus a phase perturbation
  supported wholly ahead of Gamma has exactly zero mechanical derivative,
  while its fracture derivative need not vanish.
- A Q1 node geometrically ahead may have support behind the tip. The cut case
  measures a nonzero 1.05454e-7 J/m derivative of such a basis function. Purely
  ahead directions still have exactly zero mechanical derivative. This is a
  support distinction, not a rounding or missing-normalization-adjoint error.
- `fe_normalization.cc` retains analytic normal-ray integrals, Gauss8 surface
  projection and full derivatives. No finite positive I admission threshold,
  floor, physical bound or existing normalization derivative was changed.
  The cut case's minimum I is 105.760830749 m; it does not test the I→0 limit.
- `history.cc::compute_history_candidates` updates phase-driving history only
  inside an active association. The legacy intact-point/recoverable formulas
  require an already defined q; they do not supply a traction ahead of Gamma.
  `particle_free_history.tex` explicitly requires an intact I=0 activation/
  limiting rule. Combining this observer with the total-energy endpoint solve
  would change the model without resolving the absent rule.
- P3 finite-tip physical mechanical/phase guards remain in `history.cc`,
  `reconstructed_fault_endpoint_phase.cc` and the frozen API. Read-only energy
  calls return existing complete gradients; independent energy differences
  need neither an enabled tangent nor a physical trial/publication. No solve
  eligibility or native advancement route is relaxed by the new test plugin.
- The existing `p3_native_checks` DSO gains an explicit test-only environment
  selection, `ASPECT_P4A_TIP`, for frozen observation. It exports actual native
  points/JxW, profiles/ray intervals/surface weights, physical stress, all phase
  derivatives and finite differences. Default P3 expressions and arithmetic
  remain unchanged: a new native P3 transaction reproduces 16 original exports
  byte for byte. Production APIs/defaults, BP3, and saved P3 evidence stay fixed.
- `assess_p4a_tip.py` independently integrates actual exported Q1/DGQ2 fields,
  resolving every cell, surface knot, strip boundary and represented support.
  It builds its own surface mass, DI and normalization adjoint; checks all
  gradient/load entries; and retains all nine modes and the full 3D norm.
  Gauss24/32 and native Gauss8-panel/12 agree within the predeclared screens.
  The first inside-cell endpoint is on an existing quadrature panel boundary;
  arbitrary off-panel or oblique finite endpoints are not qualified.
- Required scientific choice is ahead-of-tip mechanical activation/traction
  with an I=0 birth limit. General nonzero-tip joining remains a later choice;
  P3's admitted positive-support zero-tip/zero-new-history map is preserved.
  No finite-tip equilibrated or phase-active event/work balance is claimed.
  P2 strict branch matching stays false; all 34 protected BP3 identities and
  manual AGENTS edits match entry. No RSF/AMR/transport/MPI campaign was added.

---

# P3 native frozen-append completion audit — 19 October 2026

Entry c8e4ae65d5d6750889a11e5a1956efd303e10a37. See
[P3_NATIVE_COMPLETION_REPORT.md](P3_NATIVE_COMPLETION_REPORT.md) and its actual
native checkpoints/exports. The preceding synthetic result is preserved below.

- The material's existing FEStrainState/DGHistory owns T, Z, I and the accepted
  physical phase anchor. The manager owns q=Z/I, previous I and every rate
  representation. The fixture registers accumulated slip in the manager's
  existing property registry/checkpoint; it is not a second persistent owner.
  A zero-time append must retain Z and update q and previous I together.
- `dg_geometry.cc` stages the joint candidate with actual native owners and the
  existing manager archive. T+ = T- + Pi_Q2[2(Z+chi+-Z-chi-)S]. All nine modes
  survive, normal coefficients are copied exactly, and source projection,
  constitutive update and physical publication are not called. Old endpoint
  histories are zero; new surface records have explicit zero Z/slip/rate. No
  new bulk material or general nonzero-tip birth rule is selected.
- Candidate manager binding is restricted to nonconcurrent serial preparation;
  success/exception restores accepted owners and invalidates scratch caches.
  The unsupported zero-phase append exercises a real preparation exception.
  M5 validates parent/native clock/phase/publication state before dropping the
  canonical surface/coupling/inverse objects and publishing prepared owners.
  Affected search/QP, DG basis, normal rays and surface assemblies are rebuilt
  and used. The fixed bulk mesh and native AffineConstraints do not change.
- A fresh geometry initially computes generic bootstrap I. Its existing
  quadrature/stop criteria differ from the native analytic FE operator:
  max difference 4.12813e-8 m, scaled 8.62568e-9. The initial implementation
  incorrectly treated bootstrap values as a saved canonical I and hit the
  unchanged 1e-12 restore guard. The corrected fresh-geometry branch constructs
  I through the same canonical FE projection as fresh native initialization.
  Both values remain in `geometry-normalization-bootstrap.csv`; independent
  Gauss16 verifies canonical values to 3.03e-15 relative. No restore tolerance,
  normalization floor/derivative, generic backend or default is changed.
- Point reconstruction and native bulk-work assembly now admit a **read-only
  zero-interval observation** only under the explicit frozen diagnostic API.
  It uses beta=1, eta_ve=0, retained physical stress and q; it does not call the
  positive-interval 1/eta_ve history-correction formula at zero. All original
  entry points delegate with this option false, preserving physical laws.
  The stateless law's Vmin guard remains. For this constant mu_s=mu_d=.1,
  damping=0, zero-interval fixture, temporary admissible observation rates give
  exactly the same native load as the zero-rate rational limit. Persistent
  birth rates remain exactly zero. This does not qualify RSF/nonconstant
  friction or a physical birth evolution rule.
- An explicit setup call (no new production parameter) requires homogeneous
  cohesive accepted DG/consistent-endpoint ownership, no solvers, no RSF,
  serial execution and fixed Cartesian geometry. The simulator performs only
  initialization/postprocessing, never a physical step. Finite-tip mechanical
  and phase solves remain explicitly forbidden, also after restart. The
  special quadrature is fixed tangential8/normal12; source/default rules stay.
- Native checkpoint version 5 carries the restricted zero-time clock and a
  geometry-publication counter separate from accepted physical publications.
  Manager, mesh, native solutions, particles and material histories use the
  real Simulator create_snapshot/resume path. The zero-time observer bypasses
  ordinary start_timestep(): it therefore explicitly rebuilds current native
  constraints on resume. The first frozen shortcut had 0 versus 768 rows;
  stronger ledger comparison caught this derived-cache omission despite exact
  history/stress/load agreement. Constraint rows are now exported and compared
  exactly before/after/restart, without changing their construction. Versions 1–4 keep their existing
  format/eligibility. Pre-append restart/retry and post-commit restore match all
  persistent bytes and physical loads for both inputs. It is not conversion
  of a frozen geometry checkpoint into an evolving finite-tip model.
- Native MPI/finite-tip Stokes reassembly are not qualified here. Existing
  component-only two-rank evidence remains separate. All 34 protected BP3
  identities and incoming manual AGENTS work are preserved. P2 branch and
  accuracy limitations, including strict_reference_match_passed=false, remain.

---

# P3 imposed geometry-transfer audit — 19 October 2026

Entry reusable P2/hanging-node work is committed as `2d0d40183` (parent
`612b75d3570d81763407875f3a12c12999b9fa8e`). The selected original propagation
PLAN P3 diagnostic and its precise qualification boundary are recorded in
[P3_SYNTHETIC_EXTENSION_REPORT.md](P3_SYNTHETIC_EXTENSION_REPORT.md).

- `fault.cc::append_vertices` already preserves old vertex/property indices and
  advances geometry version once per append. Directly calling it on a managed
  fault leaves support widths and committed/current/trial rates inconsistent.
  The new, currently test-only-called manager `append_reconstructed_fault`
  validates/stages all those records together, requires explicit finite birth
  properties/rates, and swaps only after validation. Active nonlinear solves
  and trials reject geometry changes. Existing archive fields/schema persist.
- That manager operation invalidates both particle projection systems and
  Stokes-QP caches, including the indexed normal-profile search. The synthetic
  candidate rebuilds normalization, point associations, quadrature split at
  surface knots and fixed-grid constraints. Candidate associations are scratch
  owned with candidate geometry; rejection cannot expose them to accepted data.
  Configured boundary-completion routes reject append until their endpoint
  metadata/support can be rebuilt jointly; the bounded horizontal fixture has
  no nonzero completion wedge and does not exercise those routes.
- The actual cohesive native backend still assumes unit through-going geometry
  in `fe_strain.cc` (initialization, bulk point segment map, FE stress),
  `dg_history.cc` (cohesive reconstruction), and full-normal initialization in
  `manager.cc`. `validate_fe_strain_restrictions` freezes its accepted geometry.
  These guards are **not removed**. The P3 fixture exercises native containers,
  DG basis/mass integration, MPI vectors and archives, not a simulator outer-loop
  finite-tip geometry transaction. There is no live Stokes/preconditioner object
  to invalidate in that fixture; native material/solver integration is unqualified.
- The compact cap uses the existing `PhaseFieldProfile` with actual p=1 Lorentz
  degradation. Its material obstacle is sampled directly and frozen. It is not
  a new propagation seed or a long-tail/normalization-floor workaround. Boundary
  contact qualification identifies only the original left box contact with zero
  completion wedge; both right endpoints are physical interior tips.
- At frozen phase/time the required coordinate map is
  `T+ = T- + Pi_Q2[2 (Z+ chi+ - Z- chi-) S]`, retaining all nine modes, with
  `zz=-(xx+yy)`. Surface overlap Z/slip/rate rows are copied exactly; prescribed
  new surface rows are zero and old tip rows are zero. Independent Legendre
  projection and dense normalization/weak-functional calculations verify the
  native nodal projection; changes in cohesive storage and weak integration
  measure are separated from bulk stress transfer defects.
- No Maxwell update, source projection, physical phase solve, particle movement,
  production history owner, acceptance tolerance, limiter/default, BP3 source,
  package or job changes. Only a neutral manager API and bounded native-component
  tests are added. In particular this is not historical PF3 or completion of
  coupled native geometry acceptance/restart or genuine phase activation.

# P2 physical-response audit — 19 October 2026

The selected bounded experiment is in [P2_PHYSICAL_RESPONSE_REPORT.md](P2_PHYSICAL_RESPONSE_REPORT.md).
Entry revision remains `612b75d3570d81763407875f3a12c12999b9fa8e`; the complete
incoming patch and source copies are preserved under `build-particle-free/p2-response/`.
This task changes no production source, constitutive equation, solver tolerance,
acceptance rule, normalization, history owner, limiter or default.

- `fe_strain.cc::initialize_fe_shared_phase` reads the retained pre-existing
  damage prescription only for fresh step zero, forbids history conversion,
  interpolates onto native phase vertices, and leaves particle H diagnostic.
  At every selected 512-normal-cell vertex its table equals the original
  continuum BVP exactly. The finer 1D reference samples that same BVP at its
  new vertices; it does not interpolate/prolong the coarse solution or relax
  independently. Unloaded equilibrium is fracture KKT under this obstacle,
  not an unconstrained zero-damage minimization or a seeded-H evolution law.
- Existing domain-free geometry/material entry and DG source, endpoint reduced
  Newton, physical ordinary-FE fracture, pressure/plane-strain convention and
  native outer acceptance/publication remain unchanged. The selected fixed
  uniform mesh uses the identity master map. Lifecycle/derivative evidence
  from the completed hanging-node task is reused within its reported scope.
  The retained constitutive normal-friction branch uses prescribed adiabatic
  pressure (10 kPa here); the native dynamic pressure remains a distinct field
  and enters the mechanical/work diagnostics. Neither pressure is substituted
  for the other in the reference.
- `pf6a_checks.cc` gains an optional **benchmark-only** full-snapshot schedule.
  The empty default retains every-step dense exports. Phase/gradient, full
  27-coefficient DG tensor, native velocity/pressure, surface histories, clocks,
  metrics and native gates remain every-step; only dense point samples,
  residual probes and redundant flat state exports are scheduled. The residual
  probe retains its incoming anchor and explicit no-publication check.
- `pf6a_spatial_reference.py` gains an optional interval-mean boundary drive
  and finite hold. Its original default update expression is unchanged. The
  hold keeps all incoming histories and applies the same beta/kappa, contact
  complementarity, obstacle and reduced mechanical/phase equations.
- `run_p2_response.py` prepares three explicit fixtures and immutable command,
  input/source/executable identities; the native integrator controls advancement.
  `assess_p2_response.py` reuses the independent PF6a physical reference and
  work path. It samples complete native DG polynomials, uses the embedded 3D
  tensor norm and fixed peak scales, never aligns branches, and compares
  interval-average rates on common physical intervals. Fixed-phase work uses
  scalar Maxwell time moments, independently checked against the original
  full path before use. The evolving-phase path is the retained diagnostic
  linear-physical-phase/exact-Maxwell path, not a new advancement scheme.

The completed bounded response passes all predeclared practical screens:
unloaded native growth/KKT zero, loaded growth .0543973, phase temporal Linf
6.31e-8, targeted whole/localized spatial stress differences .294%/.609%, and
absolute loading work remainder .006033→.001461 J/m under one dt halving.
Separate free/contact/phase path works remain explicit and decrease; ledger
closure alone is not the criterion. All 14/318/622 native gates and 2/76/152
accepted publications pass. Same initial phase/history exports agree exactly;
all 858 production source/header and 34 protected BP3 identities match entry.
The opposite reference mirror branch remains unregistered, with phase Linf
5.91e-5 and `strict_reference_match_passed=false`. See the report and its
replayable raw/derived evidence for commands, units, thresholds and limits.
Only the next bounded P3 extension is recommended; no geometry change is begun.

# Fixed-mesh hanging-node cohesive phase — 19 October 2026

User-selected extension of the independent-master formulation below. Worktree
and branch remain `aspect-pf-rsf-particle-free` / `pf-rsf-particle-free`, revision
`612b75d3570d81763407875f3a12c12999b9fa8e`, with the incoming consolidation
preserved in `build-particle-free/hanging-phase/entry.patch` and `entry-source/`.
No BP3 artifacts, defaults, physical inputs or numerical acceptance tolerances
are replaced. Detailed measured evidence and reproducible commands are in
[PF6_HANGING_REPORT.md](PF6_HANGING_REPORT.md) and its compact verification link.

## Source ownership and discrete reduction

| Source | Change and retained contract |
|---|---|
| `include/aspect/reconstructed_fault/endpoint_phase_constraints.h` | Small coordinate helper copies closed native phase hanging rows, admits homogeneous convex within-phase partitions of unity, identifies independent physical masters, expands with native `AffineConstraints::distribute`, and applies sparse `C^T` to raw duals / `C^T H C` to raw Hessians. No histories, global dense C or acceptance owner. |
| `source/simulator/solver/reconstructed_fault_endpoint_phase.cc` | Serial, nonperiodic, homogeneous domain-free cohesive DG extension only. Expand every phase trial/action, pull back elastic and cell-fracture derivatives once. Canonical accepted lower obstacle is mandatory. KKT, active sets, positive arithmetic scales, search vectors and stability probes use masters; physical feasibility and physical L2 Rayleigh denominator use the expansion. Uniform identity retains the former assembly and arithmetic order. |
| `source/simulator/solver/reconstructed_fault_stokes.cc` | Endpoint mechanical response still returns raw physical-node phase duals. Its residual-correction action accuracy is measured after `C^T` in the same master norm as the outer solver. The outer solver alone reduces the returned action. Bulk RHS is already mechanically constrained and is not reduced a second time. Pressure scaling, constraints, actual residuals, condensation, tolerances and native publication are unchanged. |
| `benchmarks/.../pf6_transfer_checks.cc`, `pf6_interval_ramp.h`, `pf6a_checks.cc` | Reuse the existing transfer/lifecycle observer for a fixed local evolving mesh. Optional benchmark checks default off. Share the existing exact interval-mean boundary ramp without changing its formula. Keep the old frozen local patch and original observer archive record; store one optional completion flag separately. Export raw/full-master diagnostics, actual cutback and native histories. |
| `unit_tests/reconstructed_fault_phase_constraints.cc` | Native closed-chain expansion, independent dense adjoint/pullback, nonlinear constrained finite differences, obstacle preservation, exact identity and invalid-row rejection. |
| `benchmarks/.../assess_phase_energy.py`, `assess_pf6_hanging.py` | Existing independent physical energy/normalization reference gains zero-weight padding for rays crossing different numbers of local cells. Uniform arrays stay unchanged. Independently derive C from coarse physical edges, verify complex-step complete fixed-history actions and re-equilibrated native endpoint actions, master KKT, physical tensor reconstruction and Gauss24/32 controls. |

For `phi=C psi`, the new local fracture/search matrices use **raw**
`evaluate_cell_fracture()` derivatives. They do not reuse the physical system
block with its constrained-row entries. Assemble `K_F=C^T H_F C`,
`g_F=C^T g_F,physical` and the search metric
`K_F + scale*C^T diag(m_physical) C`. This is the existing physical cell-volume
mass lump expressed in master coordinates; it adds no energy or dissipation.
The old identity branch still uses its original assembled FE/CPDI block.

The endpoint response includes the full native mechanical/slip response and
normalization first/second derivatives. All its physical inputs are expanded
before evaluation; the returned physical dual is reduced once. For raw positive
roundoff majorant r, `|C^T delta_g| <= |C|^T r = C^T r`; use that positive
majorant with the original arithmetic factor and nonlinear/linear criteria.
The helper rejects inhomogeneous/nonconvex/cross-field rows; the selected local
entry excludes MPI, periodicity and legacy/mature routes. Existing mesh,
geometry, stationary population and no-transport guards remain.

All physical steps retain accepted T_n, Z_n and the same physical phase lower
anchor during every trial. Native outer pending publication, rollback, accepted
clock and checkpoint ownership are unchanged. The local mesh is prepared once
at t=0 by exact prolongation of the retained coarse sampled seed. Its growing
flank patch is `0.5<x<0.75, 0.1875<y<0.25`; 67 cells, 90 physical phase nodes,
86 masters, 8 fixed surface segments and 144 stationary diagnostic particles.
One hanging edge remains obstacle-active while others grow. This is a solver
qualification, not a new physical initialization or a spatial reference.

The complete ten-free-master Hessian is positive at the nonzero-history endpoint;
76 other masters have strict obstacle multipliers. Native symmetry defect is
9.28e-17, minimum eigenvalue 7.3572 J/m in the nodal master coordinates.
Independent gradient/action errors are at most 1.27e-14; re-equilibrated central
FD actions agree to 2.33e-9 using the established scaled error definition.
Uniform 162-file identity, actual rejected trials, native cutback, once-only
publication and same-count serial restart are recorded in the report. Native
mechanical generation guards caught an initially stale diagnostic action;
normalization guards caught an inadmissible diagnostic tail perturbation.
Both failed runs remain evidence, and only the diagnostic ordering/directions
were corrected. No guard or tolerance was bypassed.

The previously proposed formulation in the consolidation section is now
implemented only within this bounded serial admission. Its final sentence that
no implementation was included describes the earlier consolidation. Keep
`strict_reference_match_passed=false`, all previous mirror/branch differences,
Gauss6 failure and spatial/endpoint temporal findings. No AMR, transport, MPI
local-phase evolution, geometry motion, surface change or propagation follows.

---

# Evolving-phase consolidation — 19 October 2026

Selected by `../introspection.md`. Entry is `612b75d3570d81763407875f3a12c12999b9fa8e`
on `pf-rsf-particle-free` in the intended worktree; tracked files were clean.
Untracked instructions/plans and existing raw evidence were preserved. Scientific
authority remains `../current_design.md` / `../specification.tex`; the actual
derivation path is `../particle_free_history.tex`. AGENTS now fixes that path
and no longer requires an untracked PLAN file. The three development lines and
accepted-state owners are mapped in [current STATUS](STATUS.md).

## Small structural changes

Only two runtime translation units change; neither changes an equation,
selector, tolerance, checkpoint layout or publisher:

- `simulator/solver/reconstructed_fault_endpoint_phase.cc` shares the existing
  fracture-Hessian matrix application between analytic and finite-difference
  reduced actions. Matrix, scaling, operations and MPI participation remain the
  same. This is the fracture-only block, not a second elastic-energy term.
- The same solver builds its fixed phase-node lookup once per solve and shares
  the original negative normal profile derivative. The stability probe retains
  `phi>lower+1e-13`; the search retains `!contact`. They are **different masks**.
  No translation mode is imposed on the solution, and branch/tie selection is
  untouched. The new helper is restricted to the existing horizontal fixture.
- Comments and the reuse-check label identify the selected fracture backend:
  ordinary cell FE for independent DG, CPDI for the older FE-history path. The
  entry message says independent bulk history. `dg_history.cc` corrects only
  the endpoint initialization label; the stationary BP3 label remains identical.

`run_pf6_transfer.py` and its local/domain-free wrappers accept explicit core
and observer paths, record their identities, and retain original defaults.
These are build-artifact arguments, not numerical configuration switches.
The old derivative/FD, direct-mechanics and quasi-Newton references remain:
callers, benchmark checks and retained reproductions still use them. No unique
diagnostic or legacy backend was removed. No numerical defect was found or
silently repaired within this consolidation.

The previous 1,077-line STATUS is preserved exactly as
`STATUS_THROUGH_20261019.md` (SHA256
`75196219a4e3bf11b102515526d96f643a50ce7bd7e444ed1d8c589ec086129c`).
Current STATUS is a short handoff, rather than several contradictory historical
next actions. Older propagation STATUS and CURRENT_STATUS point to it. Existing
reports, archives and failed results are unchanged; no BP3 package is regenerated.

## Shared-cache and lifecycle audit

| Data / entry points | Valid lifetime and behavior in the selected cohesive route |
|---|---|
| `FEStrainState` / `dg_history` | Material owns accepted T,Z,I and physical phase; DG velocity-history block is unused/zero. `make_fe_endpoint_trial` reads that same anchor for every trial. Surface property storage/committed V remains manager-owned. |
| `fe_normalization.cc` | Ray intersections, shapes and consistent projection belong to fixed geometry/mesh. `evaluate_fe_normalization` reads the **supplied trial phase**, recomputing I and requested first/second derivatives and transpose pullbacks. It does not reuse accepted I as a trial value. |
| `history.cc`, `normalization.cc` | Mechanical preparation refreshes current normalization and surface material data. Shared value reuse requires its collective phase/geometry/material identity; geometric reuse alone is insufficient. |
| `fe_phase_energy.cc` | Explicit endpoint T_e,Z_e enter the partial phase gradient; the mechanical-response action adds their variations. Physical chi is not an independent history or a limited field. |
| `dg_history.cc` | Source geometry/mass reuse, fixed-phase point samples and Stokes-QP stress caches require `uses_stationary_mature_dg_optimizations()`. That predicate excludes consistent endpoint. Explicit trial phase also bypasses fixed-phase point samples. Gauss3/all-nine-mode projection and the operand-derived publication check remain intact. |
| `fe_strain.cc` | Accepted point-stress cache uses accepted T/Z/I/phase only and clears on publication; endpoint stress evaluation takes explicit trial phase. Remap/restart reconstructs mesh-dependent state. |
| Coupling / condensed solver | Each `linearize()` refreshes B,G,K_V. Endpoint unfiltered sparse B/G remain phase-dependent linearizations; stationary filtered-G geometry/friction caches are ineligible. The operator remains C=A-B K_V^-1 G. Homogeneous within-attempt A reuse does not freeze localization, B/G, residuals or active sets. |
| Endpoint solve | Every energy evaluation clears its tangent closure; final preparation restores the chosen endpoint normalization after probes. The new node map caches coordinates only, for one solve; derivative values use the current phase. |
| `core.cc`, endpoint publisher, `dg_accepted.cc` | A complete DG endpoint is pending through the real timestep-manager repeat decision. Accepted state publishes once; rejection/cutback restores entry solution/linearization/statistics and keeps the physical obstacle/history. Native checkpoint clocks, versioned CellId coefficients, surface state and particle/RNG persistence are unchanged. |

Thus the BP3 fixed-phase optimization is not enabled for evolving phase merely
to reuse accelerated paths. No accepted-value cache is made valid by a geometry
hit during a phase trial. The remaining particles are fixed diagnostic and
publication-check/persistence points, with matching-cell traces and no domains;
they do not supply selected material coefficients, mechanics or FE fracture.

## Fresh bounded evidence and commands

All commands run from the repository root. Separate native build:

```sh
cmake -S . -B build-particle-free/evolving-phase/native -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/gcc/12.4.0/bin/g++ -DCMAKE_C_COMPILER=/opt/gcc/12.4.0/bin/gcc -DDEAL_II_DIR=/opt/dealii/9.6-local -DASPECT_WITH_VORO=ON -DVORO_DIR=/home/ein/local/voro++/0.4.6 -DASPECT_USE_FP_EXCEPTIONS=OFF -DASPECT_INSTALL_EXAMPLES=OFF
cmake --build build-particle-free/evolving-phase/native -j 2
cmake -S benchmarks/reconstructed_fault/particle_free -B build-particle-free/evolving-phase/plugin -DAspect_DIR="$PWD/build-particle-free/evolving-phase/native" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/gcc/12.4.0/bin/g++
cmake --build build-particle-free/evolving-phase/plugin --target pf6_transfer_checks -j 1
```

The first configure lacked the Voro hint and was corrected **before building**;
the fresh baseline has the retained Voro/deal.II/compiler configuration. Entry
build/observer cost 432.98/27.98 s; the two changed units rebuild in 64.79 s.
Build logs use `refactoring_boundary/run_logged.py`, labels
`evolving-entry-build-20261019`, `evolving-entry-observer-20261019`, and
`evolving-consolidated-build-20261019`. Native case budgets are 180 s each,
one rank/thread, sequential. Preparation pins executable/plugin/input/source
hashes and refuses existing run directories. For a new label:

```sh
python3 benchmarks/reconstructed_fault/particle_free/run_pf6_domain_free.py prepare NEW --binary build-particle-free/evolving-phase/native/aspect-release --plugin build-particle-free/evolving-phase/plugin/libpf6_transfer_checks.release.so
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py NEW 180 env ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1 OMP_PROC_BIND=false /opt/openmpi/5.0.6/bin/mpirun --map-by core --bind-to core --rank-by slot --report-bindings -n 1 build-particle-free/runs/NEW/aspect-release build-particle-free/runs/NEW/input.prm
python3 benchmarks/reconstructed_fault/particle_free/run_pf6_domain_free.py finalize NEW
```

Actual full labels are `evolving-baseline-20261019` (before runtime edits) and
`evolving-consolidated-20261019`. The existing domain-free fixture uses the
supplied physical seed, unloaded histories, fixed homogeneous laws/loading,
Gauss12 physical / Gauss3 source and dt=.125 s through .25 s. Startup and the
64→256→64 identity diagnostic are retained; physical evolution takes place on
64 cells with fixed surface and 144 stationary points. This is not a new mesh
or timestep study. Initial seed representation is inherited, not a new accuracy
reference. The source change is isolated from diagnostic/build-path changes.

Both full cases pass 100 native gates, including independent complete-Q2
transfer, normalization derivatives, actual rejection/retry and terminal
inconsistent-source rejection. All 162 compared CSVs except transfer timing
are byte-identical; full tensor/phase/gradient/energy/weak-load differences are
zero. Native mechanical residual and phase-search traces agree exactly. Phase
flank growth is .0189269852885, healing zero, final peak .600001937763 and
Z_max=6870.63404561 Pa m. Final native scaled phase KKT=9.27535e-17 versus
target 6.03877e-14. Last mechanical relative bulk/fault residuals are
6.167123e-10 / 1.307184e-16; final integrated surface residual Linf is
4.12115e-13 Pa m. Original acceptance and stability checks pass.

Independent physical Gauss24/32 checks use the retained reference routines:
maximum scaled energy/load errors 6.66e-16 / 3.15e-15 (existing 1e-9 screen);
I/DI/D²I errors 6.19e-16 / 6.47e-16 / 9.62e-16 (existing screens unchanged).
Seed nodal error is zero; its initial physical representation error is
.08019663797 relative L2 and .04702055839 sampled Linf. Final bulk/cohesive/
fracture energies are .791110974827 / .256852544585 / 14.096118461785 J/m.
This checks integration/acceptance and equivalence, not a new global work-balance
or spatial/temporal qualification; retained work evidence is in PF6A_REPORT.

`evolving-half-20261019` uses `--end .125`; `evolving-continued-20261019` uses
`--restart build-particle-free/runs/evolving-half-20261019/run/restart` and the
same explicit binary/plugin. Native filesystem restore advances to step 2,
passes 19 gates and matches all 24 common pre-step/final exports and RNG exactly.
Publication is 2 at step 2. Half/continued cost 10.99/5.83 s; full baseline/
candidate cost 14.49/14.24 s. No speedup is inferred. Finalization must finish
before assessment reads `run.log`; one premature assessment was rerun in that
order, with no native rerun or numerical change.

Fresh native `--test` passes 127 cases / 165094 assertions, label
`evolving-units-20261019`, budget 120 s, elapsed 6.23 s. The complete command is
`run_logged.py evolving-units-20261019 120 env LD_LIBRARY_PATH=/opt/openmpi/5.0.6/lib OMP_PROC_BIND=false build-particle-free/evolving-phase/native/aspect-release --test`.
Candidate core SHA256 is `4b36efad2177deea24bd1aeae81b113644a2e730cca6bd623b04e8600cec6e5b`;
observer SHA256 is `4fd39189b51ff017480b0067936a5f221df75fd3e6662a358a3bf7ac4cfa9ade`.
The unchanged observer is used before/after. Existing 34-file BP3/package/live
build identity hashes pass; server inputs, outputs, checkpoints and packages
were not modified or regenerated. No server action was taken.

[Compact verification and field/source identities](../../../benchmarks/reconstructed_fault/particle_free/evidence/evolving_phase_consolidation/verification.json)
are regenerated without launching ASPECT by:

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/evidence/evolving_phase_consolidation/verify.py build-particle-free/runs/evolving-baseline-20261019 build-particle-free/runs/evolving-consolidated-20261019 build-particle-free/runs/evolving-continued-20261019 --protected build-particle-free/evolving-phase/protected-bp3.json --output benchmarks/reconstructed_fault/particle_free/evidence/evolving_phase_consolidation/verification.json
```

Raw runs/records/build logs remain local at those labels. Existing MPI,
cutback and full-interval work evidence is reused within its documented scope;
this pass does not rerun a convergence/MPI matrix. Preserve the original Gauss6
failure, spatial/endpoint temporal limitations and mirror-branch discrepancies,
including **strict_reference_match_passed=false**. No field alignment or
reference replacement enters the new comparisons.

## Next bounded task: constrained phase variations on a fixed local mesh

The current guard in `solve_reconstructed_fault_endpoint_phase` is necessary:
its phase search treats all physical nodal coordinates as independent. Frozen
64→67-cell remap/restart success in PF6_LOCAL_REPORT does not qualify this solve.
No more immediate functional blocker appeared in the renewed short control;
this constrained solver is the next prerequisite **for a locally refined
evolving-phase experiment**, not for uniform-grid continuation. The mirror/tie
reference question remains a separate scientific accuracy question.

1. Start with the existing small 64→67-cell geometry, fixed before physical
   evolution, fixed surface, homogeneous coefficients, Gauss12/source3,
   unlimited DG and the same physical seed/loading. Require the existing convex
   homogeneous Q1 hanging relations; retain unsupported periodic/moving paths.
   Record initialization representation separately and keep the current guard
   until the constrained route is verified.
2. Use the closed native AffineConstraints to define independent masters psi
   and physical phase phi=C*psi+c, with fixed admissible lifting c where needed.
   Supply C*dpsi to every normalization, energy and mechanical-response action.
   Return g_psi=C^T*g_phi and H_psi*dpsi=C^T*H_phi*(C*dpsi), including fracture,
   normalization second derivatives and mechanical response. The current DG
   cell-fracture/normalization gradients are physical-node derivatives; apply
   the reduction exactly once, not again to an already condensed legacy block.
   Use existing sparse/distributed infrastructure, not a dense global C.
3. Prove obstacle equivalence for the admitted constraints. Masters are physical
   nodes; with nonnegative closed rows and a consistent accepted anchor,
   psi>=psi_n implies C*(psi-psi_n)>=0 at dependent nodes. Conversely physical
   nodal bounds include the masters. Convex rows preserve the upper bound.
   Verify these hypotheses and the actual physical phase after distribution;
   reject unsupported rows/anchors. Do not clip constrained nodes independently.
   Define the reduced active set, KKT, search metric and positive arithmetic
   scales in master coordinates; e.g. pull back positive scales with |C|^T.
   Keep the physical energy, entry accuracy criteria and irreversibility law.
4. Independently verify C expansion/adjoint, constrained FD gradients and full
   reduced actions, including phase-dependent normalization. Require C=Id to
   reproduce the uniform control. Compare a mechanically equilibrated short
   growing case with an independently reduced functional on the same mesh.
   Check physical feasibility, mechanical/fault residuals, true energy/KKT,
   history/stress/loads and all modes. Coordinate-dependent residual norms must
   be derived explicitly, not tuned to pass.
5. Exercise repeated trials, actual rejection/retry, once-only outer publication
   and filesystem restart versus uninterrupted steps on that fixed mesh.
   Separate constrained-solver correctness from branch/accuracy conclusions;
   retain all unregistered mirror differences and the original reference flag.

That proposed task starts serial and does not change mesh during evolution.
Retain representable-phase coarsening only. Automatic AMR, lossy obstacle or
surface transfers, transport, surface birth, geometry motion and physical tip
propagation remain later tasks. No constrained-solver implementation or guard
removal is included in this consolidation.

---

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

## P4 continuation/stability diagnostic — 21 October 2026 (bounded completion)

The default bulk initial guess is dt times accepted velocity, accepted pressure
and accepted phase. Current physical boundary constraints override its boundary
values. Thus its first hold trial can retain a nonzero *interior* displacement
predictor even though boundary motion is zero. Trial constitutive variables are
always re-evaluated from the same immutable bulk owner and phase obstacle.

For constant matched parameters, let Tplus=beta(Tn+2G devsymgrad(du)) and
abar=beta an. The exact local return minimizes

    ||Tplus-2GaS||_3D^2/(4G)+G a^2/(2h)+R |a-abar|.

Its envelope derivatives are beta times physical mechanical stress and -Y.
Including AT1 gives the discrete incremental potential
Eb+Ec+Ef+beta*dt*Dfriction, subject to incompressibility and the physical phase
obstacle. This is a diagnostic of the existing BE equations, not a new viscosity
or jump law. The smooth reduced phase second variation must eliminate mixed
mechanics/pressure and local return response. Existing row scales G/Kf explain
the apparent nonsymmetric cross blocks: multiply mechanical/pressure rows by
beta*G and phase rows by Kf to obtain the symmetric physical tangent.

A test-only ASPECT_BULK_CONTINUATION_TIME selects diagnostic export at one
physical time. ASPECT_BULK_RELAXATION_GUESS then selects zero incremental
velocity, beta-scaled pressure and accepted phase before imposing current
boundary constraints. It changes neither the accepted history nor the default.
The exported tangent excludes constrained/artificial rows after native
condensation. Offline mechanically relaxed critical-space checks must not
condense it twice. Strict positive lower-obstacle multipliers remove phase
critical directions; pressure saddle eigenvalues are not phase instabilities.
Independent point-potential checks cover both stick and slip; native derivative
and stability measurements are recorded in the continuation report/evidence.

The propagation observer connects the already existing post_resume_time_step
signal, reducing only a saved pending interval to the requested maximum dt.
Accepted histories and old_time_step are untouched. Optional compact output
suppresses only dense diagnostic QP CSV, retaining every accepted phase/force/
cell-mean/reaction/history guard and checkpoint. Existing output defaults and
protected mature BP3 paths remain unchanged.

### Critical-space verification correction

The constant absolute phase FD probe can cross Coulomb branches near tiny
positive h; its failures remain evidence. A separate stability check uses
interior/near-critical phase masters and relative-to-obstacle phase directions,
with unchanged epsilon1e-7/5e-8 and both the original 1e-6 scaled bound and an
additional 1e-6 relative bound. The complete operator is P^T J P: restricting
only input directions leaves irrelevant strictly active phase residual rows in
the comparison. That initial incomplete-output test failed (1.9791e-5 relative)
and is preserved. Correct output restriction passes at 8.6009e-7, while reporting
the unprojected error separately. This is a verification-space correction;
Newton, constitutive derivatives, source projection and physical acceptance are
unchanged. Native C^T reduction is already complete and is never applied twice.
Exported master gradients and mass use C^T g/C^T m, with constrained/artificial
rows excluded. Current continuation experiments are uniform Cartesian, four-rank
fixed-mesh controls; no new hanging-node/MPI scope is implied.

The homogeneously weighted BE incremental potential gives positive relaxed
curvature .139907 at the loaded state and negative curvature -.208558 at the
predictor's hold-growth root. Its frozen-mechanics phase block is nevertheless
positive (6.37136). Thus a phase-only observer would miss the coupled instability.
The actual continuation initial guess solves with phase available throughout
Omega; its no-growth result follows strict native obstacle gaps, not a freeze.

### Enlarged-boundary diagnostic correction (P4)

The first 1.75 m control passed the native step-41 mechanics (2.49207e-18),
source and energy checks, then its postprocessor failed the unchanged natural
side residual gate. `MomentCycle::export_state()` still classified x=1.5 as a
free boundary. On the enlarged box that selects interior DoFs at y=.0078125
and .9921875, integrating only their boundary-cell contributions and omitting
cancelling interior cells. The observed false force norm .165726080264519 N/m
matches sqrt(2)*tau*Delta x/3 to 4.49e-13 relative, using the independent
homogeneous first-step Maxwell stress. It also misses the actual x=1.75 side.

The benchmark now selects supported velocity DoFs from native boundary faces
and retains the existing constraint classification and 1e-9 scaled residual
bound. No solver, equation, material or acceptance change is involved. The
failed output and old diagnostic DSO/source are preserved separately; the
corrected observer is built in `bulk-continuation/plugin-boundary`. A matching
one-step 1.5 m replay and fresh same-input 1.75 m run verify the correction.
This is a post-publication diagnostic failure, not an outer-loop rejection;
no checkpoint containing step 41 was published by the failed observer run.


### Completed P4 controls and limits

All three shared-parent timestep branches and the two fresh controls complete
through .17875 s. Predictor hold roots retain negative relaxed curvature under
halving; stable continuation is a test-only native initial guess, not a new
production acceptance policy. The refined/wider loaded curvatures are +.120646/
+2.34054; their holds have strictly positive obstacle multipliers. Exact BE
closure and physical force-displacement work are separate from stability.

The full exported-field scan finds refined onset (.12750,.12875], correcting
the preliminary log estimate. Maximum growth L2 difference on the refined
control is 7.442%; the wider-domain final difference is 94.51% of its growth
norm. Domain growth integral falls 57.26%. No reflection, history interpolation,
new phase cap, limiter, material change or tolerance relaxation is used.
The three edited implementation files are the bulk solver's opt-in diagnostics/
initial guess and the existing parallel/propagation benchmark observers. Point
law and accepted-history-owner files retain their original hashes. Source and
publication checks retain 1e-12 coverage. Frozen ID/coordinate fingerprints,
positive-step pressure gates and once-only ownership/publication are checked;
step-zero ASPECT pressure retains its existing pre-bulk-solve exclusion.

Native full-space FD failures and the incomplete critical-output test remain
recorded. Final Q1 peak errors use union-grid vertices; earlier Gauss-point peak
calculations remain labeled development evidence. Zero interval growth at the
parent is explicitly marked, without a roundoff-normalized error claim. The
[P5 specification](P5_DIRECTION_SPECIFICATION.md) is a conditional tensor-memory
constitutive proposal, not an implementation or a change to this scalar model.
Current quadrature and earlier-prefix uncertainty, branch selection, boundary
sensitivity and prior strict-reference failures are not erased by P4 closeout.
