# R6 refreshed feature-switch inventory

Post-R5 reference `3b4ae16dd`; R6a changes only M4 history CSV presentation.
This supplements and updates §5 / Appendices B–C of the rolling R0 review,
retaining that historical record. `evidence/switch-readers.json` gives exact
current reader paths, source lines and parsing excerpts for every literal
feature environment reader (production, feature headers, tests and tracked
benchmark C++). `parameter-declarations.json` records current declarations and
defaults. Source paths below are relative to the repository. Dynamic rejection
lists are distinguished from algorithm selectors. Source line numbers refer to the R6a working-tree snapshot; the two M4 getenv
sites remain in history.cc.

Conventions: **P** means `getenv` pointer presence, absent off, both empty and
`"0"` on, arbitrary other strings on; no Boolean parsing/validation. **O** is
observation, **V** verification (possibly extra work/failure), **N** numerical
selection, **B** benchmark/setup/loading. Every rank normally runs its reader.
Consistent settings are required where
a branch changes collective participation or distributed numerical inputs; this
is a caller/environment requirement, not a new global agreement mechanism.
Rank-local formatting alone (including both selected M4 streams) does not need
collective agreement; these comparisons enable it consistently to cover all files.
No control is removed or
migrated. SI quantities retain existing units; current literal CSV headers and
formatting remain the schema, never an invitation to rename historical labels.

## Production and feature-header readers

| Exact selector | Owner / readers and read timing | Default / parsing / precedence | Effects, MPI, files and consumers | Disposition |
|---|---|---|---|---|
| `ASPECT_STRESS_CYCLE_TRACE` | M4 `history.cc::compute_history_candidates`, after accepted-state sampling, each k>0 call; core `initial_conditions.cc::interpolate_particle_properties`, each transfer | P; independent streams, same `stress_trace_cells_rankR.txt` whitespace CellIds | O: M4 selected actual-parent input/candidate rows before collective validation/publication; rank query only, no added sampling/MPI. Core records actual support proposals and MPI-ADD publication, remains unchanged. BP5 `analyze_stress_cycle.py`, `analyze_stress_cycle_cells.py`, `analyze_clean_stress_cycle.py` consume them | **Selected M4 extraction only**; core reader retained |
| `ASPECT_FAULT_SOURCE_HISTORY_DIAGNOSTIC` | M4 history candidate preparation, each k>0 call | P, independent of stress-cycle switch | O: only originally inactive surface parents admitted by existing continued bulk-source logic. Rank-local actual stresses/coefficients, before validation/publication. `continued_source_history_K_rankR.csv`; BP3 bottom/top-source analysis and `check_research_restart.py` | **Selected extraction** |
| `ASPECT_FAULT_PERFORMANCE` | M4 history constructor and normalization preparation/integration; M3 manager constructor/projection; M1 particle manager constructor; M5 surface/coupling constructors and sparse setup | P; constructor timer activation additionally gated by active pcout; later reads remain per call, not cached globally | O **with extra work**: rank-local timer scopes, work summaries, RSS/time, normalization/projection counter reductions. Consistent settings required where count collectives are guarded. stdout `Fault: ...`, `Fault quadrature work`, `Fault I_h lookup work`, `Cell I_h`, sparse entries/bytes; refactoring comparers and performance scripts. Not uniformly MPI-free | Retain |
| `ASPECT_DISABLE_IH_VALUE_CACHE` | M4 `normalization.cc::prepare_normalization_reuse` reuse decision | P disables reuse; other exact keys/composition independence/collective agreement still apply | N/reference execution: recompute I_h, change cache/work counts; same intended integrals. Existing MPI reuse ordering retained; cache fixtures | Retain |
| `ASPECT_IH_BASELINE_GUARDS` | M4 normalization preparation | P | V: restore reference support checks; can reject otherwise untested configurations; no new physical parameter. Normalization fixtures/stdout/errors | Retain |
| `ASPECT_IH_COMPARE_CELL` | M4 normalization preparation/selected cell backend | P; only supported cell integration performs remote reference; incompatible with legacy completion guard | V: second integration/remote sampling/collectives, comparison reports and possible mismatch failure. Timings/RSS are not scientific fields; cache/branch work changes | Retain |
| `ASPECT_IH_REFERENCE_FACTOR` | M4 inside enabled supported cell/reference comparison | Absent 1; `stod`, must satisfy `>0 && <=1`; empty/invalid throw, `0` rejected; numeric-prefix parsing follows stod | N/V: tightens only reference quadrature/tail tolerances. No effect without comparison. Used by cell-reference experiments | Retain |
| `ASPECT_IH_COMPARE_SAMPLES` | M4 cell integrator, per integration | P | V: save direct samples, remote-evaluate and compare; extra memory/MPI and assertions. Cell backend tests; stdout comparison | Retain |
| `ASPECT_IH_VERIFY_CELL_QUADRATURE` | M4 after cell integration | P | V: independent fixed 8-panel × 32-point remote quadrature, extra geometry/sampling/collectives and mismatch checks; stdout | Retain |
| `ASPECT_IH_BOTTOM_COMPLETION_DIAGNOSTIC` | M4 legacy completion in normalization; automatic-mode compatibility in boundary_completion.cc; reference BP3 setup | Absent none; **value is a filename**, empty/0 are paths, not off. Nonempty `set_boundary_normalization_completion_file()` path takes precedence in legacy mode; automatic prescribed mode rejects incompatible explicit/legacy inputs | **N/B**, adds outside I_h RHS contributions. Collective file read/distribution, count/origin/value checks; requires frozen mature 2D; legacy env path additionally fresh uniform sliding, no compare-cell. Completion files are numerical inputs, not output. BP3 reference launchers; maintained plugin rejects this env | Retain; never classify as passive diagnostic |
| `ASPECT_BP3_UNIFORM_SLIDING` | M4 legacy-completion admission; reference_200km/uniform_sliding.h | P | B/N: required admission for legacy env completion and benchmark prescribed sliding/setup. All ranks consistent; maintained BP3 rejects presence | Retain |
| `ASPECT_FAULT_LINEAR_PERFORMANCE` | M5 `linear_performance.h::FaultLinearProfile` construction, each nonlinear linear solve | P resets/activates existing thread-local recorder | O: nested factor/inverse/A/B/G/preconditioner/FGMRES counts and seconds; pcout line `Fault linear profile`; destructor reports, no recorder collective. Existing frozen/performance comparers | Retain existing recorder |
| `ASPECT_FAULT_NONLINEAR_DIAGNOSTIC` | M5 reconstructed_fault_stokes.cc, per compatibility/iteration/trial reporting site | P, read repeatedly | O/**V**: active-bound CSV and details; also evaluates surface residual at lower rates, so can prepare caches/filter factors and invoke collectives. `nonlinear_bounds_K.csv` rank0, truncate iteration0 then append, precision17; stdout. Not merely prose; all ranks must agree | Retain; R6b extracts bound CSV/summary only into source-private reconstructed_fault_bound_diagnostics.cc; numerical probe and other output remain in driver |
| `ASPECT_FAULT_COMPATIBILITY_DIAGNOSTIC` | M5 pressure-complement compatibility check, each check | P | O: stdout existing sums/thresholds in local formatted string. Actual compatibility/fresh-residual rejection is unconditional and remains on with prose off | Retain |
| `ASPECT_K1_FLOOR_AUDIT` | M5 driver, selected assembly site | P plus existing diagnostic-site condition | V: shadow residual assembly under process-global `fault_residual_audit_channel`; restore channel before production solve, extra assembly/collectives/output. Not passive logging; exact checks/errors retained | Retain |
| `ASPECT_FAULT_EXPLICIT_B` | M5 reconstructed_fault_stokes assembler, each B linearization/timing setup | P | N: build sparse rectangular B rather than reference quadrature action; physical/homogeneous constraints and cache lifetime unchanged; distributed assembly/communication. stdout size when performance enabled; B/G comparison fixtures | Retain |
| `ASPECT_FAULT_EXPLICIT_G` | M5 surface candidate preparation, each linearization | P **and normal_filters.empty()**; filters retain reference G | N: remote point routing, multiplicity-aware sparse G build/action; collective participation must agree. Same physical pressure convention. Surface/frozen tests; sparse-size stdout | Retain |
| `ASPECT_FAULT_INTERFACE_MODES` | M5 reconstructed_fault_interface_preconditioner.h constructor; frozen test rejects incompatible use | Absent base preconditioner; `atoi` then unsigned, require 1..4; empty/0/nonnumeric invalid, numeric prefix accepted; total modes capped by existing free blocks and global cap8 | N: coarse interface correction, extra B/base/G applications, factorization and collectives; singular coarse model fallback retained. stdout/performance modes; maintained BP3 rejects presence | Retain |
| `ASPECT_FAULT_VERIFY_INTERFACE` | M5 inside each enabled interface-mode response | P, inert without interface modes | V: checks zero pressure from B-generated response; may throw, no independent mode selection | Retain |
| `ASPECT_FAULT_HISTORY_AUDIT` | M5 particle surface backend, each assembly; moments/output only on Jacobian assembly | P | V/O: extra FE tensor remote samples even residual calls; local moments; `history_surface_rankR.csv` truncate, precision17, failbit/badbit exceptions. Particle-vs-FE history research; no material commit | Retain |
| `ASPECT_FAULT_NORMAL_STRESS_DIAGNOSTIC` | M5 particle surface backend, Jacobian only | P + assemble_jacobian | O/V: true-pressure admission check (throws in adiabatic mode); local weighted moments/ranges. `constitutive_normal_K_rankR.csv`, precision17, failbit/badbit exceptions. Normal-stress pilots | Retain |
| `ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC` | M5 particle backend Jacobian, bulk slip residual, and driver's diagnostic call; reference BP3 registers observation | P | O/V: selected low/high samples and weak moments; **driver also invokes discarded bulk residual**, extra material/FE work/collectives. `stress_samples_K_rankR.csv`, `stress_weak_moments_K_rankR.csv`, `bulk_slip_transfer_K_rankR.csv` rank-local replace; sample files throw on I/O errors. BP3 work/replay/source analysis and frozen fixture | Retain |
| `ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION` | M1 RK2 integration, every call; BP5 diagnostic admission | P sets integration dt=0 rather than requested dt | **N**, changes advection/physical execution despite name. Existing benchmark protocols only; maintained BP3 rejects presence. No R6 M1 edit | Retain |

### Selected history stream compatibility

Both M4 streams use default `ofstream::open` truncation, precision17, default
(nonthrowing) stream error masks. Failed opens silently suppress rows via
`is_open`; missing/unreadable selection yields an empty set and a stress header
only if the output opens. A later stream write failure remains silently failed.
A successfully opened stream can still have failbit: retain `is_open`, not a new
`good()` criterion. No new I/O validation or cross-rank synchronization.

- `stress_update_K_rankR.csv`: 30 columns, in existing order:
  `step,time_s,dt,particle_index,particle_id,cell,x,y,ref_x,ref_y,sample_x,sample_y,beta,kappa,grad_xx,grad_xy,grad_yx,grad_yy,old_xx,old_yy,old_xy,eps_xx,eps_yy,eps_xy,crack_xx,crack_yy,crack_xy,new_xx,new_yy,new_xy`.
- `continued_source_history_K_rankR.csv`: 20 columns:
  `id,x,y,phi,chi,V,kappa,beta,eps_xx,eps_yy,eps_xy,crack_xx,crack_yy,crack_xy,old_xx,old_yy,old_xy,new_xx,new_yy,new_xy`.

Time/dt seconds; coordinates metres, reference coordinates dimensionless;
beta and phi dimensionless, chi inverse metres, V metres/second, gradients and
strain/crack rates inverse seconds, stress Pa. Both historical `kappa` columns
contain `eta_ve` (Pa s). Renaming is a future schema-compatibility decision.
No row denotes already committed history: it records a validated local candidate
before collective completeness/final publication. All rows use loop order and
actual parent data; a later failure can leave candidate rows on disk.

Core `stress_transfer_K_rankR.csv` appends without a header, with tagged
`support_proposal` / `published_FE` records at precision17. It opens only with a
nonempty selection. This distinct schema and its shared selector are retained,
not absorbed into the material recorder.

## Parameters, setters and observer alternatives

ParameterHandler readers validate declared patterns at parse time; Boolean
parameters use Boolean parsing, not P. Enumerations reject unlisted strings;
list cardinality and physical checks remain in the owner. The R0 physical
parameter table is retained; the exact refreshed declarations/defaults are in
the JSON snapshot. No duplicate viscosity, timestep, tolerance or friction
parameter is introduced. These are the feature-specific alternatives:

| Owner / selector or API | Default, timing and precedence | Category / dependencies / MPI / outputs / disposition |
|---|---|---|
| Formulation `Enable phase field`, `Reconstruct faults from phase field`, `Use implicit constitutive model` | false, startup parse; existing compatibility validation/dispatch | N, core + M2/M3/M4/M5 admission; all ranks common PRM; retain |
| M3 Fault reconstruction `Boundary completion` | `legacy`; `automatic prescribed` opt-in; startup parse | N geometry/contact qualification feeds M4 completion and M5 terms; automatic mode incompatible with legacy completion inputs. Retain |
| M3 `Structural point spacing` 1000, `Fit prescribed geometry to phase field` true, `Ridge coefficient`1, `Prescribed faults file` empty | startup parse/file load; current patterns/checks | B/N geometry, source/surface admission not interchangeable; file distribution/reconstruction collective. Retain |
| M4 `Fault constitutive mode` cohesive; `Evolve phase field` true; `Use adiabatic pressure in fault friction` false | startup parse; mature mode and frozen compatibility checks | N physics/history/pressure; evolve flag freezes driving history, does not independently skip phase solve. No environment override. Retain |
| M4 `I h integration backend` remote points, `I h quadrature tolerance`1e-8, `I h tail tolerance`1e-8, `I h surface quadrature subdivisions`1 | startup parse; selected cell backend may fall back; diagnostic reference factor only affects reference integration | N normalization with existing cache/MPI rules. Retain |
| M4 friction `Friction law`, `Use regularized formulation`, rate/weakening/state/material lists | current declaration table/JSON, parsed once; surface state fixed through Newton | N: rate-state vs rate-dependent law, no diagnostic ownership. Existing Vmin/Dc/material defaults retained |
| M4 `set_boundary_normalization_completion_file(path)` | empty until benchmark setup; nonempty path overrides env legacy input; automatic mode checks compatibility first | B/N, later collective file read, not a trace path. Retain existing setup callsites |
| M4 `set_reconstructed_fault_background_traction_property(name)` | no property until caller selects registered two-component generic Q1 traction | B/N frozen surface prestress; no bulk history injection. Maintained/reference BP3 setup; generic manager remains unaware of meaning |
| M4 `benchmark_retained_stress` | empty callback; assigned by research replay, invoked synchronously at existing sample sites | B/N overrides incoming stress for retained-history experiments; lifetime/callback caller-owned, nonserialized. Particle backend rejects it. Retain, R6c assessment only |
| M3 `set_shear_sense`, `set_prescribed_slip_rates`, `enable_bottom_source_continuation`, `enable_top_source_continuation` | defaults signed sense+1/no prescribed map/no explicit continuation; setup setters validate existing inputs | B/N updates geometry/admission or nodal constraint data; restart prescribed conditions reattached by caller. Do not migrate these generic mechanisms to BP3 |
| M5 `enable_bulk_work_measure()` | particle measure until explicit call; existing mature/2D/straight/automatic-completion admission | N distinct measures and sampling; invalidates existing linearization at existing point. Retain |
| M5 `set_normal_stress_filter(mode,length)` | raw/0; raw/projected/helmholtz with existing validation; caller-set at setup | N filtered friction input/derivative, caches and generation affected; immutable factors survive trials. Not output-only. Maintained BP3 selects raw or Helmholtz via monitor params |
| M5 `set_normal_traction_diagnostic(window,lines)`, `normal_diagnostic_observer` | empty; setter clears old snapshot; callback invoked after assembly **before factorization/publication** | O/V local capture and existing reductions; callback may write/throw. Not deferred to postprocess. Research native-QP/line consumers retain exact schema/capture |
| Simulator `post_reconstructed_fault_linear_solver` signal | no observer by default; synchronous frozen operator/RHS/preconditioner/direction references at solver site | V/O: tests may do extra solves/collectives and intentional stop. `tests/reconstructed_fault_frozen_gmg.cc`; repaired fixture must prove real AMG/GMG and preservation. R6a does not touch it |
| Stokes solver type block AMG / block GMG / default solver | existing ParameterHandler selection; reconstructed-fault default resolves AMG; fixture default requests hierarchy before resolution | N production selection, no obsolete env override; pressure/outer stopping unchanged. Retain |
| Reconstructed fault time step `Maximum logarithmic state change` | largest finite double; `Patterns::Double()` plus explicit finite positive assertion (zero forbidden) | N optional timestep cap, current state/slip/Dc and collective extrema; R0 limiter conflict was resolved, not still open |
| Reconstructed faults output `Excluded properties`, `output_requested` | empty exclusions; normal schedule unless explicitly gated by benchmark | O, output membership/schedule, registry/history unaffected. Keep consumers/checkpointed benchmark schedule distinct |
| Maintained BP3 monitor `Stationary profile file`, `Friction normal input` helmholtz, `Normal filter length`20, `Bottom velocity constraint`full, `Local state disturbance`0, `Write detailed diagnostics`false | startup parse; profile required by fixture; normal mode raw/helmholtz, length>=0, constraint full/fault parallel, disturbance[0,.02] | B/N/O separated: profile/loading/normal filter/state perturbation affect mechanics; detailed flag enables observers/output. No implicit env override |
| Maintained BP3 mesh `Target cells file`; output/setup `Bottom normalization completion file`, `Mature prestress file`, geometry/profile/loading parameters | declared defaults and readers in plugin mesh/output/bp3 files; collective setup and restart reattachment | B/N: file-backed initialization/loading. Completion remains physical input. Retain; no public callback addition |
| Maintained BP3 `Audit full state every step`false; heavy/profile time and slip intervals; `Last accepted step`2147483647; `Graceful wall seconds`86400 | parse, accepted-state output/termination sites | O with I/O/termination effects; seconds/metres, files `audit_*`, `accepted_steps.csv`, profiles and native output. Work invariants and plotting/restart consumers retain schemas. Retain |
| R6a fixture `Postprocess/History trace cells/Write selection` | true; bool parse; pre_set_initial_state writes owned cell IDs; false leaves missing input | Test-only O input, no physical change/collective beyond rank query. New fixture closes selected-cell coverage gap; not production configuration |

The full maintained-plugin target is `bp3/plugin/CMakeLists.txt`; sibling BP3,
reference_200km, BP5 and investigations files are separately built research/test
packages, not implicitly linked into maintained BP3. The live dynamic reader
`plugin/execution_environment.h::unexpected_execution_switch()` rejects presence
(including empty/0) of freeze-advection, legacy completion, uniform sliding and
interface modes. It does not implement another algorithm for those names.

## Historical names without a current production reader

`ASPECT_FAULT_SURFACE_SOLVER`, `ASPECT_FAULT_COMPARE_SURFACE_INVERSE`, `ASPECT_FAULT_VELOCITY_GMG`, and
`ASPECT_FAULT_GMG_HIERARCHY` occur in old launchers/reports (the first remains in
some recorded frozen environments). They have **no production getenv reader**;
setting them does not choose the current solver. Use the real Stokes solver
parameter and repaired hierarchy fixture. Keep archives, do not revive switches
or silently relabel AMG as GMG. Compiler definitions selecting historical BP3/BP5
plugin variants are build-time package controls, not startup environment readers.

## Separately built benchmark/test readers

All entries below are **retained**. They are available only in their owning
plugin/test/build variant; their presence in this table does not mean maintained
BP3 loads them. B/V is the conservative primary classification for experimental
controls: some replay/restore history, change laws/source/clock, add evaluations
and collectives, or deliberately terminate. Only output-specific entries are O.
All distributed experimental paths require identical selector/inputs across
ranks unless the owner explicitly restricts work to rank0; no rank-agreement
check is inferred. P controls have the exact absent/empty/0 semantics above.
Each fixture's assertions and exception/intentional-stop behavior are retained.

Reader locations (and exact parsing context) are in switch-readers.json. The
table groups ownership by actual source rather than treating all names beginning
with ASPECT_FAULT as production. The following reader families establish timing,
outputs and consumers; original CSV header literals retain SI/unit conventions
(especially old kappa labels) and are not rewritten in R6.

| Reader family / owner | Evaluation point and effects | Existing output / consumers |
|---|---|---|
| bp3/bp3.cc, work_replay.h, tests/bp3_length_scale_checks.h | Accepted-state/first-update and startup law checks; short-test checkpoint/stop behavior; full audit selection. B/V/O | Work/history invariants, audit particles/bulk, accepted_steps, work replay scripts. Separate research build, not maintained plugin |
| bp3/disturbance_diagnostic.h | Initialization, timestep proposal, accepted perturbation/control checks; can override aging/normal input and prescribed dt. B/N/V | disturbance_nodes and diagnostic reference CSVs; disturbance experiment launchers/analysis |
| bp3/investigations/{cohesion,history_load,history_mechanics}_diagnostic.h | Step0 or restored step12 audits, alternate retained stress/cohesion/load and frozen mechanics; extra material/solve work, errors and intentional stops. B/V | Frozen cohesion input, bulk_owned_rankR.csv (throwing output), history-load/mechanics reports; replay investigations |
| bp3/investigations/{junction,theta_exact,theta_history,within_step}_diagnostic.h | Selected step0/2/10/12/13 or replay target; finite differences, alternate state/history, repeated mechanics, restores and noncommitting/commit choices. B/V/N, not passive CSV | derivatives.csv, notch_input_properties.csv, state_qp_rankR.csv, candidate_commit_K.csv, fault/state/Theta snapshots; run_trace_replay, run_coupled_substeps and related analysis |
| bp3/reference_200km/{bp3.cc,uniform_sliding.h,matched_resolution.h,replay_time_step.h,refresh_check.h} | Mesh/setup/source/clock/restart/refresh checks; saved-clock matching or adaptive cap, source extension, mesh-only stop. B/N/V | mesh_guard_rankR.csv, clock/accepted-state/source files, matched mesh/fault input; frozen_gmg_repair and reference BP3 scripts |
| bp5/{clean_stress_cycle.cc,stress_cycle_diagnostic.cc,steady_initialization.h,startup_time_step.cc,test_traction_projection.cc} | Initialization/restart first pending step, captured update/transfer/traction checks; may freeze advection or set dt/source. B/V | stress_trace_cells_rankR.txt, stress_particles_before/after, inventory/clock, transfer/affine/traction CSV; analyze_stress_cycle*, analyze_clean_stress_cycle and offline traction audit |
| uniform_shear/uniform_shear.cc, evolving/seam-audit/diagnostic.cc | Accepted-state guards or periodic-image correction of audit coordinates (not production mapping). V | K3/K4 seam/state diagnostics; existing shear tests |
| tests/phase_field_fault_{boundary_completion,ih_cache,surface_system}.cc | Setup/cache replay, filter/reversed-shear mutation, basis/random B/G comparisons; extra solves/collectives. B/V/N | geometry, filter_{projected,helmholtz}.csv, saved FE/surface inputs and stdout pass/error markers |
| tests/reconstructed_fault_{frozen_gmg,mechanical_modes}.cc and frozen_profile.h | Synchronous borrowed frozen solver observer, selected step/Newton; fresh residual probes, alternate test preconditioner and deliberate stop; profile export/replay. V/B | frozen_gmg.csv (seconds excluded only), rank binary state/directions/operator data, mechanical decomposition/profile CSV; repaired comparator and mechanical-mode scripts |
| unit_tests/{phase_field_fault_ih,reconstructed_fault}.cc; tests/phase_field_periodic_domains.cc | Only selected tests: saved-state quadrature/support audit, mapping/lookup variant, remote-periodic assertion. V/N | Saved profile/geometry files, timing/digests and Catch/stdout assertions; hidden units/periodic fixture |

| Exact test/research selector | Current readers | Parsing / timing / purpose refinement |
|---|---|---|
| `ASPECT_BP3_ADAPTIVE_REPLAY` | `bp3/reference_200km/replay_time_step.h` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_BP3_ALL_SOURCE_QPS` | `bp3/reference_200km/uniform_sliding.h` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_BP3_BOTTOM_SOURCE_CONTINUATION` | `bp3/reference_200km/bp3.cc` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_BP3_COUPLED_STATE_INPUT` | `bp3/investigations/within_step_diagnostic.h` | Replay directory used only when COUPLED_STATE_REPLAY present; absent returns before replay; empty/0 literal paths. |
| `ASPECT_BP3_COUPLED_STATE_REPLAY` | `bp3/investigations/within_step_diagnostic.h` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_BP3_EARLY_TRACE_STEP` | `bp3/investigations/junction_diagnostic.h` | Absent target10; present must equal string 2 and satisfy free/noncommitting/no-within-state guards; empty/0 rejected. |
| `ASPECT_BP3_EXACT_TARGET` | `bp3/reference_200km/matched_resolution.h` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_BP3_EXPECTED_FAULT` | `bp3/reference_200km/matched_resolution.h` | Optional path; absent skips coordinate audit; read in exact-target setup; empty/0 literal paths. |
| `ASPECT_BP3_EXPORT_CHECKPOINT_BULK` | `bp3/investigations/history_mechanics_diagnostic.h` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_BP3_FROZEN_SURFACE` | `bp3/investigations/history_mechanics_diagnostic.h` | Optional directory, absent off; selected once ready at step12; empty/0 literal paths; validation can throw. |
| `ASPECT_BP3_FULLY_FRICTIONAL_REPLAY` | `bp3/investigations/cohesion_diagnostic.h` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_BP3_HISTORY_LOAD_DIAGNOSTIC` | `bp3/investigations/history_load_diagnostic.h` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_BP3_JUNCTION_DIAGNOSTIC` | `bp3/investigations/junction_diagnostic.h` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_BP3_LENGTH_COUPLED_DIAGNOSTIC` | `bp3/bp3.cc` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_BP3_LENGTH_FULL_AUDIT_FROM` | `bp3/bp3.cc` | Absent: no extra full audit; stoul threshold step, empty/invalid throws; each accepted output; audit_states also enables. |
| `ASPECT_BP3_LENGTH_QUALIFICATION` | `tests/reconstructed_fault_mechanical_modes.cc` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_BP3_LENGTH_STUDY` | `tests/bp3_length_scale_checks.h`, `tests/reconstructed_fault_mechanical_modes.cc` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_BP3_MESH_ONLY` | `bp3/reference_200km/matched_resolution.h` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_BP3_NOTCH_BOUNDARY_PROBE` | `bp3/investigations/within_step_diagnostic.h` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_BP3_REFRESH_TEST` | `bp3/reference_200km/refresh_check.h` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_BP3_TARGET_MESH` | `bp3/reference_200km/matched_resolution.h` | Required path in saved-mesh plugin; absent rejected; read at mesh setup and exact-target check; empty/0 literal paths. |
| `ASPECT_BP3_THETA_EXACT_DIAGNOSTIC` | `bp3/investigations/junction_diagnostic.h`, `bp3/investigations/theta_exact_diagnostic.h` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_BP3_TIMESTEP_SEQUENCE` | `bp3/reference_200km/bp3.cc`, `bp3/reference_200km/replay_time_step.h`, `tests/reconstructed_fault_mechanical_modes.cc` | Required filename at replay plugin initialization; absent or unreadable throws; path plus step/clock validation. Reference setup can select replay by presence. Empty/0 literal paths. |
| `ASPECT_BP3_TOP_SOURCE_EXPERIMENT` | `bp3/reference_200km/bp3.cc`, `bp3/reference_200km/uniform_sliding.h` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_BP3_WITHIN_STEP_DIAGNOSTIC` | `bp3/investigations/junction_diagnostic.h`, `bp3/investigations/within_step_diagnostic.h` | Both P admission and replay directory; absent off; read at selected trace step. COUPLED_STATE_REPLAY redirects file input to COUPLED_STATE_INPUT. Empty/0 are literal paths. |
| `ASPECT_BP3_WORK_MEASURE` | `bp3/investigations/junction_diagnostic.h` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_BP5_INITIAL_REFERENCE` | `bp5/steady_initialization.h` | Required saved initialization coefficients file for selected normal-control setup; absent/unopened throws; empty/0 literal paths. |
| `ASPECT_BP5_SHORT_TEST` | `bp3/bp3.cc`, `bp3/work_replay.h`, `tests/bp3_length_scale_checks.h`, `tests/reconstructed_fault_mechanical_modes.cc` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_BP5_TIMESTEP_AUDIT` | `bp5/startup_time_step.cc` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_DISTURBANCE_CONTROL` | `bp3/disturbance_diagnostic.h` | Exact string state or normal; absent/other (including empty/0) neither override; read per use. |
| `ASPECT_DISTURBANCE_DT` | `bp3/disturbance_diagnostic.h` | Required stod dt at timestep execute; existing reaction verifies schedule; no absent fallback. |
| `ASPECT_DISTURBANCE_EPS` | `bp3/disturbance_diagnostic.h` | Required stod amplitude each use in selected plugin; no safe absent fallback. Empty/nonnumeric throws. |
| `ASPECT_DISTURBANCE_RATIO_LIMIT` | `bp3/disturbance_diagnostic.h` | Required stod maximum V dt/Dc, used in accepted-state assertion; invalid parse throws, no absent fallback. |
| `ASPECT_DISTURBANCE_REFERENCE` | `bp3/disturbance_diagnostic.h` | Required directory in selected control; read per matched-state update, missing files throw; absent unsupported, empty/0 literal path. |
| `ASPECT_DISTURBANCE_WAVELENGTH` | `bp3/disturbance_diagnostic.h` | Cached first call; absent 200 m; stod then finite positive check; empty/0 invalid. |
| `ASPECT_FAULT_COMPARE_COUPLING` | `tests/phase_field_fault_surface_system.cc` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_FAULT_FREE_TRACE_DIAGNOSTIC` | `bp3/investigations/junction_diagnostic.h`, `bp3/investigations/within_step_diagnostic.h` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_FAULT_FROZEN_COHESION_DIAGNOSTIC` | `bp3/investigations/cohesion_diagnostic.h` | Optional filename, absent off; selected at step0; empty/0 literal paths, file/row validation retained. |
| `ASPECT_FAULT_HISTORY_FE` | `bp3/investigations/junction_diagnostic.h` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_FAULT_NONCOMMITTING_DIAGNOSTIC` | `bp3/investigations/junction_diagnostic.h`, `bp3/investigations/theta_exact_diagnostic.h` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_FAULT_PERFORMANCE_STATE` | `unit_tests/phase_field_fault_ih.cc` | Required snapshot directory in selected one-rank unit fixture; absent/missing rejected; empty/0 paths. |
| `ASPECT_FAULT_THETA_HISTORY_DIAGNOSTIC` | `bp3/investigations/theta_history_diagnostic.h` | Optional file at first step0 initialization, absent off; repeated initialization rejected; empty/0 literal paths. |
| `ASPECT_FAULT_WITHIN_STEP_STATE` | `bp3/investigations/junction_diagnostic.h` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_FROZEN_GMG_NEWTON` | `tests/reconstructed_fault_frozen_gmg.cc` | Each observer: absent4, atoi otherwise; empty/nonnumeric gives0, numeric prefix accepted; unsigned conversion retained. |
| `ASPECT_FROZEN_GMG_STEP` | `tests/reconstructed_fault_frozen_gmg.cc` | Each observer: absent2, atoi otherwise; empty/nonnumeric gives0, numeric prefix accepted; unsigned conversion retained. |
| `ASPECT_IH_CARTESIAN` | `unit_tests/phase_field_fault_ih.cc` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_IH_LOOKUP_BASELINE` | `unit_tests/phase_field_fault_ih.cc` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_IH_SAVED_PHASE` | `tests/phase_field_fault_ih_cache.cc` | Optional snapshot path, absent manufactured fixture; restores FE data inside cache test; empty/0 literal paths. |
| `ASPECT_IH_SAVED_SURFACE` | `tests/phase_field_fault_ih_cache.cc` | Required coordinate path when saved phase selected; absent or unopened throws; empty/0 literal paths. |
| `ASPECT_K4_STATE_GUARD` | `uniform_shear/uniform_shear.cc` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_MECHANICAL_DECOMPOSITION` | `tests/reconstructed_fault_mechanical_modes.cc` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_MECHANICAL_EXPORT_PROFILE` | `tests/reconstructed_fault_frozen_profile.h` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_MECHANICAL_FROZEN_PROFILE` | `tests/reconstructed_fault_frozen_profile.h` | Required directory if restore/verify enabled; absent rejected at load; otherwise P enables verify; empty/0 literal paths. |
| `ASPECT_MECHANICAL_PROBE_NEWTON` | `tests/reconstructed_fault_mechanical_modes.cc` | Each observer: absent6, atoi otherwise; empty/nonnumeric0; no new validation. |
| `ASPECT_MECHANICAL_PROBE_STEP` | `tests/reconstructed_fault_mechanical_modes.cc` | Each observer: absent11, atoi otherwise; empty/nonnumeric0; no new validation. |
| `ASPECT_MECHANICAL_WIDTH_PROBE` | `tests/reconstructed_fault_frozen_profile.h`, `tests/reconstructed_fault_mechanical_modes.cc` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_SAVED_FAULT_SUPPORT_AUDIT` | `unit_tests/reconstructed_fault.cc` | Required saved geometry directory in selected unit audit; absent/missing rejected; empty/0 paths. |
| `ASPECT_TEST_BOUNDARY_H_DRIVEN` | `tests/phase_field_fault_boundary_completion.cc` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_TEST_NORMAL_FILTER` | `tests/phase_field_fault_surface_system.cc` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_TEST_REVERSED_SHEAR` | `tests/phase_field_fault_surface_system.cc` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `ASPECT_TRACTION_PROJECTION_AUDIT` | `bp5/test_traction_projection.cc` | Required offline directory; rank0 manifest load after rank query; absent/missing rejected; empty/0 literal paths. |
| `K3_CORRECTED_PERIODIC_AUDIT` | `uniform_shear/evolving/seam-audit/diagnostic.cc` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `K3_REQUIRE_REMOTE_PERIODIC_IMAGES` | `tests/phase_field_periodic_domains.cc` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |
| `STRESS_TEST_EXPECT_AFFINE` | `bp5/stress_cycle_diagnostic.cc` | P at the owning family’s entry/callback above; selects its named fixture branch/check, not a production default. |

`ASPECT_BP3_ADAPTIVE_REPLAY` specifically changes the replay clock from strict
step indexing to next saved physical-time cap; ordinary safety controllers still
apply. `ASPECT_BP3_BOTTOM_SOURCE_CONTINUATION` is also enabled by a nonempty
completion-file parameter in the reference plugin. `ASPECT_BP3_TOP_SOURCE_EXPERIMENT`
adds top source only under its existing uniform/replay admission. `ASPECT_BP3_ALL_SOURCE_QPS`
changes source admission for uniform experiments. `ASPECT_BP5_SHORT_TEST` also
changes termination/checkpoint setup in its compiled variants. `ASPECT_BP3_EXACT_TARGET`
validates saved mesh/fault, and `ASPECT_BP3_MESH_ONLY` deliberately stops before
mechanics after output. `ASPECT_TEST_BOUNDARY_H_DRIVEN` omits the usual fixed-phase
constraint fixture. `ASPECT_TEST_REVERSED_SHEAR` changes shear sense;
`ASPECT_TEST_NORMAL_FILTER` selects bulk work/filtered test equations.
`ASPECT_FAULT_COMPARE_COUPLING` adds independent actions/collectives. None of
these is reclassified as harmless merely because it is test-only.

### R6b bound-output boundary

`ASPECT_FAULT_NONLINEAR_DIAGNOSTIC` remains a repeatedly read presence selector
(including empty/"0"), with the same collective consistency requirement. Only
file/header/row/summary formatting moves. The driver owns the per-iteration
stream, guards, lower-rate audit and current Newton state. Summary uses the same
ConditionalOStream directly; no formatting-state reset. Silent open/write errors
remain silent. All 11 CSV columns and row order remain available to BP3
`bound_roundoff_audit.py`, `bound_contact_followup.py`, `analyze_cache_audit.py`,
and BP5 `analyze_small_startup.py`. Lower-rate audit output is not a committed
state or accepted trial. Linear/nonlinear detail and trial-merit readers/output
are unchanged. See [R6b evidence](../refactoring_r6b/README.md).

### R6c benchmark boundary disposition

Accepted R6b source is `00ad5ce1c`. [The R6c assessment](../refactoring_r6c/README.md)
locates existing benchmark policy in maintained BP3, historical reference BP3
and the BP5 retained-stress fixture, and distinguishes those policies from M3/M4/M5
mechanisms. No readers/defaults/parameters or source paths change in R6c.

The legacy completion environment adapter remains in M4 normalization. A
nonempty explicit setter path takes precedence; the member-empty environment
branch retains fresh/uniform-sliding admission and cache-miss read timing.
The explicit setter is not an equivalent legacy adapter: it permits reattachment
on restart and changes path validation/invalidation behavior. Supported BP3
attachment already uses `post_advection_solver`; maintained BP3 rejects legacy
environment presence. No existing callback supplies actual private profile
integrals before projection, so completion application/output also remains.
The automatic-mode compatibility check still precedes collective cache reuse.
Any later switch migration requires its own compatibility decision; no new
signal or accessor is proposed. Source/hash evidence and environment-guard
coverage are linked from the assessment. R7 is unselected.
