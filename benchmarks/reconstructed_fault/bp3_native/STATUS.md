# Local accelerating slip-front diagnostic — completed bounded task, 2026-10-15

[Report](slip_front_report.md): coupled correctness passes41/67 positive steps toT=2e7s,
using physicalVp, filter20 and4×4/min12/max24. The isolated diagnostic selects
ordinary spatial weakening/strengthening, a Gaussian log-Omega state pulse and
explicit fixed nodal initial shear. These are not official BP3 initial data.
Peak V=21.253/20.106Vp; moving2/5/10Vp contours establish a finite slip pulse
on fixed geometry. All recurrence/state/slip/publication, phase/I_h, domain/cache,
residual and selected tangent gates pass. No crossings, births, removals or
multiple-trace events occur; population effects cannot explain this trajectory.

**Accuracy exit remains open.** Tightening predictive state bound.2→.1 advances
the peak by1.022e6s; common-time V difference reaches43% of the primary pulse
peak. Final V/theta/slip sensitivities are35.1%Vp/10.0% ofDc/Vp/1.86% of peak slip.
Actual max log-state changes.272/.146 exceed predictive bounds during acceleration.
No growing alternating slip-rate wave is observed; the sharp lower-end pressure
feature persists (raw/filtered chord3.686/.260 versus3.690/.264MPa). Endpoint
spatial accuracy is unqualified, and absence of a short-test wave is no cure.

Four plugin files change only behind the diagnostic compile definition, plus two
inputs, analysis/report and recovery notes. The coupled observer/core and all
production defaults/laws are retained. Initial/final plugin+observer builds,
exact step0/1 final-source gate,12 legacy preprocessing comparisons, ASPECT style,
Python analysis/syntax and whitespace checks pass. Three launches exit0, total
2050.86s excluding compilation; at most two ASPECT processes overlap. All incoming
work, original binaries and5×5 evidence remain; preservation checks cover6628 files.
Evidence/provenance: `build-bp3-native/slip-front/` and
`refactoring_boundary/evidence/bp3-slip-front-*`.

Next proposed scope: resolve this pulse's temporal sensitivity with consistently
tighter active restrictions, retaining sparse states for a later endpoint audit.
No extra variation, density sweep, production/full-model, unfiltered, MPI,
restart/cutback or first-event campaign is launched. Stop at this selection.

---
# Managed BP3 fixed-state event / short timestep audit — completed 2026-10-14

[Report](managed_event_report.md): keep 4×4/min12/max24; retain the completed 5×5 control.
One baseline-only replay through step44 acquires previously unsaved complete
particle/FE/RNG snapshots. All 45 accepted surface profiles match the retained
trajectory exactly. No second long density trajectory was run.

At fixed u/p/phase/V/theta/I_h, 316 births and 236 removals preserve all 65445
survivors exactly but change lower-window reconstructed history by 19.868 MPa
maximum (.333 MPa weighted RMS). Maximum nodal shear/raw/filtered load changes
are 159.32/92.37/29.57 kPa; pressure is unchanged. Native weak loads are measured
separately. A real 3.232-MPa near-face event reproduces eligible-trace jumps;
independent FEValues agrees with the average, while a one-particle substitution
changes the assembled residual by 1.027e5 Pa m. That probe is in the interior;
worst older lower-end 5.35/10.38-MPa events are not claimed reproduced.

The short native-controller pair completes T=2e7 s in 3 versus10 positive steps,
with identical step-zero initialization and one publication per accepted step.
Final maximum V/theta/slip sensitivities are .522/.444/.407% on stated scales;
all residual/history/state/domain checks pass. There are no population events
in this short early interval; late management temporal accuracy remains open.
Both observer builds and exact one-step gates pass; source changes are diagnostic
only (manager test friendship, existing observer, three fixtures and analysis).
Original binaries, physical settings, sampling rule, completed evidence and all
unrelated work are preserved. At most three serial simulations overlap.

Evidence/provenance: `build-bp3-native/managed-event-audit/assessment.json`,
`incoming.json`, `run-artifacts.json`, `final-artifacts.json`, `preservation.json`,
and `refactoring_boundary/evidence/bp3-event-*`. No selected work remains.
Next proposed scope: use saved fixed states to resolve lower-end reconstruction/
trace-gradient spatial sensitivity before increasing particle density. The
controlled-error exit remains open; no first-event, full-model, MPI, restart or
cutback run starts. Future long runs retain the at-most-four-process preference.

---
# Coupled managed-particle BP3 — completed bounded test, 2026-10-14

[Report](managed_particles_report.md): both strengthening trajectories complete80 positive steps to
T=14071021411.032413 s with identical accepted schedules. Native RK2 and12–24
management from16/cell exercise60001 crossings,1161 insertions,1087 deliberate
removals and830 outflows; the25/cell control exercises90275/1775/1366/1351.
All survivor/birth/history, geometry/cache, residual and once-only stress/state/
slip checks pass.700 assertions/11 unit cases, final builds and bit-identical
one-step gates also pass. All incoming P2/BP3 work and original artifacts remain.

The controlled-error exit remains open: final maximum V/slip density sensitivities
are5.97%/.270%; peak transient V is16.79%. Final raw/filtered normal sensitivities
are27.20%/9.57% of50 MPa, concentrated at the lower endpoint. Near-Nyquist roughness
and pressure/tensor cancellation remain unresolved.16/cell is not accuracy-qualified.
Next proposed task: isolate lower-endpoint weak-load/reconstruction changes across
population events and targeted particle/bulk/surface refinement. Await selection;
no full-model, first-event, unfiltered, MPI transport, restart or cutback campaign.
Future long tests: parallel execution with at most four processes.

The default-preserving common D3 selector keeps stress particle-owned; moving
birth H now interpolates. Stationary results/default strain selection remain.
The host-cell-only step21 diagnostic failure is preserved and resolved by checking
the native averaged FE sampling rule, with no production tolerance change.
Complete evidence, source changes and restrictions are in the linked report and
`build-bp3-native/managed-coupled/assessment.json`. Exact run commands/results are
in `refactoring_boundary/evidence/bp3-managed-*.{json,log}`. Stop at this scope.

---
# Domain conditioning and CPDI operator follow-up — 2026-10-14

The [bounded follow-up](geometry_report.md) completes the formerly failing dense
shear through all 32 positive steps to T=1, with the original 1e-10 guard intact.
The full 2,304-site patch reproduces the saved error. Consistent scale/translation
tests identify Voro++ absolute-tolerance sensitivity; centered coordinates in
typical particle spacings resolve it without changing neighbor/image admission.
Local moments, symmetric differences, CPDI and H weak contributions are compared
with independent clipping at every rebuild, including periodic seams and walls.

Operator checks also expose an independent orientation bug: 8,011/16,384 old
domains give d(y)/dy=-1. The 2-D simplex gradient now uses positive physical area.
Corrected worst local area error is 2.18e-14 and dimensionless CPDI gradient error
8.42e-14; absolute local weak bounds exclude cancellation. The matched p8/p16
pair reduces final pulse weak error .607266% → .160968% and temporal maximum
2.180573% → .543435%. Particle values remain exact and Maxwell errors retain
their established temporal trend.

Actual tests: 841 assertions/12 native cases pass, including five native
coordinate configurations. The new linear-gradient regression fails against
the original implementation as expected. Isolated native compilation/relink,
plugin build, style/syntax/whitespace and analysis pass. Six bounded simulation/
unit launches total 46.32 s excluding compilation. Core/previous plugins/server
archive and all incoming work remain preserved; corrected binaries, exact
commands, hashes and checks are separate in `build-bp3-native/transport-geometry/`.
Only particle-domain numerics, focused tests/diagnostics and recovery docs change.
No physical-law or tolerance change, complete P2 rerun, MPI transport or coupled
BP3 run is made. Next proposed scope remains common D3/G consistency and actual
coupled moving-history publication; stop before that subsequent selection.

---
# Prescribed moving-particle checks — 2026-10-14

Resumed the interrupted bounded transport task on `fault-propagation` at
`c33edaaeb`, preserving all incoming P2/BP3 edits and stationary evidence.
[Transport report](transport_report.md): seven serial prescribed-flow controls
pass with actual cell crossings. Translation particle stress is accurate to
8.15e-15 Pa; bulk refinement reduces FE reconstruction RMS .004648 to .001199 Pa.
The subcell pulse retains particle values exactly, while its actual CPDI weak
vector error decreases 11.5768% → 1.51648% → .359093% under particle refinement.
Shear-ramp stress errors nearly halve with each timestep halving, separately
from FE reconstruction and RK2 predictor errors.

The overall gate remains open. A warranted denser shear control stops at step 2,
t=.0625 s: summed domain area has a 6.24544e-9 deficit, exceeding the new test's
1e-10 guard, although below the native Debug 1e-6 guard. CPDI basis sums are
accurate to 4.44e-16. Two diagnostic replays and independent clipping localize
a discrepancy to a saved Voronoi polygon; no tolerance or provider change is
made. Last complete diagnostic row is t=.03125 s; rollback is not claimed.

Actual verification: plugin build/style/whitespace checks pass; seven completed
controls pass analysis, which exits 1 for the retained incomplete control.
Eleven launches including parse failure and diagnostic replays total 76.76 s.
Only a test-access wrapper, new isolated fixtures/observer/analysis and recovery
documents change; no production algorithm/default changes. Exact commands,
hashes and preservation checks are in `build-bp3-native/transport-checks/`.
The written law is non-rotational, so no objective-rotation qualification follows.
General D3/G consistency, actual coupled moving-history publication, MPI,
population changes and restart remain unqualified. Next proposed task is a
bounded frozen-entry Voronoi precision/conditioning diagnosis; no new run or
full BP3/first-event campaign starts automatically.

---
# Moving-particle formulation audit — 2026-10-14

Selected scope is documentation before implementation, from fault-propagation
c33edaaeb plus incoming edits. [Equation/ownership/update note and change list](moving_particle_formulation.md)
traces pre-mechanics RK2 transport, population/ghost/domain management, old-stress
FE reconstruction, common bulk/BP3 work-measure consumption, terminal particle/
surface publication and accepted-step slip. Current strain already comes from
FE velocity; particle histories, CPDI and surface state need no ownership redesign.
No d_R or FE-derived replacement stress belongs in the moving branch.

Two findings need resolution before new mechanics: select one 3D deviatoric
convention independently of history ownership, correcting the raw-gradient G
normal-traction tangent too; and explicitly retain the written non-rotational
law or select an objective replacement with its stress/background scope. The
custom path currently has no spin/rotation, consistently with current_design
section 19 and pf_rsf.tex. Standard ASPECT elastic rotation is not called here.
No new physical law is inferred. The note gives the consistent moving split of
the current non-rotational model and identifies the objective-extension decision.

Actual checks: static call-path audit, arithmetic tangent (.8 versus current
raw-G 2.4) and rigid-rotation illustrations; no ASPECT execution or compilation.
Preservation manifests and arithmetic evidence are under
build-bp3-native/moving-particle-audit/. All incoming source/input changes,
completed P2/BP3 evidence, binaries and stationary server archive are retained.
Only the audit note and two status/recovery documents change. The server input
remains stationary; no moving-particle or first-event test was launched. Next
step is the rotation/background decision followed by scoped convention/tangent
implementation and focused motion/publication gates, under a subsequent selection.

---
# Server deal.II 9.7 compile compatibility — 2026-10-14

The reported server build finds the removed ParameterHandler::Text alias in
unit_tests/phase_field_fault_maxwell.cc. Replaced it with ParameterHandler::PRM,
the identical format already used in ASPECT production code. Local standalone
syntax compilation of the complete corrected test translation unit passes with
deal.II 9.6.2 (6.35 s); the official 9.7 header confirms PRM is supported.
The separate [patch](server/dealii-9.7-prm.patch) dry-runs against the original
packaged source; git diff --check passes. Original archive, checksums, retained
executables and physical evidence are unchanged. No full core rebuild or 9.7
execution was performed locally. Next step: apply this source edit on the server
and resume its existing manual build; any further 9.7 compatibility issue remains
unverified. Scientific source/settings and all incoming edits are preserved.

---
# Fine full-mesh server handoff — prepared for manual use, 2026-10-14

User selected a self-contained ASPECT/plugin/input/environment package for manual
server builds and the first-event test. Current c33edaaeb + completed local FE
edits are included; no scientific source/default changes. Standalone input uses
fine3.90625 m/surface10/q6, actual BP3 transition, filter20/full bottom and native
stationary FE history. No moving particles, births, checkpoint/resume or automatic
cutback. Plain manual CMake/run commands target the requested WORK/SCRATCH paths.
Source snapshot configuration, rebuilt ordinary plugin, relocated parameter
validation/query and environment checks pass. No full ASPECT rebuild, fine-mesh
simulation, server test or submission was performed. Archive/inventories under
build-bp3-native/server-package/; instructions and exact checks in [server/README.md](server/README.md)
and [server/VERIFICATION.md](server/VERIFICATION.md). Spatial accuracy remains
provisional; a wall-limited incomplete event has no supported FE continuation.
Next step: user manually builds with the matching server dependency stack,
validates and submits in the selected allocation. Preserve all logs/identities.

---
# BP3 section 8 — complete bounded diagnostics / fresh preparation

Baseline c33edaaeb on fault-propagation; incoming 2385-file manifest and seven
retained binaries preserved. [One report](section8_report.md) and evidence under
build-bp3-native/section8/ record settings, commands, failures and measured fields.

| Actual check | Result |
|---|---|
| Ten-step raw control, dt 200000 to T 2000000 | Pass, 510.27 s; identical phase/I_h/mesh/rule/particle identity; final max V difference .56945%Vp; endpoint V chord 2.46e-12→8.19e-12 m/s |
| Observer-only filtered replay | Pass, 61.64 s; retained step-zero profiles/journal row byte-identical |
| Profile-only h 3.90625→1.953125 m | Pass, 10.44/22.97 s; both admitted endpoints, same surface/physical profile; min I_h +1.37%, peak-scaled chi difference 3.71%; resolution provisional |
| Fresh ordinary full BP3 input | Parameter validation passes; actual geometry/transition/state/loading; no local-test definitions; stationary/no-birth FE history, filter 20/full bottom, no FE resume/checkpoints |
| Coarse full geometry, 41226 cells/1492632 DoFs/659616 particles | Initialization +2 physical steps pass, 696.36 s; dt 1772.48070853/2166.15453236 selected by unchanged .2 log-state bound; publications once, direct stress defect ≤5.05e-9 Pa |

All 11 launches including three rejected observer/setup runs cost 1382.60 s
(23.04 min), excluding compilation. Original residual criteria pass; ordinary
startup max normalized residual 1.81e-9, state error 2.23e-16, peak virtual memory
8.42 GiB (resident peak not measured). Raw API requires explicit raw+zero length;
observer handles its optional vector. Profile zero check now inspects velocity
history separately from stored phase. Failed records/binary/source are retained.
Only benchmark diagnostics/inputs/analysis/report and status/scope docs change;
production source/defaults and read-only reference worktree remain unchanged.

No section-8 work remains. Fine full mesh/first-event/cluster jobs are unlaunched.
The candidate is provisional: coupled spatial/profile/normal-traction accuracy,
FE restart/conversion, automatic cutback and a supported history-transport model
remain requirements. Current memory is fixed Eulerian FE with stationary derived
particle samples. Next proposed task: select a coupled spatial accuracy gate;
review restart/cutback/transport scope before long-run use. Stop here.

---
# BP3 sections 7C–7D — completed bounded sensitivity / MPI recovery

Selected [SECTION7CD.md](SECTION7CD.md) only; all 7B evidence and incoming
production work are preserved. Same core/maintained coupled plugin and physics,
filter 20/full bottom/frozen fields/stationary samples. No later stage is run.

| Actual check | Result |
|---|---|
| Temporal dt 100000, twenty steps to 2000000 | Pass; Initial time step stays 200000, step-zero profiles byte-identical to 7B; final max V sensitivity .01989%Vp |
| Coupled q3→q6, original mesh/surface | Pass; phase/I_h identical; final max V sensitivity .01060%Vp; q6 selected, no q8 needed for this question |
| Actual surface 20→10 then band/endpoint bulk refinement | Both ten-step paths pass; 59→117 nodes, 4136→11528 cells; max V sensitivities .52394%/1.18694%Vp |
| Profile/normalization | Bulk changes sampled phi by ≤.021970; min I_h 12180.406→12818.608 m (5.24%); no physical profile retuning or integration-law change |
| Ordinary BP3 15–18 km material/state check | Pass; native a error 0, friction error ≤1.12e-16, theta error 0; no coupled VW/transition evolution |
| Refined q6/surface 10 one-/two-rank, three positive steps | Pass; actual Newton updates at steps 2/3; final max V difference 7.90e-23 m/s, filtered normal 3.20e-7 Pa |
| One-owner candidate rejection /identical-dt retry | Pass on both ranks, no deadlock; accepted histories/property bytes unchanged, zero pre-retry publication; all exported clean/retry fields identical |

All ten simulation launches including failed diagnostic startups used 1567.88 s
(26.13 min) of 3600 s. Native recurrence/state/publication checks and original
solver criteria pass; temporal residual max 9.69e-9 is below 1e-8. Failed include
parsing, observer manager lifetime and initial helper compilation remain recorded.
The lifetime fix retains the existing initialization-manager shared pointer.
No production C++/header or core/7B binary changed. ASPECT astyle 2.04 is applied
only to the extended observer. [One report](section7cd_report.md) links all
common-time profiles, pressure/tensor/roughness metrics, commands and provenance.
Recovery/preservation evidence: `build-bp3-native/section7cd/`.

No selected work remains. Correctness/MPI gates support a subsequent short
unfiltered diagnostic and restricted fresh-input preparation; neither is launched.
Spatial accuracy remains open: phase/I_h and endpoint traction changes require
further bulk/profile assessment before selecting a full-model resolution.
FE restart/conversion/transport/birth and general cutback recovery remain unqualified.
Next action: review/select the subsequent scope with these accuracy limits explicit.

---
# BP3 section 7B — completed serial paired comparison

Selected scope: section 7B only. First-pass manufactured checks are reused.
See [SECTION7B.md](SECTION7B.md) for the declared physical pair, fixed schedule
and 3600 s aggregate simulation budget. Same current executable and rebuilt
maintained local plugin; physical Vp=1e-9, uniform strengthening, full bottom,
filter20, stationary sampling/no population changes and exact common Gauss3.
Uniform and common .02 smooth-state controls pass ten positive steps at
dt=200000 s to T=2000000 s. One-step gates precede either trajectory.
Both one-step gates pass: FE stress error ≤1.14e-9 Pa at 16.86 kPa scale,
positive Z/zero cohesive traction, state/slip publication once; particle error
≤9.01e-10 Pa at 17.99 kPa scale. Mesh, phase, rule, physical particle identity
and initial I_h/state/background match exactly. Four failed FE setup/observer
launches remain preserved (local plugin linkage, output ordering and net-strain
roundoff scale). The corrected check uses the original 128-epsilon multiplier
with coefficient/basis operands; no physical/solver tolerance changes.
All four trajectories pass without cutback or changed schedules:

| Actual result | FE creep / particle creep | FE perturbed / particle perturbed |
|---|---|---|
| Serial wall seconds | 106.64 /62.81 | 106.44 /62.24 |
| Max normalized residual | 7.39e-10 /6.84e-10 | 7.54e-10 /1.27e-9 |
| Final max V difference /Vp | .18966% | .18963% |
| Final max raw /filtered normal difference | 2609.61 /351.44 Pa | 2657.62 /366.48 Pa |

Constant positive I_h, frozen profile/H, 66176 fixed particles, exact mature
cohesive zero, nonzero FE Z and once-only accepted state/slip/history pass.
Max direct stress defects≤9.57e-9/1.01e-9 Pa at FE/particle sample scales
156/193 kPa. All simulation launches including failures cost 423.76 s of 3600 s.
Normal perturbation RMS differences remain 10–13%; pressure/tensor compensation
and the full-gradient versus consistent_deviator distinction are explicit.
Lower-end raw normal roughness is substantial in both backends; three sampled
grading cuts have much smaller stress jumps. This is one-resolution evidence.
Compact output includes all common-time profiles and three normal cuts.
See [section7b_report.md](section7b_report.md) for settings, equations, changed
files, preserved failures, actual results and artifact/analysis links.
Incoming work snapshots/hashes are under `build-bp3-native/section7b/`.
No selected 7B work remains. Next proposed issue: targeted common endpoint
spatial/quadrature refinement to assess raw normal roughness and pressure/tensor
cancellation before interpreting an MPI test. Await selection.
No 7C/7D, unfiltered, full-model or checkpoint continuation work is selected.

---
# BP3 accepted FE history — recovery complete

Selected scope: [plan.md](plan.md) sections 1–6 with focused verification.
Baseline `c33edaaeb145e5d3dce155bb815552ef80efc5f9`, `fault-propagation`.
Read-only reference `../aspect-pf-rsf-refactor`, `pf-rsf-refactor`,
`8f126905c559df106f08af4079f678401fec4d06`.

Recovered 15 modified tracked files, no staged changes and 2346 untracked files;
reference tracked files clean, 63 untracked entries. Existing FE recurrence,
geometry/normalization dispatch, atomic publication, stationary diagnostic and
inactive-P2 exclusion were already implemented. Preserved them without reset,
merge or reapplication. Recovery corrected particle FE evaluation to use DoF
iterators, made the relaxation control respect the existing RSF minimum rate,
clarified quadrature help, and completed verification/documentation. No new
physical assumption, tolerance, law, default or generic projection change.

| Actual result | Evidence |
|---|---|
| Core and matching plugins | Release build passes; separate `build-bp3-native`, GCC12.4/deal.II9.6.2/Trilinos14.2/p4est2.8.7/Voro++0.4.6 |
| Graded q6 / q8 / uniform q6 | All 36 manufactured increments pass; 125.76/248.39/129.67 s; max stress defect .011397 of original roundoff allowance, Z relative error 1.78e-16 |
| Oblique I_h / endpoint completion | Discrete-profile error ≤5.97e-16; both contacts and nonzero exterior contribution; graded min I_h=12188.324109 m |
| Tensor / lifecycle | Divergence correction verified; zero mature cohesive traction with nonzero Z; immutable scratch, duplicate publication and particle movement guards pass |
| Compatibility / FE resume guard | 21,112 assertions/55 cases pass; intended fresh-run rejection confirmed |
| Already completed legacy controls | Two-rank Stage-J history/failure and boundary completion pass (28.17/4.07 s); not reapplied |

Earlier failed builds remain recorded, along with eight startup/dependency attempts, two
particle-iterator diagnostic attempts and the zero-slip RSF relaxation attempt.
Outputs, inputs, commands, hashes, incoming patch and preservation inventory are
under `build-bp3-native/recovery-20261013/` and uniquely labelled existing
`refactoring_boundary/evidence/bp3-native-recovery-*` records. Numerical outputs
remain outside tracked source. See [report.md](report.md) for exact changed files,
derivation, settings, measured errors and commands.

No selected implementation/check work remains. The coarse uniform profile has
substantially different I_h; these tests establish discrete recurrence/integration,
not continuum spatial accuracy. Coupled BP3, native FE MPI/cutback qualification,
restart/conversion, transport/birth, production preparation and long first-event
runs remain unqualified/deferred. Next proposed task: section 7B's bounded paired
stationary-history coupled dipping-fault test, full bottom velocity and 20 m filter.
Stop here; no later stage, server execution, commit or push is selected.
