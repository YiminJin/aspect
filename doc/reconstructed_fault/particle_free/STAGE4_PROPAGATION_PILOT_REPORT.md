# Estimate-led propagation pilot, 256x128

**Stopped after the user's cost review, at accepted70 /5.60 ms.** Trial71
(5.68 ms) was unaccepted. The constrained solve became prohibitively costly;
there is no running continuation to6 or10 ms. The current conclusion and
proposed bounded timestep check are in the final section below. Earlier pause,
restart and cost estimates remain chronological evidence.

Task began15 November 2026. The user selects one weakening-law pilot with
steady loading and free phase. The preceding overnight preparation remains a
closed control; its proposed 256x256 pair is superseded by this task.

The recovered first-free-turn reference reached x=6.243095748 m at .080 s,
with a connected phi=.05 contour, .021466 J/m added fracture in initially intact
axial material and 144 additional formed sections. That calculation used
Dc=.0005 m, age=.5 s and 128x128 cells. The current law retains
Dc=3.22111261263e-7 m, age=.001610556306315 s and 256x128 cells. The successful
reference therefore establishes a useful stress/loading and geometry reference,
not a slip-time prediction for this pilot. Its spatial/time limits and the
original failed .005 s interval remain documented in
`RECOVER_FREE_TURN_STEP16_REPORT.md`.

The selected pilot uses total initial shear .000605 and normal extension
.00015313568269, with a fresh irreversible static equilibration of the retained
.000365 preparation. The steady shear and extension rates are
.0165753424658/s and .00419549815589/s. The current weakening law, near-tip
sliding preparation, midpoint update and native acceptance gates are retained.
The working target was10 ms in checkpointed segments. The cost review below
closes that continuation at5.60 ms. This remains one specimen.

The following preparation record includes a rejected spatial loading mode.
Its fixed-phase native equilibrium screens measure unconstrained damage
directions, **not** physical time evolution or accepted phase stability.

The rejected spatial mode derives from the incompressible displacement streamfunction
psi=a exp(-((x-5.75)/1.25)^2) (y-2)^2/2. Thus
u_x=a f(x)(y-2), u_y=-a f'(x)(y-2)^2/2. It concentrates shear toward the formed
right taper while decreasing away from it; compatible normal boundary motion
avoids imposing a compressible patch. Native equilibrium, rather than the
boundary shape itself, determines the actual tip and detached-damage margins.
The affine background keeps gamma=.000365 and extension/shear=.41954981558864.
The steady physical velocity follows the same displacement pattern on a
.0365 s loading scale. Neither rupture locations nor later initiation times
are prescribed.

After rejecting that mode, three affine fixed-phase responses (the retained
zero-mode response and two new amplitudes) determined the quadratic driving
along a fixed admissible connected-profile increment and the all-orientation
intact envelope. They selected the one candidate described below. A favorable
single coefficient does not establish connected extension. Actual connected
contour, initially intact damage, new sections, signed deformation and physical
work are retained.

Root/branch remain aspect-pf-rsf-particle-free / pf-rsf-particle-free,
HEAD b32a2d4f6e21c219f9af786d1b6e3725ea78a5a4 with incoming manual work.
Protected main executable SHA256:
089824ae44c9c53b57921fe0c5019c59f4a37cf48e1f753917e232f4e4c22378.
A separate benchmark checker supplies the spatial preparation; production
source, physical midpoint solver and acceptance criteria are preserved.
Entry hashes and exact build/run commands are under
build-particle-free/stage4-propagation-pilot/. At most four aggregate workers
are used, with one thread per numerical rank. Only 2.2 GB was free at entry;
new closed diagnostics are losslessly compressed, preserving checkpoints.
No historical availability loss is reversed or concealed.

## First screen: localized mode rejected

At added amplitude .0004 the broad connected-profile direction remains
unfavorable (-.839897 Pa mean net driving), while the strongest intact boundary
row is already favorable by 4.89664 Pa near (6.140625,3.984375) m. Intact material
just ahead also has favorable rows (+.240648 Pa). The localized boundary mode
therefore fails the required spatial selection and is rejected before release.
Its checker, input, log and native fields are retained. No second amplitude of
this unsuitable shape is run.

The reference's accumulated boundary displacement at .075 s has shear .00075
and extension .000146842, a ratio .19579. This differs from the recent .41955
ratio. The next inexpensive calculation holds extension .000153138 fixed while
adding spatially uniform shear: the ratio decreases toward the successful
reference. This avoids the boundary gradients introduced by the rejected local
mode. All spatial row margins, including remote boundary rows, remain part of
selection; a uniform scalar multiplier is not inferred from a tip-force ratio.

## Reference-ratio estimate and one candidate

Two additional equilibrium solves at added shear .00025 and .0005 took about
80 seconds each. The retained zero-amplitude solution completes a quadratic
native stress-driving estimate at fixed phase/resistance. The broad forward
profile direction changes sign at added shear .000450413; the first intact
all-orientation row reaches zero at .000254199. Thus the broad-profile threshold
is unsuitable: intact rows would already be favorable by 1.10941 Pa. This
reproduces the earlier concern using a different shear/normal ratio.

One candidate at added shear .00024 (total gamma=.000605, fixed extension
.000153136) is selected for native irreversible **static equilibration**, below
the first intact threshold. Existing cap modes are already favorable there;
the static solve is needed to avoid mislabelling their initial adjustment as
physical propagation. The old broad increment remains an independent diagnostic.
After equilibration, a direction touching and advancing the measured connected
contour must be evaluated, alongside all intact/remote margins. Candidate
selection is conditional on that actual preparation, not a favorable isolated
coefficient in the unequilibrated screen. No further load sweep is selected.

## Settled candidate selected for release

The six native phase corrections converge at KKT=4.85121e-8 and mechanical
residual=1.04469e-16, with unchanged gates. Zero physical time and zero slip
increments are retained. Preparation costs 439.25 s including wrapper/compression.
Its added fracture inventory is .000153607118 J/m, separate from later physical
work and fracture. The right contour moves .0000618834 m, the left .00235655 m.
The fresh right/left x-extents are 6.00211626333 / 1.99558486898 m.

Seven previously intact cells acquire coefficients above1e-8, maximum .00212049,
mostly on the left diffuse flank. These additions must not be hidden by redefining
"intact" after preparation. Sampled phase support at levels1e-10,1e-8,1e-6,1e-4,
.001,.005,.01,.05 remains one connected component, with no new disconnected
sample area. Subsequent increments are measured relative to this settled state.

All remaining intact row envelopes are unfavorable, strongest -.259826 Pa;
right-ahead intact rows are at most -.282663 Pa, remote envelopes -.848537 Pa.
The broad shifted-profile direction still costs energy (mean net -1.35247 Pa).
In the element containing the actual right contour, two native nonnegative
basis directions (indices3,6) have net driving about +1.99e-10 Pa. Read-only
1e-4 coefficient probes move the connected contour .000122949 and .0000274261 m,
respectively. Thus they really advance this level set; they are still local
infinitesimal diagnostics, not proof of finite propagation. All nine directions
and their unequal costs are retained in `candidate.json`.

The release uses the settled phase as its lower obstacle, the same prescribed
age/sliding preparation, mechanically equilibrated physical tau/A, current
weakening law and the retained1% initial smooth age perturbation. Total initial
shear is .000605 and extension .00015313568269. The steady shear velocity
gradient is .0165753424658/s and normal extension .00419549815589/s; their ratio
is .25312. No later rupture initiation is forced. The first window comprises
eight80-microsecond midpoint intervals; checkpoint continuation permits a
common trajectory. The initial2.56 ms observation estimate was subsequently
extended to10 ms from measured native driving; see the resumption section.
Only the initial single weakening-law pilot is selected. A constitutive-law
attribution or episodic claim would require the additional controls specified
by the user.

## First physical segment and operational recovery

Seven intervals are accepted through .56 ms, although the planned eight-step
process exits124. Native wall-time termination occurred after accepted7; the
wrapper's remaining time was insufficient for checkpoint serialization. Slot01
contains mesh/metadata but no resume.z or last_good_checkpoint marker and is
**not a usable restart**. Accepted fields, scalar histories and failed evidence
are preserved; no incomplete snapshot will be used. The 300-second reserve was
too short for this more expensive specimen. Subsequent wrappers reserve1200 s
and use periodic checkpoints.

At .56 ms the right connected contour advances4.63623 micrometres
(.00014836 hx); left advance is zero. Peak rate reaches5.13797 mm/s from
.202527 mm/s initially. Added fracture is8.62952e-7 J/m; signed and accumulated
absolute physical-work errors are both5.27390e-13 J/m. Exactly intact material
relative to the prepared state remains unchanged and no new formed section
appears. This is an accelerating slipping state with cap evolution, not yet a
resolved propagation event. The history is too short to judge arrival at a tip.

The new reducer explicitly includes both DG face traces in its connectivity
graph and excludes merely adjacent disconnected cells from front candidates.
Eight versus16 nodes per cell give the same axial extrema throughout this
segment. The prepared-state probes also pass that face audit. The prior
interior-sample result is preserved separately for provenance.

A replay from the same fresh prepared state is necessary because no complete
physical restart was written. It remains the same candidate, law, loading,
step size and solver; accepted-prefix comparisons will test reproducibility.
Only full-domain diagnostic dumps change to a sparse cadence. Phase, section
rates/state, formation/angles, energy, residuals and work remain available at
every accepted step. The two diagnostic-only branches and original checker are
retained; production code and physical equations are unchanged.


The fresh replay targets .010 s (125 intervals at80 microseconds) within a
self-imposed six-hour total task envelope, including completed preparation and
the failed first process. This is a computational cap, not a physical stopping
criterion. Native wall-time termination has1200 s reserved for finalization;
the outer wrapper adds600 s for closed-file compression. Periodic restart
slots are requested every8 intervals. Full-domain row diagnostics are emitted
every32 intervals; all accepted phase, section and scalar histories remain
per-step. The tested checker and driver are frozen under `tested-replay/`.
The actual endpoint and any unresolved numerical failure will govern the
interpretation, rather than the requested endpoint.


Replay qualification: all four native phase exports and all section/plane
exports at steps0..7, together with accepted energy and pilot-summary tables,
agree exactly with the retained attempt (56 files/tables; decompressed SHA256
for field files). The comparison is in `replay-prefix-byte-comparison.json`.
Thus the two sparse-output branches leave this complete physical prefix
unchanged. This is a reproducibility check, not timestep convergence.


The first new interval (.56→.64 ms) reaches a positive-metric search plateau:
phase KKT≈1.25238e-6, mechanics≈1.13904e-13. Through iteration20 it accepts
steps as small as1.86e-9, including a shift1000 fallback, without materially
reducing the phase residual. It is manually stopped as an **unaccepted trial**
(exit143), before the native80-iteration limit. This is a bounded stagnation
stop, not a recorded native iteration-limit or line-search exception. No new
physical state was published. The completed seven-step prefix remains exact.

One recovery selects the executable's existing adaptive exact-first ordering
by omitting `ASPECT_TIP_POSITIVE_METRIC_FIRST`; trace output is enabled. No
source, equations, tolerances, timestep or loading change. It starts again from
the same prepared t=0 because no usable physical restart precedes step8.
Every accepted interval now requests a checkpoint. This is one numerical
recovery of the same specimen, not another physical case or a solver redesign.


The existing adaptive ordering clears step8 at phase KKT1.68721e-12 and
mechanics1.82466e-17, without changing gates or physics. It makes another exact
correction at residual1.17764e-6, where the positive-first ordering had retained
its shifted family. A complete0.56 ms recovery parent is preserved separately
under `accepted-parent7/`; accepted0.64 ms is also checkpointed. The eighth-stage
peak rate is8.97053 mm/s. Native initially-intact/remote region increments are
still zero. This clears the numerical obstruction; it does not yet establish
propagation or time accuracy.


## User shutdown pause

The ninth interval is accepted and its native checkpoint completes. The watcher
stops the four verified ranks after the following timestep header; no tenth
interval is accepted. Exit143 is intentional. The checkpoint is copied with
verified hashes to `paused-parent9/`, while the original rolling slots and
separate `accepted-parent7/` remain. All23 protected identities and incoming
source hashes pass the preservation check. No simulation remains running.
Further physical testing is paused until the user asks to resume.


Pause reduction: right connected contour advance is 7.10977801e-06 m
(0.000227512896 cell); left advance is zero. Initially intact
and remote phase increments and new formed-section counts are zero. Added
fracture is 1.54412099e-06 J/m; signed and accumulated absolute
physical-work errors are both 4.0314266e-12 J/m. Eight versus16
contour samples per cell give identical axial extrema. The low-phase support
checks are retained in the JSON; these sampling checks are not mesh convergence.
The maximum phase-coefficient difference from the positive-first accepted
prefix is 5.79572203e-10; maximum peak-rate difference is 3.85448121e-10 m/s.
The largest stage slip/Dc is 3.00802, so a
small work error alone does not establish temporal state accuracy. Strong
left-taper signed cancellation remains explicit in the history.

[Step9 pause history and summary](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/pilot-adaptive-summary.json)
and [rate/connected-front figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/pilot-adaptive-space-time.png)
are available. No simulation was run during that pause; the later explicit resumption is recorded below.


## Resumption, 16 November 2026 UTC

The user requested resumption. The saved step9 checkpoint, checker, all23
protected identities and incoming source hashes pass verification; no existing
simulation is active at entry. `pilot-resume9` continues the same state with
four ranks, unchanged dt/physics/search, per-interval checkpoints and the
restore audit enabled. The pause is historical, not an ongoing restriction.


The resumed parent passes the read-only audit: all section memories and plane
fields agree exactly; native history phase equals simulator phase; maximum
quadrature phase difference is1.11022e-15. The independent6x6-Gauss integration
of full3D deviatoric stress energy differs by3.73035e-14 J/m from accepted9.
Signed fault-rate section measures agree within6.77626e-20 m²/s. No fresh
preparation or age/slip reset occurs. Dense pointwise predecessor tau/A exports
were not retained at every interval, so full pointwise tensor comparison is
not claimed; checkpoint payload hashes and these independent invariants are
recorded in `pilot-resume9-restore-audit.json`.

The closed restore-audit CSVs are losslessly compressed with decompressed
SHA256 verification, saving137,359,671 bytes. Their separate online compression
manifest complements the eventual wrapper manifest; no checkpoint is altered.


At accepted13 (1.04 ms), peak stage rate is13.2210 mm/s. The two forward
probe means remain unfavorable: -.432835 / -.617678 Pa. A least-squares
extrapolation of the last six native stage observations puts their zeros near
7.46 /9.22 ms (`resume-window-estimate.json`). This is an observation-window
estimate, not a prediction of onset: phase/slip redistribution can change it.
It supports retaining the selected10 ms target rather than interpreting an
early capped history as a negative propagation result. Measured steps10..13
cost231–263 s each. The earlier six-hour whole-task estimate was optimistic and
was not a user cap; reaching10 ms may require another7–8 hours. No new specimen,
loading case, law comparison or production job is selected by this cost update.
Continue the same physical history in bounded, checkpointed segments, stopping
sooner for a clear scientific result or a numerical obstruction.

At accepted31 (2.48 ms), the resumed segment has taken2.15 hours for22
intervals, averaging352 s including checkpoint/output overhead. The updated
estimate for the remaining94 intervals and analysis is9–11 hours. This
supersedes the earlier7–8-hour estimate, without changing physics or target.

## First resumed segment: accepted38 at3.04 ms

`pilot-resume9` exits0 after native wall-time termination, with complete
slot03 and hashed `parent38-identity.json`. All23 protected identities and
incoming source files still match; only the four expected current-status
documents differ. The composed `through38` analysis preserves the intentional
exit143 of the original prefix and the exit0 of the resumed segment.

The right connected contour advances49.2868 micrometres (0.00157718 hx),
the left4.10680 micrometres. Maximum phase increment is.00137364; initially
intact and remote increments remain exactly zero. No section is newly formed,
and existing formed-plane angles are unchanged. Added fracture is1.81041e-5 J/m.
Peak stage rate is16.8239 mm/s, but the interpolated right-tip rate is only
.892374 mm/s; the right formed taper reaches10.6106 mm/s. The left taper has
nearly complete signed transverse cancellation, retained explicitly.

The rate-versus-position history shows broad acceleration inside the prepared
slipping region followed by sustained sliding; there is no resolved propagation
or arrest/episodicity claim. Both forward probe means remain negative. The
next segment continues this same state and10ms target without changing dt,
loading, law, tolerances or search ordering.

[Combined rate/front figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/through38-space-time.png),
[phase/work figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/through38-response.png), and
[quantitative summary](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/through38-summary.json).

## Second requested shutdown: accepted51 at4.08 ms

The watcher waits for completed slot01, then stops the four verified ranks at
the timestep52 header. No state52 is accepted. Exit143 is intentional. All
processes exit; the separate `paused-parent51` copy and original payload agree
byte-for-byte. Runtime/incoming-source checks pass, and both older saved parents
remain intact. Resume only after the user requests it.

The step38 restore audit has exact section/phase/plane agreement; full3D stress
energy differs by6.83897e-14 J/m, and signed measure errors are below6.8e-20 m²/s.
The complete trajectory is composed as `through51`, preserving each segment's
process status. At51 the right contour advance is.000124134865 m (.00397232 hx),
left6.16209e-6 m; no section is newly formed. Added fracture is3.42142667e-5 J/m,
and signed/absolute cumulative physical-work errors are5.61954320e-11 J/m.

One initially intact cell begins evolving at49 (3.92 ms), centered at
(6.078125,2.140625)m on the upper flank. At51 its maximum coefficient increment
is.000121118 and integrated increment1.31421508e-8 m². Sampled low-phase support
remains connected; remote material is unchanged. This is fresh flank damage,
not yet resolved axial propagation. The two axial probe means remain negative
(-.195258/-.380056 Pa), demonstrating why these probes alone cannot detect every
off-axis direction. Peak rate is18.3407 mm/s, compared with.986826 mm/s at the
right contour. The broad slipping region remains active. There is no episodic
or friction-law attribution claim; state/timestep accuracy is still unqualified.

The [current shutdown handoff](STAGE4_PROPAGATION_PILOT_PAUSE.md) links the final
figures and histories and specifies the checkpoint, restart command, disk-space
requirement and remaining question. No further simulation is authorized until
resumption. Older cleanup losses, failed attempts and both pause records remain.

## Detached continuation from accepted51

The user requests restarting the long run and then ending active monitoring.
Root/branch/revision and incoming manual source are rechecked. All23 protected
identities and every step51 saved checkpoint payload pass SHA256 verification;
no existing ASPECT process is found. `pilot-resume51` is launched with a detached
session, four MPI ranks, one thread/rank, unchanged dt80 microseconds and
adaptive exact-first ordering. The target remains step125 /10 ms. No extra case,
production-source edit, rebuild, solver change or automatic follow-on experiment.

The launch command/PID and detached console log are in
`build-particle-free/stage4-propagation-pilot/detached-resume51.{json,log}`.
Native progress is `build-particle-free/bulk-tip/runs/stage4-propagation-pilot-resume51/run.log`.
Completion is recorded by its `command.json`, the outer logged wrapper, and
`pilot-resume51-finished.json`; launcher exit0 alone is not simulation success.

Disk preparation losslessly archives12 closed, inactive checkpoint slots, saving
603580093 bytes. Each archive is read back and every member SHA256 checked
before its expanded copy is removed. The manifest and exact restoration notes
are under `build-particle-free/stage4-propagation-pilot/checkpoint-archives/` and
beside the archived slots. This includes previously incomplete/failed snapshots;
archiving does not make them valid restarts. The selected step51 parent, its
original slot01, separate step7/9 parents and original selected step38 slot03
remain expanded and byte-identical. No checkpoint data or historical evidence
is discarded; older cleanup availability losses remain unchanged.

The step51 restore audit passes, including exact section/phase/plane histories
and independent full3D stress-energy and signed-measure checks. Closed restore
exports are losslessly compressed with verified hashes, saving137,215,314 bytes.
The detached supervisor and exactly four native MPI ranks are verified running
at step52 before ending active monitoring. Results remain pending; no final
acceptance or propagation conclusion is inferred from successful startup.

## Cost review and bounded stop at5.60 ms

The user reports more than five hours for the latest continuation and asks
whether6 ms can supply a useful conclusion. At review, the displayed5.68 ms is
an **unaccepted trial71**; the last complete accepted state is70 at5.60 ms.
Five intervals, not four, remain from that accepted state to6 ms. The latest
segment ultimately records20,197.8 s (5.61 h) in the outer monotonic wall timer.
Earlier constant-cost estimates no longer describe this solver regime.

### Decision

Close this long pilot at accepted70. Reaching exactly6 ms is not a scientifically
required milestone and does not remove the observed timestep/solver uncertainty.
The accepted history already establishes substantial slip acceleration with
small connected-contour motion and fresh damage on the upper flank. It does
**not** establish resolved axial propagation, repeated rupture/arrest, a
weakening-specific cause, or absence of later propagation. Do not tune the
specimen merely because this numerically bounded pilot has not reached a cell
of extension. Retain the candidate for one local temporal/solver comparison
before deciding whether a longer physical history is worthwhile.

### Cost and numerical obstruction

| Step | Endpoint (ms) | Logged linear-solve minutes | Completed linear solves | Status |
| --- | ---: | ---: | ---: | --- |
| 52–66 | 4.16–5.28 | median5.96 per interval | 5–7 | accepted |
| 67 | 5.36 | 31.77 | 44 | accepted, 8 shifted searches |
| 68 | 5.44 | 7.56 | 8 | accepted |
| 69 | 5.52 | 76.61 | 79 | accepted, 7 shifted searches |
| 70 | 5.60 | 9.19 | 9 | accepted |
| 71 | 5.68 | 23.15 before stop | 35 | unaccepted, manual cost stop |

These times exclude assembly, output and checkpoint overhead. Step71's latest
nonlinear phase KKT is.0159771. Within its next constrained linear solve,
feasibility is lost repeatedly; the pass18 linearized phase residual reaches
3.96685e11 and mechanics1.04376e7. These are **trial linearized residuals**, not
accepted stresses or evidence of physical instability. The run is deliberately
stopped before a native iteration-limit/fallback conclusion. Exit143 is a manual
bounded cost stop, not a native solver-failure return. The unchanged acceptance
gates still hold for the saved step70 (phase KKT3.45479e-10).

The right contour increments on steps67–70 are .144763, .034442, .142425 and
.035364 mm. This approximately fourfold odd/even alternation accompanies the
expensive odd steps. It motivates a timestep check; correlation alone does not
prove the cause is midpoint integration, search, or physical dynamics. Do not
interpret this staircase as episodic propagation.

### Accepted physical evidence

At5.60 ms, the right absolute-phi=.05 contour advances1.81636 mm, only.05812 hx;
left advance is.0103191 mm. No new section is formed, and existing formed-plane
angles are unchanged. The global peak stage rate is19.3433 mm/s, whereas the
interpolated right-tip rate is1.15373 mm/s. Integrated cumulative slip is
1.74553267e-4 m². Left-taper signed cancellation remains99.33%; its small net
slip must not be read as absence of local opposite-sign deformation.

Maximum phase increment is.00347673. Initially intact increments are confined
to two upper-flank cells centered at(6.078125,2.140625) and(6.140625,2.109375)m;
the maximum is.00132787 and their integrated increment2.67430e-7 m². Remote
material remains unchanged. Sampled support remains one component at levels
1e-6,1e-4,.001,.01,.05, with no detached sampled area. The 8/16-sample contour
extraction check agrees; it is not spatial convergence. This shows fresh flank
damage connected to the diffuse support, not resolved axial extension.

Added fracture is1.00829537e-4 J/m. Signed and accumulated absolute physical-work
errors are both1.49325930e-10 J/m (about1.48e-6 of added fracture). Bounds and
irreversibility hold to roundoff. These work checks do not resolve temporal
accuracy: the largest stage slip/Dc reaches4.804. The axial forward probe means
are still unfavorable (-.079439/-.253956 Pa); off-axis flank damage explains why
these two probes alone cannot characterize all admissible damage directions.

### Preservation and one proposed next test

All four ranks and launchers exit. `cost-review-stop.json` records the exact
rank identities, signal, checkpoint hashes and unaccepted trial. The complete
native step70 is copied and verified under
`build-particle-free/stage4-propagation-pilot/cost-stop-parent70/` (slot02).
`cost-review-preservation.json` verifies all23 protected identities, incoming
manual source except the four expected status documents, and older saved
step7/9/51 parents. Native source, physics, tolerances and solver remain unchanged.
Previously archived checkpoint slots remain losslessly available; no further
data loss is introduced and older cleanup losses remain explicit.

Retain the complete native step68 in
`build-particle-free/bulk-tip/runs/stage4-propagation-pilot-resume51/output/restart/03/`;
`cost-review-parent68.json` records its hashes. The original last-good marker
selects step70, so selecting68 requires a separate copied parent with marker3,
not blindly resuming that output directory. Step69 slot01 is also retained.

**Historical proposal, subsequently attempted below:** from accepted68 at5.44 ms, run four40-microsecond
midpoint intervals to5.60 ms, compared against retained steps69/70 from the same
parent. Keep all physical inputs, initial state, bounds and gates unchanged;
cap numerical work at one hour. Compare V(x), integrated slip, phase distribution,
connected-contour increments, fresh flank damage, physical work and constrained
solve cost. If it fails or reproduces costly/alternating behavior, preserve that
evidence and stop; diagnose the responsible phase rows/active-set transitions
before another long trajectory. No mesh study, new preload or full campaign is
selected by this proposal.

[Cost and contour-increment figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/cost-review.png),
[rate/connected-front figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/through70-space-time.png),
[phase/work figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/through70-response.png),
[summary](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/through70-summary.json), and
[cost table](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/cost-review.csv).
`cost_review.py`, `analyze.py` and `plot_response.py` reproduce the reductions;
`analysis_views/through70/provenance.json` retains the chronological segment
identities and process exits. No unaccepted geometry is included in the figures.

## Selected bounded timestep check

The user selected the proposed four40-microsecond intervals from accepted68.
The [timestep-check report](STAGE4_PROPAGATION_TIMESTEP_CHECK.md) records the
common-parent verification, one-hour limit, matched-time observation rules and
actual outcome. The attempt exhausted its 3600-s numerical cap without accepting
the first new interval. Neither matched endpoint is available; no timestep
accuracy conclusion follows. Final logged phase KKT was 1.04771e-6 against
the unchanged 1e-7 gate. All processes exited, the parent is preserved, and the
long continuation remains stopped. A targeted constrained-residual/active-set
audit is proposed; no further physical case or solver redesign was launched.

The subsequent [guarded replay](STAGE4_PROPAGATION_LINEAR_RESIDUAL.md) implements
the explicit residual test, checks the full coupled tangent, and records the
accepted-state comparisons using the original physical tolerances.
