# First RSF episode: estimator repaired, server continuation prepared

2026-10-30; baseline f2f730529 plus preserved local work. **The repair is verified;
the physical episode is not yet complete.** Per the latest instruction, local
continuation was checkpointed and server materials prepared once measured cost
became excessive. No server job or finer spatial run was launched.

[Event-window PNG](../../../benchmarks/reconstructed_fault/particle_free/evidence/rsf_episode/event/event-window.png) ·
[Event GIF](../../../benchmarks/reconstructed_fault/particle_free/evidence/rsf_episode/event/event-window.gif) ·
[CSV/JSON evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/rsf_episode/INDEX.md) ·
[Estimator equations](RSF_EPISODE_ESTIMATOR.md) ·
[Server instructions](../../../benchmarks/reconstructed_fault/particle_free/rsf_episode_server/README.md) ·
[Server archive](../../../benchmarks/reconstructed_fault/particle_free/rsf_episode_server.tar.gz).

## Numerical repair

The old independent log-speed rejection was not a temporal-error test at admission:
a finite rate change can accompany a vanishing continuous-history error. Its failed
step661 is now accepted with **exactly the same** point damage, jump and velocity.
No history is clipped, reset, retied or reinterpreted. The minimum dt remains1e−10 s.
The original raw estimator stays selectable/default; the new experiment explicitly
selects `Use residual resolved estimator = true`.

The current complete residual, existing1e−8 Pa weak resolution and counted assembled
sum/product arithmetic give a force-resolution envelope. The actual constrained,
mass-normalized nonsymmetric tangent supplies its symmetric-part lower bound alpha,
including pressure coupling. Coercivity maps this to point displacement resolution
Uq and rate intervals Uq/dt. This is a **linearized assembled-system estimate**, not
a nonlinear stability, assembly-error or spatial-accuracy certificate.

Acceptance now controls BE/trapezoidal defects in jump, aging and the existing
irreversible damage map. Numerical uncertainty must itself fit the continuous-
history budget; the old slip/state/damage increment limits are strengthened by
uncertainty envelopes. Raw/resolved log-rate changes are reported, not independent
rejection gates in this mode. There is no damage cutoff. All native mechanical,
KKT/contact, state, damage, source and energy checks remain unchanged.

At the original failing candidate, raw/resolved log-speed changes are .33053/.32574,
while continuous-history error is1.08e−12. It is largely a resolved rate change,
not a significant continuous-history integration error. The selected local error
budget5e−4 is a numerical setting, **not a claim of1% global accuracy**. The retained
age/spatial/fracture-work failures and earlier shared-history errors remain.

## Verification

- Independent tests pass1343 assertions in4 cases: nonsymmetric coercive force
  perturbations, overlapping/disjoint rate intervals, finite admission and signed
  reversal, and aging versus an independent exponential solution.
- The actual failed checkpoint advances661–664 with all guards. Original candidate
  j/a/V match bitwise; no failed trial was published previously, and each accepted
  step publishes once.
- Forced rejection/retry and same-rank restart reproduce every bulk coefficient,
  all six surface histories and geometry exactly. Retained default behavior is
  bitwise unchanged in five states.
- A small2→4-rank restart restores histories exactly; its next-step coefficient
  difference is2.27e−13 and surface-history difference8.27e−24. The large continuation
  then successfully runs on4ranks. A zero-advance terminal restore reproduces
  surface/geometry/bulk-centre exports bitwise and independently validates every
  restored CellId tensor/native-memory record.

## Continuation and coupling observed so far

The four-rank native segment advances664→748 and stops cleanly under the native
collective wall criterion, checkpointing **t=374.58003334330067 s**. It recovers
from sub-nanosecond intervals to about3.83e−5 s, with continuous defect about4e−4
against the5e−4 budget. Largest observed rate .0450089 m/s occurs at374.579229367 s;
final rate .0444673 m/s is only1.2% lower. This shallow maximum is **not** a completed
peak/decay. No physical or numerical failure occurred in the continued segment.

Mean mature accumulated slip is19.693 Dc. Integrated added damage grows to
2.76832e−4 m and peak increment .00433055. Damage remains confined to amplification
of the cap/adjacent cohesive zone. Fully decohered length stays3 m; no new persistent
geometry is formed. Slip activity propagating within the mature patch is distinct
from creation of fully decohered material. The prior matched b=a control and
pre-damage acceleration still establish RSF feedback.

The new segment has mechanical weak error <=8.06e−11 Pa, surface <=9.79e−9 Pa,
scaled source <=1.30e−14 and BE ledger <=8.41e−14 J/m. Additional friction/radiation/
fracture are .0353593/.117813/.000420881 J/m. Absolute physical-work remainder
8.77e−5 J/m is .0573% of sliding dissipation but20.8% of fracture work: the1%
fracture-relative screen still fails. Algebraic closure is reported separately.

## Cost, checkpoint and scientific endpoint

Local segment:902.93 s,4ranks, about498 MiB/rank,12288cells/185667reportedDoFs.
Warm steps cost8.11 s and advance about3.83e−5 physical seconds each. At unchanged
throughput another .1 physical seconds costs about5.9 hours. The approximate
radiative low-mode scale eta_rad/k_min=.03184 s suggests order .07s for tenfold
decay after a peak if geometry remains fixed; growth can change that estimate.
These measurements justify server continuation rather than declaring a wall-time
stop to be arrest. No local MPI speedup or completed-event timing is claimed.

Figures use **t_ref=374.5 s**. The first1e−3 m/s crossing is bracketed at
[374.461102325,374.466166456]s. They show x–time slip activity separately from damage
and a=1 fronts, plus progress in accumulated mature absolute slip/Dc. Lines and
spatial interpolation are visualization only; CSV retains native Gauss values/IDs.
The declared completion diagnostic is global rate below10% of the observed maximum
for max(.005s,5Dc/V_peak), after exceeding1e−3 m/s. It has not been met.

The server package contains current source/plugins, compatible-environment records,
the complete accepted4-rank checkpoint and full zero-advance reference, portable
inputs, native wall-budgeted resume and startup/control checks. Direct `ibrun`
commands are in its README. The optional fresh coarse/fine inputs double both bulk
and surface resolution with fixed physical parameters/lengths; they remain prepared,
not qualified, and do not remap the present history. Event timing must be compared
separately from progress-aligned fields. Production/BP3 and earlier evidence remain
protected. The next physical action is to continue this checkpoint through genuine
decay, not to change material parameters or introduce a new propagation law.

## Changed files for this repair

The numerical changes are `cohesive_timestep.h`, `cohesive_evolution.h`,
`cohesive_system.h`, `cohesive_advance.cc`, `time_stepping/cohesive_rsf.cc` and
`unit_tests/cohesive_timestep.cc`. The native plugin's optional full-record export
and restore check support the lifecycle verification. The instability fixture
directory contains the selected inputs, event analysis and server-package tools.
The estimator/report, STATUS, SOURCE_AUDIT and paired current_design/specification
record scope and equations. `source-manifest.json` in the server package identifies
the complete delivered source, including preserved pre-existing local changes.
No commit, remote submission or new physical calibration is part of this handoff.
