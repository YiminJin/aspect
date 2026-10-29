# Temporary candidates and energy-accounted formation

Baseline f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved work.
The user selects the formerly proposed distinction between provisional virgin
geometry and persistent formed geometry. Production/BP3 and all prior evidence
remain protected. This is an isolated serial native qualification, not a new
constitutive law or automatic large propagation run.

## Contract before implementation

- Persistent geometry is append-only. Every cell carrying any nonzero damage,
  signed/absolute slip, cohesive reference or relaxed Maxwell jump is formed.
  These tests are exact structural tests, not magnitude thresholds. One Q1
  surface cell is the atomic geometry record: if any owned Gauss point forms,
  its entire cell becomes persistent. Existing larger polynomial pieces remain
  immutable; new candidate pieces each contain one cell.
- A trailing candidate may be discarded/reoriented only when all five physical
  Gauss histories are exactly zero, all associated nodal displacement/Maxwell
  coefficients are exactly zero, and its age follows the retained virgin clock.
  Zero memory here means no additional interface/lifting history: the physical
  bulk Maxwell tensor is generally nonzero in this loaded specimen and must be
  preserved everywhere. Tied reaction is a diagnostic multiplier, not material
  memory. No old formed
  cell is rewritten. In-plane pressure, physical stress and storage must be
  unchanged by a pure candidate replacement.
- The existing SVD constraint projection leaves ~1e-20 m modal shadows in
  materially exact-zero cells. Do not delete those from an old checkpoint.
  Use analytical elimination of Q1 nodal coordinates implied by three tied
  Gauss constraints in the NEW experimental native solve. This is exact
  constraint enforcement, not clipping or a history reset; defaults stay off.
- Start from the original accepted .410 s checkpoint BEFORE extension 2 in the
  preceding short-reserve experiment. Both existing pieces already carry real
  material history; retain them all, including every tensor/modal/Gauss history
  and clock. Only newly proposed zero records receive temporary ownership.
  This introduces ownership metadata, not a scientific checkpoint conversion.
- Direction selection samples the accepted physical stress using the same
  R=.75 m disk and persistence equation, now anchored at the PERSISTENT formed
  endpoint, not the end of an unused reserve. This explicit change to the
  propagation hypothesis removes reserve length from the sampling location.
  Both conjugates and the existing history preference remain available.
- Candidates are fixed during each nonlinear attempt. A cached candidate is
  refreshed on a declared numerical cadence (.0025 s initially), and always
  rebuilt from the current accepted stress before the first trial that would
  form new material is accepted. Reject that predictor trial; solve again from
  the unchanged incoming histories. Phase and all slip variables remain free.
  Cadence is numerical and receives one focused halving check; it is not R.
- Zero-time candidate replacement is a change of trial representation only.
  If its intact weak multiplier exceeds 3000 Pa it MUST NOT be published as an
  admissible tied state. Instead the positive-time native solve jointly admits
  slip/damage, paying Gc times actual damage plus cohesive storage and existing
  dissipation. Original pure-tied birth guards remain unchanged. Full native
  weak/contact/tangent checks decide the joint endpoint; no intermediate tied
  publication or unconditional acceptance of an overloaded candidate occurs.
- Classify and commit formed cells only after the physical trial passes. The
  metadata commit adds no storage, fracture or work beyond the just-solved
  physical update. Publish geometry and all histories atomically once. Rejected
  trials leave both the persistent registry and physical clock unchanged.
  Candidate views have separate versions; they do not count as persistent births.
- Keep at least one wholly zero candidate cell ahead of trial growth. If the
  last trial cell forms, enlarge the temporary view and restart the SAME trial
  from the original accepted state. Bound this loop; do not cap physical growth
  at one cell per step. Checkpoint persistent history and the explicitly tagged
  temporary view; verify rebuilt-context restart and changed candidate cache.

## Bounded qualification

Retain the exact cohesive–RSF calibration, pressure convention, full plane strain,
R=.75 m, resolved numerical support w=.75 m, native Q2/P1disc, Gauss6 bulk and
Gauss3 surface histories, and existing residual/FD/energy/contact tolerances.
Use the preceding physical loading turn unchanged.

First verify inactive candidate replacement in a nontrivial loaded state,
constraint equivalence, default identity, independent tangent/work derivatives,
and actual rejection/retry/publication/restart. Then run .410–.425 s at
dt=.00025 s with 2 versus 8 candidate cells (initial reserves .1875/.75 m).
Compare common formed fields/fronts, not tied tail endpoints. Include one
dt halving and one candidate-refresh cadence halving over a short own-state
suffix. There is no parameter search or extended trajectory.

Practical gates remain 5% growth/slip, one .09375 m front cell, 1 degree
selection uncertainty, 1% summed absolute physical-work remainder relative to
sliding AND added fracture, 1e-7 Pa transfer stress, and unchanged native
1e-8 Pa weak / 1e-6 derivative gates. Exact BE closure stays separate.
Preserve every failed gate, including the old point-traction and ahead-normal
failures. Admission-time microdamage is not developed propagation.

Stops: .425 s, 1.5 m new geometry per attempt, original contact/tangent/solver
limits, at most four candidate-enlargement retries per physical step, 1200 s
per trajectory, 300 s per kernel and 1.2 GiB new raw output. Preserve rejected
proposals and last accepted states. No production integration, calibration of
R, free branches, AMR, transport, MPI expansion or long propagation run.

## Numerical sampling detail (before runs)

Export the same physical R-disk tensor average at every immutable arc endpoint,
using fixed Gauss8 per bulk coordinate, independent of candidate cut partitions.
This changes only the new selector's averaging quadrature, not mechanics or
source/energy integration. Verify it once against Gauss12 and the existing
cut-split disk measure. Otherwise even zero candidates can change sampling
arithmetic through their quadrature cuts. No raw point-stress claim follows.

## Measured temporal follow-up

The first two-cell run completes .425 s with eight material-cell commits and
60 physical publications. Its coarse-step absolute physical-work remainder is
1.47568% of added fracture (0.48771% of sliding), failing the retained 1% screen.
Use the already selected ONE timestep halving over the full .410–.425 s window
from the same original checkpoint, rather than only the final suffix. This
isolates the accumulated error without changing physics or the screen. Keep
the cadence halving as the shorter own-state .420–.425 s check.

## Onset-specific cadence control

The .420–.425 suffix cadence check matches the coarse response exactly, but it
does not exercise first admission. Apply the SAME already selected .00125 s
cadence over .410–.41675 s, covering the first two coarse admissions. This
focused follow-up tests cached-predictor onset sensitivity; no additional
cadence value, loading, physical parameter or threshold is searched.

The onset cadence check completes at .41675 s: both admission clocks and all
persistent histories match the original cadence exactly. The diagnostic late
state control separately takes ONE .44625–.44650 s joint step from the archived
failed state; it is not a continuation trajectory beyond the declared .425 s
main endpoint. The original tied-birth rejection is actually repeated on a
private operator copy; joint admission then pays the new damage/cohesive energy.
