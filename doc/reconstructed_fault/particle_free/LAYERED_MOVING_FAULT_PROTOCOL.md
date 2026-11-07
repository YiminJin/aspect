> **Historical total-spin campaign.** The revised user selection resumes fresh corrected-law work under [LAYERED_BULK_CAMPAIGN_PROTOCOL.md](LAYERED_BULK_CAMPAIGN_PROTOCOL.md). Preserve the evidence and interrupted L0 below; do not resume its checkpoints under bulk spin. Its stop was explicit SIGTERM, not a demonstrated native failure.

# Layered moving-fault campaign — 5 November 2026

Selected by `CODEX_LAYERED_MOVING_FAULT_CAMPAIGN.txt`. This supersedes the older
recommendation to stop after another homogeneous timestep test. Worktree root
and branch verified at cf06525471ce89789f1ac10905dc79f3aac7562a plus retained
prescribed-exterior source/doc changes. Entry hashes are in
`benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/layered_campaign/entry.json`.
Use the matching `stage-c/exterior-stress/aspect-release` and checker; preserve
`native/aspect-release`, older providers, all outputs and checkpoints. Four local
workers maximum; sequential native runs. No production/server trajectory.

## Configuration and background

First close the existing homogeneous compatible-exterior sequence at 32²,
dt=.00625, end=.4. Keep its original 18 labels and initialization. Compare all
common times and common central/end regions against retained coarser cases.

Layered cases use G=10, eta(y)=5.5+4.5*tanh(y/.35), constant physical transition
width, domain [-4,4]×[-2,2], h=.125 initially, and the mature fault from
(-.8,-.48) to (.8,.48). The compact profile follows y=.6x, with unchanged .6
vertical width, amplitude .35 and 1.4 horizontal envelope. Surface spacing .1
produces 20 labels, fixed in time/mesh comparisons. Friction parameters,
initial material-label tractions/age, growth off, DGQ2/SSPRK3, accepted-gradient
spin, bounded MacCormack profile and native endpoint/coupled solve stay intact.
Added U=0,+3.125,-3.125 gives 1.25 m (ten coarse cells) over .4 s, with additional
shear/fault motion measured rather than assumed. L0 still moves physically. A preliminary U=2.5 run measured +.156 m of
shear/fault drift, which would reduce mean negative motion to 6.8 cells; the
main pair therefore uses |U|=3.125 to retain at least roughly eight actual cells.
The preliminary run is preserved and is not mixed into the main pair.

The benchmark background integrates the scalar recurrence in the selected task
on 64 physical depth panels with Gauss8 interpolation/integration. It retains
accepted deviatoric stress and shear gradient. Candidate rotation uses accepted
gamma, reaction uses beta=exp(-Gdt/eta), kappa=eta(1-beta), and the uniform q
chosen so integral gamma_next=2. Integrating gamma_next gives horizontal
boundary velocity U-1+integral_{-2}^y gamma_next. Interior velocity is solved.
Exterior transport receives accepted tau(y), unchanged during pure horizontal
background transport. Top total traction is (q,0), with p=tau_yy. This normal
pressure convention is explicitly selected by the layered task; for constant
eta the stress/shear recurrence reduces to the previous homogeneous reference,
but the latter used p=0 and a nonzero normal top traction. Do not equate their
pressure-inclusive fault loading without that distinction.

Initial mechanics uses .1 s and zero parent stress. Its gamma becomes the
accepted initial gradient, but its diagnostic stress is not published.
`start_timestep` builds a candidate from accepted state; ordinary postprocessing
runs only after native/controller acceptance and publishes it. Rejection does
not call that publication. Plugin save/load persists accepted state and identity;
no production checkpoint schema changes. Test scalar quadrature, constant-eta
reduction, velocity difference, normal balance, rejected candidate and matching
restart continuation before claiming usable variable timesteps.

## Bounded matrix and assessment

Run L0/L+/L- at a common timestep informed by homogeneous closure, plus one
halving and one bulk refinement of L+. Hold surface labels fixed. Use the
existing reconstructed-fault slip/state timestep controller with a measured
cap; add a Function schedule only if needed to exercise nonuniform intervals.
An intermediate checkpoint continuation must compare the same accepted times
with the uninterrupted run, preferably changing ranks. A short rejected step
must match its explicit accepted-schedule control. If directional sensitivity
exceeds discretization sensitivity, investigate within this task, with at most
one enlarged-domain comparison where finite boundaries are implicated.

Report absolute and relative history/final stress, work shear and
pressure-inclusive normal traction, slip rate/cumulative slip/age, geometry,
profile and normalization, signed and absolute work/energy/flux defects, and
wall/solver cost. Plot U*t-shifted fields and label histories; animate curve and
profile crossing the fixed cells. Distinguish pairwise sensitivity from exact
error, endpoint physical flux from SSPRK stage flux, and physical finite-boundary
response from transport error. Preserve independent cohesive failures and the
cleanup loss of 39/45 native-pilot frames and earlier temporary logs.
