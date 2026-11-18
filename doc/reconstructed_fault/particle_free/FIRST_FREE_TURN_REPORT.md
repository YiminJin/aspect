# First native free turn

**Four-rank result reviewed:** the positive-load history accepts through
**.075 s**, but the .080 s interval fails the original phase-KKT gate after
80 Newton iterations. The process exited with code 1; it did not finish the
target trajectory successfully. The last accepted checkpoint remains available.
At .075 s the absolute phi=.05 contour has a small connected asymmetric extension
beyond the original endpoint, only .0329 axial cells. This is an unresolved
turning candidate, not a qualified turn. The reflected run was **not launched**;
the originally planned paired physical comparison remains incomplete.

The latest user request selects reporting and visualization of the completed
process. No new simulation, solver correction, tolerance change, rebuild, or
commit was made during this review. Implemented and short-control-verified do
not mean physically converged or production-ready.

## Figures and reusable analysis

- [Accepted phase: contour, cells, increment map and phi0/phi/delta-phi profiles](../../../benchmarks/reconstructed_fault/particle_free/evidence/first_free_turn/free-turn-accepted-phase.png)
  ([PDF](../../../benchmarks/reconstructed_fault/particle_free/evidence/first_free_turn/free-turn-accepted-phase.pdf)).
- [Structural planes, ages, signed slip and rate](../../../benchmarks/reconstructed_fault/particle_free/evidence/first_free_turn/free-turn-accepted-histories.png)
  ([PDF](../../../benchmarks/reconstructed_fault/particle_free/evidence/first_free_turn/free-turn-accepted-histories.pdf)).
- [Accepted growth and signed/absolute physical-work history](../../../benchmarks/reconstructed_fault/particle_free/evidence/first_free_turn/free-turn-evolution.png)
  ([PDF](../../../benchmarks/reconstructed_fault/particle_free/evidence/first_free_turn/free-turn-evolution.pdf)).
- [Accepted step15 versus failed step16: convergence, search and cost](../../../benchmarks/reconstructed_fault/particle_free/evidence/first_free_turn/free-turn-four-rank-convergence.png)
  ([PDF](../../../benchmarks/reconstructed_fault/particle_free/evidence/first_free_turn/free-turn-four-rank-convergence.pdf)).

All physical fields in these figures are accepted. No reflected field is invented
by mirroring the available result. Native cellwise contours preserve separate
DG traces. The original straight-load contour is an existing same-time control;
the comparison changes loading as well as the enabled closure and does not isolate
the effect of plane selection alone. `positive-accepted-history.csv`,
`endpoint-profiles-0015.csv`, `four-rank-nonlinear-history.csv`,
`accepted-history-summary.json` and `completed-result.json` provide the plot data.
The [runner/analysis README](../../../benchmarks/reconstructed_fault/particle_free/first_free_turn/README.md)
gives exact commands.

## Selected construction and preserved behavior

[Equations, assumptions, ownership and code map](FIRST_FREE_TURN_FORMULATION.md).
The user-selected material-area/reference-arclength law is implemented with
complete vertical material strips, immutable membership and ds=dx/cos(a).
The full-height skew chart covers the material once; it is not a fan of rotating
normal rays. Plane angle, material section and measured phase contour are distinct.
Previously formed sections keep their plane and reference measure. Prospective
age starts at .5 s at t=0 and evolves during quiescence. Full native tau/A persist.

The additional plane rule maximizes the full-section integral of physical Y0
within the horizontal seed's conjugate sector, with a consistent stage RSF
solve for each trial angle. This nonlocal choice and generalized slip are
constitutive assumptions. The existing mechanics/phase solver repeats after
an angular update and publishes once. No material transport, instability,
branching or general contact is included. The fixed-direction default, BE
reference, native Bernstein phase, midpoint update, quadrature and acceptance
tolerances remain unchanged. The new checkpoint law rejects straight checkpoints.

## Completed native checks

| Check | Result |
|---|---|
| Protected/default/opt-in straight smoke, two ranks | All accepted; largest compared difference 1.32349e-23 |
| 128×128 pre-turn endpoint at .04 s | Phase and every exported section quantity exactly match the protected control |
| 32×16 append control, 16 midpoint steps to .08 s | All accepted; first new section registration at .075 s |
| Actual physical-weight rate / force adjoint at first append | Rate error 5.21e-18; power error 4.45e-15 |
| Independent direct-quadrature Y0 optimization, 12 appended sections | Maximum relative objective gain below 9.66e-16 |
| Nonuniform loaded parent14 restore | Full exported physical tensors, phase, age/slip, plane and measure exactly equal |
| Rejected first append, deterministic retry, second rollback | Existing fingerprint checks pass, including material membership and weights |
| Recovery endpoint16 versus uninterrupted endpoint16 | All compared fields exactly equal |
| Membership / metric | Unchanged x identities; dx=ds cos(a); full domain area 32 m² |

The coarse control costs 75.67 simulation seconds (66.56 coupled solver seconds);
recovery/rejection costs 72.8 seconds for its two physical intervals and repeated
trials. It is a lifecycle check, not a phase-width or turning-accuracy result.
At .075 s old ages span .2265–.5724 s; newly registered ages remain .575 s within
9.23e-13 s. New slip is 9.66e-19–8.06e-16 m and angles 10.687–10.864 degrees.
Old formed planes remain exactly unchanged. Prospective zero-rate age agrees
with .5+t to 1.12e-16 s. The coarse absolute phi=.05 contour remains inside x=6.

## Retained small-phase admission diagnosis

The first 12 coarse sections register at maximum phase 2.39118e-12 and
A norm 6.65180e-15. This is **not** evidence of resolved growth. Every affected
phase row is classified lower-active at the accepted approximate solution,
but putting the tiny values exactly on zero raises actual phase KKT to .0596.
The regularized RSF cold-creep reaction changes sharply near intactness.
Unconditional active-bound polishing cycles and fails step15 at the original
80-iteration limit, leaving parent14 intact. Its executable, source and evidence
are preserved; it is not the tested current implementation.

The current opt-in code tests an exact active-bound candidate against the full
original mechanics/KKT gates. An inadmissible zero candidate is discarded;
the original admissible positive solution remains. This uses no positive phase
cutoff, tolerance relaxation, accepted-history reset or post-publication clipping.
Infinitesimal plane registration is sensitive to the cold-creep scale and ordinary
KKT stopping accuracy. Resolved phase-path advance must be assessed separately.

## Accepted 128×128 result

The positive history is accepted through .075 s; the .080 interval failed.
The mesh spacings are hx=.0625 m and hy=.03125 m. Through .060 s no new section
is formed. The first append at .065 s is materially different from the coarse
lifecycle tail: the largest new-section phase is .00131361. Its 12 new sections
have angles .155132–.155927 rad, ages .564790–.565000 s and cumulative slip
2.21e-10–1.87e-7 m. The original planes remain exactly horizontal. The separate
first-append checkpoint is retained in `first-free-turn/first-append128`.

At .070 s a further 12 sections register at phase at most 7.30e-12, with angles
.173254–.174137 rad. Testing zero active bounds raises phase KKT to .0969/.0934;
the admissible positive solution is retained. Thus the admission-scale limitation
also occurs on the selected mesh. No positive cutoff hides these registrations.
The old/new structural plane jump reaches .155927 rad over adjacent native
sections; this is a piecewise smooth material-frame construction, not evidence
of a smoothly curved crack. The actual phase contour is measured independently.

At .060/.065/.070 s added fracture energy is .00294838/.00619679/.01264145 J/m.
The absolute phi=.05 rightmost point remains inside the original x=6 m endpoint:
x=5.92185/5.92185/5.94009 m, respectively. These intermediate states alone do
not establish a turn beyond the seed.

At .075 s the native Q2 contour extremum is **(6.00205438, 2.00936499) m**.
Relative to the original endpoint (6,2), axial advance is 2.05438 mm = .03287 hx
and signed lateral displacement is +9.36499 mm = .29968 hy. The initial absolute
contour ends at x=5.92185482 m. The original straight-loading run at the same
.075 s still ends at x=5.98673335 m. The new absolute contour is connected to the
seed in both 16- and 32-sample-per-cell superlevel checks, with no disconnected
component ahead. An independent exact Q2-root cut at y=2.00936499 gives one
phi>.05 interval [2.00286291,6.00205438] m; adjacent pieces are joined only when
both independent DG face traces exceed .05. This confirms a connection to the
interior without averaging DG jumps. These are observation checks, not spatial
convergence tests.
Contour subdivision 64→128 changes x_max by 2.83e-7 m; direct local Q2-root
optimization changes it by another 3.96e-7 m. The extraction is much finer than
the physical mesh, which remains the limiting resolution. No converged turn
angle or propagation speed is inferred from the endpoint displacement.

| Accepted time (s) | Added fracture (J/m) | Absolute contour rightmost x (m), 32 subdivisions | Central W/hy |
|---|---:|---:|---:|
| .040 | 0 | 5.921855 | 1.3941 |
| .060 | .00294838 | 5.921855 | 1.5586 |
| .065 | .00619679 | 5.921855 | 1.6438 |
| .070 | .01264145 | 5.940095 | 1.7604 |
| .075 | .02726371 | 6.002050 | 1.9011 |

The peak remains .60; W is the connected half-maximum excess-compliance width,
not the phi=.05 contour thickness. Added fracture at .075 is distributed as
.01958008 J/m central band (71.82%), .00595333 old taper (21.84%) and
.00173031 initially intact axial region (6.35%). Remote and boundary regions
have zero exported phase increment and added fracture. The separately reported
outer-taper region is a subset of the old taper and must not be added again.
The response is predominantly band/taper evolution with a small connected axial
extension; it is not solely remote or isolated damage.

There are 384 original formed sections and 60 formed since initialization.
Original angles stay exactly zero, ages span .206215–.541802 s and signed slips
29.9304–552.878 µm. Newly formed angles span 8.8884–10.8578 degrees, ages
.542354–.575 s and slips 5.32e-14–29.4084 µm. The remaining 324 prospective
sections have zero slip and age .575 s. Their displayed candidate directions
are not recorded fault planes. At step15, 12 of the 36 newly registered sections
have phase maxima 4.31e-6–.00352867, while the other 24 are below 2.67e-13.
Formation count is therefore not a resolved crack-length measure.

The final section-integrated signed fault rate is .0372478 m²/s with zero
negative-rate contribution. Local friction/radiation power is 4.96943 J/(m s),
with zero negative contribution, and equals the section power to printed
floating-point precision. This last-step reduction uses the separately exported
positive/negative local-rate integrals and the uniform section resistance.
It does not identify how many positive bands occur within each section: the
sparse point export has no step15 tensor/rate frame. Full tau/A are retained in
the checkpoint, but no final spatial stress/traction or multiple-band claim is
made without an additional read-only export. The coarse append independently
checks the actual point-weight power adjoint.

The cumulative physical-work remainder is
sum(Wext − delta stored energy − Db − Dc − Df) = **+5.58961e-5 J/m**;
the sum of its absolute per-step values is also 5.58961e-5 J/m, **.2050%** of
added fracture. Over .040–.075 s these are both 5.54408e-5 J/m (**.20335%**).
The final interval contributes 2.22324e-5 J/m, .15205% of that interval's added
fracture. The corrected ledger error at step15 is −1.84e-14 J/m; that smaller
number includes the endpoint remainder and is not a substitute for physical-work
accuracy. No work or energy result is assigned to the rejected .080 trial.

## Final-interval failure and computational cost

The four-rank continuation lasts **5170.39 s = 86.17 minutes**. Step15 uses
1099.89 coupled solver seconds (18.33 minutes); the remaining 67.84 process
minutes include the failed step16 and restart/output overhead. This exceeds the
earlier timing estimate because the final phase solve stagnated. Across all three
positive windows, recorded simulation process time is 174.32 minutes including
the deliberately discarded two-rank trial15; accepted-step solver time totals
66.58 minutes. Builds, short controls and compression are not included in these
simulation-process totals. Prefix process time is 1933.99 s; its outer wrapper
including compression is longer, so those timer definitions must not be mixed.

Step16 reaches mechanics 2.7219e-16 and section-rate defect 1.02349e-16, but phase
KKT remains **7.02479e-7 > 1e-7** at Newton80. Late searches accept only
alpha=.000488281 (approximately 1/2048), making very little progress. No virgin
plane recheck is reached in this interval, and no sector-boundary failure is
logged. The first blocking acceptance condition is the phase VI, not mechanical
equilibrium or the angular iteration. The logs do not contain residual-row or
active-set field exports, so the specific obstructing rows and whether they share
the previously diagnosed cold-creep mechanism are **not established**.

The last-good pointer is slot `01` under
`build-particle-free/bulk-tip/runs/first-turn-positive-fourcore-end/output/restart/`.
It was written after accepted15; no step16 accepted rows, field exports or joint
publication exist. `completed-preservation.json` records checkpoint hashes,
inputs, log hashes and unchanged tested/protected executable identities. Parent14
phase coefficients, solution coefficients, section/plane state and clock match
exactly after the rank transition. A parent14 full point export was not scheduled;
the independent short nonuniform rank-transition control supplies the full-tensor
transfer check. This review performs no additional recovery solve.

## Decision and next justified experiment

Retain the opt-in implementation and all successful and failed evidence. The
result supports native coupled formation with inherited section state and a small
asymmetric phase extension. It does **not** yet support a resolved free turn,
reflection symmetry, spatial/time convergence, smooth plane continuation or
production readiness. The reflected comparison and targeted physical refinement
remain unperformed; no new simulation is launched for this reporting request.

The next bounded experiment should begin from the saved .075 parent and instrument
the failing phase rows, active-set changes and search merit during the same .005 s
trial, with unchanged acceptance gates. Locate the first obstruction before
changing the solver or constitutive admission rule. Once that interval is reliable,
complete the reflected comparison at a common accepted time; only then select a
focused refinement to distinguish the subcell turning candidate. Increasing the
iteration limit alone is not justified by the nearly flat late residual.

## Reproduction and preservation

Current root/branch: `aspect-pf-rsf-particle-free`, `pf-rsf-particle-free`.
Revision at entry: `b32a2d4f6e21c219f9af786d1b6e3725ea78a5a4`, with incoming
manual changes retained. HEAD alone does not identify the executed source.
The reused build directory prints its older configured revision `5f1fc6f3c`;
that banner is not the source identity. `tested-state.json`, the source archive,
build log and executable hashes identify this tested state.
No checkout, reset, commit, server job or checkpoint-law conversion was made.

| Artifact | SHA256 |
|---|---|
| Current tested core | `089824ae44c9c53b57921fe0c5019c59f4a37cf48e1f753917e232f4e4c22378` |
| Current read-only checker | `81efcd6e5e0013f4124bf1e1ba857005a2a23447985f69b16464fbc4d3131d69` |
| Protected straight core | `034d2a848a3a10c496fa097475f31497c00bc68c3456fb6d87c1ee700f453c27` |
| Protected straight checker | `b800ef431a2469760efd16befcba94e8cd743156629b55e4c921f769255d9d84` |

[Runner and inputs](../../../benchmarks/reconstructed_fault/particle_free/first_free_turn/README.md)
and [compact evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/first_free_turn/)
include entry diff/hashes, build logs, tested-source archive, exact commands,
smoke comparison, recovery comparison, independent angular check and native
analysis. Full outputs/checkpoints remain under `build-particle-free/bulk-tip/runs/`.
Each run has an independent command/source/binary manifest. The core source
changes are the material selector/history/serialization and existing bulk-tip
coupling/solve gate. No alternate timestep driver is introduced.

To make room, 15 closed historical CSVs were losslessly gzip-compressed with
roundtrip SHA checks, saving 468,269,266 bytes; `lossless-space.json` lists them.
64 pre-existing broken aliases were recorded, not repaired or deleted. No new
physical data loss is introduced. Earlier cleanup's stated losses remain losses.
New closed exports use the same verified compression; checkpoints, runtime
libraries, protected and failed executables remain available.

## User-selected four-core continuation and space recovery

The user subsequently requested four cores. Rank changes are made only at
accepted checkpoint boundaries, with a short nonuniform two-to-four-rank
restart comparison before the larger continuation. MPI ranks bind to distinct
physical cores, one thread each. Four-rank runs defer compression and substantial
analysis until the solver exits, preserving the aggregate worker limit.

The two-to-four-rank nonuniform recovery check restores the physical parent
exactly. Its first-append endpoint differs by 1.47e-14 in phase, 2.56e-15 in
relative stress L2, 1.25e-14 in relative A L2 and 1.10e-8 rad in angle (below the
1e-7 angular recheck gate); formed flags agree. The large two-rank continuation
accepted steps13–14, then its unaccepted trial15 was intentionally stopped to
change ranks. This is an operational interruption, not a failed scientific gate.
The four-rank run resumes accepted14 and accepts15 in 1099.89 coupled solver
seconds. A comparable 75-iteration linear solve falls from 55.931 to 36.6841 s,
a 1.525× speedup. This does not establish a whole-run speedup: nonlinear search
and active-set paths can differ at roundoff. The discarded trial cost is retained.

At the user's subsequent cleanup authorization, two generated GCC precompiled
header caches were removed from `bulk-tip/native` and `material-overnight/native`,
reclaiming 1,708,278,141 bytes. No compiler/build process was active. These two
compiler intermediates must be regenerated by the next affected build; this is
the entire new availability loss. Numerical data, source, input, checkpoints,
executables and runtime libraries were not removed. Exact paths, hashes and
rebuild commands are in `four-core-cleanup.json`. Earlier cleanup losses and
broken aliases remain explicitly unrepaired. The tested C++ sources and binaries
were rehashed and remain identical to the recorded tested state.
