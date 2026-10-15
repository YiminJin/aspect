# Coupled BP3 moving particles and population management — 2026-10-14

**The bounded lifecycle/coupling correctness gate passes. The controlled-error
exit criterion remains open at16 particles/cell:** endpoint tractions and
short-wave slip rate remain materially sensitive to particle representation.
No first-event, full-model, MPI transport, unfiltered or later-stage run starts.

Baseline: `fault-propagation`, `c33edaaeb145e5d3dce155bb815552ef80efc5f9`, preserving
all incoming P2/BP3 edits and evidence. Snapshots/provenance are in
`build-bp3-native/managed-coupled/`: `incoming.{json,diff}`, compilation/link command
JSONs, `primary-v3-artifacts.json`, `final/`, `preservation.json`, and `assessment.json`.
The existing runner records exact commands, environment, logs, elapsed time and
exits as `refactoring_boundary/evidence/bp3-managed-*.{json,log}`.

The translated60-degree 2000×1000 m fixture retains4136 graded Q2/Q1 cells
(7.8125–125 m),149538 DoFs and59 surface nodes (19.90863 m actual spacing).
Phase, geometry, general consistent normalization, homogeneous G=32038120320 Pa,
eta=1e26 Pa s, strengthening a=.025/b=.015, Dc=.008 m, physical Vp=1e-9 m/s,
full bottom velocity and the effective20-m normal filter are unchanged. Both
runs use the same native executable, maintained plugin and read-only observer;
matching mechanical quadrature CSVs verify the native3×3 Gauss rule. No accelerated
loading definition is compiled. Checkpoint/resume and automatic retries are disabled.

RK2 and native point-density addition/removal start from4×4/cell, minimum12 and
maximum24. The one5×5 density control uses19–38. Addition restores the minimum,
not the initial count. Inputs differ only in population settings/output directory;
mesh, discrete phase, I_h, surface data and initial profiles match exactly.
The initial constitutive interval and first positive dt remain200000 s. The
unchanged relative-growth cap ramps dt by1.91 before its2e8-s diagnostic ceiling.
Both complete exactly80 positive steps to T=14071021411.032413 s, with identical
accepted schedules. These large timesteps establish movement, not temporal accuracy;
maximum step slip/Dc reaches28.83. Step0 is separate initialization.

The authoritative history is the particle tensor, with no competing FE d_R:

\[
\tau_p^{n+1}=\beta\tau_p^n+2\eta_{ve}
 [\operatorname{dev}_3(\operatorname{sym}\nabla u_h)(x_p^{n+1})
 -(h(\phi_h)/I_h)_pV_hS_p],\quad
\beta=e^{-G\Delta t/\eta},\quad\eta_{ve}=\eta(1-\beta).
\]

The written law is non-objective: no spin term was added. Current strain comes
directly from bulk FE velocity. Old stress is not deviated again. Mature cohesive
traction is zero; background shear/normal tractions enter friction, never bulk
perturbation history. Surface theta and slip retain their existing split update.
At step0 mechanics uses the initial constitutive interval, while stored stress
and cumulative slip remain zero.

Actual order: back up accepted particles → RK2 motion/sorting/outflow → native
count management and late-property interpolation → property refresh/ghost exchange
→ rebuild Voronoi/CPDI → refresh associations/reconstruction from current IDs,
positions and domain version → Stokes/RSF assembly with frozen incoming histories
→ construct candidates → publish stress/state/slip once. Diagnostics only read
accepted histories. Fresh H remains prescribed; moving births interpolate current
H and stress. All survivor H/stress values remain bitwise unchanged by management.
Native integrators, population algorithms, LLS, domain rebuilds, association cache
validation and terminal candidate publication already existed and were reused.

Necessary source changes are limited to the independent, default-preserving
`Particle stress strain convention = three dimensional deviator` in point response,
bulk residual/tangent, normal/shear G and particle updates; its current admission
requires fresh frozen mature2D mechanics. The legacy strain default remains.
The previously missing FE normal-G trace correction is repaired too. Moving BP3
late H now interpolates; stationary-sample birth policy is retained. No generic
projector, CPDI geometry, material, friction, normalization or solver tolerance changed.

| Actual mechanism |4×4 primary|5×5 control|
|---|---:|---:|
|Initial particles|66176|103400|
|Final particles|65420|102458|
|Cell-crossing events|60001|90275|
|Inserted|1161|1775|
|Deliberately removed|1087|1366|
|Boundary outflow|830|1351|
|First crossing/birth/removal steps|21/30/44|19/26/37|
|Largest inherited birth stress [MPa]|413.674|356.696|

Both populations remain within their selected bounds. Every particle, including
new births, passes the terminal Maxwell check; maximum defect is1.009e-6 Pa
(<.003452 of the unchanged operand-based roundoff allowance). Theta defect is
2.221e-16; phase/I_h remain bit-identical and one publication occurs per accepted
step. Combined maxima: nonlinear residual8.192e-9 <1e-8, surface RMS1.414 Pa,
domain coverage3.0e-15 and CPDI linear-gradient error2.324e-15. Totals are152/153
Newton updates and9044/9071 Krylov iterations. Filtered off-trace G finite
differences pass at steps0/2 (worst relative4.52e-8, criterion2e-7).

Initial density discrepancies are zero. Final common-node differences below use
Vp, theta0=8e6 s,1 for Omega, Vp*T for slip, and50 MPa for all traction/tensor
scales. RMS integrates the actual Q1 surface; no extra output interpolation is
counted as resolution. Full common-time profiles and metrics are retained.

|Field|Final maximum absolute difference [SI]|Maximum / scale|RMS / scale|
|---|---:|---:|---:|
| V | 5.97202e-11 | 5.97202% | 1.13685% |
| theta | 455724 | 5.69655% | 1.11479% |
| Omega | 1.04883e-12 | 1.04883e-10% | 2.08808e-11% |
| slip | 0.037975 | 0.269881% | 0.123835% |
| shear | 2.54039e+06 | 5.08078% | 0.713886% |
| raw normal | 1.36016e+07 | 27.2033% | 5.05662% |
| filtered normal | 4.78376e+06 | 9.56752% | 1.35837% |
| pressure contribution | 8.55189e+06 | 17.1038% | 7.62948% |
| tensor contribution | 5.35262e+06 | 10.7052% | 6.48563% |

Peak transient differences are16.79% V/Vp (step64),19.10% theta/theta0,
.4125% slip/(Vp*t),37.64% raw normal/50 MPa and11.01% filtered normal/50 MPa.
The worst V/raw/filtered discrepancies all occur at the lower endpoint, at
steps64/67/69, with different crossing/population-event timing. This is combined
seeding, transport, interpolation and population sensitivity; it cannot be
assigned entirely to particle projection or to one insertion event.

Pressure and tensor contributions remain separate. Primary final raw normal
ranges−65.21 to105.75 MPa, whereas the retained filter keeps friction input
compressive13.11–54.15 MPa. Pressure/tensor cancellation and large endpoint
excursions persist in both densities. Lower200-m raw local-chord roughness is
151.20/141.03 MPa; filtered17.96/16.97 MPa. V chord roughness is5.30/1.46% Vp.
Spectral V and raw-traction wavelengths are40.50 and58.73 m, only about2 and3
surface intervals. These are roughness observations, not resolved physical
wavelengths or independent error estimates. See `managed-comparison.pdf` and
`peak-sensitivity-locations.json` beside `assessment.json`.

A retained stationary FE first-step limit differs by only7.40e-21 m/s in V and
1.04e-6 Pa in shear. This small-displacement comparison uses retained evidence,
not a fresh matched long stationary pair. The legacy particle law has a different
strain convention; transported and stationary histories need not coincide later.

Preserved failures: first observer gate accessed an unconstructed particle manager
(exit−11); connecting callbacks after simulator initialization fixes it. Primary
v1 and diagnostic replay v2 stop at step21 on a host-cell-only recurrence check.
The point lies4.83 micrometres from a face: deal.II's existing1e-6 reference-cell
search tolerance admits neighboring traces. The host prediction differs194582 Pa;
a fresh native point average matches actual stress to2.91e-11 Pa. The observer now
checks that declared sampling rule, without changing production sampling or any
recurrence tolerance. Over the complete runs there are24/15 multi-trace events;
maximum host/average stress differences5.35/10.38 MPa are recorded separately.
Their local operator effect is not claimed negligible from a small global mean.
The original failures and artifacts remain; these were postpublication diagnostic
failures, not rollback tests.

Actual verification: isolated header-dependent native build/relink (13 unity
objects), matching maintained plugin/observer builds, and final style/equivalent
expression builds pass. Native units pass700 assertions/11 cases twice. Both
step0+one-positive-step gates pass; all final/earlier gate profile columns are
bit-identical. Main/control exit0 in1632.824/1744.478 s; all launched simulations,
gates and units including failed diagnostics total3863.017 s (64.4 min), excluding
compilation. Runs overlapped; these are observed wall durations, not controlled
performance comparisons. Analysis, plot rendering, Python syntax and whitespace
checks pass. No complete P2/stationary campaign was repeated.

Changed source: `include/aspect/material_model/phase_field_fault.h`,
`source/material_model/phase_field_fault.cc`, its `constitutive.cc`/`history.cc`,
`source/simulator/assembly.cc`/`newton.cc`, `source/reconstructed_fault/surface_system.cc`,
and BP3 plugin `particle_initialization.cc`/`particle_history.cc`/`work_audit.cc`.
The existing `coupled_checks.cc`, three `managed-particles*.prm` fixtures, analysis
script, design/specification paragraphs and recovery statuses document/test the
change. Exact changes and hashes are in the preservation manifest. Original
native binaries/server archive and every incoming path are preserved; no commit,
reset, cleanup, branch switch or cluster submission occurred.

Remaining:16/cell is a tested starting policy, not an accuracy-qualified choice.
The next targeted question is lower-endpoint weak-load/reconstruction changes
across population events and their particle/bulk/surface resolution sensitivity.
Include native near-face averaging in that operator audit; its local effect is
not excluded by the passing recurrence check.
The native interpolatory insertion/removal is not a conservative merge and this
smooth test does not establish arbitrary subcell-pulse preservation during count
changes. Large-dt accuracy, evolving phase/geometry, objective rotation, transport
MPI/migration, restart/conversion and automatic cutback are not qualified.
Stop here; no first-event preparation is promoted. Future long tests should use
parallel execution with at most four processes, per the user's latest preference.
