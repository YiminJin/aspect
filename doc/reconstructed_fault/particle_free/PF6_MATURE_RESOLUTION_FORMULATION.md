# Bounded mature mechanical/profile-resolution separation

Selected 16 October 2026, branch `pf-rsf-particle-free`, base a8707307008401ac9cff702a369e2e2a611fa0f7 plus retained work.
This selects fresh serial fixed-mesh, stationary mature trajectories only.
Retain unlimited DGQ2, native constitutive/RSF equations, physical loading,
Gauss6 bulk mechanics, Gauss3 DG source projection, unchanged surface mesh,
normalization law/derivatives and acceptance/publication checks. No phase
stepping, transport, limiter change, AMR, restart qualification or event run.

## Matrix and immutable fields

Compare coarse mechanics/coarse phase (CC), fine mechanics/coarse phase (FC),
and fine mechanics/fine phase (FF). Use the retained 4136/11528-cell meshes,
59 surface nodes, dt=200000 s, initial constitutive interval 200000 s and three
positive steps, T=600000 s. Repeat the three cases with matched native particle
history controls (D3 convention, actual LLS transfer, stationary points).
Particle–DG comparisons are backend differences, with neither an accuracy truth.
Native attempt budget: 3600 s total, serial one thread, at most 700 s per case.
Compilation and offline reduction are excluded. Preserve all old runs/failures.

CC/FF use the same physical native prescribed seed as the retained experiment.
FC imports **only** the accepted step-zero coarse Q1 phase nodal field, including
its hanging-node values. All T/Z histories remain unloaded; surface Theta,
physical loading, geometry, profile law and irreversibility conventions match.
Fine cells are contained in coarse cells. Their prescribed vertices evaluate
the complete coarse bilinear polynomial, so fine Q1 exactly prolongs coarse Q1.
The exterior Q1 phase lattice stays at the coarse 7.8125 m spacing; refinement
of mechanics must not resample that exterior field. The native completion
validator must still check every phase constraint, admissibility, affine cells,
reference field agreement and the exterior trace. Native integration/projection
computes I; no normalization values are imported or overridden. Check common
physical phase/chi and all nodal I before interpreting evolution differences.

The default-off immutable reference hook is restricted to a fresh serial frozen
mature fixed mesh. It supplies a physical reference evaluator and explicit
exterior spacing, not constitutive history or replacement normalization.
Existing automatic completion without the hook retains its original admission.
An explicit existing `FE bulk normal quadrature points` value above three also
selects the shared mature tensor rule for particle controls; default three
retains native particle quadrature. Verify points and weights, not just input.

## Measurements declared before validation

Use the fine mesh's Gauss3 overlay: 103752 common physical points, with explicit
native-cell trace ownership for each mesh and identical positive physical weights.
These are diagnostic integration points; Gauss6 mechanics is unchanged. Independently
verify prolongation at fine vertices and nonnodal points. Report full plane-strain
norm |tau|²=xx²+yy²+zz²+2xy², zz=-(xx+yy). Read accepted authoritative DG tau;
for particles evaluate the actual accepted native operator from immutable incoming
mapped stress and current FE strain/localization, without a second update.
Export T, Z, chi separately where owned; particles have no independent T/Z.

Report differences at all three positive times and step zero separately. Use
whole-domain and common physical-profile-weighted RMS tensor norms, absolute
maxima and symmetric relative differences (denominator sqrt((||a||²+||b||²)/2)).
The common localization weight is the native physical profile's h, independent
of either represented phase. Keep both normal-cut face traces/jumps separately.

Partition bulk points and surface rows by physical down-dip coordinate s:
**endpoints** s<125 m or s>L-125 m; **interior** is the complement. This fixed
125 m window contains the measured 121.407 m completion influence on both meshes.
Also report the full domain. Do not average ambiguous cell-face traces.

Export actual shear, raw-normal and filtered-normal weak loads and mass matrix.
Subtract M*1 times the common background traction before scaling; normalize each
pair's weak-load differences by the larger background-subtracted perturbation
amplitude of the two trajectories, never by 50 MPa. Report absolute Pa m and
mass-inverse projected Pa as well. Report V/Vp, Theta/(Dc/Vp), accumulated slip,
normalization and mechanical residual/publication results. A 1% perturbation/
Vp/Theta screen is descriptive, not a declaration of convergence from two meshes.
Compare spatial effects with retained Gauss6/8 and coarse dt/2 uncertainty,
explicitly retaining the previous failed 1% temporal-load screen. This task adds
no timestep sweep or higher spatial reference; it does not certify continuum
accuracy, either backend, temporal order, phase evolution or propagation.

## Export audit clarification before comparison reduction

The exported native mass is a **bulk-work** mass, assembled with chi and the
bulk quadrature, plus paired completion. A fixed surface grid does not make
that discrete measure identical when the bulk partition or profile changes.
Even CC/FC's identical physical localization can have a small quadrature-partition
mass difference. Keep this distinction in the reduction: subtract **each case's
own** M*1 background, compare the actual native perturbation weak loads, and
report total/background-measure weak-load differences separately. Also evaluate
a common diagnostic functional: apply the identical geometric surface Q1 mass
to each native mass-inverse projected traction minus the common background.
This isolates a shared functional without replacing the solver's native measure.
No native rule, mass, normalization, acceptance condition or tolerance is changed.
The full embedded deviatoric tensor tau is the independent reconstructed
history stress. Report pressure separately and also the full Cauchy difference:
|delta(tau-p Id)|² = |delta tau|² + 3(delta p)². Surface background traction is
separate from this bulk perturbation stress.
