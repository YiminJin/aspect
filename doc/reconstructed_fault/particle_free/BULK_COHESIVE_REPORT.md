# Bulk cohesive implementation and bounded response — 20 October 2026

**Implemented and verified within the selected serial fixed-mesh scope; finite-tip
propagation and physical energy qualification do not pass.** The new law supplies
finite intact-material driving without a surface association. The selected ramp
produces boundary damage and, on the finer mesh, slight thickening of an already
damaged cap flank. It does not advance the five tracked seed-connected centerline
contours. No load, seed, material law, phase cap or acceptance tolerance was tuned
after seeing that result.

In plain language: the specimen started with the same rounded damaged seed and
zero mechanical histories. Nothing relaxed during the unloaded hold. Shearing
then activated bulk deformation and some new damage, mostly at the boundaries.
The material ahead of the old surface now receives a computed mechanical driving
force, but the central tip remains held by the fracture/irreversibility condition.
Stopping motion retains stress; the short final hold gives Maxwell relaxation,
without further phase change or accumulated slip. The small damage increment is
spatially sensitive, and endpoint work retains a measurable timestep error. This
is an implemented experimental model, not a demonstrated propagating-fault result.

## What is solved

The user-selected [brief](codex_bulk_phase_implementation_brief.md) and
[derivation](bulk_surface_cohesive_derivation.tex) authorize a **distinct**
`Fault constitutive mode = bulk cohesive`. The default remains `cohesive`.
[Formulation](BULK_COHESIVE_FORMULATION.md) records the backward-Euler return map,
mixed intact limit, actual coupled residual/tangent, physical obstacle and exact
discrete ledger before coupled testing. [Selection](BULK_COHESIVE_NATIVE_SELECTION.md)
records the interval, physical input and numerical/physical screens.

The primary uses a unit square, 16×32 fixed cells, Q2 velocity/Q1 pressure/Q1
phase, unlimited independent DGQ2 T (all nine modes), Gauss3 source projection
and Gauss6 mechanics/local histories. G=1 MPa, eta=100 MPa s, ell=.15625 m,
Gc=26.6666666667 J/m², cohesion=1 kPa, mu=.1 and prescribed p_N=10 kPa.
The latter supplies constant frictional resistance 1 kPa; it is distinct from
the dynamic incompressibility pressure/gauge. The continuum seed, original
endpoint (.5,.5) m and unloaded obstacle are sampled at each mesh's own vertices.
All-boundary affine simple shear has rate .018 s⁻¹ from .05 to .15 s, with
unloaded hold before and zero-motion hold afterward, ending at .16 s.
Maximum imposed engineering shear .0018 is 90% of homogeneous virgin elastic onset.

The stateless constitutive module evaluates the full embedded 3D tensor law,
physical tau=T−2GaS, q, driving, energy, dissipation and tangent together.
At h=0, a=0 but q=sgn(t0)(|t0|−tau_r)+, including nonzero driving.
No surface gate, inserted segment, maximum-H law or duplicated surface energy
is used. Positive phase trials recompute the local state. Only 0<=phi<1 is
implemented; neither .6 nor .99 is a constitutive cap.

M4 owns accepted DG T, local quadrature a/signed k/absolute L/q/rate and the
physical phase obstacle. Native M5 solves the nonsymmetric coupled residual
with the existing FE, independent-master constraint coordinates,
AffineConstraints and UMFPACK. Raw physical contributions are condensed once;
KKT uses C^T g and free actions use C^T H C, excluding artificial constrained
diagonals. Every trial starts from the same accepted owner. The existing outer
pending-publication callback makes one accepted swap; conditional native
checkpoint data preserve all new histories. Default archive layout is unchanged.
The new mode explicitly rejects unsupported RSF, advection, MPI, AMR,
moving/periodic geometry and the old surface-endpoint route.

## Evidence and failures

| Check / predeclared screen | Measured result |
|---|---|
| Material return/ledger, scaled 1e-12 | 3,600 independently minimized samples; a defect 1.73e-18, scaled q 4.55e-16, ledger 1.08e-14 |
| Material branch tangents/power | Five new unit cases, 240 assertions pass, including non-proportional plane-strain history and one-sided intact activation |
| Temporal order at fixed phase | Analytic Maxwell stress error 25.624→12.840 Pa for dt .01→.005 s; ratio 1.9957 |
| Raw-weight profile identities, 1e-10 | Maximum defect 5.56e-17; intact I=0 produces no normalized row |
| Initial physical phase relative L2 | 1.33455% primary; .339721% fine, independently integrated against the same continuum seed |
| Unloaded phase drift <=1e-12 | Exactly zero on both meshes; zero a/k/L |
| Native mechanics <=1e-9; KKT <=1e-7 | Across controls: max 9.60e-10 and 9.87e-8 |
| DG source discrepancy <=1e-12; tangent action <=1e-6 | Max 5.14e-15; action differences below 4e-9 (actual values in CSV) |
| Ahead-increment proxy >.001 | .00340927, maximum at right boundary (1,.5) m; **not a tip-advance test** |
| Five connected centerline fronts | Exactly unchanged for levels .001, .01, .1, .3, .5 on both meshes |
| dt/2 phase Linf <=.01 | 4.07e-7; matched-point full-tensor stress difference 6.25e-6 |
| Spatial stress moments <=10%; growth area <=10% | Stress .9334% passes; growth area **28.48% fails** |
| Gauss6/8 phase <=.001; storage/work <=.1% | Phase 1.86e-7; storage/work 3.11e-7 passes on declared scale |
| Exact discrete ledger <=1e-10 | Max 1.73e-14 J/m across controls; physical dissipation nonnegative |
| Physical endpoint-work remainder <=1% | **2.1321% primary; 1.0786% at dt/2: both fail**, approximately first order |
| Loaded rejection/retry/publication/restart | Pass; 48+16 accepted steps reproduce ledger and 344 files bitwise |

The global-ahead and tip-window increments are explicitly labeled proxies.
Primary tip-window change is zero. Fine-mesh change .00120853 occurs on an
**already damaged cap flank**, at (.71875,.3125) m: phi .00395137→.00515990.
Initially intact nodes in the declared tip window do not change. The finer
mesh does not establish a new connected tip. Its growth area is 5.57714e-5 m²
versus 3.98877e-5 m²; initial coarse/fine phase Linf difference .0293123 is
reported separately from evolution-increment difference .00133532. Changed
mesh/rule stress comparisons use actual coarse-cell weighted tensor moments,
not invented interpolation of local constitutive histories.

At .15 s primary tip-window Y reaches 39.978 Pa and weak phase residual
density remains positive (minimum 5.352 Pa), holding the obstacle. Near the
right boundary Y reaches 77.443 Pa and the active nodes reach the free phase
condition. Intact points have finite q up to 978.265 Pa and Y up to 61.248 Pa;
the old zero-driving-ahead defect is not present. This is the causal distinction
between lack of driving in P4a and insufficient driving here.

![Specimen, actual quadrature, prescribed seed and computed increment](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_cohesive/specimen.png)

*The seed and loading were imposed; phase was solved everywhere without an
original-surface admission gate. The cap is pre-existing damage. The changed
low contour at the right boundary is disconnected nucleation. Actual Gauss
locations are shown. Primary JxW ranges 1.43321e-5–1.06906e-4 m² and sums to
1 m²; those weights are distinct from point counts. The last panel shows the
fine-mesh cap-flank increment; cyan dashed/white contours are initial/final.*

![Physical stress, q, driving and the weak phase inequality](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_cohesive/driving.png)

*Computed physical stress differs from auxiliary T. The dotted line is the old
endpoint; material to its right has no original-surface association. Nodal weak
densities and the obstacle condition are distinguished from pointwise Y. This
does not test unrestricted orientation selection or general crack growth.*

## Energy, surface observation and lifecycle

At .16 s primary endpoint external work is 1.50744101 J/m; storage change is
1.10831453 J/m. Physical bulk/cohesive viscous losses are .00111779 and
.00000758304 J/m; friction is .36586102 J/m. The remaining .03214010 J/m
equals the separately recorded time-discretization terms minus phase/mechanical
residual work. Those residual works are −9.85e-12 and 8.90e-12 J/m; this closure
does not make the 2.13% physical remainder pass. At dt/2 the remainder is
.01608555 J/m. A **post hoc diagnostic**, trapezoidal integration of the same
computed affine-boundary reactions, gives work 1.47519173 / 1.47526161 J/m
and remainders −1.09e-4 / −3.91e-5 J/m. It identifies work-time quadrature as
the main discrepancy; it does not replace the original failed endpoint screen.

![Response and distinct storage, dissipation, discretization and residual work](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_cohesive/response.png)

*Loading reaction, phase change and connected contour locations are calculated
from actual accepted states. The exact algebraic ledger and the failed physical
work screen are both visible. The hold stops boundary motion, not retained stress.*

Raw positive normal-weight profile tests retain the residual rho. A deliberately
nonuniform profile has cohesive storage .28765641 J/m² = .27744249 surface
part + .01021392 residual; friction .57863850 W/m² = .01430007 surface part
+ .56433842 excess. In the actual primary final state the analogous domain
cohesive decomposition is .01551128 J/m = .00586670 + .00964458. Discarding
that residual would remove most cohesive storage. No such reduction is made.

The zero-time observer recomputes straight-column diagnostic moments and an
appended interval mask, leaving every persistent T/a/k/L/q/rate/phase byte
unchanged. It is a pure function of copied bulk samples and weights with no
access to the mechanical solution. The 24 positive-Z newly observed columns
inherit their computed moments. This is **not automatic M3 ridge reconstruction,
physical surface birth or native fault-coupling activation**; the new law has no
surface feedback or surface solver caches to invalidate.

The first loaded run failed at the hold predictor: 27 falsely predicted free
phase masters had negative directions at their exact lower obstacle (minimum
−.00072237). No positive unprojected Newton line was feasible. The smallest
correction is a feasible projected line search in independent master space,
then native constraint expansion. The residual, tangent, obstacle, physical
upper bound and tolerances are unchanged. A checkpoint replay reproduced the
pre-failure loaded state in five files bitwise. The original failure and direction
CSV are retained; the fresh repaired trajectory and all selected controls pass.

Loaded recovery invokes the actual native solve/pending outer acceptance
callback: no trial publication, rejected solution/history restoration,
deterministic retry, and one real accepted publication. This qualifies that
rejection path, **not automatic adaptive cutback**. Separate-process checkpoint
continuation reproduces all continued histories, solution, stress, phase forces,
observer moments and ledger exactly. During four hold steps, phase/k/L are
bitwise unchanged; T and a decay with beta, positive-h q with beta and energies
with beta² to roundoff. Native/default tests pass (136 cases, 165,676 assertions).
The separate old-surface P3 control reproduces 16 physical/history/weak-load/
constraint exports bitwise.

## Handoff and limits

Entry is `f1f9f29cf40c32d1ddcea000b724375b64853532` plus retained manual/P4a
work. Exact source/input/executable hashes, bounded commands, selected actual
exports, original failures and checkpoints are in
`benchmarks/reconstructed_fault/particle_free/evidence/bulk_cohesive/`;
[fixture README](../../../benchmarks/reconstructed_fault/particle_free/bulk_cohesive/README.md)
gives direct build/replay/analysis commands. No server job was run.

The new local histories are authoritative quadrature records on the same fixed
mesh/rule; off-point interpolation, layout conversion, transport and AMR are
unsupported. Diagnostic particle/H-property infrastructure remains for generic
phase initialization/checkpointing, with domains disabled and RK2 explicitly
frozen. New mechanical coefficients, residual, driving and history update read
no particle values. No particles are removed in this task.

All 34 protected BP3 artifacts and incoming manual AGENTS/P4a work are preserved.
P2's unresolved spatial/branch findings and `strict_reference_match_passed=false`
remain. The new law is not asserted equivalent to arbitrary old P2 trajectories.
There is no qualification of fully failed phi=1, global energetic stability,
adaptive cutback, new-mode hanging-node evolution, MPI, ridge/birth rules,
RSF, transport, AMR, moving geometry or general propagation.

Stop here. The next useful bounded task would diagnose boundary loading and
cap-flank spatial sensitivity under this same law before selecting any revised
physical experiment. Do not enlarge the load, continue propagation or tune
parameters to turn this negative result into a growth picture automatically.
