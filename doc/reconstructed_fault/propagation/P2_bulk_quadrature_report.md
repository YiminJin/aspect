# P2 coupled quadrature: accuracy bought by Q3, cost and mesh comparison

Updated 2026-10-09. **After resolving quadrature error, Q3 buys about 2.8x
smaller full-phase error,8.7x smaller traction error and 7.2x smaller transverse
stress RMS than coarse Q2, at 2.11x runtime and 4.4% more peak memory.** These
are matched height-one runs to 0.625 s. The normal-refined Q2 comparator costs
about the same as Q3 and gives better phase accuracy, but worse stress accuracy.

For the stated provisional **phase/traction/slip** targets, recommend
16x32 Q2/Q1, six normal points and the existing64-segment fault: full-phase
error0.00648, traction error5.05 Pa and slip error0.0452% over common times,
at 156.7 s and 339 MiB peak memory. This is the only native candidate that meets
all three targets. Coarse Q3 retains phase/initialization error0.01374.
**Neither recommendation nor solver passes qualify bulk stress/energy, time
convergence or P3.** Q3 still has404 Pa stress RMS about its transverse mean;
its bulk storage differs from the rich 1D solution by 11.8%. The refined Q2
stress RMS is908 Pa and its storage error is55.9%.

The three-point results, corrected plane-strain law, constitutive options,
physical parameters and failed historical states are preserved. No material,
loading, phase bound, normalization projection, history law or kinetics changed.

## Settings and comparison definition

| Setting | Value / units |
|---|---|
| Domain / geometry | [0,1]x[0,1] m; one straight fault y=0.5 m, fixed endpoints x=0/1 |
| Bulk spaces | 16x16 Q2/Q1 and Q3/Q2 velocity/pressure; refined16x32 Q2/Q1; Q1 phase |
| Cell sizes | Coarse dx=dy=.0625 m; refined dx=.0625, dy=.03125 m |
| Surface / particles |64 Q1 segments,.015625 m spacing;3x3 particles/cell;2304 or4608 frozen particles |
| Associations | Entire normal cross-section, fixed width.5 m; no runtime admission |
| G / eta / lambda |1e6 Pa /1e8 Pa s /100 s, identical homogeneous constituents |
| Creep-viscosity bounds |1e6–1e10 Pa s |
| c / Gc / ell / p |1000 Pa /26.6666666667 J/m² /.15625 m /1 |
| Native m / Hc / c0 |128 /.5 Pa /8/3 |
| Friction / prescribed pressure |mu=.1 constant; p_ad=10000 Pa, manager background0; no evolving RSF state |
| Radiation / gravity |0 /0; static293 K |
| Initial phase/history |Nominal core.6, native H initialization, native CPDI relaxation; accepted lower bound protects this pre-existing damage |
| Initial mechanics |Unloaded accepted FE d_R=0, surface Z=0, bulk Maxwell stress0 |
| Loading |Top/bottom ux=.256 max(0,t-.5dt)(y-.5) m/s, uy=0; interval mean of the unchanged smooth ramp |
| Sides / pressure |uy=0, ux natural; pressure normalization=no; friction uses complete prescribed adiabatic pressure |
| Time |dt=.015625 s,40 steps toT=.625 s |
| Quadrature |Six Gauss normal points/cell; split tangential panels with 3 points(Q2) or4(Q3); same rule in all coupled bulk terms |
| Retained integration |Consistent analytic transverse normalization/projection/adjoint; sharp-surface quadrature; native CPDI fracture |

G=1e6 Pa is a dimensional synthetic verification value. V_interval is the
solved interval rate used by endpoint mechanics, not an exact instantaneous
velocity sample. Mesh/geometry, particle positions and associations are fixed
within each fresh trajectory. Transport, birth, conversion and restart remain
outside the accepted-FE option. This compatible symmetry control tests widening
of a prescribed band; it does not test propagation or general2D localization.

Targets, recorded before native runs: maximum common-time full-phase error
<=.01, maximum cohesive-traction error<=10 Pa(1% of cohesion), and maximum
slip error<=.5% of the reference final slip. Stress and energy errors are
reported separately; these are **not** targets for complete spatial qualification.
The reference is the retained 256-cell rich 1D, exact-bulk path, sampled at the
same times and loading levels. Its independent fine initialization is retained.
Native vs matching discrete1D controls isolates implementation from spatial error.

## Rule selection and coupled operator audit

The inexpensive prototype uses the actual Q2 or Gauss-Lobatto Q3 velocity
basis, Q1 phase, native sampled-H initialization and endpoint Maxwell/slip
system. Four, six and twelve normal points were tested; independent24/48-point
evaluation holds the FE anchor, phase and Z fixed. Constant-along-fault I equals
the analytic transverse integral under the retained consistent projection.
All40 incoming anchors were checked, including onset and strong growth.

| Maximum over both 16-cell degrees / path |4 points |6 points |12 points |
|---|---:|---:|---:|
| Bulk energy uncertainty |2.948e-03 J/m |9.29e-7 J/m |3.11e-14 J/m |
| Phase gradient difference, scaled by Gc/(c0 ell) |9.14e-4 |3.80e-7 |7.91e-14 |
| Phase Hessian relative difference |See audits JSON |3.05e-8 |3.15e-15 |
| Independent bulk-force defect |3.95 Pa |5.38e-4 Pa |8.87e-10 Pa |
| Six vs twelve full-phase change |— |4.01e-8 |— |

Selection thresholds were1e-5 for normalized phase gradient,1e-4 for Hessian
relative error,1e-5 J/m for bulk energy, .01 Pa for force, and 1e-5 for common-time
phase change. Four points fail the force/gradient thresholds. Six is the lowest
passing tested rule; its uncertainty is orders below remaining spatial error.
The24/48 reference differences are below 6.4e-14 for energy/normalized gradient.
Explicit energy/gradient finite differences and same-weight coupling/transpose
checks are retained in every prototype audit. No quadrature tolerance is equated
with spatial accuracy. [Selection and raw audits](../../../benchmarks/reconstructed_fault/propagation/quadrature_evidence/selection.json).

The implementation already routed these consumers through one material-owned
factory; the updated factory therefore changes the entire coupled calculation:

| Consumer | Audited path |
|---|---|
| Stokes residual/Jacobian and preconditioning |simulator/assembly.cc, both shared-rule entry points; plane-strain Newton trace term retained |
| Bulk-slip force / transpose / G |ReconstructedFaultStokes and surface_system_bulk_work.cc, same points, weights, FE gradients and cached associations |
| Constraint/lift / sparse coupling sizing |solver/reconstructed_fault_stokes.cc, shared rule |
| Bulk phase energy/gradient/Hessian |fe_strain.cc, cached bulk points/shapes and identical FEValues rule; normalization adjoint unchanged |
| Energy/work/loss diagnostics |native_fe_strain.cc, shared spatial rule and unchanged12-point time rule |

Tangential panels split at actual bulk/fault cell intersections. At least
velocity degree+1 points integrate the polynomial Stokes products; Q3's old
three-point tangential rule becomes four. This is explicit, not a hidden switch.
The current fields are tangentially uniform. General nonuniform tangential
quadrature accuracy remains a separate problem. Face and sharp-surface rules
are unchanged. The fixed selected setting builds all associations/caches from
fresh initialization; no runtime quadrature change or checkpoint conversion is
introduced. Cached QP ordering/weights remain those checked by the existing
manager. Only raising postprocessing would not have accomplished this change.

## Native physical accuracy, initialization and cost

[Phase/chi/stress profiles, phase-time and traction-slip curves](../../../benchmarks/reconstructed_fault/propagation/quadrature_evidence/accuracy-cost.png),
with dense CSV and exportable PDF siblings. Center and both fault endpoints
agree to arithmetic precision; reported maxima remain at y=.5 and their x
location is degenerate in this uniform control.

| Result to.625 s |16x16 Q2 |16x16 Q3 |16x32 Q2 |
|---|---:|---:|---:|
| Relaxed initial maximum |.613176697 |.613176697 |.603548838 |
| Initial particle H maximum |28010.58 Pa |28010.58 Pa |29167.99 Pa |
| Final phase maximum |.655706200 |.682442297 |.673958345 |
| Largest accepted phase increment |.0158470 |.0167496 |.0170325 |
| First resolved full-field growth |.390625 s |.515625 s |.515625 s |
| First maximum/core growth |.578125 s |.546875 s |.546875 s |
| phi / chi FWHM |.27885/.11925 m |.27775/.10865 m |.27585/.11605 m |
| Final q / interval V |305.229 Pa /.158216 m/s |260.845 Pa /.158813 m/s |260.105 Pa /.158763 m/s |
| Cumulative slip |.0485118 m |.0487293 m |.0487164 m |
| Full-phase error at common times |.0382306 |.0137441 |.00647828 |
| Initial full-field error alone |.0131196 |.0131196 |.00431042 |
| Phase-change error after subtracting each seed |.0305232 |.00430883 |.00319273 |
| q error / slip error |50.172 Pa /.4650% |5.787 Pa /.01880% |5.047 Pa /.04520% |
| Accurate transverse stress RMS |2919.59 Pa |404.41 Pa |907.77 Pa |
| Accurate bulk storage |5.35910 J/m |.880887 J/m |1.22746 J/m |

Growth onset means max(phi-phi_initial)>1e-5, not growth of the single phase
maximum: coarse Q2 grows in its tails long before its core maximum changes.
Phase consumes the incoming mechanical anchor from the previous step; all cases
share dt and that one-step lag. Comparisons use identical accepted physical times,
not different last-accepted times. The refined particle H maximum changes because
the same prescribed field is sampled more finely, not because material parameters
or the history definition changed. The finer initial obstacle changes the ensuing
response; subtracting each initial field separates this from total spatial error.

All three native paths reproduce their matching1D discretizations: full phase
<=1.32e-8, phase-change error<=5.91e-9, q<=1.56e-5 Pa, slip<=1.54e-11 m.
The cheap Q2 normal-mesh study at 16/32/64 cells gives common-time phase errors
about.0382/.00648/.00135;64-cell q error is.464 Pa. Only the warranted32-cell
normal refinement was run natively; no broad mesh matrix was launched.

| Measured per native process |Q2 |Q3 |Refined Q2 |
|---|---:|---:|---:|
| Wall / user CPU |75.89/71.67 s |159.90/155.06 s |156.71/151.43 s |
| Peak RSS |265.1 MiB |276.7 MiB |339.4 MiB |
| Velocity DoFs |2178 |4802 |4290 |
| Stokes matrix / RHS assembly |7.09/12.2 s |19.2/21.0 s |14.4/24.9 s |
| Stokes preconditioning / condensed solves |3.64/28.1 s |10.9/77.4 s |7.10/57.6 s |
| Phase bounded solve / phase linear solve |3.77/.450 s |2.84/.279 s |13.3/1.64 s |
| FE phase energy evaluations, inclusive |2.89 s(368) |2.88 s(314) |6.28 s(332) |
| Phase iterations, including initialization |60 |34 |42 |
| Mechanical Krylov iterations / nonlinear evaluations |3020/123 |3872/123 |2859/123 |
| Postprocessing |4.63 s |6.96 s |9.66 s |

Single-threaded processes ran sequentially with no simultaneous native/build
jobs. Peak RSS/CPU comes from POSIX wait4 for each child, not cumulative process
statistics. Builds are excluded; startup, logs and diagnostic costs are included.
ASPECT component timers overlap and must not be summed. These are single-run
measurements, not a statistical performance claim. Q3 and refined Q2 differ by
only 2% wall time here. Historical three-point timings involved concurrent runs
and are not used as an uncontended cost ratio. Prototype costs are below.6 s per
case and do not predict native 2D solver cost.

## Remaining errors, stability and energy accounting

Independent12/24-point evaluation of each frozen native final field agrees in
bulk storage to arithmetic precision. Six-point storage errors are4.26e-7,
9.28e-7 and 5.61e-10 J/m; independent bulk-force defects are2.02e-5,5.38e-4
and 4.59e-8 Pa. Current normalization/direct-source errors stay below 1.67e-15,
including side endpoints and initialization. Spatial stress error now dominates
quadrature error: the rich 1D shear stress is1255.06 Pa and storage.787585 J/m.
The sampled Q3 field still ranges from -1401 to 3267 Pa between quadrature points.

At unchanged40-step loading, moving from three to six points shifts final phase
by-.00172995(Q2)/-.000941662(Q3) and q by+3.541/+1.892 Pa. Initial phase files
are byte-identical. Correcting the rule changes the discrete energy and subsequent
history; it does not merely reinterpret the old sampled fields.

New phase-curvature checks use **current trial phase and incoming R/Z from
steps31/39**, with an immutable previous-phase obstacle. Native nonuniform phase
finite differences give best energy/gradient errors<=5.49e-9 and Hessian-direction
errors<=5.48e-10. These include the actual normalization projection and adjoint.

| Selected time / free Hessian minimum |Q2 |Q3 |Refined Q2 |
|---|---:|---:|---:|
| .5 s |9.14881 |No free nodes |No free nodes |
| .625 s |2.48066 |2.26162 |1.12397 |

No negative sampled free modes were found. Feasible +/-1e-4 mode perturbations
raise energy by 4.5744e-8,1.2403e-8,1.1308e-8 or5.6198e-9 J/m, consistent
with the quadratic predictions and separately retained 32eps allowances
1.50e-13–2.03e-13 J/m. Eigenvalues on different meshes are not a mesh-invariant
stability norm. These checks do not establish full critical-cone/path/global
stability; earlier affine negative modes remain preserved. No old curvature
qualification is inherited automatically.

| Cumulative distinct quantities, J/m |Q2 |Q3 |Refined Q2 |
|---|---:|---:|---:|
| Phase-relaxation release D_relax |.382235 |.180093 |.178718 |
| Fracture expenditure |2.161578 |3.020145 |3.008647 |
| Bulk / cohesive profile storage change |-.044885/-2.498928 |+.022286/-3.222524 |+.037140/-3.224505 |
| Sum absolute boundary-lifting work discrepancies |.444848 |.202376 |.200388 |
| Sum absolute internal-work discrepancies |3.074883 |.292542 |.358514 |
| Bulk / cohesive viscous loss |.019404/.023569 |.007548/.022568 |.008373/.022525 |
| Frictional loss |48.511818 |48.729303 |48.716433 |

Profile storage change plus fracture expenditure plus D_relax closes the
instantaneous phase segment; none is omitted or merged with the Maxwell work
error. The native mechanical diagnostic uses work of the affine boundary lifting;
the older1D diagnostic used actual bulk-rate work. They are different when the
endpoint-balanced field is not balanced throughout the segment. Both measures
are now reconstructed on the same fields/time quadrature. Matching-prototype
boundary discrepancies agree with native values within3.6e-8 J/m; internal
values within3.8e-7 J/m. Their absolute cumulative magnitudes remain separate so
cancellation cannot certify accuracy. No compensating driver is added.

Bounds, immutable-lower-bound irreversibility and publication checks pass;
measured healing and saved FE coefficient recurrence error are0. Original phase
relative tolerance1e-12, linear tolerance2e-7 and positive arithmetic allowances
are unchanged. Actual KKT targets span1.81e-13–1.01e-10 for coarse cases,
3.18e-13–7.67e-11 for refinement. Mechanical tolerance remains1e-8; terminal
bulk/fault residuals are below 3.78e-14/1.48e-14. Every native target passes.
This establishes numerical solution of the chosen sampled problem, separately
from quadrature accuracy, spatial accuracy and temporal/energetic qualification.

## Source changes and reproducibility

| Production file/function | Purpose |
|---|---|
| phase_field_fault.h |One numerical rule member, default 3; clarify authoritative 3D FE strain |
| phase_field_fault.cc :: declare_parameters / parse_parameters |Explicit `FE bulk normal quadrature points`; no new physical parameter; ignored by particle-owned quadrature |
| fe_strain.cc :: mechanical_quadrature |Selected normal rule and degree-sufficient tangential panels, shared by all audited bulk consumers |
| fe_strain.cc :: evaluate_fe_phase_energy |Timing scope only; formulas/adjoint unchanged apart from consuming the selected shared rule |
| phase_field.cc :: solve_bound_constrained_phase / solve_phase_field_system |Timing scopes only; solver/bounds/tolerances unchanged |

The assembly/coupling callers require no duplicate formula correction because
they already consume the shared factory. No generic projector or tensor helper
changes. Benchmark changes: generalize localization_1d.Reference to Q3 while
retaining Q2 defaults; quadrature_1d.py tests rules/refinement; native_fe_strain.cc
adds nontrivial trial derivative checks at the two selected incoming states;
run_quadrature.py records isolated process cost; assess_quadrature.py reconstructs
fields/work/accuracy; check_quadrature.py verifies current and preserved evidence.
The mathematical note/specification now distinguish the helper's signed
`II(tau)=-<tau,tau>3/2` from positive `J2=<tau,tau>3/2`; helper code is unchanged.

Exact source/input/build identities and task-only source/TeX diffs are in
[quadrature_evidence](../../../benchmarks/reconstructed_fault/propagation/quadrature_evidence/selection.json).
Baseline remains fault-propagation /de7beaa65846c7372711f7922abf1c5d9a745ee6.
Nineteen incoming snapshots and historical outputs are preserved; manual
pf_rsf.tex edits are retained exactly. GCC 12.4 Release Unity/deal.II9.6.2/
Trilinos 14.2, unchanged compiler configuration. Core SHA
0963b3aee8fac8fb35d5f15926549b2237d426544f03a359687a64c394d65a18;
native observer SHA940094b29c2843f66dc9d41829444cf4023e93b2a163609499c73e39eb4bad80.
Default mature regression is rebuilt for the changed header and its two CSVs
remain byte-identical. Documentation compiles in two LaTeX passes.

```sh
cmake --build build-propagation-integration --target aspect -j2
cmake --build build-propagation-integration/plugin --target native_fe_strain -j2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/propagation/quadrature_1d.py --cells 16 --degree 3 --quadrature 6
python3 benchmarks/reconstructed_fault/propagation/run_quadrature.py
MPLCONFIGDIR=/tmp/quadrature-matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/propagation/assess_quadrature.py
python3 benchmarks/reconstructed_fault/propagation/check_quadrature.py
```

Fresh-run drivers deliberately reject existing output directories; do not overwrite
retained evidence to reproduce them. Input variants for other prototype rules,
Q2 refinements and native cases are retained. No additional native time family,
material optimization, MPI/transport/restart campaign, branch rule or P3 was run.
The next spatial question is bulk stress/storage resolution; the cheap 64-cell Q2
result is promising for phase/traction but is not native qualification. The
restricted phase/traction/slip recommendation must not be relabeled a complete
damage-feedback or energetic baseline.
