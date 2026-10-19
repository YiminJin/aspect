# Bounded stationary mature DG optimization

Baseline b2dfdebe43437d6028b06d4f9f862cc80a00cc71; no tracked subsequent changes.
Local untracked evidence/instructions are retained. No server/full-event run.

Selected scope: accepted homogeneous mature DGQ2, prescribed fixed physical
phase/mesh/geometry, stationary check points, Gauss6/source3, native work measure,
Helmholtz20 and unchanged native adaptive controls/solver tolerances. Reference
and optimized local trajectories use identical explicit timestep sequences.
A reference selector preserves the old algorithms without enabling endpoint phase.

For each existing coupling sample p with native work weight w_p and Q1 surface
values N_i, the existing physical bulk derivatives are
s_p(x)=2 eta_p S_p:D3(symgrad x_u) and
n_p(x)=x_p-2 eta_p Nnormal_p:D3(symgrad x_u).
Their existing RemotePointEvaluation average over incident cells is retained.
S_Gamma and N_Gamma integrate w_p*N_i*s_p and w_p*N_i*n_p.
The native filter is H=M+ell_f^2*K with its existing boundary traces.
W_mu[i,j]=sum_p w_p*mu_p*N_i(p)*N_j(p), including off-diagonals and MPI sums.
Thus Gx=S_Gamma*x-W_mu*(H^-1*(N_Gamma*x)). This is the exact discrete
factorization; mu is not commuted through H and no nodal multiplication or dense
filtered global map is introduced. K_V/active sets and C=A-B*K_V^-1*G are unchanged.
Physical pressure scaling and homogeneous constraints remain outside these maps.

Geometry/layout/phase caches must outlive accepted stress samples but invalidate
on mesh/fault/projection changes or restore. Accepted stress samples invalidate
on publication; roundoff operands and Maxwell coefficients update from the trial
velocity and dt. Fixed DG Gauss3 mass/inverse and inverse-defect factors may be
reused independently. A and bulk preconditioner reuse starts only within one
mechanical attempt, rebuilding at its first Newton iteration (including retries).
All residuals and friction/active-set factors continue to update.

Completed local verification (see [measured report](BP3_DG_OPTIMIZATION_REPORT.md)): small and one larger oblique native fixture; identical
frozen B/G/C actions, pressure scaling, hanging constraints, endpoint continuation,
varying friction; reference segment search versus indexed search with 5776 actual
BP3 nodes; source/roundoff checks; accepted stress/load/history comparisons;
rejection/cutback, once-only publication, 1/2/4-rank ownership/restart and retained
collective wall-stop regression. Timing runs are sequential. Peak RSS and setup,
application, assembly, solver, commit and diagnostic costs are reported separately.
Existing failures and strict_reference_match_passed=false remain in force.

## Dependency and ownership audit

| Stored data | Owner and invalidation |
|---|---|
| Conservative segment hierarchy | Manager; same fault/half-width geometry and normal-profile admission. Cleared with the Stokes association cache. Reference scan remains available and handles extreme/ill-scaled queries. |
| Mechanical point units, phase and QP associations | Material; cell/quadrature indexing, fixed phase and Stokes geometry generation. Real physical QPs are inverted with the original mapping, preserving the original point arithmetic. |
| Reconstructed mechanical QP stress | Material; weak identity of the accepted DG history. Publication invalidates values without discarding geometry. Contains current Z/I correction, never particle stress. |
| Fixed publication/export points and DG basis values | Material DG space; CellId and a short point list per cell, geometry generation. No accepted tensor values. Dense export layout does not evict the particle layout. |
| Native publication-gradient sample layout and RPE | Material; exact IDs/coordinates, matching-cell ownership, geometry generation. Gradients recomputed independently from the current native FE solution each commit. |
| Source mass inverse/inverse-defect factors | Material DG space; exact Gauss3 JxW and affine cell widths. All velocity operand sums, RHS products and their error bounds recomputed for each trial. |
| S_Gamma/N_Gamma | Surface system; physical sample layout, geometry generation, DoF layout and homogeneous eta_ve. Lookup reuse/build decisions are collective, including ranks with no coupling samples. |
| W_mu, K_V and active-set inverse | Surface linearization; refreshed with the actual friction law at each required Newton linearization. No friction factor is cached as geometry. |
| Sparse B | Stokes coupling; rebuilt per mechanical linearization from the native coefficients and current closed constraints. Applications use the sparse map. |
| Bulk A and native bulk preconditioner | Simulator; first Newton iteration of every attempt and changed dt, pressure scaling, closed constraints, cell/DoF layout. Only homogeneous Stokes direction constraints are eligible. No cross-step reuse. |
| Condensed full-solution work vectors | Condensed linearization; owned/ghosted native layouts, invalid when that linearization becomes stale. |

Existing accepted-DG restrictions enforce fixed mesh/geometry, homogeneous G and
unclipped temperature-independent eta, frozen phase and stationary diagnostic
particles. Native matrix viscosity is eta_ve and independent of Newton velocity,
phase, Z, state and friction. Loading/residuals and every slip-dependent term
still update. The line-search residual's previous prescribed-boundary rebuild
must also be disabled under the same homogeneous-Stokes-direction eligibility; otherwise
it overwrites A between Newton iterations. The initial optimization failure
exposed this missing eligibility extension. The subsequent trace found that the
all-fields constraint test also counted nonzero temperature data: residual assembly
changed ||A_uu||_F from 6.21036e18 to 78. The corrected test checks precisely the
Stokes rows; non-Stokes constraints and their data remain intact. Both failed
attempts and the owned/ghosted-vector diagnostic harness error are retained.

The indexed search only excludes conservative bounding boxes. It then evaluates
the original xi, interpolated width and signed-distance expressions on sorted
(fault,segment) candidates. Strict distance comparison preserves knot ownership;
all admitted faults still trigger the original overlap rejection. Outward
padding accounts for coordinate/projection arithmetic; excessively large queries
and unusably scaled geometry fall back to the original scan. Manager-owned
endpoint continuation executes after the same normal-profile result.

The G construction retains the RPE average over all matching incident cells:
coefficient routing divides by the original multiplicity before assembling each
native cell's columns. Pressure columns use physical pressure; the condensed
wrapper applies its existing pressure scaling once and distributes homogeneous
constraints before G. B expands test constraints with the same closed relations
as native local-to-global assembly. Both references remain sampled native actions.
The complete C diagnostic applies the actual unrestricted or restricted K_V solve
and checks against those independent B/G references. The new frozen-action diagnostic tolerance (2e-11 relative) is applied against ||Ax||+||B K_V^-1 Gx||, rather than the
possibly cancelling norm ||Cx||. This is a diagnostic guard for reassociated sparse quadrature sums, well below
the native nonlinear targets; the measured discrepancies must also be reported.
It does not alter convergence,
publication or physical acceptance tolerances.

The mature publication candidate copies the accepted phase block verbatim.
Only that route evaluates its candidate using the same fixed-phase point layout;
evolving endpoint calls still pass their actual trial phase vector. Native
publication gradients remain independently sampled from the accepted FE velocity,
with matching-cell traces. Neither the particle population nor the coverage of
this check is reduced. The original operand-derived allowance is unchanged.

The native BP3 work observer now obtains the same accepted reconstruction by
cell at its existing Stokes QPs; it keeps its own weak integration and checks.
The observer's fixed-point gradient-operand sums are also evaluated by cell with
the original dof summation order. Dense Gauss3 outputs and fixed particles retain
separate geometry layouts in the short cell-indexed point list. Unprepared
arbitrary diagnostic points retain the scalar reference evaluation/fallback.
All stress values, Z and I in these calls are current; only geometry/basis/phase
are reused. Output scheduling, collective wall stopping and checkpoint payloads
are untouched.
