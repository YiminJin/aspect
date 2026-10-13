# Reusable native mechanical surface Schur solve

This bounded test preserves the reduced-Newton endpoint reference and the
physical model. It changes only the linear inversion used by mechanical Newton
and phase sensitivity solves. The direct option is experimental, serial and
default off.

The cost gate passes: the current paired eight-RHS batch costs 10.51 s with
FGMRES and 6.29 s with direct inversion, including the measured setup. One
complete growth entry and the retained saddle both pass the original criteria.
Growth costs 374 s versus the retained 664 s reference with identical outer
counts; saddle costs 616 s versus 1656 s and also uses fewer actions. Keep both
linear solvers selectable; retain FGMRES as the default. No trajectory is run.

## Fixed verification configuration

| Quantity | Setting |
|---|---|
| Domain/fault | Unit box; straight through-going fault y=0.5 m; fixed full-normal associations |
| Bulk mesh/elements | 16×512; Q2 velocity, Q1 pressure and phase |
| Integration/surface | Six normal bulk points; existing common intersection and CPDI fracture rules; 64 surface segments |
| Normalization | Existing consistent projection, including actual first/second derivatives and adjoints |
| Particles | 73,728 fixed particles; no transport or replenishment |
| G, eta, lambda | 10^6 Pa synthetic verification modulus; 10^8 Pa s; 100 s |
| c, Gc, ell, p | 1000 Pa; 26.6666666667 J/m²; 0.15625 m; 1 |
| Friction/pressure | Fixed mu=0.1; prescribed background pressure 10,000 Pa; zero radiation damping |
| Seed/history | Shared relaxed nominal-core-0.6 seed; accepted FE d_R and surface Z; H+ diagnostic |
| Phase set | Immutable previous physical phase lower bound; upper bound 1 |
| Growth entry | core-6 at t=0.53125 s, dt=1/1024 s, original immutable incoming state |
| Loading | Interval mean of ux=0.256 t(y−0.5), uy=0, existing compatible lateral conditions |
| Accuracy | Original Stokes linear $10^{-7}$ and nonlinear $10^{-8}$; phase linear $2\times10^{-7}$ and nonlinear $10^{-12}$; original energy allowance and KKT checks |

The initialization, histories, governing energy/constitutive equations, full
native degrees of freedom and outer Newton algorithm remain unchanged. No
full trajectory is launched. The existing quasi-Newton option stays off.
Set `Solver parameters/Phase field solver parameters/Endpoint direct mechanics`
to `false` for FGMRES or `true` for the serial Schur solver; both use the same
outer Newton and final checks.

## Actual block solve

The native uncondensed Jacobian has the actual signs
$$
 \begin{pmatrix}A&-B\\G&-K\end{pmatrix}
 \binom{\Delta x}{\Delta V}=\binom{f}{-r},
$$
with solver-scaled pressure and homogeneous constraints. For current free
surface indices $F$, form $W=A^{-1}B_F$ and $S=K_{FF}-G_FW$. For the existing
condensed load $b=f+B_FK_{FF}^{-1}r_F$, solve
$$
 x_0=A^{-1}b,\qquad z=S^{-1}G_Fx_0,\qquad\Delta x=x_0+Wz.
$$
Recover $\Delta V$ through the existing restricted $K_{FF}$ inverse, keeping active
increments exactly zero. B and G are evaluated independently; no transpose
assumption or change of quadrature/pressure convention is introduced.

The installed Amesos KLU solver factors the sparse combined velocity/pressure
block; LAPACK LU factors only the at-most-65-dimensional surface matrix. The
initial implementation explicitly rejects more than one rank. It leaves the
verified MPI FGMRES path intact. A two-sided verified pressure nullspace alone
permits an algebraic pin; the solution is returned to the original pressure
complement. Open/traction boundaries receive no pin. Physical pressure
normalization remains separate.

Bulk factor lifetime follows the existing homogeneous operator rebuilds, at
the first trial of a physical step. Phase-dependent B/G/K are reassembled at
each mechanical linearization. A surface factor belongs to one view and free
set; existing generation checks reject superseded views and an explicit check
rejects changed active sets or bulk factors. Diagnostic clock restoration
invalidates the bulk cache. These caches contain numerical operators, not
physical histories.

## Bounded measurement

At the retained growing field, the harness solves a batch of eight right-hand
sides: normal, translation-like and tangential phase forcing plus a known full
mechanical forcing, repeated once. Every solve starts from zero. A second batch
holds one endpoint surface coefficient active as an explicitly artificial
linear-elimination diagnostic. All native u/p/V modes remain present. The
phase forcing/gradient uses the native consistent-normalization derivatives.

FGMRES uses the original full-residual correction and phase-action accuracy
rule. Direct setup is charged to the first call, including all surface columns
and sparse/small factorizations. The existing velocity-factor setup retained
in both current production paths is also included on both sides. Common native
assembly/preparation and total process runtime are reported separately. Peak
RSS includes shared pre-existing FE/particle/cache memory. A clear cost gate
requires at least 20% setup-inclusive batch improvement and the original
2×10^-7 phase-action comparison; this is a decision rule, not a looser
physical acceptance criterion.

## Results and separate cost metrics

| Measured component | Time / size | Meaning |
|---|---:|---|
| Bulk setup including KLU factor | 2.291 s | Sparse combined-matrix construction and factor; **pure factorization-only time unavailable** |
| Free-surface setup including small LU | 2.153 s | 65 native B columns, bulk inverse/G actions, SVD diagnostic and LU; factor-only split unavailable |
| Active-endpoint surface setup | 2.122 s | 64 free nodes; bulk factor reused |
| Repeated direct condensed solve | median 0.03429 s; range .03118–.03609 s | 14 reused calls; includes original fresh residual checks; excludes setup and phase pullback |
| Peak process RSS, paired batch | 1789.6 MiB | One rank, includes initialization/common FE/particle caches |
| Retained RSS increment at first direct setup | 102.2 MiB | 1687.1→1789.4 MiB; later FGMRES rows retain these factors too |
| Full bulk / surface dimensions | 76,371 / 65 | 67,650 velocities + 8721 pressures; surface diagnostic reduces to 64; actual initialization also exercises all-active size 0 |
| Surface conditioning | 13.784 /13.643 | SVD ratio at 65/64 free nodes; no singular-value modification |

The separately requested **pure factorization time was not instrumented**. The
setup figures above are inclusive and must not be presented as pure factorization.
Standalone growth/saddle peak RSS was not captured. The optional external time
utility was unavailable before a test launched. No completed test was rerun to
fill these gaps. The existing timers and batch memory observations are retained
without estimates.

| Eight-RHS batch | FGMRES | Direct |
|---|---:|---:|
| Physical free surface, inclusive first direct setup and phase diagnostics |10.097 s /14 linear calls /254 Krylov iterations |5.882 s /8 calls /0 Krylov |
| Include common logged velocity factors on both sides |10.508 s |6.292 s |
| Artificial active endpoint, bulk factor already present |19.132 s /16 calls /484 iterations |3.520 s /8 calls |

The 20% cost gate passes by a clear margin (1.67× setup-inclusive physical-batch
speedup). Common native assembly/preparation is outside this fixed-operator
batch; full-entry timings below include it. Total batch process time is 115.44 s.
One executable/observer, one bound core and one thread are used; timing tests
are sequential. Pure triangular-application time is also not isolated: the
repeated-solve timer includes the authoritative fresh check.

The actual uncondensed direct residual reaches at most 6.14e-8 in native bulk
weak-row units and 1.14e-13 on free surface rows across these RHS. Each condensed
solve passes its **original RHS-scaled target**. Full mechanical forcing differs
from configured-accuracy FGMRES by up to 1.19e-7 m/s in the surface rate; its
FGMRES residual is about 1e-5 while direct is about 1e-13. Resolved phase
sensitivities agree relatively within 3.13e-10 (target 2e-7). Normal and
translation-like collapsed elastic actions agree with the matching discrete
prototype to 3.17e-9 and 6.93e-9 J/m. Tangential forcing retains the full native
u/p/V system and agrees with FGMRES. This fixture has open lateral normal-flow
conditions, so its verified pressure nullspace is zero and **no pressure pin**
is applied. The small algebraic test independently verifies the nullspace
quotient, prescribed-pressure elimination and a second RHS; a new closed-domain
native trajectory is not claimed.

| Complete retained entry | Growth | Saddle |
|---|---:|---:|
| Native / retained FGMRES entry wall time |374.11 /664.10 s |615.71 /1655.61 s |
| Whole process including native initialization |423.45 s |661.08 s |
| Tangent time, direct /FGMRES |104.16 /384.29 s |231.84 /1207.05 s |
| Mechanical-evaluation time, direct /FGMRES |216.71 /225.06 s |314.46 /367.69 s |
| Tangent actions, direct /FGMRES |135 /135 |316 /363 |
| Phase iterations /line searches, direct |7 /9 (same reference) |8 /8 (reference 9 /9) |
| Original raw KKT target, J/m |2.685e-11 |3.481e-11 |
| Final raw KKT, J/m |2.81e-13 |2.88e-13 |
| Healing /minimum I |0 /111.133 m |0 /149.512 m |
| Maximum accepted phase increment |.00157378 |.00618032 (noncommitting diagnostic candidate) |
| Phase difference from FGMRES |1.05e-11 |1.21e-12 |
| Maximum off-quadrature stress difference |1.68e-8 Pa |1.80e-8 Pa |
| Maximum q /slip differences |6.41e-11 Pa /3.52e-15 m |5.92e-10 Pa /2.85e-14 m |
| Largest separate storage difference |1.21e-13 J/m |5.93e-12 J/m |

Growth commits the complete state once and verifies its original publication
rules. The saddle begins at mass-normalized translation curvature −.280202 Pa
and ends at +15.992244 Pa, on the same positive-centroid branch
(.500126224153 m) as the reference. Its histories remain unchanged and are
restored after the fixed-entry diagnostic. No reflection or recentering is
used in these comparisons. All supplied incoming lower fields remain identical.
These are retained-entry replays, not fresh full trajectories.

Complete-entry FGMRES timings are reused from the verified, PMPI-instrumented
reference. They are not a newly controlled wall-time pair. The current paired
batch establishes the linear cost gain; identical growth outer counts support
the observed 1.78× entry improvement. Saddle's 2.69× improvement includes fewer
iterations/actions; average tangent time also falls from 3.33 to .734 s. Do not
attribute the entire saddle improvement to linear inversion.

## Factor reuse and remaining cost

| Scope (initialization excluded for entries) | Bulk factors | Surface factors | Direct RHS solves | Reused solve median |
|---|---:|---:|---:|---:|
| Both fixed-state batches |1 |2 (65 then 64) |16 |.03429 s |
| Complete growth entry |1 |18 (65) |282 |.03405 s (264 reused) |
| Saddle diagnostic |1 |25 (65) |649 |.03432 s (624 reused) |

A physical growth entry uses the same bulk factor for 18 surface setups, 1170
response columns and 282 condensed RHS (1452 total bulk factor applications).
Within each valid view, its surface factor is reused for the sensitivity batch.
Every new phase/mechanical linearization refreshes B/G/K and constructs a new
surface factor; bulk reuse follows only the unchanged homogeneous operator.
The saddle analog has 1625 response columns plus649 RHS. The initial zero-loading
solve has its own timestep-zero bulk factor and all-active controls; those are
not counted as physical-entry reuse. This conservative policy may refactor
identical surface operators in a new view; no further reuse optimization is
implemented or claimed.

Growth's sampled reused full tangent is about .7 s although its condensed solve
is about .034 s. Native forcing, energy and normalization derivatives plus
mechanical preparation/assembly dominate the remaining time. The linear solver
is useful for these restricted serial entries; it is not evidence that a
full native trajectory fits an overnight budget. Keep the option default off
and preserve the existing MPI FGMRES path. Direct MPI, full fresh evolution,
transport/restart and unrestricted native stability are not qualified here.

## Source changes in this task

| File/function | Purpose |
|---|---|
| solver/reconstructed_fault_surface_schur.h: EndpointBulkDirect, EndpointSurfaceSchur | Native sparse bulk inverse, verified gauge handling, exact free-surface Schur and reusable response columns |
| reconstructed_fault_condensed_system.h/.cc: Linearization::solve_surface_schur | Cache one exact current free-set factor with existing generation checks |
| reconstructed_fault_stokes.cc: solve_reconstructed_fault_condensed_system / solve_reconstructed_fault_stokes | Default-off inverse dispatch, unchanged fresh residual/compatibility checks, bulk invalidation on rebuild |
| phase_field.h; simulator/phase_field.cc: SolverParameters, declare/parse | Default-false Endpoint direct mechanics numerical setting |
| simulator.h | Fixed-step bulk factor reference; unchanged five linear-solve scales made complete for test access |
| tests/phase_field_test_access.h | Scoped mechanical/linear access and diagnostic cache invalidation |
| unit_tests/reconstructed_fault_surface_schur.cc | Small pressure-quotient, prescribed-pressure and second-RHS checks |
| native_endpoint_verification.cc | Noncommitting native batch, uncondensed residuals, complete response/action and memory comparison |
| run_direct_mechanical.py; assess_direct_mechanical.py | Sequential bounded screens, explicit cost gate, independent prototype comparison and provenance |
| specification, current_design, PLAN/STATUS, CURRENT_STATUS, AGENTS, README | Exact inversion algorithm, bounded scope, results and limitations |

These changes are linear-solver/diagnostic changes. The outer phase Newton,
energy, Hessian, CPDI, normalization projection, constitutive law, physical
phase obstacle and accepted-history publication are not changed. Previous
manual edits, failed evidence and reusable MPI work are retained.

## Evidence and reproducibility

Existing worktree/branch fault-propagation; baseline 33c0adba59132890b386f3e984b3e1650feffe44
with all incoming edits retained. Compiler GCC 12.4; deal.II 9.6.2, Trilinos 14.2,
p4est 2.8.7. Tested executable e78821a5b12b5070c1e0f7f626ed631d8f030a50ed23bc2a895f4b1713214898;
observer f5b7567699a85d305a1b9eb30824fe7d5ae75a58d3596602015e1402cbbbcc57.
The prior Newton executable is preserved as aspect-newton-before-schur-release
(SHA3257574451ef…); serial/MPI reference inputs and evidence remain intact.
The outer Newton source and the three material history/FE/constitutive source
files are byte-identical to the incoming task baseline.

Build/test commands and results: cmake --build build-propagation-integration
--target aspect -j2; corresponding plugin target native_endpoint_verification;
aspect-release --test '*Fault*,*fault*,Stage-I*':21,112assertions/55cases pass.
No tolerance/default/expected-output relaxation. One namespace qualification
was fixed during compilation. The sandbox MPI launch fails interface discovery
before ASPECT; the host-enabled one-rank batch passes. This infrastructure
failure is separate from numerical results.

Evidence under benchmarks/reconstructed_fault/propagation/direct_mechanical_evidence/: baseline.json,
preservation.json; batch/growth/saddle inputs, start/run records and logs;
batch-assessment.json, growth-assessment.json, saddle-assessment.json; separate
*-factor-reuse.json and prototype-actions.npz; compatibility.log. Native outputs
are in build-propagation-integration/runs/output-native-mpi-{batch-n1-local,
growth-direct-n1,saddle-direct-n1}. The scripts run_direct_mechanical.py and
assess_direct_{mechanical,entries}.py preserve uniqueness and enforce the cost/
growth gate before the next bounded entry. No additional run follows this task.
