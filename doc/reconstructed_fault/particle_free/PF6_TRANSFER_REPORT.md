# Bounded fixed-reference transfer qualification

16 October 2026. Base `a8707307008401ac9cff702a369e2e2a611fa0f7` plus the
preserved incoming work and this experimental extension. **Uniform Cartesian
refinement/coarsening, matching-cell ownership, rejection/publication and native
restart pass the specified serial and 1/2/4-rank checks.** Coarsening is not
lossless in general. The unchanged native quadrature misses the declared 1e-9
physical-functional screen on this coarse evolving fixture; that limitation is
retained separately from the passing remap checks.

[Rules declared before runs](PF6_TRANSFER_FORMULATION.md),
[source changes](SOURCE_AUDIT.md), [particle-read inventory](PF6_PARTICLE_READ_AUDIT.md),
[portable inputs, measurements and replay](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf6_transfer/README.md).
No transport, surface remeshing, moving geometry, birth or propagation is tested.
Unlimited DGQ2, physical laws, limiter settings and production defaults remain.

## Comparison and ownership

The publication repair is accepted only within its prior documented scope.
**`strict_reference_match_passed=false` remains false.** Its original primary,
unaligned errors, reference, failed/capped runs and binaries are preserved in
[PF6A_PUBLICATION_REPORT.md](PF6A_PUBLICATION_REPORT.md). In particular, the old
maximum scaled whole/localized tensor-stress errors are
1.499173665541632e-6 / 3.652740600152377e-6; phase relative L2/Linf are
.00011527520202700741 / 5.9085114797297056e-5; driving error is
.00023833147203112777. This task does not rerun or reclassify that trajectory.

A transfer compares one accepted branch at common physical points before/after
remapping, without relaxation, reflection, registration, a phase solve or a
clock/publication increment. MPI/restart comparisons use identical inputs and
primary unaligned values. The native between-candidate energy allowance and the
independent reference's strict comparison remain different documented branch
selection rules; no tie rule is changed here.

For the DG consistency guard, the particle's **surrounding CellId owns both
the native FE velocity gradient and the DG polynomial trace**. Position is
mapped into that same reference cell. Continuous FE velocity does not imply a
continuous gradient. Association IDs, ordering, positions and counts are checked.
Continuous phase/temperature sampling and legacy FE-history gradient conventions
are retained. No unused averaged gradient is evaluated on this DG path.

The analytic face control uses continuous piecewise-Q2
`u_x=y²+max(0,y−.5)`, `u_y=0`: one-sided `d_y u_x=1,2`, averaged value 1.5.
Matching-cell Maxwell/DG discrepancy is at most **2.04e-13 Pa** in the full tensor
metric; the averaged-gradient negative control differs by **7070.71427 Pa**.
All negative controls fail the unchanged publication guard. Original allowances
and the prior operand-based roundoff repair are unchanged. A local sampling
failure is now propagated to all ranks before subsequent collective evaluations
or terminal writes.

## Remap and persistent state

Native distributed `SolutionTransfer` handles three scalar DGQ2 fields. All nine
coefficients per component survive; mixed modes, including x²y², are retained.
Refinement restricts the parent polynomial to each child. Coarsening solves

\[
 M^K_{ij}T^K_{c,j}=\sum_{k\subset K}\int_k b^K_i T^k_c\,dx,
 \qquad M^K_{ij}=\int_K b^K_i b^K_j\,dx.
\]

Thus each target-cell mean is conserved and the error is orthogonal to Q2.
Independent Gauss5 assembly/solve verifies the actual native restriction; it is
not an oracle assembled from deal.II's restriction matrix. Full-tensor comparisons
use `xx²+yy²+(xx+yy)²+2xy²`, including zz and both symmetric xy entries.

At the accepted endpoint physical phase equals its next physical lower obstacle.
Refinement interpolates that same Q1 function. Coarsening is admitted only if
every child vertex equals the parent Q1 interpolant, with global absolute defect
at most **256 epsilon = 5.68434e-14**. Q1 equality on child vertices proves equality
throughout each child. A nonrepresentable +.01 fine peak is rejected collectively
before particle/userdata preparation and mesh mutation; accepted state is intact.
No obstacle projection, phase cap or healing repair is introduced. Current, old
and older native solution vectors transfer, as does the accepted phase anchor.

The fixed surface has eight cells/nine vertices. Its connectivity, orientation,
properties, Z/I, current/committed slip rates and history clocks are retained.
Mesh-dependent constraints, DG ownership/ghosts, quadrature/normalization rays
and associations rebuild; source-error and point caches are scratch and cleared.
Transfer applies **zero** Maxwell, source, cohesive or phase-evolution updates:

\[
 T_e=\beta T_n+2\kappa D_3(\operatorname{sym}\nabla u),\quad
 Z_e=\beta Z_n+\kappa V,\quad \tau=T-2\chi ZS,\quad q=Z/I,
\]

with the retained exponential beta/kappa and phase localization. These equations
apply once in the subsequent accepted physical step, using the remapped incoming
state. The native checkpoint infrastructure stores version-3 owned-CellId DG
payloads, geometry/phase checks and clocks, allowing repartitioned restore. Old
serial v1/v2 readers remain; no history-backend conversion is enabled.

## Measured serial transfer defects

Declared screens are scaled 1e-11 for polynomial transfer and 1e-9 for native
physical observables. Relative/scaled norms divide by `max(1, reference norm)`;
absolute units and raw exports are retained. Common physical integration uses
independent Gauss24, with Gauss32 uncertainty recorded. These are **remap-only**
comparisons; no physical step occurs between each pair.

| Check | Measured defect |
|---|---:|
| Nine monomials: prolongation / roundtrip | 1.22e-15 / 9.99e-16 |
| Independently assembled vs native mass restriction | 5.00e-16 |
| Complete non-proportional global Q2: coefficient roundtrip | 7.11e-15 Pa |
| Accepted nonzero-Z state: refinement / roundtrip stress scaled L2 | 1.38e-15 / 7.17e-16 |
| Same state's maximum physical stress difference | 5.46e-12 Pa |
| Phase and all phase time levels | <=3.33e-16 |
| Same state's common-rule bulk energy / weak shear defect | <=3.33e-16 J/m / 1.28e-13 Pa m |
| Independent ordinary-FE fracture gradient/Hessian action, all ranks | <=1.16e-14 |
| Constant fracture Hessian action, all ranks | <=1.43e-14 |

The **lossy** native fine-cell tensor control has arbitrary non-proportional
values in every component/mode, with exactly representable phase. Coarsening
agrees with the independent 9x9 projection solve to **3.22e-15 Pa**, preserves
target means to **8.61e-16 Pa**, and satisfies the full-tensor Pythagorean identity
to **1.00e-15**. Nonetheless it changes stress by **.860085 Pa L2 (.701247 scaled)**,
the quadratic norm by **−.739746 Pa² m²**, bulk energy by
**−1.84936e-7 J/m**, and a fault weak load by **.0124820 Pa m**. Those are measured
projection losses, not physical dissipation or a claim of acceptable coarsening
accuracy. Z=0 in this lossy control; when Z is nonzero the localization cross term
means reconstructed physical energy need not decrease under projection of T.

The independent checker exposed an older-solution phase-buffer reset of
.598494163 before the core began transferring that buffer. That failed evidence
is retained. Other corrections were counting Boolean refinement flags rather
than their enum values and rebuilding phase constraints before accepted caches.
The MPI observer's mixed Add/Insert vector update was corrected in the benchmark;
temporary matrix/solver workarounds were removed. No solver redesign or tolerance
relaxation is part of the final implementation.

## Native lifecycle, MPI and restart

The bounded fixture starts from the existing seed table and unloaded T/Z, with
dt=.125 s and two physical steps to .25 s. A base 2x8 grid (16 cells) undergoes
exact roundtrips, is held at 4x16, and temporarily refines to 8x32. Phase is the
same embedded base Q1 function, not a freshly sampled fine seed; this is not a
spatial-convergence study. The physical steps develop nonzero Z (981.339 then
6870.63405 Pa m), q (10.6813 then 74.7682 Pa) and phase-tail growth. Peak phase
remains .60000194. Fixed surface and no advection are explicit.

Each full rank-count case passes **70 native gates**, including unique owners,
ghost-backed evaluations, particles on child faces, invalid obstacle coarsening,
and the actual native outer rejection/retry/publisher. A last-owner rejection
after convergence preserves histories, RNG, all native buffers, clocks and
bookkeeping. A staged unpublished retry rejects remeshing, then discards cleanly.
An actual terminal publisher failure caused by a missing local evaluation cell
reaches every rank, preserves the same accepted state, and permits retry. Each
successful physical step publishes exactly once. Remapping never publishes.

| Comparison | Max scaled tensor T L2 | Max scaled physical stress L2 | Max scaled complete native fields |
|---|---:|---:|---:|
| 1 vs 2 ranks | 1.35e-15 | 1.08e-14 | 1.74e-13 |
| 1 vs 4 ranks | 1.37e-15 | 9.53e-15 | 5.16e-11 |
| Same-rank restart vs uninterrupted, each of 1/2/4 | 0 | 0 | 0 |
| Serial checkpoint restored on 4 vs uninterrupted 4 | 1.37e-15 | 1.16e-14 | 5.16e-11 |

All persistent tensor coefficients, native current/old/older/anchor fields,
phase/obstacle, surface fields, particles/properties, fracture actions, stress and
weak loads are compared. Largest absolute native-field difference is an initial
pressure old/older buffer difference of **1.51e-8 Pa**. Phase differences are
<=6.11e-16. Particle CellId mismatches are zero. Same-count and serial-to-four
restart RNG bytes match per rank; a fresh 1-vs-4 comparison has different numbers
of RNG records and is not a byte-identity test of rank layouts.

| Ranks | Full instrumented run | Half/checkpoint | Restart | 64→256 / 256→64 mesh transfer |
|---|---:|---:|---:|---:|
| 1 | 12.391 s | 10.735 s | 3.322 s | .0433 / .0368 s |
| 2 | 35.387 s | 27.619 s | 8.282 s | .1687 / .1627 s |
| 4 | 47.674 s | 37.201 s | 11.035 s | .2567 / .2428 s |

Serial-to-four restart takes 12.107 s. Runs are sequential, one thread/rank,
core-bound, at most four processes, each under a 180 s native cap. Timings include
diagnostic gathers/exports, repeated rejected solves and cache/matrix rebuilds;
the tiny replicated endpoint fixture demonstrates no parallel speedup. Bulk
DG DoFs are 432/1728/6912, phase DoFs 27/85/297 and velocity DoFs 170/594/2210
at 16/64/256 cells. Surface DoFs stay nine.

## Quadrature limitation, separated from transfer

The retained Gauss3 mass/source and native Gauss6-normal physical rule are
unchanged. Even with identical remapped physical fields, the native integration
points change on refinement. At accepted step 1 native-minus-Gauss24 bulk energy
is −3.08e-10 J/m before refinement and −1.45e-13 afterward; the largest shear-load
error falls from **8.41e-8 to 3.91e-11 Pa m**. Common-rule transfer errors above
remain at roundoff. Re-evaluating a previously equilibrated state with another
quadrature rule is not another equilibrium solve.

On evolved state 2 the native bulk energy error is **−1.52022e-8 J/m** (scaled
1.52022e-8), weak-shear error **5.90193e-7 Pa m** (scaled 4.39308e-9). Gauss24/32
uncertainties there are 2.22e-16 J/m and 2.56e-13 Pa m. Native fault KKT residual
is 1.15e-12 Pa m, whereas independent high-rule residual is 5.90e-7 Pa m.
The complete-Q2 diagnostic also slightly misses the 1e-9 load screen on the base
grid. Accordingly **`native_rule_vs_high_rule_screen_passed=false`** for all rank
counts. A solved native residual is not evidence of high-rule equilibrium.
No tolerance, quadrature or physical acceptance rule was changed to hide this.

## Particles, reproducibility and recommendation

Effective integrator is RK2, `ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1`
present, advection dt=0; constitutive dt remains .125 s. **All 144 IDs and physical
coordinate strings are identical** at initialization, before/after both steps,
every exported transfer and restart. Eighty particles lie on child faces; owner
rank/CellId/reference coordinates change consistently with refinement. This tests
the trace convention where it matters, without moving particles.

Remaining reads and proposed replacements are individually classified in the
[audit](PF6_PARTICLE_READ_AUDIT.md): H/CPDI initialization, domains/support and
H-dependent mapped-field selection, particle chemical material projection,
surface material normalization, ownership/motion checks, publication validation
and derived output, optional diagnostics, manager/RNG/ghost persistence.
Independent T/Z remains authoritative for accepted mechanics and fault loads;
stored particle stress does not feed those equations. Fixed particles do not
remove their initialization, material or lifecycle dependencies.

All ten final native cases use executable SHA256
`80aa0d4f12840524edf5311da22875d888aac2e9e6971d149f852d19eeaa6328`
and observer `be3f7d61490fd3ee6a50a61ab72dbadef5f84242b2fd88efb8d43c8b6b78c634`.
Complete inputs, argv, source hashes/overlay, original failures and checkpoints
are archived. Focused FE control reproduces 21 prior CSVs byte-for-byte; PF4
operator/control reproduces 23; PF1 independently passes 94 checks. Prior 628-unit
publication evidence is retained, not claimed as a new run. Preservation checks
cover all 6406 incoming files and retain historical STATUS/SOURCE_AUDIT payloads.

Retain unlimited DGQ2. Exact uniform refinement and representable phase/obstacle
coarsening now have bounded ownership/lifecycle evidence. **Do not enable
automatic or lossy coupled AMR from these results.** The next concrete task is a
separately selected fixed-geometry local-AMR qualification: define a physical
stress/weak-load/energy transfer error budget (including T−chi Z cancellation),
choose an admissible irreversible obstacle rule for genuinely lossy coarsening,
establish quadrature accuracy below that budget, then test hanging-node caches,
ownership and restart with the fixed surface. Surface remeshing needs its own
Z/slip/material transfer and work/energy checks if later required. General PF5,
strict branch matching, transport, moving geometry and propagation remain open.
