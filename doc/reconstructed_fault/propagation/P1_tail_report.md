# P1 initialization/tail diagnosis — 2026-10-06

The bounded diagnosis is complete. **Equality convergence and nontrivial
prescribed-H evolution pass; physical admissibility fails in every case.**
Moving the boundary away does not remove the finite-seed violation. Removing
the cap sharply reduces the initial profile discrepancy and eliminates
measured ramp healing in these controls, but leaves a resolved tail undershoot.
One finer straight-band repeat improves the overall profile approximation
while increasing the undershoot. P1 remains unqualified; stop before P2.

## Scope, identity and implementation

This is the user-selected continuation of [P1](P1_report.md), not a new coupled
experiment. Current worktree/branch remain
`/home/ein/repository/aspect-fault-propagation` / `fault-propagation`, HEAD
`de7beaa65846c7372711f7922abf1c5d9a745ee6`. Incoming changes were AGENTS, the P1
test helper/plugin/fixtures/checker, and task docs; they were preserved.
Committed AGENTS remains an exact prefix of the additive instructions. No
nested/override instruction was found. The current-design initialization
contract (§§2,13), specification phase range/API rules and refactoring guidance
were reread. The existing physical-bound failure is reported, not silently
repaired by changing the algorithm or specification.

The read-only reference core is the sibling R7b `aspect-r7b-qualified`, source
`6a3781277f2f594d52cd248d84f5b6fdc98fdeb1`, SHA256
`ed74d4d4ad1cfa5d1bd4fbade496d9748ab298c24a259a4612a94d4df375b60d`.
Its production source/headers match HEAD. Stack: GCC 12.4.0 C++, OpenMPI 5.0.6,
deal.II 9.6.2, Trilinos 14.2/Epetra, Voro, Release. CMake detects the unused
system C compiler as GNU 16.2.1; the plugin has C++ sources only. One plugin
build in `build-propagation-p1-tail` uses `-fno-finite-math-only -ffp-contract=off`;
zero core builds. Sibling build and historical outputs remain untouched.

`tests/phase_field_controlled_history.cc` now reads the actual prescribed
segment through the existing manager/closest-point API instead of hardcoded
endpoints. It checks constant core/horizontal geometry and identical material
profiles. New observations assemble the native profile and a clipped copy of
the initialized state, and save two nodal cuts plus one particle column.
Clipping is **assembly-only**: it never enters a solve or committed state.
Residual, Jacobian and roundoff evaluations, including these observations, use
scratch phi while live phi is deliberately 1.25. Normal restoration, rejected
assembly, and exception restoration checks remain intact. The finite-case
five-stage CSV is byte-for-byte identical to the previous refined P1 CSV.

The new benchmark directory contains small input overlays, a CMake entry
reusing the native test source, one-case runner, saved-evidence checker and
plotter. No constitutive equation was duplicated. Production `source/`,
`include/`, defaults, coupling and H evolution were not changed. The old
refined test source is preserved in `build-propagation-p1/refined-source.cc`
with its original `17bb9d6c…` hash; the old P1 manifest describes that historical
source version. A retained instrumentation diff reconstructs the transition.

## Bounded controls and measurements

All retain ell=0.15625, AT1/Lorentz p=1, m=128, core phi=0.8, two identical
materials, cohesive mode, Q1/CPDI, 3×3 particles/cell, no replenishment,
mechanics or advection. The native profile support radius is
**0.3116482098659328**. The original side gap 0.2 cuts into the cap support;
the padded gap 0.7 exceeds it. Top/bottom distance 0.5 exceeds support in every
case. The through-going band is x-independent and compatible with the natural
phase boundary condition; it has no radial cap within the box.

| Control | Box / seed | Cells / particles | h_cell/ell |
|---|---|---:|---:|
| finite | [0,1]² / x=0.2…0.8, y=0.5 | 1024 / 9216 | 0.2 |
| padded | [-0.5,1.5]×[0,1] / same seed | 2048 / 18432 | 0.2 |
| straight | [0,1]² / x=0…1, y=0.5 | 1024 / 9216 | 0.2 |
| straight_fine | same straight band | 4096 / 36864 | 0.1 |

The first three controls were selected before execution. The persistent
straight-band undershoot triggered the sole finer repeat in this continuation.
No further run was made. Loading retains the previous 0.1<initialized CPDI
phi<0.7 mask and max-history prescription: hold, identical-H hold, +5%, +10%,
then unchanged-H unloading. Mask sizes are 2792, 2632, 3072 and 12672. Because
the mask follows each initialized field, ramp comparisons are not identical
spatial forcing; initialization comparisons precede those modifications.

| Quantity | finite | padded | straight | straight_fine |
|---|---:|---:|---:|---:|
| Native initial profile max difference | 0.166093 | 0.177189 | 0.00847553 | 0.00229416 |
| Minimum nodal phi across five tight solves | -0.00603795 | -0.00639779 | -5.25097e-5 | -1.71362e-4 |
| Minimum CPDI phi | -0.00519851 | -0.00558315 | -4.39516e-5 | -1.51370e-4 |
| Max load-induced healing | 4.23116e-6 | 4.10329e-6 | 0 | 0 |
| +5% ramp max change | 0.00596992 | 0.00578346 | 0.00216477 | 0.00222787 |
| +10% ramp max change | 0.00573741 | 0.00555853 | 0.00212318 | 0.00218323 |
| Diagnostic bound/healing allowance | 1.49219e-12 | 1.26937e-13 | 1.46836e-13 | 1.86214e-13 |
| Runtime, seconds | 4.875 | 5.428 | 2.822 | 6.881 |
| Peak child RSS, MiB | 208.6 | 224.8 | 208.4 | 258.0 |
| Physical qualification | Fail | Fail | Fail | Fail |

All 20 fresh equality residual checks pass, with actual residuals respectively
4.32e-15…2.68e-12, 4.11e-15…7.27e-14, 4.36e-15…5.21e-15 and
8.53e-15…9.11e-15. Nonlinear relative tolerance is 1e-8; CG relative tolerance
is 2e-7; limits are 50 Newton / 1000 CG iterations with line-search setting 3.
Each solve's target is max(1e-8 × entry residual, its production roundoff
allowance). Actual target ranges are 3.94e-13…8.35e-10,
3.92e-13…8.52e-10, 4.85e-13…2.69e-10 and 9.23e-13…1.58e-10.
Newton counts are [2,0,3,3,1] for finite and [2,0,3,3,0] for the others.
Exact per-solve values are in the CSVs and independent checker records.

The largest scaled Jacobian-correction estimates are 1.49219e-13,
1.26937e-14, 1.46836e-14 and 1.86214e-14. These are directional uncertainty
estimates, not bounds on inverse-Jacobian norm or discretization error.
Identical-H repeat drift is 1.19463e-13 for finite unloading and zero for all
other repeats. The allowance is ten times max(estimate, repeat drift);
repeat consistency and both >100×allowance evolution gates pass. Significant
bounds/healing violations exceed that numerical scale by many orders of
magnitude. Their existence is resolved; their asymptotic mesh behavior is not.

## Diagnosis and limits

1. **A nearby side boundary is not necessary for finite-seed failure.** Padded
   initialization still differs from its radial profile by 0.177 and produces
   a -0.00639 tail. Its worst tight-solve node is at (1.09375,0.5), inside the
   cap support but far from the boundary at x=1.5. The cap's profile adjustment
   is visible in the [saved cuts](../../../benchmarks/reconstructed_fault/propagation/evidence/initialization.svg).
   A boundary contribution to the original case is not ruled out.
2. **The radial cap is not a stationary one-dimensional band.** Source
   `PhaseFieldProfile` and `stationary_crack_driving_force()` provide the
   straight normal profile; the manager applies them to closest-point distance,
   radial beyond a finite endpoint. For a constant-coefficient radial cap,
   laplacian(phi)=phi''+phi'/r, whereas a straight band has phi''. The extra
   curvature term is a source-based explanation for cap relaxation, not a new
   driving law or proof that it explains every negative node. Eliminating the
   cap reduces the measured profile mismatch by roughly twentyfold.
3. **A separate discrete tail issue survives in the straight band.** At y=0.1875
   the native profile is zero, yet the initialized straight-band node is
   -5.32450e-5 (h=1/32) and approximately -1.72444e-4 (h=1/64). The tighter holds
   leave the negative values in the table. On these sampled minimum rows the
   x variation is only 1.5e-16 and 1.8e-12 before the hold, respectively, so this
   is not an endpoint-localized artifact. Global profile error decreases by
   about 3.7× under refinement, but tail undershoot increases about 3.3×.
   Two meshes do not establish an asymptotic error rate or a mesh-independent
   lower-bound failure.
4. **The native H transition and assembled operator explain what to examine
   next.** At h=1/32, the central particle column has H=0.5 at y=0.1822917
   outside support, then H=0.0116585 at y=0.1927083 just inside. At h=1/64,
   those neighboring samples are y=0.1848958, H=0.5 and y=0.1901042,
   H=0.0111988. Source initialization retains baseline H outside the activation
   threshold and assigns stationary-profile H inside. This sharp transition
   is prescribed initialization, not healing of mechanical history. The Q1
   field spans it and the production operator evaluates nonlinear terms at
   CPDI-averaged phi/gradient. The native-profile nodal interpolant is not a
   discrete equilibrium: raw RHS norms are 0.24707 and 0.037699 for the straight
   cases. These unnormalized norms depend on mesh size; no convergence order
   is inferred from them. Positive tangent off-diagonal counts remain 1674
   and 176 at the first tight hold, so a discrete comparison principle cannot
   simply be assumed. Positive off-diagonals alone do not prove the mechanism
   of undershoot. These runs do not isolate quadrature, interpolation,
   profile tabulation and transition alignment contributions.
5. **Clipping is not an equality-preserving repair.** For the native initialized
   states, merely replacing negative nodal values with zero changes residual
   norms from 7.96e-6/7.66e-6/5.59e-5/4.86e-5 to
   0.29034/0.42424/0.0049458/0.014575. These observations precede the tighter hold;
   they are not claimed to use its final residual tolerance. No clipped state
   was advanced, and no clipping, H floor, active set or KKT algorithm was added.

Finite ramps still heal by ~4e-6. Straight ramps show no healing on either
mesh, and enough loading removes their undershoot (the finer +5% state still
has -4.30690e-5). Later admissible states do not erase the failed initialization
gate. No universal monotonicity qualification follows from two straight ramps.

## Verification, artifacts and handoff

The [benchmark README](../../../benchmarks/reconstructed_fault/propagation/README.md)
contains exact configure/build/run/check commands. The
[manifest](../../../benchmarks/reconstructed_fault/propagation/evidence/manifest.json),
[summary](../../../benchmarks/reconstructed_fault/propagation/evidence/summary.json)
and per-case logs/CSVs/checks preserve provenance. Each run stayed below 120 s,
2 GiB RSS and 10 MiB output. All four scratch-state/normal/exception checks
passed; expected-output filtering correctly lacks the qualification line.
The independent numerical checker exits 2 for each physical failure. The
saved-evidence consistency checker exits 0 while explicitly reporting
`physical_qualification=false`. No failing result was relabeled as a pass.

One initial plugin compilation needed an explicit cast from the Trilinos
vector proxy to double in the observation-only clipping code. A runner setup
check then needed the actual `.release.so` suffix; neither launched a case.
After the finite run, a whitespace-sensitive wrapper assertion was corrected
for ASPECT's line wrapping; saved simulation/checker results were valid and
were rechecked without another simulation. Inherited unused-variable warnings
in the included registration fixture remain. Four simulations total.

Updated files are the test instrumentation, new propagation benchmark and
evidence, benchmark index, and PLAN/STATUS/report links. Historical P1 evidence,
original instructions, production source and existing builds are preserved.
No commit or push. No P2 work, native restart, physical timestep rollback,
transport/birth check or propagation. Monotone diagnostic H cannot qualify
the production H law or answer the historical small-increment feedback-loss
question; that remains specifically a P2 investigation of the mechanical
candidate/max update under the audited sequential split.

Next proposed task remains P1: a separately selected operator-consistency
diagnosis at the straight-band H support transition, distinguishing particle
integration from FE/profile representation before choosing a remedy. No
solver/H-law change or further campaign is implicitly authorized by this report.
