# K5 condensed mechanical solve: timing and pivoted surface prototype

## Decision summary

The pivoted tridiagonal surface prototype passes the focused indefinite,
active-set, MPI, condensed-action, recovery and rollback checks. Coarse BP3
completes three real steps, and evolving K3 completes all nine accepted states,
with trajectories agreeing to roundoff and unchanged linear iteration counts.
No equations, pressure treatment, quadrature, support, tolerances, bounds or
history rules changed. The pending I_h boundary issue was not reopened.

Surface inversion is **not the mechanical bottleneck**. BP3 factorization plus
589 inverse applications drops from 0.0734 to 0.00675 s, while its timed coupled
work remains about 334 s. Stokes preconditioning takes 157 s, B 82 s and G
32 s. K3 reaches the same conclusion. No meaningful end-to-end acceleration
can be attributed to saving these milliseconds.

Stop here: retain the opt-in LAPACK prototype and UMFPACK reference. The next
separate proposal is interface-aware preconditioning, not outer CG, another
surface inverse, or a fine BP3 run. That design is documented but not implemented.

## 1. Implementation and invariants

The same semantic `ReconstructedFaultSurfaceLinearSolve::solve` interface now
supports a private `FaultSurfaceDirect` implementation shared by unrestricted
and restricted systems. Each contiguous principal free block has its own
LAPACK GTTRF factors, adjacent pivot indices and second superdiagonal. GTTRS
reuses them for every RHS. Zero/negative original diagonal entries are allowed;
nonsingular indefinite matrices require no exception or SPD assumption.

The installed Teuchos LAPACK wrapper provides the tested implementation.
There is no handwritten Thomas/Cholesky algorithm or condition estimator.
Original tridiagonal coefficients are retained for backward checks. Each free
block uses its own residual scale with the existing 100*epsilon*n_fault
allowance; a large neighboring block can no longer hide its error. Finite
input/factor/solution checks and LAPACK singular-pivot checks identify the
fault, block or vertex. Active/prescribed RHS entries are ignored and their
returned increments are exactly zero. Generation checks, complete-candidate
publication, and failed-solve/history rollback are unchanged.

Prototype switches (no new physical/numerical parameters):

- `ASPECT_FAULT_SURFACE_SOLVER=tridiagonal`: LAPACK prototype. Unset or
  `umfpack` retains UMFPACK; the production default is not silently changed.
- `ASPECT_FAULT_COMPARE_SURFACE_INVERSE=1`: additionally factor UMFPACK and
  compare **every** RHS solution at relative 2e-12. This was enabled for the
  production prototype replays. Reference work is charged to `other`, not to
  specialized factor/inverse time, but remains in end-to-end elapsed time.
- `ASPECT_FAULT_LINEAR_PERFORMANCE=1`: exclusive rank-local timing/call counts.

The current prototype uses the build's Trilinos LAPACK wrapper and retains
UMFPACK for reference. A build without that wrapper was not qualified.

## 2. Disjoint timing and accounting

One profile starts before surface linearization and ends after the current
active-set solve(s) and delta-V recovery. Nested sections charge only their
exclusive time: algebraic K_V matrix/factor construction, inverse applications,
A, B, G, Stokes preconditioning, FGMRES internal work, and other setup/vector
work. No timing collectives are inserted; these are root-rank wall times.

`FGMRES_vectors` is the installed FGMRES solve scope **minus** operator and
preconditioner application time. It includes Arnoldi/orthogonalization,
convergence/vector operations and the small projected triangular solve.
It is not an unsupported claim to have timed pure dot products separately.
`Other` explicitly includes surface coefficient/residual assembly, G lookup
construction, frozen B setup, constraint/pressure conversions, nullspace/fresh
check orchestration, and remaining vectors. The factor column is algebraic
matrix/factor construction, not those particle constitutive evaluations.
Bulk Stokes assembly and preconditioner rebuilding precede the profile and
retain their ordinary ASPECT timers. Component sums reconcile with the profile
duration; the rest of end-to-end time includes preparation, trial residuals,
advection/domains, histories, output and initialization.

### One BP3 condensed linearization: timestep zero, Newton iteration zero

| Exclusive operation | UMFPACK (s) | Pivoted LU (s) | Calls |
|---|---:|---:|---:|
| K_V matrix/factor construction | 0.001317 | 0.0000946 | 2 |
| Surface inverse, including backward checks | 0.001396 | 0.0001913 | 20 RHS |
| A action | 0.20907 | 0.20654 | 18 |
| B action | 2.72241 | 2.70443 | 19 |
| G action | 0.96716 | 0.93687 | 19 |
| Stokes preconditioner | 5.73964 | 5.64616 | 17 |
| FGMRES Arnoldi/vector work | 0.14127 | 0.13729 | 1 FGMRES invocation |
| Other setup/vector work | 1.43357 | 1.39640 | — |
| **Total** | **11.21584** | **11.02798** | |

There is one fault: the two constructions are its unrestricted factor and
its prescribed/free restriction, not per-Krylov refactorizations. Only the
free 400-vertex block is solved by GTTRS in the restricted BP3 path; deep
prescribed entries return zero. The unrestricted fault has 1156 vertices.

### Full short trajectories, disjoint cumulative seconds

| Operation | BP3 UMFPACK | BP3 pivoted | K3 UMFPACK | K3 pivoted |
|---|---:|---:|---:|---:|
| K_V matrix/factor construction | 0.03456 | 0.00142 | 0.01339 | 0.00050 |
| Surface inverse/backward checks | 0.03882 | 0.00533 | 0.04474 | 0.00273 |
| A | 6.28432 | 6.44422 | 1.09695 | 0.83229 |
| B | 81.69853 | 81.86696 | 14.27209 | 10.49767 |
| G | 32.96677 | 31.85910 | 5.32971 | 3.85609 |
| Stokes preconditioner | 155.38191 | 156.68237 | 16.08315 | 11.35887 |
| FGMRES Arnoldi/vector work | 4.17249 | 4.29639 | 0.55357 | 0.40002 |
| Other setup/vector work | 53.57451 | 53.18181 | 7.62421 | 5.86088 |
| **Profile total** | **334.15191** | **334.33760** | **45.01780** | **32.80905** |
| End-to-end wall | 616.80208 | 663.32627 | 116.59652 | 88.43777 |
| Child user CPU | 572.54020 | 587.69661 | 79.33753 | 79.85853 |
| Child system CPU | 6.58454 | 7.01089 | 1.35328 | 2.77477 |
| Peak RSS (KiB) | 4,477,012 | 4,473,840 | 519,920 | 520,568 |

The run order included partial overlap with short verification work (notably
the beginning of baseline K3 with the end of baseline BP3). These are not
statistically controlled end-to-end speed measurements: **do not attribute
K3's 28 s wall reduction or BP3's 47 s wall increase to surface LU.** K3's user
CPU is essentially unchanged, as are iteration/action counts; BP3's timed
coupled work is essentially unchanged. No repeat campaign is warranted to
resolve a speedup whose maximum possible end-to-end benefit is below 0.1 s.

Measured factorization ratios are about 24x (BP3) and 27x (K3), and inverse
ratios 7.3x and 16.4x. Those kernel ratios are useful only alongside their
absolute cost. Specialized factorization plus inversion is about 0.002% of
BP3's timed coupled work and 0.010% of K3's. Bulk preconditioning and repeated
B/G applications dominate; FGMRES orthogonalization itself does not.

| Cumulative call count | BP3, both backends | K3, both backends |
|---|---:|---:|
| Coupled linearizations / returned directions | 29 | 37 |
| Factor constructions | 58 | 37 |
| Surface RHS solves | 589 | 706 |
| A actions | 531 | 632 |
| B / G actions | 560 / 560 | 669 / 669 |
| Preconditioner applications / linear iterations | 502 | 558 |

RHS/recovery, fresh residuals and eligible pressure-nullspace checks explain
the extra actions beyond Krylov iterations. For example BP3's 589 inverses
are 502 Krylov + 29 fresh + 29 RHS + 29 recovery. K3 additionally has 37
full constrained right-null checks. Factors are reused, not recomputed for
those applications. Exact per-linearization records are in the JSON artifacts.

## 3. Numerical and lifecycle verification

One/two-rank unit tests pass **615 assertions in 12 cases**, covering SPD,
deliberately nonsingular indefinite and pivot-requiring zero-diagonal matrices,
singular-pivot rejection, nonfinite inputs, negative scalar blocks, multiple
RHS, unrestricted/middle/alternating/endpoint/all-active partitions, exact
active zeros, lower-contact release, Armijo rejection/exhaustion and rollback.
The 2x2 matrix with diagonal (0,0), off-diagonal 1 has eigenvalues +/-1 and
requires pivoting; it is not a positive-definite surrogate test.

The production fixture's independent semantic reference reconstructs K_V from
its actual action and factors it explicitly with UMFPACK. On the **same**
frozen A/B/G/K_V generation, it compares the full condensed action (2e-11
bulk-relative allowance) and recovered delta-V (2e-12) for unrestricted,
middle-active, endpoint-active and alternating free blocks. The dedicated
assembled-Stokes fixture passes on one and two ranks. The lighter prescribed-
Stokes action fixture tests F/G but skips Stage H; it is not substituted for
that dedicated comparison. This distinction was checked in the source.

Production K1, K2, K3 and BP3 replays run the pivoted backend with every RHS
also checked against UMFPACK. All returned fresh linear checks pass:

| Case | Accepted coverage | Returned fresh checks | Linear iterations |
|---|---|---:|---:|
| K1 homogeneous short case | initialization + 0.5 s | 13 | 188 |
| K2 true-pressure coarse | initialization + 0.5 and 1 s | 17 | 275 |
| K3 evolving 32x128/fault32, dt=.375 | states 0--8 through 3 s | 37 | 558 |
| BP3 coarse, Vmin=1e-20 | initialization + three real steps | 29 | 502 |

For K3 both backends pass all nine existing lifecycle/phase/reference/support/
homogeneity guards. Maximum absolute accepted-field differences are V=1.29e-16
m/s, Theta=1.06e-12 s, C=2.16e-12 Pa, I_h=9.95e-14 m, phi=2.22e-16,
particle H=1.56e-13 Pa and particle tau_xy=9.87e-10 Pa. The surface CSV does
not contain cumulative slip, so no unsupported bitwise slip claim is made.

For BP3 the accepted physical times and timesteps are identical, as are
iteration counts (3,3,13,10 including final convergence evaluations) and all
active/free partitions. Maximum accepted alpha changes are 1.28e-15 in step 2
and 5.55e-16 in step 3. Maximum accepted-field differences are V=6.20e-25 m/s,
Theta=2.79e-9 s, C=4.07e-10 Pa, slip=3.47e-18 m, q=1.12e-8 Pa and **I_h=0**.
All 29 fresh checks pass; the largest fresh/target ratio is 0.99523 in either
backend. Initial supplied histories, frozen backgrounds/geometry and subsequent
split updates pass the existing BP3 analyzer. Existing coarse support/endpoint
accuracy limitations remain and are not certified away by this solve test.

The one-/two-rank production rollback fixture deliberately exhausts nonlinear
iterations after an accepted update and verifies exact restoration of bulk,
current/committed V, surface and particle histories. Its configured failure
strategy allows exit zero; the failure-signal assertions, not exit status,
provide the evidence. No new restart I/O or fine/MPI convergence campaign was
run. The ordinary surface factorization remains non-checkpointed and tied to
the current coupled linearization.

## 4. Evidence, source and next decision

All paths below are under `benchmarks/reconstructed_fault/performance/`:

- `mechanical-bp3-result.json`, `mechanical-k3-result.json`: per-linearization
  disjoint timing, counts, resources, fresh checks and accepted-field differences.
- `mechanical_bp3_{umf,pivot}/perturbation_report.json` and `cache_audit.json`:
  dimensional tractions, histories, accepted times and nonlinear paths.
- `mechanical_{bp3,k3}_{umf,pivot}.log/.prm/.resources.json`: full run provenance.
- `mechanical-unit-{one,two}.log`: 615 passing assertions per rank.
- `mechanical_actions{,_two}`: F/G checks, 17.65/17.15 s.
- `mechanical_condensed{,_two}`: actual assembled condensed action/recovery,
  11.84/9.53 s.
- `mechanical_rollback{,_two}`: intentional-failure preservation, 8.28/8.38 s.
- `mechanical_k1`, `mechanical_k2`: short production inverse comparisons,
  24.87/79.02 s.

Core and focused plugin builds use `-j4`, Release. A test-only missing dependent
type qualification was corrected after a compile error; no failing numerical
case was retuned and no expected output was weakened. `git diff --check` passes.

Typical commands (the runner refuses to overwrite prior evidence):

```sh
cmake --build build-pf-cpdi --target aspect.exe.release -j4
build-pf-cpdi/aspect-release --test '[fault_surface_direct],Stage-I*'
mpirun -np 2 build-pf-cpdi/aspect-release --test '[fault_surface_direct],Stage-I*'
python3 benchmarks/reconstructed_fault/performance/run.py PARAMETER.prm --cap 120 \
  --env ASPECT_FAULT_SURFACE_SOLVER=tridiagonal \
  --env ASPECT_FAULT_COMPARE_SURFACE_INVERSE=1 \
  --env ASPECT_FAULT_LINEAR_PERFORMANCE=1
```

The exact resource manifests record 900 s caps for BP3, 180 s for evolving K3
and the short K1/K2 checks, MPI rank counts and additional diagnostics. No timed-
out trajectory was retried. Test logs and earlier evidence remain intact.

Changed implementation boundaries: `linear_performance.h` supplies the small
exclusive timing scopes; solver.cc, the condensed operator and B/G entries
instrument existing work. `surface_direct_internal.h` contains the LAPACK/
UMFPACK implementation, and `surface_system.cc` replaces duplicated inverse
storage/checking with that helper. The semantic public surface API is unchanged.
Unit/production action tests, performance wrappers/report scripts and the K3
wrapper-name guard registration provide the focused verification. Design notes
explicitly permit the approved pivoted prototype. Prior accepted I_h/BP3 work
and unrelated working-tree changes are preserved. No commit was requested.

**Next decision:** review `stage_K5_interface_preconditioner_proposal.md`.
It proposes a few-mode, solver-side approximation to the interface Schur
feedback, using existing semantic actions and outer FGMRES, with a setup-cost
break-even test. It explicitly rejects a dense per-vertex probing campaign and
does not alter physical pressure or active-set equations. No preconditioner
implementation, I_h repair, propagation or fine BP3 pilot is included here.
