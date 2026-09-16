# K5 explicit coupling and bounded interface-preconditioner result

## Decision summary

Explicit sparse B and G preserve the existing discrete actions. B gives the
substantial improvement; G only narrowly recovers setup at these reuse counts.
The corrected two-mode interface preconditioner does not reduce Krylov work
enough to recover its setup and extra applications, so it is not promoted.
The combined preconditioner+B+G cost improves by an estimated 1.33x in BP3 and
1.53x in evolving K3 with sparse coupling, short of the roughly 2x target.
These are same-run component-accounting estimates, not native wall-speedup
claims. Stokes preconditioning remains the dominant cost.

One-/two-rank action, derivative, recovery, convergence and rollback checks pass.
The corrected BP3 run completes initialization and three real steps with 29
genuinely accepted fresh linear checks; evolving K3 passes all nine state guards.
No equations, quadrature, history semantics, support, pressure treatment,
FGMRES, iteration budgets or tolerances changed. An initial prototype-only
pressure-destination bug was corrected; its failed attempts remain preserved.
No fine pilot or GMG implementation was launched. The next decision is review
of the separate GMG-preconditioner proposal, not more surface-inverse work.

## Frozen-operator invariants and selection

`stage_K5_coupling_addendum.md` records the approved mathematical construction.
B uses the existing bulk quadrature and frozen 2 kappa chi S. Its test rows are
expanded through homogeneous constraints before MPI ADD. G uses the existing
parent-P0 bulk samples and integrated domain test/coefficient weights, including
normal-stress and pressure terms. RPE ownership and shared-face averaging are
preserved when constructing its sparse rows. Missing requests have the original
zero contribution. **G is not B transpose.** Sparse storage expires with the
linearization; geometry, coefficients and active set are not changed in Krylov.

Original `apply_B_reference` and `apply_G_reference` remain independently callable.
`ASPECT_FAULT_EXPLICIT_B=1` / `ASPECT_FAULT_EXPLICIT_G=1` select the prototypes;
`ASPECT_FAULT_COMPARE_COUPLING=1` checks every application at relative 2e-11,
including production directions, pressure tests and recovery. Leave these
presence-based flags unset to disable them (setting `=0` does not disable).
Sparse B is recommended at the measured reuse counts. Sparse G is conditional
on sufficient reuse, not a universally cheaper default. Both remain opt-in.

`ASPECT_FAULT_INTERFACE_MODES=2` selects the tested constant/cosine correction.
The implementation permits 1--4 modes per contiguous free block, capped at
eight overall, but the other mode choices are **not performance-qualified**.
Active/prescribed rows are zero. With Y=P_A BQ, the small pivoted LU solves
T=Q^T K Q-Q^T GY; application is P_A r+Y T^-1 Q^T G P_A r. Singular T disables
this optional correction. Pressure-complement handling surrounds the same base
Stokes preconditioner. Outer FGMRES and the semantic indefinite surface inverse
remain unchanged. `ASPECT_FAULT_VERIFY_INTERFACE=1` verifies zero pressure in
setup responses. This correction remains experimental and off by default.

The already qualified pivoted tridiagonal inverse is now the preferred default;
`ASPECT_FAULT_SURFACE_SOLVER=umfpack` retains the independent alternative.

## Sparse coupling timing and memory

Times below are seconds, summed over one-rank short trajectories. BP3 has 29
returned directions/linearizations; evolving K3 has 37. Each verification run
executes both sparse and original actions on the same vectors.

| Quantity | BP3 | Evolving K3 |
|---|---:|---:|
| B/G applications, each | 560 | 669 |
| B setup | 17.6382 | 1.89935 |
| B sparse applications | 0.79857 | 0.10034 |
| Original B applications | 79.9187 | 10.5335 |
| B mean break-even applications per linearization | 4.30 | 3.29 |
| G setup | 24.9011 | 3.35951 |
| G sparse applications | 2.88208 | 0.27457 |
| Original G applications | 31.3580 | 4.05757 |
| G mean break-even applications per linearization | 16.89 | 16.06 |
| Base Stokes preconditioner applications | 502 | 558 |
| Base Stokes preconditioner time | 150.684 | 11.3880 |
| Original-action preconditioner+B+G | 261.961 | 25.9790 |
| Sparse setup+preconditioner+B+G | 196.904 | 17.0217 |
| Estimated combined reduction | 1.33x | 1.53x |
| Peak retained B arrays, bytes | 15,728,736 | 1,179,744 |
| Peak retained G arrays, bytes | 12,607,584 | 1,573,728 |
| Peak B entries | 604,704 | 51,842 |
| Peak G entries | 752,899 | 74,690 |

B setup includes the whole existing B coefficient linearization, making its
break-even accounting conservative. Break-even is setup per linearization
divided by mean original-minus-sparse application time. Memory is rank-zero
retained array capacity, not a distributed peak; process RSS also includes
temporary assembly maps and the rest of ASPECT.

## Corrected few-mode comparison

| Metric | Sparse BP3 | +2 modes BP3 | Sparse K3 | +2 modes K3 |
|---|---:|---:|---:|---:|
| Returned directions | 29 | 29 | 37 | 37 |
| Total outer iterations | 502 | 502 | 558 | 559 |
| Mean iterations | 17.31 | 17.31 | 15.08 | 15.11 |
| Min / median / p90 / max | 16/17/18/19 | 16/17/18/19 | 10/14/19/25 | 10/15/18/25 |
| Total base preconditioner calls | 502 | 560 | 558 | 633 |
| Of which setup probes | 0 | 58 | 0 | 74 |
| B sparse calls | 560 | 618 | 669 | 744 |
| G sparse calls | 560 | 1120 | 669 | 1303 |
| Base preconditioner seconds | 150.684 | 182.710 | 11.3880 | 11.0908 |
| B setup / matvec seconds | 17.638/.799 | 17.637/1.098 | 1.899/.100 | 1.786/.099 |
| G setup / matvec seconds | 24.901/2.882 | 24.729/6.349 | 3.360/.275 | 3.194/.499 |
| Interface exclusive setup / apply overhead | 0/0 | .915/7.355 | 0/0 | .071/.472 |
| Interface inclusive setup seconds | 0 | 21.977 | 0 | 1.688 |
| Dominant cost including new setup, excluding reference callbacks | 196.904 | 240.793 | 17.0217 | 17.2117 |
| Additional retained bulk responses, bytes | 0 | 5,433,040 | 0 | 602,672 |

BP3's corrected iteration list is exactly the sparse baseline's. The correction
adds work without reducing those iterations. K3 also shows no material benefit.
This rejects this bounded mode set, not every possible interface preconditioner.
Do not tune additional modes or construct a full dense fault Schur complement
as an automatic continuation.

### Disjoint timing and wall reconciliation

`ASPECT_FAULT_LINEAR_PERFORMANCE=1` records exclusive nested timings. The
inclusive interface setup line contains child preconditioner/B/G calls and
must **not** be added again to the exclusive totals. FGMRES-vector time excludes
nested operator/preconditioner work. Remaining coupled work is explicitly
reported as `other`, not silently assigned to Arnoldi.

| Run | Timed coupled work | Wall | Wall outside coupled scope | Peak RSS KiB |
|---|---:|---:|---:|---:|
| coupling_bp3 (with reference callbacks) | 366.793 | 603.115 | 236.322 | 4,569,716 |
| coupling_k3 (with reference callbacks) | 38.4769 | 92.6677 | 54.1908 | 540,860 |
| interface_zero_bp3 (native corrected modes) | 299.283 | 561.090 | 261.806 | 4,568,700 |
| interface_zero_k3 (native corrected modes) | 23.0659 | 73.8246 | 50.7587 | 515,056 |

For corrected BP3, the coupled total includes A=6.596, FGMRES vectors=4.515,
other=47.371, factorization=.00148 and surface inverse=.00739 seconds, besides
the exclusive entries above. Corrected K3 has A=.83546, FGMRES vectors=.34563,
other=4.66944, factorization=.00058 and inverse=.00302 seconds. Totals reconcile
by construction; wall outside the scope includes preparation, histories,
output and startup, not unmeasured Krylov work.

Removing timed reference callbacks gives 255.516/23.8858 seconds for sparse-only
coupled work. This is accounting, **not a native/native timing experiment**:
reference callbacks affect cache behavior and wall time. Normal diagnostics
also differ between BP3 runs. No wall-speedup claim is based on these unequal
flags. Prior native original-action runs (334.338/32.8091 coupled seconds,
663.326/88.4378 wall seconds) are retained as context, not repeated for cosmetics.

## Correctness and the preserved failed prototype

The initial mode implementation copied the nonlinear RHS as a layout template
for Y. The existing pressure inverse's zero-RHS fast branch leaves its
destination untouched. Consequently P_A BQ incorrectly retained that copied
pressure, although BQ has exactly zero pressure RHS. This was a **prototype
caller bug**, not a change needed in equations or general Stokes code.

The first BP3 direction recorded estimated/fresh residuals of 1.33083/1.54197e11
at cumulative iteration 33, then .37838/61252.1 at 66, target 1.42748. Existing
fresh checks rejected both. Residual replacement returned a genuinely passing
direction at iteration 74, fresh .979459. These artifacts remain in
`interface_bp3`; they are not used to qualify the corrected preconditioner.

The correction initializes setup responses to zero and clears reused optional
preconditioner destinations before base application. The general Stokes solver
is unchanged. The corrected BP3 first direction takes 17 iterations, with no
failed fresh attempt anywhere in the run. One-/two-rank setup, positive
convergence and failure-after-accepted-update rollback tests were rerun.

Reporting scripts initially treated every failed *attempt* as a failed returned
direction. They now preserve such attempts and require a same-target,
strictly increasing cumulative iteration count ending in a genuinely passing
fresh residual. Incomplete failures still fail. Five isolated analyzer tests
cover this distinction; no numerical acceptance criterion was relaxed.

Every sparse BP3/K3 action comparison passes 2e-11. Maximum measured G errors
in sparse-only trajectories are 5.13e-14 and 4.72e-14. The first instrumentation
reported B maxima as zero; this is not claimed as bitwise agreement. Subsequent
BP3 comparison instrumentation measured B=1.03e-15 and G=3.47e-13 even during
the discarded large-direction prototype attempts. Native corrected runs do
not measure a reference error; their report records null, not a fabricated zero.

Corrected-mode versus sparse-only accepted trajectories:

| Maximum absolute difference | BP3 | K3 |
|---|---:|---:|
| V, m/s | 3.99e-23 | 1.12e-15 |
| Theta, s | 7.50e-8 | 5.88e-12 |
| C, Pa | 2.21e-9 | 5.63e-12 |
| I_h, m | 0 | 7.11e-14 |
| Accumulated slip, m | 1.16e-16 | not exported by this surface CSV |
| q, Pa | 2.24e-8 | see saved comparison |
| Phase | frozen | 2.22e-16 |
| Particle H | see saved comparison | 4.09e-13 |
| Particle tau_xy, Pa | see saved comparison | 1.42e-9 |

BP3 controller times differ by at most 1.034e-7 s (about 1e-14 relative), not
bitwise; K3 times/dts are identical. BP3's maximum raw weak nodal residual
difference is .001477 in weak-load units, not a strong Pa traction error.
Every final bulk/surface criterion passes. The corrected BP3 physical report
confirms four accepted states and 29 passing returned directions. All nine K3
guard files pass. BP3 has no K3-style guard JSONs: an empty guard list is not
evidence. The sparse-only BP3 run lacks the optional normal-stress CSV, so its
full physical analyzer cannot run; its preserved trajectory comparison and
fresh checks remain valid, with the corrected run providing the full physical
diagnostics. No rerun was made merely to fill that optional output gap.

## Focused verification and reproducibility

Artifacts live under `benchmarks/reconstructed_fault/performance/`.
The runner records exact command, environment, cap, executable/input/phase-source
hashes, status, wall and RSS in each `<run>.resources.json`. It does not overwrite
or retry runs. Relevant commands used the following pattern (not instructions
to repeat completed studies):

```sh
cmake --build build-pf-cpdi --target aspect.exe.release -j4
cmake --build benchmarks/reconstructed_fault/bp3/build --target bp3_coupling_checks.release -j4
build-pf-cpdi/aspect-release --test 'Stage-I*'
mpirun -np 2 build-pf-cpdi/aspect-release --test 'Stage-I*'
```

| Evidence | Result |
|---|---|
| coupling_random, coupling_random_two | Seeded random and basis B/G actions, FD, condensed action/recovery and restricted masks pass on one/two ranks |
| coupling_adiabatic_two | Prescribed-pressure action/condensed checks pass, 9.846 s |
| interface_zero_stage_i, interface_zero_stage_i_two | Corrected setup-pressure check and genuine nonlinear convergence pass |
| interface_zero_rollback, interface_zero_rollback_two | Failure after accepted update restores bulk/V/histories; pass, 6.331/7.783 s |
| coupling-stage-i-unit.log, coupling-stage-i-unit-two.log | 90 assertions in 11 cases per rank pass; bound, fraction, backtracking, exhaustion and rollback coverage retained |
| coupling-analysis-tests.log | Five reporting regression tests pass |
| coupling-bp3-result.json, coupling-k3-result.json | Reference actions and production trajectory comparisons pass |
| interface-zero-bp3-result.json, interface-zero-k3-result.json | Corrected native trajectory comparisons and fresh checks pass |
| interface_zero_bp3/perturbation_report.json | Requested initialization + three real steps complete; all final criteria pass |

Final verified executable SHA256:
`6471dd21a8a13e6493f05365b88da1b5531f4baa3ed43657ec091701e6519114`.
The phase material source hash remains
`36e86323bbd508ee25ca92ccb0034646255cb43ca319beb559bf03010cd930b6`.
Earlier comparison builds and their hashes are preserved per run. The final
prototype fix only changes destination initialization; B/G discretization did
not change after the seeded-random tests.

### Source changes and remaining scope

- `sparse_coupling.h`: small rank-local CSR storage of additive operator rows.
- Stokes assembler header/source: explicit B build/action and independent reference.
- Surface-system header/source: explicit G build/action and independent reference.
- Condensed-system header/source: semantic constrained/scaled B/G operations for
  solver-side setup, referencing canonical components.
- `reconstructed_fault_interface_preconditioner.h` and `solver.cc`: opt-in
  bounded mode correction, pressure-safe base calls, unchanged FGMRES lifecycle.
- `linear_performance.h`: exclusive coupling/setup scopes, counts and error maxima.
- `surface_direct_internal.h`: prefer already-qualified pivoted inverse only.
- `tests/phase_field_fault_surface_system.cc`: deterministic random/basis comparisons.
- Performance runner wrappers/report scripts and BP3 analyzer: reproducible
  experiments and correct failed-attempt versus returned-direction accounting.
- K3 checker: recognize the new output wrapper names without changing guards.
- Current design/specification, addendum, progress and this report: record scope,
  equations unchanged, tested outcome and the separate next proposal.

Changes remain uncommitted. The selected recoverable source snapshot and its
SHA256 manifest are `performance/coupling-source.tar.gz` and
`performance/coupling-source.sha256`, relative to the benchmark directory.
This is not a clean-tree claim: unrelated pre-existing changes and artifacts
are preserved. The prior mechanical snapshot remains available separately.

No full MPI BP3/K3 trajectory, general multi-fault explicit-matrix campaign,
other mode-count qualification, broad ASPECT suite, fine BP3 pilot, or GMG code
was run/implemented. Existing coarse BP3 scientific limitations are not erased
by a performance pass. The legacy I_h reference concern remains separate.
Review `stage_K5_gmg_followup_design.md` next: start by adapting the canonical
GMG **preconditioner** while retaining assembled A and qualified fine-grid B/G;
only consider matrix-free A after independent operator equivalence. No further
run is required to make this stage's performance decision.
