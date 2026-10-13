# P2 checkpoint and bounded closeout — 2026-10-13

Retain the **consistent-endpoint reduced-Newton solver with FGMRES** as the P2
reference. Native operators, accepted core growth, a retained saddle departure,
and coordinated prepublication failure/retry have been verified. The complete
0–0.625 s trajectories are **prototype results**; fresh native evolution reaches
0.53125 s. This closes the selected implementation/diagnostic work, not full P2
physical qualification or readiness for propagation. No new experiments were
run for this closeout.

## Fixed verification model

| Setting | Reference value / units |
|---|---|
| Geometry / associations | Unit square, m; through-going fault at y=0.5 m; fixed full-normal association |
| Discretization | 16×512 cells; Q2 velocity, Q1 pressure/phase; 64 Q1 surface segments; six normal bulk points; CPDI fracture |
| Histories / particles | Accepted FE d_R and surface elastic-slip Z; 73,728 fixed particles; no transport or birth |
| Material | G=10⁶ Pa, eta=10⁸ Pa s, lambda=100 s; c=1000 Pa, Gc=26.6666666667 J/m², ell=0.15625 m, p=1 |
| Friction / loading | Fixed mu=0.1, prescribed normal pressure 10,000 Pa; zero radiation/body force; interval mean of ux=0.256 t(y−0.5), uy=0 |
| Boundaries / initialization | Compatible lateral conditions; natural phase boundary; shared relaxed nominal-core-0.6 seed; zero initial FE stress, Z and slip |
| Phase / normalization | Immutable previous physical phase ≤ trial phase ≤ 1; consistent I_h and its actual derivatives/adjoint |
| Solvers | Reduced Newton, true-energy search, original KKT checks; FGMRES mechanics; verified fixtures enable adaptive intermediate accuracy and translation preconditioning |

G=10⁶ Pa is a **dimensional synthetic verification value**. The optional FE
endpoint path remains default off and restricted to homogeneous coefficients,
fixed mesh/geometry and fresh histories. Every trial starts from the same
accepted d_R, Z and physical phase obstacle; the complete state is published
once. Predictors and intermediate iterates are not accepted history. The legacy
particle-authoritative path and its defaults remain available.

## Evidence and qualification limits

| Evidence | Established result | Limit |
|---|---|---|
| Native fixed-state operators | Unloaded, growing and asymmetric states: energy, normalization derivatives/adjoint, phase gradient, mechanical/contact residuals and reduced-tangent actions agree with retained FD/prototype checks; predictor/history poisoning passes | Sampled actions and curvatures, not exhaustive stability or spatial convergence |
| Fresh native onset | Energy-qualified 58-step prefix reaches 0.53125 s; core phase 0.60000194→0.60457720, zero healing; all seven matching-prototype field targets and separate energy gates pass; 85 min 9 s, 1684.9 MiB peak RSS | Does not reach 0.625 s or independently establish native temporal order |
| Retained native growth/saddle | Complete growth entry publishes once; noncommitting saddle diagnostic escapes to the matching stable branch with admissible energy decrease and original KKT | Imported immutable entries are not fresh trajectory continuation or a general jump-selection rule |
| MPI agreement | Original 16×512 operator/growth/saddle controls pass at 1/2/4 ranks; maximum rank phase differences ≲10⁻¹¹; energy uncertainty 3.55×10⁻¹⁵ J/m, 0.78% of the minimum original line-search allowance | Complete growth costs 664/850/717 s; substantial MPI speedup is not demonstrated |
| Outer-step recovery | Four-rank failure on rank 3 after convergence and before publication restores tested accepted histories, surface/particle data and bookkeeping exactly; retry publishes once without deadlock | Prepublication exception recovery only; not coupled checkpoint/restart or all rejected-step/transport paths |
| Full prototype trajectories | Matching 512-cell Q2 trajectories at dt=1/1024 and 1/2048 s reach 0.625 s and pass all seven unaligned temporal targets; separate mechanical, friction and phase residual works retained | Restricted normal reduction; additional native tangential modes and positional selection remain limitations |

The reference temporal targets are phase 0.001, traction 1 Pa, slip 0.1%,
common-interval rate 0.5%, and stress/storage 1%, using the retained fixed scales.
Prototype targets do not transfer automatically to a full native evolution.
Analysis-only centroid alignment never changes accepted histories or the
obstacle. Cross-mesh positional differences and localization remain unresolved.

Evidence: [native operators/reference](P2_native_endpoint_report.md),
[analytic tangent and recovery](P2_native_tangent_report.md),
[fresh onset and saddle](P2_native_solver_report.md), and
[matched MPI checks](P2_native_MPI_report.md). Earlier reports retain their
historical stopping points; the later accepted onset/saddle evidence supersedes
those particular missing tests, not their failed trials or energy limitations.

## Experimental solvers remain separate

| Experiment | Result / disposition |
|---|---|
| Phase-first single pass | Cheaper prototype paths, but fine phase error 0.05994 and traction error 36.84 Pa fail targets. Converged incoming-history phase KKT does not remove the endpoint splitting defect. Not adopted. |
| Four ordinary staggered correctors | Growth KKT fails; saddle has a small residual but negative curvature. Unsuccessful and inactive. |
| Quasi-Newton search | Full prototype comparisons pass; native growth screen stopped after 36 min 49 s before acceptance/publication. No native cost/branch qualification. Default false and inactive. |
| Direct mechanics | Experimental **serial**, default false, interchangeable with FGMRES. Actual independent bulk/surface blocks, pressure treatment and full residual/tangent checks pass; original outer reduced Newton retained. No direct MPI or trajectory qualification. |

Direct cost data: eight-RHS setup-inclusive batch 10.51→6.29 s; bulk construction
plus KLU 2.291 s; 65-node surface setup 2.153 s (64-node active control 2.122 s);
reused solve median 0.03429 s including residual checks. Batch peak RSS is
1789.6 MiB, retained increment 102.2 MiB. Pure factorization-only time and
standalone growth/saddle peaks were not recorded. Complete retained growth is
374.11 s versus 664.10 s with identical outer counts; saddle is 615.71 s versus
1655.61 s with fewer actions. The latter comparisons use retained instrumented
FGMRES timings. Growth reuses one bulk/18 surface setups across 282 RHS; saddle
one/25 across 649 RHS. Phase-dependent factors refresh with their operators.
No tests were repeated to fill missing metrics.

Reports: [single pass](P2_single_pass_report.md),
[cheaper coupling](P2_cheaper_coupling_report.md),
[direct mechanics](P2_direct_mechanical_report.md). The original equality-only,
high-load, history-law and coarse-prefix energy failures remain preserved.
Storage, viscous/frictional losses, phase-relaxation release and separate
mechanical/contact/phase residual works are not combined into a compensating
dissipation. Native conclusions remain separate from numerical pass counts.

## Checkpoint and deferred work

The checkpoint starts from branch `fault-propagation`, commit
`33c0adba59132890b386f3e984b3e1650feffe44`, in the existing worktree. The incoming
patch, source/document archive, full file hash manifest and saved serial/MPI
binaries are preserved locally. The committed
[checkpoint index](../../../benchmarks/reconstructed_fault/propagation/p2_checkpoint/README.md)
identifies source/build provenance, canonical reference options and curated
inputs, summaries and compact evidence. Source behavior was not changed for
closeout. The last retained compatibility result is 21,112 assertions in
55 cases; this task performs integrity checks only.

Deferred: fresh full native evolution to 0.625 s; native temporal/spatial and
localization qualification; comprehensive admissible stability and positional
selection; coupled restart and rejected-step coverage beyond the tested failure;
transport/replenishment and general associations; evolving RSF; scaling and
performance optimization; P3 and fault propagation. The prepared full-run input
is a plan, not evidence of execution. Reduced Newton/FGMRES remains authoritative;
no further quasi-Newton, arc-length, full-trajectory or scalability work is
initiated by this checkpoint.
