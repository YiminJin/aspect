# Restricted native P2 MPI verification (tasks 1–3)

This report covers the original 16×512 fixture on one, two and four MPI ranks.
All nine matched controls pass: fixed operators, a complete retained growth
entry and noncommitting saddle escape at one, two and four ranks. Four-rank
failure/retry restores accepted state exactly and publishes once. MPI reduces
memory per rank, but the growing step takes 11.1/14.2/12.0 minutes, so substantial
parallel speedup is not established. No smaller mesh or fresh .625-s trajectory
is run under the latest scope. Physics, energy, bounds and final acceptance are
unchanged. The committed serial results remain reproducible.

## Fixed model and numerical settings

| Quantity | Setting |
|---|---|
| Domain / prescribed fault | [0,1]² m²; y=.5 m, x∈[0,1] m; fixed full-normal association |
| Bulk / surface | 16×512 cells, Q2 velocity/Q1 pressure/Q1 phase; 64 Q1 surface segments |
| Integration | Six normal Gauss points; common surface intersections and three tangential points; native CPDI fracture |
| Particles | 73728, 3×3 per cell; fixed positions/domains; no transport or birth |
| Material | G=10⁶ Pa, η=10⁸ Pa s, λ=100 s, c=1000 Pa, Gc=26.6666666667 J/m², ell=.15625 m, p=1 |
| Viscosity bounds | 10⁶–10¹⁰ Pa s |
| Scoped entries | Growing core: t=.53125 s, dt=1/1024 s; resolved saddle: t=.5654296875 s, dt=1/256 s |
| Calibration | c0=8/3, Hc=.5 Pa, m=128.00000000016, Gc/(c0 ell)=64.00000000008 Pa |
| Friction / pressure | Fixed coefficient .1, prescribed background normal pressure 10000 Pa; radiation damping zero |
| Initialization | Shared continuum nominal-core-.6 seed; retained seeded particle H; unloaded accepted FE d_R and surface Z |
| Loading / boundaries | Interval mean of ux=.256 t(y−.5); uy=0; compatible sides with natural ux; natural phase boundaries |
| Histories / bounds | Immutable incoming accepted d_R, Z and previous physical phase; lower obstacle unchanged through trials; publish once |
| Solver | Phase nonlinear tolerance 10⁻¹², intermediate linear tolerance 2×10⁻⁷; mechanical nonlinear 10⁻⁸, linear 10⁻⁷; original true-energy allowance |
| MPI | One thread/rank; ranks bound to distinct physical cores; cases sequential; same executable/observer throughout matched suite |

G remains a dimensional synthetic verification value. This is one prescribed
through-going fault with predefined association throughout the normal section;
it does not qualify transport, birth, general support admission or propagation.

## Preserved reference and provenance

Baseline is 33c0adba59132890b386f3e984b3e1650feffe44 on fault-propagation.
The incoming tracked tree was clean. Historical raw evidence and manual work
are preserved; no reset, cleanup, checkpoint conversion or new worktree is used.
Serial executable and observer copies reside in
build-propagation-integration/serial-reference. baseline.json records their
hashes and 1505 tracked-file hashes. verification-build.json records the final
dirty source and executable hashes; production.patch records the tracked
production diff. Every run has input/executable/observer/profiler hashes,
command, rank count, output path, return code and bounded runtime.

Compiler/dependencies: GCC12.4, deal.II9.6.2, Trilinos14.2, p4est2.8.7;
existing release build and installed MPI/Amesos infrastructure. No new solver
dependency or production setting is introduced. Defaults remain unchanged;
MPI is enabled only for the explicitly selected consistent-endpoint FE option.
The earlier sequential FE option remains restricted to serial operation.
The matched MPI executable is preserved as
build-propagation-integration/aspect-mpi-verified-release, SHA256
e9c8b1423d21a4e2a2903a579eb9703e4b3d54231cfdb0e17bee2ad320d55fb2.
The observer is 8d0060654edb260326357fb59a6b91cc43e99b8793b70e64a30f193d9f03858e;
the PMPI shim is e2c23bcf11be5033f66a2aa1262d26bcb37d53412924401b786b80c8a3ed14eb.

## Ownership and publication audit

Accepted FE histories stay distributed with unique owners and refreshed ghosts.
Only phase search coefficients and immutable normal-ray geometry are replicated.
Owned bulk cells and particles assemble once. Global bulk derivatives are
reduced before the replicated cohesive term and normalization pullback are added
once. The same consistent projection and its first/second derivatives are used.
CPDI energy and bulk energy use extended local and collective accumulation.
Global mass norms, active sets, phase coordinates and convergence decisions use
complete owned contributions. No rank-local phase index is read as a global one.

Existing collective Amesos_Klu factors are retained. Their root gather/factor
bottleneck is labelled and timed; they are not a distributed factorization.
Both successful publication and failure handling retain one incoming anchor.
Particle-local FE publication validation is coordinated before any history write;
all ranks reject together. The controlled prepublication failure is injected on
only the last rank, after coupled convergence, then restoration is checked and
the identical complete step retried. Full surface property bytes, particle
properties, all phase/FE histories, rates and bookkeeping are checked.

## Retained failures and preliminary evidence

Two porting defects were exposed and repaired, with the failed runs preserved:
operators-n16-r2-a selected an artificial/coarser cell for uniform mesh metadata;
operators-n16-r2-b added a ghosted vector to an owned vector and encountered an
Epetra map error. Metadata now comes from an owned cell and FE recurrence
arithmetic uses matching owned maps before ghost refresh. No projection is added.

Before the final matched suite, all 15 fixed states passed at each rank count.
The four-rank loaded core-6 recovery control completed in 1444.54 s including
failure and retry. It published one accepted increment .0015737832, zero healing,
KKT 2.4528×10⁻¹³ J/m at the original 2.6849×10⁻¹¹ target, positive I≥111.13338 m,
and a phase difference 1.81×10⁻¹⁰ from its retained prototype. This is a scoped
complete-entry test, not a fresh physical trajectory from time zero. The old
coarse-prefix cumulative energy failure retains its failed qualification status.

## Matched comparison and cost

The final fixed-operator suite passes on the same executable/observer build:

| Ranks | Whole suite (s) | Peak MiB per rank | Largest energy difference from one rank (J/m) |
|---|---:|---|---:|
| 1 | 133.13 | 1629.2 | 0 |
| 2 | 104.85 | 1036.3 / 965.0 | 3.55×10⁻¹⁵ |
| 4 | 79.24 | 745.9 / 620.0 / 618.4 / 569.8 | 3.55×10⁻¹⁵ |

The minimum original line-search allowance is 4.5475×10⁻¹³ J/m. Measured rank
energy uncertainty is 0.78% of that allowance. Collapsed gradient differences
are ≤2.51×10⁻¹⁵ in the benchmark scale. All 15 unloaded/growing/asymmetric
states, normal/prototype and tangential actions, normalization adjoints,
CPDI quadratic reuse and predictor/history poisoning checks pass. Sampled
physically normalized tangential curvatures agree across ranks. These fixed-state
results establish operator equivalence, not full angular stability.

The complete growing entry passes at all three rank counts, with zero healing
and positive I≥111.13338 m. KKT residuals are 2.88/2.97/2.45×10⁻¹³ J/m at the
original 2.6849×10⁻¹¹ target. Largest rank differences are phase 9.68×10⁻¹²,
traction 7.24×10⁻¹¹ Pa, slip 2.97×10⁻¹⁵ m and independently sampled shear
stress 1.63×10⁻⁸ Pa. All original bounds, equilibrium and publication checks pass.
Four-rank failure on rank 3 after convergence restores every tested history and
bookkeeping value, including complete surface property bytes; retry publishes once.

| Ranks | Successful complete step (s) | Tangent actions / inner Krylov | Tangent time (s) | Speedup vs 1 rank | Peak MiB per rank |
|---|---:|---|---:|---:|---|
| 1 | 664.10 | 135 / 10660 | 384.19 | 1.00 | 1842.0 |
| 2 | 849.90 | 136 / 10775 | 644.76 | .78 | 1193.3 / 1071.5 |
| 4 | 717.18 | 140 / 11194 | 575.32 | .93 | 850.4 / 702.5 / 701.7 / 637.4 |

The four-rank failed attempt costs another 713.10 s; the whole recovery run
is 1453.24 s including setup/postprocessing. Successful retry timing is separated.
Its homogeneous factor can remain cached from the failed attempt, an advantage
of <1 s; this does not explain the ≈53-s slowdown relative to one rank.
All matched runs use the same executable and observer. Krylov totals differ by
1.1%/5.0%; action time rises 68%/50%, so iteration count does not explain most
of the slowdown. Four ranks speed mechanical trial evaluation but do not make
this complete growing step faster. This task does not optimize the bottleneck.

The growing-step scopes explain where the cost changes. The tangent total
contains its inner solve/preconditioner scopes; they are not additive timings.
Assembly of the phase-dependent coupling maps is included in mechanical trial
evaluation. The existing native TimerOutput summary also records the following
assembly/solve scopes, at its original rounded precision. Unlike the successful
entry table, the four-rank summary includes both the forced failure and retry.

| Ranks | Stokes RHS assembly (s / calls) | Bulk matrix assembly (s / calls) | Bulk preconditioner construction (s / calls) | FE tangent pullback (s / calls) | Condensed linear solve (s / calls) |
|---|---|---|---|---|---|
| 1 | 114 / 72 | 3.91 / 2 | 1.88 / 2 | 90.9 / 467 | 295 / 353 |
| 2 | 71.1 / 72 | 2.53 / 2 | 1.26 / 2 | 70.9 / 470 | 563 / 355 |
| 4, failure + retry | 70.8 / 137 | 2.04 / 3 | 1.02 / 3 | 94.3 / 968 | 1030 / 724 |

These scopes include setup and nested operations; their sum is not whole-step
wall time. The successful step's finer inner scopes are:

| Ranks | Mechanical trial evaluation (s) | Phase iterations / searches | Inner solves | Inner preconditioner (s / calls) | Other inner work (s) |
|---|---:|---|---:|---|---:|
| 1 | 225.06 | 7 / 9 | 332 | 149.11 / 10660 | 130.72 |
| 2 | 147.24 | 7 / 9 | 334 | 157.95 / 10775 | 387.60 |
| 4 | 85.47 | 7 / 9 | 344 | 167.91 / 11194 | 328.54 |

The successful two-rank entry spends 316/255 s per rank inside instrumented MPI
calls. Rank 0 performs 5.81 million all-reductions (298.84 s); rank 1 spends
131.25 s in 3.55 million barriers. Four-rank retry totals are 263/386/354/361 s.
These inclusive times include waits and single-rank communicator calls, overlap
solver scopes, and cannot be added as separate energy or wall-time contributions.
They identify expensive collective traffic within the coupled search and linear
solver path; this instrumentation does not attribute every MPI call to a caller.
The existing root sparse factors remain a labelled limitation; no new solver
or changed intermediate algorithm is introduced here.

All three saddle diagnostics depart to the same reference branch, with no
reflection or recentering of any field. Native potential decreases by
8.8141×10⁻⁹ J/m, far above the 4.5475×10⁻¹³ allowance. Physical KKT is
1.8893/1.7029/5.6942×10⁻¹² J/m at the unchanged 3.4813×10⁻¹¹ target.
Healing is zero, I≥149.51177 m and maximum trial phase change is .00618032.
Centroids agree within 1.88×10⁻¹² m at .50012622415 m. Largest unaligned rank
differences are phase 8.02×10⁻¹², traction 1.61×10⁻⁹ Pa, slip 4.64×10⁻¹⁴ m
and independently sampled shear stress 2.09×10⁻⁸ Pa. Native/prototype phase
error is ≤5.15×10⁻¹¹. Tangential phase variation is ≤8.63×10⁻¹³; the native
tangential degrees of freedom remain active.

The known admissible mode changes from L² curvature −.28020 Pa to +15.99224 Pa
at every rank count. Independent action uncertainty is ≤1.31×10⁻⁶ J/m in the
reported pairing, far smaller than the resolved sign. Only sampled mode
stability is established; a full angular spectrum is not qualified. The saddle
test is noncommitting and does not justify a physical jump transition.

| Ranks | Entry solve / whole run (s) | Actions / inner Krylov | Phase iterations / searches | Mean tangent action (s) | Peak MiB per rank |
|---|---|---|---|---:|---|
| 1 | 1655.61 / 1703.57 | 363 / 34141 | 9 / 9 | 3.32 | 1840.2 |
| 2 | 2189.84 / 2222.50 | 342 / 31625 | 9 / 10 | 5.40 | 1166.2 / 1072.7 |
| 4 | 1466.20 / 1487.86 | 260 / 23776 | 7 / 9 | 4.82 | 777.5 / 696.6 / 679.7 / 630.2 |

Four ranks solve this saddle entry 13% faster, with 28% fewer actions. The
per-action time rises 45%; the whole-entry improvement is partly a different
Newton route, not a clean parallel scaling result. Both two- and four-rank
growth actions are slower than serial actions. This does not support an
overnight full-run claim.

Rank comparisons use physical coordinates, not DoF
numbering. Native bulk/contact residuals and original KKT are measured separately.
Energy reduction uncertainty is measured between ranks and checked against the
original 32 eps max(1,|J/scale|) allowance; the common native/prototype fracture
integration offset is not labelled MPI error. No tolerance is enlarged.

The raw MPI log's legacy “bulk old-new” print compares
a local ordinary subtotal with a global extended total; its large offset is
not reduction uncertainty or physical energy imbalance. Only the globally
comparable stored energies above are used for the uncertainty check. A
label-only correction makes this distinction explicit after the matched suite,
without changing its verified operators. The timed executable above is retained;
the subsequent build and four-rank operator smoke check are recorded separately.
That final four-rank check passes in 82.30 s; all 15 operator rows and every
compared gradient value are identical to the matched four-rank result. Final
executable SHA256 is 8fba047bd33d212ec4c007b2fa5d287362634f60d1b41b236012c9b65bdd2f83.
The unchanged observer is reused. Compatibility regressions pass all 21105
assertions in 54 cases; script syntax and tracked whitespace checks pass.

Benchmark-only PMPI interception records call counts and inclusive time spent
inside collectives/point-to-point waits, including waiting for other ranks.
These times overlap assembly/solve scopes and must not be added to them.
Per-entry epochs separate initialization, controlled failure, successful retry
and postprocessing. Per-rank peak resident memory is retained. Timing does not
change messages, operators, solver settings or physical histories.

## Source changes and purpose

| File / function | Purpose / resulting behavior |
|---|---|
| include/aspect/reconstructed_fault/endpoint_mpi.h | Owned-entry gather/assign and extended energy reduction for the restricted endpoint path |
| fe_strain.cc: initialize_fe_shared_phase, prepare_fe_strain_history, mechanical_quadrature | Owner-only initialization, correct owned-cell metadata, one-time global normal rays, owned bulk quadrature and visible-cell stress lookup |
| fe_strain.cc: make_fe_endpoint_trial, prepare_fe_strain_commit | Exact Maxwell recurrence in matching owned maps followed by ghost refresh; no particle projection |
| fe_strain.cc: evaluate_fe_phase_energy_at, evaluate_fe_endpoint_action | Once-only distributed energy/gradient/action and normalization adjoint; dense Hessian remains a serial diagnostic |
| fe_strain.cc: validate_fe_strain_restrictions | Global fixed-particle restriction before collective trial evaluation |
| history.cc: validate_history_candidates, commit_reconstructed_fault_mechanical_history | Collective owner-ID and FE-sample validation before terminal publication; globally reported publication uncertainty |
| phase_field.cc: assemble_phase_field_system | Owned CPDI contributions, extended energy reduction, owner-only insertion of global elastic gradient |
| reconstructed_fault_endpoint_phase.cc: solve_reconstructed_fault_endpoint_phase | Distributed sparse search, global active/mode/KKT decisions and coordinated prepublication rejection; actual Hessian and energy unchanged |
| reconstructed_fault_stokes.cc | Label existing collective Amesos velocity factor and its root bottleneck |
| phase_field_fault.cc: parsing/declaration | Permit MPI only for the selected endpoint FE path; all defaults and other restrictions retained |
| native_endpoint_verification.cc | MPI-safe coordinate comparisons, compact independent stress, complete-entry publication/recovery, tangent/curvature and per-rank memory checks |
| run_native_mpi.py / prepare_native_mpi.py / assess_native_mpi.py | Preserved scoped fixtures; bound-core sequential run/provenance; coordinate and cost assessment |
| plot_native_mpi.py | Compact unaligned phase/localization/stress profiles and rank-cost charts |
| native_mpi_profile.cc | Benchmark-only PMPI counts/timing; no solver dependency or production behavior change |
| AGENTS.md / design / specification / STATUS / PLAN | Current stop-after-task-3 scope, ownership contract and measured qualification level |

## P2 closure limits

The deliverable stops at MPI correctness and rank comparison on the original
mesh. Fixed states and complete retained entries cannot establish a full native
trajectory, native temporal order, angular stability or physical jump selection.
The .625-s fresh trajectory, tangential-mesh study, targeted optimization,
coupled restart, postpublication rejection and transported/birth histories
remain unqualified. No P3 or material tuning is undertaken.

| Check | Result / qualification |
|---|---|
| Baseline/manual-work preservation | 1505 baseline files present, no unexpected changes; both saved serial binaries match |
| MPI operators and numerical uncertainty | 15 fixed states at each of 1/2/4 ranks; original bounds, adjoints, residuals and action checks pass |
| Complete accepted growth / history ownership | Same retained core-6 entry at 1/2/4 ranks; actual solve/publication, exact FE recurrence, positive I and zero healing |
| Four-rank failure / retry | One rank rejects after convergence, before publication; all tested bytes/bookkeeping restored, retry publishes once |
| Known saddle departure | Same stable reference branch at all ranks; energy decreases and known-mode curvature becomes positive; noncommitting diagnostic |
| Compatibility / final source | 54 cases, 21105 assertions; label-only four-rank operators/gradients unchanged |
| Parallel performance | Measured; growth has no speedup, saddle's modest speedup includes fewer iterations; inner collective traffic remains costly |
| Smaller mesh / fresh .625-s trajectory | Deferred by stop-after-task-3 instruction |
| Broader P2/P3 | Native temporal order, full angular stability, physical jump selection, coupled restart and transport/birth remain unqualified |

[Growth profiles](../../../benchmarks/reconstructed_fault/propagation/native_mpi_evidence/growth-rank-profiles.png),
[noncommitting saddle profiles](../../../benchmarks/reconstructed_fault/propagation/native_mpi_evidence/saddle-rank-profiles.png),
[rank cost and action counts](../../../benchmarks/reconstructed_fault/propagation/native_mpi_evidence/rank-cost.png).

## Recorded commands and reuse

The one-shot matched suite ran sequentially with the existing release build
and observer. These commands are recorded in the evidence; suite tags already
exist and are protected from overwriting:

```sh
python3 benchmarks/reconstructed_fault/propagation/run_native_mpi.py --task3-suite
python3 benchmarks/reconstructed_fault/propagation/assess_native_mpi.py --operators operators-final-r1 operators-final-r2 operators-final-r4 --growth core-final-r1 core-final-r2 core-final-r4 --saddle saddle-final-r1 saddle-final-r2 saddle-final-r4
MPLCONFIGDIR=/tmp/p2-native-mpl python3 benchmarks/reconstructed_fault/propagation/plot_native_mpi.py
build-propagation-integration/aspect-release --test '*Fault*,*fault*,Stage-I*'
```

For a repeat, select an existing operators/core/saddle input with a new unique
`--tag`, `--ranks` and `--profile`; use `--failure-step 1` for the loaded recovery
control. The runner preserves old outputs and enforces its local wall cap.
The final label-only build does not change a numerical operator. Long growth
and saddle evidence is reused on that basis; it is not silently relabelled as
new timings. No extra native trajectory is launched.
