# Stationary mature DG optimization — 17 October 2026

Implemented and locally qualified against reviewed base
`b2dfdebe43437d6028b06d4f9f862cc80a00cc71`, with no tracked post-base changes at
entry. Incoming manual files, prior runs/checkpoints and original failures are
preserved. The source snapshot and its manifest, rather than HEAD alone, identify
this uncommitted implementation. No server job or full event was launched.

The final sequential two-rank comparisons reduce total local runtime from
30.888 to 18.305 s and 80.940 to 46.709 s: **1.69× and 1.73×**. Peak RSS increases
by 13.5–36.0 MiB/rank. This supports using the optimized selected route for the
prepared bounded server startup; it does not predict full-grid/event performance
or resolve the inherited accuracy failures.

## Scope and implementation

The bulk mesh, 10 m surface spacing, unlimited DGQ2, frozen physical phase and
geometry, Gauss6 mechanics/source3, Helmholtz20, full 3D plane-strain convention,
constitutive laws, native adaptive controls, solver tolerances and accepted-state
publication allowance are retained. Consistent endpoint phase solve stays off.
`Optimize stationary mature DG = true` enables the optimizations only for the
guarded homogeneous mature independent-DG route; false retains the original
algorithms. The legacy/default history backend is unchanged. All four selected
server inputs explicitly enable this option.

[Formulation and cache dependencies](BP3_DG_OPTIMIZATION_FORMULATION.md) give the
signs, work measure, constraints, point traces and invalidations. In particular,
at each native sample p, w_p=JxW_p chi_p,
s_p=2 eta_ve S:D3(symgrad v), and
n_p=p-2 eta_ve (n tensor n):D3(symgrad v). The exact filtered derivative is

```
S_Gamma x = sum_p w_p N_i(p) s_p(x)
N_Gamma x = sum_p w_p N_i(p) n_p(x)
H = M + ell_f^2 K
W_mu[i,j] = sum_p w_p mu_p N_i(p) N_j(p)
Gx = S_Gamma x - W_mu H^-1 N_Gamma x
C = A - B K_V^-1 G
```

The same RemotePointEvaluation incident-cell average and physical pressure
scaling are used. W_mu includes consistent off-diagonals and refreshes at each
required friction linearization; mu is never commuted through H. There is no
dense global filtered map. Sparse B uses the retained constrained native
assembly, replacing per-Krylov bulk integration. B still builds per Newton
linearization; S/N and their lookup reuse fixed geometry and eta_ve.

Mechanical history evaluation now consumes existing Stokes-QP associations and
batches by cell. Geometry, physical-to-unit coordinates, basis and fixed phase
are cached separately from accepted tensor/Z/I values. Publication invalidates
stress samples. Fixed particle and dense-output point layouts coexist in compact
cell-indexed storage. Independent native FE publication gradients and the BP3
work observer are batched by cell; their independent evaluation and full point
coverage remain. Arbitrary unprepared points retain the scalar path.

First-time projection uses a conservative segment hierarchy in a fault-aligned
frame, followed by the original exact admission/selection arithmetic in original
segment order. Knot ownership, nonuniform spacing, overlap rejection and manager
boundary completion remain. Extreme/ill-scaled queries fall back to the scan.

Gauss3 source mass inverses and invariant inverse-defect factors are cached by
actual JxW/affine widths. All nine modes, velocity operand sums, Maxwell
coefficients and the original operand-derived allowance remain fresh. The native
bulk matrix and preconditioner build at the first Newton iteration of **each
attempt**, then reuse within that attempt. Residuals, frictional tangent and
active-set solves continue updating. Changed timestep/cutback, constraints,
pressure scaling or layout forces rebuilding; no cross-step bulk reuse occurs.
Condensed operator work vectors are reused within the linearization.

The first attempt exposed an additional eligibility dependency: residual-only
line-search assembly could overwrite A because its old boundary check considered
nonzero temperature constraints. The observed A_uu norm changed from 6.21036e18
to 78. Extending the homogeneous **Stokes-direction** check to this selected
route prevents that overwrite while retaining all non-Stokes constraints.
The saved failed attempts and subsequent owned/ghosted-vector comparison-harness
failure remain in the evidence. No convergence/publication tolerance was relaxed.
The new frozen-action comparison guard is 2e-11 relative, normalized for C by
||Ax||+||B K_V^-1 Gx||; it is a diagnostic guard, not a production acceptance rule.

## Local measurements

GCC 12.4.0, OpenMPI 5.0.6, deal.II 9.6-local, Release, one thread/rank. Final
timing runs did not overlap. Each pair uses the same compiled source and plugins,
switching only the optimization selector, with two positive steps of 200000 s.
The reference selector also reproduces the copied fresh-base binary's compared
persistent fields/stress exactly. The strengthening fixture is local test
physics; the ordinary server plugin has no local loading or wall-injection macro.

| Local mesh | Cells | Native DoFs | Independent DG DoFs | Fixed points | Surface nodes |
|---|---:|---:|---:|---:|---:|
| Small | 4136 | 149538 | 111672 | 66176 | 117 |
| Larger | 11528 | 410198 | 311256 | 184448 | 117 |

| Quantity, reference → optimized | Small | Larger |
|---|---:|---:|
| Harness elapsed, including initialization/terminal checkpoint (s) | 30.888 → 18.305 | 80.940 → 46.709 |
| Positive-step header intervals, excluding terminal checkpoint (s) | 14.830 → 8.627 | 45.503 → 23.695 |
| Total / positive-step speedup | 1.687 / 1.719 | 1.733 / 1.920 |
| Peak native RSS, ranks 0/1 (MiB) | 410.32/424.99 → 425.94/438.50 | 743.74/753.43 → 779.70/786.14 |
| A assembly (s; calls) | 2.230;7 → .976;3 | 6.260;7 → 2.600;3 |
| Residual assembly (s; calls) | 3.380;10 → 3.120;14 | 9.400;10 → 8.510;14 |
| Preconditioner setup (s; calls) | 1.340;7 → .615;3 | 3.430;7 → 1.560;3 |
| B application, exclusive (s) | 3.961925 → .121045 | 14.231666 → .179274 |
| G application, exclusive (s) | .695514 → .100904 | 2.626980 → .335753 |
| B setup, native timer (s) | .120 → .393 | .399 → 1.510 |
| G lookup builds, native timer (s; calls) | .140;7 → .0247;1 | .586;7 → .099;1 |
| Optimized complete G setup, exclusive sum (s) | .167367 | .732280 |
| Native history candidate/check (s) | 2.500 → .404 | 7.380 → 1.120 |
| Independent native-gradient sampling, nested (s) | .554 → .152 | 1.580 → .408 |
| DG source update, nested (s) | .0511 → .0429 | .154 → .119 |
| Postprocessing (s) | 3.220 → 2.290 | 10.800 → 6.970 |
| Dense authoritative stress output, nested (s) | .726 → .409 | 2.110 → 1.130 |
| Terminal checkpoint (s) | .406 → .410 | 1.030 → 1.010 |

Native timers can overlap/nest and are rounded; do not sum these rows. G lookup
is only part of complete setup. The unchanged original path has no sparse G
setup timer. Individual rank RSS peaks need not coincide; node simultaneous peak
was not measured. Sparse setup costs and memory increase are included in totals.
Newton updates are [2,0,2] in both pairs; Krylov counts are [71,0,70] and
[73,0,76], including t0. Preconditioner applications remain essentially unchanged
(small 2.988→3.005 s; larger 9.124→9.073 s).

| Instrumented rank-zero cumulative calls, reference → optimized | Small | Larger |
|---|---:|---:|
| Projection queries | 701073 → 331034 | 1983085 → 921939 |
| Exact segment tests | 81324468 → 118883 | 230037860 → 474123 |
| DG FEValues constructions | 422823 → 11060 | 1207627 → 29773 |
| Native FEValues constructions | 491075 → 12044 | 1399482 → 32896 |
| History cache misses | 287701 → 110264 | 828853 → 306296 |
| Source inverse / invariant roundoff setup | 4136 → 5 | 11528 → 6 |
| Coupling lookup / sparse G builds | 9/0 → 2/1 | 9/0 → 2/1 |
| B linearizations / bulk builds | 7/7 → 7/3 | 7/7 → 7/3 |

These low-overhead counters cover instrumented fault paths, not all FEValues
objects in ASPECT. They are local rank-zero counts and reset on process restart.

The isolated kernel uses BP3's actual anchors/resampling, **5776 nodes and 8664
queries**. Reference scan .264188 s versus indexed .002393 s (110.4×), with
.001300 s index construction; admission/segment/xi/distance are exactly equal.
Manufactured positive surface-work samples with spatially varying mu give
filtered relative difference 2.15e-16, .001600 s packing/filter setup and
.0000919 versus .0000390 s post-filter product. These microsecond product timings
exclude the shared normal action/filter solve and are not a native full-grid
work-measure benchmark. The native checks below test the actual bulk-work measure.

## Numerical and lifecycle verification

- Rebuilt native suite: **126 cases / 149074 assertions pass**. The isolated
  full-node kernel separately passes 25100 assertions. An earlier broad negative
  test filter accidentally selected two saved-state tests without their required
  environment; its failing log is retained, and the intended default suite passes.
- Frozen native source coefficient and allowance arrays are **bitwise equal**,
  including manufactured non-proportional tensor data/all nine modes. Batched
  versus independent scalar reconstructed-QP stress differs by at most 3.09e-12
  Pa (operand scale 45149 Pa). The larger control gives 4.25e-13 Pa.
- Frozen B/G/C maximum relative differences, small/larger respectively:
  B 1.03e-15/1.87e-15, filtered G 1.22e-14/1.77e-14, complete C
  4.44e-16/9.46e-16. These include live Krylov directions and manufactured probes,
  free/restricted K_V, physical pressure scaling 3.204e14, nonconstant friction,
  endpoint completion and respectively 3060/6426 stored Stokes hanging relations
  summed across ranks. Three surface rows are additionally restricted in probes.
- Optimized/reference full 3D reconstructed stress RMS is 1.30e-10/2.09e-10 Pa
  (small/larger); maxima 1.61e-9/3.00e-9 Pa. Whole relative L2 is
  5.35e-14/3.14e-13; localized relative L2 1.75e-14/1.03e-13. DG coefficient maxima
  are 2.88e-9/5.48e-9 Pa; Z differences 1.86e-8/3.35e-8 Pa m. Phase, I and native
  surface mass are identical. Max rate/state/slip differences are 8.28e-25 m/s,
  1.87e-9 s and 1.09e-18 m. Native weak-load maxima differ by <=5.97e-8 Pa m;
  worst interior perturbation-scaled difference over t0/step 1/step 2 <=6.01e-11.
  Endpoint/interior and
  common-measure diagnostics, including intermediate accepted times, are saved.
- Actual nonlinear residuals, true linear checks and native publication guards
  pass. The optimized timing runs' max normalized nonlinear residual is 2.94e-9;
  surface RMS <=.001411 Pa and publication error/allowance <=.00323. The broader
  lifecycle matrix has maximum normalized nonlinear residual 3.65e-9 and maximum
  publication error/allowance .00349. No tolerance changes.
- 1/2/4-rank ownership checks pass. Versus two ranks, stress maxima are 1.42e-8
  and 4.86e-8 Pa. A serial step-one checkpoint resumed on four ranks differs from
  uninterrupted four-rank stress by 3.53e-8 Pa maximum (9.74e-10 Pa RMS), and
  from uninterrupted serial by 2.34e-8 Pa maximum. These MPI solver/reduction
  differences are measured, not described as bitwise restart equality.
- Complete one-owner candidate rejection preserves accepted histories; the
  same-dt retry matches the uninterrupted control exactly. Actual outer-loop
  rejection at step 2 preserves accepted state/publication count, cuts 200000 to
  100000 s and then uses the native 91% growth to 191000 s. Its three-step result
  matches a prescribed matching-sequence control exactly. Four-rank nonlinear
  failure/cutback has the same sequence; stress max difference 5.60e-8 Pa versus
  the two-rank control. Every accepted observation publishes once, including
  restart; rejected attempts do not publish.
- A bounded native adaptive control retains initial/first cap 4e6 s, max 4e7 s,
  log-state .2, growth/cutback rules, and accepts dt=4e6 then 7.64e6 s. This is a
  short local strengthening control, not qualification of full weakening/event
  dynamics.
- The existing collective wall-stop regression passes again: only rank 1 requests
  stopping at positive step 2 while ordinary outputs are not due; both ranks
  select terminal output/checkpoint and terminate within the 240 s bound.
  Wall-stop restart and scheduled-output restart both match uninterrupted step
  five exactly in all compared fields, with particle/bulk audit files byte equal.
- All four server inputs pass native validation with matching ordinary plugins
  and no test macros. Independent parameter comparison verifies physical/adaptive
  equivalence; this parser check does not construct the full mesh.

## Particle dependencies and retained limitations

The effective `Particles/Integration scheme = BP3 stationary samples` remains
selected. Its update leaves coordinates unchanged. The diagnostic registry validates
all fixed IDs/coordinates at every observation; population is 66176/184448 with
zero births/deletions or movement. Native particle-property arrays still carry
comparison stress and inert H, are backed up/restored and checkpointed. Matching
CellIds/point coordinates select the independent native-gradient publication
validation traces: this check is an acceptance dependency, not merely offline
output. Ordinary particle initialization/bookkeeping and the independent work
observer also remain. No particle domain/Voronoi/CPDI/face construction or reads
feed selected initialization, homogeneous coefficients, accepted mechanics or
surface weak loads. Legacy particle/CPDI backends are untouched. Future removal
would need a mesh-owned fixed validation registry and matching checkpoint/ownership
rules; it is not part of this optimization.

Remaining measured costs include native residual/material evaluation, unchanged
Krylov/preconditioner application, per-Newton B setup, diagnostic CSVs and
replicated all_gather checkpoint construction. General arbitrary-query fallbacks
and existing coordinate-keyed diagnostic storage remain. Full-grid checkpoint
write/read memory, dense-output size and conditioning must still be measured
during startup. No extrapolated full-event speedup is claimed.

The inherited **6.598 kPa** fixed-FF spatial change, failed endpoint 1% temporal
screen (1.150/.523/1.047%), **49.704 Pa** surface-state representation imbalance
and **strict_reference_match_passed=false** remain. Neither particles nor the old
DG implementation is a physical accuracy reference. This task qualifies exact
evaluation changes locally, not propagation, phase evolution, transport, moving
geometry, full PF5 completion or first-event accuracy.

## Evidence, replay and handoff

[Compact evidence/replay](../../../benchmarks/reconstructed_fault/particle_free/evidence/bp3_optimization/README.md)
contains commands, hashes, inputs, final reductions and retained engineering
failures. Raw fields/checkpoints remain in
`build-particle-free/bp3-optimization/`; final timing pairs have the `qualified-`
prefix. Earlier timing results are retained but not used in the final tables.
Local execution uses `run_bp3_optimization.py`; native jobs run through the
bounded `refactoring_boundary/run_logged.py` environment. The server runtime
continues to use direct native launches, not a Python controller.

The regenerated source distribution is
`build-particle-free/bp3-optimization/package/bp3-dg-stationary-optimized-20261017-r3.tar.gz`;
its adjacent checksum/archive-verification record identifies the exact payload.
The earlier r2 and first-event archives are unchanged. Sources include the
native unit profile fixture previously omitted from the package closure.

Recommended next action: build the supplied matching stack, execute **startup
through step five and resume through step eight**, and inspect accepted-state,
operator/solver and resource evidence before a separate first-event decision.
See the [manual server procedure](../../../benchmarks/reconstructed_fault/particle_free/bp3_dg_first_event/README.md)
and resource sheet. The task stops with this package; no server run is submitted.
