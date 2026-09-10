# Polyline verification hold: bounded residual-consistency audit

2026-09-09. This is a diagnosis and correction proposal, not a new stopping
rule, a changed quadrature formulation, or a completed K1/K2 trajectory replay.
The polyline implementation and all its saved passing evidence are retained.

## Scope and method

The unchanged checked-in `phase_field_fault_stage_i.prm` is included by
`benchmarks/reconstructed_fault/uniform_shear/nonuniform/residual-audit/stage_i.prm`;
only library/output paths differ. Geometry, initialization, physical inputs,
support, full I_h, V_min, tolerances and iteration budgets are unchanged.
`pre-audit.patch` preserves the preceding tracked working-tree diff; earlier
untracked artifacts remain in their original directories.

The opt-in `ASPECT_K1_FLOOR_AUDIT=1` reuses the K1 affine comparison, bounded to
Newton iteration 1. Unlike the old diagnostic, it copies the original Stokes
matrix before any shadow/trial assembly and restores that matrix after each
shadow evaluation. B/G/K_V retain their original linearization caches. The
normal production residual still determines every candidate merit. Shadow
vectors never participate in convergence or acceptance.

For ASPECT's RHS b=-R_bulk the comparison is

    b_trial = b_base - A dx_represented + B dV_represented.

The measured bulk increment is the actual trial vector minus the accepted
base, after physical-pressure normalization where applicable. Constrained
rows are removed homogeneously; physical pressure is divided by the existing
solver scaling before applying A. This dynamic-pressure fixture does not
normalize the physical pressure or use a pressure quotient. The measured V
increment includes the manager's `V + (V_trial-V)` arithmetic; its difference
from explicit surface V is also checked.

The diagnostic separately assembles ordinary Newton terms and the fault-load
channel (negative frozen beta*tau_old throughout the bulk, positive profile
history, and positive B V). Two additional shadow evaluations distinguish
combining these channels after cell quadrature from combining independent
global vectors after constrained assembly. These diagnostics use the same
production material evaluation and integration, not a replacement formula.
The coupled-assembly flag stays enabled in every channel, so the existing K1
cell-constant velocity subtraction and divergence evaluation are unchanged.
They apply to this affine, all-prescribed-boundary fixture, not arbitrary
nonlinear rheologies or extra boundary-face loads.

## Measured cause

At the second linearization all **8 surface vertices are active, 0 free**.
The projected surface residual is zero; dV and the difference between explicit
and manager-represented V are exactly zero. This isolates the observed failure
from a moving active set, V representation, or a nonzero B dV contribution.
It does not newly verify B for nonzero increments; the saved production
coupled-action tests remain the independent coverage for that case.

The fixed bulk scale is 5.8889378102325262e-7, the unchanged target
5.8889378102325264e-13, and the existing precision allowance is zero. The
surface scale remains 156788247.47774017 Pa. The restricted linear solve
returns an estimated residual 3.425257633142748e-17 and fresh residual
3.4252576335537871e-17 against its target 5.7842826748736911e-17.

The full-step candidate gives the following Euclidean bulk-block norms
(in 2-D velocity weak-load units, N/m per out-of-plane thickness; the continuity
equation uses the existing reference-viscosity/length pressure scaling):

| Quantity | Velocity | Scaled continuity |
|---|---:|---:|
| Normal base residual | 5.7842826706216433e-10 | 2.2178838130288975e-14 |
| Affine prediction using represented increments | 2.1474378685609198e-18 | 3.418519882167673e-17 |
| Fresh normal assembly | 5.7437020053058799e-10 | 3.4185199630070312e-17 |
| Fresh-minus-affine mismatch | 5.7437020058527155e-10 | 2.2037605142887603e-22 |
| A-action change from represented versus requested increment | 2.1488072617267405e-20 | 1.3589797669696894e-21 |
| Newton-only affine mismatch | 4.2722966711231045e-21 | 2.2037605142886988e-22 |
| Load change after subtracting B dV | 0 | 0 |
| Cell-separated affine mismatch | 1.3643740221386957e-10 | 2.2037605142886988e-22 |
| Globally separated affine mismatch | 4.2722969575518505e-21 | 2.2037605142886988e-22 |

The Newton-only and load-only base velocity norms are
5.8889378102329317e-7 and 5.8889378102325262e-7 respectively; the load-only
continuity norm is zero. The actual velocity correction has maximum amplitude
6.1307421551090114e-21. Its represented A action is accurate far below the
required residual accuracy: representation is not the cause of this floor.

The evidence identifies cancellation when unknown-dependent residuals and
frozen stress loads share the cell/global accumulation. Keeping their cell
integrals separate is insufficient: after they are combined into a common
cell RHS, constrained global assembly still yields a 1.36e-10 inconsistency.
Independent global assembly removes the measured inconsistency by about
eleven orders of magnitude. Continuity, the frozen load itself, and the
fresh linear solve do not exhibit the failing discrepancy.

The normal candidate reduces merit from 4.8238643361083479e-7 to
4.7564163996014372e-7 and is accepted at alpha=1 with zero rejected candidates,
but remains far from convergence. In particular, the globally separated
**shadow** residual at this same candidate is still 5.7842808068903986e-10:
one must recompute the residual and Newton direction consistently, not
retroactively accept this candidate based on the diagnostic slope. The
production trajectory matches the saved unaudited run through the audited
update and all twelve nonlinear iterations: `trajectory-comparison.diff` is
empty when comparing every normal residual record and rejection count.

The original budget is exhausted with last reported bulk residual
3.4763076771419674e-10, velocity 3.4763076771419664e-10, scaled continuity
8.0945279408907076e-18 and projected surface residual zero. The new positive
test guard rejects this result explicitly, producing process exit 1 instead
of the former misleading exit 0/"verified" combination. This is still a
**numerical failure of the positive fixture**, not a newly passing solve.

## Smallest justified correction proposal — requires review

Keep the unknown-dependent weak residual and frozen Maxwell/profile load in
separate accumulators through local quadrature, homogeneous constraint
distribution and MPI compression; combine the global vectors only afterward.
Do not merely move the addition to the end of each cell. The frozen stress
load must still include beta*tau_old exactly once outside as well as inside
fault support. Preserve the existing B sign, geometry and constitutive
evaluations, and use the same residual evaluation for the Newton base and
every non-committing trial.

This is an arithmetic residual-evaluation correction, not permission to
change the discrete equations or an argument for relaxing the stopping rule.
The current diagnostic load channel includes B V, which is constant in this
audited active-set case. A production split should distinguish the truly
frozen load from the V-dependent contribution and verify nonzero dV too.
No production correction is implemented by this audit.

Focused verification for that correction:

1. On one/two ranks, require affine consistency of both bulk blocks under
   actually represented constrained increments, including a nonzero dV case.
   Include the present strongly cancelling old-stress fixture and preserve
   frozen-load/constant-history weak-load checks.
2. Require the unchanged positive Stage-I fixture to meet both final criteria,
   not merely exit successfully or retain admissible initialized histories.
3. Retain independent surface convergence, active-set, true-linear-residual,
   compatibility, exhaustion, rollback and lifecycle/restart coverage. Check
   that diagnostic mode on/off leaves production candidate decisions unchanged.
4. Only after review and focused verification, run the prepared short K1
   reference replay, then unchanged K2-64/K2-128 through 1 s. Report initial
   projection changes, actual weak endpoint traction, support/normalization
   and cost before any larger campaign.

## Verification and artifacts

All new logs are under
`benchmarks/reconstructed_fault/uniform_shear/nonuniform/residual-audit/`.
The README there records exact commands and diagnostic field definitions.

- Debug/Release builds and test/benchmark plugin builds use `-j4` and pass:
  `build.log`, `build-final.log`, `plugin-build.log`, `benchmark-plugin-build.log`.
- Quadrature plus Stage-I units pass 815 assertions in 18 cases on one rank
  and on each of two ranks: `unit-one.log`, `unit-two.log`.
- Intentional linear exhaustion passes (61.17 s); rollback passes on one rank
  (102.52 s) and two ranks (100.90 s): `focused-tests.log`, `rollback-mpi.log`.
- The strengthened positive surface-temperature test passes on one rank
  (196.35 s) and two ranks (84.70 s): `focused-tests.log`, `temperature-mpi.log`.
- The first sandboxed MPI integration launches were denied local sockets,
  not numerical failures; those outputs are retained in `focused-tests.log`.
- The unchanged positive Stage-I diagnostic replay is in `stage_i.log`.
  It exhausts twelve iterations and the positive convergence guard correctly
  fails. `stage_i.resources.json` records 1042.02 s wall, 808.49 s user CPU,
  .38 s system CPU and 633696 KiB peak RSS (about 619 MiB). This includes
  diagnostic assemblies and some concurrent focused verification, not a K2
  trajectory cost estimate. It finishes inside the 1200 s wall-time bound.
- `git diff --check` passes. No full integration suite is run.

Positive Stage-I/lifecycle, surface-temperature and benchmark postprocessors
now require `SolverControl::success` and a finite final maximum normalized
residual below the configured tolerance. They reset this evidence at each
timestep. Intentional exhaustion/rollback fixtures use their own separate
failure checks and are unchanged. No golden output is refreshed, and no
K1/K2 trajectory replay is counted as complete by this audit.

Files edited for this audit, distinct from the retained polyline work:

- `source/simulator/solver.cc`: opt-in fixed-linearization, represented-increment
  and assembly-channel comparisons; normal acceptance is unchanged.
- `source/simulator/assembly.cc` and the implementation-only
  `source/simulator/reconstructed_fault_residual_audit.h`: diagnostic assembly
  channels; the default channel executes the original residual arithmetic.
- `tests/phase_field_fault_stage_i.cc`,
  `tests/phase_field_fault_stage_j_temperature.cc`, and
  `benchmarks/reconstructed_fault/uniform_shear/uniform_shear.cc`: require
  explicit final convergence before claiming successful positive verification
  or exporting an accepted benchmark trajectory.
- This report, `stage_K_progress.md`, and the new diagnostic directory's
  README/input/evidence. No public constitutive or quadrature API changes.
