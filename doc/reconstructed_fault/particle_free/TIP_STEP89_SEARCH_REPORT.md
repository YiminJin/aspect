# Fine-grid step 88 → 89: constrained-search ordering

**Step 89 accepted with positive-metric-first ordering; default-order replay
capped without acceptance.** The opt-in variant reaches .055625 s in 454.630 s
process time. The retained default spends 2400.315 s without reaching an accepted
endpoint. Original equations and gates remain; the accepted checkpoint restores
exactly. This supports retaining the option for bounded fine-grid continuation,
not changing the production default or claiming search-order endpoint equivalence.
No simulation remains running.

## Scope and preserved parent

This bounded follow-up to [fine growth continuation](FINE_MESH_GROWTH_CONTINUATION_REPORT.md)
compares two search orders for **one interval**, .055 → .055625 s. It does not
resume the longer growth campaign. Both replays use the saved `fine-growth-warm01`
snapshot 02, accepted step 88, on the 8×4 m synthetic fixed-direction section-RSF
specimen: h=.0625 m, dt=.000625 s, native DG Bernstein phase, midpoint with exact
stage aging, two MPI ranks and one thread per rank. Physical inputs, quadrature,
full stress/cohesive A/section histories, bounds and acceptance gates stay fixed.

Entry is `pf-rsf-particle-free` at `b32a2d4f6e21c219f9af786d1b6e3725ea78a5a4`,
with the preceding continuation's uncommitted work retained. The previous tested
core/checker pair is copied to `build-particle-free/tip-step89/reference/` before
rebuilding. Incoming tracked changes, parent hashes and tested identities are in
`benchmarks/reconstructed_fault/particle_free/evidence/tip_step89_search/`.
Neither HEAD nor the stale embedded git banner identifies the tested executable.
The instrumented pair is separately preserved in `build-particle-free/tip-step89/tested/`.
The new core SHA256 is
`034d2a848a3a10c496fa097475f31497c00bc68c3456fb6d87c1ee700f453c27`;
the unchanged checker SHA256 is
`b800ef431a2469760efd16befcba94e8cd743156629b55e4c921f769255d9d84`.

## Exact comparison

The default remains adaptive exact-first: try the full constrained tangent,
then the existing positive phase metric family if needed; after a successful
shifted search, retry exact Newton after a tenfold merit reduction or within
one decade of convergence. The benchmark-only
`ASPECT_TIP_POSITIVE_METRIC_FIRST` override instead tries the same positive metric
family first whenever the normalized physical merit exceeds 10. Exact Newton
remains the fallback and the first polishing attempt at merit ≤10. No matrix,
residual, metric, slope, bound interpolation, line-search rule or tolerance changes.

Merit is max(mechanical residual/1e-9, phase natural-map KKT/1e-7).
The exact direction must retain its existing feasibility and true linearized
phase-VI certificate <1e-9. Existing inner iteration limits remain. Final
publication still requires both physical residual gates and the independent
ledger/dissipation checks. The previous within-search Krylov warm start is
present in **both** replays. Environment flags are benchmark controls, not new
checkpoint laws or production defaults.

`ASPECT_TIP_TRACE_SEARCH` logs the true linearized phase VI, mechanical and
continuity norms before each exact active-set classification, and every evaluated
line trial's actual merit, functional difference and directional slope. The
reported FGMRES residual estimate is separately retained; it is not the VI
certificate. Successful exact-solve time sums omit unfinished or shifted solves;
complete run and accepted-step timers provide total cost.

Reproduce sequentially with `bulk_tip/run_step89_search.py positive` and
`bulk_tip/run_step89_search.py default` under the existing MPI logging wrapper.
Each run has a 2400 s numerical-process cap and an end-step limit of 89.
The original parent is copied, never overwritten. The pre-step native restore
observer checks that both variants begin from the same accepted state.

## Results

The positive-metric-first replay **accepted step 89**, at .055625 s, after
20 nonlinear iterations, 1144 Krylov iterations, 17 shifted searches and three
exact polishing searches. It evaluated 170 line trials. Recorded interval cost
is 443.510 s; wrapper cost is 454.827 s. The final mechanical residual is
2.42513e-16, phase KKT 7.04101e-8 and endpoint ledger defect 7.77328e-15.
All prescribed physical gates passed. Snapshot 03 contains the accepted endpoint.
Native coefficients remain above their stored obstacle (minimum increment zero),
with maximum .95. Minimum −4.33681e-18 is identical to the parent roundoff value;
no clipping or new bound relaxation was applied.

The interval adds 3.30076777e-4 J/m of fracture energy. Its physical-work error
Wext − ΔE − Db − Dc − Df is +1.97068512e-8 J/m, or .0059704% of added fracture.
For this single interval the signed and accumulated absolute errors coincide.
The phase remainder is 1.96955772e-8; phase work is −1.12646515e-11;
mechanical work is −1.66533e-15. Midpoint stress/traction/fracture-loss channels
remain zero. This is an interval-level accounting result, not mesh convergence.

The trajectory of nonlinear iterates is not monotone in phase KKT. Full positive
metric steps repeatedly raise KKT to about .069; heavily damped accepted steps
reduce it. Accepted alpha reaches 4.76837e-7. At iteration 17, merit falls to
5.6328, triggering exact polishing. Three half steps then reach the original
physical gate. Each polishing direction satisfies the original linearized VI
certificate; accepting a cheaper direction alone never constitutes step acceptance.

The default replay reproduces the earlier first exact-search sequence: all
24 pass iteration counts match, totaling 2116 Krylov iterations and 596.399 s
of successful linear-solve time here. The logged true phase-VI defect reaches
8.15277e6; the search supplies no certified direction. Small or zero inner
residual estimates do not establish feasibility or the full VI. The positive
metric fallback then produces exactly the same first accepted search step as
in the positive-first replay. The default tenfold-merit trigger immediately
retries exact Newton. Its four-solve correction is certified (VI 1.31076e-11),
but the accepted full step raises physical phase KKT to .064212. This recreates
the interrupted trajectory and leads to another expensive exact search.
These are failures of trial directions to meet certificates, not publication
of inadmissible physical state or evidence that the accepted scheme changed.

The reference subsequently accepts one exact correction at nonlinear iteration 3
(certified linearized VI 1.60316e-13), but this raises phase KKT to .0690183.
The fifth exact-search attempt is interrupted by the **2400 s process cap** at
its pass 17 check. The last nonlinear mechanical residual is 5.88584e-12, but
phase KKT remains about 690,183 times its gate. No physical endpoint is accepted.
Underlying runner exit 124 denotes the cap; the outer wrapper exits 1 because
its child runner reports that failure. This is not a completed nonlinear failure
certificate or an accepted state with relaxed gates.

| Quantity | Positive metric first | Retained adaptive exact first |
|---|---:|---:|
| Physical endpoint | 89 / .055625 s | None; accepted parent remains 88 / .055 s |
| Process elapsed | 454.630 s | 2400.315 s, capped |
| Accepted interval timer | 443.510 s | Unavailable |
| Nonlinear corrections accepted | 20 | 4, still trial state |
| Exact searches attempted | 3 | 5, last incomplete |
| Certified exact directions | 3 | 2 |
| Successful exact linear solves | 3 | 70, plus unfinished work |
| Krylov iterations in those exact solves | 199 | 7907, lower bound |
| Time in those exact solves | 46.004 s | 2283.331 s, lower bound |
| Total accepted-step Krylov iterations | 1144 | Unavailable |
| Final physical phase KKT | 7.04101e-8 | No accepted value |

The reference spends **5.28 times** the candidate process cost without completing.
That is a bounded work comparison, **not** a completed-endpoint speedup ratio.
The default's third exact search alone costs 984.848 s in successful subsolves
and exhausts all 24 passes. Its first and third searches are uncertified; its
second and fourth certify directions but do not remove the recurring physical
KKT spikes. The candidate postpones exact work until the physical merit is ≤10;
its three polishing solves take 71, 67 and 61 Krylov iterations, with true phase
VI defects 2.02e-14, 2.81e-14 and 2.79e-14. This identifies **when exact constrained
search is attempted** as an effective cost lever for this interval. It does not
explain every failed restricted solve or prove general convergence.

## Endpoint, restart and missing comparison

Both pre-step restore observations reproduce the parent exactly: native phase
coefficients/solution, full plane-strain stress and cohesive A, signed/absolute
fault-deformation observations, and section age/slip/rate/resistance. The accepted
positive-first endpoint's read-only restore also reproduces every compared field
exactly, with accepted clock 89/.055625, publications 89, and pending clock
90/.05625 at dt=.000625. **No step 90 was advanced.** Snapshot 03 is retained at
`build-particle-free/bulk-tip/runs/step89-search-positive/output/restart/03/`.
The original step-88 snapshot hashes are unchanged. The capped default's trial
iterates are not promoted, checkpointed as accepted history, or used in growth
analysis. No numerical processes remain.

The candidate's maximum native coefficient increment during this interval is
.00449014346. Stored phase irreversibility and the existing upper bound pass.
Full endpoint fields and work channels are retained; the energy results above
use the existing independent endpoint ledger, with no modified loss terms.

**Exact missing data:** a converged default-order step-89 endpoint, its exported
stress/A/phase/section fields, accepted energy/work row, whole-step counters and
step-89 checkpoint. Therefore accepted-field differences between search orders,
accepted-work differences, and a completed-step speedup cannot be computed.
Identical parents and agreement of all 36 previously retained inner iteration
counts establish reproducibility, not uniqueness of this nonconvex endpoint.
The reducer computes these comparisons only if both endpoints actually exist.

The build, script syntax checks and `git diff --check` pass. This task changes
only an opt-in search-order policy and diagnostics; no full unit or long physical
matrix was repeated. All simulation work used two ranks, one thread each,
sequentially. Total numerical replay time is 2854.945 s (47.58 minutes), plus
8.64 s for the read-only restore wrapper. Prior failed evidence and cleanup
availability losses remain intact. No source commit or server run was made.

## Decision and next experiment

Retain positive-metric-first as an **opt-in benchmark option** for this fine-grid
control. It produces an accepted, restartable endpoint at affordable cost while
retaining exact polishing and all physical gates. Keep the adaptive default
unchanged. Its repeated KKT spikes and tiny accepted line fractions remain
visible; this result does not qualify the search as generally robust, establish
endpoint uniqueness, or resolve the specimen's spatial sensitivity.

The next justified experiment is a separately selected, bounded **89→96** window
from the verified candidate snapshot 03, using this opt-in order and unchanged
h, dt, material, loading and acceptance gates, with a checkpoint per accepted
interval. A 60-minute/two-rank cap would test whether the cost improvement persists
across the seven remaining intervals to .06 s. If it completes, reuse the retained
coarse M3 endpoint and existing spatial/work reducer to close the .04–.06 window;
do not start another mesh or loading campaign. This follow-up has **not** run.
The .06 target, .08 continuation, third mesh and instability studies remain
unqualified/deferred.

[Convergence figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_step89_search/search-comparison.png)
([PDF](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_step89_search/search-comparison.pdf))
shows physical residuals, accepted line fractions and true exact-search VI.
[Evidence/commands](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_step89_search/README.md)
and [machine-readable comparison](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_step89_search/comparison.json)
retain identities, clocks, costs, failed trials and successful certificates.
