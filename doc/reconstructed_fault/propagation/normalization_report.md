# P2 normalization representation follow-up

2026-10-06. **I_h-only row-sum lumping is implemented and passes the restricted
P1 lifecycle checks. It does not qualify the retained P2 fields.** The coarse
field is finite but poorly represented at both tested fault spacings. The finer
field is exactly saturated at the side endpoints and has an infinite transverse
and along-fault normalization. No fresh coupled trajectories were started under
the new representation; their required finiteness/accuracy gate fails.

The [benchmark model overview](../../../benchmarks/reconstructed_fault/propagation/COUPLED.md)
was updated before deterministic reproduction. The [cumulative source inventory](SOURCE_CHANGES.md)
covers actual changes from `de7beaa65846c7372711f7922abf1c5d9a745ee6`, including
untracked tests/evidence. Same worktree/branch, no commits, pushes, other-worktree
changes, H-law change, floor, phase cap or new activation mask.

## Frozen states and integrability

`normalization_capture025` and `normalization_capture0125` reproduce the old
`coupled_report025` and `coupled_report0125` trajectories using the preserved
pre-projection-change executable. Accepted `feedback.csv`, failed-entry phase
metrics and normalization-node files are byte-identical. The observer captures
1,089 Q1 nodal values and native degradation values at the failed entry, before
mechanical history publication. Existing H/stress/q/previous-I_h checks remain.
The subsequent analysis and native projection probes do not advance any physical
history or replace the live phase field. Captured CSVs and provenance are in
[normalization evidence](../../../benchmarks/reconstructed_fault/propagation/normalization_evidence).

| Frozen entry | Maximum phi (17 digits) | Location (m) | Native g |
|---|---:|---|---:|
| dt=0.25, phase consumption at t=0.5 | 0.99708667519951666 | (0,0.5) | 3.3299522134135952e-8 |
| dt=0.125, consumption at t=0.375 | **1 exactly** | (0,0.5) and (1,0.5) | **0 exactly** |

These are failed-entry phase fields, not accepted complete mechanical steps.
For Q1 on the square mesh, extrema occur at vertices; the complete nodal capture
therefore establishes the global phase maximum, not merely a sampled maximum.
The coarse field is bounded strictly below one on the compact domain, so both
integrals are finite. Its direct endpoint transverse integral is about
124200.5398701 m, and its surface integral is 5071.1477732265 m².

Source: `PhaseField::DegradationFunction::value` and
`PhaseFieldFault::normalization_integrand`. Here p=1 and m=128:

\[
g=\frac{(1-\phi)^2}{(1-\phi)^2+128\phi(1+\phi)},\qquad
h=128\frac{\phi(1+\phi)}{(1-\phi)^2}.
\]

Near the fine left endpoint, writing z=y−0.5, the actual adjacent Q1 values give

\[
1-\phi=0.2352294738484879\,x+2.69331139917211\,|z|
        +O(x|z|).
\]

Thus h is asymptotic to 256/(Ax+B|z|)². At x=0 its transverse integral diverges
as integral dz/z². For x>0 approaching zero, I(x) is asymptotic to
808.1493995858582/x, and its along-fault integral diverges logarithmically.
The right endpoint has the same behavior within roundoff. A positive surface
basis at either endpoint therefore has a divergent projection RHS. Treating
the endpoint alone as a set of measure zero does not cure the surface integral.
Finite Gauss samples miss the singular point and cannot establish integrability.
The earlier report's positive sampled fine integrals are retained, but do not
prove a finite underlying normalization.

The independent frozen-field script integrates each transverse Q1 interval
analytically. For u=1−phi, h=128(2/u²−3/u+1), with interval averages
<1/u²>=1/(u0*u1) and <1/u>=log(u1/u0)/(u1−u0), using the continuous equal-value
limit. Native nodal g agrees within 4e-16 absolute. Independent Q64 transverse
quadrature on 31 nonsingular columns agrees to 3.89e-15 relative (coarse),
1.91e-14 (fine). This quadrature comparison is not the singularity proof above.

## Projection, accuracy and source effect

Only the material model's private I_h projection changes. The existing Q1
surface mass matrix and RHS are assembled from distributed owned profiles and
MPI-summed as before. `surface_weight = segment_length * Gauss_weight` is the
arclength measure. Shapes (1−xi,xi) are nonnegative and sum to one; all iterated
Gauss weights are positive. Each assembled row sum is checked finite and
strictly positive before division. End-node denominators are spacing/2 and
interior denominators spacing (minimum 0.05 / 0.025 m for the two meshes).

New `Material model / Phase field fault / I h projection = row sum lumped`
selects I_i=b_i/sum_j M_ij. Default `consistent` and its tridiagonal solve remain
unchanged. No generic particle projection, shared matrix or unrelated field is
lumped. Constants and the quadrature-defined surface integral are conserved;
these identities do not prove spatial accuracy.

The frozen bulk field is identical at fault spacings 0.1 and 0.05 m. Surface
quadrature was refined separately (3-point Gauss on 1, 8, 32, 128 panels per
segment) to resolve the concrete endpoint-integration uncertainty. Normal
integration remains analytic. A separate adaptive surface reference splits at
the 32 bulk-cell boundaries. Dense validation includes both exact side endpoints
and geometrically clustered near-boundary points. Exported CSVs are subsampled
for compactness; all metrics use the complete validation grid.

| Coarse frozen field | spacing 0.1 m | spacing 0.05 m |
|---|---:|---:|
| Default surface quadrature relative integral error | 6.1457% | 0.04223% |
| Resolved (128-panel) surface quadrature relative error | 4.45e-12 | 6.78e-14 |
| Consistent nodal minimum, resolved quadrature (m) | −8287.44396 | −759.74597 |
| Lumped nodal minimum, resolved quadrature (m) | 565.480914 | 513.635233 |
| Lumped relative L1 representation error | **65.9692%** | **34.6472%** |
| Integral of h/I_lumped across normal profile: min / max | **0.161583 / 3.636983** | **0.271496 / 2.269670** |

Relative L1 error is integral |I_projected−I_direct| ds / integral I_direct ds.
The normalized-source column is I_direct(x)/I_projected(x), whose target is one;
it also measures the multiplicative pointwise error in h/I at fixed h>0.
Consistent interpolation crosses zero, so it does not define an admissible
normalized source. Lumping provides positive sampled coefficients but remains
poorly resolved even at halved spacing. No accuracy tolerance was loosened and
no arbitrary numerical pass was assigned to these large errors.

For the singular fine field, sampled surface integrals grow from 4449.38 to
12418.00 m² at 0.1 m spacing and from 5756.43 to 13538.32 m² at 0.05 m as panels
increase from 1 to 128. This agrees with logarithmic divergence; it is not a
quadrature convergence study of a finite answer. No finite relative spatial
error or normalization qualification is assigned to this field.

## Integration and small checks

The choice is immutable during a run. Native initialization computes current I_h
with the selected representation before setting previous I_h; subsequent current
values and the existing terminal commit consume that same field. No histories
are reset. A marker present only for lumped mode in the existing serialized
property registry rejects representation changes in either restart direction.
The default registry and old consistent checkpoints retain their layout. Native
parse/restart cache invalidation is retained; probe helpers restore values and
the temporary comparator choice on both normal and exceptional exits.

| Check | Result |
|---|---|
| Native constant reproduction / integral conservation | max absolute errors 2.665e-15 / 5.552e-17; allowances 2e-14 each |
| Native frozen coarse projections at both spacings vs independent calculation | max scaled error 3.03e-14; check allowance 5e-13 |
| Straight current default vs retained P1 phase/history | Exact |
| Lumped straight vs original phase reference; padded control | Pass, unchanged bounds/KKT/healing allowances |
| Straight serial / two-rank phase histories | <=1.11e-16 |
| Current I_h warm/cold cache | Exact |
| Native restart after +5%, distinct old/older phase | Exact subsequent phase histories, serial and two ranks |
| Restart representation mismatch | Both directions explicitly rejected before advancing |
| Pre-change consistent checkpoint (copied, original untouched) | Loads and resumes |
| Default mature frozen boundary-completion fixture | Pass |
| Existing I_h reference, retained 50-iteration budget | Pass; original 10-iteration failure remains historical |

P1 KKT allowance remains 1e-11, phase nonlinear tolerance 1e-12, CG tolerance
2e-7, nodal box exact, CPDI allowance 3e-14, healing allowances 4.922e-12 straight
and 1.298e-11 padded. The original equality failures remain. These phase-only
runs do not qualify the production H update or coupled commit/rollback.
The fine failed-entry healing 4.2683e-11 still exceeds the straight allowance;
normalization positivity does not address it. Original ~1e-16 correction/repeat
uncertainty estimates remain unchanged because the reproduced phase states match.

Build: GCC 12.4.0 core and plugins, Release, deal.II 9.6.2, OpenMPI 5.0.6,
Trilinos 14.2/Epetra, existing Voro build. Capture runs took 16.1/21.9 s;
phase checks about 4–17 s each. One MPI startup was blocked by sandbox sockets;
the authorized local-socket rerun passed. No broader coupled/time refinement,
transport/birth test, 3D or cluster run was launched. Existing Stage-I unit
results are reused for the unchanged pressure-check correction, not newly rerun.

Commands and exact identities are retained in the runners, per-run JSONs and
[manifest](../../../benchmarks/reconstructed_fault/propagation/normalization_evidence/manifest.json).
`check_normalization.py` verifies implementation checks and explicitly records
`P2_normalization_qualified=false`; a successful checker exit does not mean a
successful coupled scientific gate.

## Reproduction and checking commands

Existing labels refuse overwrite; use fresh labels for new executions.

```sh
# The capture pair used the pre-projection-change core and capture observer.
python3 benchmarks/reconstructed_fault/propagation/run_coupled.py normalization_capture025 --dt 0.25
python3 benchmarks/reconstructed_fault/propagation/run_coupled.py normalization_capture0125 --dt 0.125
# Frozen comparisons; output directories must not exist.
python3 benchmarks/reconstructed_fault/propagation/normalization_frozen.py benchmarks/reconstructed_fault/propagation/normalization_evidence/frozen-phase-025.csv /tmp/normalization-coarse
python3 benchmarks/reconstructed_fault/propagation/normalization_frozen.py benchmarks/reconstructed_fault/propagation/normalization_evidence/frozen-phase-0125.csv /tmp/normalization-fine
# Final production representation, unchanged phase-only schedule.
python3 benchmarks/reconstructed_fault/propagation/run_integration.py normalization_lumped --full-support --lumped
python3 benchmarks/reconstructed_fault/propagation/run_integration.py normalization_lumped_resume --full-support --lumped --resume normalization_lumped
python3 benchmarks/reconstructed_fault/propagation/run_integration.py normalization_lumped_mpi_allowed --full-support --lumped --ranks 2
python3 benchmarks/reconstructed_fault/propagation/run_integration.py normalization_lumped_mpi_resume --full-support --lumped --ranks 2 --resume normalization_lumped_mpi_allowed
python3 benchmarks/reconstructed_fault/propagation/run_integration.py normalization_lumped_spacing05 --full-support --lumped --spacing 0.05
python3 benchmarks/reconstructed_fault/propagation/run_integration.py normalization_lumped_padded --padded --lumped
python3 benchmarks/reconstructed_fault/propagation/check_normalization.py
```

The two deliberate mismatch checks and legacy checkpoint check used copies of
output directories, preserving original checkpoints. Their exact commands are
in the per-run JSONs. The compatibility runner is `run_normalization_compat.py`.
Core build: `cmake --build build-propagation-integration --target aspect -j3`;
plugins: `cmake --build build-propagation-integration/plugin --target production_integration coupled_feedback mature_compat -j2`.
The intermediate capture observer binary was rebuilt afterward; its hash and
source identity are recorded. The pre-change core, original 273 pinned artifacts,
and captured states are preserved. Final check binaries are pinned separately.

## Decision required before further P2

The smallest formulation question is the compatibility of admissible phi=1,
g(1)=0 and localization h=1/g−1. Keeping this localization requires a model that
stays strictly unsaturated; introducing a strict phase cap or residual degradation
would be a formulation choice, not a projection tolerance fix. Neither is added.
A reviewed fully damaged-state treatment is required for the finer trajectory.
The finite coarse field additionally requires a better-resolved surface
representation than either tested spacing. No further surface refinement or
new representation is silently selected here.

P1's restricted implementation checks remain complete. P2's cumulative mechanical
feedback conclusion is still inconclusive over the requested interval. Healing,
physical rejected-step rollback, coupled restart, transported/birth histories,
and general support admission remain open. This remains one prescribed
through-going fault with predefined full normal association; no P3 or propagation.
