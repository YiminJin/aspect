# P1 production integration: qualified for the selected fixed-support controls

**Normalization follow-up (2026-10-06):** default compatibility and optional
I_h-only lumping pass the small straight/padded, cache, serial/two-rank and
changed-history restart checks. This does not qualify normalization accuracy
for P2's highly concentrated fields. See [normalization report](normalization_report.md)
for the exact-saturation limitation and measured spatial errors.


2026-10-06 support-policy follow-up. The user selected a fixed association region
covering the whole normal cross-section for the single straight through-going
verification fault. **The affected P1 gate now passes.** The padded finite seed
retains its original support and cap treatment; prior successful evidence is reused.
The historical equality-only failure and narrow-strip omission remain below.
P2 followed automatically and exposed a different normalization-projection failure;
see [P2_report.md](P2_report.md). P1 does not qualify mechanical H, sampled-H
accuracy, general support admission, transported histories or propagating tips.

## Fixed-support implementation and recheck

`Fault reconstruction / Association support = full normal cross-section` is an
explicit default-off verification option. The default `stationary profile` path
is unchanged. The new option requires one straight fixed prescribed 2D fault in
a Box, spanning the entire tangential domain extent, with legacy completion.
Maximum normal distance is computed over owned mesh vertices and reduced across
MPI; 64 double epsilons times the fault coordinate scale includes boundary
roundoff. Actual half-width is **0.50000000000001588** for the centered unit box.
No physical profile, sharp endpoint, phase boundary or H initializer changes.

The manager commits these widths during fresh reconstruction and builds particle
and production Stokes-QP association caches before any cohesive initialization.
It reuses native initialization of q and previous I_h from the relaxed field.
The new widths intentionally affect that projection's eligibility; q is not
required to equal its former narrow-strip average. Particle H is captured before
phase relaxation/association and is unchanged exactly afterward. Phase histories
are bitwise identical to the former complete straight trajectory at all five
steps. Initial H also matches exactly between serial and two ranks (9,216 parents).

Widths use existing manager serialization; no checkpoint schema was added. Both
serial and two-rank native restart after the +5% increment preserve distinct
old/older phases and reproduce steps 3/4 exactly. Boundary probes at y=0 and y=1
(and 1e-15 beyond), all parent associations, actual cached Stokes QPs, cache reuse
and post-restart reconstruction are checked. The selected region is fixed for
all steps; there is no new admission event, phase threshold or runtime growth.

| Affected straight control | Serial | Two ranks |
|---|---:|---:|
| Maximum change versus prior narrow phase histories | 0 | <=1.11e-16 versus serial |
| Native initial H change through relaxation/association | 0 | 0 |
| Unassociated particle/mechanical-QP samples | 0 | 0 |
| Omitted normalized source | 0 | 0 |
| Bounds, KKT, holds, perturbed identical-H repeat | Pass | Pass |
| Native restart, phase histories at steps 3/4 | Exact | Exact |

The full quadrature sum after +5% is 0.99999329484756 in serial. Its approximately
6.7e-6 mechanical integration error is retained separately from the now-zero
association omission. No tolerance was increased. Earlier contact/release,
initialization failure, publication, scratch poisoning/restoration, padded-seed,
frozen compatibility and equality-failure evidence remain applicable.

The support core was built with GCC 12.4.0 (as were the final observers), SHA256
`13b178809f053d9d84319b7d4e391cbfe52e1c965685004ef344c15406d25d0d`.
Subsequent P2 changes affect only mechanical fresh-linear-residual verification
and its tests, not this phase/association evidence. Final P2 core identity is
in the P2 report. Full runs took 7.7–7.8 s; restarts 2.9–5.1 s, under 243 MiB
runner-reported RSS, with about 1.95 MB checkpoints. All work remains local.

[Independent full-support checks](../../../benchmarks/reconstructed_fault/propagation/integration_evidence/full_support_checks.json),
[commands](../../../benchmarks/reconstructed_fault/propagation/INTEGRATION.md).
The initial analytic H re-evaluation diagnostic was replaced by a stronger exact
native-initializer snapshot check after exposing a 2.68e-8 discrepancy at
H≈150141 in the reconstructed expectation. That expectation used a nominal core
value rather than preserving the complete native interpolation/evaluation sequence.
The exact snapshot check establishes unchanged H without duplicating its initializer.
Failed diagnostic attempts are retained.

The following sections retain the original integration audit and evidence;
the original support blocker is historical and superseded only for this
explicitly selected straight verification policy.

## Scope, provenance and implementation

Existing worktree `/home/ein/repository/aspect-fault-propagation`, branch
`fault-propagation`, HEAD `de7beaa65846c7372711f7922abf1c5d9a745ee6` plus the
preserved incoming P1 work and this uncommitted integration. No worktree switch,
commit, push or sibling build modification. Root AGENTS instructions are
preserved, with the current P1/conditional-P2 authorization appended.

New core: `build-propagation-integration/aspect`, SHA256
`a26571143d2731e6f04e0b45e9ae365468d415a58e5998eaa9c66ed5f16a3325`.
GCC 12.4.0, deal.II 9.6.2, OpenMPI 5.0.6, Trilinos 14.2/Epetra, Voro 0.4.6,
World Builder 1.0.0, Release/unity; NetCDF off. Configure did not enable floating
point exceptions. Both core and final observer use GCC 12.4. Earlier observer
builds accidentally inherited MPI-wrapper GCC 16.2.1; that was identified,
corrected, and the support/serial/MPI/restart controls repeated. The retained
complete straight trajectory used that earlier observer with this same core;
its support failure is independently reproduced with the final compiler.

| Files | Responsibility |
|---|---|
| `include/aspect/phase_field.h`, `source/simulator/phase_field.cc` | Optional `Solver parameters/Phase field solver parameters/Use bound constraints = true`; default false. Native CPDI energy output, projected Newton, symmetric distributed active-increment elimination, collective KKT and line search; native CG/AMG retained. |
| `source/simulator/solver_schemes.cc` | All three existing phase callers publish only the accepted phase block to `current_linearization_point`; no history-vector synchronization writes. No-Stokes path prepares dependent data after composition setup. |
| `include/aspect/material_model/phase_field_fault.h`, `source/material_model/phase_field_fault/history.cc` | Narrow phase-only normalization/initial cohesive preparation using existing initialization and paired commit. Later/resumed calls require initialized history and recompute current I_h. Mechanical history formulas are untouched. |
| `tests/phase_field_test_access.h` | Existing test friend exposes native energy, bounded scratch solve and active-system elimination for checks. |
| Propagation benchmark files | Production-path observer, fixture, runner, numerical checker and retained evidence. No independent trajectory Newton solver replaces production. |
| Design/specification, propagation plan/status/reports | Record the selected algorithm, audited ownership and unresolved support decision. |

The mature frozen-field bypass and equality path remain. The bounded path reads
H and compositions from the same particle representation and uses the supplied
trial vector for residual, tangent, energy and roundoff evaluation. Newton and
line-search trials remain scratch state. Failed convergence does not publish
an intermediate phase. The box is [0,1], with no previous-phase lower bound.

## Phase-state and initialization audit

`Simulator::advance_time()` copies the completed timestep-zero solution into
both old FE vectors, and subsequently shifts old into older and solution into
old. Consequently `old_solution` holds the last accepted phase at the next
phase-solve entry. `solve_timestep()` first calls
`initialize_current_linearization_point()`: source assigns old to the predictor,
then, for timestep >1, uses `sadd(1+dt/old_dt, -dt/old_dt, older)` on owned
copies. These are verified source assignments, not a recalled formula.
The bounded solver explicitly selects the old accepted phase and never consumes
that extrapolated phase block. After success, each simulator caller refreshes
only the predictor's phase block. Neither old vector is overwritten.

Fresh ordering is bulk initialization/pressure → particle generation and native
prescribed-profile H initialization → rounded scratch guess → converged phase
publication → initial fixed sharp geometry → final composition setup → native
surface mixture/current I_h → initial cohesive q and previous I_h together.
The guess uses native material profiles; it never recalculates particle H.
All selected fixtures have zero initial adaptive cycles. Later mechanical
initialization continues through the existing preparation path.

The timestep-zero constitutive special case still evaluates converged phi as
both current and previous without modifying `old_solution`. At later times it
reads the real previous phase. Native checkpoint writing follows vector rotation.
No new persistent phase/H copy or production checkpoint schema was introduced.

Current I_h is recomputed from accepted phase. Chemical surface properties are
projected before integration. Initial q samples the accepted FE phase at
particle positions and uses native H/degradation. Initial q and previous I_h
remain unchanged during these phase-only ramps; only current I_h changes.
Normalization's exact phase-value cache key is separate from geometry-only
point/cell and particle/Stokes-QP association caches. Restart invalidates
normalization caches; cold recomputation agrees exactly with warm values in
each tested partition. Frozen-mature restart restoration is not used here.

Scratch tests poison live phase, every predictor block and both older FE
vectors while explicit trial phase, particle H/material/domain inputs stay fixed.
Residual, Jacobian entries, energy and roundoff allowance are identical.
Normal/exception restoration of FE vectors, particle/property data and domain
geometry version is checked (including the final two-rank repeat); a one-iteration injected phase failure
leaves the supplied accepted field unchanged. A singular-normalization probe
with uninitialized q/previous-I_h sentinels confirms no partial cohesive-history
commit. This is initialization/scratch failure handling, not physical-step rollback.

## Numerical results and actual tolerances

[Machine-readable checks](../../../benchmarks/reconstructed_fault/propagation/integration_evidence/summary.json)
separate the algebraic results from the failed support gate. No tolerance was
increased. Relative phase tolerance is 1e-12, CG tolerance 2e-7, Newton limit
100, line-search limit 35, Armijo factor 1e-4. The absolute target is the larger
of relative entry-KKT tolerance and native roundoff allowance, typically
3.9e-13–4.9e-13 in these trajectories; every explicit KKT check also requires
free residual, dual violation and complementarity <=1e-11. Raw nodal bounds
are exact; the CPDI observation allowance is 3e-14, retaining the previous
2.842e-14 weight-roundoff evidence. Normalization quadrature and tail tolerances
remain 1e-8 each.

| Measurement | Straight serial | Padded serial | Padded two ranks |
|---|---:|---:|---:|
| Relaxed maximum phi (minimum 0) | 0.8084775463 | 0.8073406845 | 0.8073406845 |
| Maximum KKT defect | 1.616e-13 | 4.410e-15 | 2.086e-15 |
| +5% maximum change | 0.002153985 | 0.005786104 | 0.005786104 |
| +10% maximum change | 0.002123180 | 0.005560932 | 0.005560932 |
| Maximum incremental healing | 0 | 1.794e-20 | 7.758e-22 |
| Maximum retained-reference cut discrepancy | 8.549e-15 | 1.960e-13 | 1.961e-13 |
| Changed-history native restart | Not needed for matrix | Exact | Exact |
| Normal-strip coverage after +5% | Fails active-sample check | Pass | Pass |

Holds show zero phase change. Initial-guess relaxation is excluded from healing.
The old numerical allowances remain 4.922e-12 (straight) and 1.298e-11 (padded).
At +5%, directional Jacobian corrections are 1.975e-14, 7.295e-16 and 4.574e-16;
identical-H perturbed repeats differ by 2.257e-14, 7.216e-16 and 4.441e-16.
These support numerical uncertainty estimates, not rigorous error bounds or
proof of general irreversibility.

The native checkpoint is after +5% (completed step 2, saved counter 3), before
the next hold. Save and resume both require old and older phase to differ by
>1e-4. Restored finite vector/particle/property data and initialization markers
match exactly before a solve; resumed current/old/older nodal phase matches the
uninterrupted steps 3 and 4 exactly. Same rank count and partition only.
Serial/two-rank phase-history differences are <=7.495e-16. Manufactured lower
and upper contacts, release, and symmetric elimination pass on the actual
assembled matrix, including 50 nonzero off-rank active columns. Physical upper
contact, AMR, empty-owner and broad distributed configurations remain unqualified.

## Historical support diagnosis: resolved by the fixed verification policy above

`determine_initial_fault_support()` reads stationary-profile endpoints and
adds one global maximum cell diameter only to the **reconstruction search
radius**. Projection half-widths themselves remain profile widths, interpolated
onto resampled fault vertices. With `Fit prescribed geometry to phase field =
false`, geometry and these widths are retained: the relaxed field does not
recompute them. The measured width is 0.31164820986593172. Inside/outside
projection probes verify the actual manager strip, not only the profile formula.

The straight relaxed state and first hold have no missed active samples. After
+5%, missed mechanical-QP phi is 5.6989883e-5 and missed particle CPDI phi is
5.3617384e-5, versus activation 1e-6. Using native degradation/localization,
actual Stokes quadrature and the current projected I_h, the omitted normalized
source is 7.2555068e-7; its fraction of the full quadrature sum is 7.2555554e-7.
The full sum is 0.99999329485: existing mechanical quadrature error is also
visible and must not be confused with this additional support omission.
The omission exceeds algebraic uncertainty and the 1e-8 normalization integration
controls; it is not evidence of catastrophic mechanical error. There is no
established acceptance budget here for discarding an active evolved tail.

Phase CPDI assembly and adaptive normalization evaluate beyond this fixed strip;
particle constitutive associations and mechanical source quadrature do not.
`compute_history_candidates()` (in material history implementation) starts
from stored H and applies the cohesive-work candidate only to associated
particles. Expanding the strip therefore changes which particles receive that
feedback. A support-update policy must specify admission, cache invalidation,
retention of existing histories and initialization/interpretation of newly
associated material. Silently padding the radius, clipping the tail, changing
activation, or accepting a new truncation tolerance would choose that policy
without resolving its constitutive implications. None was done.

Padded normal-strip coverage passes; its 2,268 active cap quadrature points
beyond the true sharp endpoints are reported separately. They are the initialized
diffuse cap, not a transverse-width failure or new propagation. No fictitious
sharp-fault continuation was introduced. The sampled-H transition and L_tail
accuracy question remains independently open as in [P1_bounds_report](P1_bounds_report.md).

**Historical decision request, now resolved for these controls:** define how normal-profile support admits an evolving
transverse tail while preserving existing histories, or explicitly define and
justify a retained-strip truncation approximation. The current evidence does
not qualify either choice. After that decision, rerun the affected P1 gate;
then the already-authorized first P2 package can proceed without another
planning approval.

## Compatibility, cost and limits

- Equality controlled-H regression still rejects physical admissibility; its
  negative-phase/healing evidence is retained, not reclassified as a pass.
- The untouched reference executable and new executable both exhaust the I_h
  fixture's original ten-iteration limit, with identical printed iteration
  histories; downstream geometry is absent. This is a pre-existing failure.
  A diagnostic 50-iteration overlay passes `Distributed I_h: verified` with
  unchanged tolerance; fixture/default files are unchanged.
- Mature boundary-completion/frozen-field regression passes both default
  equality selection and explicit bound selection.
- All attempts, including parser registration, diagnostic NaN comparison,
  matrix-proxy expectation and ghost-vector norm fixes, are retained. MPI first
  failed at sandbox socket setup and ran after the tool's execution approval.
  These fixes change checks, not scientific acceptance thresholds.
- Final selected production runs take roughly 3–14 seconds, under 270 MiB
  runner-reported child peak RSS. Diagnostics remain below 10 MiB per output;
  padded checkpoints occupy about 3.93 MB. Build artifacts are separate.

[Commands, inputs and evidence](../../../benchmarks/reconstructed_fault/propagation/INTEGRATION.md).
The original integration package ran no mechanics, RSF evolution, transport/birth
qualification, physical rejected-step rollback, coupled restart or propagation.
P2 has since run as described above. See
[P2_report.md](P2_report.md) for the gated handoff.
