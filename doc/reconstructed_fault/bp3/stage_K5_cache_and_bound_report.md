# K5 bounded preparation cache and lower-bound audit

The cache/geometry work is verified, but the three-real-step smoke remains
incomplete. The authorized 1e-20 comparison exposed a bound-contact arithmetic
failure before the first step-2 trial residual; no further solve was launched.
The accepted stress-perturbation initialization is retained; Airy is diagnostic only.
No phase profile, quadrature/tail tolerance, support, pressure convention,
Maxwell/history equation, solver tolerance or iteration budget was changed.

## Changes and invariants

BP3 strengthening and initial-Theta bulk interfaces use physical depth:
`xd_equiv=(100000-y)/sin(60 degrees)`. The 15--18 km transition is horizontal
at y=87009.6189432334 and 84411.5427318801 m: 2598.0762113533 m vertically,
exactly 3 km along the sharp fault. The sampled sharp-fault fraction differs
by at most 2.443e-15 and Theta by 2.843e-14 relatively from the original
down-dip evaluation. Deep Vp still uses true fault distance. The existing
surface interface does not provide a clean benchmark override of chemical
fractions, so particle projection remains and is timed explicitly.

`PhaseFieldFault` owns a transient, exact completed-I_h cache. All ranks must
match owned phase entries, fault vertices/versions, a valid mesh lookup and
degradation inputs. Chemical projection is still performed; its values enter
the key unless material degradation laws are identical. BP3 has identical
degradation laws despite its spatially varying friction coefficients.
Cache hits issue zero integration/remote-FE requests. Restart and parameter
parsing invalidate; failed preparation leaves validity false. Mesh deformation
disables reuse. No cache is checkpointed and no analytic I_h replaces a
production integral. The authority documents record this approved extension.

Identical current/previous local phase reuses h. Identical h and I_h makes the
history-localization difference exactly zero. The existing B/G linearization
caches retain their coefficients, including localization, through repeated
actions. Moving parent coordinates and changing surface friction mixtures
are not assumed frozen merely because Eulerian phase is fixed.

Root-only begin/end messages now identify property preparation, material
projection, I_h, and particle/Stokes-QP cache construction. Coarse preparation,
projection, I_h, cache and condensed-solve timers use the normal ASPECT summary.
Fine profiling remains opt-in. The I_h breakdown separates adaptive/request
work, FE lookup/sampling, material/geometry guards, adaptive MPI, final
projection and its MPI reduction. FE timing includes lookup and its communication;
these are not additive independent timings.

## Passing focused evidence

Commands are from the repository root. Core and plugins were built with -j4.

```
c++ -O2 -std=c++17 benchmarks/reconstructed_fault/bp3/check_depth.cc -o /tmp/bp3-check-depth
/tmp/bp3-check-depth
python3 benchmarks/reconstructed_fault/bp3/run.py benchmarks/reconstructed_fault/bp3/cache_final_one.prm --cap 120
python3 benchmarks/reconstructed_fault/bp3/run.py benchmarks/reconstructed_fault/bp3/cache_final_two.prm --cap 120 --ranks 2
build-pf-cpdi/aspect-release --test '[phase_field_fault_cohesive],Stage-I*'
```

The final one-/two-rank runs passed in 7.33/7.18 s. Unchanged preparation
returned bitwise-identical I_h and zero requests; changing phase on rank zero
only forced collective recomputation; geometry replacement invalidated;
singular-profile failure did not publish validity; restoration reproduced the
original values. The checkpoint-load invalidation routine was exercised,
not full checkpoint I/O (the BP3 plugin intentionally rejects restart).
The final test also compares the frozen-localization shortcut with the original
formula at an 8-epsilon relative bound. All 71 assertions in 16 focused
cohesive/Stage-I cases passed, including exhaustion and rollback tests.
The retained coupled-action check (`cache_actions_one.prm`) passed in 12.09 s.

Preserved incomplete setup attempts: `cache_one` omitted the required particles
postprocessor; `cache_valid_one` exposed a stale executable missing the newly
added invalidation symbol. Neither is counted as passing evidence. Subsequent
fully rebuilt tests above passed. No full ASPECT suite was run.

## Saved 1e-12 replay

`cached_smoke` accepted initialization and step 1, then was stopped at the
user's request after 650.66 s (SIGTERM, not nonlinear rollback or convergence).
Peak RSS was 4480432 KiB. Its binary hash and plugin hashes are in
`cached_smoke.resources.json`; recoverable source is `cache-replay-source.tar.gz`
(SHA256 bd843dce0f1ccd0dd8637f5f4288b14601b0161cd80615c88442160aa6980d40).
This snapshot precedes only the final timer/message adjustments, the additional
localization test assertion and the later horizontal initial-Theta extension.

Cold preparation was 87.862 s: adaptive/requests 1.1821 s, FE including lookup
19.5969 s, material/geometry guards 65.7146 s, adaptive MPI 0.0058 s. Warm calls
took 1.3764, 1.4442 and 1.4372 s with zero integration requests, including
fresh friction projection and particle cache construction. The previously
unexplained cold cost is therefore predominantly material/geometry checking,
not missing FE timing. No further cold-path algorithm change was made.

Step 2 had 152 negative all-free-at-Vmin weak-density values throughout the
recorded audit. Thirteen nodes became lower-active; the bound repeatedly
limited accepted alpha (as low as about 2e-5), without Armijo rejection.
Fresh linear checks passed while bulk/surface relative residuals remained
approximately 0.464/0.196. Per-node data are `nonlinear_bounds_2.csv`; accepted
alphas and residuals are in the log and joined by `analyze_cache_audit.py`.
Fmin is evaluated at fixed current bulk state with every unprescribed V set
to the bound. It is not an independent scalar root or a replacement for
the coupled active-set test. This evidence authorizes the requested 1e-20
comparison, not a solver redesign.

## Final comparison and timing

```
python3 benchmarks/reconstructed_fault/bp3/run.py benchmarks/reconstructed_fault/bp3/cached_smoke_low_floor.prm --cap 900
python3 benchmarks/reconstructed_fault/bp3/analyze_cache_audit.py benchmarks/reconstructed_fault/bp3/cached_smoke_low_floor
python3 benchmarks/reconstructed_fault/bp3/analyze_perturbation.py benchmarks/reconstructed_fault/bp3/cached_smoke_low_floor
python3 benchmarks/reconstructed_fault/bp3/bound_roundoff_audit.py
```

`cached_smoke_low_floor` used fresh initialization, the same coarse mesh and
actual timestep controller. Only the authorized rate bound and horizontal bulk
initial-state extension changed relative to `cached_smoke`. All recorded
surface fields at accepted states 0 and 1 are bitwise identical between these
two runs (including V, Theta, C, I_h, slip and backgrounds). The final normal
timer/progress edits and localization-equivalence assertion were rebuilt and
tested before this replay.

| Accepted state | Physical time (s) | Free V/Vinit | Actual sigma_n range (MPa) | Bulk / surface relative residual |
|---|---:|---|---|---|
| Initialization | 0 | 0.987149--1.016008 | 49.829053--50.073433 | 4.744e-15 / 2.099e-11 |
| Step 1 | 2624650.7189074 | 1.000144--1.018971 | 49.886348--50.048185 | 5.081e-13 / 9.174e-11 |

Actual mean normal stresses are 49.99999321 and 49.99999382 MPa. The supplied
Theta0 and zero initial Maxwell history are retained exactly. Step-1 Theta
matches independent exact aging to 2.22e-16 relative; maximum committed
particle stress is 237601.710 Pa. Deep Vp is exact, with 400 free and zero
lower-active nodes at both accepted states. Geometry, phase, backgrounds and
I_h are unchanged. All seven returned linear directions pass fresh checks,
including the first direction of unaccepted step 2.

The run failed after 257.35 s, peak RSS 4445324 KiB (4.239 GiB), before
accepting step 2 at t=5241669.3686994 s. This is not a successful three-step
runtime and is not comparable to the wall time of an interrupted long solve.
The final normal timer summary is unavailable after the abort; do not infer
a complete assembly/linear-solve breakdown from the remaining wall time.

| Preparation measurement | Before value cache | Final cached run |
|---|---:|---:|
| First cold I_h (s) | 83.33 | 83.515 |
| Later I_h calls (s, each) | 67.37, 69.27, 67.23 | 1.422, 1.447, 1.445 |
| Four I_h calls through step-2 entry (s) | 287.20 | 87.829 |
| Later complete property preparations (s) | dominated by the above repeated integrations | 1.425, 1.449, 1.448 |

Repeated I_h preparation is **46.5--47.9x faster**, and total I_h preparation
through the same number of calls is **3.27x faster**. These are measured scoped
times, not an end-to-end three-step speedup. The earlier run's bulk friction
extension differed, but its degradation laws, phase mesh and integral problem
were the same; initial I_h agrees to 4.30e-16 relative. The focused cache tests
also compare freshly recomputed and cached values bitwise in the same fixture.

The final cold breakdown is 1.118 s adaptive/requests, 18.536 s FE+lookup,
62.464 s material/geometry guards, 0.0056 s adaptive MPI, and 1.391 s surface
material projection (including its cache build). These account for 83.514 of
83.515 s; final profile projection and remaining key/control work are below
0.001 s combined. Nested scopes are not added twice. Particle-cache builds
total 5.668 s across four preparations; Stokes QP cache builds once in 0.443 s.
Total property preparation is 88.197 s, a subset of total run time. Other
assembly/solve/output work and setup account for the remainder; its precise
subdivision needs a normally completed run, not an invented allocation.

The additional owned-phase/geometry key is about 0.31 MiB on this rank. The
large geometric lookup cache already existed; this patch does not duplicate
its 25.9 million retained requests. Per-call exact key comparison adds a
similarly sized temporary vector. No all-profile or fine-pilot run was added.

Coarse support limitations remain unchanged: 83 independent columns give
maximum omitted h fraction 1.49449e-4, tip supported-normalization error
6.526%, and interior sampled maximum 3.51e-4. Global Q3 supported-slip errors
are 1.51448e-4 and 1.51822e-4 at accepted states 0/1. The independent old-formula
history-term replay has only 1.4--2.1e-22 m^2/s of cancellation noise; production's
identical-profile shortcut is zero. No tail, support or criterion was changed.

## Newly exposed blocker: represented bound contact

At Vmin=1e-20 the first step-2 bound audit has **zero** negative Fmin values
(minimum weak-density Fmin=8.4934 MPa), versus 152 at 1e-12. Thus the lower
floor removes that measured inadmissible-root preference. It does not yet
demonstrate nonlinear convergence: the trial fails the existing rate guard.

Saved node 1154 gives:

| Quantity | Value |
|---|---:|
| Base V | 1.018971212482027e-9 |
| Newton dV | -8.573688417268437e-9 |
| Computed alpha_max | 0.11884864050105866 |
| Represented V + alpha*dV | 9.99999322030669e-21 |
| Deficit below 1e-20 | 6.779693309633094e-27 |
| epsilon times base V | 2.262570603055518e-25 |
| Existing snap tolerance | 2.220446049250313e-34 |

The 65-digit offline replay separates rounded contact-alpha error from
addition/product cancellation. The exact affine value of the *represented*
inputs is 9.9999726830861951e-21. The snap check scales with the tiny result,
although the arithmetic cancels operands of order 1e-9. This is a reproduced
bound-contact arithmetic defect, not evidence of an erroneous constitutive
Jacobian or a failed returned linear direction.
Fused multiply-add alone cannot fix the rounded contact fraction; even the
exact affine value of those represented inputs is below the bound. Moving
alpha down by one representable value also leaves the original multiply/add
result below the bound in this captured case.

Merely snapping that one value is insufficient: both residual orchestration
and acceptance convert the validated absolute trial to a difference, and
`set_slip_rate_trial` adds that difference back. Even `V + (Vmin - V)` returns
9.99999322030669e-21 for these inputs. That second reconstruction must preserve
the actual evaluated trial too. The production exception follows the existing
rollback path; no step-2 history was committed. Post-rollback vectors were not
exported in this run, so rollback verification rests on the focused tests,
not absence of output files.

**Smallest proposed separate correction (not implemented):** construct
bound-contact candidates using the known admissible fraction-to-boundary
relation without cancellation, and pass their validated absolute nodal values
through temporary evaluation and acceptance without subtract/add reconstruction.
Keep the existing local active-set tolerance, fraction formula, Armijo budget,
residual criteria, prescribed-node constraints and history-publication timing.
Regression: use these captured inputs; verify exact contact and equality of
evaluated/accepted trial values, preserve rejection of genuinely infeasible
steps, and rerun bound-release/exhaustion/rollback checks before the same smoke.

This requires a narrow solver/manager trial-value correction outside the
approved preparation optimization. No such correction or retry was made.
The preparation-speed gate passes, the second-real-step convergence gate does
not. **The 48.8 m server pilot remains unlaunched.**

## Files and provenance

The cache and profiling changes are in `include/aspect/material_model/phase_field_fault.h`,
`source/material_model/phase_field_fault.cc`, `include/aspect/phase_field.h`,
`source/simulator/phase_field.cc`, `source/reconstructed_fault/manager.cc` and
`source/simulator/solver.cc`. The phase handler exposes only a degradation-law
configuration revision; phase values themselves are compared exactly. The
existing pointwise material/simulator assembly boundary is unchanged.

BP3 changes live in `bp3.cc`, `bp3_model.h`, `run.py`, `check_depth.cc`, the
new cache/audit wrappers, `analyze_cache_audit.py`, `bound_roundoff_audit.py`
and the adjusted `analyze_perturbation.py` (reads the actual configured bound).
Tests use `tests/phase_field_fault_ih_cache.cc`, the existing private test-access
header and the existing surface-dynamic plugin translation unit. Authority
updates are in `current_design.md` and `specification.tex`; benchmark notes and
the Stage-K progress record point to this report. Earlier unrelated and BP3
stress-background modifications were preserved, not reset or recommitted.

The final executed binary/plugin/config hashes are recorded in
`cached_smoke_low_floor.resources.json`. `cache-final-source.tar.gz` preserves
the relevant final source/config/test files; its SHA256 is in
`cache-final-source.sha256`. Changes remain uncommitted. No source from the
proposed bound-contact/publication correction is present in this snapshot.
