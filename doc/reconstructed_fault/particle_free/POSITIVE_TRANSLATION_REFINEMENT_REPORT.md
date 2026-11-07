# Positive-translation constant-viscosity refinement

The two recommended fresh cases completed successfully. At fixed 64² bulk
resolution and .05 m initial label spacing, halving dt from .025 to .0125 s
reduces the endpoint energy remainder from **8.7411% to 4.4151%** of active
work, with measured halving order .9859 in its accumulated absolute value.
Surface refinement primarily changes the sampled initial surface data and
coupled traction/slip; it does not remove this energy remainder. Raw boundary
traction work worsens under the timestep halving and remains unqualified.
No native implementation, physical law, solver tolerance or endpoint convention
changed. This completes the selected two-run experiment, not all of Stage C.

## Provenance and experiment

Entry root is `aspect-pf-rsf-particle-free`, branch `pf-rsf-particle-free`,
revision `69e8a10a93a08718d31af961197899e320657c04`, with clean tracked files.
All 929 native source/build inputs and six binaries match the prior qualified
manifest. The retained executable prints its pre-commit build revision
`27178813c`; actual hashes, not that label or current HEAD, identify the code.
No rebuild or production checkpoint import was needed.

[Protocol and evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/positive_refinement/README.md),
[comparison JSON](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/positive_refinement/comparison.json),
[coupled work JSON](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/positive_refinement/coupled-work.json), and
[figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/positive_refinement/positive-refinement.png)
record inputs, norms, bounds, clocks and cost. The exact runner, command, input,
core/checker hashes and outputs stay under each case's local `material-runs/`
directory. Both new runs use four ranks sequentially, one thread per rank.

All cases use G=10 Pa, eta=10 Pa s, total material Jaumann evolution, bounded
MacCormack Q1 profile/obstacle transport, accepted-trace tensor inflow, full
normal association, growth disabled and the existing coupled endpoint solve.
The box is [-2,2]² m, with +.4 m/s translation on left/right/bottom and the
existing unloaded natural top. Initial curve, compact oblique phase, background
traction and age functions are unchanged; initialization interval is .1 s.
Only initial surface-label spacing and physical timestep vary. This remains
the mature material formulation, distinct from cohesive energy and production
BP3. No resampling, birth, moving AMR or deformed cohesive shape work is enabled.

| Case | Initial label spacing (m) | dt (s) | Physical steps to .2 s | Wall (s) | Linear solves / iterations |
|---|---:|---:|---:|---:|---:|
| retained `bpf-c-p-m-ref` | .10 | .025 | 8 | 120.54 | 27 / 1056 |
| new `bpr-c-p-m-surface64` | .05 | .025 | 8 | 106.10 | 26 / 1013 |
| new `bpr-c-p-m-halfdt64` | .05 | .0125 | 16 | 180.08 | 42 / 1669 |

The retained reference also emitted one 2.78e-17 s terminal interval; it remains
in raw data and cost but is excluded from physical reductions. Both new runs
finish their intended schedules without such an extra interval. Counts include
initialization. New timestep-halving cost is 1.70 times the .025 case; this is
one bounded timing pair, not a production scaling estimate. The old/new
spacing timings do not establish a speedup.

## Surface and temporal sensitivity

The first column below compares .10 to .05 m spacing at dt=.025. The second
compares dt=.025 to .0125 at .05 m spacing. These are pairwise differences,
normalized by the finer member, not errors to an exact continuum solution.
Bulk tensors use full 3D plane-strain L2 norms on common Gauss-4 quadrature.
Surface scalar fields are compared in nodal l2 after interpolation in the
fixed material parameter onto the finer label grid. No physical state is
resampled. Point traction uses segment Gauss-3 quadrature on that common
parameter; work traction uses its independent Gauss-8 volume/mass adjoint.

| Final quantity | Surface refinement (%) | Time refinement (%) |
|---|---:|---:|
| Profile L2 | .00000869 | .003565 |
| Bulk stress L2 | .51960 | .86138 |
| Point shear traction L2 | .60115 | .04422 |
| Work-conjugate traction L2 | .90304 | .02577 |
| Normalization I | .35186 | .002930 |
| Slip rate | .66976 | .05633 |
| Endpoint-integrated cumulative slip | .66977 | .02039 |
| Age | .04278 | .00007475 |
| Auxiliary memory Z | .66977 | .02062 |
| Final physical bulk storage | .96274 | .02325 |

Absolute bulk stress L2 differences are 2.5911e-6 and 4.2960e-6 Pa m;
geometry maxima are 7.7332e-8 and 5.3266e-9 m. Absolute field errors and
zero-reference cases are retained in the JSON; zero initial stress/memory is
not assigned a spurious relative error.

The two timestep runs have identical reconstructed initialization fields.
The surface-spacing pair already differs at initialization by .04357% in
age, .35327% in I and .66883% in slip rate, despite identical bulk phase and
zero committed stress/memory. Thus the final .66976% slip-rate difference is
largely an initial surface-data discretization effect, not accumulated
transport error. A third surface grid would be needed to establish its order.
Neither pair proves absolute sub-percent stress or traction accuracy.

At .05 m spacing, peaks are .34885278/.34882609 for dt=.025/.0125, normal
variances .030677635/.030677589 m², and mass drifts +.00006976%/−.00002899%.
Normal support loss is zero; final integral normalization errors are
.00665757%/.00665777%. No normalization floor, phase cap or mass correction
was introduced.

## Energy and material flux

The physical mature energy is |tau|₃²/(4G); auxiliary Z has no cohesive energy.
The active scale integrates absolute bulk mechanical, slip, Maxwell,
dilatation and material-energy flux terms. The much larger cancelling
background/friction work is recorded separately. Friction uses incoming
committed age, consistent with the native endpoint solve.

| Case (spacing / dt) | Endpoint signed / accumulated absolute (J/m) | Endpoint / trapezoidal relative (%) | Trapezoidal signed / accumulated absolute (J/m) |
|---|---:|---:|---:|
| .10 / .025 | −6.90349e-10 / 6.90349e-10 | 8.74125 / .072443 | 1.42415e-12 / 5.72126e-12 |
| .05 / .025 | −6.97043e-10 / 6.97043e-10 | 8.74106 / .072403 | 1.45232e-12 / 5.77367e-12 |
| .05 / .0125 | −3.51935e-10 / 3.51935e-10 | 4.41508 / .044472 | −2.55168e-12 / 3.54498e-12 |

Every physical endpoint remainder is negative; cancellation does not hide its
size. The trapezoidal sum does have cancellation and even changes sign in the
timestep pair. Its smaller absolute sum does not change the endpoint scheme
or establish second-order coupled integration. The endpoint halving order
is .9859 (absolute) / .9854 (relative), from one fixed-space halving only.
The full pressure-aware coupled endpoint sums agree to the measured surface
work residual. The final case still exceeds a hypothetical 1% active-work
screen; no tolerance or acceptance rule was changed to turn this into a pass.
Gross-work normalization would give only .000139% for that case and would
conceal its active-energy error.

For the new .025/.0125 runs, incoming material-energy integrals are
−1.67169e-11/−1.88336e-11 J/m, outgoing integrals
4.50991e-12/4.41537e-12 J/m and net outward integrals
−1.22070e-11/−1.44183e-11 J/m. The temporal net-flux difference is .027741%
of the finer active scale; the surface-refinement difference is .001131%.
These fluxes are independently integrated, not inferred from ledger closure.

New physical-step surface work residuals are at most 1.93e-17 W/m. Final
Gauss-12 versus Gauss-8 storage differences are 2.775e-14/3.079e-14 J/m
(less than .00050% of storage); surface-work differences are at most
3.48e-13 W/m. The final projected-minus-physical storage inventories are
−1.5333e-12/−1.5321e-12 J/m. That fixed-state projection inventory is not a
measure of the whole accumulated transport defect. The native quadrature
identities and the higher-quadrature continuum estimates remain distinct.

## Boundary traction remains unresolved

| Case (spacing / dt) | Final raw face power (W/m) | Final weak reaction power (W/m) |
|---|---:|---:|
| .10 / .025 | −3.10268e-6 | 1.61e-20 |
| .05 / .025 | −3.11469e-6 | 5.45e-21 |
| .05 / .0125 | −4.19177e-6 | 9.04e-21 |

Halving dt increases the raw face-power magnitude by 34.6%, despite improved
endpoint energy accounting. In the .0125 run, −4.19174e-6 W/m is rigid
translation times the raw integrated face force; relative-velocity power is
only −3.3031e-11 W/m. The weak reaction work for the declared pure-translation
boundary problem remains near zero. This tests only its power identity, not
all independent weak residual components or a converged raw traction trace.
The result neither closes the boundary-traction problem nor identifies its
cause. It must not be interpreted as missing applied energy and compensated
by an invented load or dissipation.

## Verification, preservation and next action

Both cases pass native acceptance/publication checks with exactly 8/16 history
and geometry commits and zero retries. Each records 46 summed label/step cell
crossings. Independent profile replay is within 4.996e-16; donor-bound
violation is at most 5.551e-17 (limit2e-11). Age source agreement is exact in
the exported arithmetic and memory error is at most5.421e-20 (limit1e-11).
The new comparison reducer agrees with the existing same-grid metrics to
2.760e-17 (bound2e-10). Existing rejection/restart and unit evidence applies to the unchanged source;
those suites were not redundantly rerun, and no new lifecycle coverage is
claimed. Native binaries, retained reference data and protected manifests
remain unchanged. Original source, outputs/checkpoints and manual work remain.
The recorded 39/45-frame and temporary-log cleanup losses are not repaired
or hidden. No server, production restart or deferred physical campaign ran.

The next justified task is a read-only boundary-reaction diagnosis on these
accepted states: independently assemble weak residual/reaction components
using the same physical stress, pressure, quadrature and constrained velocity
space; compare them with per-face traction resultants and jumps. Determine
whether the discrepancy comes from stress observation/projection, boundary
assembly/data, or unresolved trace convergence before another trajectory or
solver change. The unrelated cohesive endpoint-energy and localized-tip
failures remain as documented in STAGE_C_COHESIVE_DIAGNOSIS.md.
