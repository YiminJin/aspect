# K4.3 — nonuniform preflight stopped at existing accuracy

## Decision

**Nonuniform finite-width sensitivity remains unverified at the existing K2 accuracy.**

This is outcome 4 of `stage_K4_3_instructions.md`. No ASPECT or reference
trajectory was run. The existing data do not establish a temporal uncertainty
estimate for the requested nonuniform width contrast through 1 s. Stop at
preflight; do not start a K2 convergence campaign or infer zero width sensitivity.

K4 can be closed as a bounded, partially resolved investigation: homogeneous
post-transient width effects are distinguished from normal discretization in
K4.2, while the early transient and additional nonuniform sensitivity remain
unresolved. This is not an all-time K4 pass, a sharp-fault limit, or a Gate-K2
pass. No Stage K4.4 experiment or subsequent stage is started here.

## Reused evidence and numerical uncertainty

The inputs are the domain-rule K2 spatial and completed temporal audits under
`benchmarks/reconstructed_fault/uniform_shear/nonuniform/`:

- `domain-convergence/spatial-audit.json`: 32x128, 64x256, 128x512 at dt=.5 s;
- `domain-convergence-completion/temporal-audit.json`: dt=.5/.25/.125 s at
  128x512, with identical discrete initial data;
- their existing measurement reports, resolved `parameters.prm` files, and
  `stage_K2_2_domain_convergence.md` / `stage_K2_2_temporal_completion.md`;
- `stage_K2_stress_transfer_timeline.md`, to retain the scope of the later
  transfer correction rather than silently treating the old sequence as a
  current-implementation convergence study;
- K4 A/C inputs, results, and provenance under `finite-width/`, plus
  `stage_K4_2_result.md`.

The following numbers are read directly from the saved audit's `adjacent`
entries at common physical times. They are differences between coupled K2
solutions, **not** measured two-width bump-minus-control differences. RMS
norms use the existing Q1 arclength integration. Actual weak traction is
`particle_q_Q1 = M^-1 Q`, not a bulk-column mean; C below is retained cohesive
traction (`C_retained`).

| t=1 s observable | Total RMS, dt .5--.25 | Total RMS, dt .25--.125 | Mean-removed RMS, dt .5--.25 | Mean-removed RMS, dt .25--.125 |
| --- | ---: | ---: | ---: | ---: |
| Weak q (Pa) | 21.7930 | 9.59347 | 2.81087e-4 | 6.53755e-4 |
| V (m/s) | 9.68176e-5 | 3.59395e-5 | 1.80596e-7 | 9.29413e-8 |
| C (Pa) | .195820 | .0858567 | 2.66630e-4 | 3.42427e-4 |
| Theta (s) | 2.45582 | 1.06231 | .0450937 | .0224697 |
| Slip (m) | 2.17443e-5 | 9.60508e-6 | 2.86333e-8 | 3.69224e-8 |

Total differences shrink, but the q/C/slip anomalies grow by factors
2.326/1.284/1.289. Even at .5 s, where q's anomaly difference improves
(5.18622e-4 to 6.68639e-5 Pa), V's increases (1.40065e-7 to 1.51250e-7 m/s).
Thus shortening the review to the first real step does not qualify all the
requested quantities. The earlier signed-cancellation accounting is reused,
not repeated: cancellation explains a nonmonotone difference without proving
an asymptotic error estimate or identifying a production defect.

Spatial weak-q anomalies do contract: 32--64 to 64--128 gives
1.53750e-3 to 3.58263e-4 Pa at .5 s and 1.18580e-3 to 2.48890e-4 Pa at 1 s.
Those grids refine bulk tangential and fault resolution together; they do not
independently qualify a 32-tangential, fault32 A/C comparison. Initial Theta
projection errors are .0973196/.0212798/.00514082 s. They remain physical inputs
to their respective trajectories, not errors removed by initial subtraction.

The uncertainty is not confined to independent open endpoints. At 1 s the
temporal q anomaly difference grows in the same fixed K2 interior
s in [.0625,.1875] m, from 3.08513e-4 to 6.79312e-4 Pa RMS, and in the outer
regions from 2.50680e-4 to 6.27158e-4 Pa. Left/center/right differences change
from (-3.23173e-4, 5.35763e-4, -3.14199e-4) to
(-8.59680e-4, 1.04506e-3, -8.42156e-4) Pa. No endpoint exclusion makes this a
demonstrated temporally resolved interior reference.

Supporting raw-stress evidence also reverses: at 1 s the tau_xy total RMS
difference shrinks 21.79248 to 9.593347 Pa, but its anomaly grows .01785518 to
.02792598 Pa. No smoothing or new transfer/solver/geometry diagnosis is implied.

## Controls, timestep, and implementation compatibility

K4 A and C are available homogeneous controls at the requested widths and
matched relative normal resolution: 32x256 and 32x512, both fault32,
dt=.125 s. Their accepted outputs include every .125-s time through 1 s.
They use prescribed frictional normal stress 1000 Pa, volume pressure
normalization, the same shear loading, activation .1, and fixed initialized
phase profiles. They could serve **new, identically configured** bumped runs
with only the K2 physical initial Theta bump added (5%, half-width .0625 m).
No extra homogeneous-control runs would be necessary for that choice.

They cannot be directly subtracted from the existing K2 sequence to create
a matched contrast: K2's 64x256/fault32 data use dt=.5, whereas its dt=.125
data use 128x512/fault64. Coincident output times do not remove intervening
timestep/history or mesh differences. Older homogeneous K1 runs also predate
the retained transfer/periodic-domain baseline. K2.3/K2.4 homogeneous controls
use true-pressure traction boundary conditions and are explicitly ineligible.

Resolved inputs confirm that the algebraic settings need no change:
nonlinear tolerance 1e-8, Stokes linear tolerance 1e-9, and phase nonlinear
budget 50 agree between K2 and K4. All saved accepted K2 states satisfy their
solver/history checks and both separate 1e-4 support/normalization limits.
That verifies execution, not temporal accuracy. Reusing tested dt=.125 is
possible operationally; calling it qualified for nonuniform width attribution
is not supported by the existing data.

The old K2 sequence precedes incident-cell MPI ADD/count history transfer and
the K3 periodic-domain correction. The bounded corrected-transfer replay
preserves the 0/.5-s surface results but changes actual q at 1 s by 6.36592e-5
Pa RMS on the coarse mesh. It is not a replacement spatial/temporal sequence,
and does not establish a cause for the temporal reversal. Neither accepted
correction is reopened here.

## Factor-four rule and closure

The required signal remains
Delta_nonuniform Q = (Q_bump - Q_homogeneous)_half
                  - (Q_bump - Q_homogeneous)_wide.
Its magnitude is **unmeasured for every requested observable**: no compatible
narrow bumped trajectory is present. Neither the 16--17 Pa homogeneous K4
traction contrast nor a K2 anomaly alone is this signal.

For scale only, four times the finest saved temporal anomaly difference at
1 s is .00261502 Pa for q, 3.71765e-7 m/s for V, .00136971 Pa for C,
.0898788 s for Theta, and 1.47689e-7 m for slip. These are not full uncertainty
budgets: they omit other trajectories, spatial error, and any demonstrated
cancellation in the double difference. They cannot certify the required
factor-four separation. A strong width effect is not ruled out, but the
preflight does not establish an accuracy basis for attributing one. In
particular, favorable homogeneous cancellation in K4.1b is not transferable
without evidence to the nonuniform contrast.

No equations, initialization, histories, support, full I_h, quadrature,
pressure convention, tolerances, or acceptance coefficients changed. New
runs: **zero**; new builds/tests: none (documentation-only result). Read-only
JSON/parameter inspections took seconds; no dynamics, cancellation accounting,
or large export analysis was rerun. `git diff --check` verifies the report
edits. Recommend closing K4 with this explicit nonuniform limitation rather
than expanding K4.3 into another K2 convergence campaign.
