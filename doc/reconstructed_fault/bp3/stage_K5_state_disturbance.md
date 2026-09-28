# Short evolving disturbance test from accepted step 11

Follow-up: [transient friction and denominator accounting](stage_K5_transient_friction_accounting.md)
uses these same 32-step outputs to distinguish genuine absolute growth from
relative amplification, and the Q1 weak friction response from extra bulk
feedback. It adds no trajectories and qualifies the mechanism interpretation
below. Large raw exports are in the September 17 recoverable archive; compact
results and the follow-up's reduced QP cache remain active.

## Scope and status

This is a one-year, fixed-mesh disturbance experiment, not a new loading
prefix, earthquake-cycle trajectory or production formulation change.
All requested trajectories are complete. The experiment starts from the
coherent local accepted-step-11 checkpoint, not an unfinished Newton iterate.

**Decision:** the small 200-m disturbance grows in the ordinary discrete
evolution, mainly through feedback of its solved velocity into aging. At the
fine clock its V norm grows 27% from the first accepted response; replacing
only the aging rate by the evolving reference rate makes that norm fall 49%.
Using the evolving reference normal traction instead barely changes the
response. Timestep halving retains amplification, but changes the small
relative-state norm growth from 7.11% to 7.73%; this is not a converged
growth-rate or continuum-instability claim. Absolute-state norm growth is
only 0.52%, and the disturbance changes shape around the creep front.
No production algorithm, physical parameter, support, profile or tolerance
was changed, and no trajectory beyond this one-year window was launched.

## Reproducible problem

- Start: 1144267468.9434748 s, accepted step 11 (about 36.26 years).
- End: 1175825068.9434748 s, exactly one further 31557600-s year.
- Source: `first_long_run/mechanical-discrimination-roundoff-clock/restart/01/`.
- Same 300-km production Box mesh, 97.65625-m near-fault bulk cells,
  100-m shallow fault grid, ell=400 m, phase field, full I_h, endpoint
  treatments, background, material parameters and plate loading.
- Same split, exact frozen-rate nodal aging law; no within-step state coupling.
- Same bulk/linear/nonlinear tolerances, active-set rules, sparse B/G,
  pivoted surface inverse and AMG reference backend as the saved local run.
- Four MPI ranks. The original checkpoint is never overwritten.

The disposable checkpoint branches alter only the two serialized pending
clock doubles, time and dt. Their positions/layout are checked against the
qualified archive format; reversing this alteration reproduces every
uncompressed byte. Preceding dt, timestep index, all bulk/particle/history
data and all other checkpoint files remain unchanged. Initial state
perturbation is applied once after loading and before mechanics; the
benchmark's independent aging checker is given the same changed incoming
state. No fault velocity is prescribed or perturbed.

The disturbance is continuous Q1 interpolation of nodal values

    Theta_j^+/- = Theta_j^base exp(+/- epsilon w_j), epsilon=1e-4,
    w_j = cos^2(pi (xd_j-16500)/3000) cos(2 pi (xd_j-16500)/200)

inside 15–18 km, zero elsewhere. It is the same tapered 200-m pattern as
the frozen mechanical probes. Do not replace the interpolated pattern by
an analytic cosine when evaluating its production weak norm.

Use 16 equal steps (1972350 s) and 32 equal steps (986175 s). Unperturbed,
positive and negative branches share the exact schedule at each level.
Both ordinary production timestep guards remain active. A guard demanding
a shorter step, or an accepted max(V)*dt/Dc exceeding 0.25/0.125, stops
the branch instead of silently giving compared runs different clocks.
This maximum includes strengthening nodes and the complete fault.

An additional four-step, double-amplitude fine-clock branch checks linear
scaling over the first 0.125 year. It is not another physical loading case.

## Feedback controls

Both controls use the same saved **32-step evolving unperturbed reference**.

1. **State feedback removed:** mechanics retains its own incoming nodal
   Theta and solves its own V and bulk state. For the next step, the control
   state is updated from its own incoming Theta using the reference's
   accepted V at each node. In the test plugin, the ordinary own-rate
   candidate is checked first, then replaced before any postprocessor
   consumes it by a candidate evaluated from the *same* incoming Theta.
   Aging is not applied twice; stress and all other histories are untouched.
   The independent checker uses the reference V for this control only.
2. **Normal feedback removed:** friction uses the reference's actual total
   normal traction at the same owned bulk QP and accepted time. This includes
   the evolving pressure/deviatoric-stress contribution, not just 50 MPa.
   The perturbed branch still solves bulk pressure and velocity and evolves
   its own stress and state. Physical sigma_n remains a separate diagnostic.

The normal control reuses the existing prescribed-friction-pressure
constitutive/Jacobian path through a benchmark-only reference-pressure
adapter. All nonpressure adiabatic functions delegate to the original
`ComputeProfile` implementation. A test-access setter selects the prescribed
friction path only after the normal BP3 work/source mapping is reattached.
There is no pressure-vector shift or bulk boundary/pressure change.

With sigma_ref fixed during each mechanical solve, the existing derivatives
are precisely

    K = integral w N_i N_j [2 kappa chi S:S + sigma_ref mu_V + eta_d],
    G dx = integral w N_i [2 kappa S:depsilon].

The mu*N and -mu*dp terms are absent; bulk pressure still enters the Stokes
equations. Each positive-work diagnostic sample is checked against its
reference traction. Pressure/normal-strain response checks verify absence
of that friction feedback. A one-step zero-disturbance control checks that
the reference solution is retained to numerical accuracy.

Only the separate test libraries enable these controls. The maintained
BP3 library and production material/solver source contain no new runtime
selectors. The existing test-access header exposes the already-existing
pressure-mode boolean; no production class layout or public API is changed.

## Measurement conventions

Use the actual work measure w_q=JxW_q chi_q and its consistent Q1 mass M.
For every accepted state report

    P(f) = w_nodal^T M f / (w_nodal^T M w_nodal),
    RMS(f) = sqrt(f^T M f / (1^T M 1)).

Here f is the difference between the perturbed and **evolving reference**
nodal V, nodal Theta, or nodal log(Theta_pert/Theta_ref). Report both signed
200-m projection and total whole-fault weighted norm. The whole-fault
normalization measure is approximately 115466.5035 m; modal mass is
377.800817 m. Also report normalized amplitudes, with their denominators
explicit: initial epsilon and RMS(w), or epsilon*Vp for velocity.

Absolute Theta differences and relative/log differences are different
observables when the background Theta changes. Neither is silently used
in place of the other. V differences are zero at the initial checkpoint;
their first nonzero amplitude is the first mechanically re-equilibrated
accepted state, not a prescribed initial rate disturbance.

The traction budget is evaluated at every owned production work QP using
the accepted mechanics and its **incoming** Theta, not the plotted updated
state. Temporarily substituted Theta is restored by a scope guard even if
diagnostic evaluation throws. Working FE stress is the retained history
consumed by mechanics; the newly committed particle tensor is not reused
as old stress in these evaluations. Reassembled weak residuals are checked
against the native accepted surface system.

At a matched QP, telescope the exact friction difference as

    -mu_P (sigma_used,P - sigma_R)                         [normal]
    -sigma_R [mu_P(V_P,Theta_P)-mu_R(V_P,Theta_P)]          [mixture]
    -sigma_R [mu_R(V_P,Theta_P)-mu_R(V_P,Theta_R)]          [state]
    -sigma_R [mu_R(V_P,Theta_R)-mu_R(V_R,Theta_R)]          [rate].

Theta_P and Theta_R in this expression are the incoming Q1 states. Add
q_P-q_R and -eta_d(V_P-V_R) to recover the full physical residual
difference. The normal-control sigma_used,P is sigma_R; its own physical
normal stress is still exported separately. The material-mixture term is
retained rather than attributed to state. The offline evaluator recovers
the actual effective a from the production mu,V,Theta using the configured
regularized law, then independently reproduces mu before telescoping.

QP identity is (MPI rank, rank-local active-cell index, quadrature index),
with segment/xi and weights checked identically across branches. This is
a same-mesh/same-rank comparison, not a cross-partition identity scheme.

## Evidence locations and commands

Root: `benchmarks/reconstructed_fault/bp3/first_long_run/state-disturbance/`.
Each branch contains `launch.json`, exact resolved input, disposable
checkpoint, `run.log`, `execution.json`, ordinary accepted-step/history
checks, nodal diagnostics and binary QP records.

`disturbance_nodes_<step>.csv` contains incoming/outgoing Theta, accepted V,
the input mode, consistent mass bands and native weak loads. Its
`normal_load` is the normal traction used by friction: the reference load
in the normal control, not that control's physical bulk normal stress.
`disturbance_qp_<step>_rank<rank>.bin` contains 12 little-endian doubles per
record: local cell index, q, segment, xi, weight, V, incoming Theta, q shear,
physical sigma_n, mu, damping and residual. For the normal control, sigma_n
in this file is physical; the residual uses the separately saved reference
normal stress. This distinction is intentional and encoded in launch data.

```sh
cmake --build benchmarks/reconstructed_fault/performance/build-gmg --target fault_disturbance -j4
python3 benchmarks/reconstructed_fault/bp3/run_disturbance.py prepare NAME --steps 16 --epsilon 1e-4
python3 benchmarks/reconstructed_fault/bp3/run_disturbance.py run NAME
MPLCONFIGDIR=/tmp/aspect-bp3-slip-plot python3 benchmarks/reconstructed_fault/bp3/analyze_disturbance.py REFERENCE NAME
```

Use `--steps 32 --limit .125` for the finer clock. Controls additionally use
`--control state|normal --reference reference32` and the separate
`fault_disturbance_controls` build target. Explicit core placement for
independent concurrent branches changes execution placement only.

The first `reference16` attempt stopped in parameter parsing because an
obsolete clock name was still present; it never loaded the checkpoint.
`reference16-ready` is the corrected reference. The positive coarse branch
initially inherited one CPU on all unbound ranks; affinity was corrected
without restarting or changing state. Its wall time therefore is not a
performance comparison. Neither incident is a physical/numerical failure.

## Results

All unperturbed, +/- perturbed and feedback-control trajectories reached
the common one-year endpoint with their declared identical clocks.

### Full feedback and timestep check

The table uses the positive branch; the negative branch is the separate
small-amplitude symmetry check. Norms are whole-fault work-weighted RMS.

| Quantity at one year | 16 steps | 32 steps |
|---|---:|---:|
| delta V 200-m projection (m/s) | -2.0381912e-14 | -2.0737661e-14 |
| delta V RMS (m/s) | 1.3071326e-15 | 1.3356208e-15 |
| delta Theta 200-m projection (s) | 11173.5493 | 11180.8619 |
| delta Theta RMS (s) | 1350.9644 | 1351.3048 |
| log-state projection / initial epsilon | 1.0564051 | 1.0618615 |
| log-state RMS / initial RMS | 1.0710979 | 1.0773149 |

Initial absolute Theta projection/RMS are 11931.0760/1344.3121 s.
Consequently the absolute-state RMS increases only about 0.5%, even though
the relative-state RMS increases 7–8%; absolute-state projection decreases.
The fine V RMS increases 27.0% from its first accepted nonzero response,
1.0517611e-15 m/s, to the final value. Initial checkpoint delta V is zero
because V is not prescribed by this perturbation.

At the common final time, the positive disturbance fields change under
timestep halving by 2.425% for V, 0.0896% for absolute Theta, and 0.7494%
for log-state in the same weighted norm. Relative-state norm growth changes
from 7.110% to 7.731%: about 0.62 percentage point, or 8% of the finer growth.
The direction of growth is robust; a converged growth rate is not established
by two clocks. The matched unperturbed final V/Theta fields themselves differ
by 0.1226%/0.00566% globally. All disturbance measures subtract the reference
on their own clock rather than attributing this common-mode difference to
the disturbance.

Amplification is localized rather than uniform: the final nodal log-state
disturbance divided by its initial value is about 1.206 at 16.0 km, 1.319 at
16.2 km, 1.111 at 16.5 km, 0.839 at 17 km, and 0.749 at 17.5 km. This is
transient amplification around the evolving creep front in the existing
discrete model, not a measured steady eigenvalue or a continuum wavelength
qualification. The 200-m pattern spans only two shallow fault-node spacings.

### Normal-stress feedback

At one year, using evolving reference normal traction in friction gives
log-state projection/norm gains 1.0625948/1.0781454, compared with
1.0618615/1.0773149 with full feedback. The final V disturbance field changes
by 0.4040%; the log-state disturbance field changes by 0.1121%. Thus this
feedback is not the principal amplifier here; removing it slightly increases,
rather than suppresses, the measured gain.

The actual physical normal-stress disturbance is not suppressed or overwritten:
its full/control weighted RMS is 0.55787/0.55882 Pa. Only its frictional effect
is removed. The exact zero of the control's `normal_friction` column is therefore
a specified diagnostic condition, not a claim of zero physical stress response.

Final signed 200-m weak force projections (Pa), using incoming state:

| Residual contribution | Full feedback | Reference V in aging | Reference normal in friction |
|---|---:|---:|---:|
| shear driving | +1.5582896 | +1.0245099 | +1.5603775 |
| normal-stress change in friction | +0.0808652 | +0.0479022 | 0 |
| state change in friction | -77.6266669 | -39.6831419 | -77.6773902 |
| direct rate change in friction | +75.9875120 | +38.6107299 | +76.1170783 |

Damping is about +9.6e-8 Pa and the mixture term is about 1e-8 Pa.
The exact telescoping budget reproduces the independently re-evaluated
point residual difference to less than 1e-7 Pa, rather than imposing zero
on the accepted finite-tolerance residual. State/rate terms nearly cancel;
their separate large magnitudes must not be mistaken for a residual imbalance.

### State feedback and mechanism

Final fine-clock measures, with V gain relative to each branch's first
accepted response and state gains relative to the initial imposed disturbance:

| Quantity | Full feedback | Reference V in aging | Reference normal in friction |
|---|---:|---:|---:|
| delta V projection (m/s) | -2.0737661e-14 | -8.9081912e-15 | -2.0792033e-14 |
| delta V RMS (m/s) | 1.3356208e-15 | 5.3988657e-16 | 1.3400949e-15 |
| V RMS gain | 1.269890 | 0.513317 | 1.273988 |
| delta Theta projection (s) | 11180.8619 | 8523.2056 | 11183.3238 |
| delta Theta RMS (s) | 1351.3048 | 1260.5843 | 1351.3107 |
| absolute Theta RMS gain | 1.005202 | 0.937717 | 1.005206 |
| log-state projection gain | 1.061861 | 0.534828 | 1.062595 |
| log-state RMS gain | 1.077315 | 0.564966 | 1.078145 |

The first state-control mechanics solve reproduces the full-feedback V
bit-for-bit. Its control affects the subsequent aging map, not that first
incoming-state solve. At one year, suppressing feedback into aging reduces
the V disturbance norm to 40.42% of the full-feedback result and changes
growth into decay. The state disturbance is not reset to the reference;
it retains and evolves its independent initial perturbation.

The supported mechanism is a positive state/rate feedback: locally increased
incoming state raises friction and lowers the solved rate; the reduced rate
then preserves more state relative to the reference through aging. The
opposite-sign perturbation behaves nearly antisymmetrically. The shear
response and direct rate friction oppose the imposed state resistance, but
do not suppress this short-pattern transient in the tested discrete model.
Removing normal-stress feedback alone does not cure it. Its small measured
effect is below the coarse/fine V-disturbance difference, so even its precise
sign/magnitude should not be treated as temporally converged.

This establishes a feedback mechanism for this background state and grid,
not the physical admissibility of a two-node wavelength. No friction-law
change, normal-stress clamp, state smoothing or solver modification follows
from this test alone. The remaining scientific distinction is whether the
amplification persists under a spatially qualified state/friction
representation; the present task does not answer that by running a new mesh.

Completed qualification checks:

- The 16-step unperturbed/positive/negative trajectories reach the same
  one-year endpoint. The largest accepted V*dt/Dc is 0.233399607, below 0.25.
- Doubling epsilon on the first four fine steps changes the amplitude-scaled
  V field by at most 4.23e-5 relative, absolute Theta by 4.33e-5, and relative
  log-state by 9.12e-6. These are small-perturbation responses, not solver noise.
- The coarse +/- even/odd norm ratios are at most 1.308e-4 for V,
  5.318e-5 for absolute Theta, and 8.544e-5 for log-state.
- The corresponding fine maxima are 1.389e-4, 5.346e-5 and 8.812e-5.
  The symmetric odd response gives final coarse/fine differences
  2.42478%, 0.0895605% and 0.749412%, essentially identical to the
  positive-branch comparison above.
- The one-step, zero-disturbance normal control differs from the ordinary
  reference by 8.414e-21 m/s in weighted V RMS; its physical working-stress
  observer, normal-feedback derivative checks and ordinary history audit pass.

The state-feedback control has an additional independent full-history check.
Since reference and control use the same accepted reference rate in aging,
their absolute nodal difference must obey

    delta Theta_k = delta Theta_11 exp(-sum_{j=12}^k V_ref,j dt_j / Dc).

This identity uses the initial perturbation once and never resets the control
from later reference Theta. Its final predicted absolute-state norm gain is
0.937716992; the predicted log-state projection/norm gains are
0.534828072/0.564966201. The actual control is checked against this independent
prediction at all 32 states: maximum relative difference-norm error is
3.370e-11. Mechanics and stress histories are solved separately, not inferred
from this identity.

## Verification, cost and retained implementation

All ten successful runs pass, totaling **213 new accepted steps** (the copied
loading-prefix rows are excluded). All end at exactly their declared physical
times; only the amplitude/noise checks intentionally stop earlier.

- Accepted maximum V*dt/Dc: 0.233399607 on the coarse clock and 0.116738629
  on the fine clock, including strengthening nodes.
- Every new accepted state passes the genuine nonlinear/physical acceptance
  checks and fresh linear-residual check; no tolerances or budgets changed.
- Maximum recorded normalized nonlinear residual over these runs:
  1.585e-9. Maximum surface RMS: 0.001915 Pa (normal-control branch).
- Every new accepted state has 1156 free and zero lower-active nodes.
- Independent aging audit: maximum relative error 2.221e-16, against the
  retained 1e-12 threshold. The state control additionally checks the ordinary
  own-rate candidate before substituting its reference-rate candidate.
- Reassembled incoming-state native weak residual difference: zero in every
  printed audit. Exact friction-budget telescoping closure: at most 8.131e-9 Pa
  across every exported QP/accepted state, below the 1e-7 diagnostic check.
- Same-mesh QP identities, segment coordinates and work weights agree across
  compared branches. Existing BP3 fixed-geometry/profile and working-stress
  checks remain enabled. No loading prefix is replayed.
- Build: the separate diagnostic targets with `-j4`; Python syntax checks
  and `git diff --check` pass. No broad ASPECT test suite was run.

| Run | New steps | Wall seconds |
|---|---:|---:|
| reference16-ready | 16 | 440.64 |
| plus16 | 16 | 1017.50 |
| minus16 | 16 | 502.62 |
| reference32 | 32 | 905.96 |
| plus32 | 32 | 1068.43 |
| minus32 | 32 | 1181.11 |
| normal32 | 32 | 1702.56 |
| state32 | 32 | 1183.83 |
| double4 | 4 | 222.20 |
| normal-zero1 | 1 | 88.36 |

These sum to 8313.2 simulation seconds, with independent branches partly
concurrent; the sum is not elapsed experiment wall time. They are not a
performance comparison: core placement differs and the positive coarse run
had the documented initial affinity problem. Maximum reported child RSS is
1363472 KiB, not an aggregate four-rank memory measurement.

The source base is `359ea223cd0fa088ba7c8b03338cc7fe19d4acdc`, with
benchmark/test-only additions. The exact binary/library/input hashes are in
each `launch.json`; the executed commands and placement are in `execution.json`.
`diagnostic-code-and-libraries.tar.gz` in the evidence root preserves the
modified diagnostic sources and both qualified libraries. The main library
was kept immutable while controls were built separately; the first
state-control mechanics result reproduces its V exactly.

Retained task files:

- `benchmarks/reconstructed_fault/bp3/bp3.cc`: compile-time-only hooks;
  ordinary research builds do not enable them.
- `benchmarks/reconstructed_fault/bp3/disturbance_diagnostic.h`: initial
  disturbance, matched controls, exact clock and correct-time QP exports.
- `tests/reconstructed_fault_disturbance.cc`, the two diagnostic CMake
  targets, and a narrow setter in `tests/phase_field_fault_test_access.h`.
- `run_disturbance.py`, `analyze_disturbance.py`, `summarize_disturbance.py`:
  checkpoint-preserving launch, work-weighted analysis and final comparisons.

No production material/solver implementation or public class layout changed.
Unrelated working-tree changes were preserved. The diagnostic plugin is
specific to this accepted-step-11, fixed-mesh, four-rank branch; it is not a
new general restart or long-run mode. No cross-rank restart, new spatial
level, longer trajectory or model correction is qualified here.

Main outputs:

- [Growth and projection curves](../../../benchmarks/reconstructed_fault/bp3/first_long_run/state-disturbance/disturbance_growth.png).
- [Final along-fault disturbance profiles](../../../benchmarks/reconstructed_fault/bp3/first_long_run/state-disturbance/disturbance_final_profiles.png).
- [Signed force-budget curves](../../../benchmarks/reconstructed_fault/bp3/first_long_run/state-disturbance/disturbance_force_budget.png).
- [Machine-readable comparison/verification](../../../benchmarks/reconstructed_fault/bp3/first_long_run/state-disturbance/summary.json).

Each perturbed/control directory also contains `comparison.csv`, giving
absolute physical time, elapsed physical years, signed projections, total
weighted norms and each correctly timed signed force contribution at every
accepted state. The raw nodal and QP records remain available independently
of the plots.
