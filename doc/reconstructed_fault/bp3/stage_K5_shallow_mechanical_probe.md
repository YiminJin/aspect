# Frozen mechanical discrimination of the early shallow structure

Follow-up: the single authorized bulk refinement is complete; see
`stage_K5_frozen_bulk_refinement.md`. It preserves the captured physical
localization and makes the alternating mode still softer. The recommendation
below records the decision at the end of this earlier task, not a request
to repeat that refinement.

## Result and next decision

The neighboring-node alternating mode is mechanically softer, not an
unrestored zero mode. At the early step-11 state its shear restoring
coefficient is **40.8% of the broad-bump value and 37.4% of the six-node
value**. The instantaneous friction derivative is positive and large; it
must not be counted as mechanical restoring stiffness. All three tested
total modal responses remain positive with incoming state frozen.

Normal-stress feedback does not account for this short-mode softening.
Freezing each QP's actual baseline normal traction changes the alternating
mode's total restoring coefficient by **0.231%**. In the narrower
16.5–17.5-km onset neighbourhood, the weak normal-feedback response is
14.23 Pa RMS versus 1524.89 Pa RMS total response, under the arbitrary
1e-12-m/s tangent normalization. Thus this conclusion is not solely an
artefact of averaging over the nearly locked shallow part.

**Recommended next action:** one frozen mechanical spatial comparison,
halving the local bulk velocity/pressure cell size from 97.65625 to
48.828125 m while keeping the 100-m fault grid, ell=400 m, mode wavelengths,
physical phase/localization field and incoming data fixed. Prolong the
existing FE fields rather than reinitializing a different profile. Repeat
the three responses, especially the alternating/shear ratio. Because the
mechanical derivative here is history-independent, another long loading
prefix is unnecessary. A substantial stiffness recovery would implicate
bulk/source discretization; persistence would favour the finite-width
coupling itself and motivate a more specific model/discretization decision.
One refinement would not prove convergence. No refinement or timestep
comparison was launched in this task.

This does **not** establish that the observed nonlinear instability is
caused solely by this softening. These are three tangent directions, not a
stability spectrum with evolving state. The split-state time discretization
remains a separate possible amplifier; the nodal V*dt/Dc reaches about 4.49
in the patch at this step. The result selects a spatial discrimination next,
not a production correction or permission to suppress normal feedback.

## Scope and definitions

This test does not change production equations or publish diagnostic histories.
It reconstructs the early prefix with the maintained 300-km, fully frictional,
mature-fault fixture, the saved accepted clock, and the ordinary timestep guards.
The available server checkpoints are later than onset and use a different
deal.II build; they are not transplanted into the local executable.

The first attempt reproduced initialization and step 1 but stopped before
selecting step 2: the local production controller selected
2666245.4820050253 s rather than the saved 2666245.4820157574 s. That
10.7-microsecond difference exceeded the historical matching-only 1e-12
relative assertion. A separate preserved run uses a benchmark-local 1e-8
cross-build clock agreement check; the actual production controller still
selects the smaller step. No physical timestep guard or nonlinear/linear
tolerance is changed. Actual clock differences must be reported.

The test hooks into the production, freshly verified linear solve in step 11.
It selects Newton iteration 6, the final linearization in both the saved and
reproduced step 11, before acceptance/publication. The original 0.01-Pa
diagnostic trigger missed this state (accepted RMS 0.0227632 Pa); the resulting
completed prefix is retained as evidence, not reported as a successful probe.
The corrected trigger is an iteration selector, not a convergence criterion.
The prefix through step 10 is committing; the step-11 probe and perturbations
are noncommitting. The unchanged exception/rollback path terminates the run.

Three continuous-Q1 perturbations have a cosine-squared taper on 15–18 km:
a broad bump, a 600-m wavelength (six nodes), and a 200-m wavelength (alternating
neighboring nodes). Nodal amplitude is 1e-12 m/s. The broad input deliberately
does not impose zero mean. Compare responses normalized by the production
mass norm, not unnormalized nodal Euclidean norms.

For each prescribed diagnostic velocity direction, solve on private vectors

\[
 A\delta x=B\delta V,
 \qquad\delta R=G\delta x-K\delta V.
\]

The pressure remains an unknown. Use homogeneous perturbation constraints and
convert solver pressure to physical pressure exactly once. The production
work measure is \(J_q\chi_q N_i\), on uniquely owned bulk QPs, with the existing
endpoint source map. There is no replacement by particle-center weights or
an unweighted column average.

With committed/incoming state frozen,

\[
\begin{aligned}
\delta q&=2\kappa S:\delta\dot\epsilon
             -2\kappa\chi(S:S)\delta V,\\
\delta\sigma_n&=\delta p-2\kappa N:\delta\dot\epsilon,\\
\delta F&=\delta q-\mu\delta p+2\mu\kappa N:\delta\dot\epsilon
             -\sigma_n\mu_V\delta V-\eta^d\delta V.
\end{aligned}
\]

Report all terms separately. The instantaneous friction derivative is
\(\sigma_n\mu_V\), not the coupled mechanical stiffness. No aging update is
performed inside these probes.

The matched **frozen-normal diagnostic** evaluates
\(F^{fn}=q-\mu(V,\Theta_{old})\sigma_{n,base}-\eta^d V\)
at the very same bulk response. It preserves each baseline QP resistance,
and removes only \(-\mu\delta\sigma_n\). It does not fix bulk pressure, reset
normal traction to 50 MPa, or constitute a separately evolved trajectory.
These are linear response probes, not a full nonlinear frozen-normal replay.

## Checks and outputs

- Fresh residual for each private A solve: at most 1e-10 of its RHS norm.
  This diagnostic accuracy does not alter the production solve tolerances.
- Reproduce the native weak base residual within 1e-5 Pa after mass-row scaling.
- Compare the decomposed action with production \(G\delta x-K\delta V\),
  and check shear work pairing with B (relative 1e-8).
- Centered finite differences of the actual point constitutive evaluator,
  including the matched frozen-normal response (relative 1e-5).
- Verify solution, working vector, and manager velocity were not changed.
- The original plan was one four-rank run with a hard cap of 1800 s.
  The actual execution exceeded that planned aggregate cost because of
  diagnostic-harness defects, documented below. No physical failure was
  bypassed. Preserved data and exact response reuse avoided a third prefix.

`mechanical_modes.csv` contains modal restoring coefficients in Pa/(m/s),
defined as \(-\delta V^T\delta R/(\delta V^TM\delta V)\). Positive means
restoring for that tested direction; it is not an eigenvalue or proof of
positivity of the generally nonsymmetric coupled operator.
`mechanical_mode_nodes.csv` contains unreplaced weak rows, not nodal point
tractions. Dividing by `mass_row` gives labelled mass-row weak averages.
Rank-local `mechanical_mode_qp_rank*.csv` contains raw QP differences and
production weights. Summation over ranks visits each QP exactly once.

The decision is whether the alternating/broader perturbations have weak
mechanical restoring traction, and whether normal feedback substantially
changes that response. This cannot alone establish nonlinear long-term
stability or causally attribute all subsequent oscillations.

## Target state and exact reuse

The qualified local control reaches accepted step 11 at
1144267468.9434748 s (36.259648 yr), with dt=153336464.43996707 s.
All **1156 nodes are free**, with zero lower-active nodes. Final relative
bulk/fault residuals are 1.704813e-13 / 3.018811e-9; fault RMS is
0.0227632092 Pa. All fresh linear checks pass, and the accepted Theta audit
error is 2.22e-16. This is the earliest 0.1%-dip state identified in the
preceding saved-output audit, not the later strongly alternating/bound state.

The target probe retained its complete broad response on the perturbed
region, but aborted in a diagnostic MPI reduction before writing its
summary. After fixing the harness, all three complete probes were qualified
in a 47.34-s initialization run. They can be reused here without approximating
the target operator:

- Bulk eta=1e26 Pa*s and G=32038120320 Pa are uniform, with inactive cutoffs;
  geometry, FE spaces, constraints, phase, chi and boundary types are fixed.
- A's velocity block and B scale by kappa. The constrained incompressible
  solution of A dx=B dV therefore has unchanged delta-u and physical delta-p
  proportional to kappa. Delta-q and -delta-tau:N scale by the same factor.
  Solver-pressure scaling is not mistaken for physical-pressure scaling.
- kappa11/kappa0 = **38.33411519295163**. Actual target mu, total sigma_n and
  the production instantaneous friction derivative are retained at every
  QP; none is replaced by its initial value. Retained history contributes to
  the baseline but not the affine bulk derivative.
- All **4624 QPs with nonzero modal dV** are present in both exports. There
  are 11902 complete common records; one truncated final off-support record
  from the interrupted writer is explicitly skipped. The assembled mass
  coverage of every reported 15–18-km weak row agrees with the native mass
  row to 1.45e-15. Thus missing records do not truncate these responses.
- Actual xi, positions and JxW are identical; relative differences in chi
  and work weight are below 6e-17. The actual captured target V agrees with
  the completed step-11 control interpolant to 6.93e-17 relative L2.
- Scaled versus independently captured **target** broad responses differ
  by 1.62e-13 (shear), 1.94e-13 (pressure), and 2.62e-13 (deviatoric normal)
  relative L2. Recombined full/frozen-normal responses agree to 3.02e-15
  and 2.76e-15. This is measured reuse, not an assumed time rescaling of
  friction or an extrapolation of a different trajectory.

The other two target responses are explicitly labelled **algebraically
reused linear responses**, not three newly executed target-step solves.
The native action/work/finite-difference checks below belong to the complete
initialization probes. The independent target broad comparison validates
the reuse identity. No diagnostic history was committed.

## Separated restoring responses

Each entry below is -dV^T(delta weak load)/(dV^T M dV), in
**1e15 Pa/(m/s)**, with the actual production consistent mass/work measure.
Signs include the residual convention: positive is restoring for that mode.
"Pressure" means -mu delta-p in the residual; "deviatoric normal" means
+mu delta-tau:N. Neither is replaced by B transpose.

| Mode | Mechanical shear | Pressure feedback | Deviatoric normal feedback | Instantaneous friction | Full | Frozen normal |
|---|---:|---:|---:|---:|---:|---:|
| Broad 3-km bump | 2.852689 | -0.001181 | -0.000726 | 19.498439 | 22.349220 | 22.351128 |
| 600 m / six nodes | 3.112913 | +0.001791 | +0.000297 | 19.770415 | 22.885416 | 22.883328 |
| 200 m / alternating neighbours | 1.162967 | +0.023291 | +0.026634 | 20.439609 | 21.652501 | 21.602577 |

Radiation adds 4.62444e6 Pa/(m/s) to every mode, negligible at this table's
scale but retained in every calculation. Normal feedback changes the
mechanical-only restoring coefficient by -0.067%, +0.067%, and +4.29%
respectively; for the alternating mode it is mildly **restoring**, not
destabilizing in the same-mode contraction.

The direct frozen-bulk shear stiffness is nearly identical for the three
modes, about 8.409e15 Pa/(m/s). After solving bulk equilibrium, only
33.9%, 37.0%, and **13.8%** remains. The short-mode softening therefore
appears in the **bulk-relaxed mechanical response**, not in a disappearing
local friction derivative or a missing direct chi term. This alone cannot
separate finite-width accommodation from under-resolved FE coupling:
the alternating wavelength is only two fault elements/about two bulk cells,
and is smaller than ell=400 m.

The three-mode cross-action table is also saved. Normal feedback changes
its full/frozen block by about 0.212% in spectral matrix norm under the
stated individual mass normalization. This is a supplementary check for
off-diagonal effects, not a complete spectral stability analysis or an
assertion that these modes form an orthonormal basis.

## Weak versus raw response

The following are **mass-row weak-average RMS** over 15–18 km, in Pa for
a 1e-12-m/s nodal tangent normalization. This amplitude is a basis convention,
**not an admissible finite trial** at every nearly locked node. Linear
coefficients and ratios do not depend on its arbitrary scaling.

| Mode | delta-q | delta-p | -delta-tau:N | Instantaneous friction | Normal-feedback contribution | Full response |
|---|---:|---:|---:|---:|---:|---:|
| Broad | 1835.52 | 2.196 | 1.182 | 76608.96 | 1.515 | 76484.49 |
| Six nodes | 1115.90 | 2.607 | 0.415 | 46017.91 | 1.607 | 46195.88 |
| Alternating | 235.46 | 16.238 | 9.361 | 28267.94 | 13.153 | 28302.62 |

These small weak normal contributions do not mean raw pressure and stress
are individually small. With the same production QP weights, raw pressure /
-delta-tau:N / total delta-sigma_n RMS are respectively:

- broad: 1307.93 / 802.63 / 555.84 Pa;
- six nodes: 1653.04 / 633.89 / 2187.69 Pa;
- alternating: 650.68 / 913.37 / 1366.35 Pa.

The weak integration cancels much of the transverse structure. The raw
QP statistics and the weak rows are exported separately, not substituted
for each other. Actual baseline QP sigma_n in this patch has weighted mean
49.929598 MPa and range **48.515605–51.332351 MPa**. The matched diagnostic
freezes these individual values, not a common 50-MPa pressure.

To avoid masking the onset region with the strong friction derivative near
15 km, the 16.5–17.5-km weak RMS is also evaluated:

| Mode | Shear | Instantaneous friction | Normal feedback | Full |
|---|---:|---:|---:|---:|
| Broad | 2220.67 | 3588.09 | 1.213 | 5771.02 |
| Six nodes | 1435.32 | 2202.98 | 2.214 | 3635.02 |
| Alternating | 297.89 | 1216.24 | 14.231 | 1524.89 |

Normal feedback is still below 1% of full weak-response RMS in that window.
The CSV/plot retain the along-fault structure rather than only modal scalars.

## Verification and execution record

The complete four-rank initialization probes pass:

| Check | Broad | Six nodes | Alternating |
|---|---:|---:|---:|
| Private A solve iterations | 18 | 16 | 17 |
| Fresh relative A residual | 4.5664e-11 | 8.3240e-11 | 4.2528e-11 |
| Shear virtual-work relative error | 4.88e-16 | 2.05e-16 | 4.45e-15 |
| Decomposed versus native G dx-K dV | 7.09e-16 | 5.88e-16 | 1.19e-15 |
| Full point-law finite difference | 1.75e-10 | 2.50e-10 | 4.85e-10 |
| Frozen-normal finite difference | 1.04e-10 | 1.65e-10 | 3.03e-10 |

All native base weak rows pass the 1e-5-Pa reproduction check. Production
solution, working vector and manager V are checked unchanged before the
intentional stop. The ordinary solver catch path restores bulk and manager
state; history publication is not reached. This is not a new restart or
rollback stress-test campaign.

All directories below are preserved under
`benchmarks/reconstructed_fault/bp3/first_long_run/`:

| Directory | Elapsed seconds | Outcome |
|---|---:|---|
| mechanical-discrimination | 58.66 | Historical cross-build clock assertion at step 2; no physical guard relaxed |
| mechanical-discrimination-roundoff-clock | 961.98 | Genuine accepted control through step 11; original too-small diagnostic trigger never fired |
| mechanical-modes-final | 953.99 | Broad target response captured; diagnostic MPI vector-sum overload caused a lazy-symbol failure |
| mechanical-modes-preflight | 45.85 | Broad checks passed; root-only distributed norm caused a collective mismatch |
| mechanical-modes-preflight-fixed | 47.34 | All three response/check sets passed; intentional noncommitting stop |

The two harness bugs are fixed in the test plugin, not production. The
runner now binds plugin symbols at startup. Aggregate simulated execution
was about **34.46 min**, including the failed harness runs; this exceeds
the original one-prefix plan and is reported rather than hidden. Maximum
reported child-process RSS was 1361120 KiB (about 1.30 GiB, **not** aggregate
four-rank memory). No third prefix was run: the validated scaling/coverage
test above permits exact reuse of the short qualified probes.

The completed local control agrees with the server through step 10:
V relative L2 1.41e-12, Theta 4.01e-12, slip 3.79e-12; maximum weak
shear/normal differences 1.67e-5 / 6.60e-6 Pa. The physical time offset is
-0.0039782 s. The interrupted target prefix agrees bit-for-bit with the
completed local prefix in accepted records through step 10.

## Artifacts, snapshots and reproduction

No production numerical source was modified. Added/changed task code:

- `tests/reconstructed_fault_mechanical_modes.cc`: opt-in noncommitting
  response/derivative/work checks, benchmark-local saved-clock comparison.
- `benchmarks/reconstructed_fault/performance/CMakeLists.txt`: test-plugin target.
- `benchmarks/reconstructed_fault/bp3/run_mechanical_modes.py`: prepared launch,
  source/binary/input provenance, hard cap, no automatic retry.
- `benchmarks/reconstructed_fault/bp3/analyze_mechanical_modes.py`: complete
  native-output reducer; its default target comparison is step 11.
- `benchmarks/reconstructed_fault/bp3/reuse_mechanical_modes.py`: guarded
  offline reuse/recombination used for the actual reported target result.
- This report. Unrelated plotting/README and working-tree changes are retained.

The preceding bounded-diagnosis report has a follow-up pointer; its original
saved-data findings are unchanged. Base source revision is
`359ea223cd0fa088ba7c8b03338cc7fe19d4acdc`. A recoverable task-code snapshot is
stored as `mechanical-response-analysis/mechanical-probe-sources.tar.gz`.

`mechanical-response-analysis/` contains `mechanical_response.json`
(numerical checks, raw/weak statistics, cross actions and input hashes),
`step11_response_weak_profiles.csv`, and `step11_mechanical_response.png`.
The raw target and initial QPs remain in their original run directories.
No failed evidence is overwritten by a purported successful run.

The usable ordinary local checkpoint is at
`mechanical-discrimination-roundoff-clock/restart/01/`, **accepted step 11**.
It is not a beginning-of-step-11 checkpoint. The earlier requested step-10
snapshot was not made because wall-clock checkpointing had precedence.
The launcher now disables wall-clock checkpointing when selecting the
ten-step snapshot cadence; no extra run was made just to populate it.
The existing accepted checkpoint and exported frozen linear-response data
are preserved for reuse, in accordance with the user's snapshot permission.

Build command:

```sh
cmake --build benchmarks/reconstructed_fault/performance/build-gmg \
  --target fault_mechanical_modes -j4
```

Offline report reproduction (no ASPECT run):

```sh
MPLCONFIGDIR=/tmp/aspect-bp3-slip-plot python3 \
  benchmarks/reconstructed_fault/bp3/reuse_mechanical_modes.py \
  --initial benchmarks/reconstructed_fault/bp3/first_long_run/mechanical-modes-preflight-fixed \
  --target benchmarks/reconstructed_fault/bp3/first_long_run/mechanical-modes-final \
  --control benchmarks/reconstructed_fault/bp3/first_long_run/mechanical-discrimination-roundoff-clock \
  --output benchmarks/reconstructed_fault/bp3/first_long_run/mechanical-response-analysis
```

Use `run_mechanical_modes.py --help` only to prepare a future explicitly
requested run. The completed evidence is sufficient for the next decision;
do not automatically repeat these prefixes or launch a loading trajectory.
