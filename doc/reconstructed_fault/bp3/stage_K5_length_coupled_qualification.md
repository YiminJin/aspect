# Modified BP3: bounded coupled mesh qualification

## Decision

Both fresh cases completed nonlinear initialization and two matched real
steps. Coupled feasibility, current endpoint source/weak-row consistency and
the ordinary history checks pass. Rate, state and accumulated slip agree
closely, but **localized traction increments are not uniformly percent-level
resolved**. Stop here under the requested localized-discrepancy branch; do
not promote either mesh, construct a globally fine production mesh or launch
the conditional eight-step/restart/half-step sequence.

The strongest unresolved relative increment differences are at the nearly
steady bottom: 20.65 Pa in weak shear and 34.59 Pa in weak normal traction,
against fine-run changes of 59.50 and 66.02 Pa. These are small absolute
effects, not a large change in frictional loading, but dividing by 50 MPa
would hide the lack of incremental accuracy. The top shear increment differs
by 8.45%, and transition normal traction by 2.18%. The reference also creates
localized artifacts at its artificial fine/coarse patch edges. The evidence
supports targeted endpoint/patch-transition attention, not a solver change.

## Scope and preserved gate

This is the explicitly authorized diagnostic exception to the failed 1%
localization RMS-width screen, not production qualification. The original
`length-scale-study/comparison.json` and the refusal in
`length_scale_study.py prepare --qualification` remain unchanged. Its
12.20703125-m width error is still 1.92506%; the previous interior fine patch
gave 0.48062%. No equation, support, integration tolerance, nonlinear tolerance,
history rule or physical background has changed for this comparison.

Both cases start fresh with Dc=0.024 m, ell=50 m, the same continuous stationary
profile, 100-m fault grid, 300-by-100-km domain and fully frictional mature
work-measure model. The artificial Maxwell interval is 4e6 s, not elapsed
physical time. The two real intervals are both 4e6 s, ending at 8e6 s
(about 0.254 years). The existing diagnostic clock caps the ordinary controller;
it asserts if the production safety controller requires a shorter interval.

The new reference refines the existing 10–23-km transition patch and adds
the first/last 5 km down dip, each with the existing graded normal halo.
It is not a globally fine band. Its finest cell side is 6.103515625 m;
the rest retains the candidate mesh. Endpoint normalization completion is
regenerated for that exact local virtual-Q1 mesh; independent 8/16-order
generation differs by at most 4.83e-11 m. The physical background file is
identical in both cases, not fitted to either accepted solution.

## Reproducibility

Artifacts are under
`benchmarks/reconstructed_fault/bp3/length-scale-study/coupled-diagnostic/`:
`candidate-two/`, `reference-two/`, regenerated reference mesh/completion,
resolved PRMs, common clocks, input/binary/plugin hashes, accepted-state logs,
current constitutive QP and weak-row exports, and execution records.

The opt-in launcher is `benchmarks/reconstructed_fault/bp3/length_coupled.py`.
The analysis is `analyze_length_coupled.py --compare candidate-two reference-two`.
Neither overwrites the old failed screen. The normal benchmark does not enable
the added `ASPECT_BP3_LENGTH_COUPLED_DIAGNOSTIC` output switch. An optional
`ASPECT_BP3_LENGTH_FULL_AUDIT_FROM` switch gates full lifecycle exports and is
not needed for the initial spatial pair. The new mesh generator mode 2 adds
endpoint refinement; mode 1 preserves the previous interior-only experiment.

The reference hard execution cap was set to 4000 s before launch, after the
candidate measured 1293 s; its graceful cap is 3900 s. The launcher also
enforces the remaining 7200-s aggregate budget, counting the earlier six
frozen probes (618.333 s). No failed solve is automatically retried.

## Meaning of the diagnostics

`work_qp_*` contains **current constitutive stress**, evaluated from accepted
u,p,V and the frozen working FE history used by that mechanical solve. It is
not a reevaluation with newly committed particle stress. The plotted Theta
is outgoing/committed state; it is compared as state, not substituted into a
friction balance that used incoming Theta. The ordinary production residual
and exact split-history checks retain their original meaning.

The endpoint check independently sums JxW*chi*N_i, q and sigma_n over all
exported bulk QPs, including continued in-box wedges, and compares with the
actual surface weak rows. It also checks chi=h(phi)/I_h and the full current
stress/pressure decomposition. Both endpoints have nonzero source continuation.
This checks actual coupled mechanics, not just completed column integrals.

A nonzero Q1 phase tail just beyond the fixed analytic association radius is
reported separately. Those unassociated QPs are outside the admitted normal
band, not holes in the band or missing endpoint wedges. No support change is
made. Window-weighted omitted FE localization is a volume/source diagnostic,
not a claimed bound on every transverse column.

Spatial traction differences are reported in Pa and relative to the actual
evolving traction increments, not only the 50-MPa background. Initial
projection differences remain visible. Subtracting them in an increment
comparison is accounting, not removal of their influence on evolution.

## Candidate result

The candidate completed initialization and two real steps with all 1156
nodes free, zero prescribed/lower-active nodes, and all fresh linear checks.
Initial Theta agrees with its configured formula exactly; t=0 retains zero
particle perturbation stress. Subsequent exact aging checks differ by at most
2.22e-16 relative. The first stress update differs from its independent check
by 3.69e-7 Pa on a 1.24521e5-Pa scale, using zero old working stress and
2,069,010 stable particle IDs.

| Accepted step | 0 | 1 | 2 |
|---|---:|---:|---:|
| Physical time, s | 0 | 4e6 | 8e6 |
| Newton updates | 3 | 0 | 12 |
| Krylov iterations | 116 | 30 | 250 |
| Final normalized nonlinear residual | 3.00e-12 | 9.66e-10 | 1.94e-12 |
| Surface strong/RMS residual, Pa | 8.71e-6 | 2.81e-3 | 3.02e-7 |
| Minimum accepted alpha | 1 | 1 | 0.13234 |
| Maximum V dt/Dc, real steps | — | 0.184956 | 0.174100 |

Endpoint weak mass closure is within 4.0e-15 relative and current weak traction
closure within 2.7e-7 Pa. Each endpoint window contains 206 continued-source
QPs. The exported-window omitted FE-tail source fraction is 2.127e-5.

The refined initialization independently closes endpoint mass within 9.0e-15
and traction within 4.1e-7 Pa, with 740 continued-source QPs per endpoint.
Initial top strain-mismatch RMS falls from 1.41946e-13 to 3.66345e-14 /s;
bottom falls from 1.42003e-13 to 3.66353e-14 /s. Top raw normal-stress
departures from 50 MPa change from [-26462,+17769] to [-6132,+4242] Pa;
bottom changes from [-26153,+17943] to [-5772,+4456] Pa. The unrefined 60-km
control remains essentially unchanged. The residual FE-tail source fraction
in each refined endpoint window falls to 2.5812e-6.

**Reference patch-edge caveat:** the larger global initial reference extrema
are at xd=23.00611 and 23.01238 km, normal coordinates +14.8378 and -12.3059 m,
on 6.104/12.207-m cells at the patch edge. Production sigma_n there reaches
49,950,323.55 and 50,059,913.28 Pa. These are not physical-endpoint extrema.
The saved high-order visualization permits an offline t=0 location check:
with zero old stress and S:N=0, sigma_n=50 MPa+p-2*kappa*epsilon:N. Reconstructing
the Q2 velocity derivative at Gauss points reproduces those extrema within
0.46 Pa despite Float32 visualization data. `length_initial_normal.py` retains
the strongest locations. It never treats the published zero tau history as
current constitutive stress. This is a location diagnostic, not a substitute
for the double-precision production residual/weak-load checks.

The initial velocity minimum at 15 km (0.817840 Vp) and maximum at 18 km
(1.109734 Vp) already exist before aging. The unchanged physical background
contains departures of -192515 Pa and +199353 Pa there from the nominal
constant BP3 shear traction; these are not newly generated oscillations or
evidence of a history update at timestep zero.

## Cost and convergence

| Resource | 12.207-m candidate | 6.104-m local reference |
|---|---:|---:|
| Cells | 229,890 | 352,062 |
| Total DoFs | 8,256,730 | 12,501,594 |
| Particles | 2,069,010 | 3,168,558 |
| Four-rank Release wall time | 1293.453 s | 2081.486 s |
| Largest child lifetime peak RSS | 4.64 GiB | 6.80 GiB |
| Condensed solve, 18 calls | 713 s | about 1200 s |
| Stokes preconditioner, 18 calls | 55.5 s | 92.5 s |
| Fault property preparation, 4 calls | 38.0 s | 47.7 s |
| I_h, included in preparation | 37.7 s | 47.2 s |

New simulation time is 3374.939 s. Including the earlier six probes gives
**3993.271 s = 66.55 min**, below 7200 s. No timeout, retry or full-cycle run.
Sampled summed reference rank RSS reached 24.79 GiB; this is not an aggregate
peak. The 32-GiB desktop used swap (host-wide usage reached about 4.4 GiB),
so costs include that resource pressure. A dedicated >=48-GiB machine would
give comfortable margin for this *local-reference* comparison; this is not
a resource qualification of a globally fine mesh.

The reference used the same 3/0/12 Newton updates as the candidate at steps
0/1/2, with 118/30/249 Krylov iterations and minimum alpha 0.132265. Both had
zero rejected Armijo candidates. All 18 returned directions per run passed
fresh residual checks; maximum fresh/target was 0.9523 / 0.9379. At the final
state the bulk/fault ratios were 1.94e-12 / 1.04e-13 (candidate) and
2.50e-12 / 6.37e-14 (reference). Surface RMS was 3.02e-7 / 1.85e-7 Pa.
No state reached its lower bound; all 1156 nodes remained free throughout.

## Matched spatial comparison

Common accepted times are 0, 4e6 and 8e6 s. Profiles and all quantitative
rows are in `matched_fault_profiles.csv`, `spatial_differences.csv`,
`velocity_chord.csv` and `spatial_*.png`. Tables below use common physical
windows and the mean of the two native work-lumped measures. Negative
roundoff at the exact top coordinate is included; the endpoint row is not
accidentally dropped from an x_d>=0 window.

Final **total-field relative RMS differences**, percent:

| Window | V | Theta | Accumulated slip |
|---|---:|---:|---:|
| Whole fault | 0.01540 | 0.003535 | 0.01599 |
| Top 0–2 km | 0.01513 | 0.002389 | 0.06093 |
| Transition 13–20 km | 0.02637 | 0.006714 | 0.03360 |
| Interior 25–35 km | 0.001427 | 0.000243 | 0.000808 |
| Bottom last 2 km | 0.02868 | 0.007582 | 0.02657 |

Maximum local V difference is 0.43384%, at a patch edge, not a bound-clipped
node. Maximum local Theta difference is 0.05372%. The largest absolute slip
difference is 1.474e-5 m at 23.1 km. In the physical transition window,
maximum local V difference is 0.16326%. The shallow rate collapse after the
first aging update occurs on both meshes; this is not inferred to be a new
mesh-induced instability from a visually small rate alone.

For tractions, use **native work averages** R_i/(M*1)_i with the driving or
normal load in place of R_i. These are distinct from the separately saved
consistent Q1 projections. Define increment error as
RMS[(q_c(t)-q_c(0))-(q_f(t)-q_f(0))], and analogously for sigma_n.
Initial differences are retained in the CSV, not removed from either run.

| Window | Shear increment error, Pa | Fine shear increment RMS, Pa | Relative | Normal increment error, Pa | Fine normal increment RMS, Pa | Relative |
|---|---:|---:|---:|---:|---:|---:|
| Top 0–2 km | 277.93 | 3290.99 | 8.45% | 15.28 | 5101.97 | 0.299% |
| Transition 13–20 km | 201.01 | 30354.02 | 0.662% | 14.51 | 664.97 | 2.182% |
| Interior 25–35 km | 15.36 | 2333.97 | 0.658% | 0.237 | 371.50 | 0.0637% |
| Bottom last 2 km | 20.65 | 59.50 | 34.71% | 34.59 | 66.02 | 52.40% |
| Whole fault | 129.06 | 7939.01 | 1.626% | 8.00 | 868.92 | 0.920% |

The bottom is very close to steady sliding. Its tiny V and Theta *increments*
are correspondingly less well determined than their total fields: the state
increment difference is 1819.75 s versus a fine increment of only 222.55 s,
although total Theta differs by just 0.00758%. This is why neither a large
relative incremental error nor a small total-field percentage should stand
alone as the interpretation.

At step 2, the raw bottom normal-stress departure ranges from
[-29960,+19932] Pa on the candidate and [-6648,+5369] Pa on the reference.
Bottom strain mismatch falls from 1.45054e-13 to 3.73853e-14 /s. Top raw
departures change from [+860,+9782] to [+3851,+6922] Pa; the physical top
normal-stress increase is retained rather than flattened away. Transition
raw ranges are similar on the two meshes (approximately -30/+29 kPa), and
the unrefined interior control is unchanged. Raw QP windows include the
small positive-phase FE tail; the actual surface rows use only admitted
source weights, as explicitly recorded in the CSV.

## Localized structure and refinement implication

The 15/18-km structure is already present in the initialized loading. Its
largest initial neighboring-chord departure is 0.04207/0.04186 Vp at 15 km
(candidate/reference); at step 2 it is 0.02761/0.02732 Vp at 18 km. A chord
departure measures curvature, not proof of numerical ringing. These features
are not removed by bulk refinement.

The reference creates additional grid-transition structure: at 23.1 km its
step-2 chord departure is 0.002843 Vp, versus a largest 0.000045 Vp in the
candidate's same 22–24-km window. The consistent-Q1 shear difference peaks
at 2977 Pa there. Additional small features align with the other patch edges
near 5, 10 and 110.47 km. This prevents calling the patch a globally more
accurate reference simply because its smallest cells are finer. The exact
initial raw extrema are localized above; later global raw extrema are
reported in `accepted_steps.csv`, but their strongest positions outside the
exported windows were not additionally sampled by a new replay.

**Smallest justified next decision:** retain 12.2 m as a useful economical
diagnostic for V/state/slip, not as a qualified production mesh. If percent-level
traction increments remain required, concentrate the next mesh design on the
top/bottom neighborhoods and the change of resolution along the localized
source. The reference demonstrates benefit from 6.1-m endpoint cells, but its
artificial patch edges must be accounted for and kept outside the relevant
measurement windows before it can qualify small traction increments. Moving
a patch edge is not proof that its global error has disappeared; the present
2:1 grading already makes the mesh conforming. The physical
13–20-km transition shows much smaller mechanical sensitivity; no new
constitutive, support, quadrature or solver correction is indicated here.
No additional mesh has been constructed to act on this recommendation.

## Verification and unrun items

Passed: mesh/plugin builds with `-j4`; Python syntax checks; unchanged
production-gate refusal; both complete 0/1/2 accepted-state runs; fresh
linear and final bulk/surface criteria; initial Theta and zero stress retention;
exact aging updates; first Maxwell update; inert H/geometry/I_h invariants;
current working-stress observer; both endpoint source and weak-row closure;
independent completion (maximum relative discrepancy 6.34e-8 on the reference).
Explicit CSV parsing was checked against the prior reader with exact equality.

Unrun: eight-to-twelve-step startup, restart equivalence, half-timestep
comparison, ell25 evolution, global fine-band construction and full-cycle
simulation. The first three are deferred because the spatial-accuracy
condition is not uniformly met, **not because the two-hour budget expired**.
The optional full-state-audit-from-step output switch was prepared for those
conditional checks but was not enabled; its runtime path is not claimed as
tested here. A draft continuation launcher was removed rather than leaving
an unexercised launch path as an apparent qualification deliverable. All
passing and failed earlier evidence and unrelated working changes are retained.

## Files and exact commands

Changes for this follow-up (distinct from the preceding configured-Dc work):

- `tests/bp3_length_scale_mesh.cc`: reference mode 2 adds both endpoint patches;
  previous modes and physical box remain unchanged.
- `benchmarks/reconstructed_fault/bp3/bp3.cc`: selected compact current-work
  exports; optional delayed full audit for the deferred lifecycle comparison.
  No constitutive or solver change.
- `length_coupled.py`: fresh diagnostic preparation, exact common clock,
  provenance and aggregate runtime cap; production gate stays separate.
- `analyze_length_coupled.py`: source/weak-row and accepted-state checks,
  common-coordinate profile/traction/initialization comparisons and plots.
- `length_initial_normal.py`: saved-output initial raw-extrema location check.
- This report and the fixture/benchmark README links; generated inputs and
  evidence are confined to `length-scale-study/coupled-diagnostic/`.

Executed at the repository root (completed output paths must not be reused):

```sh
cmake --build benchmarks/reconstructed_fault/performance/build-gmg --target bp3_length_scale_mesh -j4
cmake --build benchmarks/reconstructed_fault/bp3/build --target bp3 -j4
benchmarks/reconstructed_fault/performance/build-gmg/bp3_length_scale_mesh 50 12.20703125 2 benchmarks/reconstructed_fault/bp3/length-scale-study/coupled-diagnostic/target_cells_reference.txt
python3 benchmarks/reconstructed_fault/bp3/length_coupled.py completion
python3 benchmarks/reconstructed_fault/bp3/length_coupled.py prepare --label candidate-two
python3 benchmarks/reconstructed_fault/bp3/length_coupled.py prepare --label reference-two --reference
python3 benchmarks/reconstructed_fault/bp3/length_coupled.py run --label candidate-two
# Reference cap/grace were explicitly increased to 4000/3900 s before this
# launch; launch.json contains the updated PRM hash and exact resolved input.
python3 benchmarks/reconstructed_fault/bp3/length_coupled.py run --label reference-two
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/aspect-length-mpl python3 benchmarks/reconstructed_fault/bp3/analyze_length_coupled.py --compare candidate-two reference-two
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/bp3/length_initial_normal.py candidate-two
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/bp3/length_initial_normal.py reference-two
```

The launcher wraps each simulation in `timeout` and
`mpirun -np 4 --bind-to core --map-by core build-pf-cpdi/aspect-release`,
clears inherited ASPECT diagnostic flags, and sets one thread per rank.
The failed original `comparison.json` SHA256 remains
`5b407f6981e0c7b98345a3a4528791b835baf6142322337e15d5f9213cb78154`.
The untouched production-gated launcher SHA256 is
`6fbb4ea1f19ba7d68c779fe4773edf536cb7c988751de96ac1d950062f57ad9e`.
