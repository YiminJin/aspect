# K2.3 targeted 32-to-64 spatial check

Current disposition: the user accepts K2.3 as completed feasibility verification
with the spatial limitation subsequently localized in
`stage_K2_3_endpoint_localization.md`. This supersedes the earlier hold on
feasibility closure below, not its numerical evidence or the unresolved
reference status. No 128x512 pair is authorized now. K2.4 preparation is
recorded separately in `stage_K2_4_preparation.md`.

## Decision: verification passes; review before K2.4

The two authorized 64x256 cases complete at 0, 0.5 and 1 s with all existing
acceptance checks satisfied. No numerical or physical rule changed. The
accepted 32x128 pair is reused, not rerun.

At 1 s the mean-removed normal-feedback change is 9.40% of the fine signal
and the profile correlation is 0.99559. At 0.5 s, however, the change is
87.93% of the fine signal and the correlation is only 0.75077. Endpoint
features change width with the surface spacing. Therefore the requested
condition is **not met over all common accepted times**. Do not mark K2.3
resolved enough for K2.4 based only on the favorable final-time result.

One 128x512 matched-pair confirmation is justified to check whether the
endpoint contribution continues to contract while the interior profile
stabilizes. It is not authorized or launched. No production correction is
proposed or inferred from the discretization differences. K2.2's reference
remains provisional, and Gate K2 remains unmet.

## Scope and representation

Artifacts are under
`benchmarks/reconstructed_fault/uniform_shear/nonuniform/true-pressure/`.
`pilot64.prm` and `homogeneous64.prm` refine only the bulk mesh and fault
spacing: 32x128 -> 64x256 cells; 0.015625 -> 0.0078125 m structural spacing;
17 -> 33 fault vertices. Both use dt=0.5 s, top normal total traction -1000
Pa, top tangential velocity, full bottom velocity, x periodicity and no
pressure normalization. Each mesh's converged initial Q1 phase field stays
fixed. Physical initial data, support, full I_h, surface domain quadrature,
continuous Q2 ADD/count transfer, histories, Jacobian, solver tolerances and
lifecycle are unchanged. No build or production edit was needed in this turn.

Define Delta z = bumped z minus the matched homogeneous z at the **same**
resolution and time. Surface p, sigma_n and tau:N are consistent Q1
representations of the actual pre-commit constitutive-domain moments, not
bulk-column averages. Compare on the union of the two physical fault grids.
Means, RMS, cross moments and correlations integrate the piecewise Q1 fields
exactly; extrema are taken on their common breakpoints. No field is smoothed,
no initial error is subtracted, and no independent per-vertex K1 solve is used.

## Normal-feedback signal and spatial change

All values in this table are **Delta sigma_n**, in Pa. RMS is mean-removed.

| Time (s) | Grid | Mean | Mean-removed RMS | Minimum | Maximum |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | 32 | -1.30e-11 | 3.828802e-5 | -5.484895e-5 | 5.484890e-5 |
| 0 | 64 | 1.38e-11 | 3.725754e-5 | -5.285966e-5 | 5.285969e-5 |
| 0.5 | 32 | 7.980015e-6 | 4.456670e-5 | -6.797583e-5 | 2.209347e-4 |
| 0.5 | 64 | 3.977563e-6 | 3.348163e-5 | -6.294923e-5 | 2.196417e-4 |
| 1 | 32 | 5.179833e-5 | 2.189361e-3 | -2.157407e-3 | 6.338665e-3 |
| 1 | 64 | 2.576367e-5 | 2.209896e-3 | -2.052830e-3 | 6.058610e-3 |

| Time (s) | Mean change, 64 minus 32 (Pa) | Mean-removed change RMS (Pa) | Change / fine anomaly RMS | Anomaly profile correlation |
| --- | ---: | ---: | ---: | ---: |
| 0 | 2.68e-11 | 1.052669e-6 | 2.825% | 0.999984 |
| 0.5 | -4.002452e-6 | 2.943932e-5 | 87.927% | 0.750767 |
| 1 | -2.603466e-5 | 2.076223e-4 | 9.395% | 0.995589 |

The total (not mean-removed) change/signal ratios are respectively 2.825%,
88.116% and 9.468%. At 0.5 s endpoint peak amplitudes are similar, but the
narrow features occupy different widths on the two grids; similar extrema
alone do not demonstrate a stable profile. At 1 s the broad central peak and
side lobes are qualitatively stable, with remaining endpoint differences.
Both later-time means roughly halve. The small mean component is therefore
less settled than the final-time dominant anomaly.

![Common-coordinate normal feedback and signed pressure/stress components](../../../benchmarks/reconstructed_fault/uniform_shear/nonuniform/true-pressure/spatial-normal-feedback.png)

The figure above is available directly at the artifact path; common-coordinate
numerical profiles are `spatial-common-{0,1,2}.csv` and complete statistics are
`spatial-comparison.json`. (For a repository-relative path, use
`benchmarks/reconstructed_fault/uniform_shear/nonuniform/true-pressure/`.)

For clarity, the bumped **actual constitutive** sigma_n is not the tiny
Delta field. Its domain-weighted mean/raw integration-point range on grid 64
is 1000.000000000 / [987.152815,1012.847183] Pa at zero,
1000.000003978 / [994.913801,1005.086768] Pa at 0.5 s, and
1000.000025764 / [991.485768,1008.506752] Pa at 1 s. The homogeneous sigma_n
remains 1000 Pa, with raw departures below 4.3e-9 Pa and projected anomaly
RMS below 1.5e-11 Pa. Raw extrema are not substituted for along-fault variation.

## Signed p and tau:N contributions

The identity is Delta sigma_n = Delta p + **(-Delta tau:N)**. These two signed
profiles are plotted separately above and exported without removing their
means. The table gives means and mean-removed RMS in Pa; correlation uses
the mean-removed signed components.

| Time | Grid | Mean Delta p | RMS Delta p | Mean -Delta tau:N | RMS -Delta tau:N | Component correlation |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 32 | -1.38e-11 | 5.648303e-6 | 2.26e-15 | 3.263971e-5 | 1.000000 |
| 0 | 64 | 1.25e-11 | 5.470344e-6 | -1.13e-14 | 3.178720e-5 | 1.000000 |
| 0.5 | 32 | 5.320912e-6 | 2.754513e-5 | 2.659102e-6 | 2.007595e-5 | 0.745408 |
| 0.5 | 64 | 2.651758e-6 | 1.963471e-5 | 1.325804e-6 | 1.700190e-5 | 0.668659 |
| 1 | 32 | 3.553575e-5 | 1.010670e-3 | 1.626259e-5 | 3.165921e-3 | -0.976843 |
| 1 | 64 | 1.767154e-5 | 1.011155e-3 | 8.092128e-6 | 3.204104e-3 | -0.988485 |

At 1 s the signed components oppose each other strongly: the fine anomaly
cross moment is -3.20254e-6 Pa2. At zero and 0.5 s their aggregate cross
moments are positive. Thus cancellation is visible, not assumed to explain
every difference. The exported identity closes within 6.2e-11 Pa across both
resolutions and their difference. This is saved-field accounting only; no
new pressure treatment or residual evaluation is introduced.

## Slip response and initialization

| Time (s) | Fine Delta V RMS (m/s) | Spatial change RMS (m/s) | Fine Delta slip RMS (m) | Spatial change RMS (m) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 8.817458e-7 | 3.569366e-8 | 0 | 0 |
| 0.5 | 1.426738e-6 | 4.563705e-8 | 7.133688e-7 | 2.281853e-8 |
| 1 | 9.505587e-7 | 3.144283e-8 | 1.187178e-6 | 3.850268e-8 |

Later-time V/slip changes are about 3.2--3.3% of the fine total response;
profile correlations exceed 0.9994. Their stability does not override the
0.5-s normal-stress result. Delta V contains the direct Theta-friction effect
as well as bulk-mediated normal feedback.

Initial physical data were not changed, but the realized initial projections
are resolution dependent:

| Initial quantity | Grid 32 | Grid 64 |
| --- | ---: | ---: |
| Mean Theta (s) | 201.875000 | 201.875000 |
| Theta range (s) | [199.999024405,210.523585777] | [199.999883836,210.128218886] |
| Mean retained C (Pa) | 318.747655768 | 317.513762685 |
| Mean I_h (m) | 108.098095081 | 108.144833844 |
| Center phi | 0.5999103233 | 0.5997640169 |

These are initialization/discretization differences, not isolated mechanical
errors. Within each resolution the bumped and homogeneous initial phase CSVs
are byte-identical. The initial projection differences remain in the ensuing
trajectories; no reset or initial-error subtraction removes their influence.

## Acceptance checks and budget

Both new runs pass all three final nonlinear criteria and 17 fresh linear
checks each (9/4/4). Maximum fresh/target ratios are 0.894827 (bumped) and
0.965791 (homogeneous). No pressure quotient is applied. At all accepted
states the fine active/free counts are **0/33**, versus 0/17 on grid 32.

| Time (s) | Bumped bulk residual | Bumped free-surface RMS (Pa) | Homogeneous bulk residual | Homogeneous free-surface RMS (Pa) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1.30088e-11 | 6.98699e-8 | 1.09217e-11 | 5.92401e-8 |
| 0.5 | 9.56714e-12 | 6.13869e-9 | 9.37169e-12 | 5.94597e-9 |
| 1 | 3.28643e-12 | 1.09387e-6 | 3.28501e-12 | 1.04998e-6 |

Fixed bulk targets are about 2.244e-5 initially and 1.414e-7 at real steps;
surface targets are about 9.53e-6, 3.41e-6 and 7.93e-6 Pa. Actual weak loads
close and their independently evaluated consistent-mass residual norms agree
with production, using the unchanged verifier. No continued-failure output
is counted as an accepted state.

Both fine runs retain the frozen phase/H exactly, retain supplied particle
stress at zero, and match the independent real-step Theta update at exported
precision. Committed stress changes are observable: maxima 377.835/91.456 Pa
for the bumped real steps and 376.318/90.481 Pa for the homogeneous steps.
Existing transfer/timeline evidence is reused, not audited again.

Across 65 profile locations and 192 slip-normalization columns at all accepted
times, omitted fraction is **5.913959442e-5** and actual normalization error
is at most **6.006427240e-5**, separately below 1e-4. Global normalization is
about 0.999939935728. No support or full-I_h alteration is made; fixed-support
truncation remains a separate approximation.

Sequential simulation wall times are **150.524 s** and **139.018 s**; peak RSS
is **1715228 KiB** and **1611212 KiB** (about 1.64/1.54 GiB). Verification took
5.371/5.383 s and comparison 0.658 s: recorded simulation/analysis time is
**300.954 s**, comfortably below the approved 600 s aggregate cap. Both cases
are below 300 s. No timeout, retry or third spatial run occurred.

Commands from repository root, first for CASE=pilot64 then homogeneous64:

```sh
ASPECT_FAULT_NORMAL_STRESS_DIAGNOSTIC=1 python3 benchmarks/reconstructed_fault/uniform_shear/convergence/run_case.py benchmarks/reconstructed_fault/uniform_shear/nonuniform/true-pressure/CASE.prm --configuration Release --timeout 300
OPENBLAS_NUM_THREADS=1 timeout 45 python3 benchmarks/reconstructed_fault/uniform_shear/nonuniform/true-pressure/verify_pilot.py CASE --runtime-limit 300
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/aspect-k23-refinement-mpl timeout 45 python3 benchmarks/reconstructed_fault/uniform_shear/nonuniform/true-pressure/compare_spatial.py
```

The executable/plugin SHA-256 values match the accepted coarse pair exactly:
`abd1a3b8ec4cac3fd025f2ce9ee727abdaef4888d5507e859ce3a1915fb9f2a9` and
`eb42df87be0f9665beb0888d21fea2923ae2292274723e51f8cbfa3188ee90a9`.
Run commands, input hashes, costs, full linear histories and final checks are
preserved in `pilot64`/`homogeneous64` `.log`, `.resources.json`, and
`-verification.json` files. ParaView collections and raw exports remain in
their corresponding output directories. Source changes in this turn are only
the execution-budget/case options in the analysis script, the comparison
script, prepared inputs and documentation. Production changes already in the
working tree belong to the preceding accepted pilot and are untouched.

## Proposed next decision, not execution

A single additional **matched** 128x512 spatial level, through the same times,
would test whether the 0.5-s endpoint feature shrinks while the interior and
1-s profile remain stable. No solver or formulation issue has been demonstrated
that would justify reopening production development. Fourfold cell/particle
growth from the measured fine pair suggests roughly **10--20 minutes per
case, 6--8 GiB peak RSS**, with unmeasured linear-solver scaling. These are
planning estimates, not approved budgets or new performance measurements.
Such a confirmation needs explicit runtime/resource approval. Do not launch
it, K2.4, temporal refinement or forcing amplification automatically.
