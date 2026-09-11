# K2.4 bounded parity/alignment sensitivity: completed pair

Accepted disposition: the user closes K2.4 as this completed bounded
sensitivity check. The approximately 1%-or-less fixed-interior differences
at real steps do not explain the larger unresolved K2.3 spatial uncertainty.
K2 remains bounded verification with unresolved convergence, not Gate K2
passage. K3 preparation is separately authorized as a sequencing exception;
see `stage_K3_preparation.md`. No 128x512 K2 run is authorized.

## Decision summary

Both approved 64x255 cases completed through 1 s, with actual exported geometry
confirming the fault at y=0 inside the central cell row. All existing nonlinear,
fresh-linear, history, weak-balance, containment and actual-normalization checks
pass. The runs took 151.164 and 131.842 s, totaling 283.007 s; neither was retried.
No production equation, parameter, tolerance, support, or pressure treatment
changed.

The bumped-minus-homogeneous profiles remain close to the provisional 64x256
baseline. At .5 s the global Delta sigma_n RMS difference is .490% of the
baseline signal. On the fixed interior excluding .046875 m at each end,
Delta sigma_n and -Delta(tau:N) differences are .998% and 1.007%; at 1 s they
are .383% and .359%. Endpoint profiles nearly overlap. These are measured
sensitivities, not newly introduced pass thresholds.

Recommend completing the authorized bounded K2.4 sensitivity check, without
claiming post-convergence verification. This combined parity, spacing, hierarchy,
particle-sampling and initialization perturbation does not isolate alignment
causally. It neither resolves nor removes K2.3's 19.6%/21.9% interior spatial
uncertainty. K2.2/K2.3 references remain provisional and Gate K2 unmet. Stop
here; no 128x512 confirmation, production correction or later campaign follows.

## Execution and actual geometry

The prepared parameter files were executed unchanged, bumped first. Its full
verification passed before launching the homogeneous case. Each process had
a 300-s subprocess timeout; there were no retries. The one-rank Release
executable and benchmark-plugin hashes exactly match those of the accepted
64x256 baseline; full hashes, commands and git revision are in each resource
JSON. The recorded source baseline is `8ace2d1b8d71d1de6d8a3ae1249fda502c276d01`
plus the retained opt-in diagnostic working-tree changes; no new production
source edits were made by this task.

Actual exported Q1 mesh coordinates (`phase_0.csv`, unchanged at later times)
show 65 x 256 vertex coordinates and 16320 cells on one level. The reconstructed
surface exports at all three accepted times in both cases show:

| Measurement | Realized value |
|---|---:|
| Physical box | [0,.25] x [-.5,.5] m |
| h_x | .00390625 m |
| h_y range | [.0039215686274509665, .0039215686274510775] m |
| Change in h_y versus baseline | +.3921568627% |
| Reconstructed vertex y range | [-1.866623e-15, -1.781580e-15] m |
| Containing normal row bounds | [-.0019607843137254832, +.0019607843137254832] m |
| Reference normal coordinate range | [.499999999999524, .499999999999546] |
| Surface discretization | 33 independent nodes; h_Gamma=.0078125 m |

Thus the physical line remains at y=0 to roundoff and passes through normal
cell interiors, not horizontal faces. It still intersects vertical cell faces.
Both cases have identical reconstructed geometry. The measured support
half-width is unchanged at .3088215939070757 m. Top normal traction remains
-1000 Pa, top tangential/bottom full velocities and bulk x periodicity remain,
and pressure normalization remains `no`. Measured prescribed-velocity errors
are <=2.711e-20 m/s. There is no pressure shift or surface endpoint identification.

## Acceptance and histories

Numbers below are actual final residuals, not inferred from process exit or
accepted-state output. Surface targets are 1e-8 times the fixed surface scale.

| Case | Time (s) | Bulk residual / target | Surface RMS (Pa) / target | Fresh linear checks |
|---|---:|---:|---:|---:|
| Bumped | 0 | 1.298e-11 / 2.235e-5 | 7.166e-8 / 9.519e-6 | 9 |
| Bumped | .5 | 9.509e-12 / 1.408e-7 | 6.177e-9 / 3.401e-6 | 4 |
| Bumped | 1 | 3.269e-12 / 1.409e-7 | 1.096e-6 / 7.905e-6 | 4 |
| Homogeneous | 0 | 1.164e-11 / 2.235e-5 | 6.078e-8 / 9.521e-6 | 9 |
| Homogeneous | .5 | 9.427e-12 / 1.408e-7 | 5.982e-9 / 3.402e-6 | 4 |
| Homogeneous | 1 | 3.295e-12 / 1.409e-7 | 1.052e-6 / 7.906e-6 | 4 |

All 34 fresh-linear checks meet their requested targets; the largest
fresh/target ratio is .953535. Pressure-quotient handling is inactive, as
required for these true-pressure equations. All 33 nodes remain free at every
accepted time in both cases. Actual domain-integrated weak terms close, the
mass-based surface norm agrees with production, and admitted particle volume
equals the surface-system mass sum. The largest weak-balance component is
8.4471e-9 in the stored weak-load units.

The independent full-profile integral ranges from 108.149281242024 to
108.149281242025 m over 65 transverse profiles. The maximum omitted fraction
is **5.9199498124e-5**, below the explicitly extended 1e-4 allowance (original
target 1e-6). Separately, the largest measured actual bulk/surface slip-
normalization error over both cases/times is **5.8724063790e-5**, below 1e-4.
The global normalization is approximately .999941275936. No tail was
renormalized away; support and full I_h are unchanged in definition.

The phase field is bitwise frozen after initialization. Stable particle IDs,
retained H, supplied timestep-zero stress [0,0,1500] Pa, and split Theta
updates pass the reused lifecycle checks. The independent exponential-aging
reference agrees with the saved Theta updates to the reported double precision
(maximum measured difference 0 s). Newly committed stress changes are nonzero:
377.068/91.312 Pa at .5/1 s in the bumped case, and 375.660/90.335 Pa in the
control. The previously verified transfer timeline is reused, not reinvestigated.

For context, actual bumped constitutive sigma_n has domain-weighted means
1000.000000, 1000.000003979, 1000.000025781 Pa at 0/.5/1 s. Raw integration-point
ranges are [987.148605,1012.851393], [994.913222,1005.088368], and
[991.484186,1008.509771] Pa. These raw ranges are not the much smaller represented
bump-minus-control feedback. The homogeneous sigma_n stays at 1000 Pa to
approximately 5e-9 Pa at worst.

## Matched constitutive and slip comparison

For each configuration, Delta means bumped minus its own homogeneous control.
Compare staggered Delta minus baseline Delta at common physical arclength.
Pressure and sigma_n are reconstructed from actual constitutive surface weak
moments with the consistent mass matrix, not normal-column bulk averages.
The identity Delta sigma_n = Delta p - Delta(tau:N) is checked. Reported
RMS/means use exact integration of the represented Q1 fields.

| Time (s) | Field | RMS difference | Difference / baseline Delta RMS |
|---:|---|---:|---:|
| 0 | sigma_n | 4.8363e-7 Pa | 1.298% |
| 0 | p | 8.2409e-8 Pa | 1.506% |
| 0 | -tau:N | 4.0123e-7 Pa | 1.262% |
| .5 | sigma_n | 1.6519e-7 Pa | .490% |
| .5 | p | 2.7504e-8 Pa | .139% |
| .5 | -tau:N | 1.4198e-7 Pa | .833% |
| 1 | sigma_n | 8.3476e-6 Pa | .378% |
| 1 | p | 2.9846e-6 Pa | .295% |
| 1 | -tau:N | 1.1329e-5 Pa | .354% |

The global mean changes in Delta sigma_n are -3.7234e-11, 1.6957e-9,
1.7077e-8 Pa at 0/.5/1 s. Mean-removed relative RMS differences are 1.298%,
.493%, .378%; removing a mean does not explain the differences. Signed p and
-tau:N means, extrema and mean-removed quantities are retained in the JSON
and CSV so cancellation remains visible.

At .5/1 s, Delta V RMS differences are 9.5849e-10/1.1462e-9 m/s
(.0672%/.1206% of baseline Delta V). Delta accumulated-slip RMS differences
are 4.7925e-10/1.0515e-9 m (.0672%/.0886%). Initial accumulated slip is zero;
the timestep-zero kinematic solution is not integrated as a real step.

The following fixed physical interior is [.046875,.203125] m, matching the
earlier exclusion of three **coarse K2.3** elements per end (six elements on
either of these fine surface grids). Denominators use this same interior.

| Time (s) | Interior sigma_n RMS difference / baseline signal | Interior -tau:N RMS difference / baseline signal | Endpoint share of total squared sigma_n difference |
|---:|---:|---:|---:|
| 0 | 5.2538e-7 Pa / 1.298% | 4.3586e-7 Pa / 1.262% | 26.24% |
| .5 | 1.7721e-7 Pa / .998% | 1.5248e-7 Pa / 1.007% | 28.08% |
| 1 | 1.0506e-5 Pa / .383% | 1.4259e-5 Pa / .359% | 1.009% |

These small parity-comparison differences are not exclusively endpoint-local.
That does not contradict the endpoint dominance of the earlier 32-to-64
spatial difference: they are different perturbations. The full JSON includes
left/right/interior measures for exclusions .0078125, .015625, .03125 and
.046875 m, including signed p and -tau:N. No threshold is fitted to the prior
19.6%/21.9% uncertainties. This check shows no large new interior response to
the chosen parity change; it does not establish spatial convergence.

![Signed matched fields at common times](../../../benchmarks/reconstructed_fault/uniform_shear/nonuniform/alignment/comparison.png)

![Normalized endpoint comparison at .5 s](../../../benchmarks/reconstructed_fault/uniform_shear/nonuniform/alignment/endpoints.png)

## Initial representation is part of the result

The physical input functions, retained histories and initialization rules are
unchanged. Normal cell size, hierarchy and particle ordinates differ. Each
mesh initializes once from those functions; no subsequent output resets a
reference or removes initial-error influence.

| Initial quantity | Baseline 64x256 | Staggered 64x255 |
|---|---:|---:|
| FE phi at physical y=0 | .59976401690679 | .59963373547259 |
| Mean projected cohesive history C (Pa) | 317.513762684786 | 318.494680773238 |
| Mean projected I_h (m) | 108.144833843899 | 108.149281240714 |
| Maximum projected Theta (s) | 210.128218885526 | 210.128218885544 |
| Mean initial bumped V (m/s) | .000315344281268 | .000314901808477 |

On the staggered mesh phi(y=0) is interpolated inside the central cell, not
read from a nonexistent y=0 vertex row. Each bumped/control pair has identical
initial phase arrays. The changed C is approximately +.981 Pa; it has not been
subtracted from the history or ignored in interpreting later mechanics. Nor
can the small matched-response difference prove that all representation effects
are independently small: cancellation between their effects is not excluded.

## Artifacts, commands and scope

All new artifacts are in
`benchmarks/reconstructed_fault/uniform_shear/nonuniform/alignment/`:

- `pilot.log`, `homogeneous.log`, and corresponding `.resources.json` files.
- `*-geometry.json`, `*-verification.json`, `*-verification.log`.
- `*-surface-{0,1,2}.csv`, `comparison-{0,1,2}.csv`, `comparison.json`,
  `comparison.log`, and the plotted PNGs.
- `pilot/` and `homogeneous/`: actual mesh/profile, surface, weak-traction,
  parent-particle and bulk diagnostics; ParaView-readable bulk, particles and
  reconstructed-fault outputs in their `solution/`, `particles/` and
  `reconstructed_faults/` subdirectories.

For each CASE, sequentially, with verification before advancing:

```sh
ASPECT_FAULT_NORMAL_STRESS_DIAGNOSTIC=1 python3 benchmarks/reconstructed_fault/uniform_shear/convergence/run_case.py benchmarks/reconstructed_fault/uniform_shear/nonuniform/alignment/CASE.prm --configuration Release --timeout 300
OPENBLAS_NUM_THREADS=1 timeout 120 python3 benchmarks/reconstructed_fault/uniform_shear/nonuniform/alignment/verify_case.py CASE
```

After both checks passed:

```sh
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/aspect-k24-mpl timeout 120 python3 benchmarks/reconstructed_fault/uniform_shear/nonuniform/alignment/compare_cases.py
```

All commands exited zero; the verification explicitly checks final residuals
and does not rely on exit status alone. Saved-state analysis ran within the
short diagnostic limits. The endpoint plot was visually inspected. Maximum
RSS was 1723716 KiB (1.644 GiB) bumped and 1587496 KiB (1.514 GiB) homogeneous.
The measured simulation sum is 283.006553730 s, below 600 s. Nonfatal MPI
network-discovery warnings appear in the one-rank logs; neither run failed.

New scripts are analysis-only (`verify_case.py`, `compare_cases.py`). Existing
parameter files, benchmark plugin and production source were left unchanged.
No broad tests, solver/transfer audits, stress smoothing, normalization changes,
or further simulations were performed. Bulk stress compositions remain old
history inputs; particle stress output is the accepted committed stress.
