# Inclined clean-start history-transfer comparison

## Decision

**Ordinary history transfer adds a measurable subcell normal-stress pattern on
the inclined fault. Native retention reduces that pattern, but does not remove
the much larger common finite-box stress variation. Its effect on the native
weak fault traction is small over these four steps.**

At 0.4 s, the normal-traction variation about each cell's affine spatial trend
is 0.581971 Pa RMS in A and 0.359707 Pa in B (38.2% lower). The matched A-minus-B
subcell residual is 0.524098 Pa RMS, almost entirely deviatoric rather than
pressure. In contrast, the A-minus-B neighboring-chord variation of the weak
fault traction is only 0.006429 Pa RMS. Native weak averaging cancels much of
the raw-QP pattern.

This is evidence that the ordinary transfer/update cycle can contribute to
raw normal-stress bands. It is **not** evidence that it dominates BP5's weak
normal-traction bands, or that B is a generally qualified production method.
No new smoothing, production change, C branch, or BP5 trajectory was run.

## Controlled problem

- Cartesian box [-0.5,0.5]^2 m, 64x64 unrotated square cells; h=0.015625 m.
- Straight prescribed 60-degree fault through the centre, from
  (-0.28867513459481287,-0.5) to (0.28867513459481287,0.5).
  Existing `Fit prescribed geometry to phase field = false` retains this line.
  Exported maximum normal displacement from the intended line is 2.78e-17 m.
- 38 vertices, 37 elements, actual spacing 0.0312081226589 m.
- Frozen Q1 AT1 distance profile, core phi=0.6, ell=0.15625 m, ell/h=10.
  The production normalization, source map and quadrature are unchanged.
- V=0.005 m/s prescribed at every surface node. No free-rate/friction-feedback
  solve is being compared; mechanical normal traction is still measured.
- 3x3 fixed particles/cell, 36864 particles. Same physical initial state and
  all-zero retained Maxwell tensor in independent A and B starts. Initial
  particle/property hash is `36864 1909751090143304754` in both.
- G=1e6 Pa, eta=1e20 Pa s; artificial dt0=0.1 s. Four real steps of 0.1 s
  end at 0.4 s. Timestep-zero evaluated stress is not retained. Each real
  stress history is published once and read by the following solve.
- Same Q2 velocity/stress FE, Q1 pressure/phase, native 3x3 bulk quadrature,
  original nonlinear 1e-8 and linear 1e-9 tolerances.

With s=(1/2,sqrt(3)/2), n=(-sqrt(3)/2,1/2), prescribe

    u_boundary(x) = 0.01 (n dot x) s

on all four walls. This affine field is exactly representable and divergence
free: s dot n=0. Measured net boundary flux is below 1e-15 m^2/s at every
accepted state. There is no mass source or periodic boundary. The mesh itself
is not rotated. This is compatible loading, **not** an assertion of zero
elastic strain or an infinite-fault analytical solution. Finite-box/end effects
are present in both branches.

Volume pressure normalization means integral(p)=0. `Surface pressure=1000 Pa`
sets the separate adiabatic friction pressure; it is **not** added to the
reported mechanical pressure or normal traction. All plots use the same
volume gauge, without independently shifting A/B profiles.

## What is compared

A is unchanged production particle update, distance-weighted interpolation,
shared-DoF averaging and constrained FE working history. B retains each cell's
native-QP full tensor, uses the existing benchmark callback in bulk and work
surface mechanics, and advances that native history once per accepted step.
B's particle stresses are shadow output, not its next mechanical history.

The exported current tensor is the implemented constitutive evaluation

    tau_k = beta tau_in + 2 kappa [sym(grad u_k) - chi V S].

It uses the incoming mechanics history, not newly published particle values;
there is no stress rotation in this law. Pressure is the accepted physical FE
pressure. We export p, d=-n^T tau_k n, and sigma=p+d at the **same actual QPs**.

For each surface row, the observational native average is

    Q_i^w = sum_q J_q chi_q N_i(q) Q_q / sum_q J_q chi_q N_i(q).

These are normalized native weak rows, not consistent-Q1 nodal coefficients
and not centreline interpolation. Geometry, quadrature coordinates, chi, basis
coordinates and weights are verified identical across both branches and all
steps; row masses sum to the total native work weight. Pressure plus d closes
to sigma. The interior row window |s|<=0.2 m contains 12 vertices with centres
in [-0.171644675,0.171644675], at least 0.405705595 m from the tips.

Two distinct variation diagnostics are kept:

1. **Along-fault weak rows:** deviation from the neighboring-node chord, with
   native row-mass weighting. This includes broad curvature; it is not all
   numerical noise. The unsmoothed profiles are retained.
2. **Subcell raw fields:** weighted residual about an affine (x,y) fit to all
   nine actual QPs of each complete interior cell. Cell centres satisfy
   |s|<=0.18, |r|<=0.035 m. This only measures subcell structure and never
   modifies a field, a history, or a traction. Common smooth curvature can
   contribute, so the controlled A/B difference is essential.

The raw-field image shows actual QPs in |s|<=0.2, |r|<=0.05 m. There is no
synthetic r=0 line and no interpolation between CSV points.

## Results

Initialization and the first real solve agree **exactly** between A and B in
the exported velocity/current stress/traction. Differences start only after
the first retained history has traversed the different paths.

### Transfer-load and velocity checks

The jump is ||C_v^T int (tau_next-tau_current):epsilon(w)||_2 in Pa*m.
An independent Python Q2 assembly reproduces the C++ jump to within 2e-11
Pa*m, with full symmetric-tensor contraction and the same four Dirichlet walls.

| Real step | A global jump | B global jump | A interior jump | B interior jump |
|---|---:|---:|---:|---:|
| 1 | 36.36045 | 2.33e-13 | 0.0814561 | 7.09e-15 |
| 2 | 46.86114 | 4.67e-13 | 0.152211 | 1.42e-14 |
| 3 | 54.86182 | 7.05e-13 | 0.221340 | 2.14e-14 |
| 4 | 61.30697 | 9.34e-13 | 0.287752 | 2.84e-14 |

The interior load norm selects already assembled velocity rows with support
point |s|<=0.2, |r|<=0.1; it does not clip their integration domains.
Large global jumps include the boundary regions and must not be attributed
entirely to the interior. B's jump is at numerical summation/interpolation scale.

Final A-B velocity RMS is 8.9961e-7 m/s over the box and 4.1194e-7 m/s in the
interior |s|<=0.2, |r|<=0.1. Interior maximum is 5.1981e-7 m/s; global maximum
is 1.8868e-5 m/s. Interior RMS is 0.00824% of prescribed V.

### Raw subcell structure, Pa RMS

| Step | A sigma affine residual | B sigma affine residual |
|---|---:|---:|
| 1 | 0.0899267 | 0.0899267 |
| 2 | 0.226709 | 0.179853 |
| 3 | 0.401042 | 0.269780 |
| 4 | 0.581971 | 0.359707 |

At step 4:

| Component | A | B | A-B affine-residual RMS |
|---|---:|---:|---:|
| p | 0.375910 | 0.377499 | 0.003281 |
| d | 0.834720 | 0.695886 | 0.524007 |
| p+d | 0.581971 | 0.359707 | 0.524098 |

The A-minus-B raw-QP map has a clear cell/QP-scale pattern in d and sigma,
whereas the pressure difference is predominantly smooth. Different signed
components can cancel: differences between the RMS norms are not the RMS
norm of the difference.

### Native weak traction, final interior, Pa

| Quantity | A chord RMS | B chord RMS | A-B chord RMS | A-B total RMS |
|---|---:|---:|---:|---:|
| p | 1.070856 | 1.069386 | 0.001539 | 0.821672 |
| d | 1.236192 | 1.240574 | 0.004999 | 0.038084 |
| p+d | 0.498819 | 0.500057 | 0.006429 | 0.799083 |

Thus B does **not** reduce the existing weak normal-traction chord variation
in this test. The A-B chord difference is 1.29% of B's chord scale. The weak
sigma mean is 680.2292 Pa (A), 681.0271 Pa (B); most of their total RMS
difference is a smooth mean shift. Raw-QP A-B sigma RMS over the whole
associated interior strip is 1.383923 Pa, versus 0.799083 Pa after weak
averaging. Do not mistake the latter for the raw-stress difference.

## Verification, provenance and cost

- Both runs: initialization plus exactly four accepted real steps, genuine
  nonlinear convergence and fresh returned-direction linear checks.
- Largest reported relative nonlinear residual: A 7.894e-10, B 7.174e-10.
- Frozen particle positions, phase/source/geometry invariants pass. Native
  incoming history equals the preceding retained history at matched QPs;
  the analogous production FE transfer identity also passes.
- Independent integration tests: constant tensor has zero free weak load;
  linear tensor matches its analytical divergence load. **2/2 pass**.
- Existing moment algebra tests **3/3 pass**; clean-cycle test **1/1 pass**.
- No new two-rank qualification or full ASPECT test suite was run.
- Release, one MPI rank, one thread: A 22.83 s / 525.4 MiB peak child RSS;
  B 34.04 s / 523.6 MiB. Two simulation runs only, total 56.87 s; each had
  a 120-second hard cap. No retry or tolerance change.

Same executable in both runs, SHA256
`d053c7a596887b6e630574c8db660520957a890d672efa8a83ec22087392006c`.
Plugin SHA256
`7373eb70a98726134d4b5ede9ce19452552f3a31bdcee2b1d0c6308418b51870`.
Source/plugin/PRM snapshots and hashes are in each branch's `execution.json`.
ASPECT production source was not modified for this task. Benchmark changes:
optional straight-fault angle in `clean_stress_cycle.cc`; matched native-QP
traction and geometry exports in `moment_cycle.cc`; inclined runner selection;
two complete PRMs, fault coordinates, analysis and two cheap tests.

Reproduction (from repository root, choose an unused output root):

```sh
cmake --build benchmarks/reconstructed_fault/bp5/build --target bp5_moment_cycle -j4
python3 benchmarks/reconstructed_fault/bp5/run_moment_cycle.py production --inclined --compatibility-audit --binary build-pf-cpdi/aspect-release --root /tmp/inclined-replay
python3 benchmarks/reconstructed_fault/bp5/run_moment_cycle.py native_history_reference --inclined --compatibility-audit --binary build-pf-cpdi/aspect-release --root /tmp/inclined-replay
python3 benchmarks/reconstructed_fault/bp5/analyze_inclined_moment.py /tmp/inclined-replay
python3 benchmarks/reconstructed_fault/bp5/test_inclined_moment.py
```

`analysis.json` contains every accepted-state metric; branch
`output-*/weak_profile_*.csv` contains unsmoothed weak profiles, and
`traction_*_rank0.csv` contains raw current-mechanics samples and incoming
tensors. `interior_tractions.png` compares weak profiles;
`matched_raw_qps.png` isolates the raw pattern at identical QPs.
Legacy horizontal-fit columns in `summary.csv` are not interpreted as an
inclined correction; A/B never replace history by that horizontal fit.

## Remaining decision

Retain the distinction between raw stress bands and the weak traction actually
entering fault mechanics. This bounded result supports a transfer contribution
to the former, but not a dominant transfer explanation of the latter. Longer
accumulation, free RSF/normal feedback, BP5 mesh grading and its physical scales
remain outside this small four-step discrimination. No production remedy is
qualified by this test alone.
