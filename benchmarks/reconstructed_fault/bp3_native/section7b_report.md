# Section 7B: bounded serial stationary-history BP3 comparison

All four coupled runs pass: uniform creep and the common smooth-state perturbation,
each with accepted FE history and retained particle history. Each reaches exactly
ten positive-time steps, dt=200000 s, T=2000000 s (23.15 days), after a separately
recorded kinematic step zero. Both one-positive-step gates passed before extension.
There were no cutbacks, changed schedules or scientific-model changes. This establishes
the restricted coupled lifecycle and measures the backend difference at one resolution;
it does not qualify temporal/spatial accuracy, later BP3 events or MPI.

## Matched setup and provenance

Baseline `c33edaaeb145e5d3dce155bb815552ef80efc5f9`, branch `fault-propagation`.
Incoming first-pass edits and historical evidence are preserved. Both backends use
the same current Release executable and freshly built maintained BP3 plugin, with
GCC12.4, OpenMPI5.0.6, deal.II9.6.2, Trilinos14.2 and p4est2.8.7. SHA256 identities:

| Artifact | SHA256 |
|---|---|
| ASPECT | `1ba2f7f43b37b75b1462d9cd5e48998fc8cc58b891e943829be8240f70776d12` |
| Maintained BP3 plugin | `676177bc1935cdee156227b9aeae1a2990a5ed0ddbf5bc34955b57a82e8dcd76` |
| Common trajectory observer | `ec028e17382d642d166f0528368c7947632fcd592577886f6a94bd65895c06da` |

The translated Box is 2000×1000 m, origin (-1234.5,137.25) m, with a straight
60° through-going fault, 4136 graded Q2/Q1 cells (7.8125–125 m), 59 surface nodes,
ell=20 m and core=.6. Frozen geometry/profile, native consistent cell-interval I_h
with eight surface subdivisions and paired automatic boundary completion are shared.
The oblique bilinear profile uses general cell-interval integration. Stress/history
evaluation uses the actual segment normal, tangent and source association, including
constant endpoint continuation in the paired boundary wedges. No horizontal-ray
formula, normalization floor or omitted completion contribution is introduced.
G=32038120320 Pa, eta=1e26 Pa s, a=.025, b=.015, Dc=.008 m,
Vp=Vinit=1e-9 m/s, reference rate=1e-6 m/s, sigma0=50 MPa, damping=4624440 Pa s/m.
The background shear is 26.546122365 MPa. Full left/right/bottom velocity loading,
natural top, true pressure, no pressure normalization and the 20 m Helmholtz filter
are unchanged. The completed velocity-lift checks report zero error on all three
prescribed sides. All 66176 particles remain at their initial positions and IDs;
load balancing is `none`, so no insertion/deletion occurs.

**Effective mechanical quadrature is checked, not inferred:** the legacy backend
uses the native 3×3 Gauss rule. Setting the existing FE option to three makes its
rule identical, including every point and weight, across the coupled operators.
No test-only core quadrature adjustment or production-default change was needed.
Initial mesh, phase DOFs, rule and physical particle identity files are byte-identical
within each pair; surface coordinates, I_h, theta and background normal tractions
also match exactly. Resolved inputs differ only in output path and history option.

The maintained plugin is compiled with its historical `ASPECT_BP3_LOCAL_BOTTOM_TEST`
control, selecting uniform strengthening and the existing smooth state initializer.
`ASPECT_BP3_LOCAL_OSCILLATION_TEST` is absent, preserving physical loading.
The second pair uses theta0 exp[.02 sin²(pi(y-origin_y)/height)] in both members;
the first uses zero perturbation. This is an explicit benchmark control.

## Positive-step correctness and publication

The stationary no-Stokes setup guard already skips velocity checks only for the
no-Stokes manufactured scheme. The coupled scheme enters Stokes/RSF assembly and
the normal accepted path, retaining full loading checks; no guard edit was necessary.
Step zero uses the initial constitutive interval for kinematics, while committed
bulk history and cumulative slip remain zero. At step one the converged initial
mechanical solution is reused (zero Newton updates), then histories commit once.
Steps 2–10 require two Newton updates each. This is positive physical evolution,
with direct state/slip/history checks, rather than another End-time-zero fixture.

The authoritative mature FE recurrence remains

\[
\tau=2G D_3(\operatorname{symgrad}d_R)-2\chi ZS,\qquad
d_R^n=\beta_n d_R^{n-1}+(\eta_{ve,n}/G)u^n,\quad
Z^n=\beta_n Z^{n-1}+\eta_{ve,n}V^n.
\]

Here beta=exp(-dt G/eta), eta_ve=-eta expm1(-dt G/eta), chi=h(phi)/I_h,
and D3 subtracts one third of the trace in the plane-strain embedding. Linearity
with the frozen profile gives the direct Maxwell recurrence tested pointwise.
Z is slip memory, independent of mature cohesive traction, which stays exactly
zero. Final FE max Z is 6.45445e7 /6.45151e7 Pa m for creep/perturbed;
the recurrence defect is at most 7.45e-9 Pa m. The particle backend stores tensor
history directly; its exported Z=0 is an absent representation, not zero physical
slip memory. Background shear/normal loads remain surface data and are never added
to the zero-initialized bulk perturbation history.

Each acceptance is audited after the maintained diagnostics: fixed phase/I_h/H,
finite positive state, one mechanical acceptance, theta aging, cumulative slip and
terminal stress/Z. There is no diagnostic second increment. Sampled direct Maxwell
checks use 216 bulk QPs and 537 stationary particles per state; native FE publication
also validates all derived particle samples. Across the trajectories the maximum
stress defects are 9.57e-9 Pa (FE) and 1.01e-9 Pa (particle), at sample stress scales
up to 156/193 kPa. Their maximum fractions of the original roundoff allowance are
.00335/.01179. Minimum I_h=12188.3241089111 m is constant; minimum theta>7.9848e6 s.
Raw normal equals background+pressure+tensor load to within 2.46e-7 Pa.

| Coupled run | Result | Wall s | Max normalized residual | Positive-step Newton / Krylov updates |
|---|---|---:|---:|---|
| FE creep | Pass, 10 steps | 106.64 | 7.39e-10 | 0 then 2; 0 then 68–70 |
| Particle creep | Pass, 10 steps | 62.81 | 6.84e-10 | 0 then 2; 0 then 46–47 |
| FE perturbed | Pass, 10 steps | 106.44 | 7.54e-10 | 0 then 2; 0 then 69–71 |
| Particle perturbed | Pass, 10 steps | 62.24 | 1.27e-9 | 0 then 2; 0 then 45–46 |

All are below the existing nonlinear1e-8/linear1e-9 criteria. Steps 2–10 have
normalized residual≤8.05e-13. Surface residual RMS is ≤1.18e-6 Pa except the
particle perturbed step-zero/step-one reuse (.001427 Pa, still accepted by the
unchanged criteria); subsequent values are ≤1.34e-7 Pa. Theta relative defects
are ≤2.23e-16. Passing FE/particle gates cost 19.81/12.99 s. Runs were serial and
sequential; timings include the compact observer and are not a general speed claim.
All simulation launches, including failures, used 423.76 s of the declared 3600 s
budget. Caps were 400 s per gate, 700 s per trajectory and explicit step limits.
Checkpoint writing/resume and bulk/particle visualization were disabled.

## Common-time physical response

Profiles at every common time are retained. Below, differences are FE minus
particle; table entries give maximum absolute difference (arc-length Q1 RMS).
Omega norms interpolate the exported nodal V theta/Dc values.

| Field | Step-zero creep max | Final creep max (RMS) | Final perturbed max (RMS) |
|---|---:|---:|---:|
| V, m/s | 4.528e-14 | 1.897e-12 (1.439e-12) | 1.896e-12 (1.424e-12) |
| theta, s | 0 | 1873.24 (1301.04) | 1873.32 (1299.16) |
| Omega | 4.528e-5 | .0016561 (.0012762) | .0016563 (.0012745) |
| Cumulative slip, m | 0 | 2.034e-6 (1.403e-6) | 2.033e-6 (1.388e-6) |
| Shear, Pa | 46.65 | 2335.51 (1723.91) | 2334.18 (1713.56) |
| Raw normal, Pa | 35.70 | 2609.61 (247.47) | 2657.62 (249.71) |
| Filtered normal, Pa | 22.43 | 351.44 (68.25) | 366.48 (66.89) |
| Pressure contribution, Pa | 33.15 | 3290.37 (257.38) | 3327.09 (260.03) |
| Tensor contribution, Pa | 28.60 | 1119.82 (165.36) | 1119.75 (164.83) |

Perturbed step-zero discrepancies are similar: max V=4.527e-14 m/s,
shear/raw/filtered=46.62/35.70/22.41 Pa, with exactly matched initial theta and
zero cumulative slip. Thus a difference exists before any transported/projected
accepted stress history. Initial mechanical response is not imposed uniform V.

Final maximum differences relative to Vp, theta0=8e6 s, Omega=1 and Vp*T=.002 m
are .190%, .0234%, .166% and .102%, respectively, in both pairs. Final shear
RMS differences are 66.0%/65.8% of the measured shear perturbation RMS, although
max differences are only .00880%/.00879% of background shear. Raw/filtered
normal RMS differences are 10.97%/12.67% (creep), 10.66%/10.01% (perturbed)
of their perturbation RMS. Relative to 50 MPa, final maxima are .00522%/.000703%
and .00532%/.000733%. Pressure/tensor RMS differences are 6.36%/6.51% and
6.14%/6.62% of the larger backend's corresponding RMS. Small background-scaled
differences therefore do not establish perturbation accuracy.

The V difference is predominantly a signed mean shift: creep mean=-1.432e-12,
oscillatory RMS=1.345e-13 m/s; perturbed mean=-1.417e-12, oscillatory RMS=1.380e-13.
At T, max V is 1.01171/1.01360 Vp (creep FE/particle), 1.01097/1.01287 Vp
(perturbed); theta, Omega and slip remain finite and smooth at the sampled scale.

The FE uses consistent_deviator (3D trace subtraction), whereas the retained
particle law uses the full symmetric gradient. The existing diagnostic evaluates
the instantaneous normal-stress shift 2 eta_ve div(u)/3 on each own path:
up to 5.459/4.974 kPa (creep FE/particle), 5.450/4.968 kPa (perturbed).
Discrete divergence is not identically zero (sample maxima≈1.28e-12/s).
Pressure can compensate part of this isotropic shift; normal and shear responses
also depend on retained-history representation and discrete transfer. This number
is not a causal subtraction of the final coupled difference. The API column named
`deviatoric` denotes the tensor load; in the legacy path it includes the retained
full-gradient convention. No legacy-law switch or projection-only attribution is made.

## Spatial diagnostics, limits and next issue

Final maximum local-chord residuals in raw normal traction, upper/interior/lower
200 m regions, are .980/.325/34.142 kPa (FE creep) versus .615/.375/30.512 kPa
(particle); filtered lower residuals are 3.820/3.427 kPa. Perturbed raw lower
values are 34.822/31.149 kPa. Hann-windowed, line-detrended short-wave normal
amplitudes (wavelength≤200 m) are 194/250 Pa raw and 40.5/52.4 Pa filtered in
creep, with peak wavelength≈73.4 m. V short-wave amplitudes are 5.34e-14/2.90e-14
m/s. These are roughness measures including genuine curvature and endpoint effects,
not independent error estimates; the minimum resolvable surface wavelength is 39.82 m.
Three normal cuts across 12 sampled grading interfaces show pressure-continuous
traces (≤2.28e-13 Pa jumps), with max tensor/normal jumps 4.429/.651 Pa for
FE/particle creep, 3.770/.701 Pa for perturbed. These cuts sample only three stations.

The specific next refinement question is whether the prominent lower-end raw
normal roughness and pressure/tensor cancellation converge under a targeted common
spatial/quadrature refinement (7C), before interpreting an MPI comparison. The
7B recurrence/publication checks pass and need no new physical assumption. Frozen
straight geometry, homogeneous coefficients, fresh fixed Eulerian sampling and
unchanged population/mesh/profile remain the supported FE lifecycle. FE resume,
checkpoint conversion, transport/birth and changing geometry remain restricted.
No 7C/7D, unfiltered test, varying-friction transition or full-model preparation was run.

## Changed files, preserved failures and reproducibility

This selection adds `coupled_checks.cc`, `analyze_section7b.py`, `plot_section7b.py`,
`section7b-common.prm` and six gate/trajectory leaves, `SECTION7B.md` and this report;
it extends only the optional diagnostic target in `CMakeLists.txt`. Updates to
`STATUS.md`, `CURRENT_STATUS.md`, `current_design.md` and `specification.tex` record
the selected/completed scope. Incoming production C++/headers and maintained plugin
sources are byte-preserved; ASPECT's astyle2.04 formatting is applied only to the new C++.
No commit, reset, merge or push was made; the reference worktree is read-only.

Build commands: configure the maintained plugin in
`build-bp3-native/section7b/bp3-plugin` with `Aspect_DIR=build-bp3-native`, Release,
`CMAKE_CXX_FLAGS=-DASPECT_BP3_LOCAL_BOTTOM_TEST`, then build with `-j2`. Configure
`bp3_native` in `build-bp3-native/history-plugin` with `BP3_COUPLED_PLUGIN` pointing
to that exact maintained library; build target `bp3_coupled_checks -j2`. The observer
links it explicitly because ASPECT loads plugins locally. Existing first-pass targets
and their evidence were not rerun.

One CMake link-signature attempt failed before compilation. Four FE launches were
preserved: missing observer library dependency (exit127), postprocessor acceptance
ordering (exit1), and two overstrict observer bounds formed from cancelled net
strain (exit1). Fixes are confined to test linkage/input order/diagnostics. The final
bound retains the original 128 epsilon multiplier and uses coefficient/basis operands,
as the production check does. No solver/physical tolerance, dt or boundary tuning
occurred. The passing FE gate observer preceded the informational bulk-cut addition;
all four trajectories used the identical final observer above. A NumPy `trapz`
compatibility failure was corrected using the explicit Q1 integral; analysis passes.

Exact launch commands/exits/times remain in
`benchmarks/reconstructed_fault/refactoring_boundary/evidence/bp3-native-7b-*.json`
and matching logs. The pattern is
`python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py LABEL CAP env ASPECT_SOURCE_DIR=PWD build-bp3-native/aspect INPUT`.
Fresh outputs are `build-bp3-native/runs/7b-{fe,particle}-{gate,creep,perturbed}`
(passing FE gate suffix `-v5`); earlier gate failures remain separate.
Each trajectory stores `coupled-profile-{0..10}.csv`, `bulk-normal-cuts-{0..10}.csv`,
`coupled-checks.csv`, `accepted_steps.csv` and compact initial identity files.

[All common-time differences](../../../build-bp3-native/section7b/common-time-differences.csv),
[metrics, runtime, oscillations and retained failures](../../../build-bp3-native/section7b/comparison.json),
[initial/final profiles](../../../build-bp3-native/section7b/profiles.pdf),
[artifact/input provenance](../../../build-bp3-native/section7b/provenance.json) and
[resolved pair differences](../../../build-bp3-native/section7b/resolved-pair-diff.json)
are reproducible with the two new analysis scripts. Incoming snapshots and
[final preservation checks](../../../build-bp3-native/section7b/final-verification.json)
are in the same `section7b` evidence directory: all 2346 incoming untracked files
remain, the 13 incoming production C++/header files and both main binaries are
unchanged, reference tracked files remain clean, and `git diff --check` passes.
