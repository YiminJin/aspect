# Bounded K1 support-resolution study and corrected mechanical pilot

**Recommendation for review:** prefer a fixture-specific containment error budget
of **1e-4 for this homogeneous, fixed-profile K1**, keeping the full I_h reference
and unchanged initialization/support policy. Do not apply it automatically to K2
or evolving profiles. The current **1e-6 gate remains failed**: this is a proposal,
not an edited threshold or a K1 pass. The first real mechanical timestep also
failed its line search, independently blocking completion.

No production file, history rule, solver setting, fixture physics or acceptance
threshold was changed in this task. The pilot overlay changes only its output
directory. The existing 600 s wall-clock limit was retained.

## 1. Width required by the saved corrected FE profile

The scan independently integrates h(phi_h) at fixed Gauss order 32, split at
every Q1 knot and the trial strip endpoints. It checks every saved native x
column, taking the worst omitted fraction, not just a favorable center column.
The preceding gate audit cross-checked orders 16/32 and adaptive quadrature.
Here I_h means the full saved-domain integral, approximately 108.072238204 m;
no field outside the bulk domain is extrapolated.

| Symmetric half-width (m) | Omitted fraction |
| --- | ---: |
| Current 0.308821594 | 5.64918e-5 |
| 0.3125 | 4.35997e-5 |
| 0.328125 | 1.37695e-5 |
| 0.34375 | 4.23236e-6 |
| 0.359375 | 1.28850e-6 |
| **0.362931072** | approximately **1e-6** |
| **0.375** | **3.91071e-7** |
| 0.4 | 5.83950e-8 |

The estimated minimum extension is **0.054109478 m on each side**, or 17.52%
more band width. Its last digits locate a numerical threshold, not a safety
margin. A practical mesh-face-aligned alternative is 0.375 m, an extension of
0.066178406 m per side with 21.43% more width. The actual production query in
the new accepted-state export gives the unchanged half-width
0.3088215939070757 m, agreeing with the independent support calculation.

### A shared-support extension changes more than the slip tail

The same manager support selects particles for initial cohesive projection.
Keeping the saved particle H and FE phi unchanged, the volume-weighted
transverse projection predicts:

| Half-width | Associated initial particles | Initialized C (Pa) |
| --- | ---: | ---: |
| Current | 5664 | 313.511760 |
| Minimum meeting 1e-6 | 6720 | 413.137692 |
| 0.375 m | 6912 | 429.368748 |

For this x-uniform regular fixture, the consistent Q1 surface projection is
constant and equals the transverse weighted mean. The current value reproduces
the independent full projection/production result. Values for extended strips
are **offline predictions**, not a production rerun with different support.

Thus a shared-support extension would alter C0 by **31.78%–36.95%**. It is not a
small correction to the same initialized mechanical trajectory. Extending only
QP support would instead separate the currently shared support policy; that is
an architectural/numerical decision requiring review, not a benchmark workaround.

## 2. Separate retained-fraction scalar diagnostic

Let r=I_strip/I_full=**0.9999435082451503**. Retain the approved scalar reference
unchanged. In a separately named diagnostic only, use

    q_k = beta*q_(k-1) + kappa*(U_k - r*V_k)/W,
    C_k = beta*C_(k-1) + kappa*V_k/I_full,
    q_k - C_k - sigma_* mu(V_k,Theta_(k-1)) - eta_d V_k = 0.

W=1 m here. Only the integrated bulk slip term changes; the cohesive denominator
remains the **full** I_h. Mechanics freezes old Theta; afterward each path applies
the same exact constant-accepted-V aging update. Both paths initialize once from
retained tau_xy=1500 Pa, C0≈313.511760147 Pa, Theta0≈200 s. At timestep zero,
evaluated q/C are returned separately; neither path evolves or replaces the
retained histories. No later ASPECT state resets either trajectory.

Two outputs are explicitly separate:

- `support-resolution/`: **prospective nominal** sequence t=0,2,4,6 s,
  U=1e-4,1.1e-4,1.2e-4,1.2e-4 m/s. Only the t=0 entry is backed by an accepted
  production state; later entries are predictions.
- `support-resolution-accepted/`: actual accepted sequence, currently **t=0 only**.

All diagnostic/reference roots are interior; numerical traction residual checks
remain <=1e-7 Pa. Over the prospective nominal sequence:

| Quantity | Maximum diagnostic minus approved-reference magnitude | Maximum relative magnitude |
| --- | ---: | ---: |
| V | 1.67806e-8 m/s | 5.97589e-5 |
| Evaluated q | 0.00327127 Pa | 3.13927e-6 |
| Retained C | 0.000564668 Pa | 1.84784e-6 |
| Theta | 0.00405351 s | 6.06388e-5 |
| Accumulated real-step slip | 6.32757e-8 m | 5.46186e-5 |

At t=6 s, approved/diagnostic accumulated slip is
0.00115850039638 / 0.00115856367208 m and Theta is
66.84690532 / 66.84285181 s. These are **not production observations**.
The local Theta-update assertion tolerance (2e-8 s) is not relaxed: this diagnostic
compares different slip trajectories, not accuracy of the update at identical V.

## 3. Corrected bounded production pilot

One rank, unchanged 16x64 mesh, 9216 particles, ell=0.15625 m, nine fault
vertices, maximum dt=2 s, original loading and solver tolerances. Executable
SHA256 `5022898bddbc9762595547433056e3d09e8fe06bb9bbb812038d80b3ceca33cd` is
unchanged from the ownership correction. The benchmark plugin was rebuilt with
`-j4`; the main executable did not need rebuilding.

**Observed:** initial phase convergence and accepted t=0 mechanics; **failure**
on the first Newton iteration of the first real step at t=2 s. The line search
exhausted its candidates (configured maximum five reductions). This is not a
600 s timeout or a 30-Newton-iteration exhaustion. The outer generic exception
suggests increasing an iteration budget, but that does not diagnose this failure.
Trial merit/scales were not exported, so the deeper cause is **not established**.
No parameter, direction, tolerance or state was patched to obtain acceptance.

Measured runtime **149.2506 s**, peak RSS **479792 KiB** (468.55 MiB), process
exit **1**. At t=0, eight Newton updates reached relative bulk/fault residuals
7.756e-15 / 5.588e-11, with no rejected candidates. At t=2, the first reported
relative residuals were 1 / 0.1545; no real-step candidate was committed/exported.
Rollback code was not independently re-instrumented in this task, so absence of
a positive-step export is not promoted to a new rollback verification result.

### Accepted initial state versus approved reference

| Quantity | Production | Independent full-I_h reference |
| --- | ---: | ---: |
| Mean V (m/s) | 3.17317926083e-4 | 3.17173797010e-4 |
| Evaluated mean q (Pa) | 1040.016567 | 1040.265080 |
| Retained C (Pa) | 313.511760147 | 313.511760147 |
| Retained Theta (s) | approximately 200 | approximately 200 |
| Retained particle tau_xy (Pa) | approximately 1500 | 1500 |
| Real-step cumulative slip (m) | 0 | 0 |

Initial V mean error is **0.04544%**, evaluated q mean error **-0.02389%**.
The retained-fraction scalar predicts V=3.17190093192e-4 m/s: omission explains
only about 11.3% of the observed V difference, not the entire numerical error.
Approved-reference velocity-profile errors are **3.44747e-8 m/s RMS** and
**5.52984e-8 m/s maximum**, evaluated at the actual bulk quadrature points.

The evaluated FE shear stress varies **normally** from 1032.5901 to 1049.2996 Pa;
its maximum along-x range at fixed y is only 5.18e-10 Pa. Do not confuse the
mean comparison with pointwise exact uniform stress: these normal oscillations
still require resolution evidence. V along-fault range/1e-4 is 1.6263e-12;
transverse velocity RMS is 1.2970e-18 m/s, divergence RMS 2.1088e-18 s^-1,
and prescribed-velocity maximum error 2.0329e-20 m/s.

### Actual normalization, not the scalar proxy

Using production QP weights, chi, V and history correction, and the surface
Q1 integral of accepted V:

    integral_bulk(chi*V + history) = 7.93248250018e-5 m²/s,
    integral_surface(V)          = 7.93294815207e-5 m²/s,
    ratio                       = 0.9999413015338395.

The normalization deficit is **5.86985e-5**; the equivalent per-unit-length
slip-rate deficit is **1.86261e-8 m/s**. History-correction magnitude is at most
1.76e-17 s^-1. This passes the existing **1e-4 normalization check** but does
not replace the separate failed **1e-6 containment check**. The discrepancy
between actual normalization and continuous retained fraction is about
2.21e-6; bulk quadrature/discrete-path effects are not removed by the scalar
diagnostic. Moving periodic particle-domain behavior and all real-step history
comparisons remain unverified because the first real solve failed.

## 4. Recommendation and bounded next steps

Prefer reviewing a **K1-only containment allowance of 1e-4**, with no change to
the approved full-I_h reference. Reasons:

- Observed omitted fraction and actual initial normalization deficit are below
  1e-4, and the predicted V/Theta relative bias is <=6.1e-5 for this short case.
  This is about 3.1% of the existing 0.2% resolved relative-error allowance,
  leaving most of that allowance for spatial/temporal/algebraic errors.
- It preserves the accepted initialization contract and C0. A shared support
  extension changes C0 by more than 30%, despite correcting a roughly 6e-5
  integrated-slip omission.
- It has no production runtime/memory cost. An extension to 0.375 m adds 1248
  associated particles (22.03%) and roughly 21.43% continuous band work, without
  reducing unknowns. That is not a reliable whole-run time multiplier: changed
  C0 and Newton convergence can dominate. Applying that factor to the entire
  observed 149 s would give about 181 s, only a rough work-scaling illustration.

This is an error-budget proposal, **not permission to reinterpret a failed
gate as passed**. The predictive transient is not production-verified; normal
stress oscillations, spatial/time convergence and the real-step failure remain.
Do not transfer this allowance to K2, true normal stress, evolving profiles or
different loading without a new error assessment.

Bounded sequence for review, not executed automatically:

1. **Diagnose the real-step line search:** one replay of the same coarse case,
   cap 180 s, recording alpha, trial bulk/surface norms and fixed normalizers,
   current/trial merit and active-set status. Stop after the first failed search.
   If needed, verify the directional Jacobian on that saved linearization before
   proposing any fix. Do not simply increase the line-search budget.
2. **Check resolution sensitivity of the proposed containment budget:** one
   initialization-only uniform 2x mesh refinement at fixed ell, H prescription,
   physical domain and fault settings, cap 300 s. Reuse the same width/integral
   audit. Stop if the profile cost or omitted fraction invalidates the proposed
   budget; this is one check, not the K1 convergence matrix.
3. **Only after the failure is understood and any required correction approved:**
   repeat the unchanged coarse 0–6 s pilot under the existing 600 s cap. Use its
   actual accepted sequence for both independent paths and verify histories,
   normalization and fixed fields. No reference-history resets.

These proposed runs have a combined wall-clock cap of 1080 s, excluding builds;
they are not a request to launch the full campaign. A refined-memory estimate
is uncertain; four times the coarse 469 MiB is roughly 1.9 GiB as a planning
allowance, not a measured requirement. Gate K1 still needs its separately
reviewed spatial/time/MPI/restart evidence afterward. If review insists on the
original 1e-6 containment target, 0.375 m is the bounded shared-support candidate,
but the C0 change must be explicitly accepted and the new initial state re-audited.

## 5. Reproduction and artifacts

All new data are persistent under
`benchmarks/reconstructed_fault/uniform_shear/diagnostics/`.

```sh
cmake -S benchmarks/reconstructed_fault/uniform_shear \
  -B benchmarks/reconstructed_fault/uniform_shear/diagnostics/pilot-build \
  -DAspect_DIR=/home/ein/repository/aspect/build-pf-cpdi -DCMAKE_BUILD_TYPE=Debug
cmake --build benchmarks/reconstructed_fault/uniform_shear/diagnostics/pilot-build -j4
cd benchmarks/reconstructed_fault/uniform_shear/diagnostics/pilot-build
python3 ../../run_pilot.py /home/ein/repository/aspect/build-pf-cpdi/aspect ../corrected_pilot.prm
cd /home/ein/repository/aspect
python3 benchmarks/reconstructed_fault/uniform_shear/analyze.py \
  benchmarks/reconstructed_fault/uniform_shear/diagnostics/mechanical-pilot \
  > benchmarks/reconstructed_fault/uniform_shear/diagnostics/corrected-pilot-analysis.json
python3 benchmarks/reconstructed_fault/uniform_shear/support_resolution.py
python3 benchmarks/reconstructed_fault/uniform_shear/support_resolution.py \
  --accepted benchmarks/reconstructed_fault/uniform_shear/diagnostics/mechanical-pilot \
  --output benchmarks/reconstructed_fault/uniform_shear/diagnostics/support-resolution-accepted
MPLCONFIGDIR=/tmp/k1-matplotlib python3 benchmarks/reconstructed_fault/uniform_shear/plot_corrected_pilot.py
python3 -m unittest discover -s benchmarks/reconstructed_fault/uniform_shear -p 'test_*.py' -v
```

Preserve existing output before rerunning. Production pilot exits 1; the
independent analyzer exits 2 for incomplete time coverage/failed containment.
The seven Python tests pass, including three new diagnostic tests for unit
retention equivalence, full-I_h cohesive law, and initial/real history semantics.
`git diff --check` passes. No complete ASPECT suite or new MPI run was performed.

Key artifacts:

- `corrected-pilot.log`, `corrected-pilot-analysis.json`, `mechanical-pilot/`:
  failure/convergence log, accepted initial comparison and raw accepted CSVs.
- `support-resolution/width_scan.csv`, `summary.json`,
  `retained_fraction_trajectory.csv`: width/cost/C0 scan and **prospective**
  scalar diagnostic.
- `support-resolution-accepted/`: diagnostic restricted to **actual accepted t=0**.
- `mechanical-pilot-view/accepted_t0_bulk_qp.vtu`: actual bulk quadrature-point
  values, not reconstructed native FE patches; render as points in ParaView.
- `mechanical-pilot-view/accepted_t0_fault.vtu`: actual fault with accepted V and
  retained Theta/C/I_h; `accepted_t0_particles.vtu` stores retained particle H/stress.
- `mechanical-pilot-view/accepted_t0_transverse.csv`, `summary.json`,
  `initial_mechanics_and_width.png`: accepted velocity/stress reference profiles
  and normalization. No t=2 accepted field is fabricated.

Files added: `support_resolution.py`, `test_support_resolution.py`,
`plot_corrected_pilot.py`, the output-only `diagnostics/corrected_pilot.prm`, this
report and generated diagnostics. The diagnostic ignore file, README and progress
links are updated. Existing production and unrelated worktree changes are preserved;
nothing was committed. K1 remains unmet and K2 remains unstarted.
