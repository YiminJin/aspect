# K5: single frozen-history 35-km junction diagnostic

## Decision summary

The four-rank diagnostic mechanical solve converged in 79.02 s with unchanged
tolerances. All seven fresh-linear checks passed. Its converged stress data
were exported before history publication. **The new rollback verifier then
failed in a Trilinos bulk-vector subtraction**, so its promised in-memory
equality checks remain incomplete. Both checkpoint copies are unchanged; no
accepted timestep or new history output was written. No second solve was run.

The dominant non-bottom concentration **does not move to 35 km**. A new,
smaller dipole appears there, but about 80% of the raw 40-km stress range and
81% of its projected variation remain. The bottom-tip minimum is essentially
unchanged. This fails the proposed simple “dominant dipole follows the current
junction” criterion.

The current junction affects the stress, but it is not the only dominant
contribution. Frozen Maxwell-history loading is the leading remaining
mechanism; the test does not establish a transfer bug or rule out stress
previously accumulated because of the original junction. Next, separate the
retained particle stress from its actual FE-transferred weak load using frozen
data, without another altered-cutoff solve or a physical-model change.

## 1. What was run, and what was not committed

Input checkpoint:
`benchmarks/reconstructed_fault/bp3/normal-stress-complete-local4/restart/03/`.
The completed baseline log establishes that this is the checkpoint saved
after step 11, before step 12. Its last-good marker now points to slot 01
because the baseline subsequently completed step 12; that does not change
the contents of slot 03. The diagnostic copied slot 03 into a separate output
directory and selected it explicitly. All source and copied checkpoint hashes
were checked before and after execution.

The only model-input change in the mechanical probe was to prescribe Vp on
the additional existing surface nodes from 35 to 40 km. Physical boundaries,
pressure treatment, background tractions, friction and material parameters,
phase field, support, quadrature, solver settings, and time/history inputs
were otherwise inherited unchanged. The original BP3 constant Wf is unchanged.
The benchmark-only switch is effective only for a noncommitting step-12 replay.

The run reproduced the baseline's step-12 time **2,232,176,379.2516127 s**
from the same checkpoint and timestep controller (dt **376,359,254.49064183 s**).
Surface mass-matrix entries agree **exactly** with the baseline. The largest
relative difference in bulk-QP integrated chi is **4.19e-16**. Thus the geometry,
measure and sampled localization comparison provides no indication of a
changed support or I_h path.

Execution directory:
`benchmarks/reconstructed_fault/bp3/normal-stress-junction35-local4/`.
Runner: `benchmarks/reconstructed_fault/bp3/run_junction_diagnostic.py`.
`provenance.json` hashes the actual compiled source/binary/plugin used.

### Mechanical verification

| Quantity | Result |
|---|---:|
| Newton updates | 6 |
| Returned linear directions | 7 |
| Total Krylov iterations | 162 |
| Largest fresh-linear residual / requested target | 0.841992195 |
| Final absolute bulk residual | 6.520747744e-4 |
| Fixed bulk scale | 6.356584567e9 |
| Final normalized bulk residual | 1.025627013e-13 |
| Final surface residual RMS | 2.717249467e-4 Pa |
| Fixed surface scale | 2.612008497e6 Pa |
| Final normalized surface residual | 1.040291205e-10 |
| Final free / lower-active RSF nodes | 350 / 0 |

Both normalized residuals satisfy the unchanged 1e-8 criterion. The generic
diagnostic branch exports the converged weak data and throws before the
history-commit call. It never invokes the accepted BP3 postprocessor and
never advances to another step.

### Rollback-verification limitation — not a mechanical failure

The ordinary solver catch path rolls back current V, restores the production
bulk and linearization vectors, and restores assembly/constraint controls
before emitting the failure signal. The benchmark verifier attached to that
signal attempted to subtract two Trilinos vectors and received error −2 in
`TrilinosWrappers::MPI::Vector::operator-=`. This secondary exception masks
the intended diagnostic-stop exception. Process status is therefore **1**,
not a successful ordinary simulation termination.

The in-memory particle/surface/geometry comparisons following that subtraction
were **not executed**. `noncommitting_history.csv` was not produced. The
report does not claim a verified bitwise rollback or provide fabricated
Theta/history values in its place. The core convergence branch precedes every
history-publication call, and no `accepted_steps.csv`, accepted step 12,
updated checkpoint, or ordinary graphical postprocessing was produced by this
probe. Original graphical output was not modified.

The verifier has been repaired to compare represented entries over locally
owned global DoF indices instead of applying vector algebra to differently
laid-out copies. The repair **compiles but has not been exercised in another
run**. The performed run remains archived unchanged. Offline analysis requires
the explicit `--allow-unverified-rollback` flag and records
`rollback_verified: false`; this flag does not bypass any mechanical residual
or traction consistency check.

## 2. The strongest dipole remains near 40 km

The following are actual parent-P0/domain-Q1 constitutive samples, not
normal-column averages or post-commit Maxwell reevaluations. The table gives
the strongest retained samples in each region. Selection retains rank-local
extrema per support class, not every pressure sample.

| Region | Accepted 40-km baseline sigma_n range (MPa) | 35-km probe sigma_n range (MPa) |
|---|---:|---:|
| Old junction near 40 km | −0.659852 to 100.176830 | **9.672915 to 89.835148** |
| New junction near 35 km | no comparable dominant raw extrema retained | **40.938090 to 59.441540** |
| Bottom-tip global tensile minimum | −20.406352 | **−20.399874** |

The old-junction raw range contracts from 100.836682 to **80.162233 MPa**,
leaving approximately **79.5%**. The new-junction range is only 18.503450 MPa.
Thus the raw extrema do not support classifying the instantaneous relocated
constraint as the sole dominant source of the existing 40-km concentration.

Compression-positive decomposition, including the fixed 50 MPa background:

| Probe sample | xd (km) | Delta p (MPa) | −Delta tau:N (MPa) | sigma_n (MPa) |
|---|---:|---:|---:|---:|
| Old junction, low | 39.961334 | −34.243044 | −6.084041 | **9.672915** |
| Old junction, high | 39.998116 | +35.130034 | +4.705115 | **89.835148** |
| New junction, low | 34.960291 | −10.499438 | +1.437528 | **40.938090** |
| New junction, high | 34.967542 | +10.933646 | −1.492106 | **59.441540** |
| Bottom minimum | 115.469484 | −21.061019 | −49.338855 | **−20.399874** |

The old-junction low sample is the same parent **217673**, cell
`0_10:3021033300`, as the baseline tensile pocket. Its current normal stress
increases by 10.332767 MPa after the mask change. The opposite side decreases
by about 10.34 MPa. The intervention therefore has a substantial immediate
effect, but a much larger residual dipole remains.

At the bottom the minimum is still parent **11936**, cell
`0_10:0011010111`, at exactly the same parent position
(21012.855768411, 81.850114902) m. Different quadrature labels can tie for the
minimum because normal traction is parent-P0 for this straight geometry.
The minimum changes by only **0.006479 MPa** (about 0.032%). Bottom projected
sigma peak-to-peak variation changes by only about 7 Pa out of 30.13 MPa.
This separates the bottom-tip problem from the junction intervention.

## 3. Consistent projected profiles also retain the old feature

Peak-to-peak Q1 variation, in MPa, over fixed 2-km windows:

| Window / quantity | 40-km baseline | 35-km probe |
|---|---:|---:|
| 34–36 km: Delta p | 0.039505 | 0.069779 |
| 34–36 km: −Delta tau:N | 0.036667 | 0.042795 |
| 34–36 km: sigma_n | 0.007310 | **0.043337** |
| 34–36 km: total shear q | 0.332296 | **2.048011** |
| 39–41 km: Delta p | 0.319999 | **0.250699** |
| 39–41 km: −Delta tau:N | 0.124822 | **0.111540** |
| 39–41 km: sigma_n | 0.444821 | **0.362238** |
| 39–41 km: total shear q | 10.012363 | **8.085655** |

The old-junction projected normal-stress variation retains **81.4%** of its
baseline magnitude. It is still substantially larger than the new-junction
variation. See `comparison.csv`, `comparison.json`, and `comparison.png`.
The old feature is not merely hidden by a positive mean projection: it
persists in both the raw constitutive extrema and the complete projected
profile.

In the probe, V=Vp on **both sides of 40 km**. Bulk-QP localization-weighted
rates in the 39.8–39.9, 39.9–40.0 and 40.0–40.09996 km intervals all equal
Vp to roundoff. The new mixed element is 34.9–35.0 km; its weighted rate is
5.91767169e-10 m/s, with nodal V(34.9 km)=2.13948118e-10 and V(35 km)=Vp.
Thus failure to remove the old current-slip gradient is not the explanation.

All probe tensile weight is now on prescribed bottom-tip support:
**5297.872946 m²**, identical to the baseline's bottom-only tensile weight.
There is zero tensile weight on unprescribed or mixed-junction rows. The
baseline's additional 1059.643059 m² junction tensile pocket disappears.
This removes that pocket in this particular diagnostic; it is not approval
to change the BP3 prescribed-slip extent.

## 4. Interpretation: distinguish current loading from inherited stress

The two solves share frozen histories. Moving a kinematic boundary for one
solve does not remove the stress accumulated under the old boundary. With
the configured eta=1e26 Pa s, G=32038120320 Pa and this accepted dt,

\[
\Delta t G/\eta=1.205784308\times10^{-7},\qquad
\beta=0.9999998794215765.
\]

Essentially all retained Maxwell stress is carried into this mechanical
evaluation. For fixed phase/I_h the profile-history crack correction is zero;
the normal contraction of the direct slip tensor vanishes (S:N=0). The
normal traction nevertheless responds through bulk velocity/pressure and
the inherited stress load:

\[
\sigma_n=50\ {\rm MPa}+p-
 [2\kappa\dot\epsilon(u)+\beta\tau_{11}]:N.
\]

Here the **surface** samples use retained parent-particle tau_11, whereas
the **bulk weak load** uses its actual constrained FE transfer. Those two
representations must not be interchanged during diagnosis. The ordinary
post-solve step-11 FE stress-composition visualization is an old-history
input, not automatically the particle tau_11 transferred for step 12.

The sustained 40-km concentration with uniform current V there points toward
the inherited stress/loading distribution (or its representation), not solely
the current kinematic-gradient term. It is compatible with a historical
concentration originally generated by the hard junction. The result does
**not** prove that the historical junction was irrelevant, nor that the
particle-to-FE transfer is wrong. No history was zeroed or reset to force the
dipole to move.

## 5. Next decision and verification status

The stipulated dominant-dipole relocation criterion is **not satisfied**.
Do not implement a smoothed junction, change normal-stress treatment, extend
RSF to the bottom, or resume the event trajectory on the basis of this probe.

The smallest useful next diagnostic is a **frozen-data load decomposition**
around 40 km, with a 35-km/interior control: compare retained particle
tau_11 with the actual constrained FE history consumed by bulk assembly,
then quantify their normal-stress contractions and the FE history weak load.
This can be an extraction/assembly audit; it does not require another
altered-cutoff mechanical solve. It would distinguish an inherited physical
stress concentration from amplification introduced by its transfer or
sampling. Keep the bottom-tip boundary question separate.

Changes in this task:

- `source/simulator/solver.cc`: opt-in converged weak-data export and forced
  pre-publication rollback; normal runs do not take this branch.
- `bp3.cc` and new `junction_diagnostic.h`: benchmark-only 35-km mask and
  frozen-state rollback verifier. The revised owned-entry verifier compiles
  but is **not runtime-verified** after the observed checking failure.
- `run_junction_diagnostic.py`: copied-checkpoint, one-run provenance wrapper.
- `analyze_normal_stress.py`, `compare_junction_diagnostic.py`: explicitly
  noncommitting/unverified-rollback analysis; no fabricated accepted output.

Release core and plugin builds used `-j4` and passed after correcting a
diagnostic output-path compile error. The repaired checker plugin also builds.
The normal accepted-state analyzer was regression-checked on the saved step-12
baseline and passed unchanged numerical assertions. The diagnostic's weak
normal-stress projection and bulk transfer consistency checks pass. No second
mechanical solve, broad tests, model correction or history publication was
performed. `git diff --check` passes. The rollback-checker defect remains an
explicit verification gap, not silently qualified evidence.
