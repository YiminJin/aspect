# K5 BP3 normal-stress diagnosis: consolidated report

This report consolidates the last three tasks: the fresh four-rank baseline
through step 12, the single noncommitting 35-km junction probe, and the frozen
step-12 history/load audit. It supersedes their intermediate recommendations,
not their evidence. Original reports and raw outputs remain preserved.

Subsequent approved work is reported in
[the frozen mechanics comparison and single FE-history solve](stage_K5_history_mechanics_result.md).
It establishes a 20% frozen bound-margin change but unchanged contact after
re-equilibration; it completes the weak-moment follow-up proposed below.

## Decision summary

The corrected Theta audit and the baseline mechanical solves pass. Two
distinct stress concentrations exist: a prescribed bottom-tip tensile minimum
of **−20.406 MPa**, and a pressure-dominated 40-km junction dipole with a
**−0.660 MPa** mixed-support pocket reaching the lower-active 39.9-km node.

Moving the current prescribed-slip junction to 35 km in one frozen-history
solve leaves about **80%** of the old 40-km feature. A smaller feature appears
at 35 km; the bottom minimum barely changes. The probe converged, but its
new in-memory rollback checker failed and remains incompletely verified.

The subsequent pre-Newton audit establishes that the 40-km concentration
already exists in retained particle stress. Its FE transfer reduces the
local normal-history RMS by **2.9%**; independent integration reproduces the
production frozen weak load to roundoff. Nevertheless, local particle/FE
differences materially affect the tensile sign, particularly at the bottom.

No production correction is justified by these results alone. Inherited
stress concentration is established; its convergence and the mechanical
significance of the representation mismatch remain unresolved. No long-run
continuation, smoothing, pressure change, or physical-model correction was
performed.

## 1. Common formulation and interpretation rules

The accepted stress-perturbation BP3 configuration is unchanged: 100×100 km
box, 60° fault, ell=400 m, frozen phase and fault geometry, full I_h and
unchanged support, fixed background tractions, true normal-stress feedback,
official lateral velocities, zero top/bottom perturbation traction, and deep
Vp=1e-9 m/s. Vmin remains 1e-20 m/s. The mesh has 36,106 cells and 1,282,726
DoFs. Runs used four MPI ranks, one thread per rank, sparse B/G and the
qualified pivoted tridiagonal surface inverse. Physics, integration and
solver tolerances were not retuned.

Compression is positive. The constitutive surface residual and normal stress
are

\[
 R_\Gamma=q-C-\mu(V,\Theta)\sigma_n-\eta^d V,
 \qquad \sigma_n=50\ {\rm MPa}+\Delta p-\Delta\tau:N.
\]

Bulk/history inputs to surface mechanics are parent-P0; surface fields and
test functions use domain-integrated Q1 quadrature. Consequently a surface
quadrature coordinate is not necessarily the parent location where bulk
fields were evaluated. Raw extrema, consistent Q1 projections and
normal-column averages are different diagnostics and are not interchangeable.

Mechanics at step 12 uses retained particle tau11 and Theta11. Bulk assembly
uses the **constrained FE transfer of tau11**; surface evaluation uses retained
parent-particle tau11. Ordinary `bulk_11` stress visualization is an older
FE input, not automatically the particle tau11 transferred for step 12.
No diagnostic recomputed a Maxwell update from already committed history.

For this fixed-profile step, the history-localization correction is zero and
S:N=0. Thus

\[
 \sigma_n=50\ {\rm MPa}+p-[2\kappa\dot\epsilon(u)+\beta\tau_{11,p}]:N,
 \qquad \beta=0.9999998794215765.
\]

Essentially all old Maxwell stress survives this step. Moving a current
kinematic boundary does not erase stress accumulated under the old one.
These conventions follow `current_design.md` and `specification.tex`; this
report makes no change to either authority.

## 2. Execution and verification ledger

| Task | Execution | Result | Publication / verification limit |
|---|---|---|---|
| Fresh baseline | Initialization + steps 1–12; 907.34 s | Exit 0; accepted-state and residual checks pass | Ordinary accepted histories and output |
| 35-km probe | One step-12 frozen-history mechanical solve; 79.02 s | Mechanical convergence; seven fresh-linear checks pass | Exit 1 after rollback-checker error; no accepted output or history publication; full in-memory equality not verified |
| Frozen history/load audit | Step-12 preparation and extraction; 26.40 s | Production/independent load checks pass | Intentional exit 1 before Newton; no mechanical solve or accepted output |

Both diagnostics copied the compatible **local step-11 `restart/03`** from
the completed baseline. Neither used the failed server in-memory step 12 or
the incompatible server checkpoint. Source and copied checkpoint hashes
remain unchanged. There was no automatic retry or solve beyond the bounded
tasks. Exit zero alone was never treated as proof of convergence.

### Baseline convergence and Theta

| Quantity | Step 11 | Step 12 |
|---|---:|---:|
| Physical time (s) | 1,855,817,124.760971 | 2,232,176,379.251613 |
| dt (s) | 884,306,322.054474 | 376,359,254.490642 |
| Newton updates / Krylov iterations | 6 / 198 | 5 / 268 |
| Minimum accepted alpha | 1 | 0.544299719945 |
| Free / lower-active RSF nodes | 400 / 0 | 399 / 1 |
| Absolute bulk residual | 0.0015190941 | 0.0006363143 |
| Fixed bulk scale | 8.44784146e9 | 3.80348150e9 |
| Surface RMS residual (Pa) | 9.64672e-8 | 2.61255e-8 |
| Fixed surface scale (Pa) | 6.57931995e6 | 3.14348262e6 |

All 13 accepted states satisfy the unchanged criteria. All **105** fresh
linear checks pass; the largest fresh/target ratio is **0.997666455**.
Total recorded Krylov iterations are 2,140.

The benchmark-only Theta reference uses the cancellation-safe expression

\[
 x=V_k\Delta t/D_c,\qquad
 \Theta_k^{ref}=\Theta_{k-1}e^{-x}-(D_c/V_k)\operatorname{expm1}(-x).
\]

Production aging and split-history semantics were not changed. Step-12
relative error is **2.22044605e-16** against the retained 1e-12 assertion;
an independent 50-digit calculation gives **2.09843747e-16**. Captured
near-bound rates, deliberately incorrect histories and zero-interval controls
pass. Original server records through step 11 agree within approximately
5.1e-12 relative for the reported time/rate/stress quantities.

## 3. Baseline: distinguish the junction from the bottom tip

These are actual constitutive samples, not projected normal stresses.

| Sample | Surface xd (km) | Delta p (MPa) | −Delta tau:N (MPa) | sigma_n (MPa) | Support |
|---|---:|---:|---:|---:|---|
| Step 11 global minimum | 115.470054 | −21.379684 | −40.855155 | **−12.234839** | prescribed bottom |
| Step 12 global minimum | 115.470054 | −21.065329 | −49.341024 | **−20.406352** | prescribed bottom |
| Step 11 junction minimum | 39.993648 | −39.670972 | −1.401657 | **8.927371** | mixed |
| Step 12 junction minimum | 39.961334 | −43.096021 | −7.563831 | **−0.659852** | mixed |
| Step 11 global maximum | 39.925530 | 37.231415 | 3.612528 | **90.843944** | mixed |
| Step 12 global maximum | 39.983708 | 43.971442 | 6.205388 | **100.176830** | mixed |

The bottom minimum is parent **11936**, rank 0, cell
`0_10:0011010111`, fault 0, segment 0. At step 12 its parent position is
(21012.855768411, 81.850114902) m, normal offset +144.528345 m; the selected
surface point is the bottom tip (21132.486540519, 0) m. V=Vp exactly.
Its domain measure is 1059.831018 m², phi=0.504199995,
chi=0.00278809649 1/m, and q=−54.134343 MPa. Its surface equation is replaced
by a prescribed-V condition, not solved as an RSF traction equation.

The junction minimum is parent **217673**, rank 2, cell
`0_10:3021033300`, segment **755**, xi=0.386655821. Its parent position is
(59065.290703567, 65315.107048699) m, offset −193.218518 m; the surface point
is (58886.846250536, 65392.469224981) m. Its domain measure is
1059.643059 m², V=6.133441789e-10 m/s, phi=0.449358031,
chi=0.00101660863 1/m, and q=27.997901 MPa. The positive counterpart lies
on the opposite side of the band. Multiple quadrature points can share a
parent-P0 stress extremum; these are not claims of unique extremal points.

### Tensile integration weight and affected equations

All weights below sum **all** raw samples with their positive Q1 test weights,
not only saved extrema. Units are m² of admitted domain measure.

| Test support | Step 11 | Step 12 |
|---|---:|---:|
| Prescribed bottom-tip basis functions | 3178.982847 | 5297.872946 |
| Prescribed junction node at 40.0 km | 0 | 414.439966 |
| Unprescribed node at 39.9 km | 0 | 645.203094 |
| Other unprescribed nodes | 0 | 0 |
| Total tensile weight | **3178.982847** | **6357.516005** |

Node **756** at 39.9 km is lower-active at step 12, not prescribed deep slip.
Tensile weight is 0.4080% of its test weight. Its signed tensile friction load
is −2.29591925e8 Pa m², or 5.58597e-5 of its net friction-load magnitude.
The other 399 currently free nodes receive no tensile weight, but that does
not remove the tensile contribution from the active-set decision.

Friction uses signed **mu*sigma_n before integration**, and K_V includes
sigma_n*dmu/dV. Positive Q1 normal-stress ranges—[25.219,52.582] MPa at step
11 and [20.853,53.081] MPa at step 12—therefore do not negate raw tension.

### Shape, growth and deep-slip transfer

Consistent Q1 peak-to-peak variations, in MPa:

| Step / window | Delta p | −Delta tau:N | sigma_n | q |
|---|---:|---:|---:|---:|
| 11 / 13–20 km | 0.223826 | 0.121747 | 0.335798 | 3.770770 |
| 11 / 37–43 km | 0.272830 | 0.097460 | 0.348600 | 8.172248 |
| 12 / 13–20 km | 0.268602 | 0.149577 | 0.407423 | 4.131953 |
| 12 / 37–43 km | 0.319999 | 0.125083 | 0.444821 | 10.012363 |

Diagnostic detrended half-amplitude widths are about 4.35–4.38 km at the
constitutive transition and 0.23–0.26 km at the junction. The latter occupies
only a few 100-m surface elements. These are single-grid descriptions, not
convergence-qualified widths or proof of a continuum singularity.

As the last unprescribed nodal rate at 39.9 km falls from 8.9869e-10 at step
5 to 6.8583e-11 at step 11 and Vmin at step 12, the junction's projected
sigma range grows from 0.001296 to 0.348600 and 0.444821 MPa. **V(40−)=Vp
in continuous Q1**: the steep transition is over the last element, not a
discontinuous slip-rate limit.

Deep V enters the bulk as chi*V, not chi*(V−Vp), and prescribed entries are
not masked out. At the bottom, 85 associated QPs give sum chi*JxW=100.976813 m
and integrated crack rate 1.00976813e-7 m²/s. The profile-history correction
is exactly zero. This verifies transfer presence, not a new full-profile
normalization claim. No missing-deep-slip defect was found.

## 4. Single 35-km intervention: immediate effect, persistent old feature

The diagnostic prescribed Vp on additional existing nodes between 35 and
40 km, retaining the same saved bulk/history input and all other settings.
It did not change the BP3 constant Wf or commit a trajectory. Surface mass
entries match the baseline exactly; integrated bulk chi differs by at most
4.19e-16 relative. Current V is Vp on both sides of 40 km, including the
bulk-QP localization-weighted values.

The solve required six Newton updates and 162 Krylov iterations. Its largest
fresh-linear/target ratio was 0.841992195. Final bulk residual was
6.520747744e-4, normalized 1.025627013e-13; surface residual was
2.717249467e-4 Pa, normalized 1.040291205e-10. It had 350 free and zero
lower-active RSF nodes. Mechanical convergence is established independently
of the subsequent checker exception.

| Constitutive feature | Baseline | 35-km probe |
|---|---:|---:|
| Old 40-km sigma range (MPa) | −0.659852 to 100.176830 | **9.672915 to 89.835148** |
| New 35-km sigma range (MPa) | No comparable dominant raw extrema retained | **40.938090 to 59.441540** |
| Bottom minimum (MPa) | −20.406352 | **−20.399874** |
| 39–41 km projected sigma peak-to-peak (MPa) | 0.444821 | **0.362238** |
| 39–41 km projected q peak-to-peak (MPa) | 10.012363 | **8.085655** |
| 34–36 km projected sigma peak-to-peak (MPa) | 0.007310 | **0.043337** |

About **79.5%** of the old raw range and **81.4%** of its projected sigma
variation remain. The old low sample's pressure changes from −43.096 to
−34.243 MPa; its sigma rises by 10.333 MPa. The new low sample has
p=−10.499 MPa and −tau:N=+1.438 MPa. The current junction therefore has a
real immediate effect, but the dominant old feature does not relocate.

The junction tensile pocket disappears in this diagnostic; all remaining
tensile weight is the unchanged 5297.872946 m² on bottom prescribed support.
The bottom minimum changes by only 0.006479 MPa, about 0.032%. Moving the
junction is not a demonstrated remedy for the separate bottom-tip issue.

### Incomplete rollback verification

Converged data were exported before history publication, followed by a forced
rollback stop. The new verifier failed with Trilinos error −2 while
subtracting differently laid-out bulk vectors. Subsequent particle/surface/
geometry equality checks did not execute; `noncommitting_history.csv` was
not produced. No accepted state, ordinary graphical output or checkpoint was
published. Both checkpoint hashes are unchanged.

The checker was repaired to compare owned global entries. That repair
**compiles but is not runtime-verified**. Analysis explicitly records
`rollback_verified: false` and requires `--allow-unverified-rollback`.
Neither mechanical convergence nor the later extraction audit closes this
verification gap. No second intervention solve was run.

## 5. Frozen tau11 audit: retained concentration and local representation error

The audit stops after normal step-12 preparation. It reconstructs the solver's
private physical constraint lift, temporarily supplies that working vector
to the actual bulk cell assembler, and restores the substituted vector on
unwinding. This is **not** another Newton iteration. It exports 317,016 unique
parents and 317,004 QPs in a diagnostic strip, while assembling all 36,106
bulk cells, including those outside fault support. Ownership is per bulk cell
with MPI ADD; exported parent/cell/owned-DoF identities are unique.

At identical current parent positions within 400 m of the fault, normal
history RMS values are:

| Down-dip window | Particle tau11:N (MPa) | Constrained FE (MPa) | FE-minus-particle (MPa) |
|---|---:|---:|---:|
| 24–26 km interior | 0.628013 | 0.621740 | 0.063276 |
| 34–36 km control | 1.107070 | 1.097639 | 0.088887 |
| 39–41 km junction | **6.418345** | **6.231753** | **0.709266** |
| 13–20 km transition | 1.749316 | 1.736299 | 0.035885 |
| 114–116 km bottom | 4.588670 | 2.663962 | 3.053236 |

These are **unweighted parent-sample RMS values**, not surface weak norms or
errors against an exact solution. At 39–41 km, particle values span −15.5985
to +16.2912 MPa; FE values span −14.1386 to +14.6907 MPa. Broad profiles
nearly coincide, but pointwise differences reach −4.948 to +5.851 MPa.
The FE transfer attenuates RMS by 2.91% at the junction and 41.94% at the
bottom; it does not generate the broad junction concentration in this step.

Published versus constrained FE histories agree exactly at these near-fault
window samples. Over the larger exported strip, maximum constraint corrections
are 0.178038 MPa in a tensor component and 0.052642 MPa in tau:N. Constraints
are therefore not the source of the sampled junction discrepancy.

The profile plot separates the two normal-side strips, each 100–300 m from
the fault, rather than cancelling the dipole in a whole-column average.
Its 100-m bins are descriptive; raw data remain available without binning.

### Verified weak load

The actual frozen RHS is

\[
 f_i^{hist}=-\int_\Omega\beta\tau_{11,h}:\epsilon(w_i)\,d\Omega.
\]

The production `local_frozen_fault_rhs` is compared with independent weak
integration using the same constitutive beta evaluation. Homogeneous
constraints and MPI assembly are retained. The V-dependent load is separate;
no current viscous, pressure or boundary-traction term is included here.

| Check | Result |
|---|---:|
| Maximum local production/independent entry difference | 9.53674e-7 Pa m, against O(1e9) cell loads |
| Full constrained global load norm | 1.60659914537e10 Pa m |
| Isolated normal-tensor load norm | 7.54636894870e9 Pa m |
| QGauss(4) minus production quadrature, relative global norm | **1.78323e-15** |
| Sum of regional vectors minus full vector, relative norm | **2.42348e-16** |

Regional constrained norms are 7.77603e9 near 35 km, 9.63052e9 near 40 km,
9.05600e9 over 13–20 km, 6.07094e9 in the disjoint bottom region, and
1.88598e10 in the remainder. They are **not additive scalar percentages**:
artificial subdomain interfaces affect these loads. A frozen-load norm alone
is not an equilibrium residual.

For comparable near-fault fine cells, local unassembled load RMS is
1.45749e9 near 40 km (168 cells), 3.10714e8 near 35 km (168 cells), and
1.45279e8 near 25 km (167 cells). The inherited weak load is concentrated,
but this FE-only comparison does not measure a particle-to-FE weak-load error.

### What enters the tensile samples

Stable-ID joins recover the exact parent input at the saved converged sample
locations to 1e-8 m. Values below are MPa; old-history columns include beta.

| Sample | p | Particle old normal history | Current strain contraction | Actual sigma_n | FE old normal history at same parent |
|---|---:|---:|---:|---:|---:|
| Baseline 39.961334 km, ID 217673 | −43.096021 | 5.216355 | 2.347476 | **−0.659852** | 0.439753 |
| 35-km probe, same old hotspot | −34.243044 | 5.216355 | 0.867685 | **9.672915** | 0.439753 |
| Probe new hotspot, ID 230195 | −10.499438 | −0.906002 | −0.531527 | **40.938090** | −0.886542 |
| Baseline bottom, ID 11936 | −21.065329 | 40.855151 | 8.485873 | **−20.406352** | 19.098902 |

As accounting only, replacing particle old history by FE history while
holding saved u,p fixed would change the junction minimum to +4.116751 MPa
and the bottom minimum to +1.349896 MPa. **These are not corrected solutions.**
Such substitution would change the specified surface-history evaluation;
mechanical re-equilibration could also change u,p. This demonstrates sensitivity
of the tensile sign, not correctness of the FE representation or permission
to substitute it.

## 6. Combined conclusion and remaining decision

Established by the three tasks:

1. The Theta-checker defect is closed; baseline step 12 genuinely converges.
2. The bottom-tip and 40-km junction concentrations are distinct. The junction
   tensile pocket has nonzero measure and reaches an unprescribed active row.
3. Deep prescribed slip is present in bulk deformation.
4. Moving only the current junction does not move the dominant accumulated
   feature. This does not exonerate the original junction's historical loading.
5. Retained particle stress already contains the junction concentration.
   This step's FE transfer attenuates its overall RMS; the frozen bulk load
   is assembled consistently with its intended formula.
6. Local particle/FE discrepancies are material to pointwise tensile signs,
   especially at the bottom, but are not proof of an implementation defect.

Unresolved: spatial convergence of accumulated stress; cumulative transfer
error over earlier steps; a resolved finite-width effect versus a discrete
junction/tip layer; and the mechanical effect of differing history
representations. The 35-km probe's repaired full-state rollback checker also
remains unverified at runtime. The extraction's unwind restore was implemented,
but no new full in-memory equality test was added; checkpoint preservation
and absence of accepted output were verified.

The smallest proposed follow-up is a **frozen, common-test-function weak-moment
comparison of particle-domain stress and its FE representation**, near 40 km
and separately at the bottom, with an interior control. It must retain actual
domain measure, not replace domains by point weights. This would quantify
representation effects that pointwise differences and FE-only load norms
cannot establish. It has **not** been performed or approved as a production
correction by this consolidation.

Do not infer permission to smooth the junction, clamp normal stress, alter
pressure or Vp, extend RSF to the bottom, replace history transfer, or resume
the first-event trajectory from these results.

## 7. Reproducibility, changes and evidence index

Exact commands, environments, compiled source/binary hashes and checkpoint
hashes are preserved in each run's `provenance.json` / `execution.json`.
The runner scripts refuse to overwrite their existing evidence directories;
commands below identify the performed runs, not instructions to repeat them.

```sh
python3 benchmarks/reconstructed_fault/bp3/run_normal_stress_local4.py
python3 benchmarks/reconstructed_fault/bp3/run_junction_diagnostic.py
python3 benchmarks/reconstructed_fault/bp3/run_history_load_diagnostic.py
```

All output paths below are relative to `benchmarks/reconstructed_fault/bp3/`.

| Task / directory | Essential evidence |
|---|---|
| `normal-stress-complete-local4/` | `run.log`, provenance, accepted states; `analysis/convergence.json`, `linear_checks.csv`, `theta_independent.csv`; step-11/12 raw extrema, Q1 profiles, tensile weak weights, bulk transfer; `analysis/feature_widths.csv`, `junction_history.csv` |
| `normal-stress-junction35-local4/` | `run.log`, `execution.json`, `convergence.json`; `comparison.csv/json/png`; noncommitting surface and raw constitutive exports; recorded rollback-check failure |
| `normal-stress-history-load-local4/` | `run.log`, `execution.json`, `build.log`, `analysis.json`; `history_load_{parents,qp,cells,loads}_rank*.csv`; `hotspot_decomposition.csv`; `history_profiles.csv/png` |

Direct links: [baseline log](../../../benchmarks/reconstructed_fault/bp3/normal-stress-complete-local4/run.log),
[junction comparison](../../../benchmarks/reconstructed_fault/bp3/normal-stress-junction35-local4/comparison.png),
[history comparison](../../../benchmarks/reconstructed_fault/bp3/normal-stress-history-load-local4/history_profiles.png),
[history-load data summary](../../../benchmarks/reconstructed_fault/bp3/normal-stress-history-load-local4/analysis.json).

Across these tasks, `source/simulator/solver.cc` gained opt-in pre-publication
exports and the intentional noncommitting diagnostic stop. `bp3.cc` and
`junction_diagnostic.h` provide the diagnostic-only mask and rollback observer;
`history_load_diagnostic.h` provides the pre-Newton private-vector audit.
The runners and offline analyzers preserve provenance, separate accepted from
diagnostic data, and generate the tables/plots. Standard graphical output is
unchanged. No physical or numerical equation was revised.

Core/plugin builds used **-j4** and passed after the recorded diagnostic
compile fixes. Theta regression controls, baseline numerical checks,
noncommitting traction/projection checks and frozen-load integration checks
passed subject to the explicit rollback limitation above. Python compilation
and `git diff --check` passed. No broad suite or new one-/two-rank campaign
was run. **This report consolidation itself changes documentation only and
runs no simulation or numerical test.**

Preserved chronological reports:

- [Fresh baseline](stage_K5_normal_stress_local4_result.md)
- [35-km junction probe](stage_K5_junction_location_result.md)
- [Frozen history/load audit](stage_K5_frozen_history_load_result.md)
