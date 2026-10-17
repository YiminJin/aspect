# Bounded stationary mature-fault DG qualification

16 October 2026; branch `pf-rsf-particle-free`, base
`a8707307008401ac9cff702a369e2e2a611fa0f7` plus preserved incoming work.
[Pre-run formulation](PF6_MATURE_DG_FORMULATION.md),
[source audit](SOURCE_AUDIT.md), [particle inventory](PF6_PARTICLE_READ_AUDIT.md),
[portable evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf6_mature_dg/README.md).

The domain-free DG extension passes native-FE reconstruction/coupling and the
short strengthening-RSF fixture. **Spatial accuracy remains unresolved**, and
perturbation-load timestep sensitivity exceeds the declared 1% screen. No phase
evolution, automatic AMR, transport, moving geometry or first-event run was made.
Hanging-node phase evolution is deferred. Unlimited DGQ2 and production defaults
remain unchanged; WENO stays experimental and is not used here.

## Equations and ownership

For fixed phase/geometry, homogeneous matched G/eta and stationary history,

\[
 \beta=e^{-\Delta tG/\eta},\quad\kappa=-\eta\operatorname{expm1}(-\Delta tG/\eta),\qquad
 T^{n+1}=\beta T^n+2\kappa D_3(\operatorname{symgrad}u^{n+1}),
\]
\[
 Z^{n+1}=\beta Z^n+\kappa V^{n+1},\qquad
 \tau=T-2\chi ZS,\quad\chi=h(\phi)/I_h.
\]

Mature cohesive resistance, traction and storage are zero. **Z=G s_mem is
auxiliary relaxation/slip memory**, independent of cohesive elastic slip and
of I_h*C=0. With fixed chi these equations exactly reproduce native direct
Maxwell stress. Native FE uses T=2G D3(symgrad d_R) and
`d_R_new=beta*d_R_old+(kappa/G)*u`. Preserve the full plane-strain tensor:
zz=-(xx+yy), norm²=xx²+yy²+zz²+2xy². No spin/rate convention changes occur.

DG coefficients own T; the reused FE velocity-history block stays zero. The
existing mature native commit owner supplies fixed phase, geometry, Z/I and
validation before a single publication. Mature DG does not enable PF5's
prescribed-transport checkpoint flag. Particles are derived diagnostic and
independent native-check points; neither their stress nor their labels feed
accepted DG/FE mechanics or fault loads. Bulk coefficients and phase/state
initialization need no domains, CPDI or particle projections in the selected mode.
Every physical phase DoF must be prescribed, admissible and exactly unchanged.

Use actual segment tangents/normals/shear sense, native bulk-source endpoint
continuation, general oblique cell-interval normalization and paired boundary
completion with exterior Q1 contributions. No horizontal P2 replacement,
normalization floor, stress bound or adaptive anisotropic rule is introduced.
All nine DGQ2 modes remain. Native source/mass integration stays Gauss3; selected
bulk mechanical Gauss3/6/8 rules are separate and shared across mature operators.

## Controls and measured correctness

Reuse section 7B's translated 2000x1000 m Box, straight 60-degree mature fault,
ell=20 m, core=.6, 4136 graded Cartesian cells (7.8125--125 m), 59 surface nodes,
G=32038120320 Pa, eta=1e26 Pa s, a=.025, b=.015, Dc=.008 m,
Vp=1e-9 m/s, initial Theta disturbance .02, sigma0=50 MPa and existing filter20.
Full bottom/side loading and natural top remain. Configured cohesion/Gc still
supply the retained prescribed profile/localization law; mature mechanical
cohesive resistance stays zero. Bulk perturbation histories start unloaded;
background stress stays surface data. All trajectories use this same physical
initialization prescription, not an imported coarse endpoint.

The particle control retains its native domains, CPDI and actual stress-to-FE
LLS transfer, with the existing optional **three-dimensional-deviator** convention
selected to match FE/DG. Its original symmetric-gradient default and evidence
remain. Mesh, initial physical phase and mechanical quadrature files match
byte-for-byte. Common physical particle IDs/coordinates/H/stress/Theta/strengthening
hash identically after resolving native static versus mapped property names.
Surface state/loading and all physical coefficients match. Background composition
labels differ in the domain-free setup; homogeneous coefficients make this
physically immaterial. Variable material/friction distributions need explicit
surface data and are outside this qualification.

- Independent 70-digit fixed-chi algebra: defect 1e-69. Gauss3 versus Gauss8
  Q2 mass and reproduction of all nine monomials, including x²y²: 6.67e-16.
- Final-build prescribed history alternates non-proportional tensor forcing and
  spatial slip forcing across dt/lambda=1e-12,1e-6,.1,1,4,.25, ending in relaxation.
  Each of two paths checks 215072 points per interval, including 13044 continuation
  samples. Max full-tensor DG/FE scaled stress defect: 3.80e-15; actual surface
  weak-load/residual scaled defect: 5.67e-16. At manufactured stresses up to
  1.11e16 Pa, absolute defects reach 35.6 Pa and 40 Pa m; these are algebra
  controls, not physical stress amplitudes.
- Invalid scratch trials preserve accepted DG coefficients/FE anchor/Z; moving a
  fixed diagnostic point rejects; duplicate terminal commits reject. Coupled
  trajectories have no cutbacks; this does not qualify mature outer-loop retry
  or restart, which remains unsupported.
- Independent fixed Gauss16/32 integration of actual interior Q1 restrictions
  and exterior completion agrees with native nodal I_h to 5.97e-16 relative;
  reference self-difference is also 5.97e-16. Maximum exterior contribution is
  5762.268 m; minimum native I_h is 12188.324109 m.
- All three ten-positive-step trajectories reach T=2e6 s at dt=200000 s. Native
  state/slip/Maxwell checks, zero C with nonzero Z, frozen phase/I/H/geometry,
  fixed point population and one publication per positive step pass. Step zero
  is a separate kinematic solve; its committed T/Z and accumulated slip are zero.
  Max Theta-aging relative defect is 2.23e-16; max normalized nonlinear residual
  is 3.03e-9. The native accepted-work reconstruction guard passes (<1e-5 Pa).
- All-point publication defects: DG <=3.23e-9 Pa, native FE <=1.26e-8 Pa; maximum
  fractions of the unchanged derived allowance are 4.49e-5/.00798 respectively.
  Fine DG <=2.03e-9 Pa and fraction 1.82e-5. The obsolete net-stress-only allowance
  would fail up to 43057/101702 samples on coarse/fine DG; those diagnostics remain
  recorded. No tolerance was multiplied to obtain these passes.

| Ten-step result, Gauss3 | DG | Native FE | Matched particle |
|---|---:|---:|---:|
| Wall seconds, including startup/exports | 106.69 | 108.19 | 97.99 |
| Final V range / Vp | .988876--1.010974 | .988876--1.010974 | .990515--1.012908 |
| Final max Z, Pa m | 6.451512e7 | 6.451512e7 | not stored |

Final DG/FE whole-domain and localization-weighted tensor relative L2 differences
on common Gauss3 points are 9.07e-14 and 2.21e-14; maximum tensor difference is
1.31e-8 Pa. Max slip-rate difference is 6.00e-24 m/s, raw-normal projection
7.46e-9 Pa, and filtered projection 1.50e-8 Pa. Gauss6 DG/FE controls agree similarly.

Particle/FE final whole/local tensor differences are **84.53%/83.34%**, local RMS
81.93 kPa; max point difference 209.26 kPa. Yet max V difference is .1933% of Vp,
raw-normal projection 3.292 kPa and filtered projection .592 kPa. Small velocity
or background-scaled differences alone would hide this history discrepancy.
This comparison measures the complete native transferred-history trajectories,
not an identical-history particle-versus-FE interpolation experiment or a
FE-versus-CPDI fracture comparison; phase is prescribed in all three.

## Quadrature, timestep and one spatial reference

Declare before validation: 1% of Vp for V, 1% of Dc/Vp for Theta, and 1% of
measured perturbation scales for stress/tractions. Use relative full-tensor norms;
normalize traction changes by background-subtracted reference perturbations,
not 50 MPa. Tests use T=600000 s and retain the same 200000 s step-zero constitutive
interval. No threshold tuning or WENO timestep sweep occurs.

| Comparison at T=600000 s | Tensor relative L2, domain/local | max dV/Vp | shear/raw/filtered traction change, Pa |
|---|---:|---:|---:|
| Gauss3 versus Gauss6 | .2193% / .07124% | .003862% | 48.78 / 25.95 / 16.87 |
| Gauss6 versus Gauss8 | .06888% / .005083% | .0003892% | 4.19 / 5.14 / 1.10 |
| dt versus dt/2, Gauss6 | .1288% / .1119% | .01179% | 141.26 / 88.04 / 33.94 |

Gauss6/8 stays below the declared screens throughout the short horizon. Gauss3/6
filtered-load sensitivity is about 1.05% at the endpoint. Timestep halving passes
V/Theta/tensor screens but fails the perturbation-load screen: maximum shear/raw/
filtered relative changes over matching positive times are **2.48%/1.13%/2.36%**
(2.21%/.953%/2.07% at the endpoint). Two timestep levels establish uncertainty,
not temporal order or a converged load history. Gauss6 is a reasonable next
fixed-rule control; the production default remains Gauss3.

The sole finer native control re-solves three physical steps on 11528 graded
cells, finest h=3.90625 m, with the same prescribed physical seed, dt, Gauss6,
loading, acceptance rules and **unchanged 59-node surface**. FE DoFs increase
149538->410198; mechanical velocity/pressure DoFs 39607->108621. DG history
has 111672->311256 scalar coefficients plus 59 Z values. Diagnostic points are
16 per cell (66176->184448), fixed throughout each separate fresh trajectory;
no particle supplies geometric support. Wall time is 174.45 s versus 60.74 s for
the coarse three-step DG control; peak RSS about 1066 MiB versus 528 MiB.

| Initialization measure | Coarse | Fine |
|---|---:|---:|
| Phase relative error on common-rule Gauss3 samples | 2.513% | .6466% |
| Phase sample Linf | .02930 | .007845 |
| I_h deviation from actual physical profile normalization | 6.630% | 1.802% |

The physical profile's fixed Gauss32 normalization is **13053.839445 m**.
These deviations are **mesh representation error**, distinct from the ~6e-16
integration error of the actual mesh field. The phase figures are sampled
quadrature norms, not independently converged exact integrals of representation
error. Both meshes use the actual native 5000-knot physical prescription.

At 450 common interior points on three normal cuts (2 m spacing, ambiguous face
points excluded), coarse/fine tensor RMS differences at T=200000/400000/600000 s
are 2.244/4.488/6.731 kPa; physical-profile-weighted local differences are
7.529/15.058/22.587 kPa. Final relative cut norms are **402%/385% of the fine
signal**, not whole-domain error estimates. Both face traces remain exported:
maximum final tensor jumps decrease 91.40->14.82 kPa, raw-normal jumps
45.74->7.75 kPa. On the same cuts, final quadrature/local timestep uncertainty
is only 1.46/3.48 Pa in localized tensor RMS, well below the spatial difference.

At the surface, final coarse/fine differences are .3322% Vp, 4351/6197/865 Pa
in shear/raw/filtered projections. Relative to fine perturbations those load
changes are 228%/210%/117%, failing the spatial screen. V alone again masks the
problem. Refinement changes both mechanical resolution and frozen phase-space
representation; this is not a separation of those errors or a converged continuum
reference. No relaxation of an imported coarse state is used.

Cancellation remains strong: on each mesh's Gauss3 measure, localized T/physical-tau
norm ratios are 33.55 (coarse) and 127.36 (fine) at the short endpoint. The common
DG/FE reconstruction and strong refinement response identify a remaining shared
mechanical/profile-resolution problem; they do not prove a unique remainder
mechanism for the oblique coupled fixture. See the
[measured spatial figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf6_mature_dg/spatial-reference.svg).

## Source corrections, exports and reproducibility

The new selectable route reuses ownership, native laws and general geometry.
The original first operator failure omitted the existing DG source allowance in
an FE-only diagnostic. The first physical-step failure incorrectly restricted a
frozen-phase callback to timestep zero. Both originals remain under
`operator-dg` and `gate-dg`; fixes use the established derived allowance and exact
accepted-phase equality, respectively. No production validator was bypassed.

Export `authoritative-stress-N.csv` (physical tau and independent T/Z/chi), all-nine
`dg-coefficients-N.csv`, physical FE coefficients, raw/filtered surface profiles,
normal cuts with both traces, and separately derived unfiltered point tractions.
Raw point tractions are p+sigma0-tau:N and tau0+tau:S, without a filter or clipping.
They use common Gauss3 diagnostic points; native Gauss6/8 coupling rules remain
separate. Nodal raw profiles are consistent weak-load projections, not raw point
extrema. At step zero stress exports represent committed zero history, whereas
surface/cut diagnostics describe the separately recorded kinematic solve.

Particle exports have no independent T/Z: T columns echo physical tau and Z is NaN
at positive steps; the surface-profile Z=0 means absent storage. The original
particle `physical-fe` history column echoed velocity and is not an independent
history quantity; subsequent observer output uses NaN. It is not used in any norm.
Static diagnostic Theta has a different native property name; the final short
gates hash the common physical field under both aliases and match exactly.
These output-only corrections do not change native equations or acceptance.

Primary ten-step artifact SHA256 identities (every case also pins its full input,
argv, source hashes, exact binary/plugin copies, log, cost and CSV hashes):

| Artifact | SHA256 |
|---|---|
| aspect-release | `9f246a95e2cdb1dbf9c0b02991d4fc8d0477d2a04d926523b2af9ec9ab19cda2` |
| libbp3_restore_150x50.release.so | `5aae1016c7c61419feb1eeb25ac89749995bb8ab83e518d014cbf992211e7006` |
| libbp3_coupled_checks.release.so | `700bbabe471df84d5d4fd425d1bb4b162e31529638a3d016b6d010e01f38eb5b` |

All primary trajectories, final algebra and sensitivity/spatial controls use the
same core and maintained BP3 library. Diagnostic observer revisions add output
metadata/common-point exports only; each is separately pinned. Rebuild commands:

```sh
cmake --build build-particle-free/native -j 2
cmake -S benchmarks/reconstructed_fault/bp3_native/server/plugin -B build-particle-free/mature-dg/bp3-plugin -DAspect_DIR=$PWD/build-particle-free/native -DCMAKE_CXX_FLAGS=-DASPECT_BP3_LOCAL_BOTTOM_TEST
cmake --build build-particle-free/mature-dg/bp3-plugin -j 2
cmake -S benchmarks/reconstructed_fault/bp3_native -B build-particle-free/mature-dg/observers -DAspect_DIR=$PWD/build-particle-free/native -DBP3_COUPLED_PLUGIN=$PWD/build-particle-free/mature-dg/bp3-plugin/libbp3_restore_150x50.release.so
cmake --build build-particle-free/mature-dg/observers --target bp3_native_checks bp3_coupled_checks -j 2
```

The selected drivers are `run_pf6_mature.py` and `run_pf6_mature_cases.py`, launched
through the established bounded logger. The evidence README gives replay and
new-label commands. All 18 native attempts, including failed ports and preliminary
controls, cost **1278.61 s of 3600 s**; compilation is excluded. Runs are sequential,
serial, one thread. No full first-event or production job was launched.

All 34 declared model defaults match actual task entry. Transport/limiter,
accepted checkpoint, remap, phase-energy, general normalization/derivatives and
particle-domain/manager source are byte-identical to entry. Existing work and
all prior evidence remain. **strict_reference_match_passed=false stays false**;
this experiment does not repair or supersede prior native-quadrature/reference
failures, qualify moving geometry, phase evolution, AMR/MPI, mature restart,
heterogeneous materials or propagation, or establish full PF5 completion.

## Recommendation and next task

Retain unlimited domain-free DGQ2 as a **verified FE-compatible experimental
mature-history backend**. At matching settings it has essentially FE accuracy
and comparable cost; the slightly cheaper particle control has much larger
perturbation-stress differences. Neither backend agreement nor small V/background
errors qualify this coarse physical discretization.

Next select a bounded **fixed-phase oblique mechanical/profile resolution study**:
hold a common physical chi and incoming histories fixed to separate mechanical
resolution from phase/I representation; then refine the prescribed profile space
separately. Use the measured cancellation/jumps and an inexpensive normal reference
to choose a graded bulk resolution, holding the surface fixed. At that resolution,
verify perturbation loads with one further timestep level before any event run.
Mature native rollback/restart/MPI and a mesh-owned replacement for independent
point checks need separate qualification. Keep the current independent publication
check and defer hanging-node phase evolution, transport, automatic AMR, moving
geometry, fault birth and propagation.
