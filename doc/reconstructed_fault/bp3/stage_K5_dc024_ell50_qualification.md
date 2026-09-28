# Modified BP3: Dc=0.024 m, ell=50 m — bounded qualification

## Decision

The parameter-consistent case and narrower reference are prepared. All six
frozen mechanical probes pass, but **12.20703125 m is not qualified**: its
realized localization RMS width has up to **1.9251%** error against the
specified 1% criterion. The 6.103515625 m interior patch reduces this to
**0.4806%**. Mechanical stiffness changes are only 0.238%, 0.658% and 1.744%,
all below 5%. No criterion was relaxed.

Consequently no real timestep, restart comparison or half-step comparison was
launched. This is a mesh/profile prerequisite failure, not a nonlinear failure
or evidence of stable earthquake-cycle behavior. The local fine patch does
not qualify a whole-fault fine mesh. The original Dc=.008/ell400 fixture and
its evidence are unchanged. No production equation or numerical algorithm changed.

## Configuration and initialization

Starting HEAD: `359ea223cd0fa088ba7c8b03338cc7fe19d4acdc`, with pre-existing
working changes preserved. New inputs:

- `benchmarks/reconstructed_fault/bp3/bp3_dc024_ell50.prm`;
- `benchmarks/reconstructed_fault/bp3/fixtures/modified_bp3_dc024_ell50/`;
- separately prepared, unrun `bp3_dc024_ell25.prm` and corresponding fixture.

Each fixture has a hash manifest. The executed binary/plugin/input hashes and
resolved PRMs are in `length-scale-study/probe-{candidate,reference}/`.
`length-scale-study/source-before/` preserves the entering BP3 source/header,
plugin and executed launcher. The final launcher additionally blocks evolution
on failed prerequisite gates; that refusal was tested without launching ASPECT.

| Quantity | Value |
|---|---:|
| Dc, ell | 0.024 m, 50 m |
| Geometry | unchanged 300×100 km box, 60° straight fault |
| Fault | fully frictional, mature C=0, continuous Q1; 1156 vertices |
| Actual element lengths | 99.960336–100.000000 m |
| G, viscosity | 32.03812032 GPa, 1e26 Pa s |
| Intended core phase | 0.6 |
| Intended full support, localization RMS width | 197.6458201 m, 22.1118411 m |
| Artificial initialization interval | unchanged 4e6 s; no elapsed physical time |
| L_b, at 50 / 60 MPa | 2.050 / 1.709 km |
| h*, at 50 / 60 MPa | 11.748 / 9.790 km |

The existing material `Characteristic slip distance` is now authoritative for
benchmark initial Theta, its independent aging checker and V dt/Dc reporting.
`PhaseFieldFault::characteristic_fault_slip_distance()` is a narrow const
accessor, not another parameter. Production friction/time stepping already
used the configured value. Initial Theta is exactly the formula with Dc=.024,
three times its .008 value, not a replacement by steady-state Dc/V.

Both four-rank initializations checked nodal and initial particle Theta with
zero relative discrepancy. Copies of the production friction law exercised
192 exact aging updates at both Dc values: maximum independent-reference
relative error 2.22e-16, initial-friction error 2.22e-16. Ten separate captured
near-bound-rate checker cases passed, including rejection of intentionally
incorrect state and retention at zero interval. The old horizontal-depth
check passed (fraction error 2.44e-15, Theta error 2.84e-14).

The background prestress file is byte-identical to the accepted physical
background. Its captured rational correction is part of that frozen function,
not the current normalization denominator. Replacing it with the new I_h
would retune the physical target. Current surface masses, work-weighted loads
and normalization are regenerated; no old nodal I_h is restored. The base
surface RMS before nonlinear initialization changes from 1,754,817.61 to
1,758,309.31 Pa between these meshes. These are **unconverged base** values,
not accepted initial tractions or a prestress fit.

## Profile and endpoint checks

Same continuous AT1 profile sampled independently on each mesh; no coarse
profile prolongation. Its production 5000-panel table agrees with the
independent construction within 1e-7 m in radius and 2e-15 in phase. Actual
exported FE/QP values reproduce that mesh's Q1 field within 1.01e-13; evaluated
chi agrees with h(phi)/interpolated I_h within 6.77e-15 /m.

Interior checks use 61 common columns over 15–18 km. Cell-split 16/32-point
independent integrations agree to 1e-9 relative for both zeroth and second
moments. No integration tolerance was weakened.

| Quantity | 12.207 m candidate | 6.104 m local patch |
|---|---:|---:|
| RMS width range, m | 22.52251–22.53751 | 22.21769–22.21812 |
| Maximum RMS-width error | **1.92506% — fail** | 0.48062% — pass |
| Maximum abs(J/interpolated I_h−1) | 0.044589% — pass | 0.0002052% — pass |
| Max column I_h error versus continuous profile | 2.79874% | 0.71721% |
| Realized phase on fault, sampled columns | 0.587971–0.598731 | 0.596927–0.599539 |
| Global maximum phase at FE vertices | 0.6 | 0.6 |

All 166,078 nonzero-phase candidate cells have h<=12.20703125 m, including
both endpoint neighborhoods. Mesh area remains 3e10 m²; h_max=12500 m.
The reference halves band and graded-halo cells over approximately 10–23 km
down dip and ±2.5 km normal distance; the probe is tapered over 15–18 km.
Its endpoints deliberately retain the candidate mesh and its completion.

Four surface integration profiles need exterior completion. At the nearest
top/bottom Gauss profiles, the in-box fraction is 0.833017 / 0.832932, not one.
Production inside+outside equals completed exactly. Completed totals differ
from independent full virtual-Q1 columns by at most 3.66e-8 relative; inside
integrals by at most 4.39e-8. Completion generation's 8/16-point discrepancy
is 1.74e-10 m. This verifies the regenerated denominator split, **not** a new
uniform-sliding endpoint source/stress replay. Existing paired source support
is retained; endpoint mechanical qualification on a globally finer candidate
remains to be done rather than inferred from zero endpoint probe amplitudes.

## Frozen mechanics and finite-band error

Each run uses the existing frozen linear-response construction, identical
100-m Q1/tapered inputs and zero incoming perturbation stress. It performs
three A dx=B dV response solves. The existing hook first computes one
uncommitted coupled initialization direction, then deliberately exits after
`MECHANICAL MODES VERIFIED`; no history or accepted timestep is published.
Exit status 1 is intentional here, not evidence of nonlinear convergence.

Table entries are **positive work-conjugate elastic shear stiffness**, in
MPa/m, converted using G/kappa. Here kappa=1.2815248119788472e17 Pa s. Neither
the instantaneous friction derivative nor normal-stress feedback is included
in these mechanical coefficients. Raw separated contributions remain saved.

| Actual tapered-Q1 mode | Candidate | Local fine | abs(coarse/fine−1) | Infinite-band prediction for same input |
|---|---:|---:|---:|---:|
| 1500 m | 107.75395 | 108.01134 | 0.23830% | 108.12047 |
| 600 m | 204.36241 | 205.71662 | 0.65829% | 206.20634 |
| 200 m alternating | 268.05422 | 272.81121 | 1.74369% | 274.50075 |

The fine mesh differs from this continuum diagnostic by 0.101%, 0.237%, and
0.615%. The prediction uses the exact Fourier transform of the actual Q1
input and exact line mass, not a nominal sinusoid. Normal-kernel sampling
refinement changes predictions by at most 7.50e-6 relative. The finite-box,
inclined FE operator is not asserted to equal the infinite homogeneous model.
No evidence here motivates enlarging the already buffered patch.

Fresh relative residuals: 2.50e-11–9.64e-11, all below 1e-10. Iterations:
candidate 16/16/16, reference 15/16/17. Work-pair discrepancies <=1.13e-14,
independent action discrepancies <=1.92e-14. Signed bulk-relaxation components
sum to their independently measured total within the existing check. Native
mass/exact Q1 line mass differs by <=4.52e-5 (candidate), 2.73e-6 (fine).

Finite-width error is separate: the nominal ell50 pure-sinusoid band/sharp
ratios remain 0.81970 at 1500 m and 0.84675 at 1800 m. These ~18%/~15%
differences are intentionally part of the proposed finite-band model, not
bulk-mesh error. At 200 m the ratio is only 0.27470; no sharp-fault accuracy
claim is made there.

The uniform-Q1 steady-sliding screen samples 200–3000 m every 10 m, including
Q1 aliases. Minimum stiffness is 60.678 MPa/m (3000 m); alternating stiffness
274.458 MPa/m. Both exceed 10.417/12.500 MPa/m at 50/60 MPa across the screened
range. This does not certify an evolving front or exclude intended long-wave
nucleation. Changing Dc still changes the physical nucleation problem.

## Cost, status and next decision

| Resource | Candidate | Local reference |
|---|---:|---:|
| Cells | 229,890 | 298,914 |
| Total DoFs | 8,256,730 | 10,654,274 |
| Particles, 9/cell | 2,069,010 | 2,690,226 |
| Total preparation/assembly/probe wall time, 4 ranks | 282.623 s | 335.710 s |
| Largest child-process peak RSS | 4.19 GiB | 5.23 GiB |

Aggregate simulation time **618.333 s**, within 7200 s. Sampled summed rank RSS
was approximately 16.6 / 20.4 GiB; those are observations, not independently
measured aggregate peaks. Small host swap use appeared in the larger run.
Separate assembly time was not instrumented; the table includes mesh creation,
CPDI/profile preparation, assembly/preconditioning and all responses.

Status: parameter consistency, six response solves, work/derivative checks,
normalization and endpoint denominator completion **pass**; candidate RMS
profile width **fails**. Accepted nonlinear initialization, 8–12 real steps,
restart, half-timestep comparison and finer endpoint mechanics are **unrun**.
The ell25 reference is only prepared (459,696 cells, 4,137,264 particles); it
has the same ell/h as the failed width candidate and is not qualified either.

Next decision: use approximately 6.104 m over the complete ell50 band and
endpoints, not just the probe patch, before the production qualification.
Extrapolating refinement of the candidate's 179,048 finest cells suggests
roughly 0.7–0.8 million cells and 50–60 GiB aggregate solver RSS. This is an
estimate, not a built or measured global fine mesh; plan a 96-GiB host for
margin. No additional mesh or physical parameter was silently promoted.
The prepared eight-step case is explicitly **blocked**, not ready for longer
evolution. No full-cycle command is authorized by this result.

## Reproducibility and changes

Commands are in the new fixture README. Results under
`benchmarks/reconstructed_fault/bp3/length-scale-study/` include:
`comparison.json`, `coefficients.csv`, `profile_columns.csv`,
`stiffness_screen.csv`, `resolution_gates.png`, per-run raw quadrature/velocity
and FE-profile exports, resolved parameters, launch hashes, logs and execution
records. The initial analysis had a NumPy-boolean JSON serialization error;
only the offline serializer was corrected and analysis rerun, no simulation.

Maintained code changes: one material Dc accessor; explicit Dc arguments in
`bp3_model.h`; configured-Dc plumbing in `bp3.cc` and the opt-in disturbance
checker; updated standalone depth/Theta checks. New benchmark tests:
`tests/bp3_length_scale_checks.h`, `tests/bp3_length_scale_mesh.cc`; existing
mechanical probe gains the opt-in 1500-m mode and consistency hook. New mesh
target added to performance CMake; no existing diagnostic mode is replaced.
Scripts: `length_scale_study.py`, `analyze_length_scale.py`. Builds used -j4;
both plugins and mesh utility built successfully. Python syntax checks and
`git diff --check` pass. No full ASPECT suite or new production trajectory was
run. No commit was made; unrelated working changes remain untouched.
