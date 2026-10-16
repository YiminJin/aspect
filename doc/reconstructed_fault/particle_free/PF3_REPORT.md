# PF3 fixed-surface phase functional — 15 October 2026

**PF3 operator and controlled-phase scope complete.** User selection: PF3 as
proposed in PF2's handoff. All 67 final assessment gates pass. The
experiment evaluates and solves a controlled phase problem at immutable incoming
DG trial stress and physical surface Z; it does not adopt a coupled phase-first
time integrator. The retained P2 endpoint formulation/defaults and its failed
phase-first trajectory evidence remain unchanged.

## Discrete functional selected before implementation

On the existing fixed affine Cartesian Q1 phase mesh, let p be the nodal phase,
T the accepted DGQ2 trial tensor, and Z the accepted Q1 surface moment. The
reference surface, normal rays, common bulk/surface intersection partition and
all histories stay fixed during a phase solve. Define

```
b_k(p) = integral along normal ray k of h(phi_h(p)),
I(p) = P b(p), P = M_F^{-1} B_F,
tau_q(p) = T_q - 2 Z_q h(phi_q)/I_q S,
E(p) = sum_q w_q |tau_q(p)|_3^2/(4G)
     + sum_k w_k Z_k^2/(2G I_k(p))
     + sum_q w_q A [alpha(phi_q) + ell^2 |grad phi_q|^2],
A = Gc/(c0 ell).
```

The bulk rule is the existing shared mechanical rule. Ordinary FE values and
gradients replace particle-domain averages **only in the selected DG phase
operator after initialization**. AT1 alpha, ell, c0 and A come from the existing
phase handler; no new physical parameters. Homogeneous coefficients are already
required by the restricted native history option. The retained FE-history path
and native startup still use CPDI fracture integration.

For a direction d, `dI = P Db[d]` and `d2I = P D2b[d,e]`; P is independent
of phase. The bulk/cohesive Hessian includes local phi terms, both phi/I cross
terms, `DI^T E_II DI`, and `sum_i E_I[i] D2I[i]`. The last term differentiates
the actual consistent mass projection. Fracture adds

```
g_i = sum_q w_q A [alpha'(phi_q) N_i + 2 ell^2 grad phi_q . grad N_i],
H_ij = sum_q w_q A [alpha''(phi_q) N_i N_j
                    + 2 ell^2 grad N_i . grad N_j].
```

The box solve uses the existing native active-set/Newton machinery, immutable
accepted phase as lower bound and upper bound 1. Rational localization still
requires every evaluated nodal phase <1 and all projected I strictly positive
and finite. Zero/inactive normalization and exact saturation are rejected;
no floor, cap or filter is introduced. A near-saturated positive profile is
an integration stress test, not authorization to regularize its denominator.

Physical cohesive redistribution at fixed T,Z is
`delta tau = -2 Z (chi_new-chi_old) S`, with `q_new=Z/I_new` and unchanged
`s_elas=Z/G`. Trial evaluations must not publish any DG, Z, I, phase, particle
or surface history. A controlled bounded solve remains a noncommitting fixture.

## Verification declared before runs

- V7: independently recomputed energy/normalization at the same quadrature,
  relative/scaled target 1e-10; energy/residual and residual/tangent central
  differences over a perturbation sweep, best resolved scaled errors <=1e-6;
  symmetric tangent <=1e-12. Include tangentially varying phase/Z/history.
- Bounds: native unchanged KKT target; exact feasibility/no healing against the
  immutable accepted lower bound, repeated trials and no accepted-state mutation.
  Report stationarity separately from minimum/stability qualification.
- V8: vary normal quadrature at fixed mesh/history/profile, including width
  smaller than a cell. Compare energy, normalization and weak loads to an
  independently resolved integral. Then vary phase/history resolution separately.
  Report CPDI and ordinary FE as distinct discrete fracture functionals.
- Compatibility: rebuild and rerun the frozen FE/DG controls against PF2's pinned
  results and existing cohesive/Maxwell units. Preserve all failed controls.

## Implementation and preservation

Worktree `/home/ein/repository/aspect-pf-rsf-particle-free`, branch
`pf-rsf-particle-free`, HEAD `3232056cde12dbae17ef4265ecb804f9098b88f1` plus
uncommitted PF2/PF3 changes. Root `AGENTS.md`, incoming manual derivation/plans,
PF0–PF2 evidence and other checkouts/builds are preserved. No staging, commit,
push, default switch or physical-parameter change was made. The incoming file
manifest and patch are local `build-particle-free/pf3/incoming.{json,patch}`.

| Owner / files | Selected change |
|---|---|
| `phase_field_fault.h`, `dg_history.cc` | Batch evaluation of independent accepted DG tensors; optional explicit trial phase in physical stress sampling. No displacement-history velocity input |
| `fe_phase_energy.cc` | Reuse the exact bulk/cohesive energy and normalization pullbacks with accepted DG T/Z. Reject FE endpoint-energy/action calls in DG mode |
| `phase_field.h`, `simulator/phase_field.cc` | Add ordinary FE fracture energy/gradient/Hessian using existing phase coefficients. Initialized DG assembly returns the complete FE functional before accessing CPDI/material particles; fracture-only diagnostic calls retain CPDI. Reuse native bounded-solver safeguards for this independent history owner |
| `fe_normalization.cc` | Group symmetric jet products before addition; arithmetic-only correction described below |
| `current_design.md`, `specification.tex` | Matching experimental operator contract and unchanged outer-lifecycle restriction |
| `phase_energy_checks.cc`, `assess_phase_energy.py`, benchmark CMake/runner | Separate native observer, independently integrated/complex-step reference, quadrature/representation controls and provenance |

The enclosing ASPECT case initializes once. A separate observer drives two
native fixed-phase mechanical commits at dt=.125 s to t=.25 s, obtaining actual
nonzero DG history and physical Z. Subsequent phase solves are **noncommitting**
controlled fixtures: all Newton evaluations read that same accepted T/Z and
lower bound; the published phase, DG, Z/I and surface traction remain unchanged.
The existing guard against physical outer DG phase stepping remains. This
explicit scope avoids adopting the historical unsuccessful phase-first coupled
time algorithm. No coupled endpoint/DG tangent or physical phase-step lifecycle
is claimed.

## Independent reference and measured V7 results

The Python reference imports no production constitutive or assembly routines.
It reconstructs DGQ2 from exported tensor values at nine Lobatto points per
cell; builds Q1 values/gradients from physical cell coordinates; integrates
normal rays using Gauss192; and independently constructs and solves the Q1
surface mass system. Complex-step differentiation of this separately integrated
energy and analytic reference gradient supplies directional residual/tangent
checks. Native bulk integration is held fixed throughout each derivative sweep.
Both directions and the phase vary along the fault; native T and Z are nonzero.
Norm scales are `max(1 SI unit, reference infinity norm)` as in PF2.

| Check | Measured result / unchanged target |
|---|---|
| Full energy / gradient against independent reference | 0 / 1.044e-14 scaled; target1e-10 |
| Projected I / directional derivatives | I error3.337e-16 scaled; both derivative directions pass1e-10 |
| Complex-step energy directions | 1.933e-14 and8.395e-15 scaled; target1e-10 |
| Independent tangent actions | 7.187e-12 and1.627e-11 scaled; target1e-10 |
| Hessian symmetry | 3.639e-13 scaled; target1e-12 |
| Central energy differences, epsilon .01…1e-8 | Best scaled errors2.864e-10 /2.803e-10; target1e-6; truncation then cancellation visible |
| Central residual differences | Best tangent errors1.973e-9 /1.982e-9; target1e-6 |
| Omit normalization pullback negative control | Gradient discrepancy1.2107 scaled; detects a missing nonlocal contribution |
| Particle material/history poisoning | Set every particle property to NaN after initialization: identical energy, residual and Hessian; restore before legitimate history validation |
| Inactive/singular profiles | Constants0,1,1.01,-.01 rejected; .999999 remains finite/admissible. No floor, cap or filter |

Native smooth-trial totals are E=62.9199096095478 J/m,
E_bulk=39.9568300534045, E_coh=.049464708920604 and
E_FE-fracture=22.9136148472226. The independent sum at the same rule agrees
at the recorded tolerance. The retained native CPDI fracture is22.9135104795492
J/m: a real difference1.04367673426e-4 J/m, independently reproduced from the
exported native domain weights/gradients to9.303e-16 scaled. Neither functional
is silently relabeled as the other.

### Bounds and cohesive phase redistribution

The native box solve takes five iterations, E57.9333040047818→56.7007298102367
J/m. KKT=4.774e-15 against1.39484e-12, the original arithmetic/relative target.
Maximum phase increment=.165941616977379, maximum phase=.613176697049462;
healing is exactly0. There are63 lower contacts and90 free phase DoFs.
A repeat from the same immutable incoming state is bitwise identical.
No upper contact is claimed: phi=1 makes this active-profile energy undefined
and is rejected, rather than replaced with a cap.

The exact free phase Hessian has minimum physical mass-normalized curvature
137.5215, with minimum contact gradient.0394579. This is a fixed-history phase
check, separate from mechanically reduced coupled stability or global minimum
selection. Resolving the final trial at Gauss192 changes energy by1.994e-11
scaled and gradient by1.780e-9 scaled; the tiny discrete KKT residual is not a
claim of quadrature-level stationarity to the same tolerance.

At unchanged physical s_elas, peak trial cohesive traction changes
67.2437217939→51.4028243343 Pa. Peak elastic slip stays.00729172980161 m.
The nonzero stress redistribution norm is17245.00336 Pa; the independently
computed finite `-2 Z delta(chi) S` agrees with actual point stress evaluation
within2.734e-11 Pa (1.585e-15 scaled). Accepted DG, phase, Z/I and surface
normalization remain exactly unchanged after all successful/invalid trials.

## V8 quadrature and representation

Normal quadrature is varied at the **same mesh, phase coefficients, DG tensor,
Z, surface projection and normal rays**. Only scratch bulk integration geometry
changes and is restored. This is not a family of differently evolved histories.
The independent Gauss192/256 reference uncertainty passes1e-8 for energy,
bulk, normalization and weak loads. All native-versus-reference comparisons
at each identical rule pass1e-10, including the surface traction-work load.

| Profile / normal rule | Total energy error | Surface weak-load error |
|---|---:|---:|
| Smooth /3 | 2.380e-6 | 2.986e-5 |
| Smooth /6 | 3.941e-12 | 4.943e-11 |
| Sharp /3 | .97575 | 1.00080 |
| Sharp /6 | .96713 | .98923 |
| Sharp /12 | .75845 | .77235 |
| Sharp /24 | .15206 | .15441 |
| Sharp /48 | 9.852e-4 | 9.995e-4 |
| Sharp /96 | 6.283e-9 | 6.373e-9 |

Errors are scaled as above; the table reports measurements, not a universal
rule choice. The sharp Q1 tent has peak.995, adjacent values.05 and normal
cell size.0625 m. Its localization characteristic width
`(1-phi_peak)/abs(grad phi)` is about.000331 m, well below one cell. Analytic
normalization remains accurate (~9.82e-13 scaled) while bulk energy/load
integration fails badly at6 points. Accurate I alone does not qualify mechanics.
Defaults remain unchanged; a sharp evolving profile cannot use the smooth
fixture's six-point result as an accuracy certificate.

Independent manufactured studies then hold integration resolved and vary phase
and history spaces separately, with the native exported G/m/A/ell. They are
mathematical approximation studies, not new native trajectories:

- Q1 phase interpolation of a narrow Gaussian has nonmonotone coarse energy
  errors18.438,17.159,20.611 J/m at16/32/64 normal cells. At256/512/1024 cells
  errors2.893/.7363/.1849 J/m show the resolved second-order trend. The last
  error is still about.0414% of the446.2598 J/m continuum reference. More
  quadrature at the coarse mesh cannot replace missing phase DoFs.
- At exact prescribed phase, DGQ1 history energy errors fall.03152→8.248e-6
  J/m from8→64 cells; DGQ2 errors fall5.620e-4→2.272e-9. Continuum reference
  uncertainty is1.137e-13 J/m. This smooth nonpolynomial history is distinct
  from PF2's exactly representable native affine-Q2 strain source.
- For one smooth 2D field, FE minus rectangular-domain CPDI fracture decreases
  .00171684→.00043758→.00010992→.00002751 J/m on8/16/32/64 cells per axis.
  Both approach the independent continuum fracture21.2089897551 J/m, at their
  distinct errors. Native Voronoi-domain results are compared separately above.

## Preserved failures and arithmetic correction

The first native run completed, but its assessment failed the1e-12 symmetry
gate:1.255e-12 scaled (6.822e-11 absolute). Analytic-ray jets previously added
mixed-derivative products left to right, so transposed entries followed
slightly different cancellation paths. Grouping each symmetric product pair
and the `second*(d_i*d_j)` product changes floating-point association only;
values, first derivatives, equations, quadrature, physical bounds and tolerances
are unchanged. The final symmetry error is3.639e-13 and independent tangent
checks pass. No output Hessian is averaged or replaced with a positive matrix.

The preliminary harness also demanded monotonic error on every unresolved
phase mesh. That additional assumption was not a predeclared accuracy target;
the coarse sequence falsifies it. Its failed assessment is preserved. The
resolution study was extended to512/1024 and now checks/reports the asymptotic
256/512/1024 sequence while retaining the coarse nonmonotonic data. The early
manufactured helper's hardcoded m=10 was also replaced by the actual exported
native m=128; no native material was changed. The initial observer compile
failure lacked the explicit particle-domain header; that log is also retained.

## Compatibility, provenance and reproduction

Final core SHA256:
`5959f41bfeb27657ee9387bce0a9c08dc3c5ef26be31c2d701f114242c840a69`.
PF3 observer:
`906872a0a25443f870f104d84fcd8c239110bbb64a593a82903107d0c782eb26`.
The same GCC12.4/deal.II9.6.2/Trilinos14.2/OpenMPI5.0.10 Release build as PF2
is used, serial and one thread. Each native run is bounded by180 s.

| Run / check | Measured result |
|---|---|
| `pf3-phase-final` | Exit0;6.227s; peak child RSS234284KiB;67/67 assessment gates |
| `pf3-fe-control` / `pf3-dg-control` | Exit0;23.301/22.015s;222900/222932KiB; same final core and identical PF2 observer |
| FE disabled-DG baseline | All bulk/surface fields byte-identical to the fresh pre-PF2 source baseline |
| Same-build FE/DG | Max field/load scaled differences3.195e-14/7.319e-13, unchanged from PF2; all authority/source/work/publication gates pass |
| Existing tests |628 assertions/8 cohesive-Maxwell cases and94/94 independent PF1 checks pass |
| Compilation | Full/incremental native and observer builds; all four affected production translation units compile independently without unity/PCH |

Full records/raw fields/binaries/expanded inputs stay in `build-particle-free/`.
[Compact evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf3/README.md)
contains the final and failed assessments, exact identities, and a compressed
fixed-entry reference archive. All failed/preliminary runs are retained.
Use unused labels/output names:

```sh
cmake --build build-particle-free/native -j4
cmake --build build-particle-free/plugin -j2
python3 benchmarks/reconstructed_fault/particle_free/run_fixed_profile.py pf3-repeat --input benchmarks/reconstructed_fault/particle_free/fixed_profile_dg.prm --plugin build-particle-free/plugin/libphase_energy_checks.release.so --postprocessor 'PF3 phase energy checks' --seconds 180
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/assess_phase_energy.py build-particle-free/runs/pf3-repeat --output build-particle-free/pf3/repeated-assessment.json
```

PF2 report commands reproduce the two frozen controls, using the new final core;
`assess_fixed_profile.py` compares those with `baseline-fe-v2`. The incoming
root instructions/manual derivation/plans and all PF2 evidence hashes are
preserved. Python syntax, archive/reference replay, style for new C++ code,
source/artifact correspondence and whitespace checks are recorded with evidence.

## Remaining dependencies and next proposal

The initialized DG phase functional needs no particle H/composition/stress or
CPDI domain values. Restriction validation still checks particle count/positions;
startup still needs particle H/CPDI/domains, and surface material projection,
association/support and derived particle publication still require managers.
This is not an end-to-end particle-free run. No restart/conversion, transport,
limiter, geometry/mesh changes, RSF evolution, coupled temporal/full energetic
or propagation qualification is added. The outer DG physical-phase guard and
historical coupled phase-first failure remain explicit.

Proposed next scope is **PF4 prescribed smooth DG material transport**, on the
same fixed geometry and constant coefficients, with no limiter initially:
constant/pulse and manufactured nonzero-divergence tests, separate space/time
refinement, native FE velocity/normal flux/inflow, and the bulk/surface
commutator negative controls. Decide on limiting only from measured transported
history behavior. Full phase/mechanics lifecycle adoption and broader coupled
qualification remain later selections. No PF4 work has started.
