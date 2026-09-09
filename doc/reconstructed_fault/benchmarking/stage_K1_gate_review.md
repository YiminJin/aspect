# Gate K1 review after the CPDI ownership correction

Subsequent authorized follow-up: `stage_K1_support_resolution.md` now contains
an accepted corrected t=0 mechanical comparison, the first real-step line-search
failure and a bounded support/error-budget proposal. The review below preserves
the earlier pre-pilot evidence. Gate K1 remains unsatisfied.

**Gate K1 is not satisfied. K2.1 has not started.** Corrected initialization
passes several prerequisites, but the existing association-strip containment
target fails and no corrected accepted mechanical trajectory or convergence
matrix exists. No equation, history, parameter, constraint or tolerance was
changed in this review.

The requested instructions are located at
`doc/reconstructed_fault/benchmarking/stage_K_benchmarking_instructions.md`;
there is no copy directly under `doc/reconstructed_fault/`. Gate K1 requires
independent finite-width reference convergence and lifecycle checks, with the
spatial/time/MPI/restart evidence specified in K1.7. The numerical targets used
below are the pre-pilot targets in `stage_K_progress.md`, “Error targets fixed
before the pilot”, not new thresholds chosen from these results.

## Evidence and missing requirements

Artifact abbreviations below refer to
`benchmarks/reconstructed_fault/uniform_shear/diagnostics/`:

- **C:** `results-corrected/`, the saved corrected initialization packet.
- **A:** `gate-k1/`, this review's independent calculations from C.
- **R:** `results/`, which retains the original baseline packet and separately
  named ownership-correction build/regression/initialization logs.

| Requirement | Status | Evidence and limitation |
| --- | --- | --- |
| K1.1 symmetry-compatible setup | Setup passes; accepted response unmeasured | Unchanged `pilot.prm`: periodic bulk x, symmetric top/bottom tangential loading, zero normal velocity, prescribed 1000 Pa friction pressure, homogeneous material/temperature. Open fault topology is retained. Actual corrected accepted boundary traces are not available. |
| K1.2 actual reconstructed geometry | Pass at this resolution | A: length 0.25 m; max abs(y)/ell=5.1893e-17 and max angle=1.5310e-16 rad, both below 1e-6. Both actual endpoints are at the intended x boundaries to stored precision. |
| Converged initial phase field | Pass | C measurements and R/corrected_initialization.log: 14 Newton updates; final independently reassembled relative residual 1.205790478e-9 < 1e-8, initial assembled-vector norm 7159.494342. Initialization diagnostic deliberately exits 1 after export, before mechanics. |
| Physical phi and nonsingular I_h | Pass for sampled corrected profile | C: phi in [1.610915258e-9,0.600582655711]; all Q1 interpolants are physical and g>0. No clamp to the activation threshold or upper clipping. |
| Initial transverse symmetry | Pass | C: maximum along-x phi range 6.88338e-15; center range 5.21805e-15. These are phase/geometry results, not accepted mechanics uniformity. |
| Independent prescribed H and phase residual | Pass for the implemented initializer/discrete equation | Prior `check_stationary_equation.py`, C and preservation comparison: initial H error against its independent tabulated-profile reference 2.72013e-7 relative; H is unchanged. The activation-switched H is deliberately not equated with an exactly stationary untruncated profile. A fully independent 1-D initialization is K3, not claimed here. |
| K1.4 production I_h versus independent same-FE integral | Pass | A: maximum consistent-Q1-projected relative error 4.64632e-10 < 1e-6. All 24 actual segment Gauss profiles checked. |
| Independent profile quadrature accuracy | Pass as numerical accuracy evidence | A: adaptive tolerance tightening gives no printed change; independent 16/32-point fixed Gauss checks differ by 2.84217e-14 m, below the original absolute 1e-9 m and relative 1e-10 targets. This is not a rigorous bound on systematic bias. |
| Association-strip containment | **Fail** | A: omitted in-domain fraction 5.649175485e-5 > 1e-6; see below. |
| Tail outside the bulk domain | Unmeasured | Small boundary phi is not an independent domain-extension/tail calculation. Nothing outside [-0.5,0.5] m is extrapolated. |
| Initial endpoint particle support and cohesive projection | Pass for initialization | A: positive endpoint mass diagonals 0.006357452016 m²; independently projected C differs by at most 2.27374e-13 Pa. Initial C=313.511760147 Pa and Theta=200 s are uniform to roundoff. This does not establish accepted initial shear-stress uniformity. |
| CPDI constant reproduction / initial particle volumes | Pass in focused scope | C and R ownership logs: no defective stencils; area 0.25000000000000006 m². Serial/MPI2 tests pass, including nonuniform cells and perturbed-domain area. |
| K1.3/K1.4 independent scalar implementation | Available; corrected trajectory comparison unmeasured | Existing `reference.py` and four previously passing reference/analysis tests are reusable. No corrected accepted timestep/loading sequence exists; prospective or ideal-profile roots are not substituted for an actual sequence. Histories must be initialized once and never reset from subsequent ASPECT output. |
| K1.2 accepted initial q uniformity; K1.5 velocity profile, u_y and divergence | Unmeasured on corrected setup | Pre-mechanics phase exports contain no accepted bulk state. Prior nonuniform, incomplete pilot observations cannot serve as a corrected mechanical comparison. |
| K1.6 V/Theta/C/q/I_h/U, cumulative slip, informative state update, coupled residuals | Unmeasured on corrected trajectory | No corrected accepted t=0 or positive-step records. The expected Theta increment must exceed 100 local update tolerances; dimensional coupled residual norms and actual accepted step sequence remain required. |
| Integrated-slip normalization | Unmeasured for an accepted solution | Full-profile versus strip integral is now measured, but is not a measurement of actual bulk-QP integral upsilon or accepted V. Its 1e-4 target does not supersede the separate 1e-6 containment target. |
| Fixed fields, history publication and rollback | Partial reusable infrastructure evidence | Post-ownership particle-domain MPI integration passes; pre-ownership Stage-J/BC/rollback/restart tests remain documented infrastructure evidence. They do not verify the corrected fixed-profile K1 trajectory or its initialized/committed distinction. |
| Periodic particle-domain behavior after motion | Unmeasured for corrected K1 | Initial volume conservation and a periodic bulk mesh do not certify individual moving Voronoi domains across periodic boundaries. No wrapping observation is used as a substitute. |
| K1.7 three spatial and three temporal levels | Unmeasured | No such K1 matrix has been run. One converged initialization is not a discretization-convergence study. |
| K1.7 selected MPI2 K1 and restart/uninterrupted comparison | Unmeasured | Prior Stage-J restart tests and current CPDI MPI tests are useful but do not constitute the requested K1 comparison. |

## The remaining measured failure

The current source/specification intentionally distinguish:

1. Normalization over the solved Q1 phase profile, with adaptive tail termination
   (`current_design.md`, section 20).
2. Finite particle/QP association strips whose widths come from the prescribed
   stationary profile (`current_design.md`, section 17; `specification.tex`,
   particle projection and current influence-width policy).

For the unchanged prescribed core 0.6 and ell=0.15625 m, independent stationary
support is 0.30882159390707253 m on each side. This agrees with the earlier
production support queries; the source, prescribed core and support-construction
inputs are unchanged by the ownership correction. The new audit recomputes that
width independently rather than claiming a new production width export.

Adaptive integration on the corrected central Q1 column gives:

| Integral | Value |
| --- | ---: |
| Full saved-domain I_h | 108.0722382037914 m |
| Inside the association strip | 108.0661330134047 m |
| Outside strip but **inside** the domain | 0.0061051903867 m |
| Omitted fraction | 5.649175485e-5 |
| Existing acceptance target | 1e-6 |

All 24 actual surface quadrature profiles give the same conclusion. The omitted
fraction exceeds the target by about **56.5 times**. Independent fixed Gauss
rules confirm it; quadrature error is far too small to explain the discrepancy.
Production I_h is approximately 108.072238254 m and agrees with the full-profile
integral. Thus this is not evidence for another CPDI ownership defect or for
an inaccurate normalization kernel. The solved FE profile extends beyond the
stationary-profile association strip.

The audit does **not** establish whether refining the current initial-value
problem sufficiently removes this mismatch, or how much is attributable to
the already-documented activation switch versus FE/CPDI error. It does not
silently change strip width, truncate I_h, renormalize slip, modify H or relax
the containment criterion. A change to the production support-selection or
normalization policy would need explicit review; none is proposed as an
automatic benchmark workaround. Resolving this prerequisite is the next gate
decision, before spending on trajectory/convergence runs.

## Work performed, provenance and reproduction

Only the missing saved-data checks were run. There was **no new ASPECT solve**,
build, full suite, mechanical pilot or refinement run. Reused corrected build:
HEAD `f4032b1824ef4892020af8c58ef981aca03dc0ec` plus the recorded working diff,
including the approved ownership change. Executable SHA256 is unchanged:
`5022898bddbc9762595547433056e3d09e8fe06bb9bbb812038d80b3ceca33cd`.
The preserved production/test diff for `particle_domain.cc` and `particles.cc`
has SHA256 `8cbed1ebfa055173232f2e3a60353cd59746a0a5b6c47ff04f16916780cf1f87`.
Input file hashes are recorded in A/audit.json. The earlier initialization cost,
29.71 s and 472872 KiB peak RSS, is reused evidence, not a cost measured this turn.

From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 benchmarks/reconstructed_fault/uniform_shear/audit_gate_k1.py
git diff --check
```

The audit exits **2 intentionally**, identifying the unsatisfied gate. It writes:

- `diagnostics/gate-k1/audit.json`: gate status, geometry, independent integrals,
  containment, endpoint/C projection, numerical cross-checks and input hashes.
- `diagnostics/gate-k1/normal_profile_integrals.csv`: all 24 actual surface
  quadrature locations and full/strip integrals, with units in column names.
- `diagnostics/gate-k1/initial_surface_projection.csv`: production/independent
  nodal I_h and C, plus actual fault coordinates.

The existing C/pre_mechanics_bulk.vtu, C/pre_mechanics_fault.vtu,
C/fault_normals.vtu, C/initial_particles.vtu, C/transverse_phi.csv,
C/centerline.csv and C/phase_and_fault.png remain the corrected ParaView/profile
evidence. They are initialization data only. Accepted t=0/t=2 files from the
old incomplete pilot were lost with its temporary directory, as already
documented; no fields have been fabricated or relabeled.

This turn adds the audit script/results and this report, and updates the
benchmark README/progress links. Production files, fixture parameters and
existing expected outputs are untouched. Nothing was committed.

Because Gate K1 fails, the conditional authorization for K2.1 has not been
activated. There is no K2 perturbation, coupled-response claim or K2.2 refinement
proposal based on an unvalidated K1 setup. No independent per-vertex K1 solves
have been used as a non-local reference. The true-normal-stress branch and
broader campaigns remain unstarted.
