# K5 mechanical work measure: bounded qualification

## Decision

The opt-in work-measure formulation passed the requested fresh four-rank,
free-top, frozen-history coupled qualification. Both boundary completion/source
corrections are retained. Mechanical surface terms now share the actual bulk
source's physical Stokes-QP measure and working FE history; generic property
projection remains unchanged. This is the explicitly authorized discrete
formulation revision, not a claim that legacy volume assembly violated its
specification. The legacy mode remains the default.

The single run took **72.161 s**, including initialization, checks and solve,
under a 600-s cap. Three full Newton updates converged with no line-search
rejections or lower-active nodes. Every returned direction passed the existing
fresh linear-residual check. Intentional exception rollback then verified that
no bulk solution, committed/current V, particle/surface property, particle
position/ID or fault coordinate was changed. No real timestep was committed.

This qualifies the formulation in the stated straight, frozen, mature, 2-D
scope. It does not establish long-term BP3 accuracy, resolve finite-domain
effects, or qualify evolving/curved faults. The next proposed action is a short
mature-RSF replay with this mode; **that replay was not launched**.

## Invariants and equations

On each locally owned physical Stokes QP use the existing source coordinate
and basis b_i, physical FE phase, completed I_h, bulk FE material fields, and
frozen working FE Maxwell stress. Each QP is visited once; the top/bottom wedge
is a coordinate case, not another integration pass. Use current surface Q1 V
and committed Q1 Theta, with constant endpoint extension. In mature mode C=0.

\[
\Delta\tau=2\kappa_b(\dot\epsilon-\chi V_\Gamma S)+\beta\tau^{old}_{FE},\quad
q=\tau_{bg}+\Delta\tau:S,\quad
\sigma_n=\sigma_{bg}+p-\Delta\tau:N,
\]
\[
R_i=\sum_qJ_q\chi_q b_i[q-\mu(V_\Gamma,\Theta_{old})\sigma_n-\eta^dV_\Gamma],
\qquad V_\Gamma=\sum_j b_jV_j.
\]
At fixed bulk unknowns, S:N=0 and
\[
K_{ij}=-R_{i,V_j}=\sum_qJ_q\chi_q b_i b_j
[2\kappa_b\chi_q S:S+\sigma_n\mu_V+\eta^d].
\]
With physical pressure perturbations,
\[
(G\delta x)_i=\sum_qJ_q\chi_q b_i
[2\kappa_b(S+\mu N):\delta\dot\epsilon-\mu\delta p].
\]
The point response supplies these same coefficients to residual/Jacobian
assembly. Existing G machinery samples the actual QP coordinates, not parent
centers. Only G's shear portion is work-adjoint to B. FGMRES, pressure scaling,
active sets, pivoted surface inverse and fresh-residual safeguards are intact.

The new mass is M_ij=sum J chi b_i b_j; m_i=(M1)_i. In 2-D, weak rows change
from Pa*m^2 to Pa*m and M from m^2 to m. R_i/m_i, M^-1R and the consistent
free-space RMS remain in Pa. The fixed K*V characteristic is converted using
this same mass. No nonlinear tolerance, Armijo rule, fraction formula or
lower-bound tolerance was relaxed. Active contact still requires R<=0 and
reaction -R/m>=0. At chi=0 all mechanical terms vanish.

Generic particle-volume property projection, particle ownership, surface
connectivity and history timing are unchanged. The fixed background and
rational shear correction were read from the existing immutable
`mature-fault-50-local4/prestress.txt`, not fitted to the revised measure.

## Fixture and checks

Artifacts are under `benchmarks/reconstructed_fault/bp3/work-measure-free-top-local4/`.
`provenance.json` records command, environment and SHA256 of tested sources,
executable, plugin, fixed prestress and paired completion table.

- Same exact target mesh: 42,880 cells, 1,518,040 total DoFs.
- Same fault: 1,236 vertices, coordinate error zero; 440 free RSF nodes,
  including top, and 796 prescribed deep nodes at Vp=1e-9 m/s.
- Same ell=400 m, paired completed I_h, support and fixed phase.
- Same artificial initialization interval 4e6 s, not elapsed physical time.
- Existing physical boundary constraints, true normal stress and zero initial
  perturbation Maxwell history retained.
- Fresh initial guess: free V increased by 0.1 Vp exp(-xd/1000 m), making top
  V=1.1e-9 m/s. A smooth nonuniform bulk velocity/pressure perturbation is
  distributed with homogeneous constraints; physical base constraints are
  applied separately. Pressure perturbation is nonzero.
- A **private probe vector only** carries manufactured smooth nonzero FE old
  stress for the affine-history/derivative checks. Its old-shear maximum is
  1,359.840 Pa. It never replaces the zero-history inputs to the coupled solve.

The fixed-background initial residual with the actual perturbed fresh guess
is RMS **221,777.556 Pa**, top row **-246,387.176 Pa**. These include the
deliberate perturbation; they are not residuals of an unperturbed Vinit root.

| Production-path check | Measured error/result |
|---|---:|
| Central directional FD, K=-R_V | 1.58422e-9 relative |
| Central directional FD, G=R_x (velocity and pressure) | 5.22659e-9 relative |
| Shear virtual work against constrained B | 1.45062e-14 relative |
| B / direct QP / shear-G work | 7.35729709600e-6 (agree to shown digits) |
| Affine old-FE-stress point equality | 3.72529e-9 Pa |
| Affine shear weak-load equality, divided by m_i | 4.77290e-9 Pa |
| Bulk vs surface kappa/chi coefficient check | 0 |
| Direct row measure vs M1 | 3.12637e-15 relative |
| Visited physical QPs | 385,920 = 42,880*9 |
| Positive-source QPs | 206,549 |
| Associated intact QPs skipped | 552 |
| Positive wedge QPs absent from old finite-profile map | 374 |

The old-stress check is nontrivial, unlike a zero-history-only equality.
Direct integration also checks the affine shear load after MPI reduction.
The common source cache and unchanged chi evaluate the same support as B;
zero-chi QPs add neither mass nor force. No particle-center chi approximation,
new admission threshold, duplicate wedge or periodic surface identification
is introduced. Derivative checks use epsilon=1e-3, with the predeclared 2e-7
action allowance; virtual work allowance is 2e-10.

## Coupled outcome

| Newton base | Bulk residual norm | Free surface RMS (Pa) |
|---:|---:|---:|
| 0 | 1.42653354e9 | 2.21777556e5 |
| 1 | 8.26939019e-1 | 1.96794574e2 |
| 2 | 1.41139457e-4 | 3.24770618e-1 |
| 3 | 7.33067764e-6 | 1.48918642e-6 |

Fixed surface scale: 1.21525979e6 Pa. Final normalized bulk/surface residuals:
**5.13876e-15 / 1.22541e-12**, both below the unchanged 1e-8 criterion.
All three accepted alphas are 1. Final velocity and scaled-continuity residual
components are 7.32638434e-6 and 2.50852977e-7 respectively.

| Linear direction | Krylov iterations | Estimated residual | Fresh residual | Required target |
|---:|---:|---:|---:|---:|
| 0 | 18 | 8.26938983e-1 | 8.26938976e-1 | 1.41630132 |
| 1 | 19 | 1.40945499e-4 | 1.40945531e-4 | 1.62780821e-4 |
| 2 | 19 | 2.32298805e-7 | 2.32298815e-7 | 3.68949071e-7 |
| 3 | 19 | 8.07980029e-13 | 8.07980185e-13 | 2.15727248e-12 |

The existing solver also computes a checked direction at the converged base;
it is not an additional accepted Newton update.

Top V becomes **9.99274867153e-10 m/s**, not prescribed. Free-node rates span
9.99274867e-10 to 1.00045951e-9 m/s; none is lower-active. Deep V remains
exactly Vp. Top m=71.47174035 m and R/m=**3.13758416e-5 Pa**, the maximum
absolute free-row traction residual. Top weak shear and normal traction are
26.54581058 MPa and 50.00003182 MPa. Free-row weak normal traction spans
49.99977548--50.00015176 MPa. These are **work-weighted weak averages**, not
raw pointwise extrema.

`noncommitting_surface.csv` contains the converged trial; `noncommitting_history.csv`
contains restored/retained state. The process exits 1 intentionally after the
convergence marker and exact rollback verification; neither exit code alone
nor the rollback message alone is used as a convergence assertion.

## Changed files and reproducibility

- `include/aspect/reconstructed_fault/surface_system.h`: narrow opt-in method
  and private assembly mode; same semantic solve/action APIs.
- `source/reconstructed_fault/surface_system.cc`: common physical-QP work
  residual, mass, K and G sampling. Default legacy assembly unchanged.
- `benchmarks/reconstructed_fault/bp3/bp3.cc`: scope guards and fresh fixture
  preparation, allowing the top source only with the qualification switch.
- `junction_diagnostic.h`: reuse existing full-state rollback observer at
  timestep zero for this opt-in case.
- `work_measure_diagnostic.h`: production derivative, affine-history,
  virtual-work and coverage checks plus nontrivial initial guess.
- `run_work_measure.py`: single 600-s, four-rank run, immutable output folder,
  provenance and explicit convergence/rollback marker requirements.
- `analyze_work_measure.py`: independent post-run residual/constraint checks
  and `analysis.json`; no simulation launched.
- `current_design.md`, `specification.tex`: authorized opt-in formulation,
  derivatives, units and restrictions; this report and progress entry.

Commands used:

```sh
cmake --build build-pf-cpdi --target aspect.exe.release -j4
cmake --build benchmarks/reconstructed_fault/bp3/build --target bp3.release -j4
mpirun -np 2 build-pf-cpdi/aspect-release --test '[fault_bottom_source],Stage-I*'
python3 benchmarks/reconstructed_fault/bp3/run_work_measure.py
python3 benchmarks/reconstructed_fault/bp3/analyze_work_measure.py
git diff --check
```

Both builds passed. The two-rank focused legacy tests passed **112 assertions
in 12 test cases on each rank**; this includes intentional failure/rollback
coverage. The initial sandbox MPI launch could not initialize and was rerun
outside the sandbox; it was not a failed physical test. The four-rank case
was run once. Analysis and diff whitespace checks passed. Build/unit logs are
`build.log`, `plugin-build.log` and `legacy-tests.log` in the artifact folder,
alongside the coupled evidence and machine-readable analysis.

No restart, timestep, refinement, generic production deployment or mature
trajectory qualification was attempted. The new mode is reconstructible
configuration, not checkpoint payload, and the current benchmark switch
requires a fresh noncommitting run. Unrelated worktree changes were retained.
