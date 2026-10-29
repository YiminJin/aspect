# Feng direction/dissipation reference evidence

Start with `doc/reconstructed_fault/particle_free/FENG_DIRECTION_REPORT.md`
and `FENG_DIRECTION_FORMULATION.md` in the repository-relative ZIP layout.
Decision: remove the old conjugate-redistribution interpretation, but **do not
proceed to propagation**. Pressure-coupled mechanical localization remains.
No production native source, BP3 file, parameter or checkpoint is changed.

Reproduce from the repository root (Python, NumPy, SciPy, Matplotlib versions
are in environment.json):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/bulk_cohesive/check_feng_direction.py
MPLCONFIGDIR=/tmp/feng-matplotlib OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/bulk_cohesive/plot_feng_direction.py
```

The test reads only the included archived-point-inputs.csv; it does not need
the original build tree. Commands overwrite only this reference's generated
CSV/JSON/PNG files. For bounded logged execution in the worktree, the retained
`refactoring_boundary/run_logged.py LABEL TIMEOUT COMMAND ...` helper was used;
use a new label since it refuses to overwrite a previous log. Final logs and
elapsed times are included. No native solve, propagation, MPI or server test
is concealed in these timings.

Data dictionary:

| Files | Meaning / conventions |
|---|---|
| spin-controls.csv, observer-spin-curve.csv | Lij=partial_j v_i, CCW spin (L21-L12)/2, s^-1; intended and literal legacy behavior distinguished. Spin agreement is diagnostic only. |
| continuity-path.csv, tie-versus-dominance.csv | Candidate and selected normal angles counterclockwise from physical +x, modulo 180°. Plane tangent is normal+90°. Candidate families are followed under stress rotation, not enumeration order. |
| direction-objective.csv | F in Pa vs normal angle; actual increment rule including min-positive normal pressure. |
| basis-covariance.csv | Full tensor/director-projector errors under constant coordinate-basis rotations, not finite-rotation evolution. |
| mature-power.csv | Actual symmetric flow W components s^-1; equivalent rate=sqrt(2) norm(W). Work and dissipation W/m³. The two-system mixture is a deliberately incompatible control. |
| redistribution.json | Old diag(e,-e,0) redistribution projected onto the selected single-flow ray; full mixed coefficient at the old 45° wave normal. |
| archived-point-inputs.csv/json | 17 original inputs plus native comparison derivatives; origin/hash recorded. T, tau, Q, p in Pa; A dimensionless, beta dimensionless, dt seconds. Derivative rows d0,d1,d2 are response/G to xx, yy, engineering xy strain; d3 is response/G to dp/G. No histories interpolated. |
| point-symbols.csv | Each independently solved point: global force margin Pa, tangent/stress differences, mixed/fixed-pressure minimum divided by G, phase-obstacle residual Pa. Different path rows have their own histories; forward-probe dt rows share one incoming state. |
| angular-symbol.csv, mixed-null-modes.csv | Perturbation **wave-normal** angles (not flow or crack angles), from +x. Full incompressible pressure elimination and null residual. Pressure amplitude normalized by G for unit transverse strain amplitude. |
| phase-range-symbol.csv | Analytic rate limit, both slip signs, h(phi); no evolving phase trajectory. phi=0 is elastic control; phi=1 is exact mature limit. |
| global-return-controls.csv | Both regular single-return roots. `point_accepted` describes the whole point; `candidate_continuity_ok` describes that row. Individual plane F=0 is insufficient; compare final global margin to allowance. |
| point-work.csv | Energy densities J/m³: storage change, Maxwell loss, actual friction, separate BE remainder, ledger defect. No claim of physical temporal accuracy. |
| *checks.json, decision.json | Numerical errors, thresholds and blocked progression; expected scientific counterexamples are checked, not suppressed. |
| parameters.json, sources.json, entry.json, preservation.json | Units, source provenance, entry revision/local protected hashes and final preservation checks. |

Tensor contractions include both symmetric xy terms and embedded zz. The flow
is in-plane trace-free; stress/mechanical derivatives are full plane strain.
No direction is a phase-front angle. There is no spatial averaging kernel,
point interpolation, phase mask or specimen trajectory in these data.

The PNGs plot only these CSVs/reference calculations. Candidate directions use
an imposed mature stress and supplied orientation preference. The mechanical
figure separates changing-direction mixed response, inadequate fixed-pressure
response, the dt limit and actual-power mismatch. No paper figures are copied.

The archive omits the downloaded manuscript and full solver archives. The
primary manuscript URL/DOI/hash, historical source excerpt and deal.II NaN
initializer provenance suffice to audit the source claims. Previous scientific
evidence and all 79 protected source/BP3/build/archive files match entry hashes;
the three index documents retain their complete original contents as tails.
