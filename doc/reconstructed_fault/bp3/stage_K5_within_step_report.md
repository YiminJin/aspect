# K5: identical-incoming-history, noncommitting state-feedback A/B

## Result

**Within-step nodal state coupling deepens the notch in this controlled test.**
Both four-rank mechanical solves converge with unchanged solver settings and
then roll back completely. With identical incoming bulk/history data, B
increases the 39.90–39.95-km velocity contrast **21.30%** and the predicted
last-element slip-gradient increment **17.13%**. No node reaches the lower
bound. The lagged state is therefore not the sole cause of this notch: replacing
it by the candidate updated state does not repair the undershoot.

This is one step, not a new trajectory, timestep-convergence result, or approval
to change the production split-state algorithm. The preceding half-timestep
study changed entire histories; this comparison deliberately does not.

## Identical starting point and controls

Use `work-replay-50-local4/restart/01`: its independent read-only export verifies
accepted **step 9**, time **483028744.2216855 s = 15.30625726 yr**, on 42880 cells.
The checkpoint clock already describes mechanics **10**, ending at
**922804465.59751701 s = 29.24190894 yr**, with
**dt=439775721.3758315 s = 13.93565168 yr**. This is a real interval; dt0 is not
reinterpreted or evolved. Both cases copy this same checkpoint, perform the
same ordinary advection/FE transfer, and consume the same incoming fields.

Per-rank bit fingerprints of the incoming complete owned bulk vector, surface
properties, committed V, and stable-ID particle properties/positions agree
between A and B. The original and copied checkpoint files retain their hashes.
All persistent fields are checked bitwise again after exception rollback.
No accepted output/history record is written. Pressure, background, mature
friction, mesh, both boundary corrections, fixed phase, full I_h, source and
surface quadrature, prescribed V_p and tolerances remain unchanged.

### Frozen normalization reconstruction, not a production restart change

The revised-work committing fixture is fresh-start-only and does not serialize
its transient completed-I_h cache. Its normal restart recomputation differed
from the stored frozen I_h by **3.33067e-16 relative**, triggering the mature
model's exact current/previous equality invariant in the new preflight.
Two preflight attempts stopped before any mechanical direction was solved;
the second exposed the local exception behind the MPI collective mismatch.
They are preserved as `A-preflight-mpi-failure/` and `A-preflight-frozen-Ih/`.

For these disposable comparisons only, the test accessor restores the exact
saved I_h after requiring that the recomputation agrees within 1e-12 relative.
It does not change the physical input, particle histories, support, integration
tolerances, or production cache/restart policy. The same restoration is used
in both A and B. **This does not qualify general restart of the committing
revised-work fixture.** No production MPI/failure-handling redesign is included.

## State and Jacobian

The [implementation addendum](stage_K5_within_step_addendum.md) gives the
equations. A uses Theta^n. B recomputes each nodal candidate from that same
immutable Theta^n on every evaluation:

\[
x_i=V_i\Delta t/D_c,\quad
T_i(V_i)=\Theta_i^n e^{-x_i}-(D_c/V_i)\operatorname{expm1}(-x_i),\quad
\Theta_q=\sum_i N_i(q)T_i(V_i).
\]

No candidate is written into the manager or advanced repeatedly. The existing
exact aging function and nodal-to-QP interpolation are retained. The derivative
uses the stable Taylor limit for small x. The surface Jacobian includes

\[
K_{ij}=\sum_q J_q\chi_q N_iN_j
\left[2\kappa_b\chi_q S:S+\sigma_n\mu_V+\eta_d
 +\sigma_n\mu_\Theta T'_j\right].
\]

This is generally **nonsymmetric tridiagonal**, because the last term belongs
to the trial column j. The final upper/lower differences are 1.38% and 1.10%
on the two incident elements; symmetrizing would discard a real derivative.
Separate upper/lower entries pass through the existing adjacent-pivoting LU,
active restriction, action and backward-residual checks. G uses the candidate
mu with V fixed; B and the bulk law are unchanged. Outer FGMRES is retained.

## Rates and hypothetical slip increment

V_p=1e-9 m/s. No values below are committed.

| Quantity | A: lagged Theta | B: candidate Theta(V) |
|---|---:|---:|
| V(39.90)/V_p | 0.957303069 | 0.954896631 |
| V(39.95)/V_p | 0.839368550 | 0.811847356 |
| V(40.00)/V_p, prescribed | 1 | 1 |
| (V39.90-V39.95)/V_p | 0.117934519 | 0.143049275 |
| 1-V39.95/V_p | 0.160631450 | 0.188152644 |
| Predicted final-element gradient increment | 0.00141283623 | 0.00165489930 |
| Old gradient + predicted increment | 0.00213375224 | 0.00237581531 |
| Hypothetical slip at 39.90 km (m) | 0.890417308 | 0.889359015 |
| Hypothetical slip at 39.95 km (m) | 0.816116854 | 0.804013700 |

Both start with the same last-element gradient **0.000720916008**. The
increment is exactly `dt*(V_p-V39.95)/50 m`; its 17.13% increase is not a
comparison of different preceding slip histories. The predicted total gradient
increases 11.34%. A reproduces every saved step-10 rate to maximum relative
error **8.65974e-15**.

The incoming states at 39.90/39.95 km are respectively **8,315,646.785** and
**8,993,659.248 s**. A uses these in mechanics. B uses candidate values
**8,377,870.173** and **9,854,069.173 s**, but rollback retains the original
incoming states. At 40 km the supplied steady state remains 8e6 s.

## Last free row: two-element force budget

The last free node is 796 at 39.95 km. Element 796 spans 39.90–39.95 km and
element 795 spans 39.95–40.00 km (file ordering is down dip to up dip).
There are **302/298 actual production QPs** on these two elements, respectively.
For this node their `sum(J chi N_i)` measures are **25.32993409/24.66716037 m**,
identical in both cases. C=0. Radiation damping is only about 0.004 Pa, not a
significant resistance here.

Each mean below is divided by its own element's row measure. The weak load
column is the actual signed integral, in Pa*m; the two means must not be added
without their weights.

| Case | Element (km) | Mean q (MPa) | Mean friction (MPa) | Mean R (kPa) | Weak R (Pa*m) |
|---|---|---:|---:|---:|---:|
| A | 39.90–39.95, free/free | 26.558292 | 26.440601 | +117.691380 | +2,981,114.906671 |
| A | 39.95–40.00, free/prescribed | 26.359359 | 26.480213 | -120.853591 | -2,981,114.906671 |
| B | 39.90–39.95, free/free | 26.587425 | 26.461859 | +125.566196 | +3,180,583.464655 |
| B | 39.95–40.00, free/prescribed | 26.374278 | 26.503218 | -128.939992 | -3,180,583.464655 |

Thus the converged node is **not in pointwise force equilibrium on either
element**. Positive driving from the free/free element balances a deficit on
the element anchored to prescribed V_p. Nodal undershoot coexists with that
weighted cancellation; no bound reaction or tensile locking is involved.
Within-step coupling increases the magnitude of both opposing weak loads by
about 6.69%, and their sum still satisfies the free-row equation.

At A's converged u,p,V, changing only the state to its candidate update produces
**R_B(A)/row_mass = -30.468443 kPa**. Element means become +86.640824 and
-150.724280 kPa (free/free and free/prescribed). This is a direct frictional
resistance increase before any mechanical adjustment and reproduces the earlier
offline lag-versus-updated diagnostic. The coupled B solve then lowers the last
free V and re-equilibrates the bulk and neighbouring nodes.

After re-equilibration, the full row's weak friction mean changes
**26.4601443 -> 26.4822644 MPa (+22.1200 kPa)**; driving shear increases by the
same amount up to negligible damping. This is not the same quantity as the
30.468-kPa frozen-u,p,V state increment.

An additional offline frozen-bulk probe replaces only the last free V by the
linear interpolation of its two neighbouring rates. The resulting row residual
is **-2.39823 MPa in A** and **-2.75844 MPa in B**, recomputing B's state from
the original history. This confirms that simply removing the undershoot is
not a solution at the existing bulk state. It is **not** another coupled solve
and cannot by itself prove which spatial discretization should be changed.

All raw normal tractions on the two elements remain compressive:
**47.208–52.673 MPa (A)** and **46.921–52.962 MPa (B)**. The actual full-row
weighted means are **50.004567/50.006702 MPa**. The state-feedback result is
therefore not explained by tension causing a bound contact.

## Verification and scope

- Nonsymmetric/indefinite surface inverse versus UMFPACK, plus existing symmetric
  cases: **545 assertions in 2 cases**, passing on one rank and on each of two
  ranks. During both four-rank solves, every actual factorization/RHS also uses
  the opt-in pivoted/UMFPACK comparison and existing backward checks.
- Existing exact aging derivative: finite-difference tests at rates
  1e-16, 1e-12, 1e-9 and 1e-8; errors at most 1.00001e-6 for a 1e-3 relative
  increment (expected central-difference truncation). State-friction partials
  for VW, mixed and VS materials agree within 3.36e-9.
- Complete B surface derivative: adjacent-column errors **4.61e-11/3.33e-11**
  for relative step 1e-4; interior control **2.01e-9**. Errors contract about
  100x from step 1e-3. Pressure G error **7.73e-11**. A checks also pass.
- A: **7 fresh-linear checks, 267 Krylov iterations**, final relative
  bulk/surface residuals **2.26587e-13 / 1.87773e-12**, runtime **140.530 s**.
- B: **10 fresh-linear checks, 411 Krylov iterations**, final relative
  bulk/surface residuals **2.27233e-13 / 1.76745e-10**, runtime **181.475 s**.
  Both are below the unchanged nonlinear target 1e-8. No tolerances, Armijo
  budget, active-set rule, pressure treatment or linear iteration limit changed.
- Both finish via the intentional noncommitting exception (process exit 1),
  **after** genuine convergence. Complete rollback and unchanged original/copied
  checkpoints pass; neither writes accepted-step output. Exit status alone is
  not the success criterion.
- Independent raw-QP friction reproduction and incident-element summation
  agree with the reported production weak rows. Nodal state interpolation is
  checked against old-state A and exact candidate-state B at every exported QP.

Implementation changes are limited to diagnostic candidate inputs/response
columns, two friction-law derivative operations, nonsymmetric tridiagonal
storage/solve support, the guarded BP3 diagnostic/rollback wiring, a test-only
normalization restoration accessor, focused tests and benchmark tooling.
The default split update is unchanged. No committing B trajectory, broad suite,
first-event continuation or physical-model change was performed.

## Artifacts and commands

Root: `benchmarks/reconstructed_fault/bp3/within-step-50-local4/`.
Each A/B directory contains `run.log`, copied checkpoint and provenance hashes,
`derivatives.csv`, `incoming_rank*.txt`, `noncommitting_surface.csv`,
`noncommitting_history.csv`, `state_qp_rank*.csv`, `nodes.csv`,
`element_balance.csv`, `execution.json` and `analysis.json`.
Build and one-/two-rank test logs are retained at the root. The two failed
preflights remain in their named directories; no failed evidence is overwritten.

```sh
cmake --build build-pf-cpdi --target aspect.exe.release -j4
cmake --build benchmarks/reconstructed_fault/bp3/build --target bp3.release -j4
build-pf-cpdi/aspect-release --test '[fault_surface_direct]'
mpirun -np 2 build-pf-cpdi/aspect-release --test '[fault_surface_direct]'
python3 benchmarks/reconstructed_fault/bp3/run_within_step.py A
python3 benchmarks/reconstructed_fault/bp3/analyze_within_step.py A
python3 benchmarks/reconstructed_fault/bp3/run_within_step.py B
python3 benchmarks/reconstructed_fault/bp3/analyze_within_step.py B
```

Runners intentionally refuse to overwrite existing attempts. The next decision
is not to adopt candidate-state mechanics as a notch repair: it deepens this
notch. Temporal qualification remains necessary, and the spatial Q1/mixed
free-prescribed balance remains an independent question. This bounded test
does not establish either a temporal limit or a preferred production integrator.
