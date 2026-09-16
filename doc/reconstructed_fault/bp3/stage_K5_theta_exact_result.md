# K5: exact 50-m Theta comparison and one conditional mechanics-13 solve

## Decision

The exact alternative friction load **does reverse the frozen bound residual**
at 39.95 km. The authorized single noncommitting solve was therefore performed.
After coupled re-equilibration the node **remains lower-active**, with a reaction
of **0.090747 MPa**, down from **1.098618 MPa** (91.74% smaller). The adjacent
39.90-km rate increases by 61.76%. The raw tensile normal-stress minimum remains
essentially unchanged: **−10.711552 → −10.641551 MPa**.

Thus nodal aging followed by interpolation materially affects the rate
depression and complementarity margin, but this comparison does **not** show
that changing the state representation removes contact or the junction stress
concentration. No production history scheme is replaced, no state is committed,
and no trajectory is continued.

Recommended next action: review the state-discretization choice using these
numbers before authorizing a persistent alternative state representation.
Do not adopt the diagnostic as a demonstrated repair of the junction anomaly.
A consistently evolved alternative trajectory remains untested; this task is
the requested one-update counterfactual only.

## 1. Exactly what was compared

Only the saved 50-m case is used: 42,880 bulk cells, 1,236 fault vertices,
four MPI ranks. Segments 795 and 796 span 39.95–40.00 and 39.90–39.95 km.
Node 795 at 40 km remains prescribed; node 796 at 39.95 km is the contact node.

Define the unchanged exact aging update

\[
U(T,v,d)=T e^{-vd/D_c}-(D_c/v)\operatorname{expm1}(-vd/D_c),
\quad D_c=0.008\ {\rm m}.
\]

At every production domain quadrature coordinate on these two segments:

\[
\Theta^A_{12}(\xi)=\sum_j N_j(\xi)\Theta_{j,12},\qquad
\Theta^B_{12}(\xi)=U\left(\sum_jN_j\Theta_{j,11},
                              \sum_jN_jV_{j,12},\Delta t_{12}\right).
\]

The saved nodal Theta12 itself is verified against U(Theta11,V12,dt12)
at **all 1,236 vertices**, to the existing relative 1e-12 tolerance.
The nodal preceding states and rates used on the patch are:

| Node / down-dip distance | Theta11 (s) | accepted V12 (m/s) | retained Theta12 (s) |
|---|---:|---:|---:|
| 795 / 40.00 km | 8.000000e6 | 1.000000e-9 | 8.000000e6 |
| 796 / 39.95 km | 1.7716204075491014e8 | 1.000000e-20 | 5.798573048667023e8 |
| 797 / 39.90 km | 2.371493895333271e7 | 2.684054936716150e-10 | 2.9805640984495442e7 |

The clock is explicit:

- t12 = 2,165,471,357.7276783 s; dt12 = 402,695,264.30232239 s.
- mechanics13: t13 = 2,232,176,379.2516127 s; dt13 = 66,705,021.523934364 s.
- Both A and B use **Theta12 in mechanics13**. B never uses trial V13 in
  the aging update. Its state is fixed during all residual/Jacobian evaluations.
- Outside the two segments, original Theta12 is unchanged. B agrees with A
  at the vertices but is not Q1 between them. It is evaluated directly, not
  projected back into the same nodal Q1 space.

This is a deliberately alternative discrete state representation. The
authoritative `current_design.md` section 25 and `specification.tex` nodal
aging rule still describe A; no implementation/specification violation is
claimed. Friction remains the regularized asinh law, with a=.025, b=.015,
mu0=.6 and V0=1e-6 m/s on the complete selected patch.

## 2. Exact production-quadrature result

The comparison covers **9,774 production points from 731 admitted real parents**:
4,884 on segment 795 and 4,890 on segment 796. The respective summed domain
weights are 79,077.897117 and 79,188.909028 m². All are owned by rank 2;
the other ranks contribute empty patch sample lists, without duplication.
No selected-extrema reconstruction or lower-order quadrature is used.

The original accepted Stokes polynomial is extracted read-only from slot02.
The step-12 checkpoint slot01 supplies the ordinary mechanics-13 advection,
particle history, FE transfer and property preparation. Only the saved Stokes
components and V13 are substituted privately for the frozen comparison.
The original weak friction and unreplaced residual are reproduced over **all
fault rows** with maximum physical-term-scaled error **1.14945e-16**.
This independently validates that the correct saved mechanical state is used.

For the same u,p,V13 and retained particle history, B changes only friction.
Every exported coordinate, weight, rate, pressure, normal/shear traction,
cohesion, phase and localization value matches A bitwise.
Independent stable aging/asinh evaluation reproduces the exported mu with
maximum absolute errors 1.84e-15 (A) and 1.95e-15 (B).
ThetaA/ThetaB at the actual points ranges from 1.00317 to 18.37404.

Let m_i=sum(w N_i); reported MPa values below are **weak load divided by m_i**,
not nodal collocation values or bulk-column means. The sign convention is
R=q−C−mu sigma_n−damping.

| Node | friction A / m (MPa) | friction B / m (MPa) | change (MPa) | R_A / m (MPa) | R_B / m (MPa) |
|---|---:|---:|---:|---:|---:|
| 795, prescribed | 27.320020 | 26.420977 | −0.899043 | −5.555210 | −4.656168 |
| **796, lower contact** | **26.564084** | **25.248103** | **−1.315981** | **−1.098618** | **+0.217364** |
| 797, free | 25.938623 | 25.468360 | −0.470262 | 1.12e-11 | +0.470262 |

For node 796, m=79,127.43054411304 m². Its original and alternative friction
loads are 2.101947722661095e12 and 1.997817505919892e12 N; the exact change is
−1.041302167412034e11 N. Its residual changes from −8.693078054563963e10 to
+1.719943619557517e10 N. This satisfies the explicit conditional-solve gate.
The previous estimated −1.31704-MPa change was close, but is superseded by
this exact −1.31598-MPa result. Other rows outside 795–797 are unchanged.

## 3. One coupled solve: re-equilibration matters

The single disposable solve starts from the ordinary step-12 checkpoint
predictor, with the same step13 timestep, geometry, support, bulk/history
transfer, true pressure, background traction, coefficients and tolerances.
The imported accepted step13 field is used only for the preceding comparison;
it is not published or used to reset any history. Both shear and normal
mechanics retain the original particle-history evaluation.

The central pointwise diagnostic state is shared by residual, K_V and G;
the derivative holds ThetaB12 fixed. Before starting the solve:

- K finite differences at h=1e-16 and 1e-17 m/s have relative errors
  2.28e-7 and 5.68e-7 (required <1e-4).
- The physical-pressure G difference at 100 Pa has relative error
  5.43e-10 (required <1e-7).

| Down-dip distance | Original V13 (m/s) | Diagnostic V13 (m/s) | Result |
|---|---:|---:|---|
| 40.00 km | 1.000000e-9 | 1.000000e-9 | prescribed, unchanged |
| **39.95 km** | **1.000000e-20** | **1.000000e-20** | **lower-active in both** |
| 39.90 km | 1.801875e-10 | 2.914775e-10 | +61.76%, free |
| 39.85 km | 3.931010e-10 | 3.572113e-10 | −9.13%, free |
| 39.80 km | 4.547802e-10 | 4.631491e-10 | +1.84%, free |
| 39.75 km | 5.459157e-10 | 5.418046e-10 | −0.75%, free |

There is one lower-active node globally in both states. At node796:

| Weak contribution / m (MPa) | Original A | Re-equilibrated B |
|---|---:|---:|
| shear driving q | 27.851904 | 27.849081 |
| cohesive resistance C | 2.386438 | 2.389495 |
| friction resistance | 26.564084 | 25.550332 |
| physical R | −1.098618 | −0.090747 |
| lower-bound reaction −R | 1.098618 | 0.090747 |

Damping is 0.0009094 → 0.0009952 **Pa**, negligible on this MPa scale.
Frozen B initially reduced friction enough to reverse the residual. The
neighbor's larger solved rate subsequently increases this row's Q1-integrated
friction by **0.302230 MPa** relative to frozen B. Together with the shear
change (−0.002823 MPa) and cohesion change (+0.003058 MPa), this takes R from
+0.217364 back to **−0.090747 MPa**. A sign reversal with other unknowns fixed
therefore did not guarantee release in the coupled solution.

## 4. Raw pressure and normal stress, not a changed plotting convention

These ranges use every actual constitutive point on the same two elements.

| Raw quantity (MPa) | Original A range | Re-equilibrated B range |
|---|---:|---:|
| perturbation pressure | [−53.030290, 51.964526] | [−52.990083, 51.916675] |
| perturbation tau:N | [−15.018914, 16.654400] | [−15.003535, 16.653722] |
| total sigma_n | [−10.711552, 109.378260] | [−10.641551, 109.302597] |

The same sample remains most tensile: parent250724,
(x,y)=(59040.862847417135,65274.418576205855) m; segment795,
xi=.04652615775470622, projected xd=39997.67369211222 m (free-side quadrature).
Its decomposition is:

- Original: 50 − 52.697742 − 8.013809 = **−10.711552 MPa**.
- Diagnostic: 50 − 52.655769 − 7.985782 = **−10.641551 MPa**.

The point pairs have maximum absolute changes 0.079003 MPa in p,
0.068740 MPa in tau:N, and 0.135379 MPa in sigma_n. The tensile quadrature
weight changes from 5,862.1970 to 5,597.2866 m² (3.7040% → 3.5366% of the
two-element measure). Tension is not eliminated. The contact row's weak
mean sigma_n is compressive: 50.188572 → 50.188964 MPa. Neither the pointwise
tensile sign nor the plot of pressure alone determines its bound reaction.

## 5. Verification, scope and recoverable artifacts

The diagnostic reaches final relative bulk/surface residuals
**4.13181e-14 / 6.77444e-11**, below the unchanged 1e-8 target.
Dimensional residual norms are 1.17475e-4 (bulk assembled-vector norm) and
9.20670e-5 Pa (surface strong-residual RMS). Ten reported fresh-linear checks
all satisfy their requested tolerances; the largest fresh/target ratio is
0.88554. Final reported estimated/fresh linear residuals are
5.8618868e-11 / 5.8618929e-11 against target1.2881462e-10.
All four line searches accept alpha1 without rejection.

The deliberate exception occurs **after convergence and before publication**.
Rollback verifies the entire bulk vector, current/committed V, every particle
and surface property, particle IDs/positions and fault coordinates unchanged.
Retained Theta equals saved Theta12 and committed V equals saved V12 exactly.
Original and copied checkpoint hashes are unchanged; no accepted-step output
or new checkpoint is written. Exit1 is the intentional rollback stop, not
evidence of convergence; the numerical and rollback markers above provide it.

Exact commands, from the repository root:

```sh
cmake --build build-pf-cpdi --target aspect.exe.release -j4
cmake --build benchmarks/reconstructed_fault/bp3/build --target bp3.release -j4
python3 benchmarks/reconstructed_fault/bp3/run_theta_exact.py export
python3 benchmarks/reconstructed_fault/bp3/run_theta_exact.py verify-export
python3 benchmarks/reconstructed_fault/bp3/run_theta_exact.py conditional
python3 benchmarks/reconstructed_fault/bp3/analyze_theta_exact.py
git diff --check
```

Both builds, offline assertions and diff check pass. Execution consists of
one 4.56-s read-only extraction and one 125.64-s conditional process (including
17.81-s cold I_h preparation). No other ASPECT solve, one-/two-rank campaign,
or full test suite was run. The complete sampled comparison and solve use
four ranks.

The export runner initially expected the accepted-state number in checkpoint
metadata. ASPECT actually advances the clock **before** writing its snapshot:
slot02 contains accepted solution13 but clock14 at t=2359582970.3623271,
dt=127406591.11071464. Subtracting dt recovers accepted t13. This assertion
was corrected offline, without rerunning extraction; the first failed
verification record is retained, alongside `verification.json` confirming
42,880 cells and 943,360 exact Stokes coefficients. The all-row A load match
provides the additional independent state check.

All new evidence is under
`benchmarks/reconstructed_fault/bp3/theta-exact-local4/`:

- `export/`: exact saved Stokes polynomial, metadata, frozen update inputs,
  expected saved weak loads, original/ corrected verification records.
- `conditional/run.log`, `execution.json`, `provenance.json`: numerical checks,
  exact command/environment, source/binary/plugin and checkpoint hashes.
- `conditional/frozen_A_rank*.csv`, `frozen_B_rank*.csv`, `solve_rank*.csv`:
  complete raw production quadrature samples.
- `conditional/analysis_exact_load.csv`, `analysis_complete_state_samples.csv`,
  `analysis_neighbouring_nodes.csv`, `analysis_raw_extrema.csv`, `analysis.json`:
  independently checked loads, states, rates and stress comparison.
- `conditional/derivative_checks.csv`, `noncommitting_surface.csv`,
  `noncommitting_history.csv`: derivative checks, final mechanical result and
  restored histories. `history_surface_rank*.csv` contains the full weak terms.

Diagnostic-only code additions are the frozen point-state opt-in in
`source/material_model/phase_field_fault.cc`, complete selected-segment sample
export in `source/reconstructed_fault/surface_system.cc`, and BP3-local
`theta_exact_diagnostic.h`, runner and analysis script. Existing BP3 checkpoint
export and rollback hooks are reused/extended for step13. No public API,
ordinary graphical output, friction law, production state update, support,
pressure convention or solver criterion changes. The opt-in override requires
the noncommitting guard and checks the exact step. Do not enable these flags
in a production trajectory. Existing unrelated working-tree changes are
preserved, and no commit has been made.
