# Projected-material and weak Q1 state initialization

## Result

The sharp initial velocity pulses disappear at their previous scale. With the
same uniform effective background, the 13–20 km velocity range changes from
`V/Vp = 0.957455–1.142298` to **0.999072–0.999963**. The native row-weighted RMS
departure from Vp decreases from **1.88247% to 0.0424138%** (44.4-fold reduction).
Small mesh-scale variations remain; this is not an exactly uniform velocity
solution or a qualification of subsequent evolution.

One fresh four-rank Release initialization was run, stopping at accepted step
zero. No real timestep was taken. The comparison baseline is the preserved
`dc010-ell100/clean-background-init` result, not the older inherited-prestress
case. No prestress correction, mechanical equation, state representation,
aging law, mesh, phase, support, I_h, loading or mechanical tolerance changed.

## Initialization procedure

This is explicitly selected by a separate benchmark library,
`build/libbp5_initialization.release.so`. The ordinary BP3 plugin and inputs
retain their existing behavior.

1. Complete the existing surface material projection and normalization setup.
2. At each fault vertex, read those projected chemical fields, use the same
   composition-fraction utility as the material model, and invert the **configured
   friction law** for Vp and the uniform target coefficient. The normal background
   remains 50 MPa; nominal shear remains 26,546,122.365139291 Pa, including damping.
3. Audit friction using actual owned Stokes QPs, the production source map,
   continued endpoint wedges, FE phase, completed I_h and work weights. The state
   at a QP is still the ordinary linear interpolation of nodal **Theta**, not of
   log Theta. The projected-nodal inverse leaves a maximum weak friction excess
   of **9,687.09 Pa**. This is significant despite its accurate nodal inversion.
4. Solve once for positive nodal state so that every native weak row balances
   nominal friction:

   `F_i = sum_q JxW_q chi_q N_i [50MPa mu(Vp, sum_j N_j Theta_j, f_Gamma(q))
                               + eta_d Vp - tau_nom] = 0`.

   Newton uses `z_i=log(Theta_i)` solely to enforce nodal positivity. Its derivative
   is `D_ij = sum_q JxW chi N_i 50MPa mu_Theta N_j Theta_j`. The sparse tridiagonal
   Jacobian is solved with existing UMFPACK infrastructure; backtracking decreases
   the maximum row-normalized residual. Only the converged positive field is
   published. The initialization target is 1e-5 Pa, separate from and much tighter
   than the unchanged mechanical acceptance criterion.
5. Preserve a copy of this initialized field for the timestep-zero retention
   audit. Later split aging and once-per-accepted-step publication are unchanged.
   Particle initial-composition values remain bootstrap inputs; the initialized
   fault state is the constitutive state actually consumed by mechanics.

Two initialization updates reduced the maximum weak excess to **1.22387e-7 Pa**.
All 1156 nodal states are positive: 25,073.7495–100,210,774.546 s. Relative to the
projected-material nodal inverse, the weak correction ranges from **−0.6700% to
+0.2108%**. It does not enforce monotonicity; small nodal overshoots are retained.

## Independent production-quadrature audit

The offline audit reconstructs loads from the accepted production QP exports,
not from the initialization's cached samples. It uses the actual projected
composition, retained state, work weights, and incoming/accepted stress timing.

| Check | Result |
|---|---:|
| Projected-nodal inverse error at vertices | <1e-7 Pa |
| Initialization log-state Jacobian directional relative error | 1.80875e-10 |
| Initialization versus production row-measure relative error, all rows | 4.44e-16 |
| Recomputed projected-nodal weak excess, transition | 9,687.086646 Pa |
| Recomputed final weak excess, transition | 1.23471e-7 Pa |
| Independent versus initializer weak-row discrepancy | <3.51e-9 Pa |
| Accepted friction-load reconstruction error | 1.38279e-7 Pa |
| Accepted shear-load reconstruction error | 1.19223e-7 Pa |
| Accepted unreplaced residual reconstruction error | 3.32183e-8 Pa |

**Weak equilibrium is not pointwise equilibrium.** At Vp, final QP friction
excesses still range from **−10.062 to +5.693 kPa** in the transition; their native
nodal weak loads cancel. The projected-nodal field alone had pointwise errors
−5.531 to +14.889 kPa. This distinction is intentional and required by the
retained Q1 state representation, rather than concealed by a nodal-only audit.

Surface composition, I_h, background arrays and geometry match the clean-background
baseline bitwise. The final initialized Theta equals the incoming and outgoing
accepted-step-zero state exactly. Retained particle Maxwell stress remains zero.

## Mechanical comparison

Tractions below are native weak row averages, not centerline samples. Delta shear
is current bulk shear relative to the unchanged nominal background.

| Quantity | Old initial state | Weak initial state |
|---|---:|---:|
| V/Vp at 15 km | 1.142298 | 0.999227 |
| V/Vp at 18 km | 0.957455 | 0.999958 |
| Largest absolute V/Vp−1, 13–20 km | 14.2298% | 0.0928212% |
| RMS V/Vp−1, 13–20 km | 1.88247% | 0.0424138% |
| Delta shear at 15 km | −40,522.4 Pa | −137.811 Pa |
| Delta shear at 18 km | +10,997.9 Pa | −138.291 Pa |

The new weak normal traction differs from 50 MPa by approximately +13.89 Pa
at 15 km and +13.24 Pa at 18 km. Throughout 13–20 km the bulk shear contribution
is −203.9 to −110.1 Pa, while normal feedback into friction is +7.01 to +7.68 Pa.
The small remaining velocity departure accompanies these finite-mesh mechanical
perturbations; no additional prestress adjustment was made to cancel them.
Globally V/Vp lies between 0.998991 and 0.999972; the minimum is at 1.1 km,
not at a transition endpoint.

The accepted mechanical surface strong RMS is **0.0248829 Pa**. Final normalized
residuals are **6.915805e-10 bulk**, **9.242475e-9 surface**, both within the
unchanged criterion. The two returned linear directions passed fresh checks:
0.8618753 versus target 1.015455, and 1.039285e-8 versus target 2.222191e-8.
There were 59 total Krylov iterations, one accepted Newton update, minimum
alpha=1, 1156 free nodes, zero lower-active nodes, and zero Theta retention error.

## Scope, files and reproducibility

New benchmark files: `CMakeLists.txt`, `weak_initialization.h`,
`run_weak_initialization.py`, `analyze_weak_initialization.py`, and this report.
The shared `bp3.cc` has three compile-time guarded integration points: include,
initialization call, and the timestep-zero expected-state audit. The default
build is unchanged. The existing offline analyzer gains an optional label so
the deliberately adjusted state is not misleadingly called pure interpolation
error. README documents the separately selected plugin and commands.

The old physical-nodal-inverse observer is not enabled for this run: its expected
state is precisely the initialization being replaced. Its replacement checks
projected nodal inversion, the assembled derivative, native weak loads and
retention of the resulting initial field. No mechanical acceptance check is
disabled. No subsequent aging, restart or multi-step experiment was run.

Build: `cmake --build benchmarks/reconstructed_fault/bp5/build --target bp5_initialization -j4`
passed. Run and analysis:

```sh
python3 benchmarks/reconstructed_fault/bp5/run_weak_initialization.py prepare
python3 benchmarks/reconstructed_fault/bp5/run_weak_initialization.py run
python3 benchmarks/reconstructed_fault/bp5/analyze_weak_initialization.py
```

Simulation wall time: **120.31 s**. Peak child RSS recorded by the launcher:
**2,680,296 KiB** (~2.56 GiB), not a sum of simultaneous MPI memory. Source,
plugin and input hashes are in `dc010-ell100/weak-state-init/launch.json`.
The previous source/plugin are preserved in `dc010-ell100/weak-init-source-before/`.

Evidence:

- `dc010-ell100/weak_initialization_comparison.png`: velocity, state, weak mismatch
  and mechanical-traction comparison.
- `dc010-ell100/weak_initialization_comparison.json`: numerical checks.
- `dc010-ell100/weak-state-init/weak_initialization.csv`: original, projected-nodal
  and final state, full work row measures and both initial weak residuals.
- `initial_friction_qp_audit.csv`, `initial_friction_weak_audit.csv`,
  `state_work_0.csv`, `work_weak_0.csv`, ordinary graphical outputs and `run.log`
  in the same case directory preserve point/weak measures and lifecycle timing.

The result supports initialization mismatch as the cause of the sharp initial
pulses in this configuration. It does not establish later-time stability or
resolve the separately recorded real-step lower-bound issue.
