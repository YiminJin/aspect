# Frozen mechanical response: signed velocity-gradient decomposition

## Result

The 200-m alternating input is accommodated predominantly by the
**along-fault gradient of normal velocity**, not by a tangential velocity
difference across the diffuse band. The normal-gradient-of-tangential-velocity
contribution to its weighted relaxation is small and **negative**. Refining
the bulk mesh strengthens the positive normal-velocity contribution more
than the negative contribution, increasing net relaxation. This explains
the direction of the stiffness change in the preceding frozen refinement.

All six signed sums reproduce the previously reported relaxation to within
4.82e-15 relative error. Repeated net mechanical coefficients equal the saved
values at exported precision. No loading prefix, history evolution, physical
profile regeneration, new mesh level, or production algorithm change was
performed. This identifies the accommodation mechanism of these frozen
linear modes; it does not by itself prove the cause or growth rate of the
later state-evolving oscillation.

## Unchanged problem and sign convention

Reuse the two meshes and all three Q1 directions from
[the frozen bulk-refinement report](stage_K5_frozen_bulk_refinement.md):
97.65625 and 48.828125 m local bulk cells, 100-m fault grid, ell=400 m,
identical prolonged physical Q1 phase, saved full I_h and coefficients.
The artificial initialization interval and ordinary timestep controller
are unchanged. Both probes stop in initialization before publication.

The production tangent points **up dip**,
t=(0.5,sqrt(3)/2), with n=(-sqrt(3)/2,0.5). Thus s increases up dip whereas
the plots' x_d increases down dip: d/ds=-d/dx_d. With
S=sym(t tensor n), the shear identity is

    2 kappa S:epsilon(du) = kappa (d_n du_s + d_s du_n).

Solve the same private, homogeneously constrained A dx=B dV. At each actual
production Stokes QP use w=JxW*chi, the existing source/surface basis and
the same consistent work mass m=dV^T M dV. The reported coefficients are

    k_sn = sum_q w_q dV_q kappa_q d_n du_s(q) / m,
    k_ns = sum_q w_q dV_q kappa_q d_s du_n(q) / m,
    k_relax = k_sn + k_ns.

Each locally owned QP contributes once and the signed sums are MPI-reduced.
These are the two terms in bulk **relaxation**, not the two terms in net
restoring stiffness. As before, k_shear=k_direct-k_relax. Instantaneous
friction, pressure feedback, deviatoric-normal friction feedback and damping
are not folded into this table.

The native uniform kappa is 1.281524811978847e17 Pa s. To compare with the
existing step-11 response table, both meshes' coefficients are multiplied
by the previously qualified factor 38.33411519295163, giving the same
kappa=4.912611976502280e18 Pa s. This is an exact uniform-coefficient scaling
of the mechanical derivatives, not a new timestep or history solve.
Velocity responses themselves do not require this scaling.

## Signed production-work contributions

Values are in **1e15 Pa/(m/s)** at that common coefficient.

| Mode | Bulk mesh | k_sn: d_n u_s | k_ns: d_s u_n | Sum: k_relax |
|---|---|---:|---:|---:|
| Broad 3 km | 97.65625 m | +5.135994 | +0.419828 | 5.555822 |
| Broad 3 km | 48.828125 m | +5.136232 | +0.419780 | 5.556012 |
| Six-node 600 m | 97.65625 m | -0.409132 | +5.704545 | 5.295412 |
| Six-node 600 m | 48.828125 m | -0.409778 | +5.716008 | 5.306229 |
| Alternating 200 m | 97.65625 m | -0.096597 | +7.343349 | 7.246752 |
| Alternating 200 m | 48.828125 m | -0.151389 | +7.810570 | 7.659181 |

The broad mode is dominated by d_n u_s. The 600-m mode already uses the
other mechanism, with substantial signed cancellation. For 200 m,
refinement changes the two contributions by **-0.054792** and **+0.467221**,
respectively; the net relaxation increase is **+0.412429**. Taking only
magnitudes would conceal this opposing contribution. The associated net
shear stiffness remains the prior 1.162967 -> 0.746124 result, not a new
traction definition.

## Velocity fields and common-window kinematics

Full Q2 response polynomials were exported in double precision from owned
bulk cells, including intact/unassociated cells. These exports retain nine
tensor-product values per velocity component per cell; no averaging or
graphical Float32 interpolation is used.

Use the identical **[-1200,1200] m normal window** at every section and on
both meshes. The captured coarse profile's nonzero-cell normal extent in
the probe patch is at most 923.222 m; exact prolongation preserves that
support. The window therefore contains the entire localization profile.
The velocity response itself need not be zero outside the profile.

Define

    D(s) = du_s(s,+1200)-du_s(s,-1200),
    N(s) = d/ds integral_{-1200}^{1200} du_n(s,n) dn.

Then D+N=integral 2 S:epsilon(du) dn. **D is a cross-band velocity
difference, not automatically a sharp-interface slip rate.** These are
unweighted kinematic quantities; they must not be confused with the
chi-weighted, kappa-weighted work coefficients above.

Each normal ray is partitioned at actual bulk-cell boundaries. Three-point
Gauss integration is exact for the exported Cartesian Q2 polynomial on
each inclined interval. We independently verify D=integral d_n du_s and
N=d_s integral du_n by centered differences of the integrated normal
velocity, including the up-dip/down-dip sign conversion.

At x_d=16.5 km, with input dV=1e-12 m/s:

| Mesh, 200-m mode | D (m/s) | N (m/s) | D+N (m/s) |
|---|---:|---:|---:|
| Coarse | +2.848312e-18 | +7.683448e-13 | +7.683477e-13 |
| Fine | -2.397659e-18 | +8.825839e-13 | +8.825815e-13 |

For a less point-specific comparison, project each unweighted column
quantity on the **actual interpolated Q1 input** over 15–18 km:
P(f)=integral dV*f dx_d / integral dV^2 dx_d. The output sampling is 10 m;
these are diagnostic profile projections, not production work quadrature.

| Mode | Mesh | P(D) | P(N) | P(D+N) |
|---|---|---:|---:|---:|
| Broad | Coarse | 0.148491 | 0.311172 | 0.459663 |
| Broad | Fine | 0.148506 | 0.311221 | 0.459727 |
| 600 m | Coarse | -0.000318689 | 0.996724 | 0.996405 |
| 600 m | Fine | -0.000317946 | 0.998288 | 0.997970 |
| 200 m | Coarse | -2.58566e-8 | 0.920682 | 0.920682 |
| 200 m | Fine | -4.28955e-9 | 0.980622 | 0.980622 |

The alternating input produces spatially oscillating normal motion inside
the band and much smaller tangential motion. The latter changes sign
across the center and decays towards the window edges. Consequently the
cross-band tangential difference is nearly zero, while the tangential
variation of integrated normal motion supplies almost all integrated shear.
The finer bulk space resolves this pattern more effectively. This is not
evidence of an opening discontinuity: the exported velocity remains a
continuous bulk FE field.

The broad mode's unweighted finite-window projection need not equal one:
its elastic velocity response extends outside the localization window.
Nor does an almost-zero D imply zero weighted k_sn: the interior gradients
can cancel in their unweighted integral but not under chi*dV weighting.

Plots and numeric profiles, relative to the repository root:

- `benchmarks/reconstructed_fault/bp3/first_long_run/mechanical-velocity-decomposition/alternating_velocity_band.png`: u_s and u_n through/along the band, matched color limits across meshes.
- `.../alternating_velocity_cuts.png` and `.csv`: transverse cuts at 16.50/16.55 km and along-fault cuts at n=0/400/1200 m.
- `.../integrated_shear_components.png`: D, N, their sum and the input Q1 dV for all modes/meshes.
- `.../normal_columns.csv`: all column integrals and independent derivative checks.
- `.../decomposition.json`: full-precision signed coefficients, checks and execution data.

## Verification and cost

- All six signed work closures: relative error <=4.82e-15.
- All repeated net mechanical shear coefficients: zero difference at saved precision.
- Fresh A-solve relative residuals <=8.325e-11; 16–18 iterations.
- Existing work pairing, full action and constitutive finite-difference checks pass unchanged; production solution, linearization vector and manager V remain unchanged.
- Fine profile prolongation max error 1.21569e-14; saved I_h retained exactly. The unused normal preparation's 3.88264e-6 discrepancy remains explicitly logged and does not enter a probe.
- Reconstructed Q2 values/gradients versus actual production-QP exports: relative L2 errors <=4.44e-13, all meshes/modes/components.
- Normal-ray intervals cover the entire window without gaps/overlap; duplicate MPI cell owners are rejected.
- D=integral d_n u_s: maximum absolute discrepancy <=8.01e-27 m/s (8.01e-15 of the input amplitude). For the nearly vanishing alternating D, division by D itself produces about 2e-10 relative error, which is cancellation against a near-zero diagnostic, not a physical discrepancy.
- N versus finite differences of integral u_n: alternating relative L2 error falls from 4.078e-5 to 1.019e-5 (coarse) and 5.184e-5 to 1.296e-5 (fine) when the offline derivative spacing is halved from 0.5 to 0.25 m. Other modes have smaller errors. The reported N uses the exact integrated FE gradient, not this finite-difference approximation.

Two four-rank, noncommitting runs only: **49.42 s coarse, 67.89 s fine**.
Recorded peak child RSS: 1261628 and 1446332 KiB, respectively (child-process
measurement, **not** total MPI aggregate memory). Return code 1 is the
intentional stop after `MECHANICAL MODES VERIFIED`, before trial/history
publication; it is not used as evidence of a nonlinear trajectory pass.

Build and execution:

```sh
cmake --build benchmarks/reconstructed_fault/performance/build-gmg --target fault_mechanical_modes -j4
python3 benchmarks/reconstructed_fault/bp3/run_frozen_bulk_refinement.py coarse --decompose --execute
python3 benchmarks/reconstructed_fault/bp3/run_frozen_bulk_refinement.py fine --decompose --execute
MPLCONFIGDIR=/tmp/aspect-bp3-slip-plot python3 benchmarks/reconstructed_fault/bp3/analyze_mechanical_velocity.py
```

The launcher refuses to overwrite run logs; saved runs need not be repeated.
Changes are confined to test/benchmark diagnostics:
`tests/reconstructed_fault_mechanical_modes.cc`, new
`tests/reconstructed_fault_velocity_export.h`,
`benchmarks/reconstructed_fault/bp3/run_frozen_bulk_refinement.py`, and new
`benchmarks/reconstructed_fault/bp3/analyze_mechanical_velocity.py`, plus
this report and generated evidence. The prior CMake target and unrelated
README edits are retained, not reworked. Base commit is
`359ea223cd0fa088ba7c8b03338cc7fe19d4acdc`; no commit was made for this task.

## Bounded conclusion

The present data answer the decomposition question without another solve:
the weak short-wave shear stiffness is associated with accommodation by
along-fault variation of normal motion within the finite-width band, and
bulk refinement permits more of it. This is distinct from transmitting
the imposed Q1 rate as a cross-band tangential velocity difference.
Finite-width versus fault-Q1 discretization effects and nonlinear
state/normal-stress amplification are not separated by this test. No
stabilization, solver change, profile change or further campaign is implied.
