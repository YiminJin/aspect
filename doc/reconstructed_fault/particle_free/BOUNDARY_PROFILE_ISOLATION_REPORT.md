# Open-boundary and profile projection isolation

User-selected continuation of the [coupled Stage C report](TRANSPORTED_OBJECTIVE_STAGE_C_COUPLED_REPORT.md).
The experiments separate incoming tensor-history data, DG front resolution,
Q1 profile representation/repeated remapping, and analytic-localization tensor
projection. They retain the existing constitutive law, solver tolerances,
endpoint scheme, production inputs and all earlier failed evidence.

Work stayed in `aspect-pf-rsf-particle-free`, branch `pf-rsf-particle-free`,
HEAD `27178813cebd471b93a7a9308f41215b71b4a20b`, with its existing dirty work.
Entry verified all 928 native source hashes and the core/four checker-library
hashes against the previous closeout. **No native core source or existing binary
was changed.** A separately named checker library adds nonpublishing experiments;
the two new coupled runs use the retained material checker and core.
[Protocol](../../../benchmarks/reconstructed_fault/particle_free/transported_objective/BOUNDARY_PROFILE_PROTOCOL.md)
and [evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/boundary_profile/README.md)
record exact commands, inputs, source/binary hashes and criteria. Five fresh
native runs used two MPI ranks sequentially; there was no server job, event
replay, production restart/conversion or propagation/crust campaign.

## 1. Native tensor transport: boundary condition versus resolution

The actual distributed DGQ2 SSPRK3 operator transports full3D plane-strain
physical tensor components with prescribed v=(+/-0.4,0) m/s on [-2,2]^2 m
for 0.4 s. These are scratch operators using native spaces/rows/ghosts; the
accepted coupled state is checked unchanged afterward. There is no mechanical
solve or Maxwell reaction inside these prescribed transport experiments.

Three cases have independently specified initial and inflow data:

1. A representable translating polynomial,
   f=1+0.15(x-ct)+0.1y+0.08(x-ct)y,
   (tau_xx,tau_yy,tau_xy)=f(1,-0.3,0.45), with **exact inflow**.
2. The identical initial polynomial with **zero inflow**. Its correct solution
   is the translated polynomial where the backward characteristic starts in
   the initial domain, and zero where the characteristic enters through inflow.
   The uninterrupted polynomial continuation is a different boundary-value
   problem and is retained only as a deliberate contrast.
3. A compact tensor with f=max(0,1-(x-ct)^2)^4 max(0,1-y^2)^4 and zero inflow.
   Its exact support never touches the boundary during the experiment.

Both directions run on 8²/16²/32² cells with dt=0.00625s. At 16² cells,
dt=0.025/0.0125s additionally isolates time integration. No limiter or
interpolation modification is used. Prescribed arithmetic screens are 1e-10
relative for the polynomial and1e-10 absolute for the energy identity.

| Cells per side | Compact tensor relative L2 error | Measured spatial order |
|---:|---:|---:|
| 8 | 2.52510e-2 | — |
| 16 | 3.52143e-3 | 2.842 |
| 32 | 4.30760e-4 | 3.031 |

Both translation directions give the same compact errors to roundoff. Exact
polynomial inflow has maximum relative error 6.69e-15 over all meshes/timesteps.
Thus the native operator preserves representable translation with consistent
inflow. At 16² cells, zero inflow differs from the continued polynomial by
13.0246%, but differs from its **own discontinuous characteristic solution**
by 4.57647%. Timestep halving barely changes either field error at fixed mesh.
The first difference includes the changed boundary condition; the second is
primarily under-resolved DG front error. They must not be conflated.
[Native measurements and commands](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/boundary_profile/boundary-transport.json)
and [figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/boundary_profile/boundary-transport.png)
retain both references, directions and all results.

## 2. Material energy flux and numerical transport loss

For these constant-velocity, constant-G=10Pa transport controls,
E_h=integral |tau_h|_3²/(4G) dx. A separate volume/face integration verifies

```
dE_h/dt + outward physical boundary energy flux + upwind jump loss = 0.
```

The RHS derivative uses the native operator rows and inflow loads, while face
flux and jumps are independently evaluated from the DG polynomials. Physical
outflow uses the interior trace; inflow uses prescribed external data. Interior
upwind loss is integral |v.n| |[tau]|_3²/(4G) ds; incoming boundaries also have
|v.n| |tau_h-g|_3²/(4G). Each interior face is counted once. The maximum
instantaneous identity defect is 4.68e-15 W/m, passing the fixed 1e-10 screen.

For positive translation on 16² cells at dt=0.00625s:

| Case | Integrated outward physical flux (J/m) | Integrated numerical upwind loss (J/m) | Energy integration remainder (J/m) |
|---|---:|---:|---:|
| Polynomial, exact inflow | 4.01913907e-2 | 4.96e-31 | -7.30e-15 |
| Polynomial, zero inflow | 5.54136046e-2 | 1.70136156e-3 | +3.66895706e-6 |
| Compact tensor | 6.69e-15 | 7.01315780e-7 | -8.54102636e-11 |

The compact exact physical flux is zero; the small numerical tail leakage is
measured, not set to zero (6.72e-11/6.69e-15/1.75e-21J/m across the three meshes).
Numerical upwind loss decreases from 1.5916e-5 to 7.0132e-7 to 2.9143e-8J/m.
It is not physical Maxwell/frictional dissipation or a compensating energy term.

The cumulative remainder is E_final-E_initial plus the **trapezoidal** integral
of flux and loss. It contains both diagnostic time-quadrature and SSPRK3 error.
For zero inflow, positive-direction signed remainders are
+5.72961e-5,+1.45778e-5,+3.66896e-6J/m on successive timestep halvings
(orders 1.975/1.990); negative direction gives the same orders. Compact signed
remainders are -1.35943e-8,-1.35383e-9,-8.54103e-11J/m. Their apparent higher
orders mix two errors and are **not** a universal integration-order claim.
Polynomial remainder is roundoff; fitted orders there have no significance.
This closes the pure-transport energy/flux check, not the complete coupled
moving-fault work balance or the separate cohesive energy gate.

## 3. Q1 profile representation and repeated interpolation

An independent compact material profile is

```
phi0 = .35 max(0,1-(x/1.4)^2)^4 max(0,1-((y-.3x)/.6)^2)^4.
```

The Python reference implements the actual convex Q1 nodal pullback under
constant translation, with explicit zero inflow. It compares repeated nodal
remapping, one pullback from the initial Q1 field, and Q1 sampling of the exact
continuum endpoint. Errors are integrated using Gauss8. It retains the signed
cross term in

```
||phi_h-phi_exact||² = ||phi_h-I_h phi_exact||²
                   + ||I_h phi_exact-phi_exact||²
                   + 2 <phi_h-I_h phi_exact, I_h phi_exact-phi_exact>.
```

| Cells per side | Relative final L2, dt=0.1s | dt=0.05s | dt=0.025s |
|---:|---:|---:|---:|
| 16 | 18.7518% | 19.1849% | 19.3913% |
| 32 | 6.74453% | 7.49742% | 7.86328% |
| 64 | 1.83809% | 2.73870% | 3.18645% |

These are profile errors, not tensor errors. Both directions agree. At 32²
cells, the endpoint representation L2 is 0.00693095 m for every dt; repeated
remapping error increases from 0.00736785 to 0.00970348 m as dt decreases.
The total error grows from 0.0129611 to 0.0151110 m, including its positive cross
term. One-shot pullback gives 0.00818357 m and is only a diagnostic reference;
it is not a proposed implementation for time-dependent solved velocity.

The mechanism has an independent exact moment test. For translation with
0<=C=|c|dt/h<=1, linear interpolation mixes neighboring samples with weights
1-C,C. After N steps it adds variance

```
Delta variance = N C(1-C) h² = |c| T h (1-C),
nu_equivalent = |c| h (1-C)/2.
```

At fixed h, reducing dt reduces C and **increases** this accumulated numerical
diffusion toward a finite spatial-discretization limit. A padded one-dimensional
compact control verifies the variance increment to 3.47e-17 m² across both
directions, three h and three dt. A finite-domain version retains its small
numerical tail mass loss and variance defect instead of applying an infinite-
domain identity to escaped numerical tails. For example the coarse two-dimensional
profile loses 7.41e-6 m² of integrated phi at dt=0.025s despite compact exact support.
No limiter, phase cap, added dissipation or altered obstacle was introduced.

[Profile measurements](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/boundary_profile/profile-projection.json)
and [figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/boundary_profile/profile-projection.png)
include mass, maximum, centroid, variance, decomposition and native replay.

## 4. Replay and two coupled compact-profile controls

Two fresh 256-cell coupled runs use the compact initial profile, dt=0.1/0.05s,
the same 0.1 s initialization mechanics interval and final t=0.4s. All other
material, traction, fault, velocity, solver and acceptance settings remain those
of the earlier fixture. Both pass four/eight physical steps (18.70/24.97s local
wall time). The core and original checker are byte-identical to entry.

Independent replay uses the **actual exported accepted velocity**, not nominal
translation, to reconstruct each native Q1 nodal pullback. Including the native
point-lookup convention, maximum discrepancy over old and new cases is below
4.45e-16 in phi. An initial mathematical pullback comparison failed its 1e-11
screen at 9.11e-8 near a cell face: deal.II RemotePointEvaluation defaults to
1e-6 unit-cell search tolerance and average evaluation on admitted neighboring
cells. Replaying their clamped reference-cell evaluations reproduces the result.
The tolerance effect is separately measured; it is not absorbed by relaxing
the replay screen. It is much smaller than accumulated Q1 diffusion. Neither
the core tolerance nor its averaging policy was changed.

Compacting the initial profile reduces the final one-step boundary-band profile
L2 defect from 0.00482820 to 0.000678007 m at dt=0.1s. The interior defect remains
0.00372300 m. At halfdt they are 0.000434641 m and0.00224212m. This removes much
of the profile-boundary interaction but does not remove bulk tensor inflow:
the mechanical strain source still creates nonzero history throughout the box.

## 5. Where the retained tensor comparison error actually lives

The final matched positive/negative runs put 99.16%/99.05% of squared tensor
characteristic error in the **inflow strip**, sign(c)x<-1.25m. At halfdt the
inflow strip still contains 98.08%. Within the earlier "interior" window
|x|,|y|<1.25 m, 97.16% of the halfdt error lies in the immediately downstream
strip sign(c)x<-0.75m. Its largest point is (-1.2218,-1.125) m, away from the
fault-center profile. With the compact coupled profile the corresponding
fraction remains 96.82%. Thus the large nonmonotone interior tensor comparison
was chiefly a boundary/front tail in that window, not an isolated profile or
stress-spin instability. The deeper core |x|,|y|<0.75m changes much less.

A separate constant-coefficient decomposition uses independently projected
analytic localization B=2chi ZS. Let delta_B=P8 B-B, where P8 is the actual
Gauss8 DGQ2 projection. The measured pre-reaction anchor gives

```
stress defect = beta (delta_B_mapped - Q delta_B_old(foot))
              + remaining projected transport/rotation mismatch.
```

Mapped B uses incoming Z on new geometry/profile/I. The remainder combines DG
transport, characteristic-reference error and finite-rotation projection; it
is not an inferred constitutive source. No energy is assigned to auxiliary Z.
Norms include the full3D metric and signed cross terms; they are not additive
error percentages. At the original matched dt=0.1s, interior total/localization/
remaining L2 norms are 2.123e-5/1.764e-5/1.415e-5 Pa m. At halfdt they are
1.182e-4/1.917e-5/1.172e-4 Pa m. The localization term does not explain the
large halfdt increase. On the finer matched mesh the three norms are
9.613e-6/8.777e-6/3.232e-6 Pa m: localization projection becomes material to
that much smaller interior error.
[Spatial localization and signed decomposition](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/boundary_profile/tensor-defect.json)
retain every step and region, including deeper-core and upstream-strip checks.

## Conclusion and next justified experiment

The selected isolation experiments are complete. Native material transport
with consistent inflow converges for the compact tensor and passes its
semidiscrete energy/flux identity. Repeated Q1 interpolation has a measured,
analytically explained diffusion floor at fixed h. The earlier halfdt tensor
comparison is dominated by a boundary/front tail in the chosen observation
window; it must not be promoted to evidence against objective stress evolution.

The next coupled accuracy experiment should specify physically compatible
incoming tensor histories (or a sufficiently buffered domain), hold CFL fixed
while refining h and dt together, and compare deeper-interior tensor/profile
errors and the full coupled work/flux balance. Keep the existing zero-inflow/Q1
path as the control. Do not launch a longer trajectory or shrink dt alone to
claim convergence. A higher-order bounded profile transport comparison is
justified only after fixing that boundary/reference and measurement contract.
The pure-transport energy pass does not repair the cohesive endpoint-energy
or localized-tip failures; those remain separate. Production readiness, moving
AMR/birth/resampling and deformed cohesive shape work remain unqualified.

Protected binaries/checkpoints, 220 original server exports and 328 cleanup-critical
files remain hash-identical. The three-paper roadmap is unchanged; no realistic
subduction example is planned. The incomplete BP3 first event, 39/45 historical
frame loss and earlier lost temporary logs remain explicit.
