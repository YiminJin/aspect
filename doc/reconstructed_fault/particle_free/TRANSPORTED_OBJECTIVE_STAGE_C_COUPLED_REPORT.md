> Current implementation follow-up: [bounded profile transport and coupled
> comparison](BOUNDED_PROFILE_TRANSPORT_REPORT.md) adds an opt-in bounded Q1
> method and identified accepted-trace tensor inflow. This report retains the
> original Q1/zero-inflow coupled control and its measured limitations. The
> complete mature transaction and heterogeneous reaction below remain its base.

> Follow-up completed: [boundary/profile isolation](BOUNDARY_PROFILE_ISOLATION_REPORT.md)
> now quantifies the inflow/front and Q1/localization errors discussed below.
> At that isolation stage, native source and the coupled law were unchanged. Pure-transport energy/flux
> passes; full coupled energy and production readiness remain unqualified.

# Stage C coupled material experiment — 3 November 2026

This is the narrower continuation selected by the user: connect the mature
moving state, add coupled spatial viscosity, and diagnose the existing cohesive
energy and tip failures. It supersedes the missing-transaction statements in
the earlier [Stage C report](TRANSPORTED_OBJECTIVE_STAGE_C_REPORT.md). It does
not close all of Stage C or qualify a production trajectory. The cohesive
interface, mature material experiment and production BP3 remain separate laws.

The worktree is `aspect-pf-rsf-particle-free`, branch `pf-rsf-particle-free`,
HEAD `27178813cebd471b93a7a9308f41215b71b4a20b`, with retained incoming dirty
source and manual work. The current executable is built from that dirty source,
not assumed to equal HEAD. Entry controls were copied before rebuilding;
[entry and evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_c_coupled/entry.json)
and the [final provenance manifest](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_c_coupled/final-provenance.json)
identify binaries, plugins and source.

## Implemented state and numerical law

`Stress rotation = Jaumann DG material Lie` is a new experimental selection;
legacy/default and stationary `Jaumann DG Lie` remain. The admitted scope is
2D full-3D plane strain, constant matched reference G/eta, mature friction,
growth disabled, fixed Cartesian bulk mesh, one interior material fault with
same labels and no computational-boundary completion. Diagnostic particles
stay fixed; independent stress history belongs to the DG owner. No resampling,
birth, moving AMR, bulk mesh deformation or deformed cohesive energy is enabled.
The optional positive `Material viscosity multiplier` defines prescribed
Eulerian eta(x,t)/eta_ref; it is not an advected compositional material field.
`Material initial phase` supplies the fresh continuum profile. Both default to
the legacy-neutral values 1 and 0 and apply only to the new selector.

Let B=2 chi Z S, chi=h(phi)/I, Z=G s_mem, and P the common Gauss-8 DGQ2
projection. Z is auxiliary memory and has no cohesive energy. From one accepted
state, the native DG SSPRK3 upwind operator A transports T and R+P B with
accepted bulk velocity v_minus. Explicit inflow is zero tensor and zero new
phase. The same accepted velocity advances labelled vertices and supplies the
Q1 characteristic profile/obstacle pullback. Phase redistribution is material
transport; it is not growth and does not retain an Eulerian damage trail.
Surface age and Z stay attached to the moving labels before their source solve.
Support widths transform by the inverse-transpose material map; geometric
normals come from the moved curve, never from the stress spin.

With Q=exp(dt skew(grad v_minus)), the staged anchor is

```
Tbar = P[Q A(Tminus) Q^T]
Rbar = P[Q A(Rminus + P Bminus) Q^T - Bmapped(Zminus)]
tau_anchor = Tbar - Rbar - Bmapped(Zminus).
```

This retains the evolving localization and complete residual history. It is a
first-order split discrete replacement, with measured projection/remapping
error, not an exact finite-step variable transformation or finite-strain law.
The total material Jaumann convention and full 3D deviator/norm are unchanged;
no localized-slip spin is subtracted.

For matched constants the original exponential reaction follows:
Tplus=beta Tbar+2 kappa D3(vplus), Rplus=beta Rbar,
Zplus=beta Zminus+kappa Vplus, where beta=exp(-G dt/eta) and
kappa=eta(1-beta). The resulting point stress is the stress in native mechanics.

For spatial eta, bulk and surface coefficients are frozen at their endpoint
positions. The source retains every coefficient/localization product:

```
Zplus_j = beta_Gamma,j Zminus_j + kappa_Gamma,j Vplus_j
Tplus = P[beta_b Tbar + 2 kappa_b D3(vplus)]
Rplus = P[beta_b (Rbar+Bmapped(Zminus))
          + kappa_b (2 L[Vplus]) - Bmapped(Zplus)]
tau_h = Tplus - Rplus - Bmapped(Zplus).
```

Here L[V]=chi V S. A projected constitutive correction is tau_h minus the
local unprojected Maxwell response. Each native solve freezes this correction;
the outer iteration updates it from the solved velocity/rate until relative
quadrature change is at most 128 machine epsilons (maximum 24 solves). It then
re-enters the original publishing native residual/KKT gate. This is a declared
converged outer iteration, not a claim that the local Maxwell matrix is the
full projected tangent. Publication independently reconstructs the candidate
and uses the existing strict stress agreement bound. No physical acceptance
tolerance or dissipation was changed.

`DGMaterialTrial` retains the accepted FE owner, manager, normalization,
temperatures and complete diagnostic particle properties. Geometry/profile and
normalization/work maps bind as solve scratch; all bulk/surface state publishes
only after outer acceptance. A rejected completed step restores the parent and
invalidates derived surface/B caches. Schema7 persists complete T/R, physical
phase, surface/material registry, geometry and clocks. Identity includes the
law/integrator, material/friction and phase inputs; incompatible imports fail.
Only new local experimental checkpoints were resumed.

## Local evidence and its limits

The native fixture has an oblique finite fault, nonuniform initial age and
traction, and a smooth modulated profile. Boundary velocity is
(vx,vy)=(+/-0.4+0.01y,0) m/s with an open top. G=10 Pa, eta_ref=10 Pa s;
eta/eta_ref=1+0.5 tanh(4x) for the heterogeneous cases. The declared domain is
[-2,2]^2 m. Initial mechanics uses the same 0.1 s interval across time controls.
The selected baseline is 256 cells, dt=0.1 s and four physical steps to 0.4 s.
See each run's input and command JSON for all RSF, phase, boundary and solver
parameters. These dimensioned synthetic fixtures are not physical BP3 inputs.

The [independent comparison](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_c_coupled/material-comparison.json)
reconstructs full Q2 polynomials from exports and independently implements
geometric localization, characteristic pullback, rotation and exponential
Maxwell/age/memory reaction. It never calls the production constitutive code.
Curve, scalar and reconstruction identities use absolute 1e-12/1e-10 screens;
PDE/profile differences are measured discretization errors, not forced to zero.
The [figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_c_coupled/material-comparison.png)
shows the accepted curve/history and evolving error.

- Both constant-coefficient directions cross bulk cells and pass exact-label
  curve/source/publication checks. At t=0.4 s the independent curve error is
  below 5.6e-17 m, age error is zero and auxiliary-memory error is below
  1.1e-19 Pa m. Stress reconstruction error is below 4.1e-16 Pa.
- Whole-domain one-step characteristic relative L2 errors are 2.757% and
  2.841%; restricted to |x|,|y|<1.25 m they are 0.01859% and 0.01926%.
  Zero inflow with nonzero interior history dominates the global difference.
  The reference is a first-order characteristic scheme, not the exact DG
  semidiscretization. Those errors must not be labelled solver tolerances or
  a completed Galilean/spatial convergence study.
- One spatial halving (256 to 1024 cells, same dt) reduces the final whole-domain
  characteristic error from 2.757% to 1.619%, interior error from 0.01859% to
  0.008425%, and fault-center error from 0.02716% to 0.006966%. Q1 one-step
  profile remapping L2 decreases from 0.005696 to 0.004120 m. This supports
  convergence but is not an asymptotic order claim from two meshes.
- Completed-trial rejection restores complete accepted tensors, phase, geometry,
  surface properties and rates. With the same accepted time sequence, all 19
  comparable bulk/surface/model exports are byte-identical to the control.
- A two-rank checkpoint at t=0.2 s resumes on four ranks through cell crossing.
  Bulk export differences are at most 1.78e-15; surface differences at most
  8.89e-15, passing per-column 1e-12 relative/absolute bounds. The comparison
  includes both continued steps, not just a restart load.
- The positive heterogeneous four-step calculation passes with three projection
  outer iterations per physical step. The first correction changes by
  2.4234e-5, 1.0628e-10, then zero. The first observed publication error is
  9.02e-16 Pa against a 2.84e-14 Pa bound. Reverse heterogeneous motion also
  passes with the native affine/frozen-load audit enabled.

The [lifecycle reduction](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_c_coupled/lifecycle-comparison.json)
records per-variable errors and input hashes. The half-dt matched run contains
an extra terminal interval of 5.55e-17 s caused by floating-point end-time
rounding; its outputs are retained, but that interval is excluded from physical
time-comparison plots and conclusions.

## Bounded sensitivity, cost and final regression

[The sensitivity reduction](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_c_coupled/material-sensitivity.json)
keeps final-state differences separate from the one-step characteristic error.
At dt=0.05 s the matched whole-domain one-step error is 2.314% and the
heterogeneous error is 1.978%, compared with 2.757% and 2.229% at dt=0.1 s.
However the interior errors grow from 0.0186% to 0.1037% and from 0.0333% to
0.0844%, respectively. Fault-center errors change from 0.0272% to 0.0260%
and from 0.0495% to 0.0426%. These spatial/projection-dependent one-step
comparisons do **not** establish monotone temporal convergence.

The actual final physical tensors differ between the two timesteps by 2.197%
whole-domain/0.3692% interior (matched), and 1.818%/0.3482% (heterogeneous).
The matched mesh-halving final-state difference is 4.682%/0.2423%. Each norm
uses the full plane-strain metric. The need for an interior/inflow-controlled
reference remains visible rather than being replaced by a solver-pass claim.

The heterogeneous four-step positive fixture takes about34 s, its eight-step
half-dt counterpart71.76 s, and the reverse audited four-step case40.71 s.
The 1024-cell matched case takes34.09 s. These are local wall times with fixed
initialization/checkpoint/export overhead, not production performance estimates.
The refined mesh and half-dt runs are single targeted checks, not a parameter
matrix or a long event replay.

The rebuilt full unit suite passes **183170 assertions in 152 cases**. The
fresh three-step legacy control's accepted-step CSV is byte-identical to the
preserved pre-Stage-C control. New heterogeneous two-to-four-rank restart
through the final two steps agrees with uninterrupted exports to8.89e-15.
The heterogeneous completed-trial rejection also restores complete accepted
state exactly. Its matched control is byte-identical through the retried step.
The final endpoint initially fails a byte-identity screen: termination limits
the rejected schedule to dt=0.049999999999999989 s, while the function-limited
control uses0.050000000000000003 s. Final export differences are at most
7.11e-15, within the same predeclared1e-12 per-column comparison bounds.
Original outputs and both timestep sequences remain unchanged.

Exact source/binary/input hashes and durable build/unit/run logs are recorded
in the final provenance manifest. No binary is identified solely by HEAD.

## Retained failures and independent cohesive diagnosis

Early local failures are retained with their inputs/logs: fixture registration,
closed-box initialization, owned/ghost vector-map mismatch, invalidating a
persistent cache owner by destroying it, and restart normalization checking the
already moved profile. The fixes use owned vector assembly, derived-cache
invalidation and validation of the loaded profile before staging transport.
The closed-box fixture was replaced by the explicitly documented open-top
fixture; no legacy tip load or tolerance was changed.

The initial reverse heterogeneous audit failed at step0 in the tiny nonzero-V
probe. Its subtraction of two full assembled residuals omitted A*x roundoff
from the error bound. The diagnostic now uses the same absolute-row precision
estimate as the existing affine audit, retaining exact frozen-load equality.
The full native physical convergence criteria are unchanged. Failed and passing
logs remain separately labelled; this is not evidence for relaxed physics.

[The cohesive diagnosis](STAGE_C_COHESIVE_DIAGNOSIS.md) uses existing 8/16/32-step
energy outputs. Signed and accumulated absolute physical errors coincide;
the error is first order and dominated by bulk BE and surface endpoint terms,
with endpoint-minus-trapezoidal work of the same order. Projection/quadrature
terms are around roundoff. The 1% fracture-relative screen still fails at
149.6/74.8/37.4%. The endpoint scheme is retained.

The localized-tip legacy and objective traces are byte-identical. The first
failure is force-residual merit decrease, with compressive contact, converged
complementarity and an accurately solved Newton linear system; admission is
not reached. Halving dt also fails. This rules out a rotation-specific cause
in that first increment, but does not identify the exact arithmetic source.

## Readiness and next justified work

Source implementation and bounded native lifecycle/coefficient verification
now include the complete mature moving transaction. Physical comparison remains
a synthetic, first-order transport/reaction comparison with quantified finite
mesh errors. Production readiness is **not established**. The moving mature energy/advective
flux balance also lacks independent qualification; exact point rotation alone
does not establish that balance. Deformed cohesive
shape/pressure work, material birth/resampling, boundary completion, moving AMR
and more general material coefficients remain outside this enabled scope.

The next useful moving-fault experiment should isolate the open-boundary and
profile projection errors with a compact interior manufactured tensor/profile
and independently prescribed inflow, including the material energy/flux
balance, before any longer physical trajectory.
For the tip failure, replay the stalled residual and active branches in higher
precision before changing the solver. No long energy matrix is justified by
the current first-order estimate alone.

The three-example roadmap remains BP3, controlled viscoelastic crust and fault
propagation. This work advances Example 2's numerical ingredients; it launches
none of those campaigns. BP3's copied first event remains incomplete. The
cleanup record's loss of 39/45 historical animation frames and the earlier
lost temporary build logs remain explicit; none were reconstructed or hidden.
