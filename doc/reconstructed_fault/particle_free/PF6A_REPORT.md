# PF6a: fixed-reference cohesive phase/mechanics

16 October 2026; worktree `aspect-pf-rsf-particle-free`, branch
`pf-rsf-particle-free`, starting HEAD
`a8707307008401ac9cff702a369e2e2a611fa0f7` plus preserved incoming work.
The selected bounded experiment is implemented and measured. **The discrete
operators and serial acceptance/restart contracts pass; spatial accuracy is not
qualified.** Normal refinement changes reconstructed stress by 42.24%, including
visible cell-scale variation without transport. This must be resolved before
propagation or moving-geometry work. Unlimited DGQ2 remains the experimental
baseline; WENO options, limiter settings and production defaults are unchanged.
This does not complete general PF5 transfer/MPI qualification.

## Formulation and ownership

[PF6A_FORMULATION.md](PF6A_FORMULATION.md) fixes the equations and verification
criteria before the native runs. The retained consistent-endpoint/reduced-Newton
design is reused, with PF3's ordinary FE fracture functional. This is a discrete
replacement for CPDI fracture in this selected mode, not a constitutive change.
No one-pass phase-first combination of observers is used.

Each trial reads accepted independent DGQ2 T_n, surface Q1 Z_n=G s_elas,n,
consistent I_n and physical Q1 phase obstacle p_n. For the interval-constant
solved velocity u and surface rate V,

```
beta=exp(-G dt/eta), kappa=-eta expm1(-G dt/eta)
T_e=beta T_n+2 kappa D3(symgrad u); Z_e=beta Z_n+kappa V
I(p)=consistent surface projection of the direct normal integral of h(p)
chi=h(p)/I(p); tau_e=T_e-2 chi Z_e S; q_e=Z_e/I(p)
E=integral_bulk |tau_e|3^2/(4G)+integral_surface Z_e^2/(2G I(p))+F_FE(p)
```

Here `|A|3²=xx²+yy²+(xx+yy)²+2xy²`, S_xy=sign/2 and
`F_FE=integral Gc/(c0 ell)*(p+ell²|grad p|²)`. Native Q2 strain fits exactly in
DGQ2 on these affine cells; all nine tensor modes remain available. No spin,
rate, pressure, Maxwell, friction, normalization or material convention changes.

Mechanics equilibrates at every trial phase. The phase row is the **partial**
derivative of E at fixed endpoint T_e,Z_e, not at fixed reconstructed physical
stress. Its reduced tangent includes du,dV from the native constrained mechanical
Jacobian, dT_e=2 kappa D3(symgrad du), dZ_e=kappa dV, and both first/second
normalization projection pullbacks. The existing endpoint friction potential
`(kappa/G) integral mu P V` enters the reduced line-search energy. Existing
active-set Newton, stability, true-energy and KKT acceptance are retained.
Physical irreversibility is p_n<=p<1; no phase cap or normalization floor is added.
H/particle stress do not drive this phase functional or provide mechanical input.

All trials retain the same accepted anchor. DG endpoint history is staged until
**after the actual timestep-manager repeat decision**, then published once before
postprocessing. A rejected candidate never becomes history. Exceptions and
cutbacks restore trial solution/linearization/statistics; native particle backup
and restore remain in the outer loop. Publication reuses material candidate
validation, derived sampling and the existing material/manager terminal writes.
The legacy FE endpoint publication branch remains unchanged.

## Source changes

| Owner | Selected change |
|---|---|
| Material header, `fe_phase_energy.cc` | Explicit DG tensor in the existing endpoint trial; fixed-endpoint energy/action and reduced tangent use that tensor, while ordinary PF3 trials still use accepted T |
| `fe_strain.cc` | Allow trial phase/I changes only in the endpoint mode; carry explicit trial phase through reconstruction and publish its accepted anchor |
| `phase_field_fault.cc`, `history.cc` | Admit only the existing DG + accepted-history + consistent-endpoint + evolving-phase selector combination; retain all other restrictions/defaults |
| `simulator/phase_field.cc` | Use PF3 FE fracture and its exact Hessian in DG endpoint fracture-only calls; avoid adding accepted-history elastic energy a second time |
| `reconstructed_fault_stokes.cc` | Preserve the DG endpoint when constructing the native mechanical-response tangent |
| `reconstructed_fault_endpoint_phase.cc`, `core.cc`, Simulator header | Stage DG publication; accept/discard after the real outer decision; restore entry bookkeeping on rejection |
| `dg_accepted.cc` | Same-mode version-2 payload separates accepted step from native next-step clock; require a canonical accepted checkpoint; version-1 PF5 archives unchanged |
| `dg_transport.cc`, `prepare_dg_accepted_step` | Reject coupled endpoint transport; the PF5 prescribed Strang path cannot silently advance PF6a |
| New `pf6a_checks` benchmark | Observe native outer advancement, residuals, histories, rejection and restart; diagnostic interval-mean loading plugin for variable-dt controls |

No root CMake, limiter implementation, transport operator/quadrature, physical
law or existing phase-report evidence is changed. The inaccurate old guard
messages were updated to describe the newly selected endpoint mode.

## Fixture and compatibility

Unit square, 2×32 affine bulk cells (bounded reference 2×64), Q2 velocity/Q1
pressure and physical phase, three independent DGQ2 components; straight fixed
horizontal surface with eight Q1 segments. At 32 normal cells: 650 velocity,
99 pressure, 99 phase, 1728 independent tensor DoFs, nine surface rate nodes;
the native shared vector has 2473 entries including other unchanged fields.
At 64: tensor DoFs 3456, phase 195, native vector 4905; surface unchanged.

G=1e6 Pa, eta=1e8 Pa s, Gc=26.6666666667 J/m², ell=.15625 m,
cohesion input c=1000 Pa, mu=.1 and prescribed compression P=10000 Pa.
The existing supplied phase seed is fixed across meshes, initially max
.6000019377630632. Histories start unloaded and develop nonzero T/Z before
phase growth. The exact interval mean of the fixed ramp
`u_x=.256*t*(y-.5), u_y=0` is imposed. No body force, radiation, RSF evolution,
bulk/surface advection, geometry update or birth is included.
The existing `ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1` control is explicit
in the run environment; it is required for this fixed-particle experiment.

Fresh final-source frozen FE/DG controls pass the full existing compatibility
suite: physical inputs match, 75 field checks differ by at most 3.20e-14 scaled,
and the retained disabled FE baseline agrees exactly. Source projection,
physical stress, shear/normal weak loads, work adjoint, nonzero elastic slip,
repeat/publication and particle-stress poisoning controls pass. This is a
compatibility control, not a coupled spatial reference.

## Derivatives, residuals and quadrature

Independent reference code interpolates exported native fields, differentiates
Q2 velocity analytically, assembles normal integration/projection and FE energy,
and uses complex-step phase Hessian actions. It does not call production
operators. Two independent directions are checked at nonzero T/Z with an
interior trial phase. Central-difference epsilons are 1e-3 through 1e-6.

| Check | Measured scaled error |
|---|---:|
| Independent energy / phase gradient | 5.55e-16 / 3.50e-15 |
| Independent I | 2.50e-16 |
| Independent fixed-endpoint Hessian actions | 6.42e-15, 6.49e-16 |
| Best combined endpoint derivative, two directions | 6.44e-11, 5.71e-11 |
| Best reduced tangent versus re-equilibrated FD | 1.69e-9, 6.42e-10 |
| Reduced tangent symmetry | 3.59e-14 |
| Native tensor source recurrence, all primary states | 4.37e-14 |
| Z recurrence, maximum absolute | 1.10e-11 Pa m |
| Independent reconstructed stress, all primary states | 3.31e-13 |
| Weak shear / normal / cohesive / friction loads | 9.22e-14 / 4.92e-15 / 8.03e-16 / 1.14e-16 |

The frozen 1e-6 FD, 1e-9 independent-reference and 1e-10 symmetry targets pass.
Repeated endpoint candidates are identical and do not publish. The maximum
exported bulk residual norm is 6.59e-6; final values are 2.11e-11–7.83e-11.
The independent consistent-Q1 **free** fault residual RMS is at most 8.30e-12 Pa,
with zero outward sticking violation and complementarity power at most
7.47e-13 in magnitude. An unprojected surface RMS near 992 Pa early in loading
is the admissible negative friction residual at V_min=1e-20 m/s, not a failed
free equation. Both raw and projected measures are retained.

No healing is measured. Native projected phase KKT is at most 5.05e-11 across
the primary matrix. The inherited criterion is the larger of 1e-12 times the
initial projected defect and its term-based roundoff allowance; it is unchanged.
The higher-rule KKT reaches 1.09e-9 for the six-point rule, versus 4.66e-11 for
the twelve-point control. Thus native KKT alone does not qualify quadrature.

Gauss3 integrates DG mass/source exactly here. The existing mechanical rule
splits tangential surface/bulk intersections and uses six normal points; a
12-point evolution control and independently generated 24/32-point rules check
the rational profile integrals. At fixed primary fields, six versus resolved
rules change energy by at most 3.69e-12 relative and weak loads by 4.07e-12
scaled; 24/32 uncertainty is at most 3.48e-14. Full evolution at 6/12 points
changes physical stress by 2.67e-10 relative L2. This passes the predeclared
1e-6 energy/load screen; gradient/KKT quadrature sensitivity is reported above.
No adaptive quadrature or transport-rule change is made.

## Coupled accuracy and work

All comparisons end at .5625 s, use the same physical seed/loading, and use the
full embedded tensor metric. Differences are evaluated at common physical
points; common-point quadrature uncertainty is separately recorded.

| Normal cells, dt, normal quadrature | Final max phase | Final max Z (Pa m) | Native seconds |
|---|---:|---:|---:|
| 32, 1/32, 6 | .620380495853 | 39122.9878376 | 33.03 |
| 32, 1/64, 6 | .620381173092 | 39123.0760678 | 41.74 |
| 32, 1/128, 6 | .620381342404 | 39123.0981254 | 90.80 |
| 64, 1/128, 6 | .648097848824 | 39149.6677748 | 93.87 |
| 32, 1/128, 12 | .620381342385 | 39123.0981252 | 126.38 |

Costs include startup, solves and diagnostic exports, with up to two concurrent
serial processes; they are not isolated solver throughput. Initial cap 180 s
excluded the first 64-cell attempt, which stopped after step 71. Its preserved
retry had a 240 s cap and completed in 93.87 s running alone. No solver tolerance
or physical parameter was relaxed.

| Relative difference | dt vs dt/2 | dt/2 vs dt/4 | 32 vs 64 at dt/4 | 6 vs 12 at dt/4 |
|---|---:|---:|---:|---:|
| Physical phase L2 | 1.284e-6 | 3.211e-7 | .05979 | 3.60e-11 |
| Independent T, full-tensor L2 | 2.514e-6 | 6.284e-7 | .13058 | 6.53e-11 |
| Physical stress, full-tensor L2 | 1.734e-6 | 4.334e-7 | .42237 | 2.68e-10 |
| Surface q | 2.167e-6 | 5.416e-7 | .07310 | 5.43e-11 |
| Weak shear load | 4.912e-7 | 1.228e-7 | .01569 | 1.22e-11 |

Stored-field temporal differences decrease by about four. This supports observed
second-order behavior for this ramp/short event, not a general temporal-order
claim for active-set changes. Raw final V samples represent different time
intervals: their 1.77%/.681% differences are not endpoint time-integration errors.
Averaging the fine V over each common final coarse interval gives 7.58e-8 and
2.91e-12 relative differences instead.

Initial phase representation error against the actual supplied piecewise-linear
seed is .00703037 relative L2 at 32 cells and .00174588 at 64. This is separated
from evolution differences, not silently assigned to the integrator. The 64-cell
run is a bounded spatial comparison, not an exact solution. Its stress difference
is 772.69 Pa absolute L2; the peak-phase difference is .02772. The plotted
cell-scale stress variation exists without advection. Neither quadrature nor dt
explains it; no limiter or compensating diffusion is introduced to hide it.

For energy accounting, define a diagnostic within-step path: linear physical
phase, fixed solved rates and exact exponential T/Z. Include changing chi/I
throughout. Independent integration verifies

```
Delta E - W_boundary + D_bulk + D_cohesive + D_friction
 = W_free_mechanical_residual + W_pressure/divergence
   + W_contact_residual + integral (partial_p E) pdot.
```

This is an accounting path, not the update algorithm. The boundary work comes
from a Q2 Dirichlet lifting; positive viscous terms use physical stress/Z.
The signed identity closes to 5.08e-14 J/m accumulated. Time Gauss12/24 uncertainty
in absolute-work diagnostics is at most 1.46e-5 J/m due to sign crossings.
Endpoint equilibrium does not make the whole within-step path equilibrated.

| Accumulated work (J/m), 32 cells | dt | dt/2 | dt/4 |
|---|---:|---:|---:|
| Delta E | 8.01564043 | 8.01566623 | 8.01567268 |
| Boundary work | 47.17356033 | 47.21193805 | 47.22552786 |
| Bulk viscous | .00680414 | .00680511 | .00681129 |
| Cohesive viscous | .01499238 | .01502761 | .01503074 |
| Frictional dissipation | 39.19323573 | 39.19324041 | 39.19323928 |
| External-balance remainder | .05711234 | .01880130 | .00522612 |
| Absolute free-mechanical residual work | .19129553 | .09671336 | .02359089 |
| Absolute contact residual work | .41507067 | .16949845 | .04244126 |
| Absolute phase work | .21142257 | .08539222 | .02165559 |

The finer-step external remainder is .0111% of boundary work. The larger
individual absolute terms are retained; cancellation is not characterized as
zero error or new dissipation. At 64 cells the remainder is .00496376 J/m;
the spatial dependence of the individual terms remains significant.

## Actual rejection, publication and restart

The native observer is attached to the real outer loop; it does not advance
accepted clocks or implement a phase/mechanical split. Tests cover:

- Repeated scratch trials at nonzero history; actual prepublication exception
  from `solve_timestep`, restoration of all saved persistent/native trial state,
  then ordinary outer advancement from the same anchor. The final coarse run passes
  110 gates, including rejection during a growing-phase step and rejection of a
  trial checkpoint. Publication count equals accepted physical step throughout.
- Deferred publication also rejects an intentionally stale solution context
  after `solve_timestep` returns, restores entry solution/linearization/statistics
  and leaves all histories unpublished. The test fails on the pinned prior core
  at the intended rollback gate and passes after the narrow catch/restore fix.
  Final-core coarse, cutback and resumed exports match their earlier references
  byte for byte. This is an error-path correction, not a different accepted update.
- Native `repeat on cutback`: at step 16 the converged candidate at .5 s with
  dt=1/32 is discarded, then retried at .484375 s with dt=1/64. The anchor still
  has 15 publications; history bytes and accepted phase are unchanged on retry.
  No manual reinitialization is used. The 22 accepted exports (including initial
  state) match the no-rejection **identical accepted timestep schedule** byte for
  byte, including all persistent/RNG/raw-manager histories and solution vectors.
- Automatic native filesystem checkpoint after accepted step 34 at .53125 s,
  with native next-step clock 35/.546875 s. Resume advances steps 35/36 and
  matches uninterrupted T,Z,I, phase/lower anchor, solution/old/older, manager
  properties/V/slip, particles/RNG, stress and weak loads exactly. Persistent
  value/RNG archives are bitwise equal. Raw manager archives differ by the
  existing 13-byte implicit-to-explicit +1 shear-sense representation; complete
  native load/resave archives agree bitwise at both resumed steps.
- Version-2-to-PF5 conversion is rejected with the intended version mismatch.
  The old PF5 version-1 snapshot still resumes with exact previous results.
  All 23 retained PF4 operator CSVs reproduce byte for byte, independent matrix/load
  errors remain below 1.6e-15 (constant defect 1.10e-14), native 628 assertions/eight cases and 94 PF1 checks
  pass. No WENO sweep or threshold tuning was performed.

The first automatic checkpoint failed because PF5's manual at-endpoint clock
contract was incorrectly reused after native `advance_time`. Version 2 fixes
that distinction and rejects noncanonical/trial snapshots. A subsequent source
audit identified the later timestep-manager rejection point; deferring DG
publication corrects that path. Both failures and the earlier incomplete
prepublication-only coverage are retained rather than presented as successful
outer-cutback qualification.

## Evidence, limitations and next task

[Portable evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf6a/README.md)
contains independent assessments, native inputs/logs/exports/checkpoints, all
build/source identities, lifecycle/compatibility checks, the comparison figure,
failed preliminary runs and a mathematical replay. Primary accuracy runs precede
the checkpoint/outer-publication corrections. Fresh final-implementation dt and
dt/2 controls reproduce every available primary state CSV byte for byte (19 and
37 states); the later observer additionally exports persistence/native vectors.
This establishes reuse of those accuracy measurements, not a claim that all
historical binaries have the final source hash. Final core SHA256 is
`c994b6b809bde83f456ebdf2d0c573ea2ff8dbf69427add898982a3b573a9693`;
final PF6a observer SHA256 is
`172dabb46f0d1fb17aad2943fdb4592e97bf3946aeeb3210bab8d0519398ebc7`.

Particles still supply startup H/CPDI/domain services, surface material mixture
projection, association/support and fixed-particle validity checks. Accepted
physical stress/H particle samples remain derived outputs, and native particle
and RNG persistence is retained. Neither incoming independent T nor endpoint
mechanics/fault loads reads particle stress. This is not end-to-end particle
removal. No advection, AMR/MPI, periodic/open transport, surface/phase transfer,
nonuniform surface transport, moving geometry, fault birth, RSF evolution or
coupled propagation is qualified.

**Recommended next task:** resolve fixed-geometry spatial representation first.
Freeze common physical phase/T/Z snapshots, separate phase-seed interpolation
from Q2 velocity/independent-tensor representation and physical reconstruction,
and compare weak equilibrium, pointwise/full-tensor stress and energy on one
bounded finer normal mesh at controlled quadrature. Then repeat a short event
with the same physical seed only if that diagnostic supports it. Keep limiter
and constitutive conventions fixed. General PF5 AMR/MPI transfer remains a
separate prerequisite to any later PF6 transfer fixture; moving geometry and
propagation are not authorized by this report.
