# Homogeneous DG configuration without particle domains

16 October 2026; base `a8707307008401ac9cff702a369e2e2a611fa0f7` plus preserved
incoming work. **The restored 64→256→64 diagnostic passes without constructing
or reading particle domains, adding particles or relaxing coverage checks.**
The new default-off `Use homogeneous mesh data` selection retains independent
DGQ2 history, fixed reference geometry, the consistent-endpoint/reduced-Newton
equations, unlimited DG, physical irreversibility and existing limiter settings.

[Declared rules](PF6_DOMAIN_FREE_FORMULATION.md), [source audit](SOURCE_AUDIT.md),
[remaining point/particle reads](PF6_PARTICLE_READ_AUDIT.md),
[inputs, measured checks and portable replay](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf6_domain_free/README.md).
All previous evidence remains, including the Gauss6 failure, the CPDI-domain
failures and **`strict_reference_match_passed=false`**. This is a setup/material
replacement in a restricted homogeneous experiment, not a constitutive change.

## Owners and the selected configuration

- The existing supplied physical phase table directly initializes the native
  Q1 phase. No H-driven initial relaxation, CPDI integration or particle-to-phase
  mapping occurs. Independent T and surface Z start unloaded. The initial phase
  is the lower obstacle for subsequent physical steps, with unchanged acceptance.
- Configured background material values supply the homogeneous law. Validate
  equality of configured G, eta, Gc, cohesion, fixed friction, damping, thermal
  conductivity and EOS entries; preserve the existing temperature-independent
  viscosity/no-clipping restrictions. Bulk fields are static. Surface chemical
  slots select the background label directly rather than solving a particle
  projection. A label changes from .5 to 0 in this fixture; both select identical
  configured coefficients. No heterogeneous-material replacement is qualified.
- The bulk mesh and fixed prescribed fault determine full-normal support.
  Initial geometric support does not call the stationary-profile construction
  in this mode. Stokes-QP associations and ordinary FE fracture use the mesh;
  fixed-surface knots and the existing physical quadrature remain shared.
- Diagnostic particle associations contain ID/position, active fault/segment and
  xi only. Their volume stays NaN and their quadrature is empty. Particle
  projection-system construction explicitly rejects these associations. Native
  matching-CellId FE gradients, DG traces and the independent Maxwell publication
  check remain active with the existing operand-derived allowance.
- The particle manager is configured with all domain/CPDI/face-data flags false.
  The observer directly verifies its domain-handler pointer is null at every
  exported point snapshot. There is no lazy constructor; a domain-handler access
  would throw. Thus no successful domain read is possible. These are an object-
  absence check plus a source/lifecycle proof, not invented access counters.
  The input uses the same 144 fixed check points and H/stress diagnostic storage;
  it removes the unused particle chemical-composition property and all mappings.

The unchanged physical equations are
`T_e=beta*T_n+2*kappa*D3(symgrad u)`, `Z_e=beta*Z_n+kappa*V`,
`tau=T−2*chi*Z*S`, `q=Z/I`, `chi=h(phi)/I`,
`h=m*phi*(1+phi)/(1−phi)^2`, `S_xy=sense/2`,
`beta=exp(−G*dt/eta)`, `kappa=−eta*expm1(−G*dt/eta)`.
The tensor metric remains `xx²+yy²+(xx+yy)²+2*xy²`.
Normal-ray integration, the consistent surface mass projection, its actual
derivatives/adjoints and DG Gauss3 mass/source projection are unchanged.
Gauss12 physical quadrature is selected in the fixture, not made a default.

The selection requires accepted DG consistent-endpoint history, a nonempty phase
seed, fixed prescribed full-normal geometry, static temperature/compositions,
homogeneous materials and disabled domains. A registry provenance tag reuses
native checkpoint ownership and rejects mode conversion. All six negative
controls reject: domains, heterogeneous G, missing seed, particle-driven fields,
phase-fitted geometry and resuming a legacy checkpoint as the new mode.

## Matched initialization and physical trajectory

Use the existing seed, material/loading/phase law and two physical steps,
dt=.125 s through t=.25 s. Initial mesh is 2x8 cells; the retained observer's
startup remaps leave 4x16 cells for physical advancement. At accepted step 1,
restore the original 4x16→8x32→4x16 (64→256→64 **total cells**) diagnostic, with
nonzero T and Z. Transfer never advances physical time or publication count.
No new timestep, quadrature or spatial accuracy claim follows from this study.

A same-build domain-backed control omits the known failing identity remap but
uses the same physical steps. Compare common physical functions and weak
functionals, not particle versus FE fracture equations: both selected DG controls
use the retained ordinary FE fracture energy. At initialization, T/Z/phase/I,
reconstructed stress and physical native fields match exactly. The supplied seed
matches every initial Q1 node exactly. Its coarse Q1 representation differs from
the actual supplied table profile by **8.01966% relative L2**, with sampled Linf
**.0470206**. This pre-existing representation error is reported separately from
initialization equivalence and subsequent evolution.

| New mode versus domain-backed control, all compared states | Maximum |
|---|---:|
| Independent T / reconstructed stress, scaled L2 | 1.44e-15 / 9.73e-15 |
| Physical phase and lower/time-level fields, Linf | 5.63e-16 |
| Z / I, scaled Linf | 1.51e-15 / 1.55e-16 |
| Common energy / weak-shear defect | 1.00e-15 J/m / 8.53e-13 Pa m |
| Native velocity/pressure/temperature/phase, scaled max | 1.36e-13 |

The complete tensor errors, integrals and statewise values are retained in
`legacy-comparison.json`. Non-authoritative FE stress slots remain static in the
new mode and differ by up to **1207.21 Pa** from the old mapped diagnostics at
step 2. Surface chemical labels and the particle property layout also differ.
These are explicitly unmatched diagnostics, not differences in authoritative
history, reconstructed physical stress, constitutive coefficients or weak loads.

## Transfer, normalization and publication

Each 1/2/4-rank full case passes **100 native gates**. Retain all nine coefficients
of each of three DGQ2 fields, including x²y². Manufactured complete global-Q2
reproduction, independent Gauss5 mass/restriction and lossy-tensor projection
checks pass. A +.01 nonrepresentable physical phase/lower-obstacle peak still
rejects before mutation; representable-phase coarsening is unchanged.

For the restored accepted-history 64→256→64 operation, all ranks give maximum
stress error **1.38e-15 scaled L2**, phase difference **3.34e-16**, common energy
defect **1.78e-15 J/m**, and weak-shear defect **1.43e-13 Pa m**. Z/I remain fixed
through remap. Raw signed quadratic-norm changes and tensor moments are retained;
roundoff in this identical-field operation is not physical dissipation.
The serial refine/coarsen costs are **.0468/.0413 s** with no domain construction.

Independent Gauss24/32 integration and a separately assembled consistent surface
mass solve check I and its derivatives at accepted step 1. With Q1 nodal direction
`d_i=phi_i*(1−phi_i)*(1+.2*x_i)`, integrate
`h`, `m*(1+3*phi)/(1−phi)^3*dphi`, and
`6*m*(1+phi)/(1−phi)^4*dphi²`. Maximum scaled I/DI/D²I errors are
**6.19e-16 / 6.47e-16 / 9.62e-16**. Native central FD, epsilon=1e-5, gives
**1.00e-10 / 1.67e-10** scaled first-derivative/Hessian-action errors, below the
predeclared 1e-7 FD screen. The independent 1e-9 derivative and existing 1e-11
normalization screens pass. Physical energy/weak-load Gauss24/32 checks also pass.

Actual outer rejection/retry preserves accepted histories, clocks, vectors,
bookkeeping and RNG; trials remain unpublished and physical acceptance publishes
once. Terminal matching-cell source rejection remains coordinated. Averaging
matching gradients fails the unchanged publication guard in the negative control.
The largest actual error/allowance ratio across these full runs is **2.44e-5**.
The publication validator and DG source/error-budget implementation are byte-
identical to the accepted preceding build. No comparison is removed or weakened.

## MPI, restart and legacy separation

| Comparison throughout the short fixture | Scaled stress L2 | Phase Linf |
|---|---:|---:|
| 1 vs 2 ranks | 1.03e-14 | 4.79e-16 |
| 1 vs 4 ranks | 1.20e-14 | 5.28e-16 |

All native field slots/time levels are additionally compared, with maximum
scaled errors **7.77e-13 / 5.18e-11** respectively, below the declared 1e-9 screen.
All 144 IDs and physical coordinate strings stay fixed through every exported
remap/restart. RK2's effective advection dt is zero; constitutive dt is .125 s.
Domain objects are absent at all eleven full-run point snapshots on each rank.

Same-count restart at 1/2/4 ranks is exact for every compared persistent history,
all native fields, point properties, stress, energies, fracture operators and
weak loads, including RNG bytes. A half-run/continued restart matches uninterrupted
advancement exactly at the common completed state. Completed restores start zero
physical timesteps and publish zero times.

Serial-to-four restore retains T/Z/phase/I, native fields and point properties
exactly against the saved serial state. The common-rule reconstructed stress is
identical; native weak-load/fracture-action differences are at MPI summation
roundoff (**1.71e-13 Pa m / 2.14e-14**). Native changed-count RNG replay remains
unavailable, with population changes disabled; the four streams match the fresh
four-rank layout. No stochastic restart/birth claim follows.

All 17 transfer/configuration cases use one core/observer set: ten complete
successfully and seven reject as specified (six invalid inputs and the original
domain-backed deep failure). Core SHA256 is
`76289837b28b3523910ce9adbfd9e67afab7b4e30e71358f78970a24a6d8c8e1`;
observer SHA256 is
`7130520fd891123809b72006dc83fb82cf06d22228d30fe86647ac5d1c0af2f0`.
Native full runs take **13.19/40.69/50.67 s** on 1/2/4 ranks; same-count restores
take approximately two seconds. Each case has a 180 s cap, one thread/rank;
jobs are sequential. These tiny instrumented runs demonstrate no parallel speedup.

The same-build legacy domain-backed control takes 14.49 s and includes different
diagnostic work; this is not a controlled speedup estimate. The separate frozen
FE/CPDI control reproduces all **21 CSVs byte-for-byte**. `particle_domain.cc`,
particle manager domain lifecycle/coverage checks, transfer operators and all
previous artifacts are unchanged in this task. The fresh legacy deep control
still fails the original neighborhood coverage guard. Its defect remains relevant
to that backend; it no longer blocks the homogeneous DG configuration.

## Remaining scope

Use this new selection for the retained homogeneous, fixed-geometry DG fixture.
Temporary particle reads now serve point generation/identity/ownership, fixed-
position checks, derived H/stress diagnostics, independent matching-cell
publication validation, RK2/backup and checkpoint/RNG persistence. No particle
chemical material or domain integration enters selected physical initialization,
accepted mechanics, phase evolution, normalization or fault weak loads.

Next selectable work is the previously identified independent-master phase
variation/obstacle KKT qualification on fixed-surface hanging-node meshes. Retain
its current evolution guard until C^T*g/C^T*H*C and the native lifecycle pass.
Heterogeneous material interpolation, removal of the remaining diagnostic point
owner, general PF5 completion, genuinely lossy obstacle transfer, automatic AMR,
adaptive quadrature, transport, moving geometry and propagation remain deferred.
