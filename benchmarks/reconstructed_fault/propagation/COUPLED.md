# Initial mechanical-feedback package

The subsequent [bounded material study](MATERIAL_STUDY.md) uses fresh core 0.6,
modest loading and reference/fourfold Gc. Its results and plots are in the
[P2 material report](../../../doc/reconstructed_fault/propagation/P2_material_report.md).
The original high-load configuration and failed qualification below are preserved.


Current result: [normalization follow-up](../../../doc/reconstructed_fault/propagation/normalization_report.md)
finds exact endpoint saturation and divergent normalization on the finer path.
I_h-only lumping is positive but poorly resolved on the finite coarse field.
The fresh coupled continuation gate fails; the commands below retain historical
reproduction instructions, not a recommendation to advance these trajectories.


## Model and question (current normalization investigation)

This small verification asks whether mechanically computed particle history H
produces resolved phase evolution, and whether cumulative feedback over the
same physical loading interval persists at dt/2. It is not an earthquake model
or a propagation benchmark. The old runs below failed before 1 s; normalization
finiteness and spatial accuracy must be resolved before fresh coupled runs.

| Geometry and discretization | Value |
|---|---|
| Domain / sharp fault | [0,1] x [0,1] m / (0,0.5) to (1,0.5) m |
| Bulk mesh / fields | Fixed 32 x 32 squares, h=0.03125 m; Q1 phase, Q2 velocity |
| Fault discretization | Q1 ordered line, spacing 0.1 m (11 vertices); frozen-field comparison also 0.05 m |
| Particles | 3 x 3 reference-cell particles per square, 9,216 total, native CPDI domains |
| Association | Fixed full normal cross-section, half-width 0.50000000000001588 m; sharp endpoints unchanged |

| Physical parameter | Value (both material constituents) |
|---|---|
| AT1 / Lorentz | ell=0.15625 m, curvature p=1, Gc=26.666666666666667 J/m² |
| Bulk density | 3300 kg/m³; gravity is zero |
| Elasticity / viscosity | G=1e6 Pa; eta=1e8 Pa s; bounds 1e6 to 1e10 Pa s |
| Intact cohesion / critical H | c=1000 Pa / Hc=c²/(2G)=0.5 Pa |
| Degradation / localization | g=(1-phi)²/[(1-phi)²+128 phi(1+phi)]; h=1/g-1; I_h has units m |
| Friction | Minimum slip rate 1e-20 m/s; stateless rate-dependent law, reference=dynamic mu=0.1; characteristic rate 2e-6 m/s (equal coefficients remove weakening) |
| Normal pressure / gravity | Adiabatic pressure fixed by surface pressure 10000 Pa / gravity 0 m/s² |
| Radiation damping | 0 Pa s/m in revised case (quasistatic) |
| Time | 0 to 1 s; maximum dt=0.25 or 0.125 s; native initialization interval 1 s |

Fresh initialization starts from the prescribed rounded band with nominal core
phi=0.8. Native prescribed particle H is retained while the bound phase solver
relaxes the field (maximum 0.8084775462894394). Cohesive traction and previous
I_h are then initialized using that accepted phase. Initial particle Maxwell
stress is zero, temperature is uniform 293 K, and chemical fraction rock=0.5.
At particle points with prescribed phi_guess>1e-6, native initialization sets
H0=64 × 0.8 / [h(0.8) g(phi_guess)²] Pa; outside that prescribed support H0
retains the intact value 0.5 Pa. This stationary-profile assignment is not an
Hc floor inside the band. Its sampled maximum is 223460.37877567374 Pa. H0 is
not regenerated after relaxation. In the original consistent representation,
initial q is about 463.56835343 Pa and I_h about 366.20754983 m.
There is no imposed seed constraint or previous-phi lower bound.

All four boundaries prescribe ux=0.4 t (y-0.5) m/s and uy=0. The coefficient
is 0.4 s^-2; integrated top-minus-bottom displacement is 0.2 m at 1 s.
Phase has the native natural zero-normal-flux boundary condition on all sides;
there is no prescribed phase value. No traction boundary is applied. Pressure
uses surface pressure normalization (10000 Pa); friction uses the prescribed
adiabatic pressure rather than dynamically computed normal traction.
Temperature is static, so no heat equation or thermal boundary evolution is
solved. Composition and stress use particle properties; no compositional
boundary influx is exercised by this restricted fixed-particle test.

Phase, velocity, dynamic pressure, Maxwell stress, mechanical candidate H,
committed max-history H, cohesive traction q, slip rate V and normalization
I_h evolve. Geometry, mesh, temperature, material fractions, associations and
particle positions/IDs are fixed. The existing frozen-RK2 diagnostic makes only
particle transport dt zero; physical constitutive dt remains unchanged. No
particle births occur. This isolates sequential H -> phase -> mechanics -> new H
feedback; it does not qualify transported histories, replenishment, nucleation,
or support admission. No outer phase/mechanics iteration or RSF theta exists.

The original **subcritical control** (`coupled_subcritical.prm`) used
ux=0.004 t (y-0.5) and radiation damping 1e7 Pa s/m. Both timestep paths completed
1 s with unchanged H and phase. The revised case (`coupled.prm`) increases the
ramp by 100 and removes radiation damping to obtain measurable quasistatic
mechanical driving. These changes preceded the normalization failure; current
normalization comparisons retain the revised parameters without further tuning.

Phase tolerances are relative 1e-12 (100 iterations, 35 backtracks), CG 2e-7;
explicit KKT acceptance is 1e-11. Mechanical tolerance is 1e-8 (30 iterations).
Normal quadrature/tail tolerances are 1e-8, with three Gauss surface samples per
segment by default. Those tolerances do not establish projection accuracy.
Bounds, equilibrium, resolved evolution, normalization accuracy and healing are
separate criteria; retained straight healing allowance is 4.922e-12.

[P2 report](../../../doc/reconstructed_fault/propagation/P2_report.md) gives the
scientific interpretation. **P1 fixed-support checks pass; P2 does not qualify.**
Positive transverse integrals project to negative nodal normalization values,
which correctly stop production mechanics. Neither path completes the selected
one-second interval. The coupled checker returns 2, preserving that failed gate.

Use the GCC 12.4.0 core/plugin configuration in [INTEGRATION.md](INTEGRATION.md).
After building the core, build the new observer and run the selected cases:

```sh
cmake --build build-propagation-integration/plugin --target coupled_feedback -j2
build-propagation-integration/aspect --test 'Stage-I*'
python3 benchmarks/reconstructed_fault/propagation/run_coupled.py coupled_report025 --dt 0.25
python3 benchmarks/reconstructed_fault/propagation/run_coupled.py coupled_report0125 --dt 0.125
python3 benchmarks/reconstructed_fault/propagation/check_coupled.py coupled_report025 coupled_report0125
```

Recorded labels already exist; use fresh labels to rerun. The runner refuses to
overwrite outputs/logs and caps each serial run at 180 seconds. It explicitly
sets the existing `ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1` RK2 control and
`ASPECT_FAULT_NONLINEAR_DIAGNOSTIC=1` read-only solver diagnostics. The support
choice is a documented PRM entry, not an environment switch. Threads are capped
at one; no cluster job is launched. Source/executable/plugin hashes are captured
before execution in final runner versions.

[coupled.prm](coupled.prm) supplies the identical quasistatic loading function,
material inputs and full support at both dt values. The observer only reads
native H, verifies trial histories, and calculates diagnostics; it never sets
H or adds an outer phase/mechanics iteration. [coupled_subcritical.prm](coupled_subcritical.prm)
retains the original damped pilot with unchanged H/phase; `--subcritical` selects
it explicitly for reproduction. The stronger quasistatic fixture was selected
once to obtain a measurable response, before the normalization failure was known.
No physical parameter was changed afterward to hide that failure.

`feedback.csv` contains accepted-step scalar diagnostics. Small radial bins and
fault-vertex CSVs record incoming/committed H, native-function candidate replay,
phase, q, current/previous I_h and V. `phase-entry-*.csv` separately captures phase
consumption when subsequent mechanics fails. Failed-entry integration files show
positive Q16/Q32 normal integrals and the negative consistent projections;
Jacobian correction/identical-H repeat files estimate numerical uncertainty.
There are no whole-domain particle-history dumps.

All attempts are retained in [coupled_evidence](coupled_evidence), including
parser registration failure, the zero-base pressure arithmetic defect, subcritical
controls, observer point-sampling correction and a failed diagnostic compilation.
Two `coupled_diagnosed*` attempts inadvertently used the preceding observer after
that compilation failure; they are retained but excluded from final evidence.
The selected `coupled_report*` runs use the successfully rebuilt final observer.
The committed-state candidate comparison tolerance was not loosened: matching
the native point sampler resolved the observer failure.

The [summary](coupled_evidence/summary.json) performs numerical checks independently
of console-output matching. The [manifest](coupled_evidence/manifest.json) pins
final artifacts and records preservation mappings for earlier P1 evidence.
Production normalization, H evolution, phase bounds and accepted-history meaning
remain unchanged. The only P2 core correction includes the nonzero linear
correction in the existing pressure roundoff calculation; its RHS check and
mixed nonlinear cap remain intact. The Stage-I unit filter passes 20,141 assertions
in 15 cases, including genuine incompatible-flux rejection.

No physical rejected-step rollback, coupled restart or transported/birth-history
qualification is claimed. The observed single prescribed region does not qualify
multi-fault support admission, bulk nucleation or propagating endpoints.

Execution accounting: this continuation records eight support attempts (four
completed final serial/restart checks, four diagnostic-H-check failures) and
17 coupled attempts (two parser failures, three pre-fix pressure-check failures,
two completed subcritical trajectories, and ten stronger-load observer/diagnosis
attempts ending at the normalization or observer check). Only the two final
`coupled_report*` runs supply the final P2 tables. All 249 prior integration
hashes and all 98 older bounds hashes remain verifiable via manifest mappings.
