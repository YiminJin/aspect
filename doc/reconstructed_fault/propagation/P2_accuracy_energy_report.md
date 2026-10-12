# P2 accuracy and energy assessment (2026-10-07)

**Direct irreversibility is successfully verified for this fixed-mesh P2 fixture.
P3 is not ready.** All three retained irreversible trajectories complete to t=1
without healing and show endpoint-dominated damage. This assessment preserves
that implementation and all selected physics. Consistent normalization at .025 m
still misses the provisional 0.5% source-error target in strongly damaged fields.
Native zero-input stress histories are timestep-consistent; their FE representation
has a separate spatial error. Profile work balances the bulk mechanical storage
and dissipation, but the selected reduced-fault phase driver does not include
the associated bulk-profile energy variation. A governing decision is required
before claiming closed evolving-profile energetics or expanding implementation.
The conditional .03125/.015625 coupled comparison was therefore **not launched**.

## Model and restrictions

| Quantity | Retained value |
|---|---|
| Domain / fault | [0,1]² m; sharp straight segment (0,.5)–(1,.5) m |
| Bulk / surface | 32×32 fixed mesh, Q1 phase (1089 nodes), Q2 velocity; retained surface .1 m, Q3 per segment |
| Particles / support | 3×3/cell (9216); frozen positions, no birth; fixed full normal cross-section association (.50000000000001588 m) |
| Phase material, both constituents | Gc=26.6666666667 J/m²; ell=.15625 m; AT1 c0=8/3; Lorentz p=1, m=128.00000000016 |
| Strength / elasticity | c=1000 Pa; G=10⁶ Pa; Hc=c²/(2G)=.5 Pa; Gc/(c0 ell)=64.00000000008 Pa |
| Viscosity | eta=10⁸ Pa s; bounds 10⁶–10¹⁰ Pa s; Maxwell time 100 s |
| Friction / pressure | Stateless rate-dependent law, reference=dynamic mu=.1, characteristic V=2e-6 m/s, minimum V=1e-20 m/s; prescribed pressure 10000 Pa; radiation damping zero |
| Other bulk data | T=293 K, static; density 3300 kg/m³, gravity zero; frozen equal chemical fractions |
| Initialization | Nominal core .6, native seeded H, relaxed maximum .60354883767362211; then unloaded q=Z=bulk Maxwell stress=0; current/previous I consistent |
| Boundary / time | All four sides vx=.256 t(y-.5), vy=0; a=.256 s⁻², 0–1 s; final imposed relative displacement .128 m; natural zero phase flux |
| Selected options | Recoverable peak H; surface elastic-slip Z; bulk-conjugate work; consistent I; bound solver with immutable accepted phase lower bound |

The audited sequence remains H → phase → mechanics → committed H, with one-step
lag and no outer iteration. Initial relaxation remains box-only [0,1]. The local
stress diagnostic overrides velocity/strain rate and slip rate to **exactly zero**,
keeps phase fixed, and restores all live histories afterward. It is not a
zero-boundary mechanical equilibrium solve of an arbitrarily stressed body.
The elastic-limit eta=1e30 is a local diagnostic only, not a material change in
any coupled trajectory. No transport, replenishment, evolving MPI or restart
qualification is added.

## Frozen normalization accuracy

Use each retained accepted Q1 field at first resolved onset and t=1, without
advancing H or mechanics. For p=1, h=m phi(1+phi)/(1-phi)². Analytic integration
of every transverse Q1 interval supplies the direct reference. Phase maxima
remain <1, so these integrals are finite. The surface mass is the same consistent
Q1 arclength mass as production. The .1 m native projection is reproduced within
1.67e-9 relative error, inside its retained 1e-8 normal quadrature/tail tolerances.
No tolerance is changed.

To keep **integration accuracy** fixed while measuring spatial representation,
split surface RHS integration at all bulk and surface breakpoints and use Q32;
Q48 changes nodal I by <=4.89e-15 relative. The native Q3 per-element rule is a
separate comparator. Thus the provisional test is not confused with quadrature
convergence. Fine-spacing results are frozen algebraic projection comparisons,
not fresh coupled trajectories. Exact constant reproduction, positive coefficients
and conservation of the quadrature-defined integral pass (<=2e-14 / 2e-15).

Maximum relative error in h/I equals |I_direct/I_projected−1| wherever h>0;
we also inspect the endpoints and optimize interior extrema within each smooth
bulk/surface subinterval. Values below are percentages at strong damage:

| Retained dt | .1 m, accurate RHS | .05 m, accurate RHS | .025 m, accurate RHS | .025 m, native Q3 RHS |
|---|---:|---:|---:|---:|
| .125 | 6.59849 | 2.88940 | .93122 | .98072 |
| .0625 | 6.62105 | 2.85397 | .96978 | 1.01930 |
| .03125 | 6.39141 | 2.57709 | .85688 | .90130 |

All miss the **0.5% maximum** target at strong damage, including .025 m. Native
Q3 versus accurately integrated nodal coefficients differs by up to .617%,
.230%, .081% at the three spacings; it cannot explain the remaining .025 m
spatial error. Native .025 m worst locations are x=.96875/.90625/.09375 m.
Endpoint errors are +.3718/+.3359/+.3012% respectively; acceptable endpoint
values do not qualify the interior. .05 m endpoint errors reach -1.6224%.
At first onset, errors are much smaller: for coarse-path t=.75, .1/.05/.025 m
accurate-RHS errors are .09578/.02637/.01272%; finer paths' first responses are
only 4.10e-5 and 7.19e-7 phase increments and almost initially uniform.

[Data](../../../benchmarks/reconstructed_fault/propagation/accuracy_energy_evidence/normalization.json),
[plot](../../../benchmarks/reconstructed_fault/propagation/accuracy_energy_evidence/normalization.png).
Positivity supplies admissibility, not the requested accuracy.

## Native bulk-history transfer isolation

The benchmark calls actual
`PhaseFieldFault::commit_reconstructed_fault_mechanical_history` followed by
`Simulator::interpolate_particle_properties` and native constraint distribution,
using the existing MomentCycle diagnostic friendship. The selected native
cell-average interpolator and shared-DoF incident-cell arithmetic mean remain
unchanged. The constitutive update reads the **parent particle stress**, not the
mapped FE field. No generic projector is modified.

Three Q1-representable tensor fields (constant, affine normal/bilinear shear,
and nonconstant mesh-scale Q1 waves), two viscosities, and the existing three dt
partitions give 18 paths / 336 native updates / 354 observed states over 1 s.
Compare particle stress to exp(-Gt/eta) times its initial value; compare FE stress
both to the underlying field and to the decayed **initial mapped FE field**.
H is checked unchanged on every update. The comparison tolerances are 2e-10 Pa for particle/drift errors and 2e-12 J/m
for final cross-partition energies; measured errors are much smaller.
Snapshot restoration is checked normally
and after an intentional exception; all four simulator vectors and particle data
are restored exactly.

| Stress pattern | Max particle error (Pa) | Additional FE drift (Pa) | FE relative L2 representation error |
|---|---:|---:|---:|
| Constant | 6.44e-13 | 7.21e-13 | 6.98e-16 |
| Affine / bilinear | 1.79e-12 | 1.39e-12 | .11020% |
| Mesh-scale Q1 | 1.60e-12 | 1.59e-12 | 5.23575% |

Final energy values agree across partitions within 2e-12 J/m. The mesh-scale
FE energy is 3.62% below the analytically integrated field energy; particle-volume
sampling has its own .22% difference. These are differences between measures and
representations, not accumulating smoothing of the parent history. Repeated
native transfers add no resolved drift on this control. In coupled evolution,
FE stress affects the next mechanical solve, so representation error can still
alter feedback; this isolation does not qualify that spatial accuracy.

[Data](../../../benchmarks/reconstructed_fault/propagation/accuracy_energy_evidence/stress.json),
[plot](../../../benchmarks/reconstructed_fault/propagation/accuracy_energy_evidence/stress.png).

## Profile work and phase-energy meaning

The [derivation](P2_accuracy_energy_derivation.md) distinguishes the native constant
rate Maxwell path from an instantaneous profile jump followed by relaxation.
Both reach the same zero-input endpoint but have different O(dt) work/dissipation.
Controlled initial-to-final profile changes balance bulk storage/dissipation to
roundoff under either **consistently defined** path. Profile work is O(1), and
should not be assumed zero. It is already present in the bulk mechanical identity.

Reanalyze retained physical paths using native Stokes JxW for all bulk terms and
native Q3 ds for **all** fault terms, with exact Maxwell time integration. This
removes the earlier Q8/Q3 surface comparison from the matched ledger. J/m units:

| Cumulative term | dt=.125 | .0625 | .03125 |
|---|---:|---:|---:|
| Ordinary bulk slip work | 62.8182 | 58.0996 | 56.4557 |
| Bulk profile work W_C | -94.5101 | -127.0881 | -151.7852 |
| Sum native bulk storage increments | 6318.8587 | 5689.8792 | 5396.9315 |
| Bulk viscous loss, native measure | 26.4354 | 22.8763 | 21.2292 |
| Cohesive storage change | 8.2463 | 6.4060 | 5.5663 |
| Cohesive fixed-Z profile storage change J_Gamma | -3.5084 | -3.7036 | -3.7939 |
| Cohesive viscous loss | .04954 | .04188 | .03868 |
| Ordinary cohesive slip work | 11.8042 | 10.1515 | 9.3988 |
| Friction loss | 55.0862 | 51.2761 | 49.2409 |
| Fracture expenditure, retained CPDI | 2.8141 | 3.3481 | 3.6237 |
| Integrated ordinary exchange defect | -4.0722 | -3.3280 | -2.1839 |
| Native QP → direct particle-interpolated QP energy difference | 22.3840 | 59.9567 | 126.9818 |
| Interstep native QP → next incoming FE energy reset | -25.9521 | -31.2584 | -37.6186 |
| Mechanical-plus-fracture gap, excluding transfer | -97.8880 | -130.0607 | -153.7990 |

Sum native bulk storage increments is **not** final stored energy: incoming FE
and parent particle histories differ. The last two transfer measures above are
also different; neither should be hidden inside profile work. Matched bulk/fault
identities close per step to <=4.54e-11 / 7.82e-14 J/m, while ordinary integrated
exchange error decreases with refinement. The remaining mechanical-plus-fracture
gap is dominated by profile work, not the smaller temporal exchange error.

The selected driver differentiates Z²/(2G I) only. At fixed displacement,
viscous internal variables and elastic cohesive slip, native elastic-limit
kinematics additionally gives

`delta E_bulk = -sum JxW (Z/G)(tau:S) delta(h/I)`.

A controlled nonuniform representable stress gives +.65894 J/m against -.75516
J/m reduced-fault variation in the actual accepted fine-path step31–32 direction.
This 87.26% opposing contribution survives changing surface spacing and a Q8 bulk
integration comparator (difference 5.96e-6 J/m). Independent finite differences
verify each derivative. This is separate from the retained .0039–.0096% directional
consistent-projection phase-energy discrepancy, and from the 6% h/I source error.
The controlled stress is not claimed to reproduce the physical trajectory. The forced initial-to-final
profile change holds Z at step32 values; it does not replay the unloaded startup.

A closed total-energy phase formulation must address that bulk variation and
the projection adjoint, with a consistent treatment of peak history and physical
internal variables. An explicitly phenomenological reduced-fault driver instead
needs a declared evolving-profile work source. The derivation presents the
candidate total variation and affected equations for review; **no compensating
term or governing revision is implemented**.

[Ledger](../../../benchmarks/reconstructed_fault/propagation/accuracy_energy_evidence/profile-energy.json),
[work plot](../../../benchmarks/reconstructed_fault/propagation/accuracy_energy_evidence/energy-terms.png),
[derivative plot](../../../benchmarks/reconstructed_fault/propagation/accuracy_energy_evidence/profile-derivative.png).

[Actual specification assessment diff](../../../benchmarks/reconstructed_fault/propagation/accuracy_energy_evidence/specification.diff)
adds the unresolved energy variation without adopting a new law.

## Decision, changes and reproduction

Evolving-profile energetics controls the **governing-model** decision. Surface
resolution independently fails the stated accuracy gate; .025 m is an improvement,
not a qualified resolution. Native authoritative stress-history partitioning is
verified, while stress representation remains an additional spatial uncertainty.
Time refinement improves the earlier same-time full-field phase differences
(.02480 to .01721 at t=1) and traction differences (52.86 to 25.63 Pa),
but cannot remove these independent errors. No dt=.015625 coupled run is justified
by these checks. No .1 m run is relabeled as a selected-resolution comparison.

**Production source functions changed: none.** Existing irreversibility,
constitutive, coupling, normalization and projector functions remain byte-for-byte
at their incoming hashes. Diagnostic changes:

| File / functions | Purpose |
|---|---|
| `accuracy_normalization.py`: `project`, `assess` | Frozen consistent projection, fixed integration accuracy, endpoint/interior extrema, native comparator |
| `accuracy_stress.cc`: `MomentCycle::pattern`, `execute` | Existing native updater/transfer sequence, exact-zero controls and recoverable snapshots; standalone plugin, do not co-load BP5 MomentCycle |
| `run_accuracy_stress.py`; `analyze_accuracy_stress.py` | Fresh local fixture/provenance; Maxwell, drift and energy representation comparisons |
| `accuracy_profile_energy.py`: `surface`, `state`, controlled/retained ledgers | Independent energy variations, native time paths and matched spatial measures |
| Benchmark `CMakeLists.txt` | Add only the optional `accuracy_stress` test plugin target |
| This report/derivation, PLAN/STATUS, current design/specification, status/index/source inventory, AGENTS | Record qualification and the unimplemented governing decision |

Commands from this worktree:

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/propagation/accuracy_normalization.py
cmake -S benchmarks/reconstructed_fault/propagation -B build-propagation-integration/plugin -DAspect_DIR=/home/ein/repository/aspect-fault-propagation/build-propagation-integration -DCMAKE_CXX_COMPILER=/opt/gcc/12.4.0/bin/g++ -DCMAKE_BUILD_TYPE=Release
cmake --build build-propagation-integration/plugin --target accuracy_stress -j2
python3 benchmarks/reconstructed_fault/propagation/run_accuracy_stress.py accuracy_stress_final
python3 benchmarks/reconstructed_fault/propagation/analyze_accuracy_stress.py
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/propagation/accuracy_profile_energy.py
python3 benchmarks/reconstructed_fault/propagation/check_accuracy_evidence.py
```

The runner requires a **new label** on repetition. Final stress probe exit 0,
11.29 s, same pinned GCC12.4 core SHA
`44fd12df9313033820771066d1c754f95ab8fa1c9ad8739b17f56c6f86e810dd`;
plugin `ea3a4bf556bf22a31502ae034c3cffb27d00dc5ef318462f69c5ce766dd25454`.
Release/deal.II9.6.2/OpenMPI5.0.6/Trilinos14.2 provenance and input identities
are pinned in [manifest](../../../benchmarks/reconstructed_fault/propagation/accuracy_energy_evidence/manifest.json).
All 3,360 prior artifacts were verified and copied before edits to
`build-propagation-integration/accuracy-energy-baseline`. An initial harness
registration error was repaired; its log is retained as a setup failure, not a
scientific failure. No old failed trajectory is modified or requalified.

The retained bounds, KKT, zero healing, accepted-H publication and mechanical
residual checks remain the earlier fixed-mesh qualification. This assessment
does not loosen their tolerances, rerun unaffected compatibility suites, or
qualify physical rejected-step rollback, coupled restart, particle transport/birth,
general support admission, spatially converged mechanics, stress scaling or P3.
Next requires review of the explicit total-profile phase variation before a
larger coupled accuracy campaign; no such implementation begins automatically.
