# Running the bounded control

[Guide](../TIP_TREATMENT.md) · [current qualification](../TIP_CONSOLIDATION_REPORT.md)

## Supported settings

The active inputs are
[`finite_tip.prm`](../../../../benchmarks/reconstructed_fault/particle_free/bulk_tip/finite_tip.prm)
and [`localized.prm`](../../../../benchmarks/reconstructed_fault/particle_free/bulk_tip/localized.prm).
Both select the semantic `phase_field` composition, discontinuous degree two and
`Discretization / Phase basis = Bernstein`. Phase initializes through the ordinary
composition initial-condition model. Bernstein is the supported/default phase
basis. Q1, CPDI and native nodal phase are retired; ordinary ASPECT fields keep
their own FE choices. Old phase representations require their retained binary.

| Setting | Actual role |
|---|---|
| `Fault constitutive mode = bulk cohesive` | Bulk recoverable tau/A and one bulk fracture inventory |
| `Bulk cohesive flow rule = bulk tip sections` | Fixed-direction signed section RSF closure |
| `Bulk tip time update = backward Euler` (default), or `midpoint` | Focused numerical choice for integrated native Bernstein; stage equations and method admission in [time integration](time_integration.md) |
| `Bulk tip solver = ASPECT` (default) | Distributed coupled solver under ordinary simulator ownership |
| `Bulk history representation = native tip stress` | Explicit physical history meaning; old `DG trial stress` spelling in `finite_tip.prm` remains accepted |
| `single Advection, iterated Newton Stokes` | Selected simulator scheme; tip branch bypasses ordinary phase transport |
| `Stokes solver type = block AMG` | Required reference inverse; `run.py` supplies it for the integrated route |
| Static temperature and auxiliary compositions | Required by the fixed-reference control; stress compositions are inert, not physical tau |
| `Surface pressure = 1000` | Prescribed frictional compression in Pa; independent of the solved pressure gauge |
| `Bulk tip initial age = 10` | Age in seconds on every section, including intact material |
| `FE bulk normal quadrature points = 6` | Tensor Gauss6 native point-history/mechanical rule |
| `Equilibrate bulk tip seed` | False in `finite_tip.prm`, true in `localized.prm`; fracture-only zero-clock obstacle solve |

The admitted physical route is 2-D, fixed Cartesian aligned mesh and direction,
matched homogeneous coefficients, full prescribed velocity boundary, no
reconstruction, transport, spin or AMR. Ordinary particle managers are unnecessary.
The phase basis does not change these restrictions. No second phase growth solve
or SSPRK3 phase stage runs inside this tip calculation.

The replicated `benchmark direct` solver is retired. `--linear-oracle` checks
the integrated solver against an independent small direct/manufactured-direction
reference in benchmark support, using the same native Bernstein basis.

## Physical fixtures

SI units and ell=.25 m are common. G=1 MPa, eta=100 MPa s,
Gc=26.6666666667 J/m², calibration cohesion=1000 Pa, radiation=1000 Pa s/m,
mu0=.6, a=.015, b=.02, V0=.001 m/s and Dc=.01 m. Thus b>a gives steady velocity
weakening; the separate endpoint monotonicity guard still applies. Fracture has
natural boundary conditions. These fixed-reference runs have no transport inflow.

| Input | Domain / coarse cells | Declared seed | Boundary velocity |
|---|---|---|---|
| `finite_tip.prm` | [0,2]×[0,1] m / 8×4 | Center (1,.5), half-length .5, half-width .25, peak .2 | (.02(y−.5),0) m/s |
| `localized.prm` | [0,4]×[0,2] m / 16×8 | Center (2,1), half-length .8, half-width .2, peak .8 | (.01(y−1),0) m/s |

Each seed formula is a product of truncated quadratics. The physical initial
field is its FE interpolant, then bounded initialization and, for the localized
fixture, fracture equilibration. The declared analytic support/peak is not a
promise about the equilibrated FE seed. Refinement holds physical inputs fixed;
initial interpolation and equilibration are mesh dependent and must be compared.
The optional remote-margin diagnostic measures actual coupled intact driving;
nominal homogeneous loading is not a substitute.

## Build and short run example

The paths below are separate candidate directories; do not rebuild a protected
executable or reuse an existing run label. Rebuild the checker against the same
ASPECT headers/build, especially after changing FE or parameter layout.

```sh
cmake -S . -B build-particle-free/bulk-tip/doc-control \
  -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=/opt/dealii/9.6-local \
  -DCMAKE_CXX_COMPILER=/opt/gcc/12.4.0/bin/g++ \
  -DASPECT_UNITY_BUILD=ON -DASPECT_PRECOMPILE_HEADERS=ON -DASPECT_WITH_PYTHON=OFF
cmake --build build-particle-free/bulk-tip/doc-control -j4
cmake -S benchmarks/reconstructed_fault/particle_free/bulk_tip \
  -B build-particle-free/bulk-tip/doc-checks -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/gcc/12.4.0/bin/g++ \
  -DAspect_DIR="$PWD/build-particle-free/bulk-tip/doc-control"
cmake --build build-particle-free/bulk-tip/doc-checks -j4
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py \
  doc-tip-control 140 python3 benchmarks/reconstructed_fault/particle_free/bulk_tip/run.py \
  doc-tip-control --seconds 120 --dt .005 --end .01 \
  --executable build-particle-free/bulk-tip/doc-control/aspect-release \
  --checker build-particle-free/bulk-tip/doc-checks/libbulk_tip_checks.release.so
```

This is a short usage example, not a new qualification result or a growth
campaign. Run heavy jobs sequentially, ≤4 build workers/MPI ranks and one numerical
thread/rank. The wrapper records timeout and thread settings. `run.py` creates
a fresh directory and records inputs, executable/checker hashes, source revision,
command, runtime and status; a source revision alone does not identify a binary.

| Runner option | Use / boundary |
|---|---|
| `--fixture`, `--dt`, `--end`, `--refinement`, `--ranks` | Select the input and bounded mesh/time/rank control; ranks are 1, 2 or 4 |
| `--unloaded` | Replace the supported fixtures' boundary shear by zero |
| `--intact` | Replace the original `finite_tip.prm` seed expression by zero; not a generic seed override for `localized.prm` |
| `--remote-margin` | Add the benchmark remote-driving diagnostic stop |
| `--phase-balance` | Export native fracture/driving rows, independently of using a driving-margin termination |
| `--integrity-stop` | Stop if actual phase exceeds 1e-6 in the documented remote or boundary region; uses the actual box geometry and implies the required observer |
| `--rheology-audit` | Run the focused aging/RSF mathematical audit through the benchmark checker |
| `--wall-hours` | Add ASPECT's graceful wall-time limit; retain the runner's outer `--seconds` cap and last accepted checkpoint |
| `--tangent`, `--tangent-step` | Coupled derivative check at the chosen step |
| `--linear-oracle`, `--linear-oracle-step` | Small integrated-system direct/manufactured-direction check |
| `--rejection`, `--rejection-step` | Checker trial/rollback/retry test at the chosen step |
| `--resume-from`, `--snapshot-id` | Copy retained restart files into the fresh run; source remains untouched; require matching law, basis and other metadata |
| `--restart-dt` | Reduce a pending restart interval through the existing benchmark signal; preserve and record the parent clock |
| `--export-parent`, `--import-parent` | Explicit serial same-mesh benchmark nodal→Bernstein parent conversion; not ordinary checkpoint migration |

Ordinary restart rejects nodal/Bernstein mismatch before deserialization. End-time
clipping and the restart loop can affect the pending interval; inspect recorded
clocks and use the exact schedules in the reports when reproducing a comparison.
Retain failed evidence, and consult [limits](limitations.md) before extending a run.

## Time-update and mechanical-screen controls

Use `run.py --time-update midpoint` for the candidate. Midpoint requires native
Bernstein and the integrated solver; no legacy/direct-oracle fallback occurs.
The archive's physical-law schema stays unchanged. The separate numerical-method
metadata rejects ordinary BE↔midpoint restart. Only an explicitly labeled
benchmark parent import can change methods at an accepted state.

`--compact-width 1.2 --fixed-phase` selects the bounded compact-loading screen
on `localized.prm`. The checker uses an existing assembler-setup signal to set the
phase upper bound equal to its accepted lower bound; the same mechanics, section
RSF and bulk history still evolve. This constraint is benchmark-only and is not
a new physical cohesive law. `--intact` removes either fixture's seed.
`--load-factor` scales only the declared compact boundary velocities. Fresh
histories are required for a new loading; do not relabel old checkpoints.

Endpoint phase/stress/age and last-stage rate/resistance are distinguished in
[time integration](time_integration.md). Time-dependent velocity constraints
remain right-endpoint sampled and are outside the midpoint accuracy claim.

`--boundary-stop` additionally stops a growing compact-loading pilot when the
previously intact .25 m boundary strip acquires phase above 1e-12. This is a
benchmark diagnostic stop, alongside the retained remote gate, not a phase cap.
Fixed-phase screens export these margins without using them to terminate.

## Selected synthetic uniform-shear preflight

The [synthetic benchmark](../SYNTHETIC_PROPAGATION_BENCHMARK.md) specifies
`bulk_tip/synthetic_propagation.prm`, its scales, expected range, mesh and cost.
Use the documented runner with `--fixed-phase --phase-balance`: the former
freezes the accepted obstacle in the integrated solver, the latter exports
actual fracture/driving rows without assuming the old 4×2 m region definitions.
The new postprocessing script uses explicit masks for the selected 8×4 m box.
Bare execution without the fixed-phase hook would enable growth and is not
the completed preflight. Existing checkpoints must not be resumed under its
changed Gc, length, confinement, Dc, age or geometry.

The overnight growth controls use `--phase-balance --integrity-stop`, with no
fixed-phase flag. The checker now obtains box origin and extents from the
geometry model. For the selected 8×4 m synthetic body the remote region is
`abs(x-4)>3 || abs(y-2)>1`; the boundary strip is within .25 m of any wall.
The actual-integrity threshold is distinct from the older .95 normalized-driving
margin and from `--boundary-stop`'s tighter historical threshold. None changes
the phase obstacle or caps the physical field. See the
[material cleanup and midpoint/spatial comparison](../MATERIAL_MODEL_CLEANUP_AND_OVERNIGHT_RESULTS.md)
for tested binary identities, run labels and current limits.

For exact failed-parent recovery, `--end-step N` selects ASPECT's existing
maximum-timestep termination criterion and `--checkpoint-steps N` selects its
existing accepted-step checkpoint frequency. Keep the physical `--end` equal
to the original trajectory when recovering a parent: changing it can change
final-step clipping. These are benchmark controls, not new physical parameters.
