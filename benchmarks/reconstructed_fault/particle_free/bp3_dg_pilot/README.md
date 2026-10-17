# Bounded stationary-history DG BP3 pilot

This is a prepared experiment, not a first-event or propagation run. Unlimited,
domain-free DGQ2 is the selected candidate. Neither it nor the retained particle
controls is an accuracy reference. See the measured
[qualification report](../../../../doc/reconstructed_fault/particle_free/PF6_MATURE_PILOT_REPORT.md)
and the copied `qualification-summary.json` in a prepared directory.

The prepared input retains the physical 150×50 km Box, straight 60° mature fault,
original anchors, native stationary phase (peak .6, ell=20 m), homogeneous bulk
coefficients and physical 15–18 km surface transition. Requested surface spacing
is 10 m (5776 nodes in the full-geometry initializer). Mature cohesive resistance
and storage are zero; auxiliary Z is retained independently. Gauss6 mechanics,
Gauss3 native source, Helmholtz20 and all limiter settings remain fixed.
Temperature/compositions are static. `BP3 stationary samples` fixes diagnostic
points; domains, CPDI and face data are disabled. Point stress is never the
accepted mechanical history. The separate FE server package is unchanged.

The graded candidate requests finest/coarsest cells 3.90625/1000 m. Retained
estimates are about 552084 cells and 8833344 diagnostic points; these are not a
new full-mesh measurement. Full-geometry mechanics, memory/cost, weakening
transients and continuum spatial accuracy remain unqualified. Reserve suitable
server memory and at least 50 GB scratch for dense tensor/FE exports and native
checkpoints; the earlier 35–100 GiB memory range is an estimate, not a DG peak-RSS
measurement. Small-control timing does not qualify MPI scaling.

## Build and prepare

Use the matching ASPECT source checkout and one consistent C++17/MPI/deal.II
stack. Local qualification used GCC12, deal.II9.6.2, Trilinos14.2, p4est2.8.7 and
OpenMPI5.0.6. Rebuild on the server if the workstation binaries' ABI does not
match. Do not force an older libstdc++ over newer dependency requirements.
The generated directory includes the base revision, source inventory,
`source.patch`, and `untracked-sources.tar.xz`. For a clean base checkout, apply
the patch and extract the untracked source inputs at its root; initialize the
base repository's submodules. The current unit-test parameter printer already
uses the deal.II `PRM` spelling.

Build ordinary BP3 with **neither** `ASPECT_BP3_LOCAL_BOTTOM_TEST` nor
`BP3_LOCAL_OSCILLATION_TEST`. Build its observer with `BP3_TRANSITION_CHECK=ON`:

```sh
cmake -S benchmarks/reconstructed_fault/bp3_native/server/plugin -B "$BP3_BUILD" -DAspect_DIR="$ASPECT_BUILD"
cmake --build "$BP3_BUILD" -j 4
cmake -S benchmarks/reconstructed_fault/bp3_native -B "$OBSERVER_BUILD" -DAspect_DIR="$ASPECT_BUILD" -DBP3_COUPLED_PLUGIN="$BP3_BUILD/libbp3_restore_150x50.release.so" -DBP3_TRANSITION_CHECK=ON
cmake --build "$OBSERVER_BUILD" --target bp3_coupled_checks -j 4
python3 benchmarks/reconstructed_fault/particle_free/bp3_dg_pilot/prepare.py "$PILOT_DIR" --aspect "$ASPECT_BUILD/aspect-release" --bp3-library "$BP3_BUILD/libbp3_restore_150x50.release.so" --observer-library "$OBSERVER_BUILD/libbp3_coupled_checks.release.so" --qualification-report benchmarks/reconstructed_fault/particle_free/evidence/pf6_mature_pilot/measured-summary.json
```

Configure compiler paths explicitly when the site's defaults differ from the
dependency stack. Pin the matching MPI runtime as well as its launcher. A new
server build needs the unit/short lifecycle preflight; local results do not
establish another stack's numerical behavior. From the prepared directory,
validate both inputs with its executable's `--validate` option. Regenerate the
package when changing its location; native inputs contain absolute paths and
`launch.py` verifies their hashes.

## Explicit bounded launch

Preparation does not launch MPI. The local delivered package is parser-validated;
no full-geometry mechanical trajectory has been launched in this task.

```sh
python3 launch.py --ranks 4
# Only for a later continuation within the same caps:
python3 launch.py --ranks 4 --resume
python3 diagnose.py output --output pilot-diagnostics.json
```

`launch.py` accepts the qualified rank counts 1/2/4, records dependency resolution
and environment, refuses an existing fresh output, and enforces a 7500 s hard
wall bound. Native BP3 requests a graceful stop after 7200 s at an accepted
state, at most eight positive accepted steps, and end time 400000 s. Physical
dt is capped at 50000 s; the retained reconstructed-fault controller may shorten
it further (maximum logarithmic state change .2, existing growth/cutback rules).
The separate step-zero constitutive interval remains **4000000 s**. Native
checkpointing occurs after each accepted step. Rejection never publishes the
DG history; successful acceptance publishes once. Resume keeps the original
step/time caps and per-launch wall limits and does not authorize a longer event.

The 50000 s cap is the finest local temporal-control level, not proof of 1%
endpoint accuracy for this different geometry/material trajectory. The measured
local endpoint temporal screen still fails slightly. Surface Q1 interpolation
of the intended initial state also produces a 49.7 Pa interior prestress imbalance
at the 10 m control points, despite nodal balance within 1.87e-8 Pa. Neither is
removed with prestress corrections, filtering changes or dissipation.

## Diagnostics and decision

`authoritative-stress-N.csv` (or rank-prefixed files) contains accepted T and
reconstructed tau, all embedded tensor components, pressure, phase and chi on
native Gauss3 diagnostic points. `dg-coefficients-N.csv` retains all nine modes;
`physical-fe-N.csv` retains physical FE fields. `surface-weak-N.csv` retains the
actual mass, shear/raw/filtered weak loads, friction and residual. The full fault
has descending native node order in down-dip coordinates; do not assume a uniform
or ascending node ordering. Use 1000 m endpoint windows, covering the retained
811.5 m completion influence in the full geometry.

`diagnose.py` subtracts each native mass times its own constant background before
reporting perturbation loads. It reads the full tensor, including zz and both xy
contributions, and reports unfiltered point tractions without clipping. For point
CSV export, use the supplied `export_tractions.py` with one authoritative stress
file and a fresh output filename. These postprocessors do not update history or
apply a second Maxwell step. Dense diagnostics can have substantial IO cost.

Keep original errors and `strict_reference_match_passed=false`. Judge the pilot
from authoritative tensors, perturbation loads, RSF state/rate, residuals,
publication/recovery and measured cost. Do not promote a smoother traction plot
to an accuracy result. Phase evolution, transport, limiting changes, AMR, moving
geometry and the full first event remain subsequent selections.
