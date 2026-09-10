# Bounded stress-transfer correction and history timeline

## Decision and scope

Select continuous Q2 Maxwell-stress compositions with the approved incident-cell
MPI ADD/count transfer. DGQ1 was tested first, but is not locally compatible
with the unchanged phase-field/particle-field batching infrastructure. Do not
broaden this task into DG compatibility work. The shared-DoF last-writer
dependence is removed; no extra computational history lag was found.

Equations, initial histories, surface quadrature, support, full I_h, solver
criteria and terminal publication/rollback code are unchanged. The transfer
rule is an explicit discrete transfer-policy correction, not a change to the
cell-average interpolator. K2.2's execution campaign remains complete, its
reference provisional, and Gate K2 unmet. No K2.3 or convergence run was made;
this result does not establish the cause of the temporal plateau.

All paths below are relative to
`benchmarks/reconstructed_fault/uniform_shear/nonuniform/history-transfer/`
unless stated otherwise.

## DG-first compatibility check

`dg-one.prm` changes only the three explicitly mapped Maxwell-stress fields to
DGQ1: composition degrees `1,1,1,2`, discontinuity `true,true,true,false`.
Their method remains particles, with the existing cell-average interpolator;
`theta_initial` remains continuous Q2/particles.

The bounded Debug check fails before mechanics in
`PhaseFieldHandler::make_sparsity_pattern`: the unchanged implementation uses
the phase component index as a vertex-local DoF index. Preceding DG fields
have no vertex DoFs, so that index is invalid (`phase_field.cc`, calls near
lines 1150, 1178 and 1195). `dg-compatibility.log` preserves the actual assertion
and stack. This check used the retained Debug binary; the relevant source
body is unchanged in the current tree.

A separate source-level barrier is the common-base-element assertion in
`Simulator::interpolate_particle_properties`. Its sole caller in
`assemble_and_solve_composition` batches all particle-mapped compositions,
including the unchanged Q2 `theta_initial`. No mixed-base batching fix,
phase-field sparsity fix or DGQ2 experiment was attempted. DG was rejected
for local implementation compatibility, not for its approximation properties.

## Approved continuous transfer correction

The only production source changed in this task is
`source/simulator/initial_conditions.cc`. Each locally owned cell proposes
one interpolated value per selected local support DoF. Sum values and unit
counts with MPI ADD, then divide at owned DoFs before the existing publication.
Unshared DoFs keep their single proposal. This is neither an L2 projection
nor a change in the particle interpolator. Subsequent physical constraints
still act on the private mechanics working vector, not published history.

The reviewed generic path also transfers continuous particle-mapped
`theta_initial`; it previously had the same competing-write exposure and now
uses the same deterministic averaging. No selected continuous particle field
in this path retains INSERT last-writer semantics. Other future particle-mapped
chemical fields using this generic routine inherit the rule too. Temperature,
non-particle advection paths, CPDI construction, history commits and surface
projection code are not changed. Mixed-base batching remains unsupported.

The extra storage is one system-layout distributed contribution-count vector
during transfer. There is no persistent history/cache state or new public API.
`current_design.md` and `specification.tex` now state the transfer rule.

## Frozen-data verification

The original INSERT source, audit plugin and comparison are retained in
`insert-baseline.tar.gz`; original output directories `one/` and `two/` are
untouched. The updated audit uses the actual configured interpolator and
matches forward transfer against production-published history before comparing
forward/reverse traversal and one/two ranks. Reversed traversal is a test-only
replay of the transfer loop, not a new production traversal API.

The 8x32-cell fixture contains 2,304 identical physical particle tuples on
both rank counts. The decisive nonconstant history is
tau_xy = 1500 + 20 sin(8 pi x) + 10 y Pa. Constant 200 is a separate control.
The actual reconstructed-fault Stokes assembler supplies the constrained weak
load; subtracting its zero-history load isolates the frozen Maxwell term.
Independent QGauss(4) integration uses the realized constrained FE history,
homogeneous Newton constraints and MPI assembly, not an assumed exact analytic
reproduction by cell averaging.

| Measurement | Result |
| --- | ---: |
| Constant reproduction, all traversal/rank combinations | exact field reproduction |
| Maximum nonconstant published/working history difference | 2.274e-13 Pa |
| Forward/reverse weak-load L2 difference, one rank | 5.860e-14 |
| Forward/reverse weak-load L2 difference, two ranks | 5.775e-14 |
| One/two-rank weak-load L2 difference, forward/reverse | 8.029e-14 / 7.834e-14 |
| Nonconstant weak-load L2 norm | 2.9459806172 |
| Independent weak-integration discrepancy | at most 3.60e-13 |
| Constant weak-load norm | at most 1.65e-14 |
| Unshared cell-center support values versus cell means | exact, all 256 cells |

Weak-load entries above are coefficients in the fixture's assembled FE
normalization, not pointwise stress errors. Published and constrained fields
are intentionally distinct: the periodic constraint lift changes the smooth
field by 1.93128 Pa RMS, identically for both traversals and MPI partitions.

Approximation of the prescribed smooth history improves but is not exact:
published/working RMS errors are 1.62336/1.56857 Pa, versus saved INSERT errors
2.74664/2.59639 Pa. The cell-center check verifies unchanged single-contribution
behavior; it is not a claim that the incompatible mixed-DG fixture passed.
See `average-comparison.json`, `average-comparison.log` and `average-one/`,
`average-two/` for the full measurements and independent controls.

## Actual history-consumption timeline

`replay-plan.md` fixed the scope before the run: reuse the existing 32x128
coarse K2 fixture, initialization and two 0.5-s steps only. The only fixture
changes are diagnostics, end time 1 s and particle VTU output. A read-only,
zero-contribution assembler installed through the existing `set_assemblers`
signal observes the actual first cell assembly. The ordinary
`pre_assemble_stokes_system` signal is bypassed by the coupled path and is not
used as a misleading observation point.

For each step, `timeline.cc` records all 4,096 cells, 36,864 bulk QPs and
36,864 particles, including 22,848 active surface parents. Stable IDs join
pre-advection, first-assembly and terminal-commit records. Pre-advection
records have the current step/time but previous positions; first-assembly
records have current positions after advection/transfer. Cell IDs and cached
fault associations are retained. Subsequent assembler visits require particle
stress to remain exactly frozen throughout mechanics.

The actual surface code (`surface_system.cc`, old-stress input construction)
reads these same immutable parent particle properties directly. This surface
provenance is checked by source inspection plus runtime immutability, not a
separate callback inside every surface constitutive evaluation. Bulk inputs
are observed directly in the first assembly's working vector and scratch.

`check_timeline.py` independently rebuilds incident-cell nodal averages from
the actual post-advection particle data and applies the fixture's existing
periodic history constraints. Both published and first-assembly working nodal
histories match exactly, for every stress component and all three steps.

| Actual step/time | History consumed by mechanics | Terminal particle change |
| --- | --- | ---: |
| 0 / 0 s | supplied tau_0 = (0,0,1500) Pa | exactly zero |
| 1 / 0.5 s | retained tau_0, exact ID-wise match | max 377.031554 Pa |
| 2 / 1 s | committed tau_1, exact ID-wise match | max 91.337576 Pa |

At zero the physical elapsed step is zero; the numerical Maxwell initialization
interval is 2 s. It does not overwrite the supplied particle stress. At 1 s,
consumed history differs from initialization by up to 377.031554 Pa; first
working tau_xy ranges from 1123.108051 to 1129.507906 Pa. Thus step 2 consumes
step 1's commit, not step 0's input. There is no demonstrated extra lag.

The verifier also reads the actual VTU arrays and TIME/position/ID data:

- Bulk `tau_xx/tau_yy/tau_xy` compositional arrays are the **published old-history
  input**, not the privately constrained field and not newly accepted stress.
- Particle `maxwell stress_0/1/2` arrays are the **terminal committed current
  stresses** at real steps; at zero they retain the supplied initial stress.

Arrays agree exactly after the output's Float32 conversion; maximum stress
quantization is 6.104e-5 Pa. No live history was refreshed for visualization,
and no Maxwell update was reevaluated on already committed inputs.

## Changed short-run results

Reuse `nonuniform/domain-convergence/space32` as the pre-correction comparison;
it was not rerun. All three phase-profile and segment CSV pairs are byte
identical, and the surface I_h arrays are identical. Initial surface projections
and the accepted V/Theta/C/weak-q/slip arrays at 0 and 0.5 s are unchanged.
Although the generic Theta FE transfer also changes policy, initialized surface
Theta in this fixture comes from particle data, not that published FE field.

At 1 s, RMS changes relative to the saved same-mesh result are:

| Quantity | RMS new minus saved |
| --- | ---: |
| V | 3.25644e-10 m/s |
| Theta | 1.87444e-5 s |
| retained cohesive C | 1.50201e-6 Pa |
| actual weak traction q = M^-1 Q | 6.36592e-5 Pa |
| accumulated slip | 1.62822e-10 m |

The first change occurs when nonconstant committed stress is transferred for
step 2, as expected. These are bounded correction comparisons, not convergence
errors against an exact reference. No temporal-plateau conclusion follows.
`timeline-verification.json` contains means and total/mean-removed differences.

## Reproduction, cost and verification limits

Build from repository root with `cmake --build build-pf-cpdi --target
aspect.exe.release -j4`; plugin builds use `cmake --build <audit>/build --target
history_transfer.release history_timeline.release -j4`. Build logs are retained.
The Release executable was rebuilt; the earlier file-target no-op was caught
before corrected tests. The current Debug executable was not rebuilt.

Simulation commands below run from `<audit>/build`; ASPECT is the absolute
repository executable `build-pf-cpdi/aspect-release` (Debug only for DG):

| Command | Result / measured ASPECT wall time |
| --- | --- |
| `timeout 120 ASPECT-debug ../dg-one.prm` | exit 134, captured compatibility assertion |
| `timeout 120 ASPECT-release ../average-one.prm` | exit 0, 1.374 s |
| `mpirun -np 2 timeout 120 ASPECT-release ../average-two.prm` | exit 0, 3.938 s |
| `timeout 120 ASPECT-release ../replay.prm` | exit 0, 38.01 s |
| `timeout 120 ASPECT-release --test 'Stage-I*'` | 52 assertions / 10 cases passed |

From the repository root, run `OPENBLAS_NUM_THREADS=1 timeout 120 python3
<audit>/compare.py --prefix average- --require-invariance` (passed, about
0.22 s), and `OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/aspect-history-mpl
timeout 120 python3 <audit>/check_timeline.py` (passed, 5.48 s). The latter
requires the already installed VTK XML reader; it imports only those VTK
modules, not optional OpenVR. No dependency installation was needed.

All exploratory executions stayed below 120 s each and below the aggregate
600-s budget. No timed-out simulation was retried. Positive replay checks
require all three final nonlinear acceptances and 17 fresh linear residual
checks, not merely exit zero. Stage-I safeguard units pass. Full restart/
rollback integration fixtures, 3-D, AMR transfer, mixed-base DG and the full
ASPECT suite were not rerun; there is no new claim of verification for them.
The timeline replay is one-rank; traversal/partition invariance is tested
separately on one and two ranks in the frozen audit.

## Changed files and recoverable evidence

This task changes `source/simulator/initial_conditions.cc`, the transfer-rule
paragraphs in `current_design.md`/`specification.tex`, this report and progress
links. Audit changes are `history_transfer.cc`, `compare.py`, `CMakeLists.txt`,
`README.md`, plus new `dg-one.prm`, `average-one.prm`, `average-two.prm`,
`timeline.cc`, `replay.prm`, `replay-plan.md`, and `check_timeline.py`. Generated
logs/CSV/VTU/JSON evidence remains beside them. Existing unrelated working-tree
changes are preserved; no commit was requested or made.

Recoverable original source/plugin evidence: `insert-baseline.tar.gz` and
the existing `nonuniform/performance/accepted-cartesian-baseline/` snapshot.
Tested corrected SHA-256 identifiers:

```text
source/simulator/initial_conditions.cc
d4d9033d385d6a2e267ada19b63ec67de369ff09d7949bfce1cc1a7c65440db7
build-pf-cpdi/aspect-release
32ecc3789e65cfcb01b56bb96c26ec47719e2a667a625781eda1c6f0dc8ec120
history_timeline release plugin
e205bb808cc56404a81811221a82acdee6f774b031cf948865db7566872cef04
```

No unfinished DG workaround or lifecycle mutation remains. The selected
continuous representation is ready for review; broader DG compatibility and
the still-provisional K2 reference are separate decisions. Do not begin K2.3
or rerun convergence studies automatically.
