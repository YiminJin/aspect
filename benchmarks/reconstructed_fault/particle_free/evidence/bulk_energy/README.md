# Aligned wide work/temporal evidence

[Report and criterion](../../../../../doc/reconstructed_fault/particle_free/BULK_ENERGY_REPORT.md).
`assessment.json` contains all three measured cases, first-growth brackets/nodes,
common-time response differences, observed orders and the proposed reporting
criterion. `*-energy.csv` contains every increment and cumulative physical/BE
budget. `*-fields.npz` contains the common-mesh physical phase and original-QP
angular averages. The PNG/PDF figures use identical phase-increment scales.

Only two native trajectories were added: h/2 and h/4, final time .16 s. The
old h result is retained and reused. `parameter-comparison.json` verifies that
parsed values differ only in the two timestep caps, diagnostic DSO and output
location. Initial phase fields are bitwise equal; no accepted history was
transferred. The native core/constitutive sources and all old binaries/DSOs are
unchanged. The two new runs overlap in local execution; their logged wall times
are not a performance comparison. No server job or additional native control
was launched.

The h/4 case has 256 regular steps plus the native 6.66e-16 s clock-closing step.
All 257 publications and its tiny dissipation are included. Old/fine data are
matched at common physical times; the last state is used if native roundoff
produces two endpoints at the same physical time to the declared precision.
The original 1% endpoint-screen failures stay in the data and old reports.
The recommended physical-work criterion is a reporting check, not a changed
solver/constitutive acceptance rule.

The test-only `ASPECT_BULK_ENERGY_FIXTURE` DSO adds accepted unconstrained nodal
boundary reactions, using full stress, actual physical pressure, native FE
shapes and constraints. Force sign is work supplied to the specimen. `force`
in `boundary-reactions-*.csv` is N/m of out-of-plane thickness, and `increment`
is m. Q=sum F_i*(y_i-.5) for x components is J/m per unit shear strain. It is
nonzero on the fixed-displacement hold. Reactions are averaged for observable
work quadrature only; no constitutive history is interpolated or modified.

Compact `quadrature-q-*.bin` stores native q in Pa, float64, active-cell/Morton
order then native Gauss8 tensor order (x fastest). Phase files retain physical
Q1 nodal data every step. Dense full stress/history/solution exports are at
initialization, loading stop and terminal time. The old baseline raw DG blocks
use active-cell order, while its local history std::map uses numeric CellId
order. Independent native QP exports validate that distinction. The initial
reader error was preserved in the build-area analysis log and corrected before
results. A second reader assertion assumed exactly 256 quarter steps; it was
corrected to retain and report native terminal clock closure. No run or input
was changed for either reader correction.

The independent checker reconstructs baseline full stress at every state,
fracture/cohesive/rate/increment terms in every case, and refined full bulk
stress/storage at the declared dense snapshots. Refined per-step stress BE
remainders remain native ledger data; the checker explicitly marks their
independent reconstruction as unavailable rather than reporting a zero error.

Build the additional diagnostic target in this current worktree:

```sh
cmake -S benchmarks/reconstructed_fault/particle_free \
  -B build-particle-free/bulk-energy/plugin \
  -DAspect_DIR="$PWD/build-particle-free/evolving-phase/native"
cmake --build build-particle-free/bulk-energy/plugin --target bulk_energy_checks -j 2
```

The retained qualified core is
`build-particle-free/bulk-source-history/native/aspect-release`, SHA256
31ad1e59820079e712bdae23732697394cbfe36e8377001fa50cdac2a05b8d5a.
Use the matching new DSO with it. In a **fresh output directory**, direct inputs
are `bulk_cohesive/energy_half.prm` and `energy_quarter.prm`, for example:

```sh
env ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1 \
    ASPECT_BULK_CHECK_TANGENT=1 OMP_NUM_THREADS=1 OMP_PROC_BIND=false \
    build-particle-free/bulk-source-history/native/aspect-release \
    benchmarks/reconstructed_fault/particle_free/bulk_cohesive/energy_half.prm
```

Do not overwrite the existing runs. `logged/*.json` preserves actual commands,
working directory, environment, bounded timeouts and exit codes. Both reusable
parameter files also pass `--validate`. No native core rebuild was needed.

From a replay worktree root containing the retained data paths:

```sh
python3 benchmarks/reconstructed_fault/particle_free/bulk_cohesive/assess_energy.py \
  /tmp/bulk-energy-replay
python3 benchmarks/reconstructed_fault/particle_free/bulk_cohesive/plot_energy.py \
  /tmp/bulk-energy-replay
```

Python analysis uses NumPy, SciPy and Matplotlib. The data archive includes the
required old/fine native exports, actual inputs, benchmark sources and one-step
failure-reader logs. It is an independent data replay bundle, not a complete
ASPECT/server source distribution. Extract relative paths into an empty replay
tree, never over differing existing files. `archive-manifest.json` hashes every
member; `archive.json` records archive and verified-member hashes. Original raw
runs and native checkpoints remain under `build-particle-free/bulk-energy/` and
the original baseline location; redundant rolling checkpoint files are omitted
from this energy replay archive. Native ownership/rejection/restart qualification
is inherited within its documented scope from the preceding source-history task.

This test resolves the specified fixed-mesh work/temporal question. It does not
clear spatial/angular accuracy, change the law/time integrator/defaults, qualify
physical propagation, or alter BP3. strict_reference_match_passed remains false.
