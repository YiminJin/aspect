# Maintained BP3 runtime

One source package supports stationary straight 2D BP3 faults with frozen mature
phase field, core automatic endpoint completion, continuous Q2 stress fields,
native LLS history interpolation and mechanical normal-traction feedback.
Accepted output observes committed state; it never performs a constitutive update.

Use the [fresh production candidate](../production/README.md). **Resolved graded
runs remain blocked by the Section-3 automatic-completion boundary-lattice
restriction.** The old `bp3_150x50_raw/filter20/filter40/first_event.prm` files and
`fixtures/` are historical, version-pinned inputs/evidence. In particular,
`first_event` resumes an old checkpoint; it is not a fresh-run template.

## Model data and setup

The native prescribed-fault file is the only external model data file. Box
settings and ordered fault contents define geometry. One straight, strictly
ordered polyline with constant peak must intersect the interiors of the top
and bottom faces. Reversal preserves physical thrust; anchors/native resampling
are not canonicalized. Unsupported curves, multiple faults and corner/tangent
contacts fail explicitly. The horizontal material extension and 15–18 km
transition remain intact; only labeled local fixtures allow truncation.

`profile.cc` prepares a cached loading primitive from the live phase-field
factory and energetic law before its first consumer. It does not depend on
monitor execution. Initial phase, startup/newborn frozen H and core completion
use that same material/profile configuration; discrete mechanical Ih is not
replaced by the loading integral. No completion, profile or mesh table is read.

`BP3 fault support` retains the established support-band/exterior grading with
native smoothing. Initial-global tagging runs before particles exist: choose
the global level for the finest cells, the minimum level for the coarsest,
and zero initial adaptive refinement. Later AMR is not selected. Geometric
coverage, resolution and MPI count checks replace leaf-ID assertions.

Build with `cmake -S .../bp3/plugin -B .../bp3/build-maintained
-DAspect_DIR=/absolute/aspect-build -DCMAKE_BUILD_TYPE=Release`, then
`cmake --build .../bp3/build-maintained -j 3`. Load only the resulting
`libbp3_restore_150x50.release.so` using the matching executable stack. The library
and registered monitor names are retained for API compatibility; they do not
restrict the dip to 60 degrees.

## Particle history and outputs

The candidate uses regular 4×4, 12–24 native point-density management and
`BP3 history linear least squares`. The adapter resolves H/Maxwell components
by name and disables boundary extrapolation. `BP3 frozen crack driving force`
shares startup/newborn initialization, retaining material Hc outside the active
region. Maxwell inheritance stays native. Survivor H, MPI audit transport,
rollback and the manager-owned RNG/checkpoint lifecycle remain unchanged.

- `accepted_steps.csv`, `stations.csv`, `restored_growth.csv`: existing accepted
  summaries and mechanical diagnostics.
- `particle_summary.csv`: global accepted count, surviving births since the
  particle backup, losses from the incoming population (native removals **or
  outflow**, not MPI migration), H range and stress-component extrema. Step-zero
  event counts are zero because no timestep backup exists; population includes
  startup management. Diagnostic scalars refresh on every backup, including
  after restart; no physical state or extra global ID map is stored.
- `profiles/fault_*.csv` and `profiles.csv`: signed-slip/state profiles on the
  existing slip/time schedule, including initial/final/event/heavy forcing.
  There is no every-step `cumulative_slip.csv` writer.
- `BP3 / Write bulk and particle visualization`: selects native visualization
  and particle writers on the existing heavy schedule. Historical default true;
  the fresh candidate explicitly uses false. Fault output and complete
  checkpoints remain enabled. It does not change the schedule references.
- `BP3 restored monitor / Write detailed diagnostics` and `BP3 / Audit full state
  every step`: opt-in per-step field/particle/work dumps, both false by default.
- `first_update_maxwell.csv`: mandatory first-real-step publication check.

`Last accepted step`, `Graceful wall seconds`, profile and heavy intervals retain
their existing meanings and never control the physics timestep. The core
log-state bound is positive, finite and unweighted. The largest-finite-double
sentinel disables it; zero and the word `infinity` are rejected. The candidate's
0.1 bound and 0.5 nonlinear-failure cutback are explicitly provisional pending
identification of the latest server input; see its settings table.

## Restart and model identity

Identity v4 records actual ordered geometry, live primitive data, material and
phase-field parameters, loading convention, mesh/refinement policy, composition/
discretization, native particle settings, effective limiter and normal filter.
Parameter values are compared as exact resolved strings; numerically equivalent
re-spellings may conservatively reject. Paths and output schedules are excluded.
Time/solver settings are printed but remain caller-adjustable on restart, as in
the existing retry tests. This is a model compatibility guard, not a universal
bitwise-replay promise across changed solver policies/builds.

New-plugin checkpoints retain the existing v6 accepted-history archive and
manager RNG payload. Only derived caches rebuild. Older model identities require
the matching original plugin; missing material identity is not silently accepted.
A changed output cadence or visualization selection retains saved last-written
references and committed slip/Theta/H/stress. Checkpoints include physical
history regardless of visualization selection.

Use `bash ../branch_output.sh PARENT CHECKPOINT_ID NEW_DIRECTORY` from this
folder for an older selected checkpoint, then resume into that new directory
with the matching model and rank count. The helper copies checkpoint metadata
and indexed profile payloads without changing the parent. New particle summaries
join the metadata prefix and its newer-than-checkpoint checks. Do not reuse a
directory containing outputs newer than the selected checkpoint.

## Source responsibilities

| File | Responsibility |
|---|---|
| geometry.cc | Native geometry parsing, validation and physical coordinates |
| configuration.cc | Resolved parameter reporting and model compatibility description |
| profile.cc | Transient live loading primitive; no mechanical Ih ownership |
| mesh.cc | Geometry-driven startup tagging and invariants |
| bp3.cc | Initial fields, loading registration, preparation/solver observers |
| particle_initialization.cc | Shared frozen-BP3 H initializer; generic fallback retained |
| particle_history.cc | Native limiter policy and birth/survivor audit lifecycle |
| bottom_constraint.cc | Existing optional experimental fault-parallel constraint; production remains full |
| monitor.cc | Filter setup, restart identity and traction/growth observations |
| work_audit.cc | Mandatory native/inert/Maxwell checks and optional replay |
| output.cc | Accepted histories, schedules, compact summaries and checkpoint metadata |

Environment guards still reject inherited scientific experiment selectors;
`ASPECT_FAULT_LINEAR_PERFORMANCE` remains allowed. The historical BP5 checkpoint
rejection key is deliberately retained as a compatibility check, not an active
BP3 model selector. No obsolete prestress-file option is registered.
