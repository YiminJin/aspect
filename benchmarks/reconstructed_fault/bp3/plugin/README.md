# Restored BP3 runtime

This self-contained source package implements the 150 × 50 km right-dipping
thrust model with frozen mature phase field, paired endpoint completion,
Q2 continuous stress fields, native linear-least-squares particle interpolation,
and the selected raw/Helmholtz normal traction. Preparation and monitor signal
ordering are required: the monitor loads the stationary boundary profile and
configures the surface filter after simulator initialization. Accepted output
never performs a second constitutive update.

## Build and inputs

```sh
cmake -S plugin -B build-bp3 -DAspect_DIR=/absolute/path/to/aspect-build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-bp3 -j4
```

Load only `build-bp3/libbp3_restore_150x50.release.so`, built with the same
compiler/MPI/deal.II/ASPECT as the executable. Keep the separate prepared
`fixtures/bp3_150x50/` inputs and resolve their paths on the server.

The parent `bp3_150x50_raw/filter20/filter40.prm` files are **detailed startup
comparisons**. `bp3_150x50_first_event.prm` is the normal continuation input;
it explicitly sets both profile/heavy intervals to 0.1 m and 31557600 s,
with both diagnostics controls false. Explicit experiment settings are honored.
The input resumes the filter20 checkpoint, after its startup gates are reviewed.

Solver selection uses ordinary `Stokes solver type`. BP3 uses convection and
reconstructed-fault timestep controllers; no BP5 startup predictor is registered.
The core reconstructed-fault controller now supports an optional unweighted
Theta predictor (requires rebuilding ASPECT):

```text
subsection Time stepping
  subsection Reconstructed fault time step
    set Maximum logarithmic state change = 0.1
  end
end
```

It bounds the predicted absolute log-change of Theta at committed velocity.
The default largest finite double disables it; `infinity` remains an accepted
alias. No production input is opted in implicitly.
This is separate from the retired BP5 predictor and its b/a weighting.
The supplied caps remain 100 s first / 4e6 s maximum. Preserve your newer server
caps when updating its input. `Mature prestress file` is deprecated and must be
empty. The maintained input prescribes left/right/bottom velocity and has no
traction plugin boundary; the obsolete zero-traction registration is removed.

## Supported diagnostics and environment

- `BP3 restored monitor / Write detailed diagnostics`: incoming Theta copies,
  quadrature/particle/fault samples, work replay and initial mesh CSVs.
- `BP3 / Audit full state every step`: bulk DoFs, particle histories and work
  replay, for short restart comparisons. Both default false.
- `BP3 / Last accepted step` and `Graceful wall seconds`: bounded graceful
  stopping, never timestep controls.
- `ASPECT_FAULT_LINEAR_PERFORMANCE`: core linear-solver profiling is allowed.
- Unset `ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION`,
  `ASPECT_IH_BOTTOM_COMPLETION_DIAGNOSTIC`, `ASPECT_BP3_UNIFORM_SLIDING`,
  and `ASPECT_FAULT_INTERFACE_MODES`. These guard live core experiments;
  setting them to zero does not bypass the guard. See the
  [consumer audit](../output-cleanup-evidence/REPORT.md).
  The two hidden LENGTH output selectors no longer affect this plugin.

Mandatory checks remain enabled: accepted Theta/compression/boundaries, fixed
geometry/completed Ih, inert H for retained particles, native weak traction, mesh verification,
and the first-real-step Maxwell publication check. The latter always writes
`first_update_maxwell.csv`. Particle replay after step one runs only when
diagnostics request it. The birth-aware audit captures initialized H after native
management and before mechanics, checks surviving IDs exactly, and discards
removed/trial-only entries. Native particle transfer carries the local baseline
through migration/ghost exchange; backup/restore signals follow particle rollback.
Version-6 checkpoints gather only current owned baselines to rank zero in the
all-rank preparation hook. Loading checks/prunes the temporarily global current
snapshot; version-5 audit maps are also readable. There is no growing replicated
historical-ID map and no extra constitutive update.

The separately qualified section-1 candidate
[`bp3_150x50_particle_lifecycle.prm`](../bp3_150x50_particle_lifecycle.prm)
includes [`particle_policy.prm`](../particle_policy.prm): native regular 4×4,
12–24 particles, point-density addition/removal, native LLS limiting of H and all
Maxwell components, and no boundary extrapolation. `BP3 history linear least
squares` resolves/validates/logs the mask by runtime field names. Local verification
uses that same include. Historical inputs remain unchanged. This candidate still
requires all historical geometry/profile fixtures; geometry/runtime cleanup and
server qualification remain deferred. The old monitor identity string is retained
for compatibility and is not evidence of the new limiter policy; use the explicit
limiter startup lines and parameter files.

The matching core stores manager and generator placement RNG streams per rank and
per particle manager, and restores both on timestep rejection. Old snapshots or
changed MPI counts reject active addition/removal; explicit `Load balancing
strategy = none` permits unrelated restart with a warning that RNG replay is
unavailable. Same-build/count/partition replay and the current birth audit are
covered in the [lifecycle report](../../particle_lifecycle/README.md).

## Outputs

| Files | Contents and cadence |
|---|---|
| `accepted_steps.csv` | Every accepted state: time/dt, peak V, nonlinear/linear counts and required consistency summaries |
| `stations.csv` | Existing stations every accepted state |
| `restored_growth.csv`, `first_event.csv` | Growth/flux summary each state; current event decision |
| `profiles.csv` → `profiles/fault_<step>.csv` | Canonical full-fault slip, V, Theta, shear/normal traction and coordinates, at full precision |
| `heavy_outputs.csv`, native solution/particle/fault files | Coordinated heavy-output schedule; completion postprocessor commits its clock |
| Raw/full-state CSVs | Explicit diagnostics only |

Profile and heavy triggers use the maximum absolute nodal change of cumulative
**signed** slip since their last write, or elapsed physical seconds. Initial and
graceful-final states are forced; onset/down-crossing/completion force profiles;
every heavy state also has a profile. Slip integrates at every accepted physical
step using committed V, with no step-zero increment, regardless of output cadence.

The plugin no longer writes `cumulative_slip.csv`. Use the maintained
`plot_cumulative_slip.py RUN_DIRECTORY`, `plot_recorded_slip.py` or
`plot_fault_evolution.py` readers. Optional
`export_profile_slip.py RUN/profiles.csv legacy.csv` creates a legacy schema plus
a mandatory provenance sidecar: **saved profiles only**, no unsaved timesteps.
Keep them together. Event colors use instantaneous profile V, never sparse
Δslip/Δtime. Consult every-step peak summaries for peaks between saved profiles.

## Restart

Version-5 layout, slip/Theta/event history, audit baselines and schedule reference
states remain unchanged. On restart, parsed PRM output intervals replace stored
intervals while retaining the last successful write's time/slip reference.
No CSV is used to reconstruct physical history.

In-place resume is allowed only when its output is no newer than the selected
checkpoint. To branch from an older checkpoint, preserve the parent and run:

```sh
bash branch_output.sh /absolute/parent 2 /absolute/new-branch
```

Then set `Output directory` to that new branch, `Resume computation = true`,
and load the matching model/library. Use the same rank count and physical
settings; changing output cadence is supported. The helper copies the selected
checkpoint and its metadata prefix, only indexed profile payloads, and native
visualization directories once when branching. Ordinary checkpoints copy small
metadata, not the growing payload trees. Keep the parent available: an index
alone cannot restore missing profile payloads. Missing or newer/conflicting
profiles fail instead of silently appending mixed histories. An old checkpoint
without a growth snapshot starts a new growth table with a header.

## Source ownership

`bp3.cc`: initial fields, loading registration, preparation/solver observers.
`mesh.cc`: prescribed mesh reproduction and verification.
`bottom_constraint.cc`: optional fault-parallel bottom constraint; the default
`Bottom velocity constraint = full` retains the existing full loading. The
experimental `fault parallel` value is in `Postprocess / BP3 restored monitor`
and requires removing bottom from all ordinary velocity boundary lists. It
retains the side corners and releases the complementary perturbation traction.
See [the isolated local test](../rotated_bottom_local/REPORT.md) for qualification
and limits; the present evidence does not support changing production runs.
`monitor.cc`: loading-profile/filter initialization and growth/diagnostics.
`work_audit.cc`: mandatory native/inert/Maxwell checks and optional replay.
`output.cc`: accepted history, schedules, event/stations and v5 restart lifecycle.
