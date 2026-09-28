# Restored BP3 runtime cleanup — 2026-09-24

## Scope and preservation

Base revision: `3ce447a17e14ac21cb24abf8fbf679d77b4f3f0e`, with existing uncommitted
ASPECT, benchmark and documentation changes. No ASPECT header/source or physical
PRM was edited by this cleanup. No commit was made.

Recoverable evidence is in `plugin-cleanup-evidence/`:

- `before.tar.gz`: original copied plugin, original development implementation
  and its headers, timestep predictor, CMake file, and the four restored PRMs.
  SHA256: `3b4bf61e14adcc04a60d43467c3fe40724fa88167c8684ee741d014d2381c8e4`.
- `before-tracked.patch` and `before-status.txt`: pre-existing working changes,
  including the untracked-file inventory. These are not changes attributed to
  this cleanup. The earlier full-source checkpoint remains valid provenance.
- `pre-cleanup-plugin.tar`: the old user-facing package, retained rather than
  silently overwritten. The new `plugin.tar` contains the cleaned runtime.
- Build/validation/test logs, before/after parameter declarations and hashes.

## Retained invariants

The right-dipping chart/shear sense, stationary profile and boundary velocities,
uniform nominal background, fully frictional continuous Q1 fault, mature C=0,
work measure, endpoint completion/source treatment, unrestricted LLS/Q2 inputs,
normal filter, nonlinear/linear safeguards and split history updates are unchanged.
The predictor source is byte-identical to the previous `bp5/startup_time_step.cc`.

Preparation remains registered before the incoming-state monitor. The material
and manager own constitutive histories; shared plugin variables are observations
and configuration, not replacement histories. Output uses the frozen mechanical
observations and the once-committed accepted state. Checkpoint keys, archive
version and serialization order are unchanged. This is not a restart conversion.

## Consolidation

`plugin/` is now the canonical restored runtime, not a second copy of the broad
development build. Its single target has five explicit, local translation units:

- `bp3.cc`: initialization, boundary models and simulator signal connections;
- `mesh.cc`: fixed leaf-tree reading/refinement/verification;
- `monitor.cc`: profile loading and raw/filtered traction/state observations;
- `output.cc`: work/history checks, accepted-state output and restart/event state;
- `state_startup.cc`: the unchanged timestep predictor.

`runtime.h` contains only the small cross-file interface. The model and three
small observer/environment headers remain. No registration or mutable variable
definition remains in a header. Unused includes were removed; required utility
includes are now explicit. The extraction preserves numerical function bodies.

Removed from the restored runtime: BP5 initialization/normal-control branches,
disturbance branches, unreachable captured-prestress loading, the unreachable
BP5-short-test output branch (already rejected by the environment guard), and
the four implementation headers `mature_fault.h`, `matched_resolution.h`,
`work_replay.h`, `restore_150x50.h`. Their necessary code moved to the `.cc`
files; their historical counterparts outside `plugin/` are retained.
The remaining length-study switches only select extra observational output.

The parent CMake build uses `add_subdirectory(plugin)` and preserves the existing
library output path. Old-chart builds are opt-in via `BP3_BUILD_LEGACY_PLUGIN`;
mesh generation/environment/model tests via `BP3_BUILD_TOOLS`. The standalone
server build needs neither option nor any sibling source directory.

## Verification

All builds used `-j4`. Standalone Release configuration/build and the parent
development target both passed and produced byte-identical libraries:
`b4162ab56684764cdb0fe62af9685f591a949490721b2334c363fdffde93ce61`.
The first include-pruning build exposed a missing explicit `aspect/utilities.h`
include in the monitor; it was added and the final builds passed.

| Check | Result |
|---|---|
| Before-cleanup shear/filter unit baseline | 100 assertions, 2 cases passed |
| `test_bp3_restored_model` | 157 exact model/event/output comparisons passed |
| `test_bp3_environment` | Clean and contaminated environment cases passed |
| Before/after `--output-json` | Only `Additional shared libraries` path differs |
| `--validate` raw/filter20/filter40/first_event | All four passed |
| `--test '[fault_shear_sense],[fault_normal_filter],Stage-I*'` | 20,236 assertions, 16 cases passed |
| Existing reflected/filter coupling fixture, 1 rank | Passed; B FD 1.603e-14, virtual work 2.918e-15 |
| Same coupling fixture, 2 ranks | Passed; B FD 1.579e-14, virtual work 2.150e-15 |
| Whitespace/diff check | Passed |

The coupling fixtures exercise the existing production residual/Jacobian/work
path, not the full restored BP3 plugin lifecycle. The model test compares the
extracted chart, analytic state/aging checks, event observer and output schedule
directly with the retained implementation. It does not replace a mechanical run.

Commands (repository root; paths to temporary standalone builds appear in logs):

```sh
cmake -S benchmarks/reconstructed_fault/bp3/plugin -B /tmp/bp3-clean-build \
  -DAspect_DIR="$PWD/build-pf-cpdi" -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/bp3-clean-build -j4
cmake -S benchmarks/reconstructed_fault/bp3 -B benchmarks/reconstructed_fault/bp3/build \
  -DAspect_DIR="$PWD/build-pf-cpdi" -DBP3_BUILD_TOOLS=ON
cmake --build benchmarks/reconstructed_fault/bp3/build \
  --target bp3_restore_150x50 test_bp3_restored_model test_bp3_environment -j4
build-pf-cpdi/aspect-release --test '[fault_shear_sense],[fault_normal_filter],Stage-I*'
env ASPECT_TEST_NORMAL_FILTER=1 ASPECT_TEST_REVERSED_SHEAR=1 \
  timeout 120 build-pf-cpdi/aspect-release benchmarks/reconstructed_fault/bp3/restore_shear_test.prm
mpirun -np 2 env ASPECT_TEST_NORMAL_FILTER=1 ASPECT_TEST_REVERSED_SHEAR=1 \
  timeout 120 build-pf-cpdi/aspect-release benchmarks/reconstructed_fault/bp3/restore_shear_test_two.prm
```

## Remaining limitations

The reported server segmentation fault is **not diagnosed or claimed fixed**.
Its backtrace and actual executable/library pair are still needed. A fresh build
against the exact ASPECT executable avoids stale-library ambiguity but is not
proof of the crash's cause.

The full-resolution restored startup and filesystem restart were not rerun:
the recorded 40–60 GiB estimate exceeds the local available memory. Consequently,
full plugin initialization/callback and restarted-trajectory equivalence remain
unverified. No large simulation or earthquake-cycle continuation was launched.
Old research variants remain outside this target instead of being deleted.

## Follow-up: confirmed constructor fault and narrow fix

The subsequent local reproduction supersedes the earlier "not diagnosed"
status above. Using the user's `build-tmp/aspect-release` and copied runtime,
GDB captured SIGSEGV in `set_normal_stress_filter`, called from
`BP3RestoredMonitor::initialize` during postprocessor parameter parsing.
The simulator constructs the surface system only after that parsing call.
The plugin and four physical input files matched the maintained copies.

Only the runtime `plugin/monitor.cc` changes behavior: profile reading stays
early, while the filter and diagnostic setters are attached to the existing
`post_simulator_initialization` signal. No equation, input, core ASPECT source,
history timing, or solver tolerance is changed. The same fix is applied to the
user's local `plugin/bp3/monitor.cc`; its library is rebuilt with `-j4`.

Added `test_constructor.gdb`: a real constructor regression that succeeds only
upon reaching `Simulator<2>::run()` and stops before mesh generation. A crash
instead produces a backtrace and nonzero status. With the fixed Release plugin,
both the original raw input and a Helmholtz-20 variant pass this check using the
user's ASPECT executable. This verifies the former crash point, not a full
mechanical startup/restart trajectory. Logs are saved in
`plugin-cleanup-evidence/initialization-fix/`; the source package is refreshed.
