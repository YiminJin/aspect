# Mature moving-step refactor evidence

[Current reading guide](../../../../../doc/reconstructed_fault/particle_free/MATURE_MOVING_FAULT.md)
· [refactor report](../../../../../doc/reconstructed_fault/particle_free/MATURE_STEP_REFACTOR_REPORT.md)

Source baseline: `a22214d57f3ce809b3b20c9883b853c96e04033f` in the original
`pf-rsf-particle-free` worktree. This changes interfaces and ownership boundaries;
no constitutive, discretization, checkpoint or convergence convention is selected.

- `entry.json`: actual root/branch/revision, incoming untracked/manual work,
  source identities and six protected current binary/library copies. Prior
  consolidation controls and scientific outputs remain untouched.
- `build.log`, `build-final.log`, `build-consistency.log`, `checkers.log`: core
  and matched checker builds. The first build overlapped final source editing;
  only the final frozen-source build and consistency rebuild supply test binaries.
- `independent-compile.json/log`: eight affected/new translation units compile
  without unity or PCH using the native compiler flags.
- `unit-tests.log`, `unit-summary.txt`: full existing unit suite. This is not
  the entire deferred ASPECT integration campaign.
- `*-launch.json`, `runs/*/{command.json,input.prm}`: exact bounded runner
  arguments, MPI counts, environments, binary/input hashes and actual results.
  All simulations run sequentially. Raw output and test checkpoints remain in
  this worktree's `build-particle-free/transported-objective/stage-c/material-runs`.
- `equivalence.json`: common accepted clocks, every bulk/surface exported value,
  independently reconstructed physical tensor/work traction, publication counts,
  iterations and costs. The declared absolute screens are 5e-13 bulk and 5e-12
  surface. Different exported columns carry different units; those maxima are
  regression screens, not a physical norm.
- `solver-text-comparison.json`: retained failed exact-text comparison of two
  alpha values differing only at roundoff. `solver-decisions.json` records all
  numerical step lengths, identical rejection counts and outer-iteration indices.
- `lifecycle-crosschecks.json`: current uninterrupted vs continued restart, and
  rejection vs the retained matched accepted schedule, using the same reducer.
- `mature-projection-inputs.csv`: analytic check using a supplied profile and
  gradient distinct from the bound fields. This catches hidden state reads.
- `final-provenance.json`: tested source/binary identities and protection checks.

Build/check commands (at most four concurrent workers):

```sh
cmake --build build-particle-free/native --parallel 4
cmake --build build-particle-free/transported-objective/stage-c/checks --parallel 2
ctest --test-dir build-particle-free/native -R '^unit_tests$' --output-on-failure
python3 benchmarks/reconstructed_fault/particle_free/transported_objective/run_mature_step_refactor.py dc sc dl sl default reject
python3 benchmarks/reconstructed_fault/particle_free/transported_objective/run_mature_step_refactor.py restart4
python3 benchmarks/reconstructed_fault/particle_free/transported_objective/check_mature_step_refactor.py
```

Runners refuse to overwrite completed output; select fresh labels explicitly
for another replay. Controls come from the prior consolidation's retained
command records, with identical physical inputs. The native runtime libraries
remain at `/opt/openmpi/5.0.6/lib`. No server job, production checkpoint
conversion or long event replay is part of this task. Previous finite-end,
cohesive energy/tip and production limits, and inherited cleanup data losses,
remain unchanged.
