# Frozen BP3 newborn H review

The subsequent [section-2 geometry review](../bp3_geometry/README.md) derives
BP3 geometry from the prescribed file/Box. Its small consumer updates keep these
fixtures buildable; the binary/output evidence below remains the accepted
`126049420` baseline, not evidence for the newer geometry implementation.

This bounded follow-up starts from `617d52838` on `pf-rsf-refactor`, after the
accepted particle/audit lifecycle correction. It stops before runtime-cleanup
section 2. The executable and earlier qualified plugin remain unchanged.

## Change and ownership

`bp3/plugin/particle_initialization.cc` supplies one stationary H initializer for
startup and later births. It evaluates the prescribed profile at the actual
particle position, uses the existing stationary-H function with BP3's stationary
material fractions and peak phase in the active region, and retains the native
arithmetic material Hc outside it. The opt-in `BP3 frozen crack driving force`
property uses the native late-initialization mode, so H exists before insertion.
The maintained birth audit compares new H with that initializer before capturing
its baseline. No survivor's committed H is replaced, including after restart.
Maxwell components retain the existing validated native interpolator.

The candidate `bp3_150x50_particle_lifecycle.prm` selects this property. Historical
inputs retain their original selection. The specialization requires mature BP3,
whose existing material invariant requires `Evolve phase field = false`. Both
the generic property and the opt-in property's fallback preserve native late
history interpolation for evolving/cohesive models. No core source, numerical
tolerance, physical parameter, particle layout or checkpoint version changed.
The prescribed stationary geometry is not generalized in this task.

## Verification

Release plugin builds pass, including 2D/3D template instantiations. Runtime tests
use the unchanged `build-refactor-r6b/aspect-particle-lifecycle-qualified` on one
or two ranks, with the native 4×4 / 12–24 policy and the previous controlled
crossing/rejection fixture. `compare.py` reports **195 passing checks**:

- Every sampled newborn matches the shared initializer exactly before its audit
  entry exists. The serial run observes 248 births: 26 active and 222 exterior;
  the two-rank case also covers both regions. Exterior values equal material Hc.
- Retry/direct comparisons are exact for particles, full bulk/fault fields,
  history/work audits and solver summaries, including the two-rank retry with
  nonzero incoming Maxwell stress. Rejection restores RNG and audit state.
- Two-rank checkpoint continuation matches the uninterrupted staggered run;
  both ranks consume placement RNG in both birth events. Continuation from the
  preceding implementation's checkpoint preserves all surviving committed H,
  including previously interpolated newborn values.
- Startup H and all non-H particle fields match the previous qualified baseline
  exactly. Bulk/fault fields, work and solver summaries also match. Within each
  trajectory survivors retain exact H. The intentional difference is confined to
  particles born under the new policy: maximum observed absolute H difference
  from prior interpolation is `196885.7332559937`.
- A standalone two-rank evolving-model guard uses the real native property
  manager to transfer synthetic history H=123 rather than reconstruct it from
  current phase. It reaches `EVOLVING H TRANSFER PASS` on both ranks, then exits
  nonzero intentionally before mechanics. This is a transfer test, not an
  evolving physical trajectory.

The first prototype tried native startup Hc initialization at late birth and
segfaulted because the initial-composition manager is discarded after startup.
The correction uses the live material Hc vector and existing stationary BP3
material fractions, preserving startup results exactly. The first standalone
evolving input also failed because an inherited verifier was unavailable; the
final isolated input removes that fixture-only registration. Raw failures are
preserved as `serial-np1` and `evolving-np2`; final passing cases are
`serial-fixed` and `evolving-fixed`. The corresponding original PRMs are retained
as development evidence, not passing cases.

Simulation/startup runs including those two failed attempts took about 105.2 s;
maximum reported child RSS was 344092 KiB (not aggregate MPI memory). No long
trajectory, server run, AMR qualification or 3D runtime test was performed.

## Evidence and reproduction

`results/checks.json`, `H_differences.json`, `birth_counts.json`, per-rank birth and
lifecycle CSVs, `runs.json`, `artifacts.json` and `baseline.json` retain compact
checks, exact commands, hashes and preservation provenance. Large raw outputs,
builds and logs are ignored. Comparisons use the preserved preceding
`particle_lifecycle` outputs and plugin; do not overwrite them.

From the repository root, configure/build the maintained plugin into
`frozen_particle_H/build/bp3`, then the observer into `build/observer`, using
`-DAspect_DIR=$PWD/build-refactor-r6b` for each CMake configuration. Both source
directories are respectively `bp3/plugin` and `frozen_particle_H/plugin` under
`benchmarks/reconstructed_fault`. Build with `cmake --build BUILD -j2`.

`run_cases.py --batch serial-fixed:1 mpi:2 retry:1 direct:1 staggered:2 create:2`
runs the fresh cases. Before the restart cases, use `prepare_restart.py SOURCE
TARGET` to clone `output-create` to the absent `output-resume`, `output-retry2`
and `output-direct2` directories; clone the previous suite's
`output-final-create-fixed` to `output-old-resume`. Then run
`run_cases.py --batch resume:2 retry2:2 direct2:2 old-resume:2 evolving-fixed:2`
and `python3 benchmarks/reconstructed_fault/frozen_particle_H/compare.py`.
Use the suite path when invoking its scripts. Preserve existing output directories
before reproducing; the restart cloning helper deliberately refuses overwrite.
An arbitrary evolving-case failure is not success: `compare.py` requires its
explicit pass marker.

The next bounded task remains section 2, making `fault.txt` the single geometry
input, after review. Broader cache/history accuracy and production qualification
remain outside this follow-up.
