# Production phase integration evidence

Normalization follow-up: `run_integration.py --lumped` selects the new optional
I_h representation; omit it for the unchanged consistent default. Same-mode
native restart is supported; changing representation on restart is rejected.
`--spacing 0.05` is the isolated surface-refinement probe (bulk mesh unchanged).
See [normalization report](../../../doc/reconstructed_fault/propagation/normalization_report.md)
and `check_normalization.py`. P1 implementation checks pass; this is not a P2
accuracy or integrability pass.


The optional production solver is implemented. The user-selected fixed
cross-section association policy resolves the original P1 support blocker.
Affected straight serial/two-rank/restart checks pass; padded evidence is reused.
The original failure and commands remain recorded below as historical evidence.
P2 then ran and exposes a separate normalization-projection failure; see
[COUPLED.md](COUPLED.md) and the [P1 integration report](../../../doc/reconstructed_fault/propagation/P1_integration_report.md).

Current affected P1 commands (use fresh labels for future repeats):

```sh
python3 benchmarks/reconstructed_fault/propagation/run_integration.py straight_support_fixed --full-support
python3 benchmarks/reconstructed_fault/propagation/run_integration.py straight_support_fixed_mpi --full-support --ranks 2
python3 benchmarks/reconstructed_fault/propagation/run_integration.py straight_support_resume --full-support --resume straight_support_fixed
python3 benchmarks/reconstructed_fault/propagation/run_integration.py straight_support_mpi_resume --full-support --ranks 2 --resume straight_support_fixed_mpi
python3 benchmarks/reconstructed_fault/propagation/check_support.py straight_support_fixed straight_support_fixed_mpi
```

The current support checker exits 0. It checks all original numerical allowances,
exact native H preservation, the previous narrow phase trajectory, all phase
histories across MPI/restart, actual cached QP/particle coverage, and zero omitted
source. Native mechanical quadrature error is recorded separately. The checkpoint
remains after +5%, before holds erase distinct old/older phases. The new width
uses the existing manager archive; no support reconstruction runs after startup.
[Full-support results](integration_evidence/full_support_checks.json) and the
[continuation manifest](coupled_evidence/manifest.json) preserve provenance.

## Build and execution

Run from the repository root. All new builds/outputs are in
`build-propagation-integration`; old task builds are retained. The recorded
Release configuration uses GCC 12.4.0 and the installed deal.II package:

```sh
env OMPI_CXX=/opt/gcc/12.4.0/bin/g++ PATH=/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:/usr/local/bin:/usr/bin:/bin cmake -S . -B build-propagation-integration -DDEAL_II_DIR=/opt/dealii/9.6-local -DCMAKE_CXX_COMPILER=/opt/gcc/12.4.0/bin/g++ -DCMAKE_BUILD_TYPE=Release -DASPECT_UNITY_BUILD=ON -DASPECT_WITH_VORO=ON -DVORO_DIR=/home/ein/local/voro++/0.4.6 -DASPECT_WITH_NETCDF=OFF
cmake --build build-propagation-integration -j3
cmake -S benchmarks/reconstructed_fault/propagation -B build-propagation-integration/plugin -DAspect_DIR=/home/ein/repository/aspect-fault-propagation/build-propagation-integration -DCMAKE_CXX_COMPILER=/opt/gcc/12.4.0/bin/g++ -DCMAKE_BUILD_TYPE=Release
cmake --build build-propagation-integration/plugin --target production_integration mature_compat -j2
```

The original core configure additionally supplied unused `ASPECT_USE_TPETRA=OFF`;
actual build is Epetra. Configure/build logs are retained in the build directory.
The observer initially inherited MPI-wrapper GCC 16.2.1; final support, padded,
MPI and restart controls were repeated with GCC 12.4.0. No external build was
reconfigured. MPI requires local socket access outside the execution sandbox.

The runner refuses to overwrite case logs or fresh outputs; use new labels for
future repeats. Recorded final commands:

```sh
python3 benchmarks/reconstructed_fault/propagation/run_integration.py straight_gcc12_support
python3 benchmarks/reconstructed_fault/propagation/run_integration.py padded_gcc12 --padded
python3 benchmarks/reconstructed_fault/propagation/run_integration.py padded_gcc12_resume --padded --resume padded_gcc12
python3 benchmarks/reconstructed_fault/propagation/run_integration.py padded_mpi_gcc12 --padded --ranks 2
python3 benchmarks/reconstructed_fault/propagation/run_integration.py padded_mpi_gcc12_resume --padded --ranks 2 --resume padded_mpi_gcc12
python3 benchmarks/reconstructed_fault/propagation/run_integration.py padded_restore_final --padded --ranks 2
python3 benchmarks/reconstructed_fault/propagation/check_integration.py straight_contact_fixed padded_gcc12 padded_mpi_gcc12 straight_gcc12_support
```

The straight support run exits 1 at step 2 with the coverage assertion; the
independent checker returns **2**, retaining `P1_production_gate: false`.
`straight_contact_fixed` is the earlier complete algebraic trajectory through
+10%, using the final core and earlier observer before coverage was instrumented.
It reproduces the reference but is not evidence that support passes. The final
compiler run independently reproduces its +5% coverage failure. The last padded
two-rank repeat adds particle/property and domain-version restoration assertions;
its phase trajectory matches the preceding two-rank run exactly.

## Inputs and controls

[integration.prm](integration.prm) includes [straight.prm](straight.prm), with
padded overlays generated by [run_integration.py](run_integration.py). Production
solves are observed by [production_integration.cc](production_integration.cc),
not replaced by the earlier diagnostic Newton implementation. The harness fixes
geometry/mesh/material/particles, disables mechanical feedback and transport,
and prescribes particle H at +5% and +10% on the relaxed partial-band mask.
These are phase-only controls, not mechanical-H qualification.

Schedule: step 0 relax/accept, 1 unchanged hold, 2 +5%, 3 unchanged hold, 4 +10%.
Native checkpoint follows completed step 2/history rotation and stores counter 3.
Both save and resume assert distinct old/older phase, then resume steps 3/4.
No checkpoint is delayed until the histories become equal again.

Compatibility inputs/logs: `compat_equality` preserves original equality physical
failure; `compat_ih` and `baseline_ih` reproduce the original ten-iteration
failure on new and untouched reference cores. `compat_ih_budget` changes only
the diagnostic iteration budget to 50 and passes. `compat_mature` and
`compat_mature_optin` pass frozen-field completion with selector off/on.
All generated overlays remain under the local build; original fixture defaults
and tolerances are unchanged. No new equality expected-output baseline was made.

## Original evidence and preservation

[summary.json](integration_evidence/summary.json) records numerical comparisons;
[manifest.json](integration_evidence/manifest.json) pins source, inputs, builds,
selected outputs, commands and inherited-evidence preservation. Per-attempt JSON
files record exit/runtime/RSS/input/executable identity where captured; all logs
are retained, including implementation-check failures. Binary checkpoints and
rank nodal CSVs remain in the task-owned build outputs and are hashed in the
manifest. Finite state/property values and initialization markers are compared
on restart; NaN payload identity is not claimed.

No physical rejected-step rollback, coupled restart or transport/birth-history
qualification was executed. P2 feedback was subsequently tested as recorded in
COUPLED.md. The retained sampled-H accuracy
question is independent of admissibility for the sampled discrete problem.
