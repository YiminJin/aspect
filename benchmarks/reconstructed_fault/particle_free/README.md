# Particle-free history experiments through bounded PF5

The current entry points are [STATUS.md](../../../doc/reconstructed_fault/particle_free/STATUS.md)
and [PF5_REPORT.md](../../../doc/reconstructed_fault/particle_free/PF5_REPORT.md).
Unlimited upwind DGQ2 is the experimental baseline. Production defaults are
unchanged; both WENO paths remain experiments. The implemented PF5 subset is
serial, fixed mesh/geometry/phase/particles, closed-boundary prescribed transport
with uniform surface V/Z. Moving geometry, birth, AMR/MPI transfers and coupled
propagation require separate work.

This directory's CMake, C++ observers, parameter files and independent Python
checks are retained for future development. Reports preserve measured outcomes
and limitations. Large evidence archives, plots, detailed source/run manifests,
binaries, checkpoints and raw outputs are retained locally and are not part of
the source commit. Historical report links to those local artifacts require the
original evidence package; a fresh checkout can rebuild and rerun the fixtures.
See [evidence/README.md](evidence/README.md) for the committed summary subset.

| Check | Native observer / inputs | Independent assessment |
|---|---|---|
| PF1 identities and counterexamples | None | `reference_checks.py` |
| PF2 native FE/DG source, mechanics and publication | `fixed_profile_checks.cc`, `fixed_profile_{fe,dg}.prm` | `assess_fixed_profile.py` |
| PF3 noncommitting phase functional | `phase_energy_checks.cc` | `assess_phase_energy.py` |
| PF4 prescribed material transport | `transport_checks.cc`, `transport_{4,8,16,32}.prm` | `assess_transport.py` |
| Experimental compact WENO / scaling | `limiter_checks.cc`, `stabilization_checks.cc` | `assess_limiter.py`, `assess_stabilization.py` |
| Native scalar EV/SUPG comparison | `scalar_method_checks.cc`, `native_method_checks.cc`, `methods-*.prm` | `assess_methods.py` |
| PF5 accepted history and filesystem restart | `pf5_checks.cc`, `pf5-*.prm` | `assess_pf5.py` |

Build against this checkout's configured ASPECT build, from the repository root:

```sh
cmake --build build-particle-free/native -j4
cmake -S benchmarks/reconstructed_fault/particle_free -B build-particle-free/plugin -DAspect_DIR="$PWD/build-particle-free/native" -DCMAKE_BUILD_TYPE=Release
cmake --build build-particle-free/plugin --target pf5_checks fixed_profile_checks transport_checks -j2
```

Use the compiler/dependency toolchain of that ASPECT build. Optional observer
targets are listed in `CMakeLists.txt`. `run_fixed_profile.py` pins the executable,
observer and expanded input into a new run directory, bounds runtime, and refuses
to overwrite a label. For example:

```sh
python3 benchmarks/reconstructed_fault/particle_free/run_fixed_profile.py pf5-new-16 --input benchmarks/reconstructed_fault/particle_free/pf5-16.prm --plugin build-particle-free/plugin/libpf5_checks.release.so --postprocessor 'PF5 accepted history' --seconds 180
python3 benchmarks/reconstructed_fault/particle_free/run_fixed_profile.py pf5-new-restart-16 --input benchmarks/reconstructed_fault/particle_free/pf5-restart-16.prm --plugin build-particle-free/plugin/libpf5_checks.release.so --postprocessor 'PF5 accepted history' --restart-from build-particle-free/runs/pf5-new-16/run/restart --seconds 180
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/assess_pf5.py build-particle-free/runs/pf5-new-16 --restart-parent build-particle-free/runs/pf5-new-16 --restart build-particle-free/runs/pf5-new-restart-16 --output build-particle-free/pf5-new-assessment.json
```

The reports specify multi-step convergence studies, negative controls and the
remaining transfer task. These examples do not authorize additional runs or
coupled deployment. Python references require NumPy/SciPy; native tests use the
existing ASPECT/deal.II infrastructure.

## Retained PF0/PF1 instructions

PF2 is complete; see [PF2_REPORT.md](../../../doc/reconstructed_fault/particle_free/PF2_REPORT.md).
`fixed_profile_{fe,dg}.prm` and the observer drive three native mechanical steps
at frozen accepted localization. `run_fixed_profile.py` bounds runs and preserves
artifacts; `assess_fixed_profile.py` checks the fresh baseline and same-build
FE/DG fields, actual loads, independent source/work references and lifecycle.
The report gives exact build/run commands, gates, failed initialization evidence
and remaining particle dependencies. Use new labels; preserve existing outputs.

The paragraphs below document the retained PF0/PF1 mathematical references.

These scripts audit a retained baseline and verify mathematical identities.
They change no production source and do not implement DG transport or a new
phase functional. Reports and the PF2 proposal are under
[`doc/reconstructed_fault/particle_free/`](../../../doc/reconstructed_fault/particle_free/STATUS.md).

From the repository root:

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/reference_checks.py --output build-particle-free/pf1-reproduction.json
python3 benchmarks/reconstructed_fault/particle_free/run_baseline.py --retained-build /home/ein/repository/aspect-fault-propagation/build-bp3-native --output build-particle-free/pf0-reproduction --seconds 180
```

Choose unused output names: existing results are never overwritten. The native
runner copies the historical executable and two matching libraries into its
output directory and expands the existing `history-uniform-q6.prm` paths into
this worktree. It does not configure/build or write in the source checkout.
Only library/output paths change. One serial process and one computational
thread are used. The local budget is 180 s for the native case and 10 s for
the mathematical checks (observed <1 s); there is no build campaign.

The native fixture deliberately performs synthetic history commits, rejected
candidate/duplicate-commit checks and restoration inside its observer. There
is no coupled solve. Its huge artificial timesteps test general Maxwell beta,
not BP3 temporal accuracy. A binary's embedded Git label need not identify its
dirty compilation inputs; see the source audit before reusing this evidence.

`reference_checks.py` depends only on NumPy and SciPy. V1 uses 70-digit Decimal
exponentials; V2 uses analytic/complex-step derivatives and a separate central
difference sweep; V3 uses independent characteristic and commutator-forcing
integrals; V4 has a closed Gaussian-overlap error and midpoint refinement.
Each numerical assertion and expected nonzero defect is saved in one JSON.
Failure returns nonzero. Norms/tolerances are in the script and PF1 report.

`evidence/` contains compact original results and exact artifact identities.
Raw native output, expanded parameters, binaries and the earlier successful
89-check development run remain ignored under `build-particle-free/`.
