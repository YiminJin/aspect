# Reproduction and provenance

Root: `aspect-pf-rsf-particle-free`, branch `pf-rsf-particle-free`,
HEAD `f1f9f29cf40c32d1ddcea000b724375b64853532` plus preserved entry work.
See `source-entry.patch` for changes relative to actual task entry rather than
attributing all existing worktree changes to this experiment.

Use the existing deal.II/ASPECT environment and CMake configuration recorded in
the retained build logs. The two isolated targets are
`cohesive_interface_path` and `cohesive_interface_turn_solver`; the latter reuses
the unchanged signed cohesive–RSF native evolution solver. No production
parameter is added. The path executable uses sparse UMFPACK, deal.II mixed
Q2/P1disc (or Q3/P2disc), Gauss6 assembly and independent Gauss8 probes. zlib
stores large double-precision basis tables losslessly. The matching CMake
cache under `build-particle-free/cohesive-interface-path/build-work/` records
the local compiler/library paths; paths are not portable build requirements.

The local configuration used Release, `/opt/openmpi/5.0.6/bin/mpic++`,
`deal.II_DIR=/opt/dealii/9.6-local/lib/cmake/deal.II` and
`Aspect_DIR=$PWD/build-particle-free/native`. With those dependencies present:

```sh
cmake -S benchmarks/reconstructed_fault/particle_free -B build-particle-free/cohesive-interface-path/build-work -DAspect_DIR="$PWD/build-particle-free/native" -Ddeal.II_DIR=/opt/dealii/9.6-local/lib/cmake/deal.II -DCMAKE_BUILD_TYPE=Release
cmake --build build-particle-free/cohesive-interface-path/build-work --target cohesive_interface_path -j2
cmake -S benchmarks/reconstructed_fault/particle_free -B build-particle-free/cohesive-interface-path/build -DAspect_DIR="$PWD/build-particle-free/native" -Ddeal.II_DIR=/opt/dealii/9.6-local/lib/cmake/deal.II -DCMAKE_BUILD_TYPE=Release
cmake --build build-particle-free/cohesive-interface-path/build --target cohesive_interface_turn_solver -j2
```

The separate `build/` shared-library location is the controller's existing C API
path. Historical `build/` and `build-c1/` path executables are retained evidence;
fresh work-reaction kernels use `build-work/cohesive_interface_path`.

Set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`; run timing cases sequentially.
From the worktree root, with `B=benchmarks/reconstructed_fault/particle_free`:

```sh
python3 $B/cohesive_interface_path.py work64 --n 64 --pressure-mode work
python3 $B/cohesive_interface_path.py work96 --n 96 --pressure-mode work
python3 $B/cohesive_interface_path.py work-step96 --n 96 --extension .375 --pressure-mode work
python3 $B/cohesive_interface_path.py work-q3 --n 64 --degree 3 --pressure-mode work
python3 $B/cohesive_interface_path.py work-time96 --n 96 --dt .000125 --checkpoint build-particle-free/cohesive-interface-path/work-temporal-incoming.json --pressure-mode work
python3 $B/cohesive_interface_path.py work-surface96 --n 96 --surface-h .046875 --pressure-mode work
```

These labels identify the **completed** cases. To rerun, use new labels/output
locations and preserve the archived originals. The time replay's incoming state
is the accepted work96 snapshot at 0.42025 s; it preserves its clock, geometry,
six histories, Maxwell memory and contact mode. Do not convert an old
point-pressure checkpoint to work-reaction mode.

```sh
python3 $B/qualify_cohesive_interface_path.py
python3 $B/analyze_cohesive_interface_path.py
MPLCONFIGDIR=/tmp/cohesive-path-mpl python3 $B/plot_cohesive_interface_path.py
python3 $B/package_cohesive_interface_path.py
```

Run qualification after timing cases finish. The qualifier deliberately submits
an oversized artificial native contact-loss trial, verifies no publication,
then retries correctly and rebuilds the native context from JSON. This failed
trial is not a part of the physical loading history.

`check_cohesive_interface_path_pressure.py --existing` reprocesses the saved
matched frozen-reference comparison; its original mode builds those bounded
kernels. Independent weak-normal, compression-identity and interpolation
results are retained in their JSON/CSV evidence and logs. Point-pressure v1,
the derivative-consistent intermediate controls, and the original C2 source
snapshot are preserved to show why the two numerical corrections were made.
No tolerance or physical parameter was adjusted to make a control pass.

Evolution timing includes newly needed extension-operator construction, native
evolution and compact history output, but excludes initial-kernel construction;
cached operators are reused. These are not cold-start comparisons. Parent RSS
is a process high-water mark; the native kernel's
peak memory was not independently measured. No full-event speedup is inferred.
Each kernel has a 300 s timeout; each trajectory has a 900 s budget and 3 GiB
owner-RSS stop, in addition to contact, residual, energy, reserve and boundary
guards. The final loading stops at .44 s, followed by .005 s hold. R=.75 m,
maximum new arc length 3 m, and all physical inputs are fixed across controls.
