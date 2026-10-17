# Environment record

Local inherited qualification: GCC 12.4.0, deal.II 9.6.2, Trilinos 14.2,
p4est 2.8.7, OpenMPI 5.0.6, **SUNDIALS 7.2.1**, Voro++ 0.4.6; bundled World Builder and Catch.
The supplied ASPECT source requires a compatible C++17 build. These dependency
installations are not copied from the workstation into the archive.

The requested `ibrun` launch uses the server site's MPI environment. Load the
compiler/MPI modules and deal.II installation built together there, including
its p4est/Trilinos/**SUNDIALS** dependencies. Require
`DEAL_II_WITH_SUNDIALS=ON`; SUNDIALS support is needed by the native fault path. Do not mix workstation OpenMPI shared libraries
with an Intel/site MPI build. Set `DEAL_II_DIR`, `VORO_DIR`, `CXX` and the documented
compiler-specific `BP3_FP_FLAGS` manually. `environment.sh` keeps module-provided
PATH/LD_LIBRARY_PATH intact and selects one thread per rank. Record `module list`,
both CMake caches and hashes of the built executable/plugins before the run.

No specific site module name or current installation path was supplied, so none
is fabricated or loaded automatically. No container, Python environment or
additional external profile/mesh file is needed by this input.
