# Post-R6 header and build integration

Baseline: `752c5bcbd96df5c9904387f74e2738cbf7e6bcac`, branch
`pf-rsf-refactor`, with the incoming BP3 documentation/PRM formatting edits and
pinned Stampede3 package retained. `evidence/baseline.json` records their hashes,
the initial status, compiler and immutable runtime reference. The reference is
`build-refactor-r6b/aspect-birth-identity-qualified`; its banner revision is not
used as a substitute for its recorded artifact hash and accepted provenance.

Header relocation is committed separately as **423469d74**. Build-rule cleanup
and its verification are a separate commit. Neither changes numerical bodies,
interfaces, ownership, observer timing or checkpoint formats. There is no new
public simulator interface and no implementation `.cc` included by another `.cc`.

## Header mapping

| Previous path under `source/` | Installed path under `include/aspect/` |
|---|---|
| `reconstructed_fault/normal_filter_internal.h` | `reconstructed_fault/normal_filter_internal.h` |
| `reconstructed_fault/surface_direct_internal.h` | `reconstructed_fault/surface_direct_internal.h` |
| `reconstructed_fault/surface_system_internal.h` | `reconstructed_fault/surface_system_internal.h` |
| `material_model/phase_field_fault/history_diagnostics.h` | `material_model/phase_field_fault/history_diagnostics.h` |
| `simulator/solver/reconstructed_fault_bound_diagnostics.h` | `simulator/solver/reconstructed_fault_bound_diagnostics.h` |
| `simulator/solver/stokes_operators.h` | `simulator/solver/stokes_operators.h` |
| `simulator/reconstructed_fault_interface_preconditioner.h` | `simulator/reconstructed_fault_interface_preconditioner.h` |
| `simulator/reconstructed_fault_residual_audit.h` | `simulator/reconstructed_fault_residual_audit.h` |

Core/test includes and active benchmark compilation/header readers now use the
canonical installed paths. Direct standard/deal.II dependencies are explicit.
No forwarding headers or source-header include-directory additions remain for
these modules. Private `SurfaceAssembly` stays private; internal namespaces and
class boundaries are unchanged. Historical evidence files are not rewritten;
old stage-specific byte-for-byte verification scripts still describe their
original stages, not a new qualification of those historical revisions.

## Build exceptions

| Source | Unity exclusion | PCH exclusion | Audit |
|---|---|---|---|
| `material_model/phase_field_fault/constitutive.cc` | removed | removed | Definitions follow class registration/instantiation; helper name is unique |
| `material_model/phase_field_fault/history.cc` | removed | removed | Split private records/helpers compile with other material definitions |
| `material_model/phase_field_fault/history_diagnostics.cc` | removed | removed | No colliding helper or instantiation; header is independently includable |
| `reconstructed_fault/manager_particle_projection.cc` | removed | removed | Each implementation instantiates only its own definitions |
| `reconstructed_fault/surface_system_particle.cc` | removed | removed | Shared private record comes from the canonical internal header |
| `reconstructed_fault/surface_system_bulk_work.cc` | removed | removed | Same; no file-local helper collision |
| `simulator/solver/reconstructed_fault_stokes.cc` | removed | removed | Focused solver/helper definitions have no collision in normal grouping |
| `simulator/solver/reconstructed_fault_bound_diagnostics.cc` | removed | removed | Non-template diagnostic functions have one definition |
| `reconstructed_fault/boundary_contact_manager.cc` | none | none | Manager now explicitly instantiates its own methods, so both files can share a unity translation unit |

The original collision was between explicit member instantiations in
`boundary_contact_manager.cc` and the whole-class instantiation at the end of
`manager.cc`. In a unity translation unit, that class instantiation also sees
(and instantiates) the already defined boundary methods. The user requested
removing the temporary separation too. `manager.cc` now explicitly instantiates
its own 28 definitions plus five existing inline accessors for both dimensions.
The split files keep their existing member instantiations. No header declarations
or algorithm bodies changed. There are **no unity/PCH exclusions for these core
modules**. File ordering is no longer the manager's instantiation mechanism.

Existing WorldBuilder dependency exclusions are untouched (the old
inline-integration branch is inactive with WorldBuilder 1.0.0; the dependency
itself excludes `parameters.cc` and `point.cc`). No global unity/PCH default is
changed. The separate OFF/OFF build is verification only.

The full independent build also exposed a pre-existing dependency error:
`particle_domain.h` used ASPECT's deal.II namespace aliases and feature
configuration without including `<aspect/global.h>`. Both GCC 16 and pinned
GCC 12 reproduce the error in the unchanged source. One direct include fixes
standalone compilation; particle-domain algorithms and interfaces are untouched.
The baseline failure is retained in `evidence/baseline-particle-domain-gcc12.log`.
Likewise, unchanged `particle/property/crack_driving_force.cc` called an
initial-composition manager member with only its forward declaration available.
It now directly includes `<aspect/initial_composition/interface.h>`; the original
OFF/OFF failure and baseline source hash are retained. Neither include correction
changes initialization or particle history.
The independent build also found two missing defining headers in unchanged
`simulator/phase_field.cc`: `<aspect/particle/manager.h>` for the particle-manager
calls and `<aspect/simulator.h>` for `ExcNonlinearSolverNoConvergence`. These are
direct-include repairs only, with original diagnostics/source hashes retained.

## Verification

Final matched stack: GCC 12.4.0, OpenMPI 5.0.6, deal.II 9.6.2, Release, native 2D/3D
instantiations, `-fno-finite-math-only -ffp-contract=off`. Separate build trees
leave the qualified reference intact:

An initial build used system GCC 16.2.1 because the MPI wrapper resolves `g++`
through PATH. Its unity build and 961 runtime checks passed, but those are
supplemental cross-compiler results with the temporary boundary exclusion.
They are not the final matched qualification. Final commands pin `OMPI_CXX`
and a compiler launcher, and use new build/output directories; earlier logs
and artifacts remain intact. The intermediate `gcc12-unity` and
`gcc12-unity-final` runs predate the last independent-build include repairs;
`gcc12-qualified-unity` and `gcc12-independent` identify the final qualification.

```sh
export OMPI_CXX=/opt/gcc/12.4.0/bin/g++
export PATH=/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:$PATH
cmake -S . -B build-refactor-post-r6-gcc12-unity \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/openmpi/5.0.6/bin/mpic++ \
  '-DCMAKE_CXX_COMPILER_LAUNCHER=/usr/bin/env;OMPI_CXX=/opt/gcc/12.4.0/bin/g++' \
  -DDEAL_II_DIR=/opt/dealii/9.6-local -DASPECT_WITH_VORO=ON \
  -DVORO_DIR=/home/ein/local/voro++/0.4.6 \
  '-DASPECT_ADDITIONAL_CXX_FLAGS=-fno-finite-math-only -ffp-contract=off' \
  -DASPECT_UNITY_BUILD=ON -DASPECT_PRECOMPILE_HEADERS=ON
cmake --build build-refactor-post-r6-gcc12-unity -j 3
# Repeat in build-refactor-post-r6-gcc12-independent with both options OFF.
python3 benchmarks/reconstructed_fault/post_r6_cleanup/run_cases.py baseline
python3 benchmarks/reconstructed_fault/post_r6_cleanup/run_cases.py gcc12-qualified-unity
python3 benchmarks/reconstructed_fault/post_r6_cleanup/compare.py gcc12-qualified-unity
# Repeat the last two commands with gcc12-independent.
```

Eight standalone header checks and an installed-header-only shared consumer
compile/link without source-tree ASPECT headers or PCH. Dependency files confirm
this isolation. The maintained BP3 plugin is rebuilt with its local diagnostic
option OFF, together with the existing lifecycle observer. Protected-file and
unchanged-body checks are in `evidence/structural-checks.json`.

Runtime verification reuses the seven birth/completion coupled cases: serial
direct/retry, two-rank uninterrupted/checkpoint/restart, and two-rank reduced-step
direct/retry. Inputs differ only in artifact/output paths. The comparator retains
the original `5e-10 * scale + 1e-22` field bound, exact solver decisions and exact
within-build replay/RNG checks. No physical parameter or tolerance is changed.
Final results (see `evidence/comparison-summary.json`, `symbols.json`, and
`qualified-manifest.json` for comparisons and source/artifact hashes):

| Check | Unity/PCH ON | Unity/PCH OFF |
|---|---|---|
| Complete GCC 12.4 Release build/link | PASS | PASS, after the three direct-include repairs above |
| Focused fault tests | 964 assertions / 39 cases PASS | 964 assertions / 39 cases PASS |
| Seven coupled serial/MPI/restart cases | 7/7 PASS | 7/7 PASS |
| Unchanged baseline comparator | 961/961 PASS | 961/961 PASS |
| Compared field differences | zero | zero |
| Required 2D/3D member entries | 200/200 present | 200/200 present |
| Baseline manager symbol signatures | 126/126 preserved | 126/126 preserved |

Solver decisions and within-build retry/restart histories, RNG and audit state
match exactly. Fresh linear residual checks pass; elapsed times are not compared.
The maintained plugin and observer build/link with pinned GCC 12.4; the plugin's
`BP3_LOCAL_OSCILLATION_TEST` is OFF. All eight installed headers compile separately
and link into the isolated consumer without PCH or source-tree ASPECT headers.
The 66 structural/protection checks pass, with one material registration retained.

Successful final build logs are `build-gcc12-unity-qualified.log` and
`build-gcc12-independent-final.log`; earlier failed and intermediate logs are
retained with their original names. The final runtime labels are
`gcc12-qualified-unity` and `gcc12-independent`. The executable banners identify
the header commit configured during the build, not the subsequent build-rule
commit; the manifest records the exact tested source and binary hashes.

## Limits

Intel/26.0 is unavailable locally. Clang 22 reproduces the baseline's four
duplicate-instantiation errors (two members, two dimensions). The same combined boundary/manager translation unit no longer produces those
diagnostics after per-member instantiation, but these supplemental syntax
checks are **not a successful Clang build**: the locally installed deal.II bundled
TBB has a pre-existing out-of-range enum constant rejected by Clang 22, including
with `-Wno-enum-constexpr-conversion`. This external dependency was not edited and
no such suppression was added to production flags. See `evidence/clang-baseline-collision.json` and
`evidence/clang-combined-members.json` with their logs. Server-toolchain verification remains necessary.

No Debug, 3D runtime, full production mesh or first-event campaign is selected.
The accepted BP3 startup/normalization/resource limitations remain unchanged.
Recommended next bounded task after review: verify the corrected normal build
with Intel/26.0 on Stampede3; keep any scientific startup issue separate.
