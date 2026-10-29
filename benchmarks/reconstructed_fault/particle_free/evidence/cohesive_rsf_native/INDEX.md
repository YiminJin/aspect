# Native ASPECT cohesive–RSF pilot evidence

[Measured report](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_RSF_NATIVE_REPORT.md)
· [equations/owners](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_RSF_NATIVE_FORMULATION.md)
· [direct build/run inputs](../../cohesive_rsf_native/README.md)
· [single evidence archive](../../cohesive_rsf_native_evidence.zip).

The **final uninterrupted** trajectory is `pilot-final-current`, .440–.462375 s,
179 positive accepted steps, two ranks. Other labels are isolated controls;
failed/superseded trajectories are never spliced into this result. Raw solver
archives and every failed checkpoint remain separately under
`build-particle-free/cohesive-rsf-native`; the archive contains compact outputs.

| File | Contents / interpretation |
|---|---|
| `native-geometry.png` | Same physical axes before/after; a, actual a=.1/a=1 front brackets, native material markers, slip-rate vectors, temporary tail |
| `native-propagation.gif` | Whole finite-wall domain physical tau_xy and enlarged surface evolution; imposed gamma and loading/hold clocks |
| `native-mature-material.png` | Native-admitted cell: slip/Dc, velocity, age, cohesive/friction/radiation resistance, current normal reaction |
| `native-profiles.png` | Surface a, signed V and normal pressure at .440/.455/.462375; raw sided diagnostics remain visible |
| `native-comparison.png`, `native-timestep.png` | Retained trajectory and focused final-window timestep comparisons |
| `native-energy.png`, `native-energy-windows.csv` | Storage, physical work/losses, endpoint numerical remainders; absolute remainder screens separately for loading/hold |
| `native-history.csv` | Every accepted time, growth/front brackets, full native diagnostics, costs/counts/memory; initial row has no step increments |
| `native-profiles.csv` | All Gauss surface records at all accepted times; SI units in headers |
| `native-material.csv`, `native-maturity.csv` | Persistent native-admitted IDs; whole-cell full-decohesion events and subsequent RSF response |
| `native-results.json` | Main numerical extrema, windows, conversion/reference differences and maturity metrics |
| `native-reference-comparison.csv` | Absolute common-time native-minus-retained differences; arc uses independent Gauss20 integration |
| `timestep-comparison.csv`, `timestep-results.json` | Increment differences after common .460 anchor, coarse/half timestep; no preclosure-step denominator |
| `native-lifecycle.json` | Ten independent comparisons of existing native controls, complete physical records/IDs/geometry/publication counts and exact execution/binary identities |
| `native-costs.csv`, `native-timers.csv` | Actual wall times/ranks/hashes and overlapping native timer scopes |
| `native-parameters.csv`, `parameter-validation.json` | Explicit retained parameter tuple and validation of all nine inputs |
| `history-qualification.json`, `history-r2.csv`, `history-r4.csv` | Earlier independent distorted-cell component evidence, scoped as components rather than Simulator qualification |
| `pressure-correction-history-identity.json`, `clock-closure-identity.json` | Preserved physical-record identity across initialization/clock corrections |
| `source-protection.json`, `entry-protection.json` | 1456-file entry audit; 12 original files changed (including the stage document), none missing/no BP3 changes |
| `source-manifest.json`, `environment.json`, `CHANGED_FILES.md` | Exact source/environment identity, dependencies and task-specific changed-file list |
| `native-source.zip` | Current source/include/CMake/contrib/unit-test support and native fixture; excludes solver archives and unrelated benchmark results |
| `source-package-configure.json` | Successful CMake configuration of the extracted source archive; this is a package check, not another rebuilt numerical trajectory |
| `entry-to-current.patch` | Changes against the actual entry snapshot for ten original core/docs files; new modules/full header are in source archive |
| `native-start-checkpoint.zip` | Native accepted step8 (.441 s), two ranks, next interval .000125 s; no prototype file needed; internal SHA manifest |
| `checksums.sha256` | Compact deliverable checksums; archive has its own adjacent SHA file |

Arc, x, y and slip are metres; integrated |jump| is m²; stresses/reactions Pa;
work and storage J/m (unit out-of-plane thickness); V m/s; time seconds. RSF age
is stored normalized, physical age=theta Dc/V0=200 theta seconds. Surface cell
averages use the exported physical Gauss3 **weights**, not point counts. Bulk
plots use native cell-centre physical full-tensor reconstruction; no point-stress
accuracy claim follows. Fronts use seed-connected a>=.1/.5/1 components and
neighboring Gauss brackets; plot positions are bracket midpoints. New material
requires a native `admit:` identity; an unused tied tail does not count as growth.
Timestep relative changes use (half-step minus coarse) divided by the **coarse
window increment**. `timestep-normalization-note.json` retains the equivalent
earlier fine-denominator values; absolute differences and physical data agree.

The clock values differ from declared times by accumulated floating-point
roundoff within the documented operation-count bound. A small signed final
energy remainder is not used to hide cancellation: sum absolute step physical
remainders, normalize separately by sliding loss and added fracture, and keep
same-field quadrature-repartition defects separate.

To repeat analysis from the retained raw runs:

```sh
python3 benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native/verify.py
python3 benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native/analyze.py \
  pilot-final-current --gif \
  --half-history build-particle-free/cohesive-rsf-native/fine-analysis/native-history.csv \
  --comparison-start .460
```

The source snapshot builds the pilot and legacy unit support; it is not a full
distribution of unrelated ASPECT example databases or regression outputs. Unzip
it to a separate source directory and use the README commands there. Place the
supplied native anchor at the README's worktree-relative path for standalone
resume. Native checkpoint compatibility is qualified on the stated environment
and same rank count; no cross-version conversion is promised.

The point/ahead normal-reaction failures, early spatial/branch findings, finite
wall dependence, uncalibrated path hypothesis and the original 7.5 m
formed-length-budget stop remain explicit. Preliminary summaries are preserved
locally but excluded from the final archive. Production/BP3/defaults are protected.
