# Directory guide — 2026-10-15

Start with [CURRENT_STATUS](../../doc/reconstructed_fault/CURRENT_STATUS.md) for
the latest scientific results and limitations. The dated task notes below are
historical evidence, not instructions to launch another experiment.
The [worktree checkpoint](bp3_native/CHECKPOINT.md) preserves the open scientific
task and explains what follows into a new checkout and what remains local.

| Location | Contents |
|---|---|
| [bp3_native/STATUS.md](bp3_native/STATUS.md) | Stationary FE, moving-particle transport, population-event and accelerating slip-front checks |
| [bp3/plugin](bp3/plugin/README.md) | Maintained BP3 plugin source shared by the diagnostic and ordinary configurations |
| [bp3_native/server](bp3_native/server/README.md) | Manual fine-mesh server handoff, environment and deal.II 9.7 compatibility patch |
| [propagation](propagation/README.md) | Preserved P1/P2 inputs, prototypes, reports and evidence; no new propagation stage selected |
| [uniform_shear](uniform_shear/README.md), [bp5](bp5/README.md), `refactoring_*` | Independent and historical verification families |
| [maintenance](maintenance/disk_cleanup_2026_10_15/README.md) | Cleanup manifests, preservation checks and restoration instructions |
| [build-bp3-native](../../build-bp3-native/ARTIFACTS_AFTER_CLEANUP.md) | Current BP3 build and retained diagnostic families, organized by task |
| [build-propagation-integration](../../build-propagation-integration/ARTIFACTS_AFTER_CLEANUP.md) | P2 build, `runs/`, `archive/baselines/` and pinned serial reference |

[Disk cleanup](CLEANUP.md) freed 2.32 GiB. Sources and compact evidence keep their
existing paths. Eight older P2 bulk CSVs require the documented restoration;
latest BP3 trajectories, particle-event states, checkpoints and server files
remain directly available.

---

Historical P2 numerical task completed: [native solver accuracy, onset and saddle escape](../../doc/reconstructed_fault/propagation/P2_native_solver_report.md).
Fresh58-step economical onset reaches .53125s in85min9s/1684.9MiB, with core
phase .60457720, zero healing, all seven unaligned field targets, complete-prefix
energy accounting and exact complete-step failure/retry. Default-false numerical
options retain governing energy, actual Hessian and final KKT. The resolved
saddle replay is noncommitting, not fresh continuation. One full154-step native
run is prepared at27–35h estimated and is not launched; full native/P3 remains
unqualified. [Field charts](propagation/native_endpoint_evidence/onset-economical-resumed-comparison.png) and [separate energy charts](propagation/native_solver_evidence/onset-economical-resumed-energy.png).

# Reconstructed-fault benchmarks and checks

Current propagation P2: [native tangent/onset and recovery](../../doc/reconstructed_fault/propagation/P2_native_tangent_report.md).
The fresh native run completes .53125s in54min18s /1719.9MiB, with three accepted
core-growth increments, zero healing, original KKT/history checks and all seven
unaligned prototype targets passing. Complete-step failure restores accepted
state exactly and retries. The154-step economical prototype reaches .625s and
passes separate field/residual-work gates. The original coarse-prefix energy
failure remains explicit; complete native .625s and P3 remain unqualified.
[Native charts](propagation/native_endpoint_evidence/onset-linearized-obstacle-recovery-comparison.png)
and [economical-schedule charts](propagation/native_tangent_evidence/economical-schedule.png).

Historical propagation P2: [optional native consistent-endpoint verification](../../doc/reconstructed_fault/propagation/P2_native_endpoint_report.md).
The full matching 512-cell FE prototype passes the unaligned temporal targets;
default-off native fixed-state operators pass, and a bounded fresh native onset
comparison verifies flank growth through .52832s and measures tangential
variation/cost. Its one-hour budget stops before core growth. The coarse
loading prefix qualifies endpoint fields only, with a separately reported
contact-path energy error. Full native evolution and P3 remain unqualified.

Historical propagation P2: [translation-mode and reflected branches](../../doc/reconstructed_fault/propagation/P2_asymmetric_branch_report.md).
Fourteen diagnostic continuations reach .625s with numerical checks passing.
Centroid motion/release are mesh dependent; exact-bulk control also becomes
unstable. Separate work errors decrease. Physical branch selection, native
adoption and P3 remain unqualified;512-cell-only new figures and old failures
are retained. No production change.

Historical propagation P2: [coupled continuation stability stop](../../doc/reconstructed_fault/propagation/P2_coupled_branch_extension_report.md).
All three partitions reach an admissible antisymmetric saddle at .59765625s,
before the requested .625s endpoint. Incoming/equilibrium checks pass and stable
interval energy errors decrease, but no unstable trial is published. Production
and prior evidence remain unchanged; branch selection, native adoption and P3
remain unresolved.

Previous propagation P2: [consistent-endpoint short branch](../../doc/reconstructed_fault/propagation/P2_coupled_branch_report.md).
The short consistent-start coupled prototype grows smoothly and its same-path
energy/residual contributions converge. Production is unchanged; full-interval,
native and P3 qualification remain deferred. Preserve all historical failures.

Historical P2: [accuracy and energy assessment](../../doc/reconstructed_fault/propagation/P2_accuracy_energy_report.md).
Direct irreversibility is verified for the fixed-mesh fixture. Consistent I at
.025 m still fails the .5% accuracy target; native bulk stress histories have no
additional transfer drift, but evolving-profile energy needs a governing review.
No production source change or P3. Earlier results below remain historical.



Historical coupling-repair result: [P2 coupling repair](../../doc/reconstructed_fault/propagation/P2_coupling_report.md).
Fixed-phase checks pass; evolving feedback remains unqualified because all
three timestep paths exhibit resolved tail healing. Historical failures remain.

Use [BP3's maintained entry point](bp3/README.md) for the current frozen-fault
model and [uniform shear](uniform_shear/README.md) for independent constitutive
verification. Run commands from the repository root unless an entry says
otherwise. Current source is the local post-R7 branch with the user's imported
CMake checks reverted in `f277a53ae`; R7 evidence remains qualified at its recorded
revision. R8 is deferred. Cleanup is not new production/scientific qualification.

| Retained entry point | Purpose and status |
|---|---|
| [bp3/plugin](bp3/plugin/README.md), [production PRM](bp3/production/README.md) | Maintained runtime and single `fault.txt`; buffered 60° boundary admission qualified, full production normalization/mechanics and Intel server run unqualified |
| [uniform_shear](uniform_shear/README.md) | Independent reference, pilot and spatial/time comparisons; `python3 -m unittest discover -s benchmarks/reconstructed_fault/uniform_shear -p test_reference.py -v` |
| [propagation](propagation/README.md) | P1 production checks and fixed-geometry P2; direct irreversibility verified, surface accuracy/profile energetics block P3 |
| [bp3_birth_completion](bp3_birth_completion/README.md), [bp3_local_tests](bp3_local_tests/README.md) | PD-reuse birth identification, buffered completion, velocity oscillation/refinement and transport/retry; preserve their recorded scientific limitations |
| [bp3_geometry](bp3_geometry/README.md), [bp3_runtime](bp3_runtime/README.md), [bp3_packaging](bp3_packaging/README.md) | Endpoint/filter derivative, single-file geometry, removal of runtime fixtures, output/restart identity; scripts/inputs remain at existing paths |
| [particle_lifecycle](particle_lifecycle/README.md), [frozen_particle_H](frozen_particle_H/README.md) | RNG/particle retry/restart, frozen newborn H and evolving-history guard |
| [particle_replenishment](particle_replenishment/README.md) | Retained independent transfer/reference tables, including the historical failed RNG gate; not a current-source qualification runner |
| [frozen_gmg_repair](frozen_gmg_repair/README.md) | Real AMG/GMG condensed-system/state-preservation check; exact historical mesh/fault/replay dependencies retained |
| [refactoring_r7b](refactoring_r7b/README.md), [post_r6_cleanup](post_r6_cleanup/README.md) | Final local numerical/build qualification; source-specific histories, not scripts to run indiscriminately on another revision |
| [bp5](bp5/README.md), normal-filter, performance/server_gmg, nonuniform shear | Retained research/manual deployment tools; unresolved findings and external inputs prevent automatic retirement |

The [dependency inventory](maintenance/dependencies.json) records shared source,
registered tests, optional CMake targets, parameter includes, Python imports,
environment-selected fixtures and checkpoint assumptions. The current
[switch inventory](refactoring_r6a/switch_inventory.md) stays at its canonical
path. Earlier R1–R6/Maxwell/restart helpers remain historical dependencies; they
may require their recorded executable and restoration of old output groups.

## Historical material and recovery

[Classification](maintenance/classification.tsv) covers every one of the 3,397
tracked files in cleanup baseline `f277a53aeadbe99ee093d5eb5421680dad2055d4`.
1,184 completed logs, expanded PRM copies and exports are untracked, with exact
local copies retained for existing comparison paths. Compact qualification
summaries, numerical inputs, deliberate references and all R7 evidence remain
tracked. No checkpoint, executable, current output directory or old archive is
removed. This reduces the tracked tree, not the Git history or local disk usage.

| Tracked benchmark tree | Files | Uncompressed blob bytes |
|---|---:|---:|
| Actual post-R7 baseline `f277a53ae` | 3,397 | 78,793,540 |
| Closeout staged tree (pending review) | 2,221 | 52,531,009 |

The final row includes the new inventory/recovery tools and the exact previously
ignored R1 profile reference. Incoming untracked server packages and ignored
local outputs/builds are not counted as tracked files.

Recover retired tracked files into a new directory without overwriting a run:

```sh
python3 benchmarks/reconstructed_fault/maintenance/recover.py \
  benchmarks/reconstructed_fault/bp3_runtime/results/staggered \
  --destination /tmp/bp3-closeout-recovery
# Inspect the preview, then repeat with --write.
```

The exact baseline commit supplies old code/configuration/committed evidence.
The ignored local archive `.benchmark-cleanup-20261006-closeout/` also contains
`retired-tracked.tar.gz` (verified 1,200 members, including 16 replay inputs
ultimately kept tracked), `incoming.tar.gz` and its checksum manifest, and the
verification logs. These are recoverable local copies, not off-machine backups.
The incoming archive preserves the full pre-shortening BP3 README and all local
edits; Git alone does not contain those edits or qualified binaries. To inspect
that archive, extract into a **new empty directory**, not over this worktree.

For raw outputs previously archived on October 2, follow the existing
[CLEANUP.md](CLEANUP.md) and `.benchmark-cleanup-20261002-refactor/restore.py`.
Do not mix partial checkpoint metadata with another run. Restore complete groups
and use their recorded source/plugin/configuration and MPI count. Old reports
remain immutable historical records; their log/result links may require this
recovery even where a local copy happens to remain available.

## Closeout checks and limitations

[Verification](maintenance/verification.json) records fresh configure/build,
parser/smoke, independent reference, dependency/syntax, recovery and preservation
checks. The maintained BP3 target is built from scratch with local diagnostics
OFF. The optional aggregate `BP3_BUILD_TOOLS=ON` target exposes a pre-existing
failure: `test_bp3_restored_model` lacks ASPECT include setup after the geometry
header gained an ASPECT dependency. Its target/source are retained; no interface
repair is bundled here. `test_bp3_environment` builds and passes independently.

Five historical PRM includes remain unavailable at this checkout's documented
working directory (listed in the audit): old `first_cycle_coarse/original.prm`,
BP5 `dc010-ell100/fixtures/candidate/case.prm`, deployment `production_input.prm`,
`cell_actions_one.prm` and `bound_tip_smoke.prm`. These were absent before cleanup;
no dependent historical run is claimed to pass. Incoming revised/server PRMs and
packages, BP5/normal-stress research, large immutable meshes and chained stage
helpers are deliberately retained. Intel, full Debug/3D, production first event,
known cohesive nonconvergence and asymmetric cache-entry limits remain open.
