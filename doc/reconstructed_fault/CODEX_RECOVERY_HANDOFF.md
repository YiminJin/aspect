# ASPECT phase-field / RSF project: recovered working context

Prepared 2026-09-25 (America/Los_Angeles).

The previous local Codex session was lost after deletion of `~/.codex`. This document reconstructs project decisions and evidence from the surviving ChatGPT discussion and uploaded reports. It is not a recovered CLI transcript, a complete code audit, or proof of the current checkout state. The repository, its working-tree changes, and the newest executed-run artifacts must establish what is actually implemented.

## 1. First task for the new session

Reconstruct the current working state before modifying numerical behavior. Do not restart the research from its earliest failures or implement an old task plan as though later work never happened.

1. Read existing applicable `AGENTS.md` instructions and this handoff. Preserve existing project guidance.
2. Inspect the actual repository location, branch, HEAD, recent commits, tracked changes, and relevant untracked source/configuration files. Do not reset, clean, switch away from uncommitted work, or overwrite files with an uploaded archive.
3. Locate the BP3/BP5 reports, benchmark plugins, current PRMs, generated fixtures, build configuration, execution scripts, and checkpoints. Use `rg` for targeted searches. Read recent summaries first; inspect large logs only to resolve a specific uncertainty.
4. Identify the executable/plugin pair used by the latest successful run. Record paths, build options, and available hashes. Uploaded source, a committed revision, a dirty checkout, and an executed binary are different kinds of evidence.
5. Write or update `doc/reconstructed_fault/CURRENT_STATUS.md` with confirmed completed work, current parameters, run locations, unresolved questions, and the next bounded task. Link the actual evidence. Identify discrepancies without silently changing the model to resolve them.
6. Merge a short navigation section into an appropriate existing `AGENTS.md` so later sessions know where these notes are. Do not replace repository-wide instructions with this research history.

This first task is an evidence/documentation recovery task. It does not require a new simulation, a refactor, a new numerical method, or a server submission. Finish with a concise account of what is verified, what remains unknown, and the immediate next action. Subsequent implementation should continue the latest project state.

Useful initial commands, run from the user's actual checkout:

```sh
git status --short
git branch --show-current
git log -12 --oneline
git diff --stat
git diff --cached --stat
rg --files benchmarks/reconstructed_fault
```

Do not run `git add .` blindly: large outputs and private/local configuration may be untracked. Preserve useful source, scripts, and parameter changes in a reviewable checkpoint; keep large simulation outputs separately backed up.

## 2. Project and implementation

The user is developing a 2D phase-field fault model with rate-and-state friction in ASPECT. The known public branch is `YiminJin/aspect:pf-rsf`; earlier development used `pf-cpdi`, and some build-directory names still contain that name. A previously inspected revision was `33228369da82011f509ee07937f27c17663c7f28`. Later local changes exist or may exist; do not reset to that revision.

The working model combines:

- Incompressible viscoelastic Stokes, ordinarily Q2 velocity / Q1 pressure.
- Particle-carried Maxwell stress history and CPDI-related phase-field infrastructure.
- AT1 phase field and a reconstructed 1D fault embedded in the 2D bulk.
- Localization normalized through `I_h`, with `chi = h(phi)/I_h`; the `h(phi)` localization function is distinct from a mesh edge length.
- Reconstructed-fault slip-rate/state fields, coupled friction, and radiation damping.
- Mature-frictional mode with a frozen phase field for the current cycle experiments.
- True mechanical normal input to friction, including the background exactly once:

  `sigma_raw = sigma_background + p - n^T tau n`, compression positive.

This is a BP3-parameter research model, not an exact standard SEAS BP3 implementation. Differences include the incompressible bulk, finite domain, diffuse fault, filtered normal input, and fully frictional deep extension. Do not introduce compressible elasticity, 3D faults, or fault propagation as part of session recovery.

Historical environment: ASPECT 3.1.0-pre, deal.II 9.6.0, Trilinos 15.0.0, p4est 2.8.5. Verify the actual environment rather than rebuilding those versions from memory. AMG has been the reliable server path; GMG previously worked locally but failed in a server configuration. Reuse the known working environment and scripts.

## 3. Current state: BP3 restoration already progressed

An older handoff requested BP3 restoration. More recent evidence, dated September 25, shows the restored model has already completed three short filter-comparison runs. Do not redo the restoration merely because the old session is gone.

The latest uploaded `original(3).prm` specifies:

| Item | Observed configuration |
| --- | --- |
| Domain | 150 x 50 km; x origin -60 km |
| Root repetitions | 75 x 25 |
| Initial refinement | Global 1, adaptive 8; saved target-cell fixture |
| Runtime mesh refinement | Disabled |
| Bottom and side boundaries | Prescribed `reconstructed fault BP3` velocity |
| Explicit prescribed traction list | Empty |
| Gravity / surface pressure | Zero / zero |
| Pressure normalization | `no` |
| Phase-field length | 20 m |
| Reconstructed-fault spacing | 20 m |
| Phase evolution | Frozen |
| Friction parameters | a = 0.010 / 0.025; b = 0.015; Dc = 0.008 m |
| Reference friction / rate | 0.6 / 1e-6 m/s |
| Shear modulus | 32,038,120,320 Pa |
| Bulk viscosity | 1e26 Pa s |
| Radiation damping | 4,624,440 Pa s/m |
| Normal input | True mechanical stress, Helmholtz filter 20 m |
| I_h integration backend | `cell intervals` |
| Particle advection / layout | RK2; 3 x 3 particles per cell initially |
| Composition space | Continuous Q2 |
| Interpolation setting in this PRM | `linear least squares`, limiter false |
| Maximum first timestep | 100 s |
| Maximum timestep | 4e6 s |
| Material artificial initial interval | 4e6 s |
| Maximum logarithmic state change | 0.02 |
| Linear / nonlinear tolerance | 1e-9 / 1e-8 |
| Startup termination | Last accepted step 10; graceful wall limit 3600 s |

The PRM's plugin path is `plugin/build/libbp3_restore_150x50.release.so`, with geometry/profile assets under `fixtures/`. These paths are relative to its launch directory, not necessarily the repository root. The timestep controller is still named `BP5 state startup`; a legacy name alone does not establish incorrect behavior.

The design intent was a 60-degree fault from (0, 50 km) to approximately (28.8675 km, 0), full length 57.735 km. Along-fault a transitions over 15--18 km, and the strengthening fault remains frictional to the bottom. The finest target bulk edge was 3.90625 m, coarsest 1 km, with broader grading bands. Actual mesh counts/support widths must come from the generated mesh. A prior approximate 0.49-million-cell calculation was an estimate, not a measured ASPECT inventory.

Dirichlet loading should impose a smooth two-component relative plate motion across the diffuse fault consistent with the localization and signed slip convention. The far-side relative rate is 1e-9 m/s, not twice that value. It controls boundary motion, not every interior deep slip rate. Preserve the working top/bottom normalization and profile assets unless a new change requires regeneration.

## 4. Most recent completed result: restored BP3 filter startup

Source: uploaded `README(3).md`, titled "Restored BP3 filter comparison -- 2026-09-25", plus `original(3).prm`, `growth.png`, and `final_profiles.png`. Its analysis command is:

```sh
python3 benchmarks/reconstructed_fault/bp3/compare_restored_filters.py \
  benchmarks/reconstructed_fault/bp3/filter-test
```

Raw, 20 m, and 40 m cases reached accepted states 0--10, ending at 1133.7390530913 s, about 19 minutes. Every state took two Newton updates, each run had 493 total Krylov iterations, and the reported residual/state checks passed. This is startup evidence, not qualification of years of loading or an earthquake.

The report recommends 20 m provisionally. Relative to the raw observational Q1 profiles, its tangential chord roughness fell about 8--11 times. Final bottom-1-km velocity chord RMS/Vp changed from 5.9573e-7 (raw) to 2.1584e-7 (20 m) and 1.9408e-7 (40 m). These are tiny absolute velocity differences. The extra smoothing at 40 m had no solver-iteration advantage.

The roughly 1.0892%-of-Vp local chord departure near 15 km remained in all cases. Increasing this normal filter does not cure that feature. Raw bulk stress persists in the filtered cases. The artificial initialization response is much larger than changes in the first real timesteps; analyze those phases separately.

Do not rerun these three startup studies just to rediscover their conclusion. Locate the evidence and continue from the qualified configuration, subject to any newer completed work in the checkout.

## 5. Recent questions that remain unresolved in this handoff

The user subsequently asked whether the first-step cap could be restored from 100 s to 4e6 s and whether the logarithmic state-change bound could rise from 0.02 to 0.1 or more, since nonlinear solves use only 2--3 updates. This handoff does not contain a confirmed later decision or an accuracy qualification. Inspect newer notes before changing either.

Keep separate:

- The first real accepted timestep and its adaptive bounds.
- The material/artificial initialization interval.
- Newton convergence, which does not by itself establish temporal accuracy of state evolution.

Earlier BP3 initialization calculations expected about 8,000 s shallow Theta and 8,000,000 s deep Theta for a particular constant-prestress initialization. Treat these as reference calculations to check against the actual restored plugin, not as permission to overwrite newer initialization choices. Verify the background/incremental stress split and that artificial timestep-zero stress is not accidentally retained twice.

**Interpolation discrepancy to check:** the favorable local comparison in section 6 applied unlimited LLS only to Maxwell stress through a routing adapter. The September 25 PRM selects unlimited LLS globally. Verify whether the current implementation still routes non-stress properties separately or whether this was a deliberate, separately checked change. Record the answer; do not silently change or label it a confirmed bug.

A limited cleanup was recommended before long BP3 work. Whether it has been completed must be determined from the checkout, not inferred from that recommendation.

## 6. What the interpolation comparison established

Sources: uploaded `README(2).md`, `comparison(2).json`, and `comparison(9).png`; repository area `benchmarks/reconstructed_fault/bp5/interpolation-inclined/`.

The existing local fixture used a 64 x 64 Cartesian mesh, a 60-degree fault, ell/h=10, prescribed slip, zero retained initial stress, frozen phase, and four 0.1-s real steps. It advected 36,864 particles with production RK2. No particles changed host cell in that short interval. One-/two-rank results were checked.

| Case | Stress interpolation / FE space | Final interior transfer-load jump (Pa m) | Whole-domain jump (Pa m) |
| --- | --- | ---: | ---: |
| A | DWA, linear weights / continuous Q2 | 0.293767 | 63.3274 |
| B | Unlimited LLS / continuous Q2 | 0.060060 | 64.3627 |
| C | Unlimited LLS / DGQ2 | 0.523758 | 96.9949 |

Case B reduced the interior load jump by about 79.6%, but increased the whole-domain value by about 1.6%. The weak normal-traction chord RMS was almost unchanged: 0.493023, 0.493074, and 0.493392 Pa. The main B-minus-A normal-profile change was a mean shift of approximately 0.55 Pa. Pointwise tensor transfer RMS did not improve.

Conclusion: prefer continuous-Q2 LLS as a BP3 pilot candidate; stop the DG investigation for now. This is not proof that LLS eliminates normal-stress bands, preserves all weak moments, or qualifies free-friction feedback. Non-stress particle properties retained DWA in the controlled comparison.

The inspected production publication routine already sums incident-cell proposals and divides by the contribution count after MPI addition. It avoids "last writer wins" for continuous DoFs. Continuous nodal averaging and constraints still modify independent cellwise fits. Do not reintroduce overwriting, or assume DG is necessary to make the current routine deterministic.

DWA's "linear" describes a radial weight, not affine-field reproduction. Unlimited LLS had already passed an affine-tensor transfer test. Do not rerun manufactured tests without a specific code change that could invalidate them.

## 7. Earlier results and mistakes not to repeat

- A long BP5-like run reached nucleation above its 30--33 km transition at about 168 years. Parameters included ell=100 m, Dc=0.1 m, a=0.004/0.04, b=0.03. This milestone is real evidence; do not describe all earlier models as having failed to nucleate.
- Bottom traction loading allowed deep slip to slow. This motivated bottom Dirichlet velocity for restored BP3.
- A prescribed/free internal fault junction previously caused a notch. The fully frictional deep extension removed that mismatch. Do not casually restore the internal clamp at 40 km.
- Missing/truncated endpoint contributions to I_h caused an earlier bottom hotspot. Retain qualified endpoint completion; distinguish it from obsolete traction-loading corrections.
- Oscillations also occurred in uniform fine cells and wholly within an MPI rank. Refinement changes or rank boundaries cannot explain the entire normal-stress pattern.
- Raw pressure and deviatoric normal contributions partly canceled. Raw samples were already oscillatory before consistent-Q1 surface projection; mass inversion was not the sole source.
- A frozen BP5 history-cycle audit found strong subcell variation in incoming particle history, with no demonstrated index, tensor-component, time-level, or constitutive-dt defect in that captured cycle. Smaller dt did not proportionally reduce its re-equilibration response. Do not interpret all small-dt stress changes as time-integration errors.
- Clean-start native-history/moment experiments established a transfer contribution to raw stress structure. They did not establish transfer as the dominant cause of the MPa weak-normal bands.
- The inclined fixture prescribed slip and used a separate constant/adiabatic friction input. Its diagnostic mechanical pressure/traction was not the friction input. A good mechanical fixture result alone cannot establish free-RSF feedback.
- Qualified pressure-compatibility/convergence fixes were developed while diagnosing small-residual solves. Do not remove them just because their origin was diagnostic work, and do not relax tolerances to make a new test pass.

## 8. Normal filter: preserve its meaning

The selected filter acts on the total normal input to friction using the reconstructed-fault Q1 basis and native work weights:

`(M + Ls^2 K) z = b(sigma_raw)`, followed by evaluation of the Q1 field at work points.

The physical arc-length derivatives and consistent mechanical derivatives are part of the implementation. The filter modifies friction input; it does not overwrite raw pressure, current bulk stress, or particle history. Recompute the right-hand side from the current raw mechanics, not from previously filtered output.

- Raw pointwise mode is distinct from zero-length projected mode.
- Preserve the global work-weighted mean, not every local window mean or every friction load when mu varies.
- Use physical endpoints and existing normalization; do not make plotting windows into filter boundaries.
- Retain current stress/state labels. Mechanics uses incoming state; plotted committed Theta can be post-update. Omega=|V|*Theta/Dc needs consistent time-level labeling.
- BP5 100/200-m restart-filter trials cannot be carried unchanged into ell=20-m BP3. Enabling a filter midway through a loaded checkpoint can cause an immediate imbalance/transient; a clean-start comparison answers a different question.
- Filtering has not been shown to eliminate accumulated raw-history modes. Keep raw and filtered diagnostics separate.

## 9. Cleanup and validation policy

Do a limited production-path cleanup, if still needed. Preserve a source/configuration checkpoint first. Keep qualified correctness fixes, explicit filter choices, and the intended LLS treatment. Isolate prescribed-slip, frozen-advection, native-history, diagnostic-pressure, and other experimental overrides in benchmark configurations/plugins. Remove obvious temporary clutter only when its role is understood.

Defer broad class/ownership refactors, mass renaming, new history representations, and a solver redesign until the first BP3 event has been assessed. Keep cleanup commits separate from changes to physics, geometry, loading, or timestep policy.

Reuse focused tests and existing reports. Add or rerun checks only for a concrete changed behavior. Do not launch full suites, broad parameter sweeps, or another complete BP5 trajectory as part of recovery.

The current research goal is a reproducible BP3-parameter first-event run, with raw/filtered stress, state, slip, endpoint loading, timestep limits, and solver health monitored. A successful ten-step startup is not an event. Stop-after-first-event logic should include acceleration, peak, and decay/arrest, not only the first threshold crossing.

## 10. Where to look for surviving evidence

Repository paths are navigation hints; resolve their current locations:

- `benchmarks/reconstructed_fault/bp3/`: restored model, initialization, filters, run monitors, fixtures, and first-event logic.
- `benchmarks/reconstructed_fault/bp5/`: old BP5 results and the stress-cycle/moment/interpolation diagnostic fixtures.
- `source/material_model/phase_field_fault.cc` and `source/material_model/rheology/fault_friction.cc`.
- `source/reconstructed_fault/surface_system.cc` and related fault manager/system files.
- `source/simulator/assemblers/reconstructed_fault_stokes.cc` and reconstructed-fault solver code.
- `source/simulator/initial_conditions.cc`: particle-to-FE publication and shared-node averaging.
- `source/particle/interpolator/` and Maxwell stress property/update code.

Important uploaded names may differ from their original repository names:

| Uploaded artifact | Meaning |
| --- | --- |
| `README(3).md`, `original(3).prm`, `growth.png`, `final_profiles.png` | September 25 restored BP3 filter startup; newest report read for this handoff |
| `bp3(20260925-195648).cc` | September 25 BP3 source snapshot; includes other local headers and is not a complete build by itself |
| `plugin.tar` | User-uploaded plugin archive; contents not audited for this handoff; inspect in a separate location before comparing or restoring |
| `README(2).md`, `comparison(2).json`, `comparison(9).png` | DWA/LLS/LLS-DG inclined comparison |
| `report(5).md`, `summary(3).json`, `offline.png`, `evolution.png` | Earlier BP5 checkpoint filter experiments |
| `stress_cycle_report.md`, `report(1).md` through `report(4).md` | History/weak-load/moment/traction-definition investigations |
| `bp3_restore_150x50_instructions.md` | Earlier design specification; partly implemented and superseded by later evidence; not a fresh to-do list |

Do not expect access to this ChatGPT conversation, its Library, or its scratch paths from the new CLI session. The user must place missing reports/snapshots in the repository or another accessible local folder. Existing equivalent repository reports are preferable to duplicate uploads. No report or archive should be treated as an automatic overwrite instruction.

## 11. Durable context for future sessions

Maintain a small `doc/reconstructed_fault/CURRENT_STATUS.md` after meaningful milestones. Record:

1. Date, active branch/commit, meaningful uncommitted changes.
2. Current executable/plugin/configuration and exact launch/restart commands.
3. Completed checks and evidence paths, including numerical outcomes and limits.
4. Decisions and why; rejected approaches and what their tests actually established.
5. Current run/checkpoint locations and the next bounded task.

Keep a separate `doc/reconstructed_fault/DECISIONS.md` only if the existing project notes do not already serve that role. Link larger reports instead of copying them into the automatic startup instructions.

Suggested short addition to the existing applicable `AGENTS.md`:

```markdown
## Phase-field / RSF research context
- For RSF work, first read doc/reconstructed_fault/current_design.md and doc/reconstructed_fault/specification.tex.
- Use doc/reconstructed_fault/CODEX_RECOVERY_HANDOFF.md for recovered historical context.
- Verify current code and executed-run evidence before treating an old plan as unfinished work.
- Preserve uncommitted work and simulation checkpoints; keep numerical changes separate from cleanup.
- Update CURRENT_STATUS.md after meaningful milestones, including evidence, limitations, and the next task.
```

Version-control the compact research notes and useful configurations/scripts, and back up the repository plus important untracked plugins, fixtures, checkpoints, and reports outside the machine. Keep credentials and large generated output out of source commits. A copy of `~/.codex` can help preserve future sessions, but project knowledge should also survive independently in these repository documents.


Official Codex references for launching a session and repository instructions:

- https://developers.openai.com/codex/cli
- https://developers.openai.com/codex/guides/agents-md
- https://developers.openai.com/codex/learn/best-practices
