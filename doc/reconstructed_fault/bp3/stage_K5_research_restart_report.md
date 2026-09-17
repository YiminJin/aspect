# Modified-BP3 research restart qualification

## Scope and implementation

This task attempts to qualify the cleaned, fully frictional, continuous-Q1 mature C=0
work-measure fixture on the same four MPI ranks. It does not change the split
aging cycle, equations, background, endpoint corrections, solver tolerances,
mesh, or saved clock. Cross-rank restart and long-cycle behavior are outside
scope. The uninterrupted reference remains
`benchmarks/reconstructed_fault/bp3/fully-frictional-cleanup-local4/`.

The Release executable is unchanged:
`build-pf-cpdi/aspect-release`, SHA-256
`a71425a9af08d5f506011d63cfcb5715d37e8625f07e71b2454b30907bbe0e11`.
Only the benchmark plugin was rebuilt (`cmake --build
benchmarks/reconstructed_fault/bp3/build --target bp3 -j4`).

### Maintained inputs

`benchmarks/reconstructed_fault/bp3/fixtures/modified_bp3/` contains immutable,
byte-identical copies of the bulk leaf-cell tree, continuous fault, fixed
prestress coefficients, paired endpoint completion and seven-step clock.
`manifest.json` records SHA-256 and original provenance; the launcher checks
each digest. Original artifacts were not moved or removed. The maintained
PRM and runner no longer need an investigation directory to run this fixture.

### Checkpoint ownership and restoration

The ordinary simulator/manager checkpoint already owns bulk vectors, particles,
fault state/V, background coefficients and geometry. No new copy of physical
history was introduced. Reconstructible caches, B/G, and factors are rebuilt.
The benchmark's version-4 record additionally serializes the initial audit
baseline (stable-ID inert H, frozen geometry and completed Ih), and validates
the fully-frictional/work-measure selectors. Existing slip, preceding Theta,
last accepted step, event and output schedule serialization is retained.

The runtime work measure, both source extensions, completion input and fixed
background-property selector are reattached by the existing preparation hook
before mechanics. Restart does not reload/recalibrate initial prestress or
recapture an audit baseline from the resumed state. Earlier work-measure
version-2/3 checkpoints lack this audit data and are rejected rather than
silently reinterpreted. This is why a fresh prefix was required here.

Before editing, the prior benchmark source, runner and plugin were preserved
in `restart-qualification-preserved/`; the preceding cleanup snapshot and
successful trajectory remain intact. Unrelated working-tree changes remain.

## Bounded qualification

```
python3 benchmarks/reconstructed_fault/bp3/run_research.py \
  --mode restart-qualification \
  --output benchmarks/reconstructed_fault/bp3/fully-frictional-restart-qualified-local4
python3 benchmarks/reconstructed_fault/bp3/check_research_restart.py
```

The two invocations use the complete original clock, with end-step limits 4
and 7, not a shortened first-leg physical end time. At normal checkpointing,
ASPECT has advanced its clock to step 5 while holding the state accepted after
step 4. Thus the checkpoint retains the already-selected next dt; it is not a
newly committed step-5 state. `step4_checkpoint/` preserves the ordinary
checkpoint tree before subsequent rotation. Its last-good ID is **02**.

The fresh prefix completed in **460.011502 s**, with 43 passing fresh-linear
checks and 760 Krylov iterations. Initialization and steps 1–4 passed their
physical/history audits. Peak child RSS was 1,532,260 KiB (not aggregate MPI
memory). The separate process resumed at step 5 on the original clock.

The initial sandbox MPI attempt failed before ASPECT execution because local
sockets were disallowed; its log remains in `fully-frictional-restart-local4/`.
The qualification above ran with the required MPI permissions, not changed
numerics. There is no automatic failed-simulation retry.

## Comparison definitions

`check_research_restart.py` reuses the established **1e-8 per-component**
comparison coefficient. Geometry, frozen phase/Ih/background, IDs, masks and
clock are checked exactly. The retained **1e-12** aging reference check is not
relaxed. Separate comparisons cover:

- Nodal V, Theta, slip, C and shear; all 1236 nodes free, none prescribed/active.
- Current constitutive QP p/stress/strain, full source coverage and localization.
- Stable-ID committed particle H/stress, distinct from current FE constitutive
  stress. Source-update diagnostics contain retained **particle** old stress,
  not a mislabeled FE-history output.
- Boundary-source old stress equals the preceding accepted particle array by
  ID, and each exported candidate equals its newly committed array. The
  Maxwell formula is checked independently from its recorded coefficients.
- One exact slip increment and one frozen-rate aging update per accepted step.
- Actual assembled segment source integrals, including both endpoint wedges,
  and native work-weighted weak loads.
- Available ordinary Float32 bulk-output arrays, without conflating their
  precision with the separate full-precision stress/strain CSV checks.
- Genuine final nonlinear convergence and fresh linear checks in both logs.

The working FE history remains frozen throughout mechanics; the accepted
work-QP observer uses that same working history, not a refreshed published
array. Matching its stress and strain plus the identical coefficients tests
the mechanical use of the restored FE history, as well as its publication.

## Bounded continuation

`run_research.py --mode continuation` requires explicit physical end time
(seconds), absolute last accepted step and wall budget. It removes the replay
time-step model and uses only the ordinary convection/fault controllers. It
retains physical acceptance checks and ordinary checkpointing every accepted
step and on termination. The soft wall limit precedes the process cap by
60 s; a solve crossing that limit can be killed at the cap (15-s TERM grace),
leaving the preceding complete checkpoint. No trial history is checkpointed.

The prepare-only example in `research-continuation-preflight/` uses 10 yr,
step 20 and 2400 s. Its PRM passed ASPECT syntax validation. A missing-bounds
invocation was rejected with exit 2 before creating output. **No adaptive
continuation was launched.** These are explicit safety bounds, not a claim of
adaptive long-time scientific qualification. See the maintained fixture's
`continuation.md` for exact commands and recovery restrictions.

## Final comparison

**Restart is not qualified.** The resumed process failed after **35.916975 s**,
before its first mechanical linear solve or accepted history publication.
The combined simulation time was **495.928477 s**. No simulation retry or
long continuation was launched.

The exact failed invariant is in
`PhaseFieldFault::compute_cohesive_response()`:
`previous_cohesive_traction == 0 && current_h == previous_h && current_I_h == previous_I_h`.
Restart deliberately invalidates transient normalization caches, then rebuilds
Ih from the unchanged frozen profile. The uninterrupted run retained its
completed-value cache. Saved profile exports show:

| Check | Result |
|---|---:|
| Profiles compared | 3705 |
| Profiles differing after cold recomputation | 24 |
| Profile coordinates, weights, outside completion | exactly identical |
| Maximum inside/completed integral difference | 3.637978807091713e-12 m |
| Difference scaled by maximum Ih | 2.8061187753494915e-16 |
| Largest profile ID / segment | 3240 / 1080 |
| Offline linear Q1 projected difference | 2.2177730968538584e-12 m |

The Q1 figure is the independently computed solution of the **difference**
projection system, not a direct export of private values at the failed call.
The source and exports identify cold recomputation versus a bitwise frozen
history requirement as the blocker. The exact origin of the last-bit change
within FE evaluation/integration/reduction has not been isolated; no geometric
or physical change is evidenced. The largest changed profile has zero outside
completion, so the observed change is not a changed endpoint-completion input.
No tolerance was changed and no invariant bypass was installed.

Commands/artifacts:

```
python3 benchmarks/reconstructed_fault/bp3/analyze_research_restart_ih.py
python3 benchmarks/reconstructed_fault/bp3/check_research_restart.py --prefix-only
```

The latter **passed**: accepted states 0–4 match the cleaned uninterrupted run
exactly for V/Theta/slip/C, current stress/strain, stable-ID H/stress, old and
new endpoint particle stress, actual source integrals and weak loads. Frozen
fields, backgrounds and clock match exactly. The selected next timestep is
**17300729.400961984 s**, also exactly matching the reference. The independent
aging error is at most **2.217194920185595e-16** (existing limit 1e-12), and the
recorded endpoint Maxwell update agrees to **1.7171334090810495e-16** relative.
The independent slip audit uses fused multiply-add, matching the Release
accumulation: an initial two-operation Python expression differed by one ulp
(8.67e-19 m), not by an extra update. No comparison coefficient was relaxed.
The CSV loader was corrected to admit header-only MPI ranks in endpoint-only
exports, without dropping any populated rows.

Evidence in `fully-frictional-restart-qualified-local4/`:
`prefix_equivalence.json`, `restart_ih_diagnosis.json`,
`restart_ih_profile_difference.csv`, `restart_ih_linear_projected_difference.csv`,
`run.log`, `resume.log`, both execution/provenance files, and the preserved
`step4_checkpoint/`. Its directory name denotes the intended test, **not a pass**.

### Smallest next action requiring review

Restore the persisted frozen normalization as the authoritative current
normalization on a validated frozen mature restart, without restoring geometric
search caches or changing the invariant/tolerances. Its eligibility must verify
the applicable frozen phase/geometry/degradation/completion state; changed
states must still rebuild or fail. This requires a material-model lifecycle
change and rebuilt core binary. It cannot be honestly called qualification of
the requested unchanged executable. Do not implement it via a benchmark
test-access setter, overwrite previous-Ih history with new rounded values, or
weaken equality merely to pass.

After approval, reuse the accepted step-4 checkpoint for the targeted resumed
check; do not repeat the historical campaign. The full steps-5–7 comparison,
first-resumed history publication and adaptive execution remain unverified.
Continuation execution is explicitly disabled (prepare-only) until that gate
passes. Cross-rank restart remains deferred.
