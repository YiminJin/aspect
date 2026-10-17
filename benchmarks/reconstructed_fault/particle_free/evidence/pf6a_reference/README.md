# Matched PF6a spatial-reference evidence

[Report](../../../../../doc/reconstructed_fault/particle_free/PF6A_REFERENCE_REPORT.md),
[pre-native formulation/audit corrections](../../../../../doc/reconstructed_fault/particle_free/PF6A_REFERENCE_FORMULATION.md),
[status](../../../../../doc/reconstructed_fault/particle_free/STATUS.md).
**1D mechanical candidate: 512. Native confirmation: failed publication at step 1.**
Unlimited DGQ2, physical laws, native quadrature/defaults and acceptance preserved.

- `assessment.json`: main common-time stress/localized stress, phase/gradient,
  traction/slip/energy/driving, seed representation, mechanical versus phase
  resolution, temporal and quadrature controls, and direct localization remainder.
  Thirteen main reference cases contain 1152 accepted 1D steps. Every main onset
  is .53125 s. 256 fails growing 1%/2% stress targets; 512 passes practical 1D
  targets, with soft translation-position uncertainty retained.
- `operators.json`: independent full Q2 mechanical matrix equivalence and
  energy/gradient/reduced-tangent directional checks, including distinct phase
  and mechanical meshes. The historical split time driver is never used.
- `native.json`: startup matches and exact failure classification; **not a pass
  of native evolution**. Exit 1 after 15.152 s, zero accepted positive steps.
  The assertion's actual error/bound/position are not printed; numerical cause
  remains unresolved. No postfailure rollback-byte equality is claimed.
- `history-checks.json`: independent KKT/irreversibility/unloaded checks on all
  main 1D cases. `selection-before-native.json` preserves the actual pre-native
  decision; targets were declared before that confirmation. The subsequent
  source-origin correction changes no selected native nodal value or target.
- `seed-origin-diagnostic.json`, `literal-table-onset.json`: retained source-
  origin discovery. The physical continuum seed equals every supplied native
  table node exactly. Literal interpolation of the 512 table at phase 1024
  creates an unloaded driving defect/early growth, unlike sampling the existing
  continuum prescription at added nodes. Those literal-interpolant trajectories
  remain separate controls, never the main convergence reference.
- `history`, `snapshots`, `onset` PNG/PDF figures show the main reference;
  `literal-onset` shows the distinct interpolant control. Curves are 1D results;
  there is no native onset/growth curve at the selected resolution.
- `provenance.json`, `archive-manifest.json`, `reference.tar.xz`: exact native
  input/core/observer/source hashes and failure logs, all reference inputs and
  results, failed preliminary controls/searches, BVP seed data and Python source
  dependency closure. Reused <=512 trajectories have byte-identical physical
  initial nodal data under both sampling descriptions; main 1024 cases are fresh.
  Raw diagnostics and pinned binaries/DSO also remain in this worktree's build
  directories. Duplicate archive contents use internal hardlinks.
- `preservation.json`: 9452 incoming source/document/evidence files retained;
  only benchmark README and status/audit additions changed among them. No
  production source/CMake, historical evidence or manual work was overwritten.

Replay without native jobs, with Python, NumPy, SciPy and Matplotlib:

```sh
python3 benchmarks/reconstructed_fault/particle_free/evidence/pf6a_reference/replay.py \
  /tmp/pf6a-reference-UNIQUE-replay
```

The output must not exist. Replay verifies every archive hash, recalculates the
independent operator/measurement checks, classifies the native failure and makes
figures. It does not rerun trajectories or call ASPECT. Numerical evidence
comparison tolerance is 1e-10 absolute/relative; scientific targets are unchanged.
Native confirmation remains false even when the evidence replay passes.
The original native cap was 900 s, serial/one thread; mathematical campaigns
used at most four independent processes with 180-second per-case caps.
The serial DG guard prevents using MPI without separate implementation work.

Stop after this study. The next task is a bounded diagnosis of the actual
publication mismatch, retaining the guard, before another native confirmation.
No AMR/MPI/transfer, moving geometry, birth, coupled transport, RSF evolution or
propagation qualification is asserted; PF5 remains incomplete.
