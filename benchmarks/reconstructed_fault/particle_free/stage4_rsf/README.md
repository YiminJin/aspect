# Stage 4 RSF stability

Selected instruction: `CODEX_STAGE4_RSF_STABILITY.txt`. Run in the current
worktree. The protected original physical executable/checker are independently
hashed; the source worktree is not assumed to match those binaries. Existing
manual changes and the closed overnight budget remain intact.

`run_endpoint.py` copies only fine accepted16/.0775 slot02 into a fresh label,
runs the unchanged pending .0025 interval with four bound cores and the original
gates, and enforces 5400 numerical seconds. Large closed CSVs are compressed
only after MPI exits, with roundtrip SHA256. Original checkpoints are untouched.

`prepare_checker.py` generates a separate diagnostic checker from the existing
bulk-tip checker, adds the narrowly scoped `stiffness.h`, and uses retained
ASPECT plugin compiler/link settings. It does not edit/rebuild the protected
core or checker. The build manifest records exact inputs and commands.
`run_stiffness.py` takes one frozen-phase interval from the retained coarse
.075 parent and performs six smooth restoring actions plus one cell-scale control. See
[DERIVATION.md](DERIVATION.md) for the physical operator and all restrictions.

The native diagnostic reuses material derivatives, distributed FE assembly,
section integration/adjoints and the ordinary ASPECT Stokes inverse. Friction is
removed from this restoring measurement; its independent response is combined
only afterwards. All section responses and full bulk/cohesive perturbation
energies remain in the checks. Accepted state is compared before/after.

`after_endpoint.py` waits for the closed endpoint/compression record, then runs
one diagnostic build and one four-core native test sequentially, followed by
single-thread analysis. It never overlaps numerical workers with the endpoint.
Fresh labels are mandatory; existing results are not overwritten by runners.

```sh
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py stage4-fine-endpoint 5580 python3 benchmarks/reconstructed_fault/particle_free/stage4_rsf/run_endpoint.py
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py stage4-sequential-followup 9300 python3 benchmarks/reconstructed_fault/particle_free/stage4_rsf/after_endpoint.py
```

`analyze_endpoint.py` uses native Bernstein contours and the exact 12 previously
identified registration labels, without repeating first-registration analysis.
`stability.py` combines native restoring actions with the actual friction law,
checks the law against native API outputs, evaluates transient and lower-speed
operating points, and selects one synthetic Dc adjustment from the measured
margin. The old sharp-interface operator is not reused.

`reduced_pair.py` is explicitly a small **Ritz diagnostic**, not a native coupled
trajectory. It retains all responsive section ages, nonlinear signed friction,
common Maxwell relaxation and the native midpoint/exact-aging equations. A
single scalar reference coefficient makes b=a and weakening initially match.
Both receive the same force kick and constant generalized driving. Each has its
own unperturbed transient background. Full-step/half-step comparisons and a
variational trajectory test perturbation accuracy independently of work balance.
The native boundary/preload realization remains a separate qualification.

```sh
python3 benchmarks/reconstructed_fault/particle_free/stage4_rsf/reduced_pair.py
```

Native runs, inputs/logs/checkpoints are under `build-particle-free/bulk-tip/runs/`.
Compact measurements, figures and manifests are in `../evidence/stage4_rsf/`.
Do not commit numerical outputs, binaries or checkpoints under the ASPECT source
baseline. No commit is selected by this testing task.

The completed scientific decision is in
[STAGE4_RSF_STABILITY_REPORT.md](../../../../doc/reconstructed_fault/particle_free/STAGE4_RSF_STABILITY_REPORT.md).
The fine attempt capped; the successful native stiffness label is
`stage4-fixed-phase-stiffness-homogeneous`. Earlier diagnostic labels retain the
missing-increment-context failures. `after_endpoint.py` stopped at the initial
sandbox-denied MPI launch; the subsequent explicitly logged retries are separate.
Do not rerun that completed orchestration over existing output directories.

`probe_stability.py` adds the already measured cell-scale action to the same
initial operating point. Its faster unstable mode prevents adopting the
six-mode result as spatial qualification. No extra native solve is required.

`prepare_native_candidate.py` writes four cold-start conditioning inputs and a
cost record. `run_native_candidate.py` uses fresh labels, four bound cores and
the protected physical pair; it cannot resume a checkpoint. These inputs have
not been run and do not realize the reduced preload/drive. The .020 s cold prefix
cost estimate is 2.37 hours per history; the .1323 s diagnostic-length native
comparison would cost about 62.5 hours for both backgrounds and perturbations.
The original boundary speed can reach radiation-dominated sliding, so these
conditioning controls are not a qualified low-speed instability pilot.
`--free-phase` only provides the corresponding fresh-t0 configuration; no such
run is authorized automatically or claimed complete. Resolve the spatial mode
and native preparation first.

`verify_preservation.py` checks entry binary/parent hashes, unchanged incoming
non-document source patch, copied fine snapshot and native restore fields.
Figures have PNG and PDF versions; outputs remain uncommitted.
