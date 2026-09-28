# Frozen width comparison (no evolving histories)

Report: `doc/reconstructed_fault/bp3/stage_K5_frozen_mechanical_width.md`.

`ell400` and `ell200` use the same 48.828125-m probe patch. The one permitted
paired local refinement is `ell400-finer` and `ell200-finer`, at 24.4140625 m.
Every case recomputes its own phase, I_h and completion data. No prestress
recalibration or accepted timestep occurs.

Start with `comparison-finer.json`, `coefficients-finer.csv`, and
`profiles_and_inputs-finer.png`. The corresponding files without `-finer`
preserve the first comparison, including its narrow-profile normalization
limitation. Raw cell/QP and surface data remain in each case directory.

Exit code 1 is intentional ONLY when the log contains
`MECHANICAL MODES VERIFIED; intentional stop before trial/history publication.`
The driver requires this marker and all in-process numerical checks. It does
not infer success from ordinary process termination.

`preparation-history/` contains preparation-only attempts; `ell400/sandbox-mpi.*`
records an MPI interface restriction before ASPECT started. No failed physical
result was overwritten or retuned.
