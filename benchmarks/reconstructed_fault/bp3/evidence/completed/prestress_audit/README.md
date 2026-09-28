# Pre-mechanics Airy audit

See [the report](../../../../doc/reconstructed_fault/bp3/stage_K5_prestress_audit.md).

The completed attempt is `audit2.prm` / `audit2.log` / `output2/`. It stops in
the existing assembler-setup callback before the first nonlinear residual or
linear solve. Exit 1 is intentional; `run.py audit2` requires the completion
and stop markers and checks that no mechanical-iteration messages exist.
Do not rerun it merely because the process exits nonzero.

`audit.prm` / `audit.log` preserve a setup-only rejected no-Stokes harness.
It is not a physical result and cannot preserve the required boundary setup.

Common-grid stress and differences: `output2/particle_common.csv` and `.vtp`.
Production bulk QPs: `quadrature.csv`. Radial data: `radial.csv` and
`radial_full_box.csv`. Global constrained loads: `loads.csv` and `weak_loads.vtp`.
Raw per-cell norms: `cells.csv`. Assembled norm attribution and saved-correction
correlation: `cell_correlations.csv` and `cell_diagnostics.vtp`.

Analytic stress is total stress; Maxwell histories are deviatoric. The common
file exports both, with analytic pressure. Published and constrained working
FE fields are separate. The constant-shear control uses the original variable
bottom traction; its known boundary mismatch is an explicit diagnostic, not
a boundary-condition change. All analysis is offline/noncommitting.
