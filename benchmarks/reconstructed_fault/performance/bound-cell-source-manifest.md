# Bound/contact and cell-profile review snapshot

This is an uncommitted, recoverable source snapshot, not a tested commit.
Base HEAD: `fb4411915e267ba27bd0c306b9e00386f1a99cab`.

Archive: `bound-cell-source.tar.gz`

SHA256: `5f6db8802516da0ab7fbfdafff0e9e3db7ea804ba97027c35112ccaaa095fda5`

Final Release executable SHA256:
`aa81d9f76fd7821553af64c2a89281821b229d72f6df56353c33444654218093`.
Each run's `.resources.json` records its actual tested revision, including runs
made before the final read-only boundary diagnostic was added.

The archive contains full selected source/header/test/benchmark-input and
documentation files, including retained accepted K5 work. It is not an archive
of every unrelated working-tree edit or of generated binaries/large outputs.
Use `tar -tzf bound-cell-source.tar.gz` to inspect the exact contents. Restore
into a new scratch directory for comparison rather than overwriting the worktree.

Numerical status: bound correction and three-real-step BP3 smoke pass. K3
evolving and K2 saved-state cell-profile comparisons pass. The alternate cell
backend remains opt-in; BP3 comparison is blocked by a confirmed legacy boundary
omission and smaller unresolved interior errors. The hidden test
`[.ih_boundary_reproducer]` intentionally fails pending correction approval.
No production boundary-panel correction, fine pilot, or commit is included.

Detailed report:
`doc/reconstructed_fault/bp3/stage_K5_bound_and_cell_profiles_report.md`.
Performance/trajectory comparison: `k3-cell-result.json`.
BP3 convergence/history report: `../bp3/bound_tip_smoke/perturbation_report.json`.
Boundary evidence: `../bp3/cell_boundary_audit.log` and its profile CSVs.
