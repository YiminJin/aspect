# Mechanical surface-solver snapshot

Uncommitted selected-source archive: `mechanical-source.tar.gz`.
SHA256: `c61f9000486631aafe4a32dfebabaa6986fe7055993694d4ac93c128c2eb4abc`.

This is an overlay of this task's source/test/input/documentation files on the
retained working tree, not a repository reset or a complete archive of unrelated
changes. The preceding accepted baseline is recoverable from
`bound-cell-source.tar.gz` and its manifest. Base git HEAD remains
`fb4411915e267ba27bd0c306b9e00386f1a99cab`.

Tested executables:

- Original timed UMFPACK baseline:
  `78720bc68980cae2591b1d67d178074252ddabb84b0ab9f7c767d3cac4307267`.
- Pivoted prototype and focused tests:
  `b9ae3053fb6e0e03600e7e9c7bf4b69baa14cc998737c1341ccbaca1ab72a31f`.

The exact executable hash in the per-run `.resources.json` is authoritative.
No I_h source change occurred: `source/material_model/phase_field_fault.cc`
remains SHA256 `36e86323bbd508ee25ca92ccb0034646255cb43ca319beb559bf03010cd930b6`.

The LAPACK path is opt-in, UMFPACK remains the default/reference, and no
interface preconditioner is implemented. Detailed numerical/performance status
and the separate proposal are in
`doc/reconstructed_fault/bp3/stage_K5_surface_solver_report.md` and
`stage_K5_interface_preconditioner_proposal.md`.

No commit, fine BP3 pilot, propagation run or pending legacy I_h repair is
included. No claim of an overall speedup is made from the single-run wall times.
