# Cold I_h fallback: bounded diagnosis

The retained nine-profile Release reproducer was stopped at its first
unaccelerated lookup. `cold-fallback-captured.log` records the actual request:
`(0.00044024088038695236, 0.50000042146977086)`, tolerance `1e-6`.
The saved FE mesh has y bounds [-0.5, 0.5] and cell width 1/256.
The outside distance is 4.2146977086e-7 physical units, or
1.0789626134e-4 reference-cell units. It passes the absolute 1e-6 bounding-box
prefilter but cannot pass the 1e-6 reference-cell containment test. The
unaccelerated routine then expands through the mesh, up to all 65,536 cells.
This is a conservative false-positive in the broad search, not evidence of
incorrect point ownership or a missing geometric cache.

The installed deal.II 9.6.2 path already uses the persistent GridTools::Cache,
its vertex/cell R-trees, and a previous-cell hint within each request batch.
Both hint and nearest-vertex candidates fail for this outside point. Increasing
R-tree depth does not remove the absolute-margin admission. The public
RemotePointEvaluation options provide no separate conservative rejection
predicate: changing its tolerance would also change reference-cell acceptance.

No production optimization is applied in this task. The smallest justified
next proposal is a mapping-aware rejection before exhaustive fallback, with
certified enclosures of tolerance-expanded reference cells and the original
search for unsupported mappings. Implementing this at the ASPECT layer would
also need filtered-request/index handling; it is not a safe one-line parameter
change. A fixture-specific y-bound shortcut is deliberately not introduced.

The first debugger invocation exited before execution because of shell quoting;
the corrected capture completed within its 120-second cap and deliberately
killed its inferior at the breakpoint. Neither is a numerical pass. No
all-profile or trajectory run, timing retry, production edit, or change to the
verified lookup reuse occurred. New cold-time/memory/integral comparisons are
not claimed because there is no candidate implementation. Existing passing
evidence remains valid. Only the debugger script and diagnostic artifacts were
added; the older unfinished performance-report addendum remains unfinished.
