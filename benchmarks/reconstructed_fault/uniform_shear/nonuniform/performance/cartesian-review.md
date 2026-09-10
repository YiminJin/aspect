# Production Cartesian verification

Production passes its actual `MappingCartesian<2,2>` to I_h through the
phase-field GridTools::Cache. The previous exact-type guard excluded it;
exact Cartesian support now uses the same conservative enclosure. The mapping,
reference/search tolerances, integration, support, and ownership are unchanged.

Same-Cartesian nine-profile comparison: cold 17.216761 -> 3.714599 s (4.63x),
warm 0.320077 s. All nine integrals and the complete found/missing digest
match; counts are 811,863 found / 522 missing. Peak RSS changes by +404 KiB.

Boundary/shared-face/shared-vertex and MPI-indexing tests pass: 59,243
assertions on one rank, 59,243/2,521 on two ranks (four cases each).

The real K2 t=0/.5-s prefix logs Cartesian, eligibility=1, and 2,448 excluded
requests per preparation. Cold/warm profile preparation is 3.194/1.636 s;
warm reuse hits all 1,295 batches. Both nonlinear criteria and 13 fresh-linear
checks pass; all 19 CSVs equal the saved prefix byte-for-byte. Runtime: 30.92 s.

Evidence: `cold-cartesian-comparison.json`, `cartesian-prefix-comparison.json`,
`cartesian-unit-{one,two}.log`. Release builds used -j4. Testing stayed below
two minutes per run and ten minutes total; no broad campaign ran.

K2 resumes from saved spatial/partial-temporal evidence. The unresolved
mean-removed temporal trend remains the review point. Record opt-in performance
on the next necessary authorized run. Changes remain uncommitted.
