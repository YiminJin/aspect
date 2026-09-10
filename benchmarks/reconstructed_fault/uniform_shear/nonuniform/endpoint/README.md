# Bounded endpoint-traction diagnosis

Follow-up: `doc/reconstructed_fault/benchmarking/stage_K2_endpoint_moment_regression.md`
records the manufactured constant/even/odd transverse-stress tests in
`../endpoint_moments.py`, raw moments in `moments/`, and the independent
Voro++ area cross-check built by this directory's CMake file. Ten analysis
tests pass. The evidence supports a surface quadrature revision for review,
not a periodic-domain or history-transfer patch. No further mechanical replay
has been run; the 64/128 through-1-s comparison awaits correction approval
and focused verification. The record below is the prior saved-data diagnosis.

See `doc/reconstructed_fault/benchmarking/stage_K2_endpoint_diagnosis.md`.
This is saved-data analysis plus one diagnostic-only 64x256 replay through
1 s, not another refinement campaign or a corrected trajectory.

The dominant measured endpoint signature is the changing point-volume/Q1
quadrature of transverse stress at fixed-wall-clipped particle domains.
Correct old particle history is joined by stable ID; the replay distinguishes
published FE history from the constrained working history actually assembled.
Both are exported, with no history or solution mutation by the diagnostic.

`replay64.prm` changes only the output location and end time of the accepted
pilot. Cost announced before execution: 200--350 s / about 1 GiB. Actual cost:
283.290 s / 916352 KiB on one rank. Build with -j4. Accepted fields match the
saved pilot at 0, .5 and 1 s. Five diagnostic tests and fifteen existing
reference/analysis tests pass. The report records exact reproduction commands.

`summary.json`, `adjacent_split_at_1s.csv`, `endpoint_diagnosis.png` and the
`*-analysis` / `saved32` / `saved128` reports are local diagnostic evidence.
Previous-measure projections are algebraic diagnostics, not smoothing,
proposed corrected histories or an independent mechanical reference.

No periodic-domain fix, history-transfer fix, support/I_h change, endpoint
identification, Jacobian change, temporal sweep or true-pressure run was made.
Review the endpoint-moment consistency issue before a production correction.
