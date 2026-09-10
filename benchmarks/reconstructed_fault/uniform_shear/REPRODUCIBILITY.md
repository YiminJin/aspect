# Saved correctness baseline

The repository contains the production corrections, regression tests, benchmark
plugin and inputs, independent reference/analysis scripts, and compact final
K1/K2.1 verification summaries. The current decisions and limitations are in
`doc/reconstructed_fault/benchmarking/stage_K1_verification.md` and
`stage_K2_1_pilot.md`; earlier reports are historical evidence, not overrides.

Raw run directories, checkpoints, build products, full diagnostic logs and
recovery archives remain local and are intentionally not versioned. In
particular, the reports' ParaView and raw-QP links refer to local generated
artifacts. Recreate the final runs using the checked-in parameter files and
runner commands documented in the reports. The runner records actual accepted
execution commands, executable/plugin hashes and resource use. Saved hashes
identify the measured binaries; rebuilding on another machine need not produce
byte-identical binaries.

The final K1 five-case cross uses `residual-floor/convergence/*.prm`, with
one-/two-rank and restart evidence in `residual-floor/`. The bounded K2.1
fixture is `nonuniform/pilot.prm`; compact sampled fault profiles and its
measurement report are in `nonuniform/measurements/`. The subsequent K2.2
refinement is in `nonuniform/refinement/`. The user explicitly revised the
original 1e-6 containment allowance to provisional 1e-4 for this fixed-profile,
prescribed-pressure K2.1/K2.2 family only; this is a separate approval, not
automatic inheritance of K1's allowance. Actual normalization remains 1e-4.
The approved K2.2 execution campaign is complete, but its numerical reference
remains provisional and Gate K2 remains unmet. The current assessment is in
`doc/reconstructed_fault/benchmarking/stage_K2_2_temporal_completion.md`;
earlier spatial reviews retain their historical context. The subsequent
bounded stress-transfer correction and consumption/output timeline are in
`stage_K2_stress_transfer_timeline.md`. Compact JSON evidence is versioned
alongside the fixtures; full CSV/VTU exports and recovery archives remain local.
The accepted Cartesian lookup/rejection optimization is retained. K2.3 is the
next authorized stage, not part of this saved implementation or verification.
