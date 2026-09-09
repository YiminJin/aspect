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
measurement report are in `nonuniform/measurements/`. No K2.2 campaign has
been run. K2's original containment requirement remains unmet, and the
reviewed K1-only allowance does not transfer to it.
