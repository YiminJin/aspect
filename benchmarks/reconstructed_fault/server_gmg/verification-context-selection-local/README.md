# Context test-selection correction

The supplied server `output-coarse-context-*` files each ran only the baseline.
This verification tests the replacement of the environment selector by the PRM
`Postprocess / Fault GMG coarse probe / Test mode = context` and the new output
checker. Production core source was not changed or rebuilt in this correction.

The plugin rebuilt successfully with local GCC 12.4/OpenMPI/deal.II 9.6.2.
The direct ASPECT invocation was:

```bash
env -u ASPECT_FAULT_GMG_COARSE_CONTEXT \
  PATH=/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:/usr/local/bin:/usr/bin:/bin \
  ASPECT_SOURCE_DIR=/home/ein/repository/aspect \
  ASPECT_FAULT_GMG_DIAGNOSTICS=0 DEAL_II_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  timeout --signal=TERM --kill-after=30s 300s \
  build-tmp/aspect-release /tmp/aspect-gmg-context-selection/input.prm
```

Exit status was zero. All three cases passed and the 96-line numerical output
is byte-identical to `../verification-context-local/after/coarse_diagonal_probe.txt`.
The resolved PRM records `Test mode = context`. The checker accepted the complete
context result, rejected the supplied Release baseline-only result, and rejected
a copy of the old local output truncated to its first two cases (64 lines).
The corresponding checker logs are adjacent. Shell syntax and whitespace checks
passed. All 12 supplied context-output files passed the adjacent preservation
manifest check.

No Stage-I/BP3/BP5 trajectory, Intel-server test or Python script was run locally.
Build log, input, output, command, resolved selector and artifact hashes are
preserved here. The old environment variable no longer selects any test case.
