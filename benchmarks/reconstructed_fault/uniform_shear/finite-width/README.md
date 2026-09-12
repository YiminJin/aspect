# K4 finite-width verification

The reviewed scope and results are recorded in
`doc/reconstructed_fault/benchmarking/stage_K4_plan.md` and the adjacent
K4.1, K4.1b, post-transient, and K4.2 reports. K4.2 qualifies the approved
post-transient interval [4, 6] s; it does not resolve the early transient or
start K4.3.

This directory records the benchmark scripts, A/B/C parameter files,
reference initialization parameters and scalar trajectories, comparison
tables, guard/convergence summaries, resource measurements, and summary
figures. `uniform_shear.cc` supplies the opt-in `ASPECT_K4_STATE_GUARD` hook.
No production equation or solver criterion is changed by that hook.

## Evidence and reproduction

The result reports contain execution and analysis commands. The scripts
reuse the existing uniform-shear plugin, K1 reference, and independent K3
phase initialization; they do not reset reference histories from later
production output. The initialization JSON files retain the fully resolved
input parameters and source hashes. Fresh initialization with
`reference_check.py` additionally reads the locally retained resolved K1
file `../residual-floor/convergence/space32_dt05/parameters.prm`.

Large raw ASPECT output directories, build products, sampled CSV/NPZ
archives, and exploratory logs are deliberately not versioned here. They
remain in the working directory. In particular, the saved-data reference
tests and profile plotting/comparison scripts require the corresponding
`k41/*/profile.npz` and trajectory `dt*.npz` archives. The production
analysis scripts require the raw A/B/C CSV exports. The committed JSON
summaries are review evidence, not replacements for those raw inputs.

With those local reference archives present, run the cheap checks with:

```
python3 -m unittest discover -s benchmarks/reconstructed_fault/uniform_shear/finite-width -p 'test_*.py'
```

The completed local verification passed all 11 tests. The preserved-first-
attempt regression additionally needs `attempt1-volume-guard/k42_A/` and
its sibling log; it explicitly skips when that raw evidence is absent.
Without the reference archives, the full saved-data suite is not
self-contained. The synthetic Q2 reconstruction and boundary-trace checks
in `test_production.py` do not require simulation output.

The first-attempt failure and benchmark-only diagnostic correction are
documented in the K4.2 result report. Raw failed evidence was preserved,
not overwritten. No ASPECT rerun is needed merely to inspect this record.
