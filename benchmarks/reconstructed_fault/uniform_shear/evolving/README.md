# K3 evolving-profile preparation

**Current disposition:** corrected periodic fault32 runs at normal resolutions
128/256/512 pass through 3 s at dt=.375. Total phi improves, but cumulative
I_h feedback and some H/stress errors are nonmonotone. One frozen-profile
scalar impact check finds small direct mechanical effects from the I_h
increment uncertainty. Further spatial pursuit of that sub-metric is closed
unless a later benchmark demonstrates that it matters. This is not a claim
of fully converged K3 or a change to its criteria. See
`doc/reconstructed_fault/benchmarking/stage_K3_ih_feedback_disposition.md`,
`stage_K3_normal512.md`, `ih-feedback-impact.json`, and `normal512-errors.csv`.
Entries below preserve the earlier chronology, not current blockers.

**Latest coupled fault-resolution result:** `spatial0375_n128_f32/` and
`spatial0375_n256_f32/` verify the actual coupled normalization improvement
with freshly initialized 32-element faults. Max full-profile defect <4.45e-9;
actual normalization passes at all states. The cases nonetheless stop on
endpoint I_h homogeneity at 3/2.625 s, respectively, and remain explicitly
nonpassing as complete K3 cases. See `stage_K3_coupled_fault_resolution.md`,
`coupled-fault-resolution.csv`/`.json`/`.png` and per-case identity audits.
No further level or separate I_h representation was introduced. The older
fault16 representation limitation below is resolved for these coupled cases;
the homogeneity limitation is retained for review.

**Current review conclusion (2): coarse surface I_h representation limits the
common-step spatial test.** `spatial0375_n128/` and `spatial0375_n256/` preserve
the matched .375-s cases, both guard-failed at 1.125 s. Full-profile identity
defects, not excessive tails, dominate the normalization failure. See
`doc/reconstructed_fault/benchmarking/stage_K3_common_timestep_spatial.md`,
`common0375-identity.png`, and each case's `identity_audit*`/`projection_audit*`.
The independent projection reproduction and offline scalar-resolution test
identify the limiting component; neither is a production fix. No 512-level
case or further temporal trajectory was launched. All preceding evidence is
retained; the earlier reference-only and preparation dispositions below are
historical and superseded by this review point.

**Reference-only support diagnostic completed:** `timestep-support/` preserves
all 45 states of the .5/.25/.125-s sequences through 3 s, full phi/H snapshots,
signed instantaneous/history tails and offline required widths. All sequences
pass 1e-4 normalization at unchanged support, but H/phi/I_h feedback vanishes
at .125 s under the existing maximum rule. See
`doc/reconstructed_fault/benchmarking/stage_K3_timestep_support_diagnostic.md`.
No ASPECT invocation followed the refined support failure. A common-dt spatial
pair is proposed for review, not prepared or run. No support/criterion change.

**First normal refinement stops for review:** `normal256/` is the one authorized
32x256 run with unchanged tangential/fault discretization. It took 87.979 s and
909 MiB; phase/mechanical/lifecycle gates pass, but final supported normalization
is 1.526976e-4 > 1e-4. The exact-time independent reference also exceeds the
limit (1.116639e-4). `normal256-comparison.json` records `complete_smoke=false`;
the final saved mechanical state is diagnostic, not benchmark-accepted.
See `stage_K3_normal_refinement.md`, `normal256-comparison.png`, and the full
`comparison_phase_*.csv` / `comparison_H_*.csv` in both case directories.
No retry or further level is authorized. The original smoke was not rerun.

**Approved corrected invocation passes:** `smoke/` and its log/resource JSON
now hold initialization plus two real steps, completed in 43.412 s on one rank.
The first failed attempt is preserved under `attempt1-fingerprint-failure/`,
with its separate failed plugin/header snapshots. See
`doc/reconstructed_fault/benchmarking/stage_K3_smoke_result.md`. Do not rerun
or expand the campaign automatically.

**Report/review before running `smoke.prm`.** The .009 and .0045 m/s ramps fail
normalization even though h omission passes. The authorized .00225 m/s candidate
passes independent and conditional reference checks. The overlay now selects
that peak; full I_h, support, initial loading and both 1e-4 criteria are unchanged.

The scientific cycle, results, commands and limitations are recorded in
`doc/reconstructed_fault/benchmarking/stage_K3_bounded_adjustment.md` and the
historical `stage_K3_reference_preflight.md`.
`reference.py` reads the fully resolved validated K1 `parameters.prm`, including
the configured activation threshold .1, and initializes histories once.
All new profile/report directories are cheap reference data, not new ASPECT
output. `--ramp-peak .00225` selects the passing candidate; the default .009
preserves the original diagnostic. `--initial-data` selects a separately
labeled conditional calculation initialized only from saved step-zero K1 data.
`assess_adjustment.py` measures both predeclared budgets and signal/noise;
`test_reference.py` checks the bounded decision and saved results.

`Postprocess/Uniform shear pilot/Evolving profile = true` selects the new
benchmark-only mode. It removes the benchmark phi freeze and H-frozen assertion;
`Material model/Phase field fault/Evolve phase field = true` still retains its
existing meaning. Defaults preserve the fixed K1/K2 path. The smoke overlay
targets a separate `evolving/build/libuniform_shear.so`, so building it later
need not replace the accepted benchmark binary. The built plugin's
noncommitting residual instrumentation now passes normal and exceptional
restoration checks in the approved run.
Full CPDI weight/gradient export remains a fallback. `smoke-reference/` uses
the exact accepted time/dt/U without resetting independent histories, and
`smoke-comparison.json` records the primary comparison and unchanged gates.
