# Four normal steps versus eight half-steps: precommit stress experiment

**Current direct-input workflow:** see `normal_stress_AB_README.md` and
`normal_stress_experiment_A.prm` / `normal_stress_experiment_B.prm`. They use the
runtime restart hook and need no schedule generation or archive editing. The
staging instructions below are the retained earlier workflow, not prerequisites
for the new files. Do not combine a pre-edited B checkpoint with the new hook.

This package **prepares but does not launch** the requested experiment. Both
branches start from the original accepted-step-5612 checkpoint at
**5310111071.5634108 s**. A determines four actual ordinary timestep lengths;
B is generated only after A finishes successfully. No extra trajectory,
initialization, nonlinear control solve, or tolerance change is included.

## What is measured

The current PhaseFieldFault implementation omits objective rotation, consistently
with its specification. There is no rotation term to export or reconstruct.
At each production coupling QP, using the **working FE history consumed by
mechanics**, it evaluates

```
beta      = exp(-dt*G/eta)
kappa     = -eta*expm1(-dt*G/eta)
inherited = beta * incoming_working_FE_stress
strain    = 2*kappa*strain_rate
slip      = -2*kappa*(history_localization + chi*V)*S
update    = strain + slip
mechanics = inherited + update
d(term)   = -term : (n tensor n)
```

These use the production constitutive coefficients, not a second material law.
The original mechanics arithmetic is unchanged. `history_localization` is the
existing cohesive/profile correction (zero in this frozen mature fixture).
The direct slip term has zero normal contraction for the straight fault because
`S:N=0`; its indirect effect through the bulk solution is **not** zero.

The accepted linearization is captured before particle history publication.
Rejected trials never become output. The postprocessor verifies convergence and
exact agreement with accepted V before writing. Incoming particle values are
captured at the same stage; newly committed particle stress is not substituted.

New raw columns include the full inherited/strain/slip/update tensors,
`d_history`, `d_strain`, `d_slip`, `d_update`, decomposition closure,
`incoming_FE_*`, `particle_interp_*`, and `ref_x/ref_y`. The latter particle
values come from the configured particle interpolator at the **same points**;
they are distinguished from the constrained FE history actually used. Original
weights, cell ID, rank, QP index, fault segment/xi, signed r, phase, chi and I_h
remain present. All stresses are Pa.

Every accepted step produces:

- `normal_qp_STEP_rankR.csv`: unchanged native production quadrature in
  70–71 and 79–80 km. These are the mechanically weighted samples.
- `normal_incoming_particles_STEP_rankR.csv`: incoming particle tensors,
  stable IDs, positions, cell and owner rank for the interpolation stencils.
  Ghost copies are explicitly labeled; do not sum them as unique parents.
- `normal_line_STEP_rankR.csv`: native FE evaluation on the physical fault
  centerline in those windows, at approximately 2-m spacing **and at every
  crossed cell boundary**. Both one-sided traces survive, with cell ID and
  reference coordinates. These are not interpolated CSV values. Their weights
  are deliberately zero: they diagnose representation, not a new weak rule.
  This optional diagnostic supports the current axis-aligned Box cells only.
- Existing `normal_profile`, `normal_summary`, `normal_totals` and
  `normal_checks` files: native weak components, consistent Q1 projections,
  accepted state/slip, clock, history checksums and closure checks.

The line's QP index is its local evaluation-point index, not the production
Gauss-point index; `sample_kind` explicitly distinguishes the two files.

## Existing-data clue, not a causal conclusion

At saved step 5617, the 70–71 km center strip (`|r|<=1 m`) has 27 samples,
all on rank 12, with local bulk spacing 24.4140625 m and fault spacing about
99.964 m. QP indices **1, 4 and 7** carry the strong negative band
(roughly -2.57 to -3.11 MPa); the others are approximately -0.05 to -0.84 MPa.
Their r ranges overlap. This implicates the sampled within-cell structure,
not an MPI interface in this window, but does not yet identify inherited versus
current stress. The new decomposition and exact-centerline traces resolve that
remaining distinction. Existing plots are under
`output-normal-diagnostic/stress_experiment_plots/`.

## Build and preserve the server restart fix

Rebuild ASPECT and **all loaded plugins** together; response/diagnostic layouts
have changed. No checkpoint/state serialization layout has changed.

```bash
cmake --build /path/to/aspect-build --target aspect -j4
cmake -S /path/to/aspect-source/benchmarks/reconstructed_fault/bp5 \
      -B /path/to/bp5-diagnostic-build -DAspect_DIR=/path/to/aspect-build
cmake --build /path/to/bp5-diagnostic-build \
      --target bp5_steady_initialization bp5_normal_stress_diagnostic -j4
```

Merge the diagnostic edits rather than overwriting the server material file:
retain the separately accepted **restart-only** normalization comparison allowance
used for the previous replay. The local source still has the original strict
restart check. Do not loosen the production 1e-10 quadrature/tail tolerances.

## Stage and execute A

Use the actual flat/resolved original parameters, original job's fixture files,
and checkpoint `restart/01`, **not** the diagnostic's final checkpoint. These
arguments must remain identical when preparing B. The destination must be new.
Use the original modules, runtime environment and MPI rank count (32 for the
supplied server checkpoint). Do not source a launcher that rewrites the input.

```bash
python3 /path/to/bp5/prepare_normal_stress_experiment.py A \
  --checkpoint /original/output/restart/01 \
  --input /original/output/parameters.prm \
  --job /original/job \
  --destination /new/experiment-A \
  --bp5-library /path/to/bp5-diagnostic-build/libbp5_steady_initialization.release.so \
  --diagnostic-library /path/to/bp5-diagnostic-build/libbp5_normal_stress_diagnostic.release.so

# Inside the original-size allocation; this invokes ASPECT exactly once.
python3 /path/to/bp5/run_normal_stress_experiment.py \
  /new/experiment-A /path/to/aspect-build/aspect-release ibrun
# For local MPI, replace 'ibrun' with 'mpirun -np 32'.
```

Staging copies complete checkpoint files, scientific fixtures and libraries,
records SHA256 hashes, and retains `production_input.prm` unchanged. The ordinary
four steps remain adaptive. The runner stops once, without retry, on failure or
incomplete output. It has a 900-s hard cap; the existing diagnostic has a 600-s
accepted-state wall stop. A wall-limited branch is incomplete, not a comparison.
The runnable input embeds the original text literally before overrides: this
avoids ASPECT's regex `include` expansion interpreting `$0` in documentation
comments of a fully resolved `parameters.prm`. No physical setting is removed.

## Stage and execute B only after A passes

```bash
python3 /path/to/bp5/prepare_normal_stress_experiment.py B \
  --checkpoint /original/output/restart/01 \
  --input /original/output/parameters.prm \
  --job /original/job \
  --destination /new/experiment-B \
  --bp5-library /path/to/bp5-diagnostic-build/libbp5_steady_initialization.release.so \
  --diagnostic-library /path/to/bp5-diagnostic-build/libbp5_normal_stress_diagnostic.release.so \
  --a-output /new/experiment-A/output-normal-diagnostic

python3 /path/to/bp5/run_normal_stress_experiment.py \
  /new/experiment-B /path/to/aspect-build/aspect-release ibrun
```

ASPECT checkpoints already contain the pending next time/dt. On **B's disposable
copy only**, preparation changes precisely those two double values to the first
half-step; it retains the previous dt, step number and all histories. It validates
the known archive framing/unique clock and proves every other decompressed byte
unchanged. Unknown archive layouts fail closed. Original checkpoint files are
never edited. The rest of B's clock is a `function` timestep **cap**, added to all
original timestep safety models. If a safety model selects a smaller step, an
exact-clock check stops B **before mechanics**. It does not force a larger dt or
silently take a ninth step. The runner additionally checks identical binary,
plugins, launcher and relevant environment between branches.

### Full-precision clock caveat

At 5.31e9 s, one double-precision absolute-time ULP is
9.5367431640625e-7 s. Two equal floating-point half increments need not land on
A's represented endpoint. Therefore the first substep is exactly `A_dt/2`; the
second is `A_endpoint - B_midpoint`. Its departure from exact half is required
to be at most one absolute-time ULP and is recorded explicitly in
`experiment.json`; `expected_clock.txt` stores 17-digit timestamps/dt. On the
existing saved four-step clock, the largest such adjustment is about 6.27e-7 s
(0.045% of a half-step). These are **rounding-adjusted half steps**, not a claim
of identical decimal halves. All four common accepted endpoints are bitwise
equal. Report the recorded correction alongside temporal differences.

## Analyze without additional mechanics

```bash
python3 /path/to/bp5/analyze_normal_stress_experiment.py \
  /new/experiment-A/output-normal-diagnostic \
  --other /new/experiment-B/output-normal-diagnostic
python3 /path/to/bp5/analyze_normal_stress_experiment.py \
  /new/experiment-B/output-normal-diagnostic
```

Use `--step N` for an earlier accepted state. Band plots show inherited/update
against actual r, local QP index, reference-cell coordinates and fault xi; native
centerline plots retain both traces without smoothing. Transfer plots distinguish
configured particle reconstruction from mechanics' working FE tensor. Matched
profiles compare all four common times, including final velocity, committed
state, slip, pressure and total deviatoric/normal traction, in full and windowed
views. The normal plot removes only the explicitly exported background.

Inherited/current-update terms span **different time intervals** in A and B;
compare their sum for endpoint accuracy. Do not interpret the final half-step's
smaller update alone as improvement. Mechanics uses incoming state; committed
Theta is labeled and is not used to reconstruct a friction residual.

## Verification and changed source

The Release core and both plugins build with `-j4`. Cheap Python tests cover
tensor/mass consistency, checkpoint staging, plotting conventions and the exact
clock-only edit. No 12-solve BP5 trajectory has been executed in this preparation.
The new native-line/particle exports still require this server experiment for
end-to-end validation; compilation and point-response tests are not a substitute.

Preparation checks (2026-09-22):

```
python3 benchmarks/reconstructed_fault/bp5/test_normal_stress_diagnostic.py
# Four tests passed; includes synthetic A/B staging and headerless restart logs.
ctest --test-dir build-pf-cpdi/tests --output-on-failure \
      -R '^phase_field_fault_surface_dynamic_pressure$' -j1
# 1/1 passed, 34.95 s. Point response/derivatives, not native-line export coverage.
aspect-release --validate normal_stress_diagnostic_restart.prm
# Actual staged A input with the supplied restart parameters: valid.
```

New production edits in this task:

- `include/aspect/reconstructed_fault/surface_system.h`
- `source/reconstructed_fault/surface_system.cc`

Also required from the preceding stress-decomposition diagnostic:

- `include/aspect/material_model/phase_field_fault.h`
- `source/material_model/phase_field_fault.cc`

Benchmark code is `normal_stress_diagnostic.cc`, staging/preparation/runner/
analysis scripts and their tests in this directory. The BP5 physical initializer
and BP3 model are not modified by this experiment.
