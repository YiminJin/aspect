# P5c selection, 22 October 2026

Sequential gates: corrected material baseline, exact-intact native activation,
nonuniform tensor histories on 1/4 ranks with rejection and same-rank restart;
only then the freshly initialized finite specimen. Original P4/P5a/P5b packages
remain immutable. See evidence/p5c_native/parameters.csv and baseline-correction.json.

Use P4 fracture.prm physical data: 1.5 x 1 m finite body, 96 x 128 Cartesian cells,
Gauss8 mechanics/source3, continuum peak-.6 seed ending at (1,.5) m, m=4/Kf=8 Pa,
released lateral horizontal motion with lateral vertical restraint. Compare fixed
and radial tensor modes only. Unlimited DGQ2, native FE source complement,
physical irreversibility, current tolerances and local relaxed-stability gate
(cap 1024 critical masters) remain. Fresh zero histories, same sampled seed.

Bounded interval: initial rest through .05 s, original symmetric shear rate
.018/s through .140 s, then a short prescribed-displacement hold to .145 s.
This deliberately ends the loading earlier than P4's .175 s endpoint; the
original loading path is identical up to .140 s. It is a first-growth comparison,
not a replay of P4's rapid late event. No direction bias is imposed.

Start dt=.00125 s. Save an accepted checkpoint at .120 s for each freshly
initialized trajectory, then repeat only .120-.145 s at dt=.000625 s from
that same-mode accepted state. Report this common-prefix limitation, onset
brackets and growth-relative temporal defects (5% interpretation target;
not a relaxed native acceptance criterion). If onset precedes .120 s, disclose
that it lies outside the halved interval rather than silently moving the window.

Prior P4 timings suggest 15-35 minutes per fresh trajectory and 15-35 minutes
per active half-step repeat, roughly 1-2.5 hours total on four local ranks.
Measure actual costs; run sequentially, bounded timeout 2700 s per invocation.
Stop on unchanged solver/stability/capacity checks and report the accepted common
window. Do not enlarge the stability space or relax tolerances.

Measure initial representation separately; common-time phase growth, connected
.1/.2/.4/.6 contours, stress and driving, angular flow only where active,
reactions, physical work/remainders and local stability. These are finite-body
results. P4 failed temporal/spatial targets, boundary sensitivity and
strict_reference_match_passed=false remain. No claim of propagation accuracy.

Checkpoint-clock clarification before halved runs: native checkpoint scheduling
at the next step number stores accepted step95 (.11875 s), rather than accepted
step96 (.120 s), for the interval16 setting. Inspect the serialized owner clock;
use that existing same-mode checkpoint and halve .11875-.145 s (42 intervals).
No clock/state bytes are rewritten. This adds one coarse interval to the
predeclared active comparison. The fresh trajectories are unchanged.
