# Separate-global-accumulator correction and bounded replays

The user approved the correction in `stage_K2_residual_consistency_audit.md`.
`pre-correction.patch` preserves the preceding tracked working tree; prior
polyline/audit evidence is retained in its original directories.

Only arithmetic accumulation changes. The cell assembler puts +BV into the
unknown-dependent RHS and -beta*tau_old + 2*kappa*history*S into a separate
frozen-load RHS, with the stress contractions integrated over the same QPs.
Both vectors pass independently through homogeneous constraints and MPI
compression, then are added globally. No equation, initialization/history
semantics, surface quadrature, support, full I_h, V_min, tolerance or budget
changes. This path is common to each Newton base and trial.

The opt-in audit now instruments the first linearization. `unknowns` includes
BV, and `frozen` excludes it. In addition to the actually represented bulk/V
Newton increment, an uncommitted positive V probe of
`1e-12*(1+0.1*vertex_index)` verifies a nonzero B action and bitwise unchanged
frozen loads. The diagnostic affine checks use 1e-10 of the compared action
scale plus 32 machine epsilons of the independently assembled frozen-load
norm and 32 times the existing absolute-row-sum precision bound applied to
the represented direction. The latter avoids a relative-only check on a
nearly cancelling continuity action. These are regression checks, not
nonlinear acceptance parameters.
Normal criteria and positive convergence guards remain unchanged.

## Verification gates and prospective resources

1. Debug/Release build and relevant plugins, all with `-j4`.
2. One-/two-rank represented-increment/nonzero-V consistency, unchanged
   positive Stage-I convergence, frozen-stress weak loads, coupled actions,
   temperature/state lifecycle, changed loading, exhaustion and rollback.
   Restart create/resume run sequentially against a fresh reference.
3. Only after these pass, short homogeneous K1 through 1 s, then unchanged
   K2-64 and K2-128 through 1 s. No larger campaign or true-pressure branch.

The prepared `../polyline-quadrature/k1_short.prm`, `k2_64.prm`, `k2_128.prm`
retain their physical settings. The wrappers here change output paths only.
Before execution, estimate Release one-rank costs of 100--400 s / 0.5--1 GiB
for short K1, 300--1200 s / 1--2 GiB for K2-64, and 1500--5000 s / 4--6 GiB
for K2-128. These conservatively include the newly integrated surface rule;
the old point-rule 128 run through 2 s took 2413 s / 3 GiB. Reassess from the
completed smaller replays before launching 128. Runner wall caps are 1800,
2400 and 7200 s respectively, not changed solver iteration budgets.

After the smaller replays: K1 took 48.96 s / 478764 KiB; K2-64 took
271.05 s / 1329992 KiB. Both reached 1 s with positive convergence checks.
The K2-64 measured support and normalization allowances pass. Before launching
128, revise its expected one-rank Release cost to 1200--2000 s (20--33 min)
and 4--6 GiB, with the original 7200 s runner cap unchanged. This is still
only the approved through-1-s comparison, not a longer convergence campaign.

Final result: 128 completes in 1651.82 s / 4494064 KiB (27.5 min / 4.29 GiB).
All three replays satisfy their final nonlinear criteria and measured support
and normalization allowances. The coarse K1 initial raw point-stress error
retains its documented same-mesh baseline limitation. The K2 endpoint anomaly
mismatch at 1 s is reduced by about 64x/52x at left/right; whole-fault RMS is
about halved, but the new initial central discrepancy is larger and raw bulk
stress differences remain. See the complete numerical tables and caveats in
`doc/reconstructed_fault/benchmarking/stage_K2_global_accumulator_review.md`.
No longer trajectory, temporal campaign, true-pressure branch or further
production correction was run.

The endpoint report is reproducible with `summarize_replays.py`; it reads
the old saved `../output` and `../refinement/space128` measurements without
altering them. New actual weak loads/M come from `surface_weak` exports.
`compare_cases.py` supplies the native-FE raw-stress comparison. All outputs
are same-support comparisons, not full-profile references or scalar solves
at independent fault vertices. The JSON reports retain realized histories
and differences from the initial error snapshots separately.

The K1/K2 fixed-profile omitted-fraction allowance is the separately approved
1e-4 for these fixture families, replacing the original 1e-6 target. Actual
slip normalization remains a separate 1e-4 requirement at measured columns
and accepted times. Same-support comparison does not establish full-profile
equivalence. Use exported `surface_weak` records for actual pre-publication
traction and balance; published particle histories are separate diagnostics.
