# K5: paired top completion/source experiment

## Decision

The paired top treatment succeeds for prescribed uniform sliding: the top
strain and velocity errors fall to the local interior discretization level,
and the large top normal-stress anomaly collapses. The successful bottom
treatment remains unchanged. Both fresh four-rank cases completed initialization
and the same two real timesteps, with unchanged physics, constraints, solver
tolerances and history timing.

**Do not yet enable this at the free RSF top.** The current endpoint velocity
and its bulk derivative are correct, but the bulk-only extension does not add
the corresponding surface integration measure. The added wedge accounts for
57.0% of the endpoint B virtual work for the specified admissible test, while
187 positive-phase source-only parents have zero surface weight. The existing
G is still the derivative of the existing particle/domain residual; this is
not a claim that G must equal B transpose, or a demonstrated algebraic Jacobian
error. It is an unresolved free-endpoint weak-form qualification. No mature-RSF
replay was launched, and no surface measure was silently changed.

## Experiment and invariants

Cases under `benchmarks/reconstructed_fault/bp3/`:

- `top-source-control-50-local4`: observer-only repeat of the accepted
  `bottom-source-complete-wedge-50-local4`, now including inactive top QPs.
- `top-source-paired-50-local4`: same case plus top completion and source.
- `top-source-comparison`: machine-readable comparisons, figure and test logs.

Both retain the 42,880-cell mesh, 1,236 vertices, 60-degree straight geometry,
ell=400 m, mature frozen phase, Vp=1e-9 m/s everywhere, zero initial perturbation
history, original side velocities and top/bottom perturbation tractions.
Accepted times are 0, 2487214.2056652424 and 4970143.2669182401 s; real timesteps
are 2487214.2056652424 and 2482929.0612529977 s. The artificial initialization
interval remains 4e6 s and is not physical elapsed time.

The top table uses the missing outside-top portion of the same virtual Q1
profile. Its maximum discrepancy against saved physical top FE samples is
5.38458e-15 in phi; the order-8/order-16 integral discrepancy is 2.54659e-11 m.
Fourteen surface profile requests receive a top addition. Contributions enter
the existing consistent-Q1 projection RHS, with no mass change or outside
mechanical domain. The top endpoint Ih changes from 6647.7667264488373 to
12691.224796361214 m. Bottom endpoint Ih stays 12691.226484242241 m.

The in-box tangent wedge uses the last segment at xi=1, physical FE phase,
and constant endpoint Ih/material fields. No additional normal-width cutoff is
applied in the wedge; zero physical phase still gives zero source. Existing
normal-profile admission outside it is unchanged. Assembly and particle
Maxwell strain subtraction share this source map. Slip comes from the current
trial/accepted endpoint field, not a new hardcoded bulk rate.

At every accepted state, all 187 newly included positive-phase bulk QPs agree
with the continued reference source to <=8.078e-28 /s. Bottom chi is bitwise
unchanged. The observer control reproduces the saved control's chi, velocities,
pressure and all three stress components bitwise at their common QPs.

## Mechanical comparison

The all-QP set includes inactive positive-phase points, not just associated
points. The top 0–200 m region contains 351 QPs and 400543.212890625 m² of
quadrature measure. Percent strain errors below use the **same continued
reference crack-strain tensor RMS**, not each case's own clipped source norm.
Velocity errors use the independent stationary-profile sliding primitive.

At the final accepted state:

| Quantity | Bottom-enabled control | Paired top + bottom |
|---|---:|---:|
| Top 0–200 m source RMS error/reference | 43.3139% | 0.00423532% |
| Signed integrated source deficit/reference, same region | 2.62283% | 0.00104796% |
| Top strain-mismatch RMS (1/s) | 1.77497e-13 | 9.26061e-15 |
| Top strain mismatch/reference | 25.5699% | 1.33406% |
| Top velocity RMS error/Vp | 1.64844% | 0.313300% |
| Top raw bulk sigma_n (MPa) | 49.8569035–50.1589862 | 49.9977879–50.0016529 |
| Top surface raw retained-extreme sigma_n (MPa), first 2 km down dip | 49.8318558–50.0987553 | 49.9971332–50.0025861 |
| Top surface weak-mean sigma_n minus 50 MPa (Pa), first 2 km | -43766.100–2017.123 | 4.922–58.075 |
| Top consistent-Q1 sigma_n minus 50 MPa (Pa), first 2 km | -65883.832–2316.367 | 3.957–60.281 |

The signed source deficit is not a norm: missing wedge source and an excessive
clipped-denominator source partly cancel in the control. The RMS measure
exposes that error. Small remaining deficits belong to the unchanged ordinary
support/tail approximation outside the wedge; no renormalization was used.

Raw top bulk tensor ranges at the final state, in kPa:

| Quantity | Control | Paired |
|---|---:|---:|
| p | -93.919–101.910 | -0.4552–0.8854 |
| tau_xx | -92.130–123.098 | -2.4765–2.9579 |
| tau_yy | -123.454–123.049 | -4.4270–5.1005 |
| tau_xy | -117.682–35.766 | -2.4943–3.5737 |

Normal stress is evaluated as 50 MPa+p-tau:N, not inferred from an individual
tensor component. Raw bulk QPs, retained surface extrema, weak means, and
consistent-Q1 surface reconstructions remain separately labeled.

Controls at the final state:

| Region | Control strain/reference | Paired strain/reference | Paired velocity error/Vp |
|---|---:|---:|---:|
| Top 200–1000 m | 3.04945% | 1.36827% | 0.317809% |
| Top 1000–2000 m | 1.38663% | 1.36693% | 0.318911% |
| Bottom 0–200 m | 1.3340937% | 1.3340935% | 0.313427% |
| Interior, down dip 59–61 km | 1.3667338% | 1.3667328% | 0.317260% |

Bottom raw normal extrema move by about 0.0183 Pa; interior extrema by about
0.0105 Pa. These are small global solution responses, not changes to their
source policy. Interior weak normal deviations remain -28.56 to +36.68 Pa.
The top improvement is already present at initialization and the first real
step; their paired top strain RMS values are both 9.07506e-15 /s, versus
1.73503e-13 /s in the control.

## Coupling check: pass for B, unresolved surface measure

At the top wedge, xi=1 selects the exact current endpoint V. Production
assembly reads it through `interpolate_slip_rate`; the absolute residual and
B action use the same endpoint shape. Maxwell-history subtraction also reads
that current field. A separate noncommitting initial-state probe changes only
the explicit top endpoint input by +/-0.125 Vp:

- ||B deltaV|| = 40521334.782813773 in the assembled vector's units;
- centered absolute bulk-residual derivative versus -B: relative 1.70589e-15;
- explicit sparse B versus independent quadrature B: relative 2.50868e-16;
- committed V is unchanged by the probe.

The virtual-work diagnostic uses w=Vp*s*X*(1-X)*Y, X=x/100km, Y=y/100km.
It belongs to bulk Q2 and vanishes on the prescribed lateral boundaries.
For deltaV_top=Vp the **additional** wedge B contribution is

    integral_wedge 2*kappa*chi*Vp*S:epsilon(w) dOmega.

It is 3.05629e-5 W/m at initialization and 1.89714e-5 W/m at the final state,
57.0073% of the corresponding complete endpoint B work at each state. The
absolute and signed contributions coincide for this test, so this is not
cancellation of large opposite terms.

However, source-only parents still have inactive ordinary surface associations.
At each real step 187 of them have positive physical phase. Their full parent
measure is about 228883 m²; their surface quadrature weight is exactly zero.
Some *already admitted* domains cross the endpoint and already contribute via
endpoint-clamped domain quadrature. Those contributions are retained; they do
not turn the source-only parents into admitted surface samples.

The implementation invariant is explicit:

- manager bulk/QP mapping admits the continued wedge;
- `assemble_surface_system()` skips inactive ordinary parents before producing
  residual, K_V and CouplingPoint data;
- both reference and explicit G are built from those CouplingPoints only.

Therefore no new R_Gamma/K_V/G weight accompanies the added source-only
parents. Existing G continues to differentiate the selected residual correctly.
The above 57% is a **B-work fraction**, not a claimed 57% residual or G error.
The current specification uses particle/domain volume for surface weighting,
bulk-QP localization for B, and explicitly does not identify G with B transpose.
A free-endpoint virtual-work closure cannot be inferred from the successful
B finite difference or from prescribed-slip convergence.

**Smallest next action:** review an endpoint-local surface-measure extension
for the continued kinematic test, specifying its traction/resistance weights
and consistent R_Gamma/K_V/G derivatives before changing admission. Do not
merely admit the entire geometric wedge with unit volume weight: unlike the
bulk chi source, the friction residual does not vanish at phi=0. Preserve
parent ownership and open topology, and retain original admitted contributions.
Then test this free-endpoint formulation and only then run the short mature-RSF
replay. This is a discrete-formulation decision, not a solver-tolerance fix.

## Verification, cost and recoverable evidence

- Release core and plugin builds, -j4: passed.
- `mpirun -np 2 build-pf-cpdi/aspect-release --test '[fault_bottom_source],Fault normal-profile*'`:
  37 assertions / 4 cases passed on each rank. Includes top/bottom exact endpoint
  coordinates, unchanged ordinary surface admission, retained admitted mapping,
  and rejection outside the physical box.
- Control: 125.272 s; paired: 129.805 s. Four ranks, fresh start, no automatic retry.
- Each case has six passing fresh-linear checks. Worst fresh/target ratios:
  control 0.830874, paired 0.877410.
- Paired final relative bulk residuals for steps 0/1/2:
  5.28736e-10 / 4.95836e-10 / 8.74313e-10. Surface solver residual is zero because
  every row is prescribed, **not evidence of free-RSF equilibrium**.
- Geometry and clocks match; V, Theta, C and accumulated slip match between
  cases. Mature H remains unchanged for all 385920 stable particle IDs.
  Supplied timestep-zero stress is retained at zero.
- Newly sourced top particle stresses equal actual committed properties
  bitwise. Independent Maxwell formula error: 0 Pa at step 1 and <=4.548e-13 Pa
  at step 2. No diagnostic reevaluates an update using already committed history.
- `git diff --check` and Python syntax checks pass.

The first offline report attempt hit a CSV row-schema error (positive-phase
parent counts are available only at real steps); its log is retained as
`csv-schema-failure.log`. This was a reporting-only fix. A further region
accounting check found the exact top at xd=-7.28e-12 m in the rotated chart;
the final weak tables include it using a 1e-7 m selection allowance. Only the
saved weak data were refreshed (`--weak-only`); neither correction reran ASPECT
or changed a physical/solver tolerance.

Each run's `provenance.json` records binary/source/input hashes. Exact commands
and source-selection safeguards are in
`benchmarks/reconstructed_fault/bp3/top_continuation.md`. No free-RSF,
restart, refinement, or long-trajectory test was run. No commit was made.

Files changed for this task: manager header/source (opt-in top bulk map),
`unit_tests/reconstructed_fault.cc` (geometry checks), BP3 `bp3.cc` and
`uniform_sliding.h` (guarded selection and observations), the two authoritative
design files (bounded experimental scope), and new `run_top_source.py`,
`analyze_top_source.py`, `top_continuation.md`, this report and saved evidence.
Existing unrelated working-tree changes were retained.
