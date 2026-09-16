# K5: isolate the growing cohesive force (bounded diagnostic)

Subsequent task: the user authorized relaxing the benchmark's exact-timestep
matching guard. See `stage_K5_cohesion_mechanism.md` for that separate adaptive
replay. The guarded partial-run results and stopping point below are retained
unchanged as the record of this experiment.

## Mathematical decision before execution

The authoritative cohesive update, evaluated at a fault coordinate, is

`C_eval,k = (kappa_Gamma,k V_k + beta_Gamma,k I_(k-1) C_hat_(k-1))/I_k`.

Here `beta=exp(-dt G/eta)` and `kappa=-eta expm1(-dt G/eta)`.
The committed Q1 history is the particle/domain projection of this evaluated
quantity, not an independent exact recurrence at each vertex.
For frozen phase and I_h the profile-history crack-strain correction is zero,
but `C_eval,k=beta C_hat_(k-1)+kappa_Gamma V_k/I_h` still accumulates slip.
In the nearly elastic BP3 regime this is approximately `G_Gamma Delta slip/I_h`.
The bulk stress is independently `beta_b tau_old + 2 kappa_b (epsdot-chi V S)`.
With `R=q-C-mu sigma_n-damping V`, there is no automatic positive C term
in q that cancels this resistance. At fixed bulk velocity its slip derivative
has the same resisting sign. The background traction includes the initial
cohesive offset once, and is not subsequently updated.

More precisely, before spatial projection,
`C_eval,k-C_hat_(k-1)=kappa_Gamma (V_k/I_h-C_hat_(k-1)/eta_Gamma)`.
Growth requires `V_k>I_h C_hat_(k-1)/eta_Gamma`, rather than merely a frozen
phase. This threshold is roughly 1e-16 m/s for MPa-scale C here, far below
plate-scale rates. At an exactly bound vertex the local source can stop, but
adjacent moving quadrature points and the consistent projection still matter.
There is no assertion of monotone nodal growth for every rate and mesh.
The general update accumulates `I_h C`; an evolving profile could offset that
accumulation through increasing I_h. Frozen phase removes that possible
softening route. The surviving nearly elastic slip-dependent resistance is
an additional feature of this modified-BP3 model, not an aging-state update.

The saved original 50-m run confirms early growth: at 39.95 km, the weighted
cohesive force rises from 0.500839 MPa at initialization to 0.645463 MPa at
step 6, while q changes only from 27.046990 to 27.041161 MPa. The rate falls
to 0.810109 Vp. At step 8 C is 0.939616 MPa and the rate is 0.489379 Vp.
The numerical initialization interval differs from the first real timestep;
the small first-step C decrease is not evidence against subsequent growth.

## Single counterfactual

Keep timestep zero unchanged and freeze the *evaluated initial resistance*

`C_star(s)=beta_0 C_hat_0(s)+kappa_Gamma,0 V_0,accepted(s)/I_0(s)`.

This retains the actual initial mechanical force balance, rather than replacing
it by the different retained C0. Snapshot Q1 inputs once and evaluate their
ratio at each coordinate; do not project C_star back into Q1.
For real steps only, use C_star in R and remove kappa_Gamma/I_h from minus dR/dV.
The original nodal Theta update, bulk Maxwell history, fixed background,
support, I_h, phase, pressure, mesh, timestep guards and tolerances remain.
C/H are still calculated and published as **shadow histories**, not reset:
in this frozen-profile experiment they cannot feed mechanics through phase
evolution or the identically zero profile-history term. Thus this isolates a
force contribution; it is not a proposed energy-consistent alternative law.
The uniform BP3 surface Maxwell material constants make the snapshot fixed.

In this fixture G=32.03812032 GPa, eta=1e26 Pa s, so the relaxation time is
about 98.9 million years. Near 39.95 km I_h=12960.248927 m: the elastic
cohesive stiffness G/I_h is about 2.47 MPa per metre of slip. The retained
initial nodal C is about 0.491669 MPa, whereas the initial weak evaluated
resistance is 0.500839 MPa. These are intentionally distinguished.

For the frozen profile the fixed-bulk tangent is

`-dR/dV = 2 kappa_b chi (S:S) + kappa_Gamma/I_h + sigma_n dmu/dV + damping`.

The diagnostic removes only its second term. Fixed background tractions have
zero derivative. G still carries the original strain and true-normal-pressure
terms. An eventual mechanical increase of q is a response to force balance,
not an algebraically prescribed cancellation of C. For example, with identical
bulk/surface beta and kappa, subtracting the two update equations gives slip
terms `-2 kappa chi V(S:S) - kappa V/I_h`, not opposite-sign terms.

The diagnostic is opt-in and fresh-start-only. It includes production weak
residual finite differences for K and G before the first real solve, and checks
that the evaluated cohesive vector is unchanged by trial V. Compare against
the original nodal-Theta 50-m trajectory, not the functional-Theta experiment.
Run initialization through the same step-13 comparison time, using the saved
accepted timestep caps and retaining controller guards. No automatic retry,
retuning, or continuation beyond that comparison is authorized.

Run: `python3 benchmarks/reconstructed_fault/bp3/run_frozen_cohesion.py`.
Evidence: `benchmarks/reconstructed_fault/bp3/frozen-cohesion-50-local4/`.
Both Release executable and plugin built successfully with `-j4`.
Standalone check:
`g++ -std=c++17 -O2 benchmarks/reconstructed_fault/bp3/cohesion_snapshot_test.cc -o /tmp/bp3-cohesion-snapshot-test`
then `/tmp/bp3-cohesion-snapshot-test /tmp/bp3-cohesion-snapshot-input.txt`:
PASS (initial evaluated resistance, endpoint values, nonlinear ratio, rejected
geometry mismatch). Production finite differences at step 1: K relative
errors 3.60677494e-7 and 3.50877755e-8 for increments 1e-15 and 1e-16 m/s;
G pressure relative error 9.30134409e-13. Trial-V cohesive vectors were
bitwise equal. These checks use the full four-rank production surface rule.

## Result

**The bounded replay demonstrates that growing cohesion materially drives
the early slowdown and junction concentration. It does not establish the
late bound-contact outcome: the unchanged timestep guard stopped it.**

One fresh four-rank replay ran for 1159.34 s (19.32 min). States 0--9 were
fully exported on the original clock, through 15.30625726 yr. Step 10 at
29.24190894 yr also reached genuine mechanical convergence (relative bulk
2.05587e-13, surface 2.31805e-12), but before ordinary accepted-state output
the controller selected a smaller next timestep:

| Next step | Saved dt (s) | Selected dt (s) |
|---|---:|---:|
| 11 | 839971627.82783842 | 352165721.66361439 |

The replay cap correctly aborted rather than overriding the controller. No
retry, smaller-step continuation, or tolerance change was made. Step 10 has
some mechanical diagnostics, but no complete ordinary accepted-state export;
it is **not** used as an accepted history/slip endpoint or restart state here.
The requested step-13 endpoint, 70.73340112 yr, was not reached.

### Early force budget and matched comparison

Below are actual production weak loads divided by the corresponding row
weight, at the last free vertex (39.95 km). They are neither particle-centre
estimates nor normal-column bulk averages. A = original cohesion;
C* = frozen initial resistance. Values in MPa, except V/Vp.

| Time (yr) | Case | q | C force | friction | V/Vp |
|---:|---|---:|---:|---:|---:|
| 0 | both | 27.046990 | 0.500839 | 26.546151 | 1.000029 |
| 0.594802 | A | 27.044338 | 0.536431 | 26.507907 | 0.958142 |
| 0.594802 | C* | 27.043902 | 0.500839 | 26.543063 | 0.996684 |
| 2.190142 | A | 27.041161 | 0.645463 | 26.395698 | 0.810109 |
| 2.190142 | C* | 27.033106 | 0.500839 | 26.532267 | 0.982388 |
| 8.010105 | A | 27.071489 | 0.939616 | 26.131873 | 0.489379 |
| 8.010105 | C* | 26.986464 | 0.500839 | 26.485625 | 0.918950 |
| 15.306257 | A | 27.152692 | 1.200502 | 25.952190 | 0.306602 |
| 15.306257 | C* | 26.929983 | 0.500839 | 26.429144 | 0.842091 |

Damping is only about 0.002--0.005 Pa at this node in these states, retained
in the CSV and residual checks. At the final matched time the original
cohesive increase is 0.699663 MPa, while q has risen only 0.105702 MPa:
the growth does not cancel. The remaining balance comes mainly from reduced
friction as the velocity and state evolve. Free-node residuals in both runs
are roundoff-small; that is not the scale used to judge the force change.

The weak C* mean changes by only 0.007841 Pa between initial and final
matched states. C* is fixed at surface coordinates, while particle/domain
weights move. Do not confuse it with the diagnostic's normally committed
shadow C: at 39.95 km the latter reaches 1.565314 MPa, not 0.500839 MPa.

| At 15.306257 yr | Original | Frozen C* |
|---|---:|---:|
| V/Vp at 39.95 km | 0.306602 | 0.842091 |
| V/Vp at 39.90 km | 0.511783 | 0.889096 |
| V/Vp at interior control 25 km | 0.398464 | 0.697044 |
| Slip at 39.95 km (m) | 0.2324583 | 0.4337108 |
| Prescribed-node slip at 40 km (m) | 0.4830287 | 0.4830287 |
| Last-element slip gradient (m/m) | 0.005011409 | 0.000986359 |
| Penultimate-element slip gradient (m/m) | -0.001463279 | -0.000291607 |
| Lower-active free nodes | 0 | 0 |

The last-element gradient is reduced by **80.32%**. The first saved state
below 0.9 Vp shifts from 2.19 to 15.31 yr; the diagnostic has not fallen
below 0.5 Vp at the comparison endpoint. This is a slowing of the slowdown,
not its complete elimination. The independently documented original contact
at step 12 (68.62 yr) has no matched diagnostic counterpart in this replay.

### Pressure and actual constitutive normal traction

Raw bulk pressure below is sampled at the same exported Q1 VTU vertices in
39--40.5 km, within 1500 m normal distance of the fault. It is not substituted
for the production particle/domain constitutive traction.

| Time (yr) | Case | min delta p (MPa) | max delta p (MPa) | Peak-to-peak (MPa) |
|---:|---|---:|---:|---:|
| 2.190142 | A | -0.304460 | 0.294127 | 0.598587 |
| 2.190142 | C* | -0.025188 | 0.029472 | 0.054659 |
| 8.010105 | A | -3.124276 | 2.969040 | 6.093316 |
| 8.010105 | C* | -0.425055 | 0.424320 | 0.849375 |
| 15.306257 | A | -8.301141 | 7.846876 | 16.148017 |
| 15.306257 | C* | -1.533113 | 1.502738 | 3.035851 |

The final pressure range is reduced by **81.20%**. The physical junction and
its open/free-to-prescribed topology have not moved.

The raw production samples whose Q1 support spans the 40-km interface have
the following exact extrema (MPa) at the final matched state:

| Case | min sigma_n | its xd (km) | its delta p | its -delta tau:N | max sigma_n |
|---|---:|---:|---:|---:|---:|
| A | 41.587038 | 39.982042 | -7.783819 | -0.629143 | 58.162923 |
| C* | 48.378603 | 39.997074 | -1.428746 | -0.192651 | 51.573679 |

Each minimum obeys `sigma_n=50 MPa+delta p-delta tau:N`. These are mixed-support
samples on segment 795, not a purely prescribed-row or nodal traction.
The range decreases from 16.575885 to 3.195077 MPa. For purely free support,
the original minimum is 42.813039 MPa near 39.949620 km; after freezing C*,
the global purely-free minimum shifts to the 15--18-km transition, 48.058103
MPa at 17.746698 km. Selected extrema are not used to infer unexported
pressure extrema or whole-support averages.

By contrast, the deep prescribed tip near 115.47 km retains almost the same
global normal-stress range: [33.834096,59.732322] MPa originally versus
[33.846080,59.745408] MPa in the diagnostic. Thus the junction improvement
does **not** mean all finite-domain stress concentrations have disappeared.

### Verification and recoverability

- Release executable/plugin builds with -j4 and the standalone snapshot test
  passed. Production K/G derivative checks are listed above.
- All 84 returned linear directions (including step 10) passed their fresh
  residual check; worst fresh/target ratio 0.9975724. All fully exported states
  passed the actual nonlinear solve and ordinary BP3 history checks.
- Offline analysis passed for all ten fully exported states: exact same mesh
  (42880 cells), same accepted clock, 1236-node fixed geometry, fixed I_h and
  background tractions, exact deep Vp, nodal Theta aging and accumulated-slip
  updates, and reconstruction of q-C-friction-damping from production loads.
- Independent raw-point friction reconstruction using **preceding nodal Theta**
  agreed to maximum absolute mu error 2.11e-15. The functional-Theta diagnostic
  was not enabled.
- Initial Theta, C history, I_h and background were identical. Maximum initial
  V difference was 8.27e-25 m/s, q difference 1.12e-8 Pa, and projected evaluated
  C difference 2.91e-10 Pa. No initial rebalance or history reset was introduced.
- Run log, source/binary/input hashes, command, environment, execution status,
  snapshot and all outputs are under `frozen-cohesion-50-local4/`. This guarded
  partial run remains preserved, not labelled a completed step-13 trajectory.
- Analysis command: `python3 benchmarks/reconstructed_fault/bp3/analyze_frozen_cohesion.py`.
  `comparison/summary.json`, `series.csv`, `force_budgets.csv`,
  `junction_profiles.csv`, `raw_support_extrema.csv`, and `comparison.png`
  contain the numerical evidence. No broad test suite or second replay ran.

Task edits: the opt-in point-response branch in `phase_field_fault.cc`, private
`fault_cohesion_diagnostic.h`, BP3 snapshot/derivative wiring in `bp3.cc` and
`cohesion_diagnostic.h`, runner, analyzer, standalone snapshot test, and this
report. Default production equations are unchanged. The opt-in diagnostic is
not a supported restartable physical model; existing unrelated changes remain.

### Decision

An independent cohesive hardening contribution is established, and removing
its growth substantially reduces early junction slip deficit and stress
concentration without modifying nodal Theta, transfer, or pressure treatment.
This does not disprove the separately measured Theta interpolation effect,
nor prove that freezing C would prevent later contact.

**Recommended next action: review the physical role of frozen-phase cohesive
hardening in the modified BP3 model before promoting any constitutive change.**
The counterfactual keeps normally evolving C/H shadow histories while removing
their surface force, so it is mechanism evidence, not an energy-consistent
production law. Continuing it to the original late endpoint would additionally
require a newly approved guard-respecting timestep comparison; none was run.
