# Mature bulk-spin correction — 5 November 2026

## Selection and preserved baseline

The selected handoff is `CODEX_BULK_SPIN_CORRECTION_HANDOFF.txt`. Work remains in
`/home/ein/repository/aspect-pf-rsf-particle-free`, branch `pf-rsf-particle-free`,
HEAD `cf06525471ce89789f1ac10905dc79f3aac7562a`. HEAD was **not** a clean source
baseline: exterior-inflow implementation, documentation and untracked scientific
fixtures were already present. Entry diff, hashes and source copies preserve
that state. No checkout/reset, commit, production conversion or server submission
was performed. The previously stopped layered campaign stays stopped, including
its incomplete L0 output and its failed evidence.

Evidence lives in
[`bulk_spin/`](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/bulk_spin/).
`entry.json` and `entry.diff` identify the incoming state; source copies are in
`build-particle-free/transported-objective/stage-c/bulk-spin/baseline-source/`.
The canonical retained binary and the matched exterior-stress core/checker pair
are preserved byte-for-byte. A retained executable is not assumed to match HEAD;
the new executable is `stage-c/bulk-spin/aspect-release`, with its own matched
`checks/libtransported_bulk_spin.release.so`. Per-run `command.json` records the
actual executable/checker/input hashes. The old configured ASPECT banner hash
is not used as source provenance.

## Selected law and discrete implementation

This is an explicit constitutive change for mature frictional DG histories.
For $L_{ij}=\partial_jv_i$, the existing source association supplies

$$
 J_f=\frac{h(\phi)\widetilde V}{\widetilde I}(s\otimes n),\qquad
 D_f=\operatorname{sym}J_f,\quad W_b=\operatorname{skw}(L-J_f),
$$
$$
 D_t\tau-W_b\tau+\tau W_b+G\tau/\eta_b=2G(D_3(v)-D_f).
$$

`Constitutive spin = bulk slip subtracted` selects this law; `total material`
remains the default and retains the old meaning. `Stress rotation` continues to
select the integrator/storage path independently. No viscous spin, inferred
bulk velocity, polar-rotation history, spin dissipation or endpoint-spin Newton
term was added. Full 3-D plane-strain deviator and energy contractions remain.

The shared `MatureLocalizationSample` carries active support, fault/segment,
geometry version, xi, signed local frame, h and normalization. Rates and I are
interpolated separately before division. `twice_symmetric(V)` retains the old
mechanical arithmetic; `slip_gradient(V)` supplies its full nonsymmetric dyad.
Thus the old source is **2 sym(Jf)**, not Jf; Z remains auxiliary stress-times-
length memory and is never used as an instantaneous rate. There is no new
endpoint continuation, closest-point clamp, floor, cap or phase threshold.

`prepare_material_history` freezes unprojected angular rates from accepted
velocity, committed V, Q1 phase, normalization and accepted geometry, before the
manager swap. CellId, Gauss order, parent, step and geometry/rate identities
identify the attempt. The rates are Eulerian, are not transported, and are
multiplied by the actual attempted dt. With unchanged DGQ2/SSPRK3/projection,

$$
 \bar\tau=P_h[Q_b A_h(\tau^n)Q_b^T],\quad
 \tau^{n+1}=P_h[\beta\bar\tau+2\kappa(D_3(v^{n+1})-D_f^{n+1})].
$$

The endpoint source still uses endpoint geometry/profile and private Newton V.
Both separated transported tensors, T and M=R+P(B), use the same accepted spin;
residual reconstruction subtracts candidate B as before. The stationary mature
anchor uses the same sampling and stable `QBQᵀ−B` residual increment. Its exact-
zero shortcut tests the selected spin, including zero-total/nonzero-fault spin.
The ordinary simulator controls ordering/acceptance and the coupled solver
controls projection consistency; no alternate driver was introduced.

Initialization takes the accepted parent without rotation. The initial coupled
solve commits converged V with its accepted velocity without advancing physical
stress. First-step sampling uses that committed pair, not the initial .001
fixture predictor. Scratch spin is discarded with the transaction and is never
checkpointed. Moving schema7 includes a versioned constitutive-spin identity;
stationary total-spin schema6 retains its meaning, and stationary bulk-spin
schema8 adds explicit identity. Unsupported particle/cohesive combinations are
rejected at parameter parsing. Their kinematics and constitutive histories have
not been qualified by this mature derivation.

For observer rotation, $W_b^*=QW_bQ^T+\dot Q Q^T$. Since
$\tau:(W_b\tau-\tau W_b)=0$, the continuum energy identity remains
$D_t[\tau:\tau/(4G)]+\tau:\tau/(2\eta_b)=\tau:D_3-\tau:D_f$.
For curved/varying slip, $v=H(\rho)j$ leaves $L-J_f=H\nabla j$; those genuine
regular gradients are retained. This is the chosen small-elastic-strain
bulk-corotational closure, not a finite-strain hyperelastic or sharp-tip result.

## Source and verification map

| Responsibility | Implementation / check |
|---|---|
| Explicit selection/admissibility/identity | `phase_field_fault.cc`, `dg_accepted.cc` |
| Shared source sample and stationary anchor | `dg_objective.cc` |
| Accepted spin and attempt lifetime | `material_history.cc`, `material_history_internal.h` |
| Angular-rate rotation primitives | `objective_stress.h`; legacy helpers keep their arithmetic |
| Both transported history representations | `stress_evolution.cc`; explicit cell/q evaluator in `material_tensor_projection.cc` |
| Source/sign/width/curvature/covariance/resolution | Five `[bulk_spin]` cases in `phase_field_fault_maxwell.cc` |
| Actual staged anchor/committed-state/Newton checks | benchmark `native/bulk_spin_checks.h` |
| Anisotropic prescribed native operations/stationary shortcut | benchmark `native/bulk_spin_prescribed.h` |
| Bounded runs and independent reductions | `run_bulk_spin.py`, `bulk_spin_matrix.py`, `bulk_spin_lifecycle.py`, `assess_bulk_spin.py` |

The final core passes **183390 assertions in 157 test cases**. Source identity
checks cover three segment angles, both senses, varying V/I, separate
interpolation, inactive support and independent component formulas. The native
finite-end fixture additionally exercises thousands of active/inactive Gauss
points per step. Independent native source/spin errors are near roundoff, and
both T and M rotations are compared to componentwise formulas before projection.
The shared mass projector is retained, not claimed to be independently derived
by that rotation check.

For anisotropic stress `(xx,yy,xy)=(7,-2,1.3)`, fixed accumulated slip .3 over
widths .05/.1/.4 gives legacy angles −3/−1.5/−.375 rad and zero corrected spin.
The analytic rotating/relaxing checks are below 1e−13 in the completed tensor
norm, include genuine spin .63, the infinite-viscosity limit, both senses,
zero-total/nonzero-fault spin and no-slip limits. Curved independently
finite-differenced velocity retains the analytic regular gradient within 2e−9;
observer spin is retained within 1e−14. The time-dependent-observer Lie test
gives errors .0305638, .0151875, .00757012, .00377915 for 10/20/40/80 steps,
measuring first-order convergence rather than claiming discrete exact covariance.

A smooth tanh jump interpolated in Q2 has absolute residual spin that converges
at second order. For width .1, reducing h/width from .25 to .03125 lowers its
L∞ residual from 6.59614e−3 to 1.07740e−4 s⁻¹, and L2 from 9.59772e−4 to
1.50675e−5. Widths .2 and .4 give the same refinement behavior. No smoothing or
projection was inserted to force cancellation on a coarse field.

The native prescribed check transports a uniform anisotropic DG stress with
accepted-trace inflow, samples a representable pure-slip velocity, stages the
profile, rotates, and exercises the native reaction without publishing its
synthetic parent. Corrected residual spin is 3.33e−16 s⁻¹; native rotation and
interior Maxwell relaxation errors are approximately 1e−14 Pa. Genuine rigid
spin .3 is retained. Fresh attempts with dt and dt/2 use the same parent.
The stationary test includes nonzero auxiliary B, stable residual rotation,
zero-total/nonzero-fault spin, and bitwise coefficient preservation in the exact-
zero branch. The synthetic state is rolled back before the ordinary coupled run.

**Preserved failed fixture evidence:** the first independent native reference
incorrectly read `FEStrainState::m`, which is unused/zero in this mature path;
the common fixture's actual localization coefficient is 50. The corrected
reference uses its documented coefficient. Separately, a uniform profile is
not preserved near incoming boundaries under the retained zero exterior phase
rule. Its endpoint forcing changes there, leaving a 0.0653256 Pa remainder
against the *uniform pure-slip* reference. Rotation still agrees throughout the
box. The test reports this boundary remainder and checks pure relaxation in the
unaffected interior; it does not change profile inflow or call the global
endpoint state pure slip. Failed logs/outputs remain available.

## Bounded coupled physical comparisons

All ten paired/refined runs completed. The common fixture is a 16² fixed mesh
on [−2,2]², G=10 Pa, eta=10 Pa s or `10*(1+.5*tanh(4*x))`, the existing compact
Q1 phase, fault from (−.8,−.24) to (.8,.24), label spacing .1, translation .4
m/s plus boundary shear .5 s⁻¹. Initial mechanics uses the same .1 s interval
and zero physical stress for every run; physical steps generate nonzero stress.
Growth is disabled. Ordinary DGQ2, SSPRK3, bounded MacCormack profile, full
normal support, accepted tensor trace, zero natural top traction, endpoint
forcing and all convergence criteria are unchanged. The run horizon is .1 s,
with dt=.025, plus direct-mode dt=.0125 controls for both viscosities.
These are **synthetic mature-fault tests**, neither production BP3 nor the
cohesive-interface formulation, and not the stopped equilibrated-layer campaign.

Relative differences below compare corrected versus total spin at the same
mesh/dt; they are constitutive differences, not error against an exact solution.

| Viscosity / storage | Stress L2 | Work shear traction | Endpoint V | Accumulated slip |
|---|---:|---:|---:|---:|
| constant / direct | 7.2590e−6 | 1.1974e−5 | 7.3913e−5 | 4.1088e−5 |
| constant / separated | 7.2590e−6 | 1.1974e−5 | 7.3914e−5 | 4.1088e−5 |
| layered / direct | 7.4688e−6 | 1.2246e−5 | 7.5486e−5 | 4.2484e−5 |
| layered / separated | 7.4688e−6 | 1.2246e−5 | 7.5486e−5 | 4.2484e−5 |

For direct storage, central/endpoint regional stress differences are
1.8483e−5/1.0553e−5 (constant) and 1.8960e−5/1.0838e−5 (layered). The machine-
readable comparison includes xx/yy/xy norms, pressure-aware normal traction,
shear traction, V, age, Z, cumulative slip, geometry, profile and normalization.
The reported work/point traction differences refer to the stress contribution;
the common prescribed background is unchanged.
Central means material-label fraction [.2,.8]; endpoint regions are its
complement, with a .6 m normal band for bulk metrics. Tractions use common
surface quadrature, not just vertex values at finite ends.

The direct/separated physical stresses differ by 1.4592e−5/1.5543e−5 in constant/
layered cases because their retained finite-space reconstructions differ.
Endpoint regional differences are 3.9449e−5/4.1973e−5, exceeding the central
1.3724e−5/1.4488e−5. This is consistent with the known finite-end representation
limit; the correction does not claim to repair it.

Timestep halving changes direct stress by 1.4806%/1.4163% and accumulated slip
by 12.3075%/12.8475% (constant/layered). Thus timestep sensitivity dominates the
small constitutive difference in this short, weak-slip coupled fixture. The
paired run is an implementation/physical comparison, not temporal qualification
of an event. The prescribed nonzero-stress tests isolate the constitutive
property much more strongly than these weak-slip runs.

The independent physical energy ledger retains transport flux, Maxwell loss,
bulk work and slip work. For constant viscosity, signed/absolute accumulated
endpoint remainder is −.0380002/.0380002 J/m at .025 and −.0192491/.0192491 at
.0125. For the layer it is −.0363116/.0363116 and −.0184174/.0184174 J/m.
Endpoint normalized absolute remainders are 20.8105%→10.5385% (constant)
and 19.7814%→10.0313% (layered); trapezoidal work
remainders are 0.1893%→0.1035% (constant) and 0.2212%→0.09452% (layered).
Signed and accumulated absolute errors, flux and all work terms are retained
in `coupled.json`/`coupled.csv`. A smaller trapezoidal diagnostic does not change
the endpoint scheme or establish exact energy conservation.

Each old/new pair has identical summed linear iterations: 1076 (constant direct),
911 (constant separated), 1120 (layered direct), 1044 (layered separated).
Measured wall times are 26.07/26.51, 13.39/13.19, 26.52/26.17 and 36.18/36.03 s
(corrected/total). These single observations include benchmark audits, which
are stronger in corrected mode; they are not a precise isolated kernel speed
measurement. The new snapshot stores 64 doubles per owned cell at Gauss8:
131072 scalar-data bytes on this 16² serial mesh, plus map/vector overhead and
18 committed rates. It is temporary, not another physical history field.

## Lifecycle, provenance and reusable inputs

The final four-rank comparison differs from serial by at most 9.05e−15 in
bulk exports and 1.60e−14 in surface exports. A one-rank checkpoint restarted on
four ranks differs by at most 7.99e−15 / 1.07e−14. Loading the preserved old
**total-spin executable's** checkpoint under the new total-spin core also passes
(7.33e−15 / 7.11e−15). The separated bulk-spin checkpoint restarted from one to two ranks agrees
within 7.29e−15 / 1.78e−14. The unchanged total-spin fresh trajectory is bitwise equal
to the retained matched core/checker control. Both cross-law checkpoint attempts
fail with the explicit material/constitutive identity diagnostic. The native
stationary serialization test round-trips schemas6/8 (36280/36362-byte packets)
and refuses cross-law reading in both directions, without publishing its scratch
state. This is format/guard coverage plus the stationary anchor tests; a complete
stationary restart trajectory was not added to the moving-run matrix.

Forced rejection of physical step 2 and a fresh matched timestep schedule have
bitwise identical bulk and surface exports and matching publication counts.
The accepted schedule is .025, .0125, .023875, .025, .013625 s (the last interval
is shortened to the requested endpoint). The .023875 interval comes from the
existing timestep-growth cap; it is not a newly tuned parameter. Rejection
restores the complete accepted parent and the native checker verifies the new
attempt's source, spin and anchor.

Particle, cohesive, growth-enabled and legacy-integrator combinations all fail
early with the new applicability diagnostic. The first negative fixture was
itself intercepted by the existing profile/inflow applicability guard; removing
those unrelated options let the targeted guard be verified. That failed fixture
and its diagnostic are preserved.

An independent export-based reconstruction checks all six max/L2 spin metrics
against the *previous accepted* velocity, profile, geometry and converged V.
Its largest absolute metric difference is 5.89e−15. Replacing initial converged
V by the .001 predictor fails this comparison, verifying the initialization
commit rather than merely checking a snapshot against itself. In the native
matched-slip check the accepted bulk spin is below 3.34e−16 while substituting
the staged profile changes it by .0989 s⁻¹; perturbing private Newton rates still
leaves the anchor unchanged. The actual translated compact-profile case has a
3.773e−4 s⁻¹ candidate-state gap. These are discriminating time-level tests.

The final core and units are identified separately from the earlier comparison
builds. The last rebuild only clarified the parameter help text; all 157 unit
cases were rerun on that exact executable.

Final lifecycle results and the source/binary/input manifest are recorded with
the evidence (`lifecycle-comparison.json`, `timelevels.json`, `guards.json`,
`manifest.json`).
The maintained guide documents model, discretization, ownership and checkpoint
identity. The generic runner preserves its previous defaults; the new wrapper
selects the dedicated tested core/checker and explicitly requests bulk spin.
No existing production input or checkpoint is converted. A complete portable
bounded input, fault file and runtime instructions are in
[`bulk_spin_server/`](../../../benchmarks/reconstructed_fault/particle_free/transported_objective/bulk_spin_server/README.md).
It reproduces the tested .0125 layered fixture fresh; it is not an unqualified
long-event or physical BP3 input. No server execution occurred.

![Spin checks](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/bulk_spin/spin-checks.png)

[Coupled energy/slip figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/bulk_spin/coupled.png)
and CSV/JSON tables retain the numerical comparisons.

## Recommendation and remaining limits

Use the explicit bulk-spin option for **fresh mature particle-free DG examples
whose intended interpretation is Maxwell bulk surrounding localized relative
slip**. Preserve total spin as a different-law control. The correction removes
the width-dependent artificial rotation in pure slip and retains genuine
regular/observer rotation, without changing endpoint mechanics or transport.

Implementation and bounded local verification are distinct from physical
comparison and production readiness. These tests do not qualify sharp tips,
long moving profiles, event trajectories, arbitrary heterogeneous materials,
cohesive birth/propagation, or a finite-strain elastic spin. Existing cohesive
energy/tip failures remain unresolved and separate. The prior cleanup losses
remain **39 of 45 native-pilot frames and earlier temporary logs**; no deleted
campaign was recreated.

The next justified scientific experiment is a **fresh**, compatible exterior/
top-loading layered mature-fault run with the corrected spin and a timestep
pair, retaining the existing labels and reporting separate finite-end errors.
The current weak-slip fixture's 12–13% accumulated-slip time sensitivity is a
reason to refine time before claiming an event comparison. The old total-spin
layered checkpoint must not be resumed under the corrected law. No such server
run is launched here.
