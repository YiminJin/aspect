# Native bounded profile transport and coupled accuracy comparison

This is the implementation follow-up to the completed boundary/profile
isolation. It changes the experimental mature moving-fault selector, not
production BP3 or the newer cohesive-interface law. The current worktree,
branch `pf-rsf-particle-free`, revision
`27178813cebd471b93a7a9308f41215b71b4a20b`, incoming dirty source, outputs,
checkpoints and manual work were retained. Entry copies protect the incoming
core and four checker libraries. A retained binary is never identified merely
by HEAD; every run records the actual core, checker, input and runner hashes.

The [protocol](BOUNDED_PROFILE_PROTOCOL.md) states the comparison before its
completion. The implementation is in `dg_material_profile.cc`, with explicit
tensor inflow in `dg_material.cc` and selectable identities in
`phase_field_fault.cc`. The [governing extension](TRANSPORTED_OBJECTIVE_SPEC.md#103-opt-in-bounded-q1-material-profile-and-accepted-trace-tensor-inflow)
contains its discrete equations and limits.

## Implemented method and transaction

`Material profile transport = bounded MacCormack` corrects the existing Q1
accepted-velocity pullback with half its forward/return defect. It clamps each
candidate to extrema of the accepted Q1 vertices in its departure cells.
The existing remote-point tolerance and multiple-donor averaging are retained.
Missing return coverage falls back to the forward value; incoming phase stays
zero. Q1 storage, the physical transport equation and physical bounds remain.
With growth disabled, this same mapped field is the transported material
irreversibility obstacle. No Eulerian maximum, phase cap or normalization floor
is introduced. This is not a conservative remap; mass drift is measured.

The update uses the existing accepted material state and velocity, alongside
T/R, curve labels, age and auxiliary slip memory. Their normalization and work
maps rebuild before the original coupled solve. Publication is still once per
accepted physical step, with whole-state restoration after rejection. There is
no extra checkpoint history. Both new selectors enter schema7 identity when
either is nondefault; the original default identity is unchanged.

`Material tensor inflow = accepted trace` freezes incoming accepted T and
U=R+P B throughout DG advection. The original rotation and reaction follow
once. This explicitly supplies constant-normal continuation of the accepted
tensor traces. It is a finite-domain boundary approximation, not an exact
infinite-domain inflow or a new exterior constitutive law.

The fixture supplies pure translation (+/-0.4,0) on left/right/bottom and the
existing natural top, zero initial tensor data, and a compact interior oblique
profile. Thus the former imposed box-wide shear source is absent. Both methods
and the finer reference solve this same boundary problem. Reference comparisons
do not conflate the earlier zero-inflow/sheared fixture with this problem.

The initial `bp-*` matrix exposed an in-scope mapping defect: its inherited
stationary support tube clipped a profile that I integrated across the full
normal section. The corrected `bpf-*` comparison uses full-normal coverage.
The manager now admits that existing policy for finite endpoints on the mature
transported route, and either new nondefault selector requires it. Finite
material endpoints and the inverse-transpose width map remain unchanged; there
is no tangential continuation. Full-normal policy enters checkpoint identity.
Truncated and early unidentified-support checkpoints are rejected, not converted.
The original Q1/zero-inflow/stationary-support control remains available.
The truncated runs, failed first full-support admission and exact source/core
pairs are retained as diagnostics, not primary physical qualification.

The total material Jaumann spin, full 3D plane-strain tensor norm, matched and
layered exponential Maxwell reactions, RSF age update, auxiliary-memory meaning,
endpoint solve, heterogeneous projection outer iteration and acceptance
tolerances are unchanged. Q1 remapping and zero inflow remain defaults.

## Comparison and interpretation

The matrix uses G=10 Pa and eta=10 Pa s or
10*(1+.5*tanh(4*x)) Pa s, the same nonuniform age/background traction, fixed
initialization interval 0.1 s, and final physical time 0.2 s. Both translation
directions run on 16² and 32² cells; independent space/time halvings run in the
positive direction. Bounded 64², dt=.025 s calculations provide references.
These are finer numerical references, not exact continuum solutions. Costs
must be compared at equal mesh, timestep and rank count.

Independent reductions reconstruct full Q2 tensors/velocities and Q1 phase
from the native Gauss exports and calculate localization separately. They
check the actual bounded update, accepted donor extrema, exact label/age/memory
source laws, physical stress, normalization, point traction and coupled state.
The point traction metric is not a substitute for a native weak residual.

The original prescribed-translation evidence is reused. The new independent
bounded control at dt=.1 reduces analytic profile errors from18.75% to14.02%
on16², from6.744% to3.856% on32², and from1.838% to0.958% on64². This only
establishes the numerical profile mechanism; coupled adoption requires the
matrix below.

**Adopt the bounded method as an opt-in improvement for the restricted mature
configuration; retain Q1 as the default/control.** The corrected 28-run matrix
reduces profile diffusion and improves resolved coupled fields in both
viscosity cases and directions. It does not eliminate spatial, surface or
endpoint error, and it is not a production-readiness certificate.

At 32² cells, dt=.05, errors to the 64²/.025 reference are percentages:

| Viscosity / direction | Profile Q1 → bounded | I Q1 → bounded | Stress Q1 → bounded | Point traction Q1 → bounded | Work traction Q1 → bounded |
|---|---:|---:|---:|---:|---:|
| Constant / + | 4.699 → 2.882 | 5.898 → 2.912 | 3.895 → 3.559 | 2.752 → 2.415 | 2.897 → 2.120 |
| Constant / − | 4.699 → 2.882 | 5.898 → 2.912 | 3.806 → 3.461 | 2.717 → 2.394 | 2.844 → 2.075 |
| Layered / + | 4.699 → 2.882 | 5.898 → 2.912 | 3.863 → 3.524 | 2.712 → 2.385 | 2.874 → 2.090 |
| Layered / − | 4.699 → 2.882 | 5.898 → 2.912 | 3.845 → 3.496 | 2.743 → 2.409 | 2.902 → 2.122 |

Coarse point traction is worse with the bounded method in all four cases:
for example 7.652% → 7.886% for constant/+ on 16². Coarse bulk stress and
work traction still improve. A sharper profile alone is not a traction pass.
Separate h/dt controls reproduce Q1's repeated-interpolation diffusion:
at 16², halving dt increases profile error 15.241% → 15.448%, while bounded
transport changes 12.789% → 12.756%. At 32² the corresponding changes are
4.326% → 4.699% and 2.877% → 2.882%. Spatial refinement is necessary; no
higher temporal order of the complete coupled scheme is claimed.

At positive 32²/.05, the phase peak is .339196 (Q1), .346404 (bounded), versus
.348853 in the fine reference. Normal variance is .0333232, .0326425 and
.0306776 m² respectively. Bounded mass drift is +.00986% versus Q1's +.00006%;
the largest bounded drift in the matrix is .1410%. There is no mass correction.
The bounded centroid x=.079793 m compares with .079978 m in the fine reference;
full centroids/variances for both directions are in the data.

Measured normal support loss is zero in every primary case. Integral
normalization error |∫χ/LΓ−1| is at most .043995%, distinct from the I-field
error to the finer reference in the table. Native profile replay agrees to
4.996e-16; reconstructed donor-bound violation is at most 5.551e-17, within
2e-11. Native bounds remain strictly checked. Age and memory source identities
agree to 1.776e-15 and 5.421e-20 (limits 1e-11). The matrix records 340 summed
label/step cell crossings. Geometry differences to the fine references are
below 6.96e-7 m.

Positive 32²/.05 cumulative slip errors decrease .04569% → .04398%
(constant) and .04577% → .04408% (layered). Slip-rate errors decrease
.12213% → .11837% and .12190% → .11818%; age errors are about .00016%.
Memory errors decrease .04677% → .04498% and .10982% → .10787%.
These common-label-grid differences exclude the separate surface-resolution
error below.

Positive 32²/.05 wall times are 33.29 → 30.38 s (constant) and
75.03 → 71.26 s (layered), each on two ranks. Linear solve/iteration counts
are unchanged at 17/635 and 25/948, including initialization. Single timings
include startup/output and exclude queue waits; they do not establish a
speedup. Four-rank fine references cost 109–337 s. All 42 corrected invocations
(38 successful, four intended/corrected refusals) total 1,938 s of measured
runtime, excluding builds/reduction/queueing; retained truncated controls cost
another 3,553 s. No longer trajectory was needed.

[All measured tables](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/bounded_profile/MEASUREMENTS.md),
[CSV](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/bounded_profile/comparison.csv),
[profile figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/bounded_profile/coupled-profile-comparison.png) and
[energy/cost figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/bounded_profile/coupled-energy-cost.png)
retain the full refinement and both-direction results.

## Surface resolution sensitivity

At 32² cells and dt=.05, halving initial label spacing from .1 to .05 m changes
bulk stress by 0.5368%/0.5427% (constant/layered), slip rate by
0.6695%/0.6695%, memory by 0.6695%/0.6946%, age by 0.04278% in both, and
work-conjugate traction by 1.4140%/1.3760%. The profile changes by only
7.56e-8/8.65e-8 relative. Normalization I changes by 0.4121% in both.
Comparison interpolates diagnostics in the fixed material parameter only;
neither physical state nor its localization is resampled. Initial continuous
age/background data are sampled on each label grid, so this measures their
surface discretization as well as surface interpolation. The common .1 m
label-grid reference is not a surface-continuum reference. These errors must
be reduced before claims about sub-percent coupled state/traction accuracy.

## Lifecycle and preservation

The final core unit suite passes 183,170 assertions in 152 cases. The preserved
entry executable/default checker pair and rebuilt default path differ by at
most 1.954e-14 in exported columns, within the stated per-column roundoff bound.
The nondefault path is independently replayed from accepted native exports;
this is distinct from the existing unit suite's scope.

For constant and layered viscosity, completed-trial rejection matches its
accepted cutback schedule within 8.882e-15; two-to-four-rank restart through cell
crossing differs by at most 5.329e-15 and 7.105e-15 respectively. These are
numeric comparisons, not byte-identity claims. Complete bulk tensors, physical
phase, velocity/gradient/pressure, geometry, widths, I, age, V and Z are checked.
History/geometry publication counts equal accepted step numbers; only the
rejection runs record one retry. Changed profile identity, old truncated-support
history and nondefault clipped-support input all fail with the intended useful
diagnostic. No production checkpoint is imported.

Three early primary coarse runs (`c-p-q-base`, `c-p-m-base-live`, `c-n-q-base`)
used the saved support-admission build. The final build adds the explicit
coverage guard and support checkpoint identity, with unchanged numerical
expressions. All other successful primary and lifecycle runs use the final
core. Each command records which binary actually ran; the early checkpoint
files are retained evidence, not compatible restart inputs for the final core.

Exact inputs, commands, source/core/checker hashes, expected and corrected
failures, and source recovery are linked by the
[evidence index](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/bounded_profile/README.md).
An already completed finer constant reference has a roundoff-only terminal
interval of 2.78e-17 s; raw files remain, while physical comparisons use its
eight intended steps. Later fresh schedules use the representable accumulated
end clock without extending requested time or changing solver tolerances.

## Energy accounting and scientific limits

The independent bulk-branch balance uses psi=|tau|_3²/(4G), mechanical power
tau:D3, slip power tau:L[V], physical Maxwell loss |tau|_3²/(2eta), dilatation
psi*div(v), and outward material-energy flux (v.n)*psi. It reports both endpoint
and trapezoidal work remainders, signed cumulative sums and sums of absolute
step errors. The remainder measures splitting, remapping/projection, transport
and time-quadrature error; it is not fitted away as physical dissipation.

A separate diagnostic checker exports pressure for independent background,
friction, radiation and mechanical-face work. Friction uses incoming committed
age, as native mechanics does; aging occurs once after acceptance. Compare
surface work and slip adjoint identities, distinguish direct face traction
power from weak reaction power, and report both active and gross work scales.
Large cancelling background/friction work must not conceal storage error.
The initialization row has zero committed stress but the inherited finite-step
velocity/slip predictor; it is not an accepted physical step. Surface work
checks therefore apply to steps 1 onward. The initial predictor convention is
retained in both methods and all time refinements.

Raw face stress is an unequilibrated DG trace. Its mechanical power is dominated
by rigid translation times the residual integrated face force, whereas the
native weak reaction power is zero for this pure-translation, no-body-force
fixture. The exported pressure permits both to be measured independently. The
raw face value is not substituted into the weak energy ledger or repaired with
extra dissipation. Strong boundary traction and an infinite-domain continuation
are not qualified by agreement with this finite-domain numerical reference.
Quadrature sensitivity is measured separately.

For positive bounded runs, the continuum endpoint remainder as a fraction
of integrated absolute active bulk work is:

| Viscosity | 32², dt=.1 | 32², dt=.05 | 64², dt=.025 | Trapezoidal diagnostic at these resolutions |
|---|---:|---:|---:|---|
| Constant | 33.312% | 17.204% | 8.741% | 1.635%, .300%, .072% |
| Layered | 31.471% | 16.289% | 8.287% | 1.918%, .424%, .076% |

The first halving is at fixed h; the last also refines h and is not an isolated
temporal-order test. The coupled pressure-aware work balance agrees with the
bulk balance to the independently measured surface residual. The smaller
trapezoidal diagnostic does not change accepted endpoint physics. At constant
32²/.05 the signed/absolute endpoint sums are −1.3389e-9 / 1.3389e-9 J/m;
at the fine reference, trapezoidal signed/absolute sums are 1.4241e-12 /
5.7213e-12 J/m. The latter shows cancellation, which is why both are reported.
Normalizing the 17.204% endpoint defect by gross background/friction work gives
only .000529%; that scale would conceal the active-energy accuracy limit.

At bounded positive 32²/.05, incoming/outgoing material-energy integrals are
−1.52046e-11 / 4.67051e-12 J/m (constant), and −1.39482e-11 / 4.73663e-12
J/m (layered). Net flux errors to the fine reference are .0200%/.0177% of
its active work scale. Final storage errors are 1.613%/1.543%. Flux sign is
outward-positive; inflow contributes negative net outward flux. The complete
step/endpoint/trapezoid flux records are retained, not inferred by closure.

Across the ten pressure-aware reductions, physical-step surface work residuals
are at most 5.67e-15 W/m and the slip adjoint defect is at most 1.33e-23 W/m.
Final Gauss-12 versus Gauss-8 storage sensitivity is at most .00201%; the
surface-work sensitivity is at most 5.84e-12 W/m. Native Gauss-8 identities
and higher-quadrature continuum estimates are different checks. At constant
32²/.05, ∥PB−B∥L2=1.104e-5 Pa m and substituting projected localization
changes physical storage by −3.046e-12 J/m (about .0503% of storage).
The projection orthogonality identity closes below 9.10e-24 J/m across the
matrix. This fixed-state projection inventory is not the entire per-step
transport-energy defect.

The raw face-work discrepancy does **not** decrease along the tested
refinement: positive constant bounded final powers are −2.462e-6,
−2.810e-6 and −3.103e-6 W/m, while weak reaction power is near 1e-20 W/m.
At 32²/.05, −2.80963e-6 W/m comes from rigid translation times the raw
integrated face force, versus −1.24e-11 W/m from relative velocity.
Thus raw physical boundary tractions remain unqualified; this evidence does
not demonstrate their convergence or an infinite-domain boundary model.
This limits physical interpretation of the finite-domain comparison even
though its native weak residual and work adjoint pass. No compensating load,
damping, tolerance change or fitted boundary work was introduced.

The cohesive endpoint energy screen still fails at149.6/74.8/37.4% of new
fracture work; its BE and surface endpoint remainders are first order. The
localized tip fixture fails force-residual line search with both rotation and
legacy evolution, before contact/admission, also at smaller dt. Those retained
failures are independent of this mature-fault profile improvement. No loading,
solver tolerances or constitutive damping was altered to conceal them.

Broader Stage C still lacks birth/resampling, computational-boundary fault
completion, moving AMR and deformed cohesive shape/pressure work. The old
Stage C missing-transaction text is historical: the complete restricted mature
transaction and coupled layered reaction already exist and are tested here.
Production BP3 remains accepted step4237/latest spatial4182 with its first event
incomplete. Protected RSF marker2/step748 is not resumed. Cleanup's39/45 lost
historical frames and earlier lost temporary logs remain recorded. No server,
long replay, crust or propagation campaign is launched; no subduction example
is planned.


## Next justified experiment

**Completed follow-up:** the selected two cases are now documented in
[positive-translation refinement](POSITIVE_TRANSLATION_REFINEMENT_REPORT.md).
The recommendation below records the original selection; the new report
supplies results and the next boundary-reaction diagnosis.

Run two short positive-translation constant-viscosity cases at 64² with
initial label spacing .05 m and dt=.025/.0125, retaining the same endpoint
scheme, inflow and mechanical data. Comparing the first to the existing
64²/.1 m/.025 reference isolates surface resolution; the second isolates time
error at the finer surface grid. Track signed and absolute active-energy
remainders, raw versus weak boundary work, and traction/slip errors. This is
the next controlled accuracy check, not permission for a server campaign or
a claim that the unrelated cohesive energy/tip failures are solved.
