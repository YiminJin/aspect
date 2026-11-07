# Corrected-spin layered campaign — 5 November 2026

Selected by the revised CODEX_LAYERED_MOVING_FAULT_CAMPAIGN.txt. This supersedes
only the old campaign selection; its partial L0 and other evidence are retained.
L0 stopped by explicit SIGTERM during step 27 after accepted step 26 at .325 s,
not by a demonstrated constitutive failure.

The candidate copies the verified bulk-spin core (SHA256
17f7152ca5e5f78b309cfebf4586a82ac1926e1ef2b9231e225d2afdc3e525ed)
and rebuilds benchmark plugins against current headers. HEAD remains cf065254
plus preserved dirty source. Native canonical and exterior candidates are untouched.

## Frozen physical choices (before new coupled runs)

G=10 Pa, eta=5.5+4.5*tanh(y/.35) Pa s, velocity difference=2 m/s,
profile denominator in y-.6*x is .6 m (normal-distance scale .51450 m),
initial phase and finite fault dip=.6, endpoints
(-.8,-.48),(.8,.48), material label spacing=.1 m (20 labels), normal background
compression=5 Pa. Friction stays a=.025, b=.015, Dc=.008 m, Vref=.001 m/s,
mu0=.6, radiation coefficient=1. Initial age=10+2*sin(2*x_initial) s.
Growth disabled, direct physical stress, DGQ2, SSPRK3, Jaumann DG material Lie,
bulk-slip-subtracted accepted spin, prescribed exterior stress.

Initial material-label shear loading=3.7+.15*cos(3*x_initial) Pa. This is a
physical loading choice, not a friction or solver adjustment. The independent
initial scalar RSF/radiation estimate at compression 5 gives V=.0973–.1111 m/s
for ages 8–12. Using actual initial sampled chi_max=3.3532 /m, the .8-s
constant-rate envelope estimate is .1387 rad at age 10. Coupled elastic
backstress, pressure and evolving age will change this; no V is imposed and
no observed constitutive difference was used to choose loading.

T=.8 s gives eight lower-layer Maxwell times and .8 upper-layer times.
U=0,+2,-2 m/s; domain [-6,6] x [-2,2]. Added translation 1.6 m is 12.8 coarse
cells. The initial compact support plus estimated shear displacement retains
more than 2 m lateral clearance. Verify actual support and curve clearance.
All physical choices are identical across direction, law and refinements.

## Bounded run matrix

One homogeneous bridge: 32², dt=.0125, T=.4, original loading 2.7, new bulk spin;
compare the retained total-spin compatible/exterior case only as a law comparison.
Layered L0/L+/L-: h=.125, dt=.0125, T=.8. Representative L+ at dt=.00625;
h=.0625 at dt=.00625. Total-spin L+ controls at both timesteps. No separated
reruns. All cases start fresh. A new-law adaptive L+ uses existing slip/age
controls with a cap selected from these results; check matching checkpoint
continuation and one provider rejection. Add a schedule if adaptivity is constant.
Address materially different opposite-direction behavior and plausible finite
boundary influence with targeted refinement/domain checks, not a Cartesian sweep.

The layered scalar provider advances only on acceptance. Transport supplies the
unreacted parent tau(y) on every inflow face; mechanics gets endpoint v(y),
constant q and top traction (q,0), p=tau_yy. Initialization computes gamma with
.1 s but keeps stored stress zero. Exterior J_f=0, so its spin is total gradient.
There is no equivalent frozen-geometry control with this exact moving history
path; U=0 remains a physically deforming fault.

## Assessment and limits

Compare complete material-label histories, central/ends separately, stress,
pressure-inclusive normal and work shear traction, cumulative slip, age, geometry,
profile/localization, signed/absolute energy and flux remainders and cost.
Use actual chi, accepted omega/omega_f/omega_b, sampled-max exposure envelope,
material-label exposures and stress-weighted rotation tendency. Undo U*t only
for displayed/comparison coordinates; fixed finite boundaries are not Galilean
transformed. Compare law differences against each law's temporal sensitivity.
Keep endpoint law, spin lag, quadrature, solver criteria, incoming age and
checkpoint meanings unchanged. No production launch, old-law checkpoint
conversion, cohesive qualification, realistic subduction or general refactor.
Cleanup losses remain 39/45 native-pilot frames and earlier temporary logs.

## Numerical grid amendment before the fine run

After L+ completed in 2349.92 s, choose a 1.5x bulk refinement for the bounded
spatial comparison: 144x48 cells, h=1/12 m, dt=.00625. The initial box has 9x3
cells and four uniform refinements. This replaces the proposed h=.0625 mesh
above before any fine-grid or old/new comparison results. All physical inputs,
T=.8, domain, profile width and 20 material labels remain frozen. It is a genuine
isotropic refinement of h=.125 with non-nested grids, sufficient for the requested
pairwise spatial sensitivity; no asymptotic order or exact-error claim is made.
The runner records depth repetitions, and reconstruction/metrics use actual h.

## Admission correction after the first fine-mesh attempt

The 144x48 physical mesh and timestep remain selected. The first attempt exposed
a bitwise-coordinate Cartesian guard, before physical stepping. Measured
roundoff was at most 1.78e-15 m. The separate `layered-bulk-cartesian` candidate
permits coordinate-scale floating-point roundoff only; a sheared negative
control still rejects. This is an implementation defect fix, not a changed
solver tolerance or mesh/physics tuning. Preserve the failed `plus-fine` and
use fresh `plus-fine-cartesian` output for the corrected attempt. The original
candidate remains the identity of all completed full coarse controls.

## Adaptive policy selected from the completed timestep pair

Use L+ on the coarse h=.125 mesh with maximum dt=.00625 s and the existing
reconstructed-fault predicted logarithmic age-change bound .05. The uniform
cap has central/end final cumulative-slip pair sensitivity .709%/1.357%; its
maximum actual age log-change is .13518. The .05 bound targets resolution of
that early state transient, while preserving the cap later. It is a policy
to assess against complete uniform histories and cost, not an accuracy gate.
No new controller, tolerance or physical parameter is introduced. Use the
separate Cartesian-admission candidate, validated bitwise on the coarse-grid
prefix, then restart its own intermediate checkpoint with two ranks.

## Conditional old-law mesh control selected

The completed corrected-law mesh comparison gives central/end cumulative-slip
sensitivities 2.775%/4.838%, versus the coarse-grid law difference 3.418%/4.216%.
Surface stress sensitivities are 8.710%/4.923%, exceeding the 2.667%/2.166% law
difference. Therefore run the handoff's conditional matched total-spin case on
144x48 at dt=.00625, T=.8, with the same 20 labels and physical inputs. Use the
same Cartesian-admission candidate as the corrected fine result. This choice
is made before that old-law fine result; no loading or other coefficients change.

## Conditional domain control selected

Refining L+ reduces its profile/central-slip discrepancy from U=0 substantially,
but translation-removed velocity still differs .9725%, versus .2470% under mesh
refinement. Run one U=2 case with halfwidth8 (128x32, h=.125), dt=.0125, T=.8,
matching the retained coarse L+ otherwise. This measures plausible fixed-boundary
influence while preserving physical cell size and surface labels. It is a
domain-sensitivity control, not a change to the frozen principal matrix.

## Existing-controller scheduling audit

The adaptive comparison retains the actual pre-publication scheduling order.
The next proposal sees the earlier committed age/rate, so its .05 predictor
bound is not a strict actual endpoint-change bound. Early recorded maximum
actual change is .08273. Complete predicted/latest-parent/actual histories are
reported by `check_layered_timestep.py`. No controller or acceptance gate is
changed in this campaign; the cap and measured sensitivity define the practical
assessment. This observation does not authorize silent endpoint publication
before an outer rejection decision.

## Closeout, 6 November 2026

The selected run matrix and both conditional controls (matched total-spin fine
mesh and halfwidth 8 domain) are complete. Adaptive cap .00625 s with the existing
.05 predictor completes 217 steps; its 58-state 4→2-rank continuation and provider
rejection control pass. The report retains the predictor/actual distinction,
finite-domain response and nonconservative profile balance defect. No native
integrator, endpoint law, spin lag, physical input or solver criterion changed.
Only Cartesian coordinate admission was corrected for measured roundoff. The
normal-distance clarification for the profile denominator changes no input.
Further profile implementation and pilot work are recommendations, not queued
continuations of this campaign.
