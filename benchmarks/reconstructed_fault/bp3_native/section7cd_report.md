# BP3 sections 7C–7D: sensitivity and restricted MPI recovery

The temporal, coupled-quadrature, surface and bulk controls pass their prescribed
positive-time schedules. The ordinary BP3 transition material check and the clean
one-/two-rank controls also pass. The controlled identical-dt retry matches the
clean trajectory exactly in every exported surface and physical FE field.
Spatial sensitivity remains substantial; MPI reproducibility is a separate result.
Only the perturbed FE 7B trajectory is refined. Its evidence and all production
sources/defaults are preserved; no new particle matrix, q8, extra timestep,
unfiltered comparison or full-BP3 preparation is run.

## Matched definitions and initial convention

Baseline `fault-propagation`, `c33edaaeb145e5d3dce155bb815552ef80efc5f9`.
The same ASPECT/deal.II9.6.2/Trilinos14.2/p4est2.8.7/OpenMPI5.0.6 stack is used.
Core compiler provenance is GCC12.4.0; new diagnostic/transition builds use the
current MPI compiler wrapper with GNU16.2.1 and the exported matching Aspect_DIR.
Full binary identities and compiler records are linked below.

All coupled controls retain the 7B translated 60° fault, frozen phase/geometry,
physical loading 1e-9 m/s, homogeneous G=32038120320 Pa and eta=1e26 Pa s,
uniform strengthening a=.025,b=.015,Dc=.008 m, radiation damping 4624440,
background shear 26.546122365 MPa/normal 50 MPa, full bottom/side loading,
filter 20, stationary particles and nonlinear/linear tolerances 1e-8/1e-9.
The common smooth theta perturbation remains .02. No transport, births, adaptation
during stepping or checkpoint continuation is admitted.

The mature FE recurrence stays d_R_new=beta d_R_old+(eta_ve/G)u and
Z_new=beta Z_old+eta_ve V, with the existing exact Maxwell coefficients.
Z remains independent of identically zero mature cohesive traction. Background
tractions stay on the surface, and the physical segment/source map and paired
endpoint completion are retained.

The twenty-step dt 100000 run reaches the same T=2000000 s as 7B, keeping
`Initial time step=200000`. Its complete step-zero exported profiles are
byte-identical to 7B; mesh, phase, particle initialization and I_h also match.
Thus halving the positive-time interval does not silently alter the initial model.
Step-zero kinematics still depend on that common virtual constitutive interval;
committed d_R/Z/stress/slip remain zero. Interval-independent initialization is
not claimed. One halving measures sensitivity, not a convergence order.

The q6 run changes only the existing FE mechanical-rule option on the original
mesh/surface and dt 200000 schedule. The native tensor rule has 36 rather than 9
points. Residual/tangent assemblers, Stokes QP associations, slip/bulk-work
operators and the maintained diagnostics all dispatch through
`fault_mechanical_quadrature`; the observer checks every selected point/weight.
General consistent cell-interval normalization retains its own integration and
eight surface subdivisions: phase and I_h are exactly unchanged between q3/q6.
The measured q3/q6 response is small relative to the demonstrated spatial/backend
differences, so q6 is selected for the spatial/MPI controls and q8 is not warranted
for that question. This does not establish exact coupled q3 or refined-basis q6 accuracy.
No new FE/particle comparison mixes different rules.

## Common-time sensitivities and actual resolution

All profiles are compared at common physical times/arc lengths, using exact Q1
norms on the union of their actual surface meshes. Omega interpolates the exported
nodal product V theta/Dc. Final maximum absolute differences are:

| Field | dt 200000→100000 | q3→q6 | Surface20→10, q6 | Bulk-band refinement, surface 10/q6 |
|---|---:|---:|---:|---:|
| V, m/s | 1.989e-13 | 1.060e-13 | 5.239e-12 | 1.187e-11 |
| theta, s | 202.77 | 116.37 | 6071.98 | 13527.00 |
| Omega | .0002242 | .00009189 | .0044600 | .0101440 |
| Cumulative slip, m | 2.499e-7 | 1.268e-7 | 6.571e-6 | 1.485e-5 |
| Shear, Pa | 199.36 | 127.41 | 5419.26 | 13024.17 |
| Raw normal, Pa | 125.02 | 86.88 | 39225.26 | 43848.20 |
| Filtered normal, Pa | 44.60 | 47.82 | 2070.75 | 3349.99 |
| Pressure contribution, Pa | 148.04 | 21.60 | 42680.58 | 61008.31 |
| Tensor contribution, Pa | 23.02 | 81.43 | 3455.32 | 17160.11 |

Max V sensitivities are .01989%, .01060%, .52394% and 1.18694% of Vp.
Temporal raw/filtered normal RMS sensitivities are .404%/.896% of their measured
perturbation scales; quadrature values are 1.147%/.934%. Surface refinement gives
83.44%/20.24%, then bulk refinement 72.90%/61.69%. Background 50 MPa alone would
hide these differences. Full initial/common-time maxima, RMS, signed means,
oscillatory errors, pressure/tensor components and Z are retained in the data links.
No temporal order, spatial error bound or projection-only causal attribution follows.

Surface refinement creates 117 actual nodes/116 intervals (9.9543 m), replacing
59/58 (19.9086 m); it is not extra interpolated output. Bulk cells and phase
coefficients remain unchanged, but the consistent surface normalization is rebuilt:
min I_h 12188.324→12180.406 m, max common-location difference 10.188 m.
The next control uses the same surface 10 and q6, raising native startup refinement
by one level across the band and protected completion strips. It has 11528 cells
instead of 4136, fine width 3.90625 instead of 7.8125 m, with coarse 125 m unchanged.
Both endpoint completion admissions pass; grading/hanging constraints remain valid.

The physical profile stays peak phi=.6, ell 20 and the same AT1/Lorentz parameters.
Its new Q1 representation differs on three common normal cuts by max .021970,
RMS .003120. Min I_h rises to 12818.608 m (5.24% versus the previous minimum);
common-location max/RMS differences are 638.25/629.06 m. The physical profile is
not retuned; these discrete-profile/normalization changes accompany bulk refinement
and must not be assigned to quadrature error. The cut measure is sampled, not a
whole-domain phase-error norm.

| Final roughness diagnostic | q6/surface 20 | q6/surface 10 | q6/surface 10/refined bulk |
|---|---:|---:|---:|
| Raw normal short-wave amplitude, Pa | 197.24 | 1363.56 | 2.27 |
| Its peak wavelength, m | 73.41 | 27.73 | 48.53 |
| Filtered short-wave amplitude, Pa | 41.50 | 44.35 | 1.00 |
| V short-wave amplitude, m/s | 3.45e-14 | 4.61e-13 | 2.93e-15 |
| Lower raw local-chord max, kPa | 34.89 | 65.72 | 19.19 |
| Lower filtered local-chord max, Pa | 3789.65 | 2470.61 | 659.65 |
| Lower pressure/tensor max, kPa | 47.98/21.61 | 86.03/23.00 | 25.03/5.84 |
| Lower cancellation ratio | 2.331 | 1.703 | 1.608 |

Spectra use each native surface grid, linear detrending and a Hann window, selecting
wavelength ≤200 m. Chords measure curvature/roughness, not independent error.
Cancellation ratio=(RMS pressure+RMS tensor)/RMS(pressure+tensor) on the lower 200 m.
The apparent 73 m feature does not retain a stable wavelength/amplitude under
resolution changes. Bulk refinement sharply reduces oscillatory content, while
the remaining endpoint traction/profile sensitivity prevents a convergence claim.
The tensor law remains FE consistent_deviator; the legacy full-gradient distinction
from 7B is unchanged and no new particle control is needed to identify this uncertainty.

## Transition, MPI and publication

An ordinary maintained-plugin build omits the bottom-test definition. Nine material
probes span the actual 15–18 km transition, including its endpoints and both constant
regions. Native initial-composition assignments, horizontal bulk/sharp-fault charts,
mixed a=.010→.025, configured state inversion and the background convention pass.
Native a error=0, friction error≤1.12e-16, theta initialization error=0; theta ranges
8000→8000000 s while background shear stays 26.546122365 MPa. The unchanged small
mesh also verifies actual particle/surface initialization and zero bulk perturbation.
Probes outside that small mesh test material definitions; this is not a resolved
coupled weakening-transition or first-event experiment.

7D selects the passing refined-band q6/surface 10 configuration,184448 stationary
particles, dt 200000 and T=600000 s. One-/two-rank clean runs accept steps 0–3;
positive steps 2/3 each require two Newton updates. The serial prefix profiles are
byte-identical to the retained ten-step bulk run. At 600000 s, max one-/two-rank
differences are V 7.90e-23 m/s, theta 4.84e-8 s, slip 4.90e-17 m, Z 1.57e-6 Pa m,
raw/filtered normal 2.83e-7/3.20e-7 Pa and I_h 7.28e-12 m. Across 118968 physical
coordinate/component entries, final max velocity/pressure/d_R differences are
1.38e-23 m/s, 4.68e-8 Pa and 8.84e-18 m; phase is identical. Coincident constrained
Q2 DoFs have identical traces and are grouped by physical coordinate/component.
Raw distributed vector ordering is not compared.

Predeclared reproducibility allowances are 1e-7 of physical field scales (ten times
the larger existing solver criterion), 1e-10 for relative I_h and 1e-13 absolute phi.
Actual differences are far below them; iteration-count equality was not required.
Clean max nonlinear residuals ≤7.21e-10; all direct Maxwell and once-only
state/Z/slip/particle publication checks pass on each rank.

At step 2, only rank 1 deliberately mismatches one fixed-sample registry entry;
actual particle positions and physical parameters remain unchanged. After native
history candidates are built, the existing collective FE staging guard rejects
before vector collectives/publication. Both ranks catch the expected rejection,
with exactly unchanged accepted d_R/Z/I/state and all surface/particle property
bytes, including NaN sentinels. The fixture restores only its injected metadata,
rolls back nonpublishing scratch mechanics and lets the ordinary driver retry the
identical interval. Clean/retry differences are zero at every accepted time in all
exported surface fields and 118968 physical FE entries. One terminal publication
per positive step and the native validation of every derived particle sample pass;
there is no diagnostic second increment. This is in-memory
candidate-boundary recovery; FE checkpoint/restart and automatic cutback schedules
remain untested/unsupported here.

## Verification, preservation and next decision

| Fresh control | Accepted positive steps | Actual wall s |
|---|---:|---:|
| Temporal /q6 /surface /bulk | 20 /10 /10 /10 | 194.87 /168.97 /169.70 /527.26 |
| Transition material check | 0 | 5.82 |
| Selected serial /two ranks | 3 /3 | 184.62 /141.25 |
| Two-rank rejection/retry | 3 | 165.78 |

All trajectories retain finite positive I_h/state, frozen phase/H/geometry,
zero mature cohesive traction and independently evolving Z. Max ten-step stress
defect is 2.80e-8 Pa, at most .6271% of the original 128-epsilon operand allowance.
Temporal max normalized residual is 9.69e-9 and surface RMS .01314 Pa; q6/surface/
bulk max normalized residuals are 7.69e-10/7.71e-10/7.21e-10. All satisfy the
unchanged criteria. Theta aging defects are ≤2.23e-16 relative. Startup failures are preserved: transition include parsing selected
an unloaded manufactured postprocessor, then its observer used a released
initialization manager. The fixes use a narrow adapted fixture and the existing
shared-pointer lifetime API. A first MPI observer compilation failed on inaccessible
scratch types; the final test uses existing native calls/collective guards. Analysis
initially rejected legitimate coincident constrained DoFs; it now checks their traces.
No scientific failure is relabelled or tolerance changed. Compilation is excluded
from the declared3600 s simulation budget; all simulation launches including
failed startups total 1567.88 s (26.13 minutes). Run caps were 400/700/700/1200 s
for temporal/quadrature/surface/bulk,200 s for transition and 600 s per MPI control.

Changes are confined to benchmark observer/CMake, eight `section7cd-*.prm` fixtures,
this report, `SECTION7CD.md`, `analyze_section7cd.py`, recovery/status and specification
scope records. Core and maintained coupled-plugin binaries remain exactly7B's;
new diagnostic/ordinary-transition artifacts are separate. Incoming production
C++/headers,7B files/results and the read-only reference worktree are preserved.

The correctness/MPI gates support proceeding to the short otherwise-identical
unfiltered diagnostic and drafting a fresh full-BP3 input under these same fresh,
fixed, stationary-history restrictions. These would remain bounded diagnostics/
preparation, not evidence for removing the filter from a qualified model. The5%
I_h change and endpoint pressure/normal sensitivities remain the specific accuracy
issue; a further bulk/profile comparison should determine the eventual full-model
resolution. No resolved full-model/long-run, restart or transport qualification
follows. Neither later stage is started automatically.

Evidence under `build-bp3-native/section7cd/`:
[all common-time metrics](../../../build-bp3-native/section7cd/comparison.json),
[field differences](../../../build-bp3-native/section7cd/common-time-differences.csv),
[full/endpoint profiles](../../../build-bp3-native/section7cd/profiles.pdf),
[artifact identities](../../../build-bp3-native/section7cd/artifact-provenance.json),
[compiler records](../../../build-bp3-native/section7cd/compiler-provenance.json) and
[final preservation/verification](../../../build-bp3-native/section7cd/final-verification.json).
Exact commands/exits/times are the matching `refactoring_boundary/evidence/bp3-native-7cd-*.json`
and logs. Coupled inputs set explicit end/accepted-step/wall limits, abort on
unexpected failure and disable checkpoint writers; the material control solves no
mechanical step. New libraries are built against the current
`build-bp3-native` in separate serial/MPI/transition directories; no full core rebuild,
complete manufactured/history campaign, reset, commit or push is performed.
