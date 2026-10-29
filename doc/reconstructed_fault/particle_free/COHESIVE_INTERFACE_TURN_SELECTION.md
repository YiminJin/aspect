# Resolved turning and one bounded selected extension

Baseline f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved local work.
Retain quasi-static bulk mechanics, radiation damping, the cohesive–RSF point
law, G/eta/Gc/strength/state calibration and a single Gc integral of surface a.
No production/BP3, free-network, finite-rotation, opening or dilation change.

## Turning kinematics and mechanics

Retain the old closed sharp-corner control: continuous vector traces and zero
opening on two nonparallel faces imply zero jump at their vertex. It is not
removed. Instead qualify C2 graph geometry y=g(x), with continuous tangent t(x)
and exactly tangential jump [u]=t(x) delta(x). A symmetric two-sided lifting is

  u = u_cont + (sign(y−g)/2) sum_j phi_j(x,y) t(x) delta_j.

The bulk native Q2/P1disc mixed solve retains both vector components and sided
pressure traces. A Q4 geometry field represents the piecewise quartic graph
exactly. Its jump lifting is a selectable analytic enrichment, not interpolated
nodal tangent vectors. Include the complete gradient

  grad(phi_j t) = t tensor grad(phi_j) + phi_j t'(x) tensor e_x.

Then [u]·n=0 pointwise, and pressure work at the closed interface is zero without
dropping any constraint. Use actual arclength and full 3D plane-strain energy.
Condense the native mixed bulk operator; preserve its complete current-pressure
response in the nonsymmetric point-return tangent. Verify against independent
higher-order physical integration, an affine patch and gradient finite differences.

The old segment is x=.25→.5, y=0, in a finite 1×1 m block. Beyond x=.5,
g'=tan(alpha)(3z²−2z³), z=(x−.5)/w, over width w, then constant slope to x=.75.
The fault ends there; further graph flattening is only a bulk-mesh coordinate
map, not another fault. Start w=.25 m, test w=.125 m at resolved spatial scale.
This geometric length describes a finite turn, not a material regularization or
a claim about the zero-radius limit. Do not erase its physical effect.

First use a fully mature, prescribed curved surface with nonuniform age and
pre-existing stress-free slip. Require sustained nonzero rate throughout the
turn, positive sided contact, small normal gap and physical work/dissipation
accuracy. Compare 32×16 and 64×32; check turn width and one timestep halving.
Finite slip remains small compared with curvature radius, as required by the
retained small-strain, fixed-reference contact model.

## Additional bounded direction rule (experimental hypothesis)

The tested surface law does not select geometry. Add a **finite-horizon candidate
rule**, not a global incremental potential: among forward continuations in
alpha=[−60,60] degrees, compare trial cohesive–RSF evolution from the SAME incoming
state under the SAME loading. Rank the mean new cohesive damage on the append,

  S(alpha) = integral_new a ds / length_new.

This measures resolved formation over an explicit preview interval; it is not
first-point activation or a Coulomb-angle intact criterion. Normal pressure
affects subsequent friction, not the unchanged intact strength. The candidate
family, finite turn width, preview horizon, and normalization by actual append
length are additional path assumptions. Established geometry never rotates.

Start with −60,−40,−20,0,20,40,60 degrees. Inspect the two half-angle neighbours
around the leading resolved sector, within the same angular bounds. Compare the
two leading candidates at one finer mesh, one temporal halving, and alternate
turn width. Check an exact reflected-loading/geometry control. Require a score
gap exceeding twice the observed combined spatial/temporal uncertainty before
claiming a unique selected candidate; otherwise report the competing sector.
Compare scores at 80% and 100% of the prescribed loading interval to expose
preview-horizon sensitivity. No endless angular search or straight-mesh campaign.

Use a fresh **nontrivially loaded, equilibrated** old straight cohesive segment,
not a rotated imported developed trajectory. Homogeneous initial stress has
tau_xy=G gamma0 with gamma0=.0003 and the declared axial/shear load ratio below.
Old a=.75+.05 sin²(4 pi (x−.25)), old scalar jump is a compact nonzero Q2 bump,
and old reference b is chosen so q=G gamma0 on the horizontal old segment:
e=C0 a/(1−a) G gamma0, b=jump−e. This is strictly below the old damage obstacle.
Old age is nonuniform, old relaxed-jump memory is zero; the nonzero homogeneous
Maxwell tensor is retained as the affine stress mode. These are explicit prepared
initial histories, not a claimed fresh-loading trajectory or a history reset.

All candidates share that old material geometry/IDs/histories. New records are
born tied with a=jump=b=relaxed memory=absolute slip=0 and the accepted virgin
age. At this initial stress all new planes remain below 3000 Pa. At zero-time
insertion verify old stress, bond/fracture/bulk storage and history identity;
the complete weak tied certificate gates admission. Candidate previews remain
unpublished. Only a sufficiently distinguished choice may be committed and
replayed from the unchanged incoming state; otherwise retain the alternative
branches explicitly. Check deterministic retry/rejection/restart and counters.

## Loading and bounded gates

Prescribe affine boundary strain E0 with E_xy=1/2, E_xx=−sqrt(3)/2,
E_yy=+sqrt(3)/2, so homogeneous maximum shear planes are ±30-degree conjugates
relative to the appropriate principal axes, not friction-shifted Coulomb planes.
Increase gamma from .0003 to .00145 at .004/s (duration .2875 s), then hold .025 s.
The nominal intact far-field maximum shear is 2900 Pa, below strength. This is a
finite specimen; no distant-boundary or unrestricted remote-nucleation claim.
For reflection reverse shear sign and reflect y, while keeping the diagonal
loading unchanged; signed slip changes sign, normal pressure and work do not.

Initial timestep .0005 s, split exactly at the loading endpoint. Retain solver
weak residual 1e−8 Pa, derivative 1e−6, ledger 1e−10 scaled, positive contact
and the existing admissible symmetric-tangent gate. No general RSF potential
stability theorem is inferred. Practical checks: <1% absolute physical-work
remainder/sliding (and new fracture where nonzero); <5% spatial response changes;
geometric normal gap <1e−12 m; gradient FD <1e−6; mechanical integration <1e−4
relative. Scores must reflect >=4 resolved append cells at a>=.05 before direction
evidence is used. Declare failed/ambiguous results instead of forcing a choice.

Sequential local kernels/reference runs only: 300 s/kernel, 900 s/trajectory,
3 GiB RSS and ~1 GiB additional output. Preserve failed trials and last accepted
state. Source and native FE operators compute response; no scalar straight-strip
kernel is substituted for curved mechanics. Keep all prior failures/archives.

## Measured pilot diagnosis and the single main-specimen choice

The unit block's loaded cohesive seed produces no resolved new cohesive zone:
all Q1 candidates have zero cells with mean a>=.05; the largest mean append a
is only 4.35e−6. No direction is selected from these pilots. The full-Q2 trace
also exposes a native signed line-search stop at first weak admission; a fixed
active-set predictor and bounded step checks do not repair it. Preserve that
unresolved numerical evidence. Do not weaken the complete signed certificate.

Use the previously qualified **Q1 incremental jump space**, as an exact Galerkin
restriction of the same native vector mixed operator. Bulk Q2/P1disc and the
curved analytic lifting remain. Preserve the old full-Q2 permanent jump as an
affine offset: delta=delta_old+Q1 cumulative_increment. All six initial Gauss
histories, their nonzero slip/reference/age, and the uniform nonzero physical
Maxwell tensor remain identical. This is a discrete replacement, not a history
projection or constitutive change. General remapping of arbitrary old elastic
Maxwell modes into a smaller space is NOT qualified.

The small seed's stress-drop scale Delta_tau² Lseed/G is too small relative to
Gc for a resolved formation test. Make one informed enlargement, using an exact
isotropic similarity of the homogeneous native bulk problem: a **12×12 m finite
block**, old segment x=3→6, append ending at x=9. No accepted pilot history is
scaled or imported; initialize the same stated physical histories freshly.
The old material law and Gc remain unscaled. Boundary strain, hence background
stress, is prescribed; wall displacement scales with specimen size.

For scale S, D=diag(S,1,...), the exact native mechanical transformation is
K_S=D K_1 D, pressure/stress response F_S=F_1 D/S, arclength weights S w_1 and
area weights S² w_1. This retains the actual vector mixed curved operator, not
the scalar straight-strip kernel. Independent physical energy uses the transformed
Gauss8 Gram, and contact/pressure/derivatives remain fully coupled.

Choose axial ratio .5 tan(40 degrees)=.41955: homogeneous shear planes include
the +20-degree forward direction. Increase gamma .0003→.0022 at .004/s and hold
.025 s (load duration .475 s). Nominal far-field maximum shear is 2872 Pa<3000.
This loading and size are fixed across ALL main spatial/angular/time comparisons.
Primary turn width is 1.5 m (normalized .125), alternative 3 m (.25). Main bulk
64×32 and focused 128×64 control resolve the primary turn with 8/16 cells;
the alternative is also resolved. These remain deliberately finite boundaries.
Initial main timestep .00025 s; the selected temporal control halves it.

The allowed candidate sector, preview rule, score-gap/uncertainty criterion,
four-cell cohesive requirement and original solver/work gates remain. Report
near-zero-response relative-work failures as failures; their absolute remainders
and direct score sensitivity must be visible. Do not infer direction from an
unresolved tiny response. No further specimen/parameter search follows this choice.

## Common-window cap before focused validation

The +20-degree trial reaches the candidate-end neighbourhood (14/16 append
cells with mean a>=.05) at .45475 s. Its later near-saturation is not direction
qualification. Retain the full diagnostic trial, but set the **common direction
comparison to .45 s**, gamma=.0021, before this event. All earlier loading/history
is identical; no accepted clock/history is rewritten. The new angular/fine/time
controls start freshly with that loading endpoint and stop there, with no hold.
The separate mature-turn controls retain their long hold. The direction test is
shorter because of its candidate boundary, not because hold removes stress.

At .45 s the +20/+40 trials have 10/7 resolved append cells. Select the two
half-angle probes around the leading coarse sector, then the two leading
candidates for 128×64 and timestep halving. Compare the same fixed seven-angle
family once at the alternative turn width; refine its leading candidate. Use
the reflected geometry/load at the leading fine candidate. This is one bounded
geometric sensitivity control, not a material-parameter or straight-path sweep.
Report ranking changes with width/horizon and unresolved competitors explicitly.

## Implementation clarifications retained before closeout

The implemented old seed is the complete quadratic bump
`shape=4z(1-z), z=(x-3)/3`, with `a=.75+.05 shape` and
`theta=1.2+.1 shape` on the old 3 m segment. This corrects the initial
sin-squared wording above; all main cases use this identical prescription.
The main-specimen amendment supersedes the unit-block loading and costs above.
Raw native operator/field output ultimately occupies about 2.5 GiB, exceeding
the initial 1 GiB output estimate; its data are retained in the work directory.
The deliverable archive excludes these redundant large response tables and
includes compact operators, checkpoints, plotted data and their source instead.

Only Q1 **incremental** jump modes are used in the qualified evolution.
The full Q2 permanent jump offset and every Gauss-owned history remain.
This does not establish a repair of the retained first-admission Q2 failure.
The chosen path rule is an anticipatory finite-horizon experimental closure.
Changing its turn width or horizon changes the physical question; neither is
silently optimized away. A selected primary candidate can be committed within
that declared closure while a geometry-independent angle remains unqualified.
