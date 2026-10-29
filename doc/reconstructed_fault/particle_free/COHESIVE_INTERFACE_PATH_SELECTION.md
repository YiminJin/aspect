# Accepted-state incremental path experiment

Baseline f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved local work.
Quasi-static closed contact, full current pressure, the cohesive–RSF return,
G=1e6 Pa, eta=1e8 Pa s, Gc=1 J/m², peak=3000 Pa and radiation remain.
The previous future-endpoint damage score is not the adopted rule here.

## Angular convention and formation work

Angles denote unoriented tangents counterclockwise from +x, modulo 180°;
normals are n=(-sin(theta),cos(theta)). Old tangent is 0°. Under the retained
loading tau=G gamma [[-2a,1],[1,2a]], a=.5 tan(40°), the principal compression
axis is -25°, maximum-shear tangents +20° and -70°. Isotropic pressure does not
change them. With Rstress=G gamma sqrt(1+4a²), d=theta-(-25°),
tau_tn=Rstress sin(2d), pn=p0+p'-Rstress cos(2d). The diagnostic constant-Coulomb
maximizers are 17.144703° and -67.144703° for mu=.1. They are NOT the current
intact criterion. Physical stress, flow/slip direction and geometric/front
tangent are exported separately.

Minimizing the virgin bond plus Gc a gives, for small signed slip e,
psi_b+Gc a = peak |e| - peak e²/(2 delta_f), delta_f=2Gc/peak.
Thus first-order formation work is |t.sigma.n|-peak; subsequent friction and
radiation enter at higher order in virgin slip. This justifies a maximum-shear
formation preference without a Coulomb strength shift or a global RSF potential.
It does not, by itself, prescribe a continuous spatial path.

## Additional physical hypothesis

Use the full tensor averaged with W=(1-r²/R²)^2 inside a physical disk of
radius R=.75 m, centered one R ahead of the accepted geometric endpoint along
its tangent. Choose the two analytic maximum-shear planes of that accepted
tensor. They are conjugate ties; the existing fault supplies a directional
preference: choose the plane with smaller rotation from the old tangent. Report
both work-equivalent planes. If their rotations are indistinguishable within
uncertainty, do not manufacture a unique choice. Search diagnostics use the full
unoriented angular circle; a graph-unrepresentable selected plane is a stop,
not a reason to narrow the search sector.

New virgin geometry has directional persistence over the SAME physical length R:
theta1=theta_star+(theta0-theta_star) exp(-Delta_s/R). This is an explicit,
uncalibrated nonlocal formation hypothesis, not an extra stored bending energy
or an equation implied by surface RSF. R=6.75 G Gc/peak² is fixed across all
mesh/step/angular comparisons. No artificial term is added to fracture work.
Only virgin, tied representation is created; established directions never change.

Each new arc is a C2 graph whose slope interpolates tan(theta0) to tan(theta1)
with 3z²-2z³. Choose its projected length to obtain the prescribed arclength
Delta_s. Its interpolation width equals the numerical extension step, not a
new physical turn radius. Halving Delta_s tests that approximation. In constant
stress the endpoint-angle update has an exact semigroup and tends to theta_star;
the smooth graph approaches the continuous directional-persistence trajectory.
This distinguishes physical R from numerical interpolation width.

The path is reserved while still tied, using only accepted state. After a .15 s
preparation, append until the connected a>=.05 zone has R of tied geometric
reserve. Do not exceed 2.25 m new arclength. A contour is a support-scheduling
diagnostic, never a damage mask. Tied birth must satisfy the full signed weak
strength certificate, preserve stress/storage and consume no physical time or
fracture energy. If already overloaded, preserve the trial and stop/diagnose;
do not accept an artificial zero-slip damaged birth.

## Avoiding evolved-history remapping

Use a fixed Cartesian mixed bulk FE mesh with a nested analytic jump enrichment
L_j=sign(y-g)/2 * (1-|y-g|/h_y)_+ * b_j(x) t(x), where b_j is Q1.
The envelope width h_y is a numerical lifting support, not a physical band.
Integrate its complete gradient, split at g and g+-h_y and all x support/graph
knots. Assemble the continuous native mixed matrix A independently of geometry.
Old basis supports end at the old endpoint and remain literally unchanged.
An extension adds the old-endpoint mode and new internal modes; no old mode is
rewritten. K=D-B^T A^-1 B and all old physical stress/pressure columns therefore
remain unchanged. Old Maxwell coefficients are zero-extended, not projected.
Old surface Gauss records/IDs and six histories remain unchanged; new records
start tied, a=jump=reference=relaxed memory=absolute slip=0, virgin age at the
accepted clock. Permanent incoming slip is a separate unchanged affine offset.

Prove/check nesting for the entire old operator span and actual nonuniform
evolved coefficients, including physical stress, pressure, storage, weak loads,
contact, retry and restart. All physical updates use the existing native C++
signed solver. No production owner/backend is replaced.

## Focused controls and gates

Fresh 12x12 m finite specimen; old segment (3,0)->(6,0), same quadratic prepared
seed and gamma0=.0003 as the turning milestone. Load at .004/s to .45 s, then
hold .01 s if admissible. Initial Delta_t=.00025 s. Stop on contact loss,
unchanged native signed/tangent/energy limit, boundary margin <R or capacity.
Save all failures. Cap 300 s/kernel, 900 s/trajectory, 3 GiB RSS, 1.5 GiB new
raw output. No server job, inertia, broad convergence or Q2-admission repair.

First qualify a small fixed-mesh enriched affine/curved control with independent
higher-order integration. Then compare Q2 bulk on two meshes, one Q3 alternative,
Delta_s=.75/.375 m and analytic/full-circle sampled angular preference. Keep R,
materials and initial continuum data fixed; do not transfer between resolutions.
Use common-time growth-relative functionals, actual fronts/tangents, slip,
cohesive-zone geometry and work. Practical targets: <5% integral growth/slip
changes, front differences below one fine surface cell, <3° local tangent
changes when growth is resolved, <1% sum-absolute physical-work error relative
to sliding and new fracture. Retain weak 1e-8 Pa, stress 1e-7 Pa, FD 1e-6,
normal gap 1e-12 m and scaled ledger 1e-10 gates. Report failures/uncertainty.

The maximum-shear/persistence rule is an adopted experimental propagation
hypothesis. It is not the configurational-torque law for potential brittle
fracture: that derivation assumes a different gradient-dynamics setting
([Hakim–Karma](https://arxiv.org/abs/0806.0593)). No such stability or path theorem
is inherited for this dissipative current-pressure RSF/contact problem.

## Pilot-informed bounded horizon (before the main comparisons)

The 32-square pilot demonstrates three admissible insertions, including a
strongly nonuniform evolved Maxwell state, but its process zone reaches the
2.25 m prospective endpoint during the .46 s hold. Preserve those late data as
capacity-influenced, not path validation. Use a common fresh loading endpoint
.44 s and .005 s hold for the main controls, with maximum 3 m new arclength.
Stop if the R reserve is exhausted at that capacity; do not pin a physical
front at an unreported computational endpoint. Material/seed/loading rate,
physical R and all acceptance gates remain fixed. The two initial insertions
at .15 s prefill the tied reserve; distinguish them from later growth-triggered
successive extensions. No arbitrary future damage endpoint enters selection.

Front diagnostics beyond the old endpoint use the actual Q1 jump threshold
only after verifying its damage relation against every owned new Gauss record.
Otherwise retain cell brackets. A fully decohered new front is available only
after a=1 reaches new material; its zero-advance baseline is not an existing
fully mature initial tip. No constitutive histories are interpolated or changed.

## Consistent interpolation correction after the declared step test

The retained C2 zero-curvature-per-segment approximation changes growth about
20% on Delta_s halving. In the stated continuous hypothesis
`dtheta/ds=(theta_star-theta)/R`, curvature is generally nonzero at a newly
formed segment boundary. Forcing it to zero at every boundary produces
derivative oscillations that affect bulk strain through `t' delta`; matching
positions/tangents alone is insufficient. This is a numerical interpolation
defect, not a reason to tune R or the cohesive law.

Use cubic Hermite interpolation of the graph slope p: match p0,p1 and
`dp/dx=sec(theta)^3 (theta_star-theta)/R` at both ends. Adjust the projected
length to preserve Delta_s. The graph is C1 across accepted-driver changes,
with bounded curvature jumps, continuous normals and exactly zero opening.
It converges in curvature to the same physical persistence equation. No sharp
corner, opening, dilation, bending energy or fracture inventory is introduced.
Old geometry and all old basis columns remain unchanged.

Keep the complete v1 source/binary/data. Use a separate build and v2 cache for
the corrected approximation. Require an independent homogeneous ODE/curvature
check and a small native energy/gradient/contact check before comparisons.
Bound corrected follow-ups to 64/96 Q2, 96 with half extension step, one Q3
alternative, and one late-window time halving from its own checkpoint. Stop
after these controls, preserving failed spatial targets rather than starting
a larger mesh campaign. Physical R, loading, materials and tolerances stay fixed.

## Normal-contact discrete representation check

The cut-cell point-pressure diagnostic exposes a separate limitation. For a flat
interface inside one background cell, pressure and the continuous velocity
gradient have identical traces. The tangential lifting gives [div r]=m', so
the full 3D deviator produces [pn]=2G m'/3 rather than a common contact traction.
This cannot be repaired by a tighter nonlinear tolerance. Preserve point-pressure
runs and their failed spatial targets. Do not shift/lag/prescribe friction pressure.

Before another evolution, test a genuine variational normal reaction. Enlarge
the discrete virtual space with one normal-jump bubble per surface cell,
`Ln=H rho * 4z(1-z) n(x)`, and impose its amplitude dn=0 with a cellwise normal
multiplier lambda. Tangential physical trials are unchanged. The normal row is
`Fn - Mn lambda=0`; the constraint row gives dn=0. Therefore
`lambda = p0 + (Dn,t + Bn^T A^-1(-Bt) + divLn*pmean) c / Mn`,
with `Mn=-integral_bulk divLn` (the actual native bulk-work measure, equal to
the surface bubble integral). Include the complete n'(x) gradient and physical
mean-zero pressure correction. This is the Schur elimination of a closed-contact
mixed formulation, not normal opening, dilation or a new resistance law.

Use this current DG0 multiplier at the Gauss history points. Its derivative is
the complete linear work operator, so the existing nonsymmetric tangential
Jacobian uses beta*Nwork rather than a fixed/lagged normal pressure. Normal
contact work remains zero. Preserve raw sided Cauchy tractions unfiltered as
diagnostics. Compare the multiplier against a matched independently fitted
native reference and a nearly equal-resolution cut-grid placement before using
it. Physical calibration, p0 and the single Gc inventory remain unchanged.
Keep point-pressure the legacy experimental default; select work-reaction only
in explicitly labelled fresh controls. No checkpoint conversion between them.
If the check fails, stop physical qualification rather than add another remedy.

The frozen test gives .84–.94% RMS weak-normal perturbation differences from
the independently fitted/bubble-projected reference, with both aligned and
cut placements. Raw point tractions remain unfiltered diagnostics; their
non-vanishing jump is not relabelled as equilibrium. The small independent
bulk-work quadrature and complete pressure-coupled FD check also pass. Proceed
to labelled fresh work-reaction trajectories, keeping the legacy point-pressure
trials as controls. No existing history is converted to a different pressure mode.
Repeat the same bounded spatial/extension/alternative/time comparisons and one
surface refinement; stop after those measurements. This is a change of the
experimental contact discretization, not the continuum cohesive–RSF law.

The initial raw-output estimate was insufficient for the two demonstrated
numerical corrections. Preserve all runs and allow at most 2.75 GiB task output;
use lossless compression for new native field tables and retain a disk margin.
Compressed/uncompressed matrices/fields must be bitwise identical. No existing
BP3 data, build, output or checkpoint is removed.
