> Chronological selection record: the opening section predates the joint
> prototype. Its original failed local-inverse conclusion is retained.
> Current completed native scope is in COULOMB_GRADIENT_REPORT.md; subsequent
> selections and demonstrated implementation corrections follow below.

# Bounded gradient follow-up: independent controls after the point gate fails

The selected experimental energy is described in COULOMB_GRADIENT_LITERATURE.md
and COULOMB_GRADIENT_FORMULATION.md. No native or phase-active option has been
enabled. A saved phi=.99, Xxy=-.1 Pa conditional point problem has three
independently confirmed admissible roots. The proposed use of an independent
unique prescribed-X material inverse therefore fails its prerequisite.
Do not remove that guard, select an arbitrary local root, or call homogeneous
short-wave results a qualification of the joint nonlocal problem.

Complete only independent useful controls before reporting:

1. The complete fixed-phase point action on explicit unique-branch controls,
   preserving the original Coulomb point-test physical scales, epsilon=1e-5
   and derivative bound1e-6. Preserve earlier unit-normalization failures and
   the actual multiple-root counterexample.
2. Full mixed homogeneous branch symbols at phi=.2,.6,.75,.8,.9,.95,.99;
   k ell_s from0 to1000, including both systems and changing directions.
   Illustrative ell_s=1,3,6mm are **uncalibrated wavelength translations**,
   not mesh choices or selections for a native run.
3. One inverse-manufactured strip energy control, not a forward mechanical
   evolution solve. Fix phi=.8 and all existing G/eta/Gc/ell/m/mu/c_r/p0.
   Length .25m, 64-point Gauss integration; fresh histories. Ramp stress to
   .99 of the residual pure-shear strength by .05s, then prescribe
   `Kxy=.01(t-.05)[1+.3cos(2pi x/.25)]` through .125s, then hold the computed
   displacement until .15s, dt=.00625s. Use ell_s=0,3,6mm. This is the same
   K target but a separately calculated loading in each length case; it is
   not an identical-load physical-response comparison. Microtraction is zero.
   Stop if the assumed hold violates the actual yield inequalities.
   Check the exact extra gradient storage, BE remainder and weak microforce
   work independently. Add one affine-K manufactured microtraction identity.

Use bound1e-10 for the work ledger,1e-12 for scaled force/Maxwell identities,
and keep actual pressures strictly compressive. No field solve, native
refinement, native publication/restart or phase-active pass is claimed by
these controls. The intended next implementation must solve the joint
microforce/flow problem without assuming its conditional local inverse is
single-valued; global branch/stability and discretization need qualification.

## Subsequent joint prototype and bounded native selection

The smallest numerical correction is the already derived joint normal map,
not a new material root rule. Its spatial P1 mass-lumped strip prototype has
now passed homogeneous identity, independently predicted discrete wavelengths,
complete coupled K/strain/pressure actions,320 loading/hold/reversal intervals,
fixed-length33/65/129-node controls, one halving and an immutable-parent retry/
serialization check, including one phi=.99 trajectory. The original local
inverse failure remains. These tests do not prove global uniqueness/stability.

Proceed within the user's conditional native authorization: serial fixed
Cartesian confined blocks, fresh phase=.8, unchanged physical coefficients,
Gauss6/source3, unlimited DGQ2 T, frozen geometry and diagnostic particles.
Use the joint microforce formulation, never the independent prescribed-X
inverse. Main comparison: ell_s=6mm on4/8/16 cells per coordinate through.125s,
dt=.00625s, matched original loading. One3mm control and one halving on8×8;
one4×4 load/hold/restart/rejection control may use the existing fixture through
its .25s loading-to-hold transition. Both lengths are uncalibrated experiments.
Retain all old failures and compare common physical times.

Keep the experimental length selection confined to this default-off native
benchmark route and record it in the material checkpoint tuple. Initial zero
histories only; old coefficient tuples cannot be converted. Serial, fixed,
uniform positive phase and tensor Cartesian layout are the bounded eligibility
conditions. No production/default, phase-active, MPI/AMR or specimen expansion.

For native K use a continuous bilinear auxiliary field with nodes at the actual
Gauss6 history points, linear extrapolation through the first/last two nodes
to physical boundaries, and natural zero microtraction. This preserves affine
fields; it does not impose a K=0 or zero-gradient boundary skin. Constitutive
mass/working is the unchanged native positive Gauss6 weights. Integrate the
new gradient energy exactly on that auxiliary space (Gauss2 per subinterval).
This is an explicit discretization choice, to check independently before runs,
not a new physical length or a change of mechanics/source quadrature.
Existing native cell/QP K records are its coefficients; geometry matrices are
derived caches. No history smoothing or transfer is performed during a step.

Use the actual full u/p/K tangent, complete pressure-dependent projection
actions, native AffineConstraints once and retained residual/source/work/contact
allowances. Keep the original elastic-trial closed-contact admission too.
Mechanical conditioning means the true relaxed u/p Schur operator after K
elimination in the same physical Riesz metric as the audit; do not call the
normal-map iteration matrix a mechanical stability certificate. Measure slip
profiles, widths, storage and work alongside costs, not iteration counts alone.

## Native execution corrections (original failures retained)

The first 16×16 attempt stopped at its fresh boundary-layer Newton guess.
A first-iteration elastic lifting, with K fixed only for that guess, removes
that initialization artifact. The subsequent direction-change failure exposed
an implementation error: sticking values were snapped before EVERY Newton
solve, including far-from-equilibrium candidates. That can redistribute K and
its backstress without a merit-tested coupled step. The correction performs
exact sticking roundoff cleanup only AFTER complete residual convergence;
active sticking equations still enter every Newton solve. No constitutive
root is selected, contact guard bypassed, iteration budget enlarged, or
acceptance tolerance changed. The old failure logs, last accepted checkpoints
and trial matrices remain. A same-parent retry reaches .125 s, followed by
a fresh fine-mesh confirmation of the final implementation.
