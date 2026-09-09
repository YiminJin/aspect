# K1 CPDI cause, minimum correction proposal, and stationary-equation audit

Original status: **cause established; production correction awaiting approval**.
The subsequently approved correction, deduplication audit and unchanged-parameter
initialization comparison are recorded in `stage_K1_ownership_correction_review.md`.
The remainder of this document preserves the pre-correction investigation.
Stationary-profile H initialization, including activation and baseline values,
is unchanged. No phase initialization or mechanics solve was repeated in this
task. Three bounded debugger probes stopped during particle-domain generation,
before the phase solve. Existing saved initialization data were reused.

## 1. Confirmed cause

The failure is a roundoff gap in **unique FE-cell ownership of CPDI sampling
points**. It is not an H-initialization defect, a phase convergence failure or
a reconstructed-fault algorithm problem.

`source/particle/particle_domain.cc:964`, `is_inside_unit_cell()`, tests internal
faces with exact half-open inequalities: reference coordinates must be >=0
and <1. The existing eps=1e-12 is applied only at physical domain boundaries.
At lines 1102–1107 and 1113–1118, each Voronoi corner/centroid is independently
mapped into candidate cells; a rejected point contributes zero basis values.

A debugger inspected actual production polygons, candidate reference
coordinates and final stencils without changing any value. For particle 3836,
two corners have physical y=-8.6736173798840355e-19 m. At the shared y=0 face:

| Candidate cell | Computed reference y | Existing decision |
| --- | ---: | --- |
| Cell immediately below y=0 | 1 exactly | Reject: eta >= 1 |
| Cell immediately above y=0 | -5.5511151231257827e-17 | Reject: eta < 0 |

Neither cell owns those corners. The lower-cell mapping rounds away the tiny
distance from its upper face; the upper-cell mapping retains it as a negative
coordinate. This is a gap between two independent floating-point tests, not a
finite geometric hole. The centroid and other two corners have one owner each.

The observed polygon is divided into the same centroid triangles as production.
For a constant nodal field, its retained sample values are (0,1,1,0) at corners
and 1 at the centroid. Integrating those values predicts sum(w)=2/3. The
signed-face integration predicts the measured summed gradient (approximately
0,192) m^-1. This accounts for **both** constant-reproduction failures.

Two interior controls establish that the mechanism is not confined to the
x-periodic boundary:

| Particle | Position near y=0 (m) | Corner owner counts | Predicted / production sum(w) |
| --- | --- | --- | --- |
| 3836 | (0.00260417,-0.00260417) | 0,1,1,0 | 2/3 / 2/3 |
| 3983 | (0.0703125,-0.00260417) | 1,1,1,1 | 1 / 1 |
| 3986 | (0.07552083,-0.00260417) | 1,1,1,0 | 5/6 / 5/6 |

`check_cpdi_cause.py` independently integrates the captured polygon sample
ownership, and checks weight sums to 1e-13 and summed gradients to 1e-10 m^-1
against the previously saved production stencil data. All three cases pass.
These tolerances verify the causal replay; they are not relaxed production
acceptance thresholds. The broader prior packet records 68 affected particles;
this task directly traces three representative production polygons, not all 68.

## 2. Smallest proposed correction

Limit the fix to sampling-point ownership in the **2-D CPDI construction** in
`source/particle/particle_domain.cc`, with a file-local implementation helper
if needed. Do not alter particle geometry, volume, triangulation, integration,
phase assembly, H, activation, normalization or solver tolerances.

For each polygon corner and, where used, centroid:

1. Determine one owner in the existing candidate-cell patch. Preserve the
   existing strict half-open choice when it uniquely succeeds.
2. Only if no strict owner exists, consider cells whose mapped point lies
   within the existing eps=1e-12 reference-cell tolerance. Select one owner
   deterministically (closest reference-cell distance, stable cell-ID tie-break).
3. For that fallback only, project the near-face reference point onto the
   selected unit cell, then evaluate its Q1 basis once. This removes tiny
   out-of-cell coordinates rather than admitting negative basis values.
4. Never add contributions from multiple tolerance-neighbor cells. If the
   point is outside every candidate even at tolerance, report the actual
   missing-support failure rather than silently inserting zero values.

Cache the chosen owner/reference point for reuse by the existing cell loop;
the two-dimensional triangle integration remains unchanged. No new public
interface or user parameter is needed. On an exactly shared conforming face,
both candidate cells restrict to the same global Q1 trace, so the deterministic
owner does not select different physical data. Hanging-node and MPI cases
still need regression coverage through their existing constraint infrastructure.

A one-line symmetric tolerance expansion is **not sufficient**: it can make
two cells own the same point and double-count its contribution. Renormalizing
assembled weights, clipping gradients, imposing symmetry or substituting an
ideal phase profile would conceal the ownership error rather than repair it.

The separately observed polygon-orientation sign of affine gradients is not
part of this constant-reproduction correction. A per-particle common sign
reversal cancels from this phase gradient bilinear form. It remains documented
separately; do not fold an unrelated orientation change into the proposed fix.

### Regression scope after approval

- Capture-based tests for these failing/control polygons, exercising actual
  production ownership and CPDI integration; require sum(w)=1 and sum(grad w)=0.
- Exact shared-face samples and next-representable perturbations on both sides:
  exactly one owner, no dropped or doubled sample.
- Preserve the unaffected stencil and original particle volumes. Check
  constant reproduction across the full existing regular 3x3 particle layout,
  including interior and periodic-edge locations, on one and two ranks.
- Exercise nonuniform neighboring cell sizes/constraints if the helper is
  used on such meshes; do not rely on conforming-grid half-open arithmetic to
  guarantee uniqueness there.
- No complete ASPECT suite or later benchmark family.

## 3. Independent H/profile phase-equation check

Let A=Gc/(c0 ell)=64 Pa, ell=0.15625 m, alpha(phi)=phi, core c=0.6, and
h(phi)=1/g(phi)-1. The implemented strong-form phase equation, corresponding
to `PhaseFieldHandler::assemble_phase_field_system()` at
`source/simulator/phase_field.cc:806`, is
\[
    H(y) g'(\phi) + A \alpha'(\phi) - 2 A \ell^2 \phi'' = 0.
\]
The intended stationary profile constructed by `PhaseFieldProfile` at
`source/simulator/phase_field.cc:173` has the first integral

    ell^2 (phi')^2 = alpha(phi) - alpha(c) h(phi)/h(c).

On its positive support, differentiating gives

    2 ell^2 phi'' = alpha'(phi) - alpha(c) h'(phi)/h(c).

The stationary-H formula implemented at
`source/simulator/phase_field.cc:761` is

    H_star(phi) = A alpha(c) / [h(c) g(phi)^2].

Because h'=-g'/g^2, substitution cancels the phase residual exactly. There is
no sign or factor-of-two mismatch between these **untruncated positive-support
formulas**. Independent numerical evaluation gives maximum residual
2.84217e-14 Pa at the listed test points. Finite differences of the independently
integrated/inverted profile confirm its second derivative: residual errors
decrease as spacing is halved from 2e-4 to 1e-4 to 5e-5 m, reaching at most
9.87506e-6 Pa across five smooth-profile samples.

### Activation and compact support are distinct

The actual initializer does not use H_star everywhere on the positive profile.
`ReconstructedFaultManager::initialize_crack_driving_force()` at
`source/reconstructed_fault/manager.cc:412` uses the strict test phi>0.1.
Where this fails, it retains the baseline particle H=H_c=0.5 Pa, established
by the crack-driving-force property. Thus, on the intended profile,

    H_init(y) = H_star(phi(y))   if phi(y) > 0.1,
                0.5 Pa         otherwise.

The reference profile returns zero beyond its finite support; it is not
truncated to zero at the activation threshold. Its independent half support
is 0.308821593907 m, while phi=0.1 occurs at |y|=0.204000486270 m. The actual
particle data contain 3,744 activated and 5,472 baseline particles; the last
active row lies at |y|=0.200520833333 m and the first baseline row at
|y|=0.205729166667 m. This matches the documented strict activation rule.

On the still-positive but inactive tail, 0<phi<=0.1, substitution yields

    R_init(y) = [0.5 Pa - H_star(phi(y))] g'(phi(y)),

which is **not zero in general**. Representative values:

| Intended phi | H_star (Pa) | H_init (Pa) | R_init (Pa) |
| --- | ---: | ---: | ---: |
| 0.100001 | 16.8966321 | 16.8966321 | 0 to roundoff |
| 0.1, inactive side | 16.8962125 | 0.5 | +11.0751591 |
| 0.08 | 9.89284647 | 0.5 | +9.67640710 |
| 0.02 | 0.691498640 | 0.5 | +1.99609853 |
| 0.01 | 0.268899431 | 0.5 | -5.83878853 |
| phi -> 0+, inside intended support | 0.05 | 0.5 | -57.6 |

Outside support, phi=phi''=0 and the baseline gives
0.5*g'(0)+64=0 because g'(0)=-128. The support's inner limit above must not be
confused with that intact exterior. The intended compact profile has vanishing
first derivative at its endpoint, so this is not a delta-function jump in
the diffusive flux. The H activation switch also adds no derivative-of-H term
to this equation; it changes the local source term.

The production profile is additionally tabulated with 5,000 points using
trapezoidal quadrature and linear interpolation. The previous packet's
2.72e-7 relative H-reference discrepancy quantifies that tabulation effect at
the sampled particles. An exact continuum identity is not an exact identity
of a tabulated profile or of its Q1/CPDI discretization.

**Conclusion:** the stationary-H/profile formulas satisfy the same phase
equation on the positive support before the activation substitution. The
actual activated/baseline H and that intended profile do **not** form an exact
global stationary solution. This is the documented initialization rule, not
permission to change it. H initialization remains unchanged as requested.
Correcting CPDI therefore need not produce a peak or center value exactly 0.6;
the activation mismatch, tabulation and spatial error must remain distinct.
No numerical attribution of the old 0.402 peak solely to activation is made.

## 4. Comparison after correction approval

Once the ownership correction is approved and its focused tests pass, repeat
the same bounded initialization-only run: same mesh, particles, H initializer,
core input, activation, length scale, material data, constraints and solver
tolerances. Do not run mechanics or the full K1 pilot as part of that check.

Compare against the persistent baseline packet:

- CPDI weight/constant-gradient errors and discrete x-translation symmetry;
- full transverse profiles at interior and near-periodic-edge x locations;
- phi(x,0): its minimum, maximum and along-x variation;
- maximum phi and its transverse position (including whether the central
  trough and off-center maxima persist);
- maximum along-x variation at every y, initially 0.0596760807714;
- H/particle volumes unchanged, and actual phase residual meeting the same
  requested 1e-8 relative tolerance;
- actual reconstructed y(x), reported separately from phase-field error.

Keep the intended reference, activation-aware residual audit, and corrected
discrete solution separately labeled. Do not use equality to the intended
0.6 core as an acceptance condition. Other K1 gates and the runtime limit are
not presumed resolved by this correction.

## 5. Reproducibility and changed files

New diagnostic files under
`benchmarks/reconstructed_fault/uniform_shear/diagnostics/`:

- `cause_probe.gdb`: bounded inspection of the actual production polygon,
  reference-coordinate transforms and stencil; stops before phase assembly.
- `check_cpdi_cause.py`: causal replay of the three captured cases.
- `check_stationary_equation.py`: independent phase-equation/activation audit.
- `results/cause_probe{,_3983,_3986}.log`, `results/cpdi_cause.json`,
  `results/stationary_equation.json`, `results/stationary_equation_samples.csv`.

From the diagnostic build directory, the debugger command is:

```sh
timeout 180 gdb -q -batch -ex 'set $target_particle = 3836' \
  -x /home/ein/repository/aspect/benchmarks/reconstructed_fault/uniform_shear/diagnostics/cause_probe.gdb \
  --args /home/ein/repository/aspect/build-pf-cpdi/aspect ../initialization.prm
```

Repeat with target 3983 or 3986 for the saved control cases. Each completed
with exit 0 and the `CPDI_CAUSE_PROBE_COMPLETE` marker, **before any phase solve**.
Debugger process creation required permission outside the sandbox. This does
not build or alter the executable; no corrected candidate is injected.

From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 benchmarks/reconstructed_fault/uniform_shear/diagnostics/check_cpdi_cause.py
PYTHONDONTWRITEBYTECODE=1 python3 benchmarks/reconstructed_fault/uniform_shear/diagnostics/check_stationary_equation.py
```

Both checks passed. The causal replay verifies three captured cases, and the
equation check verifies cancellation, independent finite-difference convergence,
activation mismatch and intact-exterior equality. No production build was
needed; the executable and production working-tree diff remain unchanged.

**Approval requested:** the 2-D sampling-point ownership correction in section
2, with H initialization unchanged, followed by the focused tests and unchanged-
parameter initialization comparison in section 4.
