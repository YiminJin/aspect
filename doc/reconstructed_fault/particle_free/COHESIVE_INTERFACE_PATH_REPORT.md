# Causal selected extensions: bounded result

Baseline `f1f9f29cf40c32d1ddcea000b724375b64853532` plus preserved local work,
branch `pf-rsf-particle-free`. [Figures, animation and indexed data](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_path/INDEX.md) · [equations, predeclared gates and retained corrections](COHESIVE_INTERFACE_PATH_SELECTION.md).

**A short sequence of computed extensions and subsequent sliding/damage is
demonstrated. Quantitative spatial acceptance fails.** The adopted causal rule
is an explicit formation/persistence hypothesis, not a final propagation law.
This is an isolated native deal.II mixed operator and C++ nonlinear evolution
solver with benchmark JSON ownership, not production Simulator integration.
Production/BP3, the point law, solver tolerances and previous results are unchanged.

## Angles and the physical choice

Tangents are counterclockwise from +x; planes are modulo 180°, with
`n=(-sin θ,cos θ)`. The old tangent is 0°. For the imposed homogeneous strain,
`τ=Gγ[[-2b,1],[1,2b]]`, `b=tan(40°)/2`, the principal compression axis is −25°
and the conjugate maximum-shear tangents are **+20° and −70°**. The μ=.1
constant-Coulomb diagnostic gives +17.1447°/−67.1447°; it is not intact admission.
Isotropic confinement changes normal pressure, not these principal directions.
Analytic reversal, coordinate-basis rotation and full-circle sampling checks pass.

The retained bond energy is `(1-a)e²/(2C0 a)`, `C0=2Gc/peak²` and
`δf=2Gc/peak`. Minimizing virgin bond storage plus `Gc a` gives
`peak|e|−peak e²/(2δf)`. Hence its first formation variation is
`|t·σn|−peak`. Damage-weighted RSF/radiation contribute at higher order in
virgin slip. Maximum incipient shear work is therefore consistent with this
unchanged pressure-independent intact strength. A Coulomb formation angle would
require a different intact law and energy calibration; none is inserted here.

The additional hypothesis averages the **accepted physical tensor** with
`W=(1-r²/R²)²` in a disk of radius R, centered R ahead of the accepted geometric
endpoint. Choose an analytic maximum-shear plane, preferring smaller rotation
from the existing tangent between work-equivalent conjugates. Exact ties and
undefined zero shear are reported unresolved. There is no restricted-angle
search. A graph-incompatible result stops the calculation.

New, still-tied geometry obeys `dθ/ds=(θ*−θ)/R`. Here **R=.75 m is an
uncalibrated physical coherence/persistence length**, also used as the tied
reserve. It is 6.75 times `G Gc/peak²`; that ratio is an exploratory choice,
not a derivation or calibration. Reusing R for sampling, persistence and reserve,
freezing admitted tied directions, and extending only the right tip are explicit
assumptions. No future damage endpoint is optimized. No bending storage is added;
the single fracture inventory remains `Gc∫Γa ds`. This is a causal local path
preference, not a proof of global free-path optimality for RSF/contact.

## Representation, contact and history

The 12×12 m confined finite block has old Γ `(3,0)→(6,0)`. The fixed background
mixed FE matrix is independent of Γ. Nested tangential jump functions include
the full tangent derivative; new functions extend the representation without
rewriting any old function. Zero-extending Maxwell coefficients preserves
the entire old discrete stress/pressure span, not only one prepared state.
Arbitrary coefficients in that retained span are covered; independent tensor
fields outside this experimental representation are not qualified by the check.
The numerical lifting width `12/n` is not a physical band width.

At every insertion, old native response hashes, full physical stress/pressure
columns and old six Gauss histories are bitwise unchanged. Actual evolved
nonuniform Maxwell memory reaches 1.13 mm. Measured zero-time stress, storage
and old contact-load changes are zero. New records have zero signed slip,
damage, cohesive reference, Maxwell slip memory and accumulated absolute slip;
their dimensionless virgin age is `1+V0 t/Dc`. Signed weak tied admission is
checked before publication. No fracture energy or physical time is created at birth.

Two numerical defects were diagnosed, with original source/data retained:

- Forcing zero curvature at every appended segment caused about 20% extension-step
  sensitivity. Hermite slope interpolation now matches the persistence equation's
  endpoint curvatures. Independent arclength-weighted relative curvature errors
  fall from .679% to .114% to .0159% on successive step halvings. The earlier
  sampled estimates are also retained. Old geometry stays unchanged;
  normals are continuous, with bounded curvature jumps when the driver changes.
- A cut-cell raw point pressure cannot represent the correct sided contact
  reaction: even a flat interface gives `[pn]=(2G/3)dm/ds`. Use a closed normal
  virtual bubble `Ln=Hρ4z(1-z)n` with amplitude constrained to zero and a DG0
  multiplier. Elimination gives
  `λ=p0+∫bulk[τ:dev3(sym∇Ln)−p′divLn]/Mn`, `Mn=−∫bulk divLn`.
  This uses the actual bulk-work measure, including normal derivatives and the
  mean-zero pressure correction. Its full current-pressure derivative enters
  the nonsymmetric tangent. It changes the experimental contact discretization,
  not the law: no opening, dilation, pressure lag, clipping or prescribed friction
  pressure. Fresh `work*` runs select it explicitly; old point-pressure histories
  are never converted. Normal work is zero.

Against a separately solved fitted curved reference at matching physical Gauss
points and bubble functional, weak-normal perturbation RMS differences are
.84–.94% on nearly equal shifted/aligned grids. Independent bulk-work and coupled
finite-difference errors are `2.49e−14` and `2.87e−9`. Raw point tractions remain
unfiltered: final paired RMS mismatch is **226.5 Pa** in work96. This residual
point-trace defect is not relabelled equilibrium or hidden by the multiplier.

## Measured evolution and uncertainty

G=1e6 Pa, η=1e8 Pa s, Gc=1 J/m², peak=3000 Pa, p0=10000 Pa,
μ0=.1, Dc=.001 m and all RSF/radiation coefficients are unchanged. Each mesh
samples the same prepared quadratic seed afresh. Initial bulk/bond/fracture
inventories are 11.04249/.109094/2.35 J/m. The primary initial physical tensor
error is `1.40e−12 Pa`, weak residual `1.89e−13 Pa`, and the cohesive obstacle
ratio is .5000. The initial state is loaded and admissible.

Load at γ̇=.004/s through .44 s, then hold displacement through .445 s;
Δt=.00025 s, surface spacing .09375 m. Four .75 m insertions occur: two reserve
prefills at .15 s, then growth-triggered decisions at .431 and .44225 s.
Accepted preferences are 20.000°, 20.000°, 20.148° and 20.244°.
The prospective terminal tangent is 19.822°. It is **not** the front angle.

At .445 s, work96 has connected fronts:

| Surface damage | Position (x,y), m | Actual tangent |
|---|---:|---:|
| a=.05 | (7.66994,.35985) | 18.007° |
| a=.5 | (6.86839,.12583) | 13.812° |
| a=1 | (6.24106,.01228) | 5.541° |

The fully decohered advance is .24147 m along Γ, including two complete new
surface cells. The cohesive region between a=1 and .05 spans 1.4733 m, about
16 surface cells. New `∫a ds=.920301 m`; no detached threshold regions occur
on the admitted right path. At the loading endpoint, full-front x advance is
.04909 m; it continues during the hold under retained stress/history, reaching
.24106 m. Peak final slip rate is .03169 m/s. This is quasi-static evolution,
not validated physical rupture speed or a ban on hold growth.

Common-time comparisons use growth-relative integrals and actual fronts:

| Control | New-damage difference | Slip-increment difference | a=1 front difference |
|---|---:|---:|---:|
| 64²→96² Q2 bulk | .089689 m / **9.75%** | 2.28% | .08057 m |
| Δs .75→.375 m | .002909 m / .317% | .0782% | .00185 m |
| 64² Q3 versus 96² Q2 | .159703 m / **17.35%** | 3.83% | **.10764 m** |
| Late Δt halving, .42025→.445 s | .002662 m / .323% of window growth | .173% | .00120 m |
| Surface spacing halving | .000894 m / .0973% | .106% | .000265 m |

The predeclared 5% growth target fails for both bulk controls; the alternative
also exceeds one fine surface cell in full-front position. Process-front bulk
differences are .09584/.18504 m. Tangent differences stay below 1.95°, but this
does not establish precise path accuracy. Q3 has 97,537 bulk masters versus
100,609 for 96² Q2, providing an informative alternative approximation rather
than an accuracy oracle. The current rule computes analytic planes, eliminating
angular search error; retained full-circle 7° sampling selects 22° and changes
the older trajectory materially. Coarse angular labels are not reliable evidence.

## Work, lifecycle and conclusion

Primary changes/work in J/m: bulk storage +500.45570, bond +.164067,
fracture +1.547719 (including .627418 in the old seed), friction 3.340769,
radiation .754560 and Maxwell loss 1.822152. Trapezoidal work is 508.077902.
The sum of absolute physical remainders is .007064 J/m: **.1725% of sliding
dissipation and .4564% of added fracture**, below the declared 1% criterion.
All remainders have the same sign, so net closure hides no cancellation.
The exact backward-Euler ledger closes separately to `9.21e−13 J/m` maximum.
Hold absolute remainder falls from .002127 to .001059 J/m on time halving.

All accepted true weak residuals are below 1e−8 Pa; gaps below 5.5e−20 m,
compression above 8.12 kPa, and complete coupled FD errors below 3e−9.
The retained restricted symmetric-tangent diagnostic is positive; this is not
a general stability/potential certificate for the nonsymmetric RSF problem.
An actual artificial native contact-loss trial is rejected before publication;
accepted geometry/history remain unchanged. Deterministic geometry retry,
stale/duplicate rejection, exactly one commit and rebuilt-context JSON restart
are verified, including nonzero evolved memory. Physical runs publish 1780
times; geometry events have separate zero-time counters.

Sequential evolution/controller wall times are 104, 237, 356, 222, 139 and 599 s
(64², 96², half extension, Q3, late half time, half surface). Parent high-water
RSS reaches 512 MiB; native peak memory was not separately measured. These times
include newly needed extension operators but exclude initial-kernel construction;
cached operators are reused. They are not cold-start or solver-speedup measures. Raw caches
are indexed separately; compact source/data/figures are archived with checksums.

**Qualified:** causal selection without future-endpoint scoring, successive
closed-contact extensions, history-preserving loaded insertion, and a short
computed deflection with actual developed advance. **Unresolved:** quantitative
bulk/contact spatial accuracy, raw point traction continuity, calibration of R
and the tied-reserve/persistence assumptions, competing free paths and general
network/finite-sliding mechanics. Historical 65.4% early spatial sensitivity,
full-Q2 jump-admission failure and earlier branch/energy findings remain intact.
The smallest useful next task is a fixed-path mixed bulk/contact approximation
check at a saved evolved state to explain the Q2/Q3 spread before another path
campaign; physical calibration of the nonlocal rule is a separate decision.
No additional run or production adoption starts automatically.
