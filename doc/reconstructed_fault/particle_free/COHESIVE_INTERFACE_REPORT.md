# Intact fracture compliance → geometric-interface RSF: small-reference result

**The separated formulation passes intact-initiation and mature-sliding checks,
and produces a small seed-connected cohesive-zone extension. It is ready for a
bounded native interface-realization test, not a general propagation run.**
The first-activation time and near-tip peak are still spatially sensitive; no
fully decohered extension occurred in the bounded seeded test.

This is a new experimental **surface cohesive-damage law**, not a repair that
pretends to retain every old bulk phase equation. Production/BP3, native source,
checkpoints and prior evidence are preserved at `f1f9f29cf40c32d1ddcea000b724375b64853532`
plus the pre-existing local work. [Equations and history contract](COHESIVE_INTERFACE_FORMULATION.md),
[predeclared selection](COHESIVE_INTERFACE_SELECTION.md),
[downloadable archive](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface.zip), and
[indexed figures/data](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface/INDEX.md).

## Physical changes and intact calibration

Recoverable bonds and frictional contact act **in parallel** on one geometric
jump. Damage a creates bond compliance `c=c0 a/(1−a)`; bond storage is
`(δ−b)²/(2c)`. Fracture inventory is `Gc a` on the surface, counted once.
The contact contribution is `a R(δdot,pn,Θ)`, with actual solved normal pressure,
one interface age, and the retained RSF/radiation coefficients. At a=1 bond
traction and storage vanish exactly; the equation is geometric-interface RSF.
At a=0 the interface is exactly tied, with one-sided admission at |s|=3000 Pa.
There is no phase floor, width-dependent rate, residual spring, or history reset.

The 3000 Pa peak is an explicit reference calibration: the old 2000 Pa strength
drop plus .1×10000 Pa nominal friction. Gc=1 J/m² gives δf=.666667 mm and
c0=2.222222e−7 m/Pa. G=1e6 Pa, η=1e8 Pa s and RSF parameters are retained.
The old m=4 bulk degradation is **not** retained as an additional surface energy;
ell=.046875 m is available for a Γ geometric lift, not an extra fracture charge.
This is an intentional constitutive difference, not a numerical error.

Linear contact activation a, the prospective material age clock, and fixed
reference peak strength are explicit assumptions. New-plane formation maximizes
recoverable compliance release; existing structural planes retain their direction.
A declared geometric preference resolves conjugate ties, with unresolved symmetry
reported when no preference is supplied. Formation is not constrained by the old
maximum-frictional-speed direction rule. Opening, junctions and secondary fracture
through an existing record remain outside this closed-contact reference.

## Measured checks

| Check | Result |
|---|---|
| Exact intact onset | analytic 0.375704888 s; brackets [0.374,0.376], [0.375,0.376], [0.3755,0.376] s for Δt=.002/.001/.0005 s |
| Intact → complete decohesion | exactly 1 J/m² added fracture; final bond storage exactly zero |
| Formation physical-work accuracy | sum of absolute step remainders / Gc: 0.19329%, 0.09664%, 0.04832%; endpoint BE ledger retained separately |
| Mature sliding | 7.26104 Dc cumulative absolute slip through velocity step, hold and reversal; zero new fracture and zero bond storage |
| Same-law interface identity | maximum traction difference 7.85e−9 Pa; force allowance remains 1e−8 Pa |
| Mature physical-work accuracy | absolute remainders / friction+radiation: 0.01706% → 0.008553% under halving |
| Complete pressure-coupled tangent | centered-FD relative errors 1.426e−6, 3.566e−7, 8.919e−8 with refined FD step; fixed 1e−6 target passed |
| Actual pressure coupling | 360.10 Pa pressure range in the nonuniform tangent test |
| Mixed-layer independent integration | strain-energy / boundary-work mismatch ≤4.58e−14 relative; Gauss24/48 comparison; pn=p0+p′−τnn identity error ≤2.63e−11 Pa |
| Direction / symmetry | basis covariance error 2.22e−16; reversal retains the material plane; no-preference tie remains unresolved |
| Mature mixed symbol | full pressure/state modes from wavelength 2 m to 6.28 μm decay; maximum Re λ≈−1.0000001/s for tested V=.001 m/s |
| Owner lifecycle | discarded trial leaves accepted state unchanged; retry and JSON restart bitwise identical; duplicate publication rejected |

At common 0.01 s output times, formation-damage differences halve from
1.369e−5 to 6.853e−6; shear differences decrease from .6265 to .3266 Pa.
Mature timestep halving changes peak-scaled traction by .2219% and slip by
.00604%. These response checks are separate from the energy ledger.

The mature symbol is for a fixed geometric plane, not an arbitrary changing-
direction bulk law. The normal-pressure Schur term is retained. Its large-k
limit is regular here: elastic stiffness grows as G|k|, the normal cross-symbol
vanishes, and the state branch tends to −V/Dc. Physical finite-wavelength RSF
instability remains possible in other regimes. No old symmetric phase Hessian
is reused.

### Loaded history conversion is conditional, not a reset

The test retains full T,A,K and reconstructs τ=T−2GA with error 4.55e−13 Pa.
It preserves geometric slip, cohesive traction, fracture inventory and all
non-proportional tensor residuals. A compatible loaded example has switch-work
error 4.16e−17 J/m². A generic loaded example requires **−0.006578916 J/m²**
explicit recoverable energy extraction. A zero-work conversion of that example
is rejected. The old matched-Maxwell residual-storage ledger closes to
6.69e−19 J/m²; no new damping coefficient is introduced. Tensor serialization
retains every component. The mapping is an instantaneous reference, not a
qualified production-checkpoint conversion. Nonuniform old cross-band ages
have no uniquely equivalent single RSF age; they are not silently averaged.

## Small seeded test

A 2 m periodic straight candidate interface separates clamped layers of heights
.075/.05 m. The mature seed is |x|<.125 m; all remaining path points are exactly
intact and tied. Fresh histories have zero stress/slip and Θ=1. Initial fracture
inventory is .25 J/m, excluded from added-fracture work. Uniform relative wall
speed .001 m/s acts to .30 s, followed by a hold to .32 s. All candidate points
can form: no tip mask, artificial continuation or initial damaged cap is used.
Different layer heights generate actual normal-pressure feedback and a small
left/right asymmetry. This is a finite periodic specimen, not a distant-boundary
approximation or a free-path 2-D phase-field calculation.

| Run | First new cell bracket (s) | Final ∫Δa ds (m) | Peak new a | Runtime (s) |
|---|---|---:|---:|---:|
| N=128, Δt=.002 s | [.228,.230] | .00232923 | .047480 | 0.60 |
| N=256, Δt=.002 s | [.208,.210] | .00241453 | .056972 | 1.49 |
| N=256, Δt=.001 s | [.208,.209] | .00242107 | .057072 | 2.73 |
| N=512, Δt=.001 s | [.192,.193] | .00244257 | .062923 | 9.66 |

Runtimes are local wall times for the final reference implementation, measured
sequentially with one BLAS thread; exact values are in JSON. They are not native
ASPECT or full-event performance predictions.

At N=512 the a=.01 connected front reaches x=−.1640625/.16015625 m and a=.05
reaches −.1328125/.12890625 m, beyond the original ±.125 m endpoints. The a=.5
and .95 fronts remain at the original endpoints. New damage is therefore
**cohesive-zone extension, not a fully developed crack advance**. No remote
activation occurs for |x|>.5 m. Actual pressure is 9248.09–10749.68 Pa; mean
shear is 2298.99 Pa. Added fracture, bond storage and sliding loss are
.00244257, .00235242 and .02817664 J/m respectively.

Timestep halving at N=256 changes integrated new damage by **0.2793%**, slip by
0.1360% and mean shear by 0.00522%. The N=128→256 difference in growth is
**3.942%**, failing the original 2% spatial target. One focused refinement gives
N=256→512 differences **1.0296% growth**, 0.1597% slip and 0.00671% mean shear.
All recorded front-position differences satisfy the declared two-fine-cell
allowance. The original failed coarse comparison remains in the CSV.

These integral results do **not** qualify onset or peak damage: the first-cell
bracket shifts by 16 ms between the last two meshes, and peak new damage changes
by about 9.3%. A sharp pre-existing mature tip has a singular incipient traction
field before its cohesive zone develops; a first discrete-cell threshold is
therefore particularly resolution-sensitive. That explains the trend but is
not a substitute for a qualified physical onset definition. No convergence
order or propagation speed is claimed.

The finest sum of absolute physical-work remainders is 1.2663e−5 J/m:
0.04136% of formation+sliding, **0.5184% of added fracture**, and 0.04494% of
sliding loss. The BE ledger defect stays below 5.30e−15 J/m. Bulk BE and signed
cohesive endpoint remainders remain separate; they are not called physical
losses. The tested free-interface symmetric Jacobian part stays positive
(minimum 3.95e7 Pa/m on the fine run). This establishes a local incremental
check, not a universal phase-active stability theorem. The fine owner publishes
320 times for 320 accepted steps; rejection/retry and saved-state restart are
bitwise identical.

## Source corrections, preserved failures and reproducibility

Only new independent Python references and their documentation/data were added.
An early absolute-jump Newton variable lost digits in tiny reversal increments
and failed the 1e−8 Pa force check at 1.103e−8 Pa. Solving for Δδ directly fixes
the demonstrated arithmetic cause without changing its tolerance. A coarse
finite-difference step exceeded the derivative target; FD-step refinement
exposes the expected second-order error and passes the unchanged target.
A compatibility-root search initially straddled an old compact-support cell
transition; narrowing to a smooth fixed-support interval gives the actual
compatible root .8672300331 and is not a physical parameter tuning campaign.
At N=512 separate column FFTs exceeded an absolute reciprocity diagnostic;
constructing the analytically even/odd circulant kernels enforces reciprocity.
The raw-to-structured change is at most 1.20e−7 Pa/m in K (roundoff relative to
its scale), independently checked against an operand-scale roundoff bound.
No physical acceptance tolerance is relaxed.

Run from the worktree root with one BLAS thread:

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/bulk_cohesive/check_cohesive_interface.py
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/bulk_cohesive/check_cohesive_admission.py
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/bulk_cohesive/run_cohesive_interface.py
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/bulk_cohesive/plot_cohesive_interface.py
```

The full equation/parameter table, small reference code, PNG/GIF, checkpoint
samples and compact CSV/JSON evidence are indexed and archived. Historical
failed spatial/energy studies remain intact.

## Recommendation and remaining physical assumptions

Proceed only to a small fixed-geometry native Γ realization: demonstrate the
same tied intact constraint, geometric jump/contact law, once-only surface
fracture inventory, physical trace/work adjoint, complete pressure tangent and
atomic history publication. The present sharp mixed-layer solution is its
reference, **not** proof that a finite-width native Γ discretization is already
equivalent. That implementation must explicitly separate bulk Maxwell strain
from interface compliance rather than double-counting them in a diffuse lift.

Keep the linear contact-activation rule, fixed-reference strength, material age
clock and surface damage interpretation visible for scientific review. The new
model has no independent physical diffuse-band width; ell can set geometric
regularization, whose numerical equivalence is unqualified. General direction
birth/junctions, loss of contact, arbitrary old cross-band-age conversion,
fully developed tip advance and quantitative onset remain open. No native
propagation, BP3 change or automatic larger campaign is authorized by this report.

Changed-file inventory: five new scripts in
`benchmarks/reconstructed_fault/particle_free/bulk_cohesive/`
(`cohesive_interface.py`, `check_cohesive_interface.py`,
`check_cohesive_admission.py`, `run_cohesive_interface.py`,
`plot_cohesive_interface.py`); `COHESIVE_INTERFACE_SELECTION.md`,
`COHESIVE_INTERFACE_FORMULATION.md`, this report; additive entries in
`STATUS.md`, `SOURCE_AUDIT.md`, `../CURRENT_STATUS.md`; the indexed
`evidence/cohesive_interface/` data and archive. **116 protected files were
rehashed unchanged**; all three previous index contents remain intact as suffixes.
No commit was created.
