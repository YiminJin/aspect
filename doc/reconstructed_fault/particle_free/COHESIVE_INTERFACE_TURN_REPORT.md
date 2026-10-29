# Closed-contact turning and bounded direction selection

**Sustained sliding through a smooth turn is demonstrated. A computed deflection
is demonstrated within an explicit candidate rule; a geometry-independent
propagation angle is not qualified.** The turn-width control changes the best
terminal angle. No sharp-corner constraint, constitutive tolerance, radiation
term or production/BP3 behavior is removed.

[Figures, GIF and indexed data](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_turn/INDEX.md)
· [formulation and predeclared checks](COHESIVE_INTERFACE_TURN_SELECTION.md).
Baseline: `f1f9f29cf40c32d1ddcea000b724375b64853532`, branch
`pf-rsf-particle-free`, with the entry local work preserved.

## Mechanics and the added path hypothesis

The earlier sharp corner remains a control: continuous vector traces satisfying
zero normal jump on two distinct planes force zero vector jump at the corner.
This task instead uses a resolved C2 graph, with tangential jump
`[u]=t(x) δ(x)` and **both** vector displacement components in the native mixed
Q2/P1disc mechanics. The symmetric lifting is
`u=u_cont+sign(y-g(x)) t(x) Σφ_j δ_j/2`.
Its gradient includes `t⊗∇φ_j + φ_j t′⊗e_x`; omitting the second term would
change stress, divergence, pressure and work. A Q4 geometry mapping represents
the graph exactly. Normal contact is enforced kinematically, not dropped or
penalized. All tensor contractions retain full 3D plane strain.

The 12×12 m **finite** block has old Γ from (3,0) to (6,0), and a prospective
append ending at x=9 m. Over physical turn width w,
`g′=tan(α)(3z²−2z³)`, `z=(x−6)/w`, then the slope is constant.
The primary w=1.5 m and alternative w=3 m are uncalibrated geometric choices,
not material regularization lengths. Beyond the Γ endpoint, graph flattening
only completes the bulk mesh. There is no extra fault there.

The unchanged surface law uses `e=δ−b`,
`ψ_b=(1−a)e²/(2C0 a)`, `C0=2Gc/τp²`, `δf=2Gc/τp`, and irreversible
`a=max(a_old,min(|e|/δf,1))`. Tied records have zero slip increment and an
admissible weak reaction bounded by ±τp. Sliding resistance is
`q+a[μ(|V|,Θ) pn sign(V)+ηrad V]`, with the existing RSF/state update,
`V=Δδ/Δt`. **Current** `pn±=p0+p′±−n·τ±n`; the law uses their mean and
both traces must remain compressive. The numerical mean-zero p′ reference is
distinct from physical p0=10000 Pa. No pressure lag or fitted pressure shift.

The additional direction rule previews the same loading from the same incoming
state on separate candidate continuations and maximizes
`S(α)=∫new a ds / length_new` at 0.45 s. Initially α=−60:20:60°;
the leading sector gets 10° and 30° probes. This is **finite-horizon formation
selection**, not Coulomb-angle intact initiation, global potential minimization
or an instantaneous universal crack-direction law. The candidate family,
width, horizon and normalization are extra assumptions. Losing previews never
contribute fracture energy or histories to the accepted branch.

## Histories, loading and discretization

G=1e6 Pa, η=1e8 Pa·s, Gc=1 J/m², τp=3000 Pa, Dc=.001 m and all RSF/radiation
parameters remain unchanged. Prescribed affine strain is γE0,
`E0_xy=1/2`, `E0_xx=−.4195498156`, `E0_yy=+.4195498156`.
The selection experiment ramps γ=.0003→.0021 at .004/s over .45 s. Its nominal
far-field maximum shear is 2741 Pa; the loading is below remote intact strength.
There is no distant-boundary interpretation. The mature control ramps from
zero to .0022 over .55 s and holds displacement for .025 s.
That control initializes the whole curved Γ as already mature; it does not
rotate or reinterpret a previously developed straight zone.

The old loaded segment is prepared in mechanical equilibrium: nonzero permanent
slip, `a=.75+.05·4z(1−z)`, nonuniform age, and nonzero Cartesian Maxwell stress.
The reference slip b gives q=300 Pa on that old segment. This is a declared
prepared state, not a claimed prior loading trajectory. New records are tied,
with zero slip/damage/reference/relaxed memory/absolute slip and virgin Θ=1.
Zero-time insertion preserves every old history bitwise. Initial physical stress
changes ≤1.49e−11 Pa in the qualified selection cases; bulk storage discrepancy
≤5.88e−12 J/m. New bond/fracture storage and clock advance are zero.

Bulk assembly remains Gauss6, with independent Gauss8 physical reconstruction;
surface Gauss3 records own all six histories. The qualified increment uses the
previously tested **Q1 jump trace**, an exact Galerkin restriction of the full
native vector operator. Bulk Q2/P1disc is retained. The original full-Q2
permanent jump is an affine offset, not projected away. Arbitrary old elastic
modes cannot be transferred to a smaller space under this result.
The production DGQ2 history backend and its settings are untouched.

The 12 m bulk scaling is an exact homogeneous mechanical similarity:
`D=diag(12,1,…)`, `K12=D K1 D`, stress/pressure response `F12=F1 D/12`.
Surface/volume measures scale by 12/144; physical constitutive parameters and
histories do not. This reuses actual curved vector FE operators, not the
straight-strip kernel. All nonlinear native states are solved freshly.

## Measured results

| Mature turn control | ∫|Δslip| ds [m²] | Minimum late through-turn rate [mm/s] | Minimum contact [Pa] |
|---|---:|---:|---:|
| 64×32, w=1.5 m, Δt=.00025 s | .01281245 | 3.843 | 7181 |
| 128×64, same width/step | .01292174 | 3.876 | 7053 |
| 128×64, Δt/2 | .01292305 | 3.878 | 7053 |
| 128×64, w=3 m | .01445230 | 5.106 | 7024 |

Spatial slip change is .85%; temporal change .0101%. Width changes slip by
11.8%, but sustained sliding survives both widths. Thus the existence of
through-turn sliding is robust here, while its amount depends on the geometry.
Maximum `|κδ|=.00221`: this is small-strain contact on a fixed curved reference,
**not** finite-slip contact-point transport or the zero-radius corner limit.

| Primary candidate at .45 s | Coarse score | Fine score | Fine Δt/2 score |
|---|---:|---:|---:|
| +20° | .309518 | .304387 | .301930 |
| +30° | .345702 | .342977 | .341488 |

The half-step score gap .039558 exceeds the predeclared twice-combined observed
spatial/temporal uncertainty .023603. +30° is distinguished **in this sampled
primary family**. For it, spatial/temporal new-damage-integral changes are
.794%/.436%; slip-increment changes .554%/.155%. Common physical cell averages
give spatial append-damage/slip L2 differences .751%/.888%. No exact continuous
optimal angle or convergence order is inferred.

At the fine half step, the +30° branch has 18 connected whole cells with mean
a≥.05 (1.804–1.913 m arclength front bracket), and **four complete new cells
with all Gauss histories a=1**, totaling .37533 m arclength/.375 m projected
advance. There is no detached a≥.05 cell. The fully decohered front has only
turned approximately 5°; the damage-weighted new-surface tangent is 12.27°.
The prospective terminal 30° must not be called the current crack/front angle.

At w=3 m the best sampled terminal angle is +60°, at the sector boundary;
refinement changes its score .253%. Its damage-weighted tangent is 9.67°.
Positive deflection is supported by these controls, but the terminal angle is
**not independent of the prescribed turning geometry**. The wider family was
not searched beyond its declared bounds. At 80% of the preview interval the
response is still tiny: selection depends on a resolved finite preview, not
first-point damage. The original +20° trial approaches its candidate endpoint
at .45475 s; data beyond the common .45 s window remain diagnostic only.

Reflected geometry/loading reverses signed slip and swaps pressure sides:
damage error 8.34e−14, signed-slip error 5.57e−17 m, pressure error 3.49e−10 Pa.
Reflected bulk cell-mean full-tensor/pressure errors are 2.81e−10/4.86e−10 Pa.
The straight and negative-angle controls remain separate competing specimens.

## Work, numerical checks and lifecycle

For the selected half-step branch, initial bulk/cohesive/fracture storage is
11.04249/.109094/2.35 J/m thickness. Changes are 520.05464/.122370/1.759637 J/m;
friction/radiation/Maxwell losses are 3.605525/.903311/1.874860 J/m.
Trapezoidal external work is 528.31519 J/m. The sum of **absolute per-step**
physical remainders is .005153 J/m, .1143% of sliding and .2928% of added
fracture; it halves with Δt. The signed remainder is −.005153 J/m. No net
cancellation is used to pass. Mature-turn absolute work error is .00586% of
sliding, also including its hold. Fracture is counted once as Gc∫a ds;
old inventory is retained, and extra fracture includes further old damage.
`energy-accounting.csv` separates all terms. The exact backward-Euler ledger
(≤1.59e−12 J/m in these controls) is a separate algebraic check, not physical
energy qualification.

Independent Gauss8 bulk Gram versus Gauss6 condensed energy differs <1e−13
relative. Lift-gradient FD error ≤1.28e−10. Gauss3 versus Gauss12 geometric
mass and uniform-stress weak patch integrals differ ≤9.05e−8 relative; this
does not independently re-quadrature evolved nonlinear histories. Native
point-return/reference discrepancy is 6.78e−21 scaled; complete current-pressure
coupled derivative checks are <1e−8 at the unchanged 1e−6 gate. Signed weak
residuals remain ≤1e−8 Pa. Selected total normal gap ≤4.61e−18 m, normal-work
error ≤1.52e−16 J/m. Contact remains positive. The retained symmetric-tangent
gate is an admissible algebraic check, not a general non-associated RSF theorem.

Discarded geometry trials leave accepted state unchanged; retry is bitwise.
Wrong geometry and duplicate geometry/physical publication are rejected.
JSON restart with an actually destroyed/rebuilt native context gives the same
next state bitwise. A freshly computed accepted +30° evolution reproduces its
preview bitwise: one geometry publication, 1800 physical publications for 1800
steps. **These are isolated benchmark owners/native mixed operators and the
existing C++ signed solver, not production Simulator/Manager integration.**

The archived sharp default is numerically preserved: stiffness difference
2.17e−16 relative and declared physical full-tensor/pressure probe difference
2.59e−12 Pa, within the existing 1e−7 Pa check. Its initial bitwise expectation
failed and is recorded; point-law data, geometry and Q remain bitwise. A rebuilt
smooth case is bitwise in every exported field/operator.

Native 128×64 setup/export takes ~100–104 s per geometry; fresh fine/half-step
selection evolution takes 27/52 s, mature fine/half 22/44 s, sequential with one
BLAS thread. The selected lifecycle/recomputation uses 242 MiB peak process RSS;
native setup peak RSS was not measured. Raw operators/output occupy ~2.6 GiB,
exceeding the initial 1 GiB estimate; compact evidence excludes large raw tables.
All completed pilots/controls total 516.5 s of evolution and 754.8 s of native
kernel setup/export, sequentially; failed partial runs, compilation and
postprocessing are excluded. [Changed files](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_turn/changed-files.json)
and [commands](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_turn/COMMANDS.md)
identify the exact experimental footprint.

## Limits retained and next decision

The undersized unit pilots do not resolve new formation. Their near-zero-fracture
relative-work failures and the full-Q2 first-admission native line-search failure
at .2515 s remain preserved and unresolved; Q1 is an explicit discrete choice,
not a repaired Q2 solver. The historical 65.4% early spatial failure, first-tip
admission and finite-boundary findings remain. No tolerance was relaxed.

This closes the bounded turning/selection experiment. A robust physical angle
requires a justified **geometry/path-admission length and horizon**, or another
explicit path law, before treating this preview closure as propagation physics.
Generic old Maxwell-mode remapping, finite sliding along curved material contact,
production ownership, free networks and corners remain separate tasks. Existing
damaged geometry never rotates. No inertia, rupture-speed claim, bulk phase
evolution, RSF recalibration, or automatic further run is introduced.
