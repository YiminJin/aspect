# Frozen locally split Q2 stress comparison

**The improvement justifies a bounded coupled direct-stress prototype.**
At 64², resolving the two finite localization-end lines with independent local
Q2 pieces reduces constant/layered tensor L2 error by **78.36× / 77.93×**, for
**4.05% more stress coefficients**. Constant-viscosity work-traction error falls
from **0.31725% to 0.00025134%**. This is frozen approximation evidence, not an
implemented coupled option or a qualification of moving-cut transport.

[Protocol](FROZEN_SPLIT_Q2_PROTOCOL.md) ·
[Evidence/reproduction](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/frozen_split_q2/README.md) ·
[Figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/frozen_split_q2/frozen-split-q2.png)

## Scope and unchanged reference

Entry verified root `aspect-pf-rsf-particle-free`, branch `pf-rsf-particle-free`,
revision `cf06525471ce89789f1ac10905dc79f3aac7562a` and a clean tracked worktree.
Existing untracked research/manual files are retained. Native source, CMake,
working binaries and all 63 protected artifacts were hashed and remain unchanged.
There was no native build/run, rotation recheck, general refactor, checkpoint
conversion or server submission. Python/BLAS used one thread.

The [preceding report](FROZEN_FINITE_END_REPORT.md) established the uncut
approximation limit. This experiment loads its complete saved
`frozen-input.npz`, not a reinitialized or remapped field: the same 64² Q1 phase,
18 labels and I, full-normal support, G=10 Pa, m=50, dt=.1 s, zero incoming
stress, D=(.03,−.03,.01) s⁻¹ and V_j=.002(1+.2 cos X_j.x) m/s. The viscosity
laws remain 10 Pa s and 10(1+.5 tanh(4x)) Pa s. The existing evaluator supplies

$$
 f(x)=2\kappa(x)[D-\chi(x)V(x)S],\qquad
 \kappa=-\eta\operatorname{expm1}(-G\Delta t/\eta).
$$

The source still terminates at the original finite curve ends. Neither smoothing
nor tangential extension is introduced. These are mature localization ends,
**not cohesive physical tips**. Production BP3 and cohesive-interface histories
are unaffected and carry no new qualification from this experiment.

The same 41,600-triangle common 128² integration partition resolves the fixed
Q1 cells, tensor cells, label/end lines and diagnostic windows. Duffy Gauss6/10
supplies the convergence check. The retained uncut cut-integrated coefficients,
source norms and errors are reproduced rather than another q8→q24 sweep.
The old best-space uncut result is the comparator throughout; thus the gain is
not merely better quadrature applied to an underintegrated control.

## Implemented approximation

For each candidate Cartesian tensor cell K, intersect it with the three
half-plane regions separated by the two lines normal to the fault tangent
through its finite ends. Only crossed cells gain pieces. On every nonempty
piece P store an independent restriction of **Cartesian Q2**, nine coefficients
per tensor component. For this fixture each crossed cell has two pieces; no cell
contains both end lines. Internal label, Q1 and integration-triangle boundaries
add quadrature points but **no history degrees of freedom**.

$$
 V_h^{split}|_K=\bigoplus_{P\subset K}Q_2|_P,\qquad
 \int_P N_i(\tau_h-f)\,dA=0,\qquad M_P a_P=b_P.
$$

This space contains the original uncut Q2 restriction. It changes approximation
freedom, not the physical source. Coordinates are scaled separately in x/y to
each polygon's bounding box. This is a basis change within Cartesian Q2, not a
rotation to a different polynomial space. Whole-cell mass remains diagonal in
the original Legendre basis; cut pieces use full 9×9 mass matrices, diagonal
equilibration and Cholesky solves. No positive-area piece or mode is dropped,
merged or regularized. The full admitted normal lines are split even where phase
is zero; cost is not reduced using a source-amplitude threshold.

Implementation is benchmark-only:
[compare_frozen_split_q2.py](../../../benchmarks/reconstructed_fault/particle_free/transported_objective/compare_frozen_split_q2.py),
with a saved-result [side-limit audit](../../../benchmarks/reconstructed_fault/particle_free/transported_objective/audit_frozen_split_q2.py)
and [plotter](../../../benchmarks/reconstructed_fault/particle_free/transported_objective/plot_frozen_split_q2.py).
The native `TensorField` and solver interfaces are unchanged.

## Approximation accuracy

Tensor L2 errors use full 3-D plane strain and have units Pa m. Both complete
source and localized-source denominators are retained in `comparison.json`.
The complete source norms are .34085116/.33644409 Pa m; localized-source norms
are .00554305/.00549068 Pa m (constant/layered).

| Tensor grid | Uncut best L2, constant | Split L2, constant | Gain | Split L2, layered | Layered gain | Extra history |
|---|---:|---:|---:|---:|---:|---:|
| 16² | 5.40920e−4 | 8.81317e−5 | 6.14× | 8.92563e−5 | 5.98× | 15.625% |
| 32² | 4.10764e−4 | 3.12071e−5 | 13.16× | 3.10028e−5 | 13.05× | 8.008% |
| 64² | 2.88812e−4 | 3.68589e−6 | 78.36× | 3.65101e−6 | 77.93× | 4.053% |
| 128² | 2.04254e−4 | 1.29993e−6 | 157.13× | 1.28523e−6 | 156.57× | 2.039% |

At 64² the split error is **0.066496% of the localized source** for both laws,
versus about 5.21% uncut, and .001081%/.001085% of the complete source. Even
16² split has lower bulk error than 128² uncut, while storing 7,992 rather than
442,368 scalar stress coefficients; this cross-grid observation uses the same
64² physical profile, not a coarsened input.

The central window remains .3 m inside both ends with |normal|<.4 m. End windows
extend .3 m to either side of each end with |normal|<.6 m. Constant-eta central
errors are unchanged: 7.76983e−5, 2.66600e−5, 1.98866e−6, 6.83626e−7 Pa m.
Split end errors are 3.99527e−5, 1.59555e−5, 3.10327e−6, 1.10561e−6 Pa m;
end-error gains are 13.40×, 25.69×, 93.06×, 184.74×. At 64², split component
errors (xx,yy,xy,zz) are (1.43467e−6,1.43467e−6,2.17591e−6,0) Pa m.
Layered component/window values are retained alongside constant values.

Non-cut cells agree with their retained uncut polynomials to at most
2.93e−15 Pa m in aggregate. Thus the improvement is localized to the new history
freedom. Split one-halving global orders are 1.498, 3.082, 1.504 for constant
eta, with similar layered values. These mixed orders **do not establish a
uniform third-order convergence claim**: the same piecewise Q1 profile and
piecewise-linear surface data still have internal derivative changes that were
not added as representation cuts.

## Traction, weak work and energy

Point traction is integrated along the same finite curve, with cell/label/window
splits. Work traction uses the **same converged surface mass** for source and
both approximations: M_ij=∫chi N_i N_j dA and b_i=∫chi N_i(tau:S) dA. Regional
work errors partition the tangential coordinate and integrate the full normal
profile; they are distinct from restricted-normal bulk windows.

| Grid | Point error uncut → split, constant | Work error uncut → split, constant |
|---|---:|---:|
| 16² | 1.46630% → .389519% | 1.03112% → .0105806% |
| 32² | 1.64240% → .160899% | .637178% → .00230312% |
| 64² | .936337% → .0206369% | .317249% → .000251336% |
| 128² | .579369% → .00671604% | .158823% → .0000320235% |

At 64², layered point/work errors are .0206097%/.000250061%. Constant split
point errors are .0144979% centrally and .0291120% at the ends; corresponding
work errors are .0000170312% and .000425697%. Work-traction inversion couples
surface nodal values, so work error can improve centrally even though central
bulk stress coefficients are unchanged.

The one-sided jump audit explicitly selects inside/outside polynomials at the
same endpoint, avoiding the old measure-zero ownership diagnostic pitfall.
For constant eta the signed relative jump errors are +1.8781%, +.053588%,
−.013479%, −.0005419% with refinement; layered values are similar. The source's
nonzero jump is represented rather than spread over one uncut cell, but its
one-sided amplitude still has approximation error.

Constant-eta signed slip-power errors are +4.08102e−9, +5.11693e−10,
+7.13819e−12, +8.87862e−13 W/m; divided by the negative reference power
−7.73678e−5 W/m, they are −.00527483%, −.000661378%, −.0000092263%,
−.0000011476%. Layered signed errors and ratios are saved separately. These
are weak forcing-power differences, not frictional dissipation claims.

For constant kappa, orthogonality and the constant strain forcing imply
`delta(slip power)=||f−split||²/(2*kappa)`; the saved results satisfy this to
1.23e−19 W/m. Thus the signed power and projection-energy gains are related
consequences of the same projection, not independent proofs of coupled energy
accuracy. This constant-coefficient identity is not imposed on the layered law.

The signed converged-measure projection energy losses E(f)−E(split), in J/m,
are 1.94180e−10, 2.43470e−11, 3.39646e−13, 4.22477e−14 (constant) and
1.99167e−10, 2.40294e−11, 3.33248e−13, 4.12966e−14 (layered). They agree
with ∫|f−split|²/(4G) to 2.05e−18 J/m. At 64² the uncut constant loss was
2.08531e−9 J/m. No trajectory, boundary flux or time-integration energy balance
is being tested here.

Normalization is unchanged: ∫chi dA=1.67048942 m versus length1.67044904 m,
a +.0024172% difference. The new small work-traction error is relative to the
**same retained I/localization**, not a claim that this normalization discrepancy
or every physical traction error has disappeared.

## Numerical qualification and cost

All predeclared numerical screens pass without changing thresholds:

- Gauss6→10 split-field L2 change ≤2.67e−14 Pa m; work traction change
  ≤1.95e−16 Pa. The retained uncut-reference bridge is ≤2.94e−15 Pa m.
- Independent Green-theorem polygon moments check every cut 9×9 mass matrix:
  maximum absolute difference 9.39e−16. Piece areas close to 9.03e−17 m².
- A Cartesian Q2 polynomial including x²y² is projected and evaluated at
  independent bounding-box points; maximum error 1.93e−11, below 1e−9.
- The nested-space identity ||uncut−f||²=||split−f||²+||split−uncut||² closes
  to 2.15e−21 Pa² m² using direct residual integration.
- The smallest piece area fractions are .00024, .00096, .00384 and .000026667.
  Bounding-box-basis mass condition numbers are ≤8679.23 raw and ≤2877.84
  after diagonal equilibration. These are basis/mass results, not stability
  bounds for explicit advection through a small moving cut cell.

| Grid | Extra pieces | Uncut → split scalar stress coefficients | Extra stress bytes | Optional cut full-mass cache bytes |
|---|---:|---:|---:|---:|
| 16² | 40 | 6,912 → 7,992 | 8,640 | 51,840 |
| 32² | 82 | 27,648 → 29,862 | 17,712 | 106,272 |
| 64² | 166 | 110,592 → 115,074 | 35,856 | 215,136 |
| 128² | 334 | 442,368 → 451,386 | 72,144 | 432,864 |

Counts are per **one physical tensor** (three stored components, inferred zz),
not doubled for the two viscosity experiments. A cached dense mass or factor
needs 81 doubles per cut piece; the table budgets one block per piece, excluding
geometry/MPI metadata and solve scratch. Independent T/R storage would require
separate history vectors; it is not the representation evaluated here.

The full comparison took43.45 s on one thread: common partition1.03 s,
independent error/work evaluation26.09 s. At64², split assembly+solve took
1.3211 s versus1.2797 s for uncut; at128², 1.3276 versus1.2890 s. Split geometry
construction was .203/.788 s respectively. These matched passes integrate both
laws and the polynomial check on the same4.16 million reference points. They
measure additional frozen projection work, **not native solver speed**. The
reference oversamples coarse grids; practical coupled quadrature, communication,
metadata and factor reuse have not been benchmarked. No speedup is claimed.

## Decision and next justified experiment

**Yes: the end/traction gain at modest history cost merits a coupled prototype.**
Keep it opt-in and initially direct-stress only. The next bounded experiment
should freeze this same geometry/profile and introduce the split space into
coupled constant/layered endpoint mechanics, retaining the existing law and
projection-correction iteration. Use the identical material-side polynomial in
bulk stress, surface work traction, residual observations and the final candidate;
compare converged stress/slip/work and iteration cost with the retained uncut
control and this frozen reference. A finite-end cut is a representation boundary,
not a new traction boundary condition or cohesive tip law.

Before a moving-fault qualification, explicitly address cuts moving across cells,
transfer of complete accepted stress, matching work integration, small-piece
transport/stability, partition identity and joint rejection/restart. A successful
frozen mass solve supplies none of those by itself. Do not introduce damping,
mode truncation, source smoothing or constitutive changes as hidden fixes.

Implemented: benchmark split approximation and independent comparison. Locally
verified: frozen mass/source/traction/energy identities and storage cost.
Physically compared: the selected common frozen source only. Coupled/native
implementation and production readiness: **not established**. Broader Stage C,
energy/flux, AMR/birth, boundary traction and independent cohesive tip/energy
failures remain open. No unavailable historical data were recreated: the
39/45 native-pilot-frame and temporary-log losses remain explicit; production
BP3 and retained RSF evidence are preserved.
