# Native finite-tip selection before the first coupled run

20 October 2026. This selects the bounded native part of the authorized bulk
cohesive stage, conditional on material-point and profile checks passing.
No tuning campaign or forced growth is authorized by a failed screen.

- Unit square, fixed uniform 16×32 Cartesian cells; native Q2 velocity/Q1
  pressure/Q1 phase. The one spatial control is 32×64 at the same physical
  ell=.15625 m. No surface association, spatial gate or extra intact damage.
- Existing compact radial prescription: core .6, original seed from (0,.5)
  to (.5,.5) m with its rounded cap. Sample the continuum at each mesh's own
  vertices; initial T/a/signed/absolute shear all zero. The seed is the physical
  obstacle, not independently relaxed on each mesh.
- G=1e6 Pa, eta=1e8 Pa s, Gc=26.6666666667 J/m², p=1 Lorentz m from native
  Kf/Hc, Hc=cohesion²/(2G), cohesion=1000 Pa. Constant mu=.1 and prescribed
  p_N=10000 Pa imply tau_r=1000 Pa, homogeneous virgin onset approximately
  2000 Pa. These are the selected bulk law's physical parameters.
- All-boundary affine simple shear ux_dot=.018*(y-.5) m/s during
  .05<t<=.15 s; zero rate before and after. Maximum imposed engineering shear
  .0018 is 90% of the homogeneous elastic peak criterion. Local concentrations
  can activate the finite seeded fault; remote activation is measured, not
  prohibited. Holding motion does not remove retained stress or Maxwell memory.
- Physical accepted target .16 s; set End time=.16 s in all cases. The native
  end-time criterion is strict greater-than and clips the last interval to
  the exact cap. Checkpoints then store the next pending clock; comparisons
  use the material's accepted time, as verified by the unloaded startup.
  Use one dt/2 control on the same mesh. Gauss6 per
  coordinate for residual/tangent/storage/physical local histories; Gauss3
  independent DGQ2 T projection. One Gauss8 control on the base mesh. This
  is a new bulk model, not BP3 or the old strip's Gauss8-panel/12 diagnostic.
- Domain 0<=phi<1 only, no constitutive cap at the initial .6 or at .99.
  A failed admissibility/solve terminates at the declared numerical limit;
  it does not authorize flooring h/g, changing the law or enlarging the run.
- Native coupled normalized mechanical residual <=1e-9, natural master-space
  phase KKT <=1e-7; at most 80 Newton steps, 30 line-search reductions. Native
  AffineConstraints define the actual admissible space. Independent DG source
  discrepancy <=1e-12 scaled by max(1000 Pa, |T|_3D).
  One native coupled tangent action at .11 s uses two one-sided Richardson
  differences (epsilon 1e-7 and 5e-8); normalized error target <=1e-6.
- Unloaded/subcritical hold: phase drift <=1e-12 and no signed/absolute shear
  before yielding. Mechanism screen: max physical phase increase ahead of
  x=.5 exceeds .001. Also distinguish any growth outside the initial compact
  support (phi_n=0) from growth within its already damaged cap. Track remote
  change at |y-.5|>.35 and x<.125; it is diagnostic, not a hidden activation gate.
- Minimal comparison screens, not quantitative propagation validation: common-
  time phase Linf <=.01 for dt halving; spatial full-tensor coarse-cell moment
  relative L2 <=10% (not a pointwise interpolation of internal variables),
  phase-growth area difference <=10%, contour-front difference <=one coarse
  tangential cell. Quadrature phase Linf <=.001 and storage/work differences
  <=.1% on max(1 J/m, reference). If these fail, retain the failure; no mesh sweep.
- Exact discrete ledger error <=1e-10 on max(1 J/m, storage/work operands).
  Separate each physical dissipation, Maxwell/phase discretization remainder,
  mechanical/pressure and phase residual work. A cumulative external-work
  remainder <=1% on max(1 J/m, accumulated boundary work), with improvement
  on dt halving, is required for a bounded physical energy qualification.
- A small loaded-state test exercises repeated fully converged trials, outer
  rejection, deterministic retry and once-only publication, followed by an
  actual simulator checkpoint and resumed continuation. One zero-time observer
  rebuild selects an appended diagnostic interval and inherits moments from
  bulk histories without a P3 T correction or physical birth reset.

This selection does not assume that the chosen load will pass the mechanism
screen. If it does not, report the actual response rather than raise the load
or modify the initialization to manufacture advance.
