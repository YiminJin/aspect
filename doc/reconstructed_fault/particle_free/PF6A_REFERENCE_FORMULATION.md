# Matched PF6a evolving spatial-reference study

Selected 16 October 2026 at a8707307008401ac9cff702a369e2e2a611fa0f7 plus
preserved incoming work. This task is serial, fixed geometry/grid, no transport,
through t=.5625 s. Unlimited DGQ2, native quadrature 12, material/loading laws,
phase irreversibility and native acceptance rules are fixed. No imported loaded
coarse state is a main reference. No AMR/MPI, propagation or limiter work.

## Physical initialization and equations

Every trajectory interpolates the SAME supplied physical table
`benchmarks/reconstructed_fault/particle_free/pf6a-initial-phase.txt` at its
phase nodes. This is the documented pre-existing damage prescription; do not
rerun mesh-dependent seeded-H relaxation. Its piecewise linear table is the
physical data for this study, not an analytic infinite-resolution seed.
Incoming trial tensor, surface Z and accumulated slip start at zero. Natural
phase boundary, unchanged ramp u_x=.256 t (y-.5), exact interval-mean rates,
G=1e6, eta=1e8, mu P=1000, ell=.15625, AT1 ordinary FE fracture, h as retained.
Physical accepted phase is the lower obstacle on every endpoint trial.
Report seed Q1 interpolation L2/Linf/gradient-energy errors separately; their
effect on the subsequent obstacle is not removed by subtracting initial offsets.

Extend the inexpensive retained 1D mechanical/reference equations with the
same consistent-endpoint phase partial and reduced Newton tangent. The old
split `Reference.run` is not used. For constant coefficients/uniform shear,
write P for the normal-cell L2 projection onto P1 derivatives of Q2 velocity,
C=integral chi, D=integral (chi-P chi)^2, B=beta integral Tn+kappa U,
H=C^2+D+1/I, f=mu Pcompression. On unit height,

```
Ze = max(beta Zn, (C B-f)/H), V=(Ze-beta Zn)/kappa
lambda = B-Ze C, Te=lambda+Ze P chi, tau=lambda-Ze(chi-P chi)
Ebulk=(lambda^2+Ze^2 D)/(2G), Ecoh=Ze^2/(2G I)
J=Ebulk+Ecoh+F_FE+(f/G)(Ze-beta Zn).
```

This is algebraic elimination of the retained equations for compatible unloaded
histories, not a new constitutive law. Constant-stress continuum mechanics
uses P=Id, D=0 on the same phase space. Retain sampled C and its derivatives,
normalization derivatives, f in the reduced potential and fixed-Te/Ze phase
partial. No frozen-stress phase gradient. Derivative/tangent and mechanical
matrix equivalence checks precede trajectories. No artificial symmetry lock.

Use mechanical/phase meshes 64/64,128/128,256/256, and continuum mechanics at
phase 64,128,256,512,1024. Compare projected versus continuum at identical phase
resolution first; refine the continuum phase/obstacle separately. Check the
previous frozen physical profile at mechanical 256 against its existing
constant-stress reference and measure the actual evolving remainder throughout.
The 256 candidate is selected by measured remainder, not the extrapolation.
A bounded temporal control uses dt/2, and if needed dt/4 only in the inexpensive
model. Normal quadrature 12 is compared with 24 independently in that model.

## Practical targets, declared before native confirmation

Measure at every common accepted time, with onset defined by phase growth
>1e-5 above the mesh's initial seed. Use fixed reference PEAK scales over
0--.5625 s, not startup denominators. Full tensor shear errors include sqrt(2).

- Mechanical resolution: whole full-tensor L2 <=1% of reference peak tensor
  stress; localization-weighted full-tensor RMS <=2% on the same scale.
- Phase shape: relative L2 <=1% and absolute Linf <=.005; phase growth onset
  difference <=one selected timestep. Phase-space 512/1024 uncertainty must be
  reported separately, not assumed zero.
- Cohesive traction q, final-scale cumulative slip and total energy: each <=1%.
- Phase driving: bulk+cohesive energy derivatives, pulled back to common 64-cell
  Q1 tests, differ <=2% on the fixed peak driving scale; also report all terms,
  mass-weighted driving shapes and phase-gradient measures. Raw gradients on
  different meshes are never compared as identical vectors.
- Temporal/quadrature uncertainty: <=one fifth of each applicable practical
  target and smaller than material measured spatial differences. For tiny or
  vanishing differences report absolute uncertainty rather than claiming an
  error hierarchy. Targets are not native solver acceptance tolerances.
- Native versus matching 1D projected candidate: physical fields/q/slip/energy
  <=1e-6 scaled; independent native KKT/source/load/quadrature checks retain
  prior PF6a thresholds. No tolerance relaxation to rescue a failure.

Record all target misses. The native confirmation tests implementation of the
selected resolution even if a spatial target fails; it does not upgrade failed
accuracy. Run exactly one fresh native trajectory, t=0--.5625, selected normal
mesh and timestep, unchanged existing observer/core, capped at 900 s serial and
one thread. Full state/phase/stress/load diagnostics cover onset and growth.
Stop after measured results and recommendation; no automatic finer native run.

## Bounded candidate extension after the 256 diagnostic

Before native confirmation: the frozen 256 prediction is verified, but its
matched growing trajectory reaches 1.176% whole / 2.439% localized error and
misses the unchanged 1%/2% mechanical targets. Add one 512/512 projected 1D
candidate and its half-step control; no parameter search. Select the single
native resolution from these measured results. Native serial guards are kept
despite permission to use up to eight processes; reference jobs use at most four
independent processes. The 900-second native cap remains fixed.

Preliminary reference failures are retained: an incorrect enlarged-space
stability assertion, then negative translation curvature, then insufficient
reference branch-search convergence. Use the retained positive spectral SEARCH
metric and admissible energy-decreasing translation departure; exact physical
energy, Hessian, KKT and irreversibility remain unchanged. Do not freeze global
reflection symmetry or silently accept a translational saddle.

## Source-origin correction to finer phase initialization

The literal-table controls exposed an unloaded Q1 driving defect at 1024:
refining a fixed piecewise-linear 512-node interpolant is not sampling the
original documented continuum seed at new nodes. Source audit confirms every
native table entry is EXACTLY the retained `matched_initialization` continuum
BVP value at that coordinate. The BVP file is existing evidence, not a newly
relaxed or fitted seed. The main phase-refinement reference therefore evaluates
that SAME retained definition at each mesh's nodes. At all meshes <=512,
including the selected native input, this changes no initial value at all.
Only the finer 1024-node reference changes its between-table-knot sampling.
Report interpolation error against that physical continuum definition. Preserve
all literal-table trajectories/onset failures as a separate diagnostic; do not
silently substitute them for the main reference or rewrite the original decision.
Thresholds, constitutive equations, unloaded T/Z, irreversibility and the single
native run remain unchanged. The failed native publication is independent of
this correction and is not rerun within this task.
