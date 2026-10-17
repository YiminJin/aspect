# PF6a bounded fixed-geometry spatial diagnostic

Selected 16 October 2026 at a8707307008401ac9cff702a369e2e2a611fa0f7
plus preserved PF6a/WENO/manual work. Unlimited DGQ2, constitutive equations,
normalization, physical bounds, production defaults and limiters stay unchanged.
No AMR/MPI, transport, moving surface, birth or propagation is selected.

Two different questions are measured separately:

1. Install identical physical Q1 phase, complete DGQ2 tensor and Q1 Z from the
   quadrature-resolved 32-normal-cell PF6a snapshot on nested 32/64 meshes.
   Evaluate them on identical physical points, with the same energy functional.
   This checks evaluation/representation only, not mechanical resolution.
2. Freeze that same physical phase and the preceding incoming T/Z. Re-solve
   native mechanics at the original endpoint clock/rate on 32/64 normal cells.
   Reconstruct the equivalent continuous Q2 accepted FE displacement history
   from the complete saved native velocity recurrence, checking its tensor
   against the actual DG history first. Compare DG/FE at matching fields using
   the common ordinary-FE fracture diagnostic; evaluate native CPDI fracture
   separately on the FE path. No particle-authoritative history comparison or
   checkpoint/AMR conversion is inferred from these diagnostic imports.

Use the existing `propagation/localization_1d.py::Reference` and
`normal_resolution.py` derivative/moment tools with supplied physical fields,
not their historical initialization or trajectories. In the uniform shear
reduction, T_e=beta T_n+kappa u_y, Z_e=beta Z_n+kappa V and
chi=h(phi)/I. If P denotes cellwise L2 projection onto normal P1 derivatives
of Q2 velocity, equilibrium implies

```
a=beta T_n; r_a=(1-P)a; r_chi=(1-P)chi; C=integral chi; L=1
T_e=P a+kappa u_y+r_a
mean_tau=(integral a+kappa U-Z_e C)/L
physical_tau=mean_tau+r_a-Z_e r_chi
q+mu Pcompression=integral chi*physical_tau  (sliding)
```

When incoming T_n lies in this derivative space, r_a=0: physical cell-scale
variation is the localization projection remainder, not additional DG transport
modes. Solve the existing 1D matrix equations and check this explanation against
native full mechanics, including mean/load effects. Also retain a constant-stress
continuum mechanical reference with the same incoming fields/profile/loading;
it is not an evolving reference or full 2D qualification.

Export T_xy, Z chi, actual reconstructed tau_xy, all nine Q2 cell Legendre
moments of these terms and one-sided interface jumps. Full stress norms include
zz=-(xx+yy) and 2xy². Report whole-domain and chi-weighted errors on common
points, total/bulk/cohesive energy and phase gradients pulled back to the common
32-cell Q1 test space; never compare raw coefficient gradients on different
meshes. Common fields/functional must be distinguished from actual retained
FE+CPDI equations. Native particle transfer is not used as a substitute reference.

Initial checks: common-field tensor/stress/energy/gradient agreement <=1e-9 on
fixed scales (1000 Pa for stress; max(1,reference) for energy/gradient), native
versus 1D mechanics <=1e-6 on those scales, immutable incoming histories and
no publication. Independent quadrature 24/48 and native 12/24 checks must be
small compared with measured spatial effects; no tolerance relaxation. Expect
approximately second-order stress projection error for this piecewise-smooth
profile on aligned nested cells, but measure rather than assume it.

Only after frozen 32/64 diagnostics, use one finer normal mesh (128). A short
native evolving event is conditional on the frozen checks identifying a concrete
error that refinement should reduce. If justified, initialize a common imported
physical entry at .53125 s, then use the actual unchanged native endpoint loop
through .5625 s (four dt=1/128 steps, with a bounded half-step control). Such an
imported common-entry experiment is not a production restart/transfer or the
original uninterrupted fine-grid trajectory. Keep timestep/quadrature differences
below spatial effects and retain irreversibility/acceptance checks. Stop after
this diagnostic and report remaining limitations.
