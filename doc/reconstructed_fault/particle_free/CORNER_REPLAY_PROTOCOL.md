# Frozen non-affine corner velocity replay — 5 November 2026

Selected follow-up to CORNER_ISOLATION_REPORT.md. Root is the existing
`aspect-pf-rsf-particle-free`, branch `pf-rsf-particle-free`, revision
`cf06525471ce89789f1ac10905dc79f3aac7562a`, with seven pre-existing dirty
tracked documentation files and retained untracked experiments. Entry hashes
are in `benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/corner_replay/entry.json`.
No native source, physical parameters, checkpoints or existing binaries change.

Freeze the complete continuous Q2 velocity from `shear32-dt025`, step16,
t=.4 s. Reconstruct each cell from all nine saved Gauss3 values. Check the
saved gradients independently, shared-face continuity, and interpolation into
native velocity DoFs. Require velocity error <1e-12 m/s and gradient error
<1e-10 /s. Use the same 32² Cartesian box, ordinary DGQ2 and native distributed
SSPRK3, q8 operator quadrature. No new mechanics trajectory or rotation check.

Let b(y)=.4+.5y, f(x,y,t)=1+.15(x-bt)+.1y+.08(x-bt)y and
H*=f A, with (Axx,Ayy,Axy)=(1,-.3,.45), Azz=-.7. This is exactly the
prior affine pure-transport reference, a smooth nonzero exterior field, and
is Q2 in space and linear in time. For the frozen saved velocity v prescribe
S=[(vx-b)(.15+.08y)+vy(.1+.08x-t(.5(.15+.08y)+.08b))] A.
Thus ∂tH*+v·∇H*=S exactly. S is a manufactured transport source, **not**
Maxwell forcing or a change to the physical stress law. Geometry, fault history,
mechanics and constitutive reaction remain outside this scratch experiment.

Run t=0… .4 s, identical initial H*, at macro dt=.1,.05,.025 s. Compare:
0. Exact H*(x,t) at each SSPRK stage.
1. Exact H*(x,macro-parent time), held during that macro step (lag only).
2. Numerical parent trace, held during that macro step (lag plus feedback).

Choose the .1-s subdivision count as the next multiple of four satisfying
δt inverse_advection_time <= .1, and use the **same δt in all nine cases**.
This holds RK error fixed while varying the refresh interval. Native CFL ceiling
.125 is unchanged. Exact linear-time/Q2 manufactured history should be preserved
within 1e-9 relative full-plane-strain L2 error; no pass threshold is imposed on
the held boundary's model error.

Measure whole-box, upper-left .5×.5 m corner and |x|,|y|<1.5 m interior errors,
peak full-plane-strain stress norm and corner share of squared error. With
ψ=H:H/(4G), G=10, verify at every substep endpoint

    dE/dt + F_supplied + D_upwind - ∫ψ div(v) - ∫H:S/(2G) = 0.

Use native q8 face signs, include internal jumps and incoming-boundary penalty.
Require identity defect <1e-10 max(1,E). Also retain interior-trace boundary flux,
signed accumulated and accumulated-absolute trapezoidal balance remainders.
The velocity is weakly, not pointwise, incompressible. Omitting divergence work
would test a different transport equation. Independent exact-solution screens
and FE-velocity checks accompany the algebraic balance.

One initialization-only four-rank run, timeout600s; build at most two workers.
Preserve all prior outputs and failed attempts. The checker must leave accepted
native solution and vertices unchanged; it owns no physical history pointer.
No ownership/checkpoint change calls for a new production restart. Earlier
39/45 native-pilot frame and temporary-log losses remain unrepaired.

Interpretation: this isolates response of the transport/inflow operator under
one saved problematic velocity. It cannot reproduce time-dependent feedback
through a mechanics solve, select physical exterior loading, qualify the original
coupled shear problem, or replace the separate cohesive/BP3 evidence.
