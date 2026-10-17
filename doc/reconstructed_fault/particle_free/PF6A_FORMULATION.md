# PF6a: fixed-reference-surface cohesive endpoint experiment

Selected 16 October 2026, starting a8707307008401ac9cff702a369e2e2a611fa0f7
plus the preserved pre-PF7 diagnostic work. This is a bounded serial fixed-grid,
fixed-surface, no-advection experiment. Unlimited DGQ2 remains the baseline;
no limiter, production default, rate/spin convention or physical law changes.
The derivation is actually at `../particle_free_history.tex`, not the stale
`particle_free/particle_free_history.tex` path in AGENTS.

The retained P2 consistent-endpoint/reduced-Newton design governs the coupled
solve. PF3 supplies the ordinary FE fracture functional; this differs discretely
from retained CPDI. Combining phase-first and mechanical observers is not the
algorithm. The previous split proposal in the derivation is not adopted here.

## Ownership and discrete equations

Accepted at t_n: independent DGQ2 T_n (Pa), Q1 surface Z_n=G s_elas,n,
I_n, physical Q1 phase p_n (also the immutable lower obstacle), committed V,
slip/manager/particle histories and native solution/checkpoint bookkeeping.
No bulk/surface transport, changing surface labels, geometry, mesh or spin.
For interval-constant solved rates u,V and dt>0:

```
beta=exp(-G dt/eta), kappa=-eta expm1(-G dt/eta),
T_e=beta T_n+2 kappa D3(symgrad u), Z_e=beta Z_n+kappa V,
I(p)=M_surface^-1 B_surface b(p), b_r(p)=integral_ray h(p_h),
chi=h(p_h)/I(p), tau_e=T_e-2 chi Z_e S, q_e=Z_e/I(p).
```

The native Q2 source belongs exactly to DGQ2 on this affine Cartesian grid;
Gauss3 mass/source projection retains all nine modes. T_e/Z_e are scratch
endpoints reconstructed from n on EVERY trial. They never replace incoming
mechanical history. Established full 3D metric is
`|A|3²=xx²+yy²+(xx+yy)²+2xy²`. S_xy=sign/2.

Solve the existing endpoint mechanical rows (velocity/pressure and constrained
surface friction rates), with the current trial p and physical stress tau_e:

```
R_u(v)=sum_bulk w tau_e:D3(symgrad v)-sum_bulk w pressure div(v)-external(v),
R_p(q)=-sum_bulk w q div(u),
R_V=Q1 weak(q_e+mu P-L_p^* tau_e),
```

Use the retained prescribed-rate/sticking complementarity and prescribed
compression convention. No external shear offset, radiation, variable friction,
body force or RSF state is added. The native coupling/constraints/normal stress
rows, including their scales and signs, remain authoritative.

At each mechanical endpoint define

```
E(p;T_e,Z_e)=sum_bulk w |tau_e|3²/(4G)
            +sum_surface w Z_e²/(2G I(p))+F_FE(p),
F_FE=sum_bulk w Gc/(c0 ell) [p_h+ell² |grad p_h|²].
```

The phase variation is the PARTIAL derivative at fixed endpoint T_e,Z_e,
including all normalization derivatives and its consistent projection adjoint.
The reduced phase gradient evaluates this partial derivative AFTER equilibrating
u,pressure,V at that p. Equivalently, stationarity follows from endpoint stored
energy plus `(kappa/G) integral mu P V`; that friction potential contributes to
the reduced line-search energy, not directly to the fixed-T_e,Z_e phase partial.
No peak H drives PF6a after initialization. Accepted p_n <= p <1; the upper
endpoint p=1 is undefined for this rational active-profile energy and rejected,
not capped. Positive finite I is required at every evaluation.

## Consistent tangent and acceptance

For a phase direction d, dI=P Db[d] and
`dchi=h' B d/I-h dI/I²`. The existing partial mechanical forcing is

```
rhs_u=E^T W (2 Z_e dchi S), rhs_pressure=0,
rhs_surface=T^T W dchi (tau_e:S-2 Z_e chi S:S)
            +N^T Ws Z_e dI/I².
```

Solve the current full constrained mechanical Jacobian for du,dp,dV, then
`dT_e=2 kappa D3(symgrad du)`, `dZ_e=kappa dV`, and
`dtau_e=dT_e-2(dZ_e chi+Z_e dchi)S`.
Differentiate the phase partial including both `DI^T df_I` and
`(D DI[d])^T f_I`; add the exact FE fracture Hessian. This is the reduced
consistent tangent. Source projection and quadrature are fixed during each
finite-difference check and nonlinear solve. Existing active-set Newton,
true-energy line search, final KKT and stability checks are reused unchanged.

Acceptance publishes T_e,Z_e,I(p_e),p_e, derived particle samples and manager
slip once through existing candidate validation/noexcept swaps. Rejection of
any trial or the actual outer solve restores accepted histories, geometry and
bookkeeping; a retry starts from the same n. No observer substitutes a manual
sequence of phase and mechanical commits. Restart reuses native infrastructure
with a same-mode endpoint payload; PF5 prescribed-mode conversion is rejected.

## Verification frozen before native runs

First independently check fixed-endpoint energy derivatives and combined
phase/mechanical directional derivatives: best central-difference relative errors
<=1e-6, independent integration <=1e-9, symmetry where applicable <=1e-10.
Retain original mechanical/KKT/energy acceptance tolerances. Compare frozen-phase
FE/DG source, reconstructed stress and weak loads. Include nonzero T and Z and
measurable physical phase growth; a residual pass without growth is insufficient.

Run short native outer-loop controls with new unique labels, 180 seconds per
serial one-thread process initially (at most two concurrent), and save failures.
Use dt,dt/2,dt/4 at common final time; compare one bounded finer normal mesh
separately from quadrature at fixed fields and from timestep changes. A quadrature
screen requires relative energy/I/weak-load changes <=1e-6 at fixed fields;
report remaining spatial/temporal errors without claiming asymptotic convergence.
No tolerance relaxation or compensating dissipation. Test repeated trials,
controlled prepublication outer failure/retry, one publication and actual native
filesystem restart against uninterrupted advancement.

Energy accounting must keep physical endpoint storage/fracture, exact Maxwell
interval work/viscous integrals at frozen final profile, finite phase/profile
redistribution, friction, free mechanical/contact/phase residual work and endpoint
finite-step discrepancy distinct. Endpoint equilibrium does not make finite-step
work close exactly; report that discrepancy and its timestep sensitivity.

This does not complete PF5 transfer/MPI requirements, moving geometry, birth,
PF7 or general propagation qualification. A scientific/design conflict stops the
affected implementation and is reported while independent checks continue.

## Native lifecycle audit findings (after the frozen formulation)

The actual outer timestep manager evaluates its repeat decision after
`solve_timestep` returns. Therefore the DG endpoint must remain staged through
that decision; acceptance publishes before postprocessing, and cutback discards
it while restoring entry bookkeeping. The real `repeat on cutback` fixture uses
an exact interval-mean ramp and compares with the identical accepted schedule.
A prepublication exception alone does not verify this second rejection point.

Automatic native snapshots follow `advance_time`, so the saved native clock is
one step ahead of accepted constitutive history. PF6a payload version 2 records
both clocks and requires the canonical accepted solution/phase anchor and one
publication per accepted step. Existing PF5 at-endpoint version 1 is unchanged.
Trial snapshots and cross-mode conversion are rejected; there is no conversion
rule inferred from matching dimensions.

For work verification the implemented independent diagnostic uses linear phase
within each interval with exact exponential T/Z and fixed solved rates. The
chain rule includes changing chi/I, pressure/divergence, free mechanical,
contact and partial-phase work. This is a diagnostic interpolation of endpoint
data, not a split update or an assertion of equilibrium throughout the interval.
The external-balance remainder and individual absolute work contributions are
reported separately. See PF6A_REPORT.md for measured temporal and quadrature
uncertainties and the unresolved normal spatial sensitivity.

Deferred publication validation is covered separately: deliberately change the
trial solution after staging, require context rejection before history writes,
and restore entry bookkeeping before retry. The regression fails on the earlier
pending-publication wrapper and passes with its exception rollback. This changes
no accepted physical update or limiter/production setting.
