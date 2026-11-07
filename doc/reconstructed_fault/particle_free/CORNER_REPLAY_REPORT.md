# Frozen non-affine corner velocity replay — 5 November 2026

The selected replay is complete. Native DGQ2/SSPRK3 transports a manufactured
stress under the saved problematic non-affine velocity to roundoff with exact
exterior data. Holding the exact exterior over each macro step gives first-order
lag error. Feeding back the numerical parent trace leaves a substantial boundary
bias that barely decreases with the macro timestep, but does **not** reproduce
the severe coupled stress amplification in this control.

[Protocol](CORNER_REPLAY_PROTOCOL.md) ·
[Evidence and commands](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/corner_replay/README.md) ·
[Figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/corner_replay/corner-replay.png)

## Starting state and scope

Root `/home/ein/repository/aspect-pf-rsf-particle-free`, branch
`pf-rsf-particle-free`, revision `cf06525471ce89789f1ac10905dc79f3aac7562a`.
Seven pre-existing tracked documentation edits and the untracked scientific
experiments were preserved. Entry/final provenance verifies 994 prior files,
including the six retained native/checker binaries, 63 protected artifacts,
original shear data, and the previous corner plugin, reports, failed attempts
and results. Identity comes from file hashes, not HEAD or embedded build banners.
No native source, default, checkpoint, constitutive convention, quadrature,
convergence criterion or existing binary changed.

This is the specific follow-up proposed in CORNER_ISOLATION_REPORT.md. It uses
one saved velocity, one mesh and nine short scratch histories, not another coupled
refinement campaign. The host performs its usual fixture initialization only;
`material-summary.csv` has only step0 with zero history/geometry publications.
No material step, checkpoint resume, new loading selection, stress-rotation check,
server job or production run occurred. No global refactor was performed.

## Reference and controls

The replay velocity is the final accepted continuous Q2 field from
`shear32-dt025`, step16 at t=.4 s. This is a frozen snapshot, not a replay of the
sequence of accepted-parent velocities used in that trajectory. Nine saved
Gauss3 values reconstruct each cell polynomial. Independent saved-gradient and
shared-face checks precede interpolation into native velocity DoFs.

| Replay check | Maximum error or magnitude |
|---|---:|
| Saved velocity reconstruction | 6.66e-16 m/s |
| Derivative versus saved velocity gradient | 4.26e-14 /s |
| Shared-face velocity jump, 17 points per face | 2.89e-15 m/s |
| Native FE versus reconstructed velocity, Gauss8 | 8.22e-15 m/s |
| Native FE versus reconstructed gradient, Gauss8 | 3.18e-13 /s |
| Saved Gauss3 speed magnitude | 1.82390 m/s |
| Departure from (.4+.5y,0), saved Gauss3 | .742531 m/s |
| Absolute pointwise divergence, saved Gauss3 | 21.6509 /s |

These are sampled maxima, not certified continuum extrema. Weak incompressibility
does not set the last quantity to zero.

Use exactly the earlier affine pure-advection reference:

    b(y) = .4 + .5 y
    f(x,y,t) = 1 + .15(x-bt) + .1y + .08(x-bt)y
    H* = f A,  (Axx,Ayy,Axy) = (1,-.3,.45),  Azz = -.7.

Under saved v, prescribe S = ∂tH* + v·∇H*, as expanded in the protocol.
The scratch equation is the existing material transport equation
∂tH+v·∇H=S. S is a manufactured transport load; it is **not** Maxwell forcing,
a replacement constitutive law, or the physical forcing of the saved fault.
H* is Q2 in space, linear in time, and explicitly continued outside the box.
G=10 only sets the diagnostic energy scale. No rotation/reaction occurs in these
scratch histories; the physical solver's stress rate remains untouched.

Same 32² mesh, initial H*, q8 native operator and 0… .4 s duration in every case.
Macro dt=.1/.05/.025 s gives 4/8/16 steps. The native speed/h bound is
27.638246 /s; a common RK substep δt=.1/28=.00357142857 s gives CFL=.0987080,
below the unchanged .125 ceiling. There are 28/14/7 substeps per macro step,
112 per case. This deliberately fixes RK error while changing only boundary
refresh; it is not a temporal convergence study of the full physical solver.

- **Exact stage exterior:** H*(x,t) at each RK stage.
- **Exact parent exterior:** H*(x,macro-parent time), held during that macro step.
  This isolates temporal lag without numerical-trace feedback.
- **Numerical parent trace:** previous numerical boundary polynomial, held during
  that macro step, exactly the retained accepted-trace interpretation.

## Measured errors

Full 3D plane-strain tensor L2 metric, including Hzz=-Hxx-Hyy. Corner is
x<-1.5,y>1.5; interior is |x|,|y|<1.5. Table errors are **percent**, at t=.4 s.

| Exterior | Macro dt (s) | Whole box (%) | Corner (%) | Interior (%) | Peak tensor norm (Pa) |
|---|---:|---:|---:|---:|---:|
| Exact parent | .1 | .322629 | 3.67140 | .00835979 | 2.35070 |
| Exact parent | .05 | .156365 | 1.77140 | .00341764 | 2.33372 |
| Exact parent | .025 | .0772238 | .872956 | .00168670 | 2.32531 |
| Numerical parent | .1 | 1.40409 | 16.6736 | .00385059 | 2.39170 |
| Numerical parent | .05 | 1.39063 | 16.6668 | .00370404 | 2.36147 |
| Numerical parent | .025 | 1.36781 | 16.4660 | .00274282 | 2.34150 |

Exact stage exterior gives relative whole-box error <=9.16e-15 at the final
states (all stepwise screens <1e-9), corner <=5.60e-15, and final peak2.31686 Pa.
Its total energy agrees with the retained independent affine reference histories
within 7.0e-15 J/m, despite the different velocity/source. The exact-parent
whole-box orders are 1.045 and 1.018. Numerical-parent error decreases only 2.58%
from the largest to smallest macro step; it remains a boundary-model error.
The corner contains 64.28% of its squared global error at dt=.025, not the
99.86% observed in the original coupled temporal difference.

The retained affine held-trace control had 1.38819% box and 16.8755% corner error
at dt=.025. The frozen non-affine result is comparable (1.36781%,16.4660%),
although its manufactured source differs and the old RK substep was larger.
Thus the large non-affine gradients in this snapshot do not by themselves make
this smooth manufactured transport/inflow response reproduce the coupled growth.
All saved replay peaks are <=2.54586 Pa and decrease with replay time. This is
not an absolute comparison to the original 58.796 Pa physical stress: initialization
and forcing differ. It is neither a general stability proof nor a bound for
arbitrary discontinuous histories or a time-dependent sequence of velocities.

## Energy, flux and cost

With ψ=H:H/(4G), verify the material-form identity

    dE/dt + F_supplied + D_upwind - integral(ψ div v)
            - integral(H:S/(2G)) = 0.

D includes internal jumps and incoming-boundary mismatch. Native q8 signs and
weights are used. Largest instantaneous defect is 1.18e-15 W/m. Omitting
divergence or source work would invalidate the audit. The following totals are
J/m at dt=.025, with outward flux positive:

| Quantity | Exact stage | Exact parent | Numerical parent |
|---|---:|---:|---:|
| Energy change | -.0771913673 | -.0770350976 | -.0751046159 |
| Supplied upwind boundary flux | .0728591496 | .0727020521 | .0707608774 |
| Interior-trace boundary flux | .0728591496 | .0727169509 | .0707999723 |
| Upwind loss | 1.65e-30 | 2.09e-7 | 1.39e-7 |
| Divergence work | 1.96e-7 | 1.41e-7 | -9.41e-6 |
| Manufactured source work | -.0043324132 | -.0043329955 | -.0043341982 |
| Signed balance remainder | -1.05e-14 | 1.77e-8 | 8.67e-9 |
| Sum of absolute macro-step remainders | 2.58e-14 | 1.77e-8 | 8.67e-9 |

The balance integrates instantaneous rates by trapezoidal quadrature at **every
RK substep endpoint**, then sums by macro step. It is a measured quadrature/RK
accounting remainder, not physical dissipation. Absolute totals above mean
absolute macro-step balances; they do not assert absence of substep cancellation.
All held/exact-parent macro balances are positive here. Largest accumulated
remainder across nine cases is 4.53e-8 J/m, for exact-parent dt=.1. Smaller
remainders do not imply smaller boundary-model errors: the numerical-parent
case illustrates the distinction.

One four-rank run took 69.39 s wall time, including host initialization, with
58.3 s reported in postprocessing for nine scratch trajectories. The separate
benchmark-only DSO built in 10.54 s with two build workers. Native core/checker
DSOs were not rebuilt. All nine cases completed, 84 macro-step records and
1008 RK substeps in total. Accepted native solution and mesh vertices are
checked unchanged; no physical history pointer is borrowed. There were no failed
simulation attempts. An initial successful compile's indentation warning was
fixed before the clean build/run; both build logs are retained. Existing unit,
rollback and restart evidence is reused because no native ownership or
checkpoint behavior changed.

## Conclusion and next justified experiment

**Implemented:** separate saved-velocity reader, nonpublishing manufactured
transport checker, three inflow controls, reproducible runner and compact
analysis/figure. **Locally verified:** native FE reconstruction, exact exterior
reference, material energy identity and immutable accepted solution/geometry.
**Physical comparison:** this is a manufactured operator isolation only. The
original coupled shear problem remains unqualified. **Production readiness:**
not established; generic accepted-trace inflow still lacks the required physical
exterior model. Mature, cohesive and production BP3 claims remain separate.

Together with the preceding compatible-top control, this narrows the remaining
question to the mixed-boundary mechanical response and its evolving feedback
with stress/inflow; the tested frozen transport response does not explain the
severe growth. It does not prove which mechanical contribution is responsible.

Next justified experiment: a **frozen-parent mechanical corner audit** using
saved bulk stress, pressure, profile and surface history. First separate weak
momentum residual, prescribed-velocity reactions and free-top traction residual
in the upper-left cells from raw face traces. If a re-solve is required, use one
fixed-history mechanics solve for original versus the already-defined compatible
top traction, with no material time advancement. This would locate the mechanical
mismatch before another coupled trajectory, boundary-model adoption or solver
change. It is proposed, not executed or automatically authorized here.

Unrelated cohesive tip/energy failures, finite-end representation limitations and
broader Stage C work remain open. Inherited losses of 39/45 native-pilot frames
and earlier temporary logs remain unrepaired. All retained failed evidence stays.
