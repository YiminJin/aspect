# Bounded phase-release pair and ahead-tip driving

This is the selected 256×128 weakening / matched b=a pair following
`STAGE4_SHORTWAVE_CONTINUATION_REPORT.md`. Both start freshly at t=0 with the
exact retained phase and static equilibrated stress/cohesive preload, the same
1% smooth age perturbation, and uniform boundary velocity. Midpoint dt=8e-5 s,
eight intervals to .00064 s, unchanged acceptance criteria, four ranks total,
45 minutes per case. The reference inputs are copied verbatim; removal of the
fixed-phase benchmark constraint enables the existing coupled obstacle solve.
No old growth checkpoint is converted. No production code is changed.

`prepare.py` builds a new checker from the retained tested pilot source and
`observe.h`; it preserves the old checker and main executable. `run.py` starts
one fresh law. `batch.py` launches precisely the two cases sequentially under a
durable supervisor; `build-particle-free/stage4-release/stop-batch` prevents the
next launch. Failure stops the batch for inspection. Outputs/checkpoints stay
in the ordinary bulk-tip run directories. No restart observer is claimed: this
observer explicitly requires its fresh initial reference.

## An admissible increment, including the intact reaction

Use the native Bernstein direction chi_i equal to the product of piecewise
linear tents at each coefficient's lattice point. Each tent's breaks align
with cell faces, so this is exactly the represented Q1 tent inside DGQ2.
All chi_i are nonnegative. The fixed transverse tent covers y=1.875…2.125 m.
Two fixed x supports distinguish the actual diffuse tail:

* Probe 0: x=6.03125…6.15625 m, immediately beyond the initial .05 contour.
* Probe 1: x=6.3125…6.4375 m, the first fully intact band beyond all represented
  seed support. Tiny coefficients (~1e-13) between 6.1875 and 6.3125 are retained.

The probes stay in these physical locations as phase evolves. They are not
moving windows or a localization rule. `admissible_stage_epsilon` records the
maximum positive amplitude consistent with the actual upper bound. At t=0 it
is B+epsilon chi≤1; in midpoint coordinates b_m+epsilon chi≤(1+B_n)/2, equivalent
to an endpoint increment 2 epsilon chi. The nonnegative direction respects the
transport-free physical lower obstacle. A positive reported net driving means
that this particular admissible direction lowers the reduced functional; a
negative value rules out that direction, not every possible localized shape.

For c=m phi(1+phi), H=c/[c+(1-phi)^2], G fixed and S:S=1/2, hold the total elastic
predictor T=tau+2G A and section resistance f fixed. Eliminating e=2A:S from

    |T-2G e S|²/(4G) + G e²/(2 h_c(phi)) + f e

gives the phi-dependent part -H(phi) F²/(2G), F=T:S-f. Its negative derivative
is Y=H'(phi)F²/(2G), with H'=m(1+3phi)(1-phi)/[c+(1-phi)^2]^2.
This is the existing point-elimination driving force. At phi=0 its right limit
is m(T:S-f)²/(2G); holding A=0 and reporting zero would miss the admissible
coupled deformation. At t=0 the verified preload supplies T and f, including
the closure's intact reaction. This is not an extra time step or a change to
the resistance law.
The t=0 direction S is the retained initial section plane. The diagnostic does
not additionally maximize over possible virgin planes. Later stages use the
native solver's actual selected planes and intact reaction; ordinary direction
selection remains enabled in unformed sections. Thus the first change of an
intact probe can include plane selection as well as loading. Neither negative
probe is a proof against every admissible shape/orientation.
At a solved section, the first variation of its friction potential is f times
the rate/slip variation. The local constant-f potential therefore gives the
same first-order phase term after section elimination; it does not assert that
f remains fixed for a finite physical phase change. The actual coupled solve
continues to recompute the section law and every cross derivative.

For each probe the observer records

    D_mech = integral chi Y dx,
    D_frac = chi . gradient E_fracture,
    D_net = D_mech - D_frac.

The native `assemble_dg_fracture` supplies the full AT1/SIPG derivative, including
face consistency and penalty terms. The normalization `D_net/integral chi dx`
has stress units (Pa); unnormalized derivatives have J/m per unit amplitude.
The actual material initialization sets m=2 G Kf/strength². Here Kf=2 Pa,
G=1e6 Pa and strength=1000 Pa, hence m=4. In the completely intact probe,
D_mech/D_frac is the chi-weighted mean of (F/1000 Pa)². This provides a useful
loading-margin interpretation; it is not a direct multiplier for the prescribed
boundary shear, since mechanical redistribution and section resistance also
contribute to F.
At t=0 one-sided increments 1e-4,1e-5,1e-6 compare the actual native fracture
energy change plus the local reduced-potential change against D_frac-D_mech.
This verifies the diagnostic's partial derivative at fixed predictor/resistance;
it is not an extra globally re-equilibrated phase experiment.

After physical intervals, the observer uses the actual accepted midpoint Y and
fracture gradient. Their time is t_n+dt/2. Phase, slip, age and fracture inventory
are endpoints. In particular, it does not assign an invented endpoint reaction
to intact material or call a last-stage force an instantaneous endpoint force.

## Phase, fracture work and preservation

`release-phase-NNNN-rR.csv` stores all nine native coefficients per cell, allowing
polynomial/contour observations without full quadrature-history dumps.
`release-regions.csv` uses fixed geometric masks: central band, old taper/tail,
initially intact surroundings (axial and transverse), and remote material.
The exact quadratic fracture increment is partitioned by native rows as
delta B_i (gradient_i(initial)+gradient_i(endpoint))/2. Face interactions are
included and assigned through their rows; a regional contribution need not be
positive. The sum is checked against the full native fracture inventory change.
This partition is not a unique pointwise fracture-energy density.
Analysis also separates the AT1 potential increment Kf integral(delta phi) from
the remaining gradient/face increment; Kf=2 Pa for these exact inputs. Their
cancellation in the fracture inventory is distinct from numerical physical-work
error and is not hidden by reporting only the net fracture change.

The existing section, endpoint energy, work, acceptance and timer observations
are retained. The physical-work remainder is Wext-delta E-Db-Dc-Df, before
subtracting the nonlinear cohesive midpoint remainder or residual work. Report
both signed cumulative and accumulated absolute errors. The initial reference
is retained in memory only for observations and is never a physical owner.
