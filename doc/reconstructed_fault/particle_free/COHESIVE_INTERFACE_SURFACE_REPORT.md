# Focused surface/jump-resolution comparison

**Completed:** halving surface spacing gives small changes in the developed
finite-strip response. It does **not** qualify precise initiation timing.

[Selection](COHESIVE_INTERFACE_SURFACE_SELECTION.md) ·
[figures/data](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_surface/INDEX.md) ·
[compact archive](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_surface.zip).
Baseline `f1f9f29cf40c32d1ddcea000b724375b64853532`, branch
`pf-rsf-particle-free`, plus preserved local work. Production/BP3, the native law,
previous failures and previous archives are unchanged.

## What was compared

Two fresh evolutions use **256/512 Q1 surface cells**, Δs=62.5/31.25 mm, and the
existing exact incompressible mixed-strip operator projected onto those spaces.
The specimen is unchanged: L=16 m, H=1 m, fault y=.4 m, physical x=[−4,12],
fixed Γ=[−3.5,11.5], mature seed [2.5,5.5], periodic sides, impermeable shear-loaded
walls and physical p0=10 kPa. Loading remains 1 mm/s to 2.55 s, then hold to 2.58.
This is the established **finite-wall specimen**, not a distant-body test.

Both meshes start with zero jump/Maxwell histories, unit age and the same aligned
mature seed. Initial inventory is exactly 3 J/m on each; initial jump and inventory
representation differences are zero. No nonlinear history is transferred between
meshes. G=1e6 Pa, η=1e8 Pa s, Gc=1 J/m², peak=3000 Pa, RSF .005/.01/.1,
V0=5e−6 m/s, Dc=.001 m and radiation coefficient remain unchanged.

The common timestep sequence is the completed native half-step trajectory's
actual sequence: .0025 s through 2.30, .00125 s through the pending interval
ending at 2.425, then .000625 s to 2.45 and .0003125 s afterward (actual
roundoff/clipped endpoint clocks retained). A **512-cell half-step replay** starts from
its own saved 2.42375 s state and halves every subsequent interval. It controls
late temporal uncertainty, not early onset. No native bulk mesh is rebuilt.

Each surface cell retains three independent Gauss histories and the current-pressure
coupled tangent. This comparison refines the combined jump/surface-history
discretization; it does not separately qualify nonlinear surface quadrature.
Damage is surface **a**, distinct from provisional bulk φ. Fracture is counted
only as Gc∫a; the initial seed is excluded from new fracture.

## Measured developed response

| At 2.58 s | 256 cells | 512 cells | 512, half Δt |
|---|---:|---:|---:|
| Left full advance (m) | 3.213986 | 3.213151 | 3.212277 |
| Right full advance (m) | .097386 | .097893 | .097896 |
| New fracture (J/m) | 4.204249 | 4.203057 | 4.201703 |
| Integrated absolute jump (m²) | .007530151 | .007528519 | .007528143 |
| Left cohesive-zone width, a=.01→1 (m) | 1.01821 | 1.02068 | 1.02009 |

The 512-cell cohesive zone spans about **33 cells**. Continuous contour positions
come from Q1 jump/critical-slip crossings after verifying that they reproduce
the actual stored Gauss damage; no new constitutive state is interpolated.

At the endpoint, surface refinement changes integrated growth **.001192 J/m
(.0284%)**, integrated slip **1.633e−6 m² (.0217%)**, and the left full front
**.836 mm (.0260% of advance)**. Fine-grid temporal changes are **.001354 J/m
(.0322%)**, **3.761e−7 m² (.00500%)**, and **.874 mm**, respectively. Spatial
and temporal changes are comparable for late growth/left-front position; slip
retains a larger spatial contribution, although both changes are small.

The right full-front spatial change is **.507 mm (.518% of its advance)**,
versus .00363 mm temporally. Near its first full decohesion at 2.55, the same
.449 mm spatial difference is **5.34%** of its short advance. Relative errors
near a newly formed full front must not be hidden by the much larger left front.

Across 2.45/2.50/2.55/2.58, the largest spatial integral-growth change is **.1003%**
and left-front change **1.163 mm**. The declared 5% growth/slip and .03125 m
front targets pass. Actual .125 m physical-box Gauss averages give final
growth-relative L² difference .0597%, normal-traction RMS .392 Pa and shear-load
RMS .951 Pa. Their fine temporal counterparts are .0674%, .297 Pa and .626 Pa.
These are projected coupling quantities, not raw point-trace error estimates.

At the same 256-cell surface/time sequence, replacing native bulk FE by the
analytical bulk changes final growth **.000682 J/m (.0162%)**, slip .0107% and
left full advance **.470 mm**. Thus the surface comparison is separated from
bulk mechanical approximation. Independently projecting the computed fine Q1
jump into its best coarse Q1 L² representation gives .0106% global relative
error, .0842 μm tip RMS and .885 μm maximum at 2.58. This diagnostic projection
never initializes or changes either trajectory.

![Separated space and time changes](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_surface/space-time-errors.png)

## Early admission remains sensitive

First positive new damage is bracketed by **[1.4800,1.4825] s** on 256 cells and
**[1.4075,1.4100] s** on 512: a **72.5 ms** shift, versus the common early
Δt=2.5 ms. Initial fields are exactly matched, so this is surface evolution/
discretization sensitivity, not initialization by prolongation. At the
diagnostic ∫(a−a0)=.01 m threshold, ordering reverses: [1.8300,1.8325] versus
[1.8400,1.8425] s. These are measured brackets, not continuum activation times.
The focused late halving cannot qualify early temporal error. Preserve the
historical **65.4% early spatial failure**; small late differences do not erase it.

![Early response](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_surface/early-response.png)

## Checks, work and computational cost

The cached-layout extension reproduces the entry unit-domain reference bitwise
over 20 loaded steps. Three frozen native steps match histories to <3.75e−14,
jump/memory <3.80e−17 m, shear <4.24e−10 Pa and normal traction <5.10e−11 Pa.
Independent complete pressure-coupled derivatives differ by ≤5.14e−9 against the
unchanged 1e−6 gate. Alias300→600 changes symbols by ≤2.98e−7 relative and final
shear loads by <2e−5 Pa, far below the measured spatial changes.

The 512-cell previous-rate predictor stopped at the first hold step. The saved
incoming state reproduces that stop. A **zero-increment initial guess only**
recovers a candidate with complete signed weak residual 1.05e−11 Pa, retained
admissible tangent 9.78e6 Pa/m and nonzero damage increment .01056. Histories,
pressure, tolerances and phase freedom are unchanged. The stopped attempt is
preserved; continuation resumes its valid state. This is not a ban on hold growth.
NPZ round trips reproduce all history arrays bitwise on both meshes; rejected
trials leave incoming state unchanged and logical counters advance once.
This reference lifecycle check does not add native/production checkpoint claims.

Maximum accepted weak residual is 9.99e−9 Pa, BE ledger discrepancy <5.04e−14
J/m, contact stays positive and the retained admissible symmetric tangent stays
positive. That check is not a general RSF potential/stability theorem.
Raw physical-work remainders, separately from exact ledger closure:

| Fine interval | / new fracture, Δt→half | / sliding loss, Δt→half |
|---|---:|---:|
| 2.45–2.58 s | .325→.163% | .251→.126% |
| Hold 2.55–2.58 s | .382→.191% | .274→.137% |

Absolute and signed sums have equal magnitude, so no cancellation hides timestep
error. Hold external work is exactly zero. All declared physical-work targets pass.
Evolution cost is **18.1+58.9+39.2=116.2 s**, including the stopped fine attempt
and completion; evolution peak RSS **137 MiB**, one thread, sequential runs.
Native archive reading, independent qualification, plotting and packaging are
excluded from those evolution costs.

## Conclusion and next limit

The developed advance, cohesive zone and hold response of this **finite,
quasi-static prescribed-path specimen** are insensitive to this one surface
halving at the stated practical scales. Two meshes and differences comparable
to temporal uncertainty establish no convergence order or exact continuum error.
Precise first admission remains unqualified. No new law, native production
option, pressure fit, friction lag, limiter or fracture reservoir was introduced.

Before interpreting front speed physically, decide whether to retain this
quasi-static benchmark or qualify bulk inertia. The previous hold-front/model
shear-speed comparison remains unresolved. A future initiation study must target
the initial mature-tip admission separately. No further mesh, free-path, bend,
native long-run or propagation campaign starts here.
