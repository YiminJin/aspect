# Matched PF6a evolving spatial-reference study — 16 October 2026

Follow-up: [publication repair and native completion](PF6A_PUBLICATION_REPORT.md)
reproduces/fixes the first-step guard failure and completes all 72 native steps
within 900 s. Strict final unregistered phase-branch agreement remains open;
the original failed attempt and its conclusions are preserved below.

**The 1D study selects 512 normal cells for the fixed practical stress targets;
the single native confirmation fails its existing publication check at step 1.**
No native onset/growth qualification follows. Unlimited DGQ2, physical laws,
quadrature, acceptance rules and production defaults remain unchanged.

Root/branch: `aspect-pf-rsf-particle-free` / `pf-rsf-particle-free`, revision
`a8707307008401ac9cff702a369e2e2a611fa0f7` plus preserved incoming work.
[Pre-native formulation and audit corrections](PF6A_REFERENCE_FORMULATION.md),
[evidence/replay](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf6a_reference/README.md).
No production source or CMake changes in this task. Six new benchmark scripts
and documentation implement/measure the independent reference and native attempt.

## Initialization and reference equations

Every main trajectory starts unloaded, T=Z=slip=0, with the **same documented
continuum seeded-H stationary initial damage definition**, protected by physical
irreversibility. This reuses the retained `matched_initialization` BVP data
(SHA256 `7a7a11c52388b3a00aef1b586c844036b139dc2046d7bfca28321329f71020e7`).
The supplied native 512-grid seed table equals that definition at every table
node **exactly**. At every mesh <=512, sampling the continuum definition and
sampling the table gives byte-identical nodal initial phase. The finer 1024
phase reference samples the same continuum definition at its additional nodes.
The selected native input is unchanged. No loaded coarse state is imported.
No mesh-specific H relaxation or new seed fit is used.

A preliminary interpretation refined the literal 512-node table interpolant.
That same physical piecewise-linear field is represented exactly at 1024, but
its enriched phase space has unloaded KKT defect 7.84064e-4 (gradient divided by
A), first-step growth 7.84064e-6, and premature threshold onset .4453125 s.
The retained source-origin diagnostic shows the underlying continuum seed has
zero unloaded defect at 1024; its values between old knots differ by <=3.29445e-5.
These literal-table runs and their onset failure are preserved as a separate
initialization/obstacle diagnostic. They are **not** the main convergence
reference. The formulation records this source-origin correction and the
unchanged native nodal values; pre-native decision files are preserved as made.

For the main physical continuum seed, relative phase interpolation L2 errors
at 64/128/256/512/1024 are
.00176892/.000442203/.000110758/.0000276911/.00000692278.
Corresponding phase-gradient L2 errors are
.0918743/.0459309/.0229999/.0115005/.00575021; fracture-energy errors are
-.0131203/-.00329691/-.000826281/-.000206594/-.0000516505 J/m.
All main initial unloaded KKT checks pass. Seed representation is reported
separately; it is not subtracted to hide the subsequent obstacle effect.

The inexpensive 1D model retains the actual consistent-endpoint phase partial,
normalization derivatives and reduced tangent. The historical split
`Reference.run` is not used. Unit normal height; P is the cellwise L2 projection
onto P1 derivatives of Q2 velocity; chi=h(phi)/I, C=integral chi,
D=integral[(Id-P)chi]^2, B=beta integral(Tn)+kappa U, f=mu Pcompression.
For compatible unloaded homogeneous shear histories, mechanical elimination is

```
beta=exp(-G dt/eta), kappa=-eta expm1(-G dt/eta)
H=C^2+D+1/I; Ze=max(beta Zn,(C B-f)/H); V=(Ze-beta Zn)/kappa
lambda=B-Ze C; Te=lambda+Ze P chi; tau=lambda-Ze(Id-P)chi
J=[lambda^2+Ze^2 D+Ze^2/I]/(2G)+F_FE+(f/G)(Ze-beta Zn).
```

Continuum mechanics sets P=Id, D=0 at the **same phase resolution**. C and its
actual derivatives are retained, not hardcoded to one. The phase derivative is
at fixed endpoint Te/Ze after mechanical equilibration; the reduced Hessian
contains mechanical response. This is algebraic elimination of the retained
1D equations, not a different law or a reduced native history backend. Native
DGQ2 retains all nine tensor modes; the homogeneous shear source occupies its
normal P1 subset. The 1D mathematical sticking rate is zero; the unchanged native
numerical minimum is 1e-20 m/s, below all comparison scales.

Independent checks of reduced energy/gradient/tangent pass; best directional
energy errors <=5.5e-12 and tangent errors <=7.9e-11. The retained full Q2
mechanical matrix solve agrees to <=6.37e-9 Pa, Z <=2.79e-9 Pa m.
Every trial uses immutable incoming mean history/Z and accepted phase obstacle.
The retained positive search metric and feasible energy-decreasing translation
departure handle real translational saddles. No reflection/symmetry lock is
imposed. Earlier incorrect diagnostic assertions and incomplete branch searches
are retained; no physical Hessian or KKT tolerance was relaxed.

## Spatial comparison throughout onset and growth

Time 0--.5625 s, dt=1/128, fixed ramp .256 t, same G/eta/ell/Gc/friction and
natural phase boundary. Projected Q2 mechanics versus continuum at phase
64/128/256/512 separates mechanically induced error. Continuum mechanics at
phase 64/128/256/512/1024 separates phase-space/obstacle error. Controls use
dt/2 and quadrature 12/24. Complete common-time curves, field snapshots, energies,
slip/traction and phase driving are retained, not only endpoint values.

Targets were fixed before native confirmation: whole/localized full-tensor
stress 1%/2%; phase relative L2 1%, Linf .005; traction/slip/total energy 1%;
common phase driving 2%; onset within one dt. Fixed **reference peak** scales
avoid startup denominators: tensor stress 1855.656 Pa, q 312.1469 Pa, cumulative
slip .03922565 m, energy 22.21806 J/m. Full shear tensor norms include sqrt(2).
Localized RMS is sqrt(integral chi*|error|3^2 / integral chi).
Phase gradients/driving are pulled back to identical 64-cell Q1 test functions;
all bulk/cohesive/fracture components and mass-dual norms are retained.

| Mechanical cells / matching phase cells | Max whole stress error | Max localized error | Max phase L2 error | Max common driving error |
|---:|---:|---:|---:|---:|
| 64 | 16.867% | 34.111% | 3.2145% | 10.993% |
| 128 | 4.6553% | 9.6331% | .20757% | .78452% |
| 256 | 1.17594% | 2.43869% | .004320% | .029862% |
| 512 | .294387% | .610657% | .010149% | .021928% |

The requested old frozen profile at 256 gives **.689260% whole and 1.265139%
localized**, verified with quadrature 24/48. Its errors are 12.60634/23.13897 Pa.
During growth the direct Z(Id-P)chi remainder reaches 21.8220/45.2554 Pa at 256,
versus 5.46285/11.3318 Pa at 512. Thus the frozen extrapolation is valid for that
snapshot but underpredicts the grown-profile errors. 256 misses the unchanged
mechanical targets; the one bounded 512 candidate passes and was selected.
No limiter or artificial stress bound addresses this mechanical projection error.

At continuum mechanics, refining phase 256 -> 512 -> 1024 reduces traction and
energy differences against phase 1024: traction .028088% -> .005619%, energy
.004755% -> .000953%. Phase relative L2 differences .050069% -> .046752% are
less regular because of small unaligned translations between nearly neutral
branches. These position differences are retained, not reflected/registered away.
All main cases have declared growth onset **.53125 s**. This does not establish
an asymptotic spatial order for branch position or a unique global evolution.

Against continuum mechanics/phase 1024, the selected 512 case has maxima:
whole/localized stress .294391%/.610652%, phase L2 .046765%, phase Linf .00025202,
traction .005618%, cumulative slip .00004519%, total energy .00099355%, common
phase driving .087581%. All practical 1D targets, including onset, pass.
Stress maxima occur at .5625 s; phase/driving maxima occur at .546875 s;
traction/slip differences peak at .5234375 s. Main-reference final phase is
.65440197, q=268.680285 Pa, slip=.03922565438 m and total energy 22.21806453 J/m;
selected 512 values are .65439921, 268.692283 Pa, .03922564196 m, 22.21828528 J/m.

Halving selected dt changes stress fractions by <=3.66e-6, phase L2 by .0001153,
phase Linf by 5.92e-5, driving by .0002385, energy by 7.33e-8. Finer-reference
half-step phase L2/Linf differences are 2.29e-7/1.43e-7. These meet the fixed
uncertainty screens and are below the material spatial differences. Selected
phase/position uncertainty remains appreciable relative to its small spatial
shape difference; no temporal/spatial order claim follows for that soft mode.
Fixed-field selected quadrature 12/24 differences are <=4.57e-13 (gradient),
1.39e-13 J/m (energy), 4.44e-10 Pa m (Z), 2.92e-11 Pa (mean stress).
The complete 256 quadrature-control trajectories also agree within roundoff.

## Single native confirmation: failure retained

Fresh native 2×512, dt=1/128, quadrature 12, original seed file, histories
unloaded, existing consistent-endpoint/reduced-Newton solver and observer.
ASPECT core SHA256
`c994b6b809bde83f456ebdf2d0c573ea2ff8dbf69427add898982a3b573a9693`;
observer DSO `172dabb46f0d1fb17aad2943fdb4592e97bf3946aeeb3210bab8d0519398ebc7`.
Native DG serial guard is preserved despite permission for up to eight processes.
Reference jobs use at most four independent processes. Native budget 900 s;
actual exit 1 after **15.152 s**, maximum RSS **552192 KiB**.

Native startup matches prescribed phase exactly, has zero T/Z/slip and zero
publications. I differs from the independent reference by <=1.43e-13;
fracture energy by <=2.31e-14 J/m, phase gradient by <=1.50e-13.
There are 1024 cells, 1539 phase DoFs, 10250 velocity DoFs, 27648 tensor DoFs,
9 surface rate nodes and 38953 shared-vector entries. Four startup gates pass.

At t=.0078125, native mechanics converges and the endpoint is staged; the
publication check in `history.cc::commit_reconstructed_fault_mechanical_history`
fails `error <= 128 epsilon max(1,evaluation_scale)` with
`Accepted FE strain does not reproduce the native Maxwell publication.`
This compares the DG-derived stress with the independently computed native
Maxwell publication candidate at particle positions. It occurs before terminal
history/particle publication. No positive accepted-step export exists.
The exception does not print error/bound/component/position values, so its
numerical cause is **unresolved**, not established to be harmless roundoff.
Positive-step rollback bytes and native growth accuracy were not verified here.
No rule was relaxed, no second native run launched, and no failed state accepted.

## Recommendation and limits

512 is a measured **1D mechanical candidate**, not an adopted or native-qualified
resolution. Retain unlimited DGQ2 and the current physical/acceptance equations.
The next concrete task is a bounded publication-consistency diagnosis at this
same first native step: capture error, bound, components, CellId/position,
native velocity gradients, projected DG source, incoming T/Z and localization;
compare them independently before considering a correction. Preserve the guard
and failed evidence. Only after that diagnosis can a native onset/growth
confirmation be selected. No immediate larger native mesh or propagation run.

The source-origin seed distinction and literal-interpolant diagnostic remain
explicit; main references use the existing continuum prescription. Small
translation branch/phase-space uncertainty and finite final time remain limits.
Particles still own startup H/CPDI/domains, material projection, association,
derived samples, publication validation and persistence; their stress does not
feed independent history or mechanics. No AMR/MPI, surface transfer, moving
geometry, birth, coupled transport, RSF evolution or general propagation is
qualified. PF5 is not declared complete. Prior work/evidence and defaults are
preserved; the evidence package records source/data/commands and preservation.
