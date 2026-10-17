# PF6a publication-consistency diagnosis and native confirmation

Selected 16 October 2026, revision `a8707307008401ac9cff702a369e2e2a611fa0f7`
plus preserved incoming work. Serial, fixed Cartesian mesh/reference surface,
2×512 cells, native q12, dt=1/128, unchanged physical seed/unloaded histories,
loading, consistent-endpoint/reduced Newton, constitutive/rate/acceptance rules,
and unlimited DGQ2. No advection, AMR/MPI, moving geometry, birth or propagation.
The original failed confirmation and all prior evidence remain preserved.

**The publication defect is repaired and the native run completes all 72 steps
in 879.915 s. Operator/lifecycle checks pass; strict final unregistered reference
agreement does not pass because the phase chooses the opposite mirror branch.**
All threshold misses remain explicit below. No general propagation qualification.
[Common-time curves](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf6a_publication/confirmation.png)
([PDF](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf6a_publication/confirmation.pdf)).

## Reproduction and independent diagnosis

`pf6a-publication-reproduce` uses the original first step and original guard;
only failure diagnostics were added. Exit 1 after 15.064 s, before either terminal
publisher. Particle 0, CellId `0_1:0`, physical point
(.083333333333333329, .00032552083333333332), reference coordinates
(.16666666666666666, .16666666666666666). Association sample position, particle
reference position and independently mapped reference position agree.

| Component | DG reconstruction (Pa) | Native Maxwell candidate (Pa) |
|---|---:|---:|
| xx | 5.6127218327007511e-7 | 5.6127218624662077e-7 |
| yy | -1.1226235677804302e-6 | -1.1226235692687031e-6 |
| xy | 7.8121948290598242 | 7.812194829059238 |

Original stored-tensor error 8.2901549768918641e-13 Pa; bound
6.6336671742158303e-13 Pa; ratio 1.2497092119897992. Complete local native
coefficients/support points, basis values/gradients, q3 mass/source operands,
incoming/outgoing DG coefficients, both stress candidates, actual remote-point
velocity gradient, effective rate, phase, I/Z/V, beta/kappa, vertices, clocks and
locations are saved before publication in `publication-failure.csv`.

The independent `replay_publication.py` uses **80 decimal digits**, treating every
saved double as its exact binary value. It differentiates all nine nodal Q2
coefficients, reassembles the projection from exported rounded FE operands,
and separately differentiates the exact Q2 polynomial at Gauss points. It does
not call native constitutive/projection/evaluation functions.

- Actual remote du_x/dy=.00099999999991843701; exact coefficient-polynomial
  derivative=.0009999999999185166. Difference=-7.967681298695596e-17 /s.
  Its derivative dot-product absolute operand sum is 1.6667037037045005 /s.
  A small gradient results from cancellation of nearly constant velocity values
  in a thin cell. This conditioning is absent from the old stress-only allowance.
- The DG source evaluated at that point differs from exact strain by
  (-1.59653e-19, 7.98267e-20, -2.33662e-18) /s. The independently integrated exact
  polynomial reproduces its point derivative to <=9.84e-20 /s (rounded rule).
  Max DG coefficient versus exact-polynomial projection discrepancy is
  7.80626e-17 /s. Projection/evaluation rounding is present but smaller here.
- Relative to the exact coefficient-polynomial stress, xy errors are
  -3.69869e-14 Pa (DG) and -6.23185e-13 Pa (native Maxwell). Their difference
  accounts for the failed comparison. xx/yy stress errors are <=2.50e-15 Pa.
- Both paths use the same accepted native velocity coefficients, dt=.0078125 s,
  G=1e6 Pa, eta=1e8 Pa s, beta=.99992187805167831,
  kappa=7812.1948321658811 Pa s. Incoming T/physical stress and Z are zero.
  Current/old phase are 1.5465604147304825e-10; local-versus-remote phase
  difference is 1.07e-26. Every old/new I is identical; Z recurrence discrepancy
  is <=3.35e-33 Pa m, with native minimum V=1e-20 m/s. Localization correction
  is only 1.42340e-26 Pa. Neither inconsistent state, a different localization
  law nor an incorrect Maxwell recurrence explains this failure.

## Repair and operand-bound derivation

The old allowance `B0=128 eps max(1,2(current_scale+beta old_scale))` remains
for history/localization arithmetic. Its comment incorrectly treated resulting
old/new tensors as a bound on forcing arithmetic. Triangle inequalities bound
an exact resulting strain, **not roundoff in differentiation of cancelling
velocity coefficients**, nor the intermediate projection. These omissions are
now accounted for separately. No constitutive/source/projection formula is
changed, no component is clipped, and the old allowance is not multiplied.
The particle-authoritative/default FE paths retain their existing allowance.

For unit-roundoff u=eps/2, write gamma_k=ku/(1-ku), n=9, Q=9. Bounds below are
nonnegative component bounds. On the restricted affine Cartesian cells,
L_i are the scalar nodal Q2 basis, |L_i|<=1, |L_i'|<=4, |L_i''|<=8. Their
expanded monomial absolute coefficient sums are <=8 (value), <=12 (derivative).
Six arithmetic operations bound quadratic Horner/product and simultaneous
value/derivative evaluation. Define

```
g_ab = 4 sum_i |u_(a,i)| / h_b
D(g)_aa = 2/3 g_aa + 1/3 sum_(d!=a) g_dd
D(g)_ab = (g_ab+g_ba)/2, a!=b
r_point = (1+gamma_(2n+10))(1+8 gamma_6)(1+3 gamma_6)-1
v_q = 8 gamma_6 + 4 gamma_4
r_q = (1+gamma_(2n+10))(1+v_q)(1+3 gamma_6+2 gamma_4)-1
s_q = (1+v_q)^2-1
r_b = (1+r_q)(1+gamma_(2Q+10))-1.
```

2n covers gradient dot products; the extra 10 covers mapping and symmetric/
deviatoric arithmetic. The Q2 polynomial absolute sums cover basis cancellation.
Gauss-abscissa representation is included in v_q/r_q; 2Q+10 covers quadrature
products/sums, weights/Jacobian and tensor combinations. These are operation
bounds for this fixed Q2/affine experiment, not fitted error multipliers.

Let A be the **actual computed inverse** of the assembled scalar mass. The
analytic mass is `M=|K| (M1 tensor M1)`, with
`M1=[[4,2,-1],[2,16,2],[-1,2,4]]/30` in support-point order. Compute
R>=|AM-Id| in long double, including gamma_(2n+2dim+4) in that precision for
its products and analytic-mass rounding. This avoids assuming a backward-stable
inverse or using a fitted condition number. With
`w_j=sum_q |JxW_q L_j(q)|`, `w_total=sum_q |JxW_q|`, the bound implemented is

```
e_b,j = [r_b w_j + w_total s_q (1+r_b)] D(g)
e_P,i = sum_j R_ij D(g) + sum_j |A_ij| e_b,j
        + gamma_(2n) sum_j |A_ij rhs_j|
e_P = componentwise max_i e_P,i.
```

The source-value factor D(g) bounds the exact derivative at all support and
Gauss points; therefore this is deliberately conservative in cancellation cases.
The last term bounds the actual inverse/RHS multiplication. Positive bound sums
use long double and outward conversion. This publication-candidate metadata is scratch:
it is neither history uncertainty accumulated between steps nor checkpoint data.
It does not enter mechanics or any physical publisher.

At a particle, Lambda=sum_i |L_i(unit)|. For each reference coordinate d,
`e_d=gamma_4 (|x_d|+|a_d|)/h_d` bounds subtraction/division in each real-to-unit
map. Gradient-coordinate sensitivity is bounded by
`c_ab=sum_d (2 if d=b else 4) e_d g_ab`; both maps are counted. Thus

```
e_native = D(r_point g + 2 c)
B_source = 2 |kappa| ||e_native + Lambda e_P||_3
B = B0 + B_source.
```

The full embedded plane-strain norm includes zz=-(xx+yy) and both symmetric xy
entries. The repaired DG comparison uses that norm too. This allowance comes
from velocity coefficients, shape/mass/source operations, widths and coordinates,
not the observed stress discrepancy. Deliberately inconsistent staged/incoming
histories cannot enlarge this velocity-derived allowance. Terminal validation
still precedes particle/surface publication and the accepted DG swap.

## Bounded diagnostic cost correction

The first repaired confirmation retains its 900-second cap and reaches accepted
step 71 (t=.5546875), then times out during the final step. Its original source,
records and all partial outputs remain pinned as `pf6a-publication-native512`.
This is a performance failure, not a publication-guard or physical acceptance
failure. Every accepted step retains a single publication.

The first repair computed scratch forward-error metadata during every mechanical
and phase Newton trial. That metadata is only needed by the terminal comparison.
The final source requests it explicitly from `prepare_fe_strain_commit`; ordinary
noncommitting DG projections omit that diagnostic calculation. Mass/source
assembly, inversion, tensor coefficients and every physical residual/algorithm
remain the same. A native coefficient-by-coefficient control verifies that
requesting the bound leaves all nine tensor coefficients identical. Small
positive products for the bound are accumulated as sums/products of error
factors instead of subtracting one from a nearly unit product. This stabilizes
the bound calculation; its formula/operation counts are unchanged. The repeat
confirmation keeps the same fresh input, end time and 900-second cap; this does
not extend the capped process or import its state.

The terminal-only version also reaches the 900-second cap in step 72, after
71 accepted publications; it is retained as `pf6a-publication-terminal-native512`.
Dense diagnostic output was still constructing two point FE evaluators and a
real-to-unit map separately at each of 147456 points per accepted snapshot.
The final benchmark adds a default-off **diagnostic-only** batch evaluation:
use the same DG coefficients, quadrature and accepted phase/Z/I in one FEValues
call per cell. All nine coefficients remain. The production publication guard
still independently checks every particle before publishing, with its original
point evaluators; batching is entirely after publication.

The full original diagnostic evaluator is compared at steps 0/1/68/72, using
the existing 1e-9 scaled physical-stress verification threshold and full 3D
metric, and accepted state is checked byte-for-byte. First-step control has
max difference 8.80e-15 Pa (7.96e-16 scaled), point cost 1.190 s versus batch
.00930 s for 147456 points. Production defaults, constitutive/solver equations,
physical inputs, timestep and the 900-second per-attempt cap are unchanged.
The two capped attempts are not relabeled as completed confirmations.

## Verification and native confirmation

Final-source controls: core SHA256
`7c76461c54b07f705683d4e4b572043ad1f70c6377cb1de018d3e458d699445f`,
verified observer SHA256
`3679a047d80676dc860a78f60239fa7ede95fb60accd2b6c381352621b7ced4a`.
`pf6a-publication-verified-controls512` exits 0 in 25.278 s. All 23 gates pass,
including exact equality of all DG coefficients with/without error metadata,
affine source, incoming xx/yy/xy and staged-history negative controls, native
outer rejection, persistent/RNG/solution/bookkeeping rollback, discarded staged
retry and exactly one successful publication. Native timestep-manager cutback
is separately verified at t=.0078125 -> .00390625, with zero publications at
retry entry and exactly two publications after two accepted steps. Every
original failed/capped/intermediate control is retained; final verified
observer source is pinned alongside its binary identity.

Fresh same-build frozen FE/DG compatibility passes. The unchanged FE baseline
is reproduced exactly; FE/DG fields differ by <=3.20e-14 scaled, weak loads
<=7.32e-13. Particle-poison load/solve errors remain exactly zero. Source checks
include non-proportional tensors and all nine Q2 modes; DG source scaled error
<=2.38e-15, FE <=1.21e-14. Three publications/backend, nonzero physical elastic
slip/cohesive traction and rejection checks pass. The prior PF4 operator/control
outputs replay byte-identically (23 CSVs). Final native Maxwell/cohesive units
pass 628 assertions in eight cases; independent PF1 references pass all 94 checks.

### Completed native trajectory and independent verification

`pf6a-publication-batched-native512` exits **0 after 879.915 s**, below the
unchanged 900 s cap, maximum RSS 587856 KiB. Core hash is the same final core
above; batched observer SHA256 is
`c39e5eb7d68cc0c1baecca76474ecf24a8b1bd62c62644a2185d219d0339286e`.
Resolved parameters differ from the preceding attempt only in output/library
paths and the two diagnostic-batch switches. No loaded state is imported.
All 72 common earlier snapshots (0--71) are **byte-identical** for phase, all
DG tensor coefficients, velocity, surface history, persistent state and RNG.

There are 73 complete snapshots, 72 accepted physical publications, no healing,
and 372 passing native gates. Every one of 663552 particle consistency samples
passes the repaired guard. Maximum full-3D error is 1.23831e-9 Pa, maximum
allowance 8.76479e-5 Pa, and maximum *per-sample* error/allowance is .000465792.
The old allowance fails 3275 full-3D comparisons; this counter uses the repaired
3D norm, whereas the original reproduced failure used its original 2D norm.
This is a conservative arithmetic allowance, not a physical stress envelope.
Both discarded/corrupted trials and actual timestep-manager rejection still fail
or discard before any publication; a successful retry publishes exactly once.

The original point evaluator is checked at 0/1/68/72 over all 147456 points each.
Maximum direct/batch difference is 1.15246e-9 Pa (6.36e-13 scaled), with unchanged
accepted state. All 73 batch tensor evaluations total .704 s; the four complete
point-evaluator comparisons total 4.875 s (evaluation cost only, excluding I/O).
All dense physical fields are still exported; no output-point thinning occurs.
The ASPECT version banner caches an older revision; actual source hashes,
patches, HEAD and executable/DSO hashes are the provenance authority.

At every accepted time, independent complete-Q2 reconstruction verifies the
Maxwell source recurrence (max scaled 3.98e-13), surface Z recurrence (max
1.01e-11 Pa m), physical-stress energy (scaled <=1.29e-15), and same-field shear /
cohesive weak loads (<=3.55e-14 / 5.43e-16 scaled). Native free bulk residual
norm reaches 7.44e-6 in its original weak-vector units; projected fault KKT
is <=2.17e-11 Pa. The raw fault residual RMS reaches 992.19 Pa during sticking:
its native sign is shear minus resistance, so a negative gap at the prescribed
minimum rate is admissible, not a failure of fault complementarity. Native
phase KKT <=1.45e-11 in the original energy-gradient units; irreversibility
and exact publication-count gates hold at every step.

Independent normal-ray integration (Gauss192 per Q1 normal interval), and bulk
normal quadrature 12/24/32 at steps 1/68/72, give gradient errors <=4.07e-13,
normalization <=1.57e-15 scaled, reconstructed stress <=9.08e-13 scaled, native
shear weak load <=2.01e-13 scaled. Quadrature energy discrepancy <=1.70e-16
scaled, load discrepancy <=4.95e-14, and 24/32 uncertainty <=6.35e-14.
These are far below the measured spatial/branch differences.

For the final interval, the retained independent diagnostic path (linear phase,
exact Maxwell histories at constant interval rates) has storage change
.294598898086 J/m, boundary work 1.423315817379, bulk/cohesive viscous losses
.000126826920/.000826169777, and friction work 1.127736689409 J/m. Its external
remainder -2.72332e-5 J/m is accounted for by free-mechanical .041712964551,
contact -.016571857384 and phase -.025168340354 J/m (pressure work negligible).
Accounting identity defect is **6.23e-15 J/m**, time-quadrature uncertainty
1.59e-14 J/m. The interpolated diagnostic path is not equilibrated between
endpoints; these free/contact/phase works must not be silently omitted or called
a new physical dissipation. This verifies accounting, not a general temporal
qualification of continuous coupled phase evolution.

### Comparison throughout onset/growth: strict target miss retained

The selected independent reference remains `p512-m512-d128-q12`; it is not
replaced or aligned. Onset agrees exactly at .53125 s. Through step 71 the
largest scaled whole/localized stress differences are 1.862e-8/2.308e-8, phase
L2 1.910e-10, driving 5.380e-9. At the final step, opposite translational
branches produce these maxima (fixed original reference peak scales):

| Quantity | Maximum difference | Original strict target 1e-6 |
|---|---:|---|
| Whole full-tensor stress | 1.49918e-6 scaled | **miss** |
| Localization-weighted stress | 3.65275e-6 scaled | **miss** |
| Phase relative L2 | 1.15276e-4 | **miss** |
| Phase Linf | 5.90852e-5 | **miss** |
| Cohesive traction | 7.91687e-10 scaled | pass |
| Cumulative slip | 6.84734e-10 scaled | pass |
| Recoverable energy | 1.57487e-10 scaled | pass |
| Common physical phase driving | 2.38332e-4 scaled | **miss** |

No strict criterion is relaxed; `strict_reference_match_passed=false` is retained
separately from passing operator/publication checks and complete execution. All
listed practical benchmark screens pass, but this does not erase the strict
miss. Final maximum phase is .654399205861; surface-mean Z=39155.469199536 Pa m,
I=145.726065559535 m, q=268.692282668811 Pa, V=.144350296244835 m/s,
cumulative slip=.039225641963914 m, physical elastic slip Z/G=.039155469199536 m,
energy=22.2182852822033 J/m. Elastic and cumulative slip are distinct histories.

A **read-only** reduced-1D endpoint diagnostic at the same incoming state finds
native/reference phase centroids .500008508314/.499991491744 m, separation
1.70166e-5 m. Along-fault native phase spread is <=3.30e-10. Their reduced
potential difference is -4.12115e-13 J/m; translation curvatures are both
+.14055 J/m. Native logs also resolve negative curvature before departure and
positive curvature after it. The x-averaged native phase agrees with the *mirrored* selected
profile to 2.01e-10 Linf, but that diagnostic is **not used** in any primary error
or acceptance test. No reference field, native field, obstacle, target, sign or
symmetry constraint is changed. The discrepancy is comparable to the previously
recorded phase-position sensitivity, not an unresolved Maxwell publication defect.

Source review finds a numerical tie distinction: native departure candidates
must improve the **current best** beyond the arithmetic allowance; the independent
reference requires a resolved improvement over entry, then compares competing
candidates strictly without that between-candidate allowance. On this symmetric
fixture the two branch energies differ below the native 32-epsilon energy screen
(about 4.55e-13 J/m). This can select opposite otherwise indistinguishable branches.
This is recorded as a reference/branch-selection limitation, not corrected by
retuning or replacing the chosen reference. General 2D stability, uniqueness and
strict phase-trajectory equivalence remain unqualified.

The independently selected 512 candidate's prior continuum-reference mechanical
errors (.29439% whole / .61065% localized) remain the practical spatial evidence.
This native execution supports the retained equations and publication repair;
it does **not** justify claiming all strict native/reference targets passed.
[Portable evidence and replay](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf6a_publication/README.md)
retain the failed assertion, capped runs, full coefficient histories, diagnostics,
plots, independent checks and all target flags.

## Remaining dependencies and next bounded task

This proof is for the existing nine interior particles per affine Cartesian cell,
fixed mesh/reference surface, serial, no transport, homogeneous constant G/eta.
Particles still supply startup H/CPDI/domains and material projection, association
and fixed-support bookkeeping, derived samples/publication validation, and
persistence/RNG. Their stored stress does not feed DG history, mechanics or fault
loads. The diagnostic batch changes neither this ownership nor physical histories.
The new source bounds are not accumulated constitutive uncertainties or serialized
checkpoint fields. No restart conversion or general PF5 completion is claimed.

Retain unlimited DGQ2 and existing defaults. Before treating this as a strict
transfer/restart trajectory reference, specify a common unresolved-branch/tie
comparison and test both endpoint branches without aligning away primary errors.
Do not impose a phase-sign or symmetry lock. The concrete subsequent task remains
fixed-geometry AMR/MPI: first specify and test accepted T transfer retaining all
nine modes, physical phase/lower-obstacle transfer, and Z/normalization ownership;
then verify one/two/four-rank rejection, single publication and restart against
serial common-field stress/energy/weak-load functionals. Keep the reference
surface mesh fixed initially, so no surface interpolation is silently introduced.
If the intended fixture changes that mesh, separately specify physical elastic
slip/Z transfer and recompute I consistently; q=Z/I is derived.

A specific admission check is needed before that task: native
`VectorTools::point_gradients(..., avg)` can average adjacent-cell gradients for
a face sample, while DG reconstruction uses an explicit CellId/one-sided trace.
Those are distinct discrete quantities. This experiment's particles are strictly
interior and fixed; no face ambiguity occurs. AMR can put an existing interior
particle onto a child face, so matching-cell evaluation/trace ownership must be
verified rather than charged to roundoff. MPI point reductions and curved cells
also require their own operation audit. No guard for unsupported AMR/MPI,
periodic, moving geometry, coupled transport, surface birth or propagation is
weakened, and none of those paths is qualified or launched here.

Portable replay verifies all 3402 archived file hashes and independently
reproduces the 80-digit first-cell diagnosis, complete common-time measurements,
and branch diagnostic without launching ASPECT. The replay preserves
`strict_reference_match_passed=false`. Archive size is 126150224 bytes; original
raw outputs/binaries remain local. The original reference report, status/audit
payloads and all unrelated incoming source/evidence are preserved.
