# Controlled candidate comparison, frozen before transport validation

This is the user-selected pre-PF7 audit, on branch `pf-rsf-particle-free`, starting
at `a8707307008401ac9cff702a369e2e2a611fa0f7`. It does not start PF7 or adopt a
limiter. Preserve unlimited upwind DGQ2, defaults, physical laws and all earlier
evidence. Runs remain serial, fixed affine Cartesian mesh, prescribed transport,
with immutable incoming stage state and no accepted-history publication.

Primary reference: Zhu, Zhong, Shu and Qiu (2016),
[author manuscript](https://archive.ymsc.tsinghua.edu.cn/pacm_download/113/1132-CiCP-749.pdf),
DOI [10.4208/cicp.070215.200715a](https://doi.org/10.4208/cicp.070215.200715a).
Sections 3.2 and 4.2 fit on a neighbor's own cell subject to a target-cell mean
constraint; the 2D construction additionally uses selected other-cell mean
information. The paper uses total-degree polynomial spaces and its own
detector/smoothness/normalization and system treatment. This experiment replaces
only candidate construction in the existing Q2 tensor framework. It is not a
complete implementation or reproduction of the paper's shock tests.

## Selected Q2/tensor equations

Let K be the target, J one genuine immediate face neighbor, and P the target
plus available face neighbors. Averages below are physical cell averages. Let
`mu_I=<A_I>_I`, `a_J(q)=<q>_K`, and use the established full-3D metric
`|A|3²=xx²+yy²+(xx+yy)²+2xy²`. For each J, useful cells are other face neighbors
I whose `|mu_I-mu_K|3²` is strictly smaller than the maximum over those others.
Ties within 64 machine epsilons relative to that maximum are excluded; an empty
set is allowed. This tie convention is numerical, not threshold tuning. There
are no diagonal or neighbor-of-neighbor DG histories in the stencil.

For J, solve over **all nine Q2 coefficients**, with one common useful set:

```
min_q  <|q-A_J|3²>_J + rho sum_(I useful) |<q>_I-mu_I|3²
subject to <q>_K = mu_K,
rho = |J| / |reference domain|.
```

The reference domain is the fixed rectangular box, not the patch. Its area is
1 m² in the selected unit-square experiment. Thus this is the paper's relative
physical-integral weighting on the nondimensional unit square: dividing its
objective by neighbor-cell area produces rho. No mesh-dependent parameter is
tuned. Taking a cell-normalized mean penalty of unit weight would be a different
experiment and is not substituted. At boundaries, use only actual face neighbors
and the corresponding reduced useful sets; no exterior polynomial is invented.

In J's nodal basis, let M be its cell-average mass matrix, c_I the row averaging
that basis over I, a=c_K and u_J the original coefficients. The common scalar
spatial system for every tensor component is

```
H = M + rho sum_I c_I c_I^T
b = M u_J + rho sum_I c_I mu_I
[ H  a ] [ q      ] = [ b    ]
[ aT 0 ] [ lambda ]   [ mu_K ].
```

The constant positive-definite tensor metric cancels from this common spatial
KKT solve. The implementation uses its Schur complement and precomputes linear
correction maps for each possible useful set; it does not solve a data-dependent
9x9 system per stage. Native candidate values are then evaluated on K. The
existing target-mean restoration removes only roundoff residue. The center
candidate remains the original complete target polynomial. An independent
reference solves the full 10x10 KKT system in tensor-product Legendre coordinates
with a diagonal exact neighbor mass and Gauss5 average rows.

## Fixed ingredients and bounded measurements

Keep KXRCF threshold 1 and safeguarded RMS Fu–Shu threshold .1, their original
unmodified-polynomial indicators, common full-tensor weights, d0=.996 and four
neighbor weights .001 (omitted boundary weights go to the center), the existing
dimensionless epsilon 1e-6 and zero-beta branch. Keep the exact existing
first/second target-reference derivative smoothness sum, Gauss3 reconstruction,
Gauss4 transport/source/face rule, Gauss8 projection, unlimited upwind material
operator and all SSPRK3 stages. Only the candidate polynomial passed to the
unchanged smoothness formula differs. No component bounds, clipping or chi
limiting. The scalar comparison is the existing proportional signed tensor pulse.

First reproduce the old 32x64 base-step outputs. Then independently check KKT
maps, stationarity/constraint, global affine and complete Q2 including x²y²,
zero/constants, non-proportional patches, mean preservation, finite normalized
weights, amplitude scaling, in-plane tensor-basis covariance and traversal order.
Local identity targets are 1e-12 relative where conditioned, KKT/assembled
comparisons 1e-10, physical mean drift below 1e-9 Pa. Inspect failures without
relaxing tests or changing formulas to improve transport results.

On 32x64 run sharp and fixed ell=.15625 tanh pulses at dt, dt/2, dt/4, both
detectors and both constructions, plus the unchanged unlimited baseline. The
base pulse has 103 SSPRK3 steps to .2 s, v=(.4,0); finite profile uses
delta=ell/ln(9), including transition overlap. Trace all physical means before
and after each limiter call; retain all-cell maxima/integrals plus the complete
y=.5+hy/2 row. At fixed call/time samples record candidate coefficients, beta,
weights, indicators, face-averaged values and subsequent native mean-RHS flux
changes. Test 12 held-field limiter calls without transport. Separate initial
projection extrema, errors, front widths, direct/RK-weighted norm changes,
peak attenuation, temporal differences and cost. There is no threshold or
coefficient search. Every run is capped at 180 seconds, one rank/thread,
at most two concurrent, splitting detector/case jobs to respect that bound.

Use the existing PF4 transport screen for temporal contamination: the final
dt/2-to-dt/4 full-tensor field difference should be at most 1% of the dt/4
analytic error and decrease from the dt-to-dt/2 difference. This is a bounded
fixed-grid screening criterion, not a coupled-physics tolerance or proof of a
limiter's continuous-time limit. Also report actual peak/front-width changes;
do not claim convergence merely because overshoot is smaller.

Improvement in this linear transport fixture would support only this candidate
adaptation. Failure to improve or uncontrolled timestep sensitivity is retained
evidence, not permission for compensating diffusion or automatic adoption.
