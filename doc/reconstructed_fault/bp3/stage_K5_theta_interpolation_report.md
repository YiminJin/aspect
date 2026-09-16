# K5: nodal aging versus pointwise aging on the last two RSF elements

## Result and scope

**The two operations do not commute, and their late discrepancy is large.**
After update 12, interpolating the updated nodal state gives a last-element
midpoint Theta about **18 times** the value obtained by updating the
interpolated preceding state with the interpolated accepted rate. This occurs
on both saved fault grids. Early in the trajectory the corresponding
difference is only **0.0164%**.

The source and the saved friction samples confirm the split timing:
mechanics 13 consumes Theta12, not newly committed Theta13 and not Theta11.
No extra computational lag or implementation/specification conflict was
found. The specification explicitly requires the current nodal update.

The reconstructed, correctly timed change in the last unprescribed row's
weak friction density is approximately **-1.4555 / -1.3170 MPa** for the
100 / 50-m grids. This is potentially material relative to the existing
**0.9572 / 1.0986-MPa** bound reactions.

**Important completeness limit:** the saved data contain complete weak
moments and selected raw samples, not the complete quadrature-point list.
The state comparison and selected-sample friction changes below are exact
to floating-point accuracy. The full alternative weak loads are explicitly
**estimates**, not a reassembly of the original production quadrature.
Neither active-set release nor a correction is claimed as verified.

No ASPECT execution, constitutive/history change, alternate mechanical solve,
or trajectory continuation was performed. All original artifacts are intact.

## 1. Exact path to friction quadrature

The last two elements mean the **RSF-side elements next to the 40-km
prescribed-slip junction**, not the bottom tip of the complete fault:

| Grid | Penultimate RSF element | Last RSF element | Last unprescribed node |
|---|---|---|---|
| 100 m | 39.8--39.9 km | 39.9--40.0 km | 39.9 km |
| 50 m | 39.9--39.95 km | 39.95--40.0 km | 39.95 km |

For a domain quadrature point on segment j, with N0=1-xi and N1=xi:

1. [The surface assembler](../../../source/reconstructed_fault/surface_system.cc#L329)
   uses the domain partition's own segment and xi, **not the parent's single
   projection coordinate**. It evaluates the explicit current/trial rate as
   Vj + xi(Vj+1-Vj), with exact endpoint branches reading the stored nodal
   values. A lower-active node has Vmin exactly; interior quadrature points
   still have the Q1 rate, generally much greater than Vmin.
2. [The material point response](../../../source/material_model/phase_field_fault.cc#L621)
   reads the **committed surface-property Theta**, through
   [the scalar interpolator](../../../source/material_model/phase_field_fault.cc#L315):
   Theta(xi)=N0 Theta_j + N1 Theta_j+1. It does not interpolate log(Theta),
   apply aging at that point, or interpolate nodal friction coefficients.
   After initialization it does not obtain the friction state from the
   particle's initial-Theta field or its bulk FE transfer.
3. It evaluates mu from these interpolated V and Theta and the surface
   mixture. Throughout these last two elements the BP3 mixture is fully
   velocity-strengthening: **a=0.025, b=0.015, mu0=0.6, V0=1e-6 m/s,
   Dc=0.008 m**, with the regularized asinh law. The normal traction is the
   correctly timed particle-history constitutive value, including background
   and current bulk response. Bulk/history inputs remain parent-P0; surface
   Q1 inputs vary over the domain.
4. The positive weak friction contribution is

   \[
   L_{i,k}^{\rm fric}=\sum_{p,q}w_{pq,k}N_i(s_{pq,k})
     \sigma_{n,pq,k}\mu(V_k(s_{pq,k}),\Theta_{k-1}(s_{pq,k})).
   \]

   It enters the physical surface residual with a **minus sign**. MPI sums
   unique real-parent contributions; no nodal value is substituted for the
   integrated nonlinear response.
5. Only after convergence does
   [the history update](../../../source/material_model/phase_field_fault.cc#L1180)
   call the [exact aging law](../../../source/material_model/rheology/fault_friction.cc#L33)
   **at each Q1 vertex**, using accepted V, previous nodal Theta and dt.
   The global Dc is unchanged. The supplied timestep-zero Theta is retained.

This agrees with `current_design.md`, section 25, and the explicit Q1-vertex
aging rule in `specification.tex` near line 1721. Replacing it by a different
state representation would be a discrete-formulation decision, not a repair
of a discovered violation of the present specification.

## 2. The two operations and the correct clock

Write

\[
 U(T,V,d)=T e^{-x}-\frac{D_c}{V}\operatorname{expm1}(-x),
 \qquad x=Vd/D_c.
\]

The compared states after update k are

\[
 \Theta^A_k(s)=\sum_jN_j(s)U(\Theta_{j,k-1},V_{j,k},\Delta t_k),
 \qquad
 \Theta^B_k(s)=U\!\left(\sum_jN_j(s)\Theta_{j,k-1},
                       \sum_jN_j(s)V_{j,k},\Delta t_k\right).
\]

A is the saved production state. B is a **one-update counterfactual** using
the same saved preceding Q1 state and accepted rate. It is not a recursively
evolved alternative trajectory. Both agree at every vertex; between vertices
B is generally not Q1.

The friction comparison is then

\[
 \Delta L_{i,k+1}^{\rm fric}=\sum_{p,q}w_{pq,k+1}N_i\sigma_{n,pq,k+1}
 [\mu(V_{k+1},\Theta^B_k)-\mu(V_{k+1},\Theta^A_k)].
\]

It uses **the following solve's accepted V, stress and domain measure**.
Using either updated state in mechanics k would incorrectly change the
approved split scheme into an implicit/same-step comparison. Domain motion
does not change this indexing: the saved Q1 fields are evaluated at the
following state's quadrature coordinates on the fixed fault.

| Test | Update k | Update time (s) | Update dt (s) | Friction evaluation |
|---|---:|---:|---:|---|
| Early, before strong depression | 4 | 18770511.282268524 | 9057973.508357061 | step 5, t=36071240.68323051 s |
| Late, first contact update | 12 | 2165471357.7276783 | 402695264.3023224 | step 13, t=2232176379.2516127 s |

## 3. State discrepancy from the saved nodal histories

Midpoint values, in seconds:

| Update | Grid | Element | A: update nodes, interpolate | B: interpolate, update | A/B |
|---|---|---|---:|---:|---:|
| 4 | 100 m | last | 8.125011e6 | 8.123674e6 | 1.000164634 |
| 4 | 100 m | penultimate | 8.214671e6 | 8.214566e6 | 1.000012838 |
| 4 | 50 m | last | 8.124773e6 | 8.123441e6 | 1.000164010 |
| 4 | 50 m | penultimate | 8.215892e6 | 8.215796e6 | 1.000011632 |
| 12 | 100 m | last | 2.907853e8 | 1.600000e7 | **18.17408** |
| 12 | 100 m | penultimate | 2.961808e8 | 3.758324e7 | **7.88066** |
| 12 | 50 m | last | 2.939287e8 | 1.600000e7 | **18.37054** |
| 12 | 50 m | penultimate | 3.048315e8 | 5.965885e7 | **5.10958** |

At update 4 the last nodal V/Vp is about 0.958, and its new Theta is about
8.25e6 s. Both methods therefore see almost uniform rates and states.

At update 12, V=1e-20 m/s at the last unprescribed node while V=Vp=1e-9
at 40 km. The contact node ages almost by **+dt**:
Theta11=1.70875e8 / 1.77162e8 becomes
Theta12=5.73571e8 / 5.79857e8 s. The prescribed endpoint stays at 8e6 s.
Interpolating these updated endpoints spreads the large nodal state across
the whole element. But at the element midpoint Vk=Vp/2, x=25.16845, so the
pointwise update has almost fully relaxed to **Dc/(Vp/2)=1.6e7 s**.
This explains the mismatch without a cancellation error or a clock error.

Its maximum ratio on the last element is 18.1776 / 18.3740. Halving the
element size does not halve this state-amplitude mismatch; it confines a
similar nodal transition to a shorter physical interval. This is consistent
with the preceding fault-grid result, but is not by itself a proof that this
mechanism created the original rate depression.

## 4. What the raw saved quadrature samples establish exactly

There are **80 distinct saved production samples per grid/test**, 320 total
in these two-element windows. Recomputing their friction with Vk+1 and
Theta^A_k reproduces the saved mu to a maximum absolute error
**1.832e-15**. Using newly committed Theta_k+1 instead gives discrepancies
up to 1.661e-3; adding an extra lag and using Theta_k-1 gives discrepancies
up to 1.816e-2. Thus the saved friction really used the intended state.

At these same points, retaining their actual sigma_n and integration weights:

| Friction step | Grid | Range of sample delta(mu sigma_n), Pa |
|---|---|---:|
| 5 | 100 m | -123.632 to -0.00221 |
| 5 | 50 m | -123.174 to -0.10575 |
| 13 | 100 m | -4.61456e6 to +0.23136e6 |
| 13 | 50 m | -4.77477e6 to +0.46771e6 |

The positive late values occur on tensile samples: reducing mu makes their
negative friction contribution less negative. No tensile clipping is used.
The exact per-sample changes, including w N0 and w N1 weak contributions,
are exported separately. These extrema-selected points **must not be summed
and presented as the full weak load**.

## 5. Full weak-load estimates and their limitations

For each following state, reduce the saved all-sample mass matrix M,
normal load b_sigma and actual friction load over four ranks. On the same
fault grid construct the ordinary line mass matrix M_line and the Q1 signed
normal-load density t_h by

\[
 M_{\rm line}t_h=b_\sigma.
\]

This reproduces all saved Q1 normal weak moments exactly. Evaluate
integral N_i t_h(s) delta_mu(s) ds on each of the two elements. This is a
**moment-based reconstruction of the omitted within-element data**, not the
original domain quadrature. It does not substitute FE particle history.

| Following friction step | Grid | Actual production L_fric/m (MPa) | Estimated delta L_fric/m (Pa) | Estimated delta L_fric (Pa m²) |
|---|---|---:|---:|---:|
| 5 | 100 m | 26.466909 | **-44.325** | -7.00631e6 |
| 5 | 50 m | 26.466427 | **-43.863** | -3.47090e6 |
| 13 | 100 m | 26.808730 | **-1.455474e6** | -2.30062e11 |
| 13 | 50 m | 26.564084 | **-1.317038e6** | -1.04214e11 |

These are the last unprescribed row, whose whole support consists of the
two audited elements. The late density changes split as follows:

| Grid | Last element contribution (MPa) | Penultimate contribution (MPa) |
|---|---:|---:|
| 100 m | -0.874350 | -0.581124 |
| 50 m | -0.878369 | -0.438668 |

The estimated total friction-load changes, summed over all three test
functions on the two-element patch, are -4.67699e11 / -2.12164e11 Pa m²
at the late state. These patch totals include the prescribed endpoint's
weak load and are **not** a bound-reaction total.

Checks on the reconstruction:

- Re-evaluating the **original** friction load gives last-row errors of
  -5.01 / +7.61 Pa early and -6066 / -7855 Pa late (at most about 0.030%
  of the original late friction load).
- A separate reconstruction uses a Q1 measure density matching M times one,
  multiplied by the actual consistent-Q1 normal-stress projection. It changes
  the late delta estimates by **less than 0.57 Pa**. Using 50 MPa only as an
  additional diagnostic normal scale changes them by about 3.7 / 4.2 kPa;
  it is not a change to the production pressure treatment.
- Reconstructed mass diagonal/offdiagonal comparisons over these rows have
  maximum relative discrepancy **0.001401**. This explicitly confirms that
  the reconstruction is not identical to the original domain rule.
- Increasing offline Gaussian order from 128 to 256 changes the computed
  delta by less than **9e-9 Pa**. Integration error of the reconstructed
  functions is negligible; this does **not** bound the missing subelement
  normal-load information.

The saved normal moments and good original-load reproduction do not uniquely
determine the alternate nonlinear weak load. The checks support the scale of
the estimate, not a certified error bar. In particular, the 6--8-kPa original
load discrepancy must not be asserted as a rigorous bound on the alternative.

## 6. Implication for the bound and next decision

The early effect is negligible on the approximately 26.5-MPa friction scale.
The late estimate is about **5.43% / 4.96%** of the existing friction load
and **1.52 / 1.20 times** the existing local bound reaction. Since the
residual subtracts friction, the estimated change in physical R/m is
positive. At fixed other inputs it would change the existing -0.957235 /
-1.098618 MPa to approximately **+0.498239 / +0.218420 MPa**.

This is a strong candidate for material state-representation sensitivity,
not a demonstrated active-set release: neither an exact complete-domain
alternative load nor a re-equilibrated solution was computed. It also does
not establish that the nodal update alone caused the preceding depression;
the altered comparison starts from the already developed saved trajectory.

**Smallest next action:** obtain one complete, frozen step-13 quadrature
evaluation restricted to these two elements, with both Theta constructions,
before changing the state representation. It needs no nonlinear solve and
must preserve Vk+1, stress, weights and the k-to-k+1 timing. The omitted
data are precisely (segment, xi, weight, sigma_n) at every contributing
production quadrature point; saved nodal histories already provide V and
both Theta constructions. No additional convergence campaign is indicated
by this audit alone.

## Artifacts and checks

Runner (offline only):

```sh
python3 benchmarks/reconstructed_fault/bp3/analyze_theta_interpolation.py
```

- [State profiles](../../../benchmarks/reconstructed_fault/bp3/theta-interpolation-audit/theta_profiles.csv)
- [Exact comparisons at selected production samples](../../../benchmarks/reconstructed_fault/bp3/theta-interpolation-audit/selected_production_samples.csv)
- [Full weak-load estimates](../../../benchmarks/reconstructed_fault/bp3/theta-interpolation-audit/weak_load_estimates.csv)
- [Element contributions](../../../benchmarks/reconstructed_fault/bp3/theta-interpolation-audit/element_weak_load_estimates.csv)
- [Checks, source-data hashes and indexed inputs](../../../benchmarks/reconstructed_fault/bp3/theta-interpolation-audit/summary.json)
- [Analysis log](../../../benchmarks/reconstructed_fault/bp3/theta-interpolation-audit/analysis.log)

Saved inputs are `junction-matched-qualified-local4/refined/` and
`fault-grid-50-local4/`. Nodal exact-aging reproduction, endpoint equality,
uniform/contact reproduction, fixed coordinates, MPI moment reduction and
the correctly timed raw-mu reproduction all pass. The offline calculation
takes about one second. Only this analysis script, its output directory and
this report were added; production code and saved trajectories were not
modified. No ASPECT build or simulation test was needed.
