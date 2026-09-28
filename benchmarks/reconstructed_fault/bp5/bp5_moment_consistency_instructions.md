# Focused moment-consistency comparison for the clean stress cycle

## Task and scope

Implement and run a benchmark-only comparison on the existing horizontal,
periodic, fixed-particle clean-start fixture. Determine whether preserving the
weak stress load across history publication removes the spurious next-step
forcing, while preserving the physical mean shear accumulation.

Use the uploaded `stress_cycle_report.md` and clean-start `README(1).md` as
evidence. They establish: (i) the saved BP5 particle history already contains
large subcell stress; (ii) the clean fixture generates quadrature-scale stress
from zero history; (iii) its first generated mode has zero resolved equilibrium
moments; and (iv) the actual publication/transfer introduces a nonzero weak load.
They do not establish the origin of the full BP5 normal-stress amplitude.

Implement the small comparison below, not a general replacement stress-storage
architecture. Do not start a BP5 trajectory, change I_h, smooth fault properties,
change solver tolerances, or repair the unrelated near-zero-source pressure
guard as part of this task. Keep production defaults unchanged. A narrowly
scoped, default-off hook is acceptable if the benchmark needs one; document it.

## 1. Preserve the qualified fixture

Use the successful finite-source problem, including:

- 0.25 x 1 m box; 16 x 64 square cells; periodic x; Q2 velocity/history;
  the existing Q1 phase/fault representation and native 3 x 3 bulk quadrature.
- The same horizontal fault, ell/h = 10, stationary phase profile, frozen
  particle positions, and production localization/normalization.
- G = 1e6 Pa, eta = 1e20 Pa s, prescribed V = 0.005 m/s, and unchanged
  top/bottom loading. These are prescribed rates, not a prescribed bulk solution.
- Zero initial stress; unchanged artificial initialization treatment; four
  real accepted steps of actual dt = 0.1 s.
- The same 3 x 3 parent layout and ordinary state lifecycle.

Run each branch independently from zero. Do not sequentially switch treatments
on an already evolved branch. Log parameter differences and initialization
state hashes. Proposed new benchmark mode names below are declarations to add,
not existing production PRM keys.

## 2. Define the quantity that must be preserved

For a tensor field sampled at native quadrature points, define its assembled
free velocity load by

\[
  F_i(T)=\sum_{K,q} JxW_{Kq}\,
               T_{Kq}:\varepsilon(v_i)_{Kq},\qquad
  F_f(T)=C_v^T F(T).
\]

Here C_v maps independent homogeneous velocity variations to full velocity
DoFs. Apply periodic identifications and any hanging-node constraints correctly;
eliminate prescribed velocity variations only after the appropriate constraint
assembly. Use the production sign convention consistently. Include *all* free
velocity tests, not just the tangential component or divergence-free tests.

Use the same actual bulk integration measure and shape gradients as mechanics.
Do not insert an extra chi weight; localization is already part of the
constitutive tensor. For xx, yy, xy components, the symmetric tensor contraction
contains the factor 2 on the xy product.

Let T_k be the fully accepted current constitutive stress at real step k, and
H_(k+1) the actual incoming representation prepared for the next solve. Measure

\[
   J_k=\|F_f(H_{k+1})-F_f(T_k)\|_2.
\]

Evaluate this at fixed accepted geometry, velocity, and fault source, before
another mechanics solve. It is a representation jump, not that next solve's
nonlinear residual. Report also the beta-weighted history-force comparison
using the actual next-step beta. Beta is effectively one in this fixture, but
do not apply it twice or silently replace it in constitutive calculations.

The target is equality to the old accepted load, not a requirement that the
stress load itself always be zero. Pressure and boundary reactions can balance
a nonzero stress load in other cases.

Free-load preservation alone cannot detect loss of a spatially uniform shear
stress. Therefore also retain and compare cell means, domain mean tau_xy, and
top/bottom tangential reactions. These protect the physical loading mode.

## 3. Three branches, four accepted steps each

### A — `production`

Run the existing retained-parent Maxwell update and complete production
particle-to-FE transfer, including incident-cell ADD/count averaging and the
actual working-field constraints. This is the baseline and verifies that new
benchmark switches do not change the default path.

Expected reference values from the previous report:

- First three transfer jumps: approximately 0.0672939, 0.116882, 0.168486 Pa m.
- Within-cell particle tau_xy RMS after steps 1–4: approximately
  1.54453, 3.08905, 4.63393, 6.17999 Pa.
- Physical mean shear increases by about 500 Pa per real step.

### B — `native_history_reference`

Use accepted native-quadrature stress as the authoritative history for this
branch. This is an exact-transfer reference on the fixed mesh, not a proposed
production architecture.

Initialize H_0(K,q) = 0. During a solve, retain H_(k-1) unchanged and use the
same production constitutive formula:

\[
 T_k(K,q)=\beta_k H_{k-1}(K,q)
       +2\kappa_k\varepsilon(u_k)(K,q)
       -2\kappa_k\chi(K,q)V_k S(K,q).
\]

Use the actual strain operator from the production implementation if it differs
from this notation. After the single accepted commit, set H_k(K,q) = T_k(K,q).
Do not update history during Newton iterations or rejected candidates. Reuse
the production constitutive evaluation rather than independently rewriting it.

All mechanics accesses to retained stress in this branch must use this same
history. Do not override only a diagnostic or add a cancelling RHS force while
leaving constitutive stress inconsistent. If non-native-point values are needed,
a cell-local tensor-product Q2 polynomial through the nine native Gauss values
provides an explicit reconstruction with exact agreement at those points. Keep
cell traces distinct; do not globally average it into continuous Q2. List every
history consumer that was redirected, including any surface traction coupling.

Particles may remain for phase/other properties and shadow diagnostics, but
their stress must not silently feed back into branch B. The meaningful history
roughness measure here is native-QP stress, not an unconsumed particle field.

J_k should be at assembly roundoff. Subcell stress may continue to accumulate:
that alone is not failure. This branch tests whether exact retention prevents
a mechanically invisible mode from becoming a new load through transfer.

Quadrature-point history is a standard implementation option; see deal.II
step-18: https://dealii.org/developer/doxygen/deal.II/step_18.html . This reference
supports the storage technique, not qualification of this custom fault model.

### C — `horizontal_moment_update`

Test a simple, explicitly fixture-specific resolved-history publication rule.
Its moment consistency must be measured *after* the complete production
transfer; it must not be assumed from the local fit.

First verify that the accepted field remains tangentially homogeneous to the
solver's numerical accuracy. For each cell K, use native JxW weights and
xi = (y-y_K)/h_K. Fit the fully accepted current tensor T_k to

\[
  T_K^*(y)=A_K+B_K\xi,
\]

component by component for xx, yy, xy. Determine A_K and B_K from

\[
 \sum_q w_q(T_K^*-T_k)=0,\qquad
 \sum_q w_q\xi_q(T_K^*-T_k)=0.
\]

Solve the 2 x 2 moment system using the actual weights; do not replace it with
an unweighted fit to parent values. On this symmetric cell quadrature,
A_K is the weighted cell mean and B_K is the first moment divided by
sum(w_q xi_q^2).

Publish parent stress by **replacement**:

\[
   P_{k,p}^{new}=T_K^*(y_p).
\]

The accepted T_k already contains the inherited stress and current increment.
Do not add beta P_old again. Do not project just the latest increment and then
add the unresolved retained parent tensor back. That would preserve the very
history component this candidate is intended to remove.

Now run the complete existing production DWA -> incident-cell averaging ->
continuous Q2 -> physical-constraint path. The resulting *actual working field*
is the history consumed by step k+1. Apply the alternative publication exactly
once after acceptance, with all other updates unchanged.

For the reported representative first-step cell, the Gauss values
(494.1645261204, 507.2960583940, 494.1645261204) Pa reduce to an approximately
500.0007626864 Pa constant shear with zero slope. This is a useful verification
of the local fit, not a value to hard-code across cells or future steps.

Why this is a reasonable narrow candidate: under the fixture's x-invariant,
periodic shear symmetry, the normal-direction Q2 shear test derivatives require
the constant and linear normal moments. The nonlinear localized source produces
a higher subcell mode. The periodic assembly is essential to this argument.
It is *not* a general statement that {1,y} preserves all local 2D Q2 tensor
moments. In particular, derivatives of general 2D Q2 tests contain additional
mixed/higher terms. Do not generalize this rule to the inclined BP5 fault.

Separate the following diagnostics:

\[
 J_k^{proj}=\|F_f(T_k^*)-F_f(T_k)\|_2,
 \quad J_k^{map}=\|F_f(L(P_k^{new}))-F_f(T_k^*)\|_2,
 \quad J_k^{total}=\|F_f(L(P_k^{new}))-F_f(T_k)\|_2.
\]

L includes the actual constraints and publication path. Retain the signed
vectors or their inner products: norms are not additive and cancellation must
not hide two large errors. Check cell moment differences after L as well as
before it. A local fit that passes while final transfer fails is not a
moment-consistent cycle.

C can still regenerate the same pointwise zero-moment mode in its current
constitutive tensor on every solve. Distinguish that current tensor from the
history retained after publication; do not demand that both be pointwise smooth.

## 4. Measurements and acceptance

For each accepted step, write one compact summary record containing:

- Actual dt, beta, kappa; solver residuals and iteration counts; history mode.
- Current and next incoming stress means; mean-shear increment; tangential
  boundary reactions, with pressure handled consistently where relevant.
- Native-QP within-cell tensor RMS; consumed parent-history RMS for A/C;
  quadrature norm of stress removed by C. Label measures and weights explicitly.
- J_proj/J_map/J_total where applicable; incoming/current load norms; corresponding
  maxima; cell zeroth/first moment discrepancies.
- Bulk velocity and native strain-gradient differences from B at matched physical
  steps, using actual fields. Distinguish the imposed loading from the extra
  re-equilibration response. A converged final residual is not a substitute.
- Changes in particle positions, phase, chi, prescribed V: expected zero.

Use the independent FEValues load calculation already checked in the previous
test. Keep its comparison with production assembly. Do not normalize load
differences by the near-zero net first-step load. Use a noncancelling scale from
the magnitudes of element/quadrature force contributions, and report dimensional
norms in Pa m as well.

Predeclare these targets rather than loosening them after observing the results:

1. A reproduces the baseline within explained solver/build roundoff.
2. B's load jump is at the independently measured assembly floor. A larger jump
   is a reference implementation/lifecycle problem and must be investigated
   before drawing conclusions about C.
3. For C, seek at least a 10^4 reduction of each nonzero baseline transfer jump,
   with absolute load error at or below max(1e-9 Pa m, 1e-10 F_abs). F_abs is the
   explicitly documented noncancelling assembly scale. Report a partial reduction
   honestly if this target is missed; do not change the tolerance to obtain a pass.
4. C's physical mean-shear increment and boundary loading remain consistent with
   B (target relative differences <= 1e-6 on their nonzero physical scales).
   A zero-stress field can pass a free-load test and still fail this requirement.
5. C substantially reduces the accumulating unresolved particle content and
   agrees with B's resolved response to the extent justified by the measured
   load error. Do not require B's pointwise subcell stress to be smooth.

Raw first-step quadrature values and a few cells' moment/transfer traces suffice;
do not export domain-wide giant CSVs. Preserve existing checks against incorrect
time levels and repeated commits. For this new logic, test moment preservation,
constant retention, and removal of the reported zero-moment mode using the same
3-point Gauss weights; these tests verify mathematical properties, not copies
of the implementation.

## 5. Run budget, interpretation, and stopping rule

Start with three one-rank branches, four real accepted steps each, using the
existing 120-s per-run cap and output-overwrite protection. Inspect B before
interpreting C. If C meets the gates, perform one two-rank replay of C to check
periodic/incident-cell publication and compare by physical cell/parent identity.
Stop after that. No resolution, particle-count, interpolation, or timestep sweep.

If J_proj is large, the symmetry-specific fit is insufficient or incorrectly
assembled. If J_proj is small but J_map is large, the original transfer still
changes the moments: do not call C a qualified correction. Report which stage
fails before designing a general constrained transfer. Do not silently replace
DWA with another interpolator to make this comparison pass.

If B preserves loads and C also preserves loads/physical loading while suppressing
retained subcell content, the supported conclusion is that a consistent history
publication can break the demonstrated forcing cycle in this fixture. It is
not proof that pointwise subcell stress is intrinsically invalid, that all MPa
normal bands have this origin, or that stress filtering is energy-consistent.

The next scientifically distinct test would be a small inclined prescribed-slip
fixture with total normal-traction diagnostics. Free RSF, variable coefficients,
particle advection, mesh transfer, long-time energy behavior, and production BP5
remain outside this qualification. Do not launch that next stage in this task.

## Deliverables

- Benchmark-only source/hook diff and three complete mode-specific PRM files,
  with the exact new parameter declarations and build/run commands documented.
- Compact machine-readable comparison and one short Markdown report: baseline
  reproduction, where each jump occurs, physical-loading preservation, and a
  clear pass/partial/fail conclusion for each branch.
- At most two figures: load-jump versus step (absolute and normalized), and
  physical mean shear plus within-cell stress amplitude versus step. Avoid
  presenting different statistical measures as directly interchangeable.
- Exact record of any unrun/failed control and any remaining implementation
  limitation. No claim that a production correction has been selected.
