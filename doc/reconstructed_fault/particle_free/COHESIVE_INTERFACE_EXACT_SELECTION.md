# Bounded exact-jump fixed-surface qualification

Selected by the user after the native-component report. Preserve production/BP3,
prior evidence, the bond/contact law, RSF parameters, tolerances, Gauss3 surface
histories and signed-rate ownership. Serial fixed horizontal through-going surface,
periodic x, the same two clamped layers (heights .075/.05 m). No birth, propagation,
old-history conversion, MPI or production Stokes-driver replacement.

Use u=u_bar+(H(y-yΓ)-Fχ(y)) δ_h e_x. Its regular lift strain is
L_j=dev3[-χ N_j S+(H-Fχ) N'_j e_x⊗e_x], and divergence d_j=(H-Fχ)N'_j.
Retain both in the mixed assembly. Allow separate native Q1 pressure traces at
Γ by duplicating only interface pressure supports, with corresponding periodic
constraint expansion. Continuous u_bar and the known enrichment give exactly
[ux]=δ_h, [uy]=0. Pressure is free to jump as required by normal traction; do not
impose a fixed-pressure closure or add a pressure penalty.

A W=-C, C_ij=∫(2 dev ε_i:L_j-p_i d_j). Reconstruct total regular elastic strain
E_j=dev ε(W_j)+L_j. Form relaxed K from ∫2G E_i:E_j, and independently check
mixed generalized work including p d_j. Use actual two-sided normal traction
traces (arithmetic mean) for friction; export their mismatch as FE error. This
mean is a consistent numerical flux for the continuous physical normal traction,
not an average of constitutive directions or ages. Gauge: zero bulk-mean p′.

Before running: original force <=1e-8 Pa, tangent FD <1e-6, ledger <=1e-10 max(1,work),
uniform traction agreement <=1e-7 Pa. Generalized volume work/stress/energy checks
retain prior tolerances. Test same-space sharp Fourier K and N; target <2% for the
resolved 1 m mode and decreasing two-sided traction mismatch under one bounded
refinement (8/16/32 along x, up to 160 normal). Separate trace approximation from
algebraic work identities; do not force trace mismatch to roundoff. Same-grid
width change should leave sharp operators invariant to 1e-9 relative because
Fχ changes by an exactly represented continuous Q2 field when support knots
align. Bulk Gauss3/Gauss6 comparison target 1e-10 relative.

Reuse short uniform initiation and mature reversal, nonuniform developed sliding,
complete current-pressure tangent, rejected candidate/retry/duplicate publication
and real ASPECT restart. Compare new law returns against the independent reference
using identical accepted steps. Report physical-work error separately from BE
closure. Preserve any failure; no tolerance or physical-parameter tuning.
