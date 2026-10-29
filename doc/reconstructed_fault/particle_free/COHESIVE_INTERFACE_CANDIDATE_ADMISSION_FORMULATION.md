# Formation, provisional geometry and physical work

This is the opt-in serial exact-jump cohesive–RSF experiment. It does not change
the production reconstructed-fault registry or its constitutive backend.
[Declared controls](COHESIVE_INTERFACE_CANDIDATE_ADMISSION_SELECTION.md).

## Physical registry and trial view

Write the computational view as Γv=Γf∪Γc. Γf is persistent formed geometry;
Γc is temporary, with a=s=sref=m=absolute_slip=0 at every owned Gauss point.
Here m is interface/Maxwell lifting memory, not the generally nonzero bulk
Maxwell tensor. All bulk history and its physical stress are preserved.
Candidate age is the existing prescribed virgin clock, not established sliding
history. A cache with nonzero physical history, nonzero associated nodal mode,
or inconsistent virgin age cannot be retired, regardless of its magnitude.

One new Q1 surface cell is the atomic geometric record. Any nonzero physical
history makes that cell persistent; there is no damage threshold. Geometry,
directions and Gauss identities of persistent cells never change. Zero Gauss
points within a partly formed cell share its fixed geometry. This remaining
one-cell extent is numerical and requires refinement before quantitative path
claims. Disconnected formation is detected and rejected, not hidden by a mask
or silently bridged with committed virgin geometry.

Existing polynomial pieces are retained verbatim; new pieces each contain one
cell. No polynomial restriction/reparameterization or Gauss-history interpolation
is needed. The initial .410 s checkpoint has two already formed pieces and no
wholly virgin piece requiring reinterpretation.

## Unchanged law and residual

With e=s−sref, the interface energy per physical area is

\[
 B(e,a)+G_c a=\frac{1-a}{2C_0a}e^2+G_c a,
 \quad C_0=2G_c/\tau_p^2,\quad d_f=2G_c/\tau_p.
\]

The intact extension is B(0,0)=0 and B(e≠0,0)=infinity. The retained update is
`a_new=max(a_old,min(|e_new|/df,1))`. Condensation of virgin damage gives
`F(e)=tau_p |e|−tau_p e²/(2 df)` up to df, then Gc. Thus intact admission is the
signed subgradient `|lambda|≤tau_p`, and complete decohesion costs Gc exactly
once. There is no separate energy charge when a record enters the registry.

The unchanged sliding resistance is q+a sign(V)[pn mu(|V|,theta)+eta_rad |V|],
with the retained regularized RSF and implicit age update. The pressure is the
current variational normal reaction `pn=p0+N c`, with p0=10000 Pa. It is not
lagged or replaced by prescribed friction pressure. Total physical stress is
`sigma=tau−(p0+pprime)Id`, with mean pprime=0 and full 3D plane-strain tau.
There is no opening or dilation.

The two prescribed lab-frame affine memories and every signed lifting memory
advance with the same Maxwell recurrence:

\[
 c_b^{n+1}=\beta(c_b^n+\Delta b),\qquad
 c_m^{n+1}=\beta(c_m^n+\Delta d),\qquad
 \beta=(1+G\Delta t/\eta)^{-1}.
\]

Using native mixed bulk-condensed K, the complete fault residual is

\[
 R=(Kc)_m+Q^TW r,\qquad
 J=\beta K_{mm}+Q^TW[r_sQ+r_p\beta N_m].
\]

The derivatives include damage, implicit age/rate and current pressure. This
nonsymmetric operator is unchanged. Tied Gauss constraints require Q Δd=0
and a bounded signed multiplier; all rows enter the weak certificate.

For independent Q1, three tied Gauss values imply zero at both local nodes
(one free node on an endpoint cell). The optional solver eliminates exactly
those coordinates before SVD of the remaining constraints. It solves in the
same admissible space C using CᵀR and CᵀJC, without penalty diagonals, clipping,
or a phase/rate floor. This prevents numerical SVD shadows being mistaken for
new physical memory. Incoming history is never zeroed; defaults retain the old
projection path. The independent projector/native comparison quantifies equivalence.

## Representation change versus formation

All kernels share the same fixed bulk space. Retained nonzero physical basis
columns and their native response/pressure vectors match bitwise. Removed
candidate coordinates are exactly zero; retained coefficients are copied.
Consequently a pure candidate replacement preserves the entire represented
bulk tensor/pressure field, rather than merely its coefficient labels. Old
Gauss coordinates, weights, histories and IDs remain unchanged. Direct stress
and storage defects are checked with the existing allowances.

An overloaded candidate is **not** an admissible tied birth. The legacy tied
birth guard remains intact. A new temporary view may be used as the input
representation of a positive-time joint solve, including when its virtual
intact multiplier exceeds tau_p. It is not published as material in that
state. Only the joint endpoint satisfying the complete native weak/contact/
tangent/history checks can be published. Its actual damage pays Gc Δa and its
cohesive storage and sliding dissipation are included normally.

The exact backward-Euler identity is retained:

\[
 W_{end}=\Delta E_{bulk}+\Delta B+G_c\!\int\Delta a\,ds
 +D_{fr}+D_{rad}+D_M+E_{BE}+R_{surf}.
\]

Here `E_BE=.5 Delta c^T K Delta c`, `D_M=2G/eta dt E_bulk_new`, and
`R_surf=int(q_new Delta s)−Delta B−Gc int(Delta a)` is the explicit endpoint
surface remainder; it need not be positive during softening. Friction and
radiation dissipation are nonnegative in maintained contact. Independent
trapezoidal reaction–displacement work and the sum of absolute per-step physical
remainders assess accuracy, separately from algebraic ledger closure.

Classification of that already-solved endpoint changes metadata only:
the energy computed on all Γv equals the persistent Γf inventory because the
remaining candidate contribution is exactly zero. No second Gc times newly
registered length, no history reset, and no hidden transfer work is introduced.
Old stress/storage preservation is checked before the physical trial; formation
energy belongs to the physical update, not to a zero-time geometry publication.

## Direction, candidate completion and lifecycle

The same maximum-shear/persistence hypothesis samples accepted stress in the
R=.75 m disk R ahead of the **formed endpoint**. This anchor replaces the old
unused-reserve endpoint explicitly. Both conjugate planes remain available;
the retained nearest-history preference resolves a non-symmetric continuation.
New curves obey `dtheta/ds=(theta_star−theta)/R` and join with continuous tangent.
No old curve or tensor is rotated. R and numerical lifting half-width .75 m
remain distinct from the number of temporary cells.

Fixed Gauss8 bulk-cell integration of the selection disks removes dependence
on candidate cut partitions; Gauss12 checks this numerical choice. Mechanics,
normal-work, source and energy quadratures remain unchanged. Sampling-disk
accuracy is not a tip point-stress certificate.

Candidates are cached as numerical predictors between declared refresh clocks.
Any trial forming a new cell is rejected if its direction was not selected from
the actual incoming accepted state. Rebuild and re-solve from that same state;
do not compose histories from the rejected predictor. Enlarge and retry if trial
growth reaches the final candidate cell. A wholly zero witness cell must remain;
there is no one-cell-per-step growth mask. Completion/retry loops are bounded.

After native acceptance, registry promotion and history publication occur
atomically once. Candidate-cache replacement alone changes its view version,
never material generation, physical publication or clock. Restart stores the
provisional predictor/cadence as numerical state. Same-cache restart is checked
bitwise; changed-cache comparisons force the same fresh selection in both
controls. Arbitrarily changing a stale predictor between refresh clocks is
not assumed to be an exact algorithmic identity; cadence sensitivity is measured.

This does not supply a global incremental potential or a new RSF stability
theorem. The retained admissible tangent check and old approximation/branch
limitations remain in force. Serial, one connected graph, no transport or AMR;
general topology, MPI production ownership and free secondary fractures are later work.
