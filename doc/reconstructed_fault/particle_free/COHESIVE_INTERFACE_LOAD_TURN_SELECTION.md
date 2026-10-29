# Boundary-driven loading turn and reserve comparison

Baseline f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved local work;
production/BP3 and prior evidence remain protected. Experimental serial fixed
bulk mesh, closed contact, full 3D plane-strain Maxwell history and unchanged
cohesive–RSF/radiation law. Physical R=.75 m; numerical lifting/normal virtual
support half-width=.75 m, separately named and fixed.

## Physical loading and histories

Use two independent fixed, laboratory-coordinate strain tensors:

`E0=[[-b,.5],[.5,b]]`, `E1=[[-.5,-b],[-b,.5]]`, b=.5 tan40°.
They are trace-free, orthogonal and equally normalized. Boundary displacement
is `u_D=(a0 E0+a1 E1)x`, with
`a=gamma(t)*(cos(2 alpha(t)), sin(2 alpha(t)))`.
Gamma starts .0003 and increases at .004/s through .455 s; displacement then
holds through .460 s. Alpha is zero through .410 s, changes smoothly to −20°
over .410–.440 s with quintic smoothstep, and then stays fixed. In the retained
control alpha remains zero. This changes physical boundary displacement; it
does not rotate any accepted stress, geometry or history. Both prescribed
affine Maxwell memories advance as `b_new=beta*(b_old+Delta a)` in their fixed
basis. Signed surface memory/age/damage retain their established updates.
Boundary work is the sum of the two conjugate reactions times their physical
increments. p0=10000 Pa and mean-zero p′ remain unchanged.

The homogeneous elastic preferred direction changes from +20° toward 0°;
actual accepted stress direction follows the Maxwell recurrence and slip
redistribution and is measured independently. Initial .410 s growth is already
measurable in the accepted fixed-support fixture (~.038 m new ∫a ds); confirm
it in the actual new run. No Coulomb-angle intact admission is introduced.

## Reserve and scheduling

Keep append-only permanent geometry, including virgin tied records. No damaged
or virgin committed material is rewritten. Compare reserve targets S=.75 and
.1875 m using the SAME numerical extension quantum Delta_s=.375 m and surface
spacing .09375 m. Selection uses the accepted stress tensor and the same R
disk/persistence rule. Append while geometric end minus connected a=.05 front
is below S (roundoff tolerance 1e−10 m). Resulting reserve lies in [S,S+Delta_s)
at publication. This removes the old strict-equality extra-append ambiguity.
Record actual reserve, formation time/direction and later activation time for
each segment. Scheduling quantization and permanent virgin-direction freezing
are distinct assumptions. S is not R or the lifting width.

Primary bounded matrix: changed loading with both reserves on 64² Q2, and
unchanged loading with small reserve. Then one 96² changed/small-reserve spatial
control and one own-prefix time halving across the turn. If substantial reserve
dependence is observed, halve the extension quantum on the small-reserve case
to separate scheduling from committed-direction effects. No broad search.

Representative frozen checks after the turn use the same physical Γ, total
jump and relaxed memory/loading field, independently reconstructed on the
matched bulk/fitted reference. Check full sampled stress tensor/preferred angle,
work-conjugate shear loads, normal reactions, energy and incompressibility.
Retain unfiltered point-traction defects and the earlier ahead-region normal
failure. Fitted geometry is a fixed-path reference only, never an implicit
conversion of accepted evolved history.

## Gates and stops

First qualify homogeneous two-component Maxwell recurrence, physical work,
single-component identity, coupled pressure tangent, old-field preservation,
rejection/retry, once-only publication and rebuilt-context restart.
Retain all original residual/FD/contact/ledger tolerances. Practical comparison
targets: <5% growth/slip integrals, front difference <.09375 m, preferred-angle
uncertainty <1° and full-tensor sample discrepancy <2% on meaningful amplitudes;
physical absolute-work error <1% of sliding and added fracture. Preserve failed
targets rather than modifying them. A low-damage front is not a mature tip.

Bound at .460 s, 4.125 m new arc, positive contact, existing solver/tangent
limits, boundary margin at least R, 300 s per native kernel, 1200 s per trajectory
and 2 GiB new raw output with a disk margin. Preserve all failed trials and last
accepted checkpoints. If either control reaches a stop early, compare common
times and label unequal termination. No long run, R calibration, opening law,
material change, AMR, transport, MPI expansion or production integration.

## Observed admission stop and retained diagnostic convention

The first S=.1875 m trial stopped at accepted time .44625 s before an append:
the independent bounded weak multiplier required 3065.401664535 Pa, exceeding
3000 Pa. No birth damage, strength adjustment or altered reserve is used to
bypass this guard. Retain the failed trial and reproduce it with snapshot
archiving, then compare common accepted times. The large-reserve trajectory
continues to the originally declared endpoint. Unequal termination is part of
the result, rather than a reason to extend loading or search reserve values.

Under changing loading, current slip need not remain monotone. Therefore front
positions are bracketed directly by consecutive owned irreversible Gauss damage
records (a=.05,.5,1); they are not inferred from current slip or constitutively
interpolated into another state. Report detached qualifying Gauss points
separately. This changes diagnostics, not history or the return law.

The unchanged-loading S=.1875 m control also reaches an inadmissible tied birth
at .43775 s (3004.068150654 Pa). Add one otherwise identical unchanged-loading
S=.75 m control to compare both mature fronts through the intended loading/hold
endpoint. This reuses an already selected reserve, not a reserve-parameter search.

At common .445 s the primary reserve cases differ by 10.94% of the large-reserve
new-damage integral, while sampled preferred angles differ only .0125 degrees.
This exceeds the 5% practical integral target and triggers the predeclared
half-append-quantum control (Delta_s=.1875 m, fixed S=.1875 m, surface cell
spacing unchanged). The spatial and timestep controls retain the original
Delta_s=.375 m, so these effects remain separate.

The half-quantum control reuses the SAME accepted .410 s checkpoint as the
half-timestep control. It preserves the initial .375 m committed append exactly
and changes only future append quanta to .1875 m. This isolates post-turn
scheduling without rewriting any previously committed geometry or history.
The fresh 96² control reaches the same tied-birth stop at .446 s (3083.918 Pa);
its new-damage difference is 1.22% at .445 s, below the reserve effect. Identical
front Gauss brackets are not sub-grid front-position accuracy.
