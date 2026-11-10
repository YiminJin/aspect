# State dictionary and constitutive trace

This page describes the retained **BE default**. The selectable midpoint stage,
exact aging, endpoint publication and work identity are in
[time integration](time_integration.md); do not apply BE loss formulas to that mode.

[Guide](../TIP_TREATMENT.md) · [equations and executable entries](discretization.md)

All tensors are symmetric 3×3 plane-strain tensors; contraction counts both
symmetric off-diagonal entries. Length is m, time is s, stress is Pa. Energies
integrated over the 2-D specimen are J/m of out-of-plane thickness.

| Symbol | Units | Code / storage | Meaning and lifetime |
|---|---|---|---|
| tau minus | Pa | `BulkCohesiveState::LocalHistory::stress` | Accepted physical deviatoric stress at native mechanical quadrature; immutable during trials |
| A minus | 1 | `LocalHistory::A` | Accepted physical cohesive strain tensor; full point history, not a section moment |
| tau, A | Pa, 1 | `Tip::Response::state` | Current eliminated trial point state; copied to candidate only after convergence |
| Q, q=Q:S | Pa | `Response::Q`; accepted `LocalHistory::q` | Derived cohesive conjugate stress; scalar accepted diagnostic does not replace tensor history |
| phi minus | 1 | `BulkCohesiveState::tip_phase` | Accepted locally relevant native DG field; native Bernstein; basis-guarded checkpoint |
| B minus | 1 | `BulkTipCoupling::lower` | Bernstein obstacle copied from accepted native b (no coordinate conversion); scratch for one attempt |
| B, a=E_native B | 1 | coupled coordinate phase block; ordinary `solution` phase block | Trial coefficient vector; the native vector stores B. Physical phi_h is its FE evaluation |
| e=2A:S | 1 | local `e` in `Tip::update` | Engineering cohesive strain, not total strain |
| w | 1/s | `Response::rate`; accepted `LocalHistory::rate` | Fault-deformation rate only |
| V | m/s | `Evaluation::rate`; accepted `Section::rate` | Signed normal integral of w; resolved scalar-law rate also retained separately for cancellation checks |
| f=R(V) | Pa | `Evaluation::laws[k].law.traction`; accepted `Section::force` | Section resistance, applied by the actual integration adjoint |
| theta | s | accepted `Section::age`; `SignedEndpoint::age` | One age per entire section, initialized everywhere including intact material |
| slip | m | accepted `Section::slip` | Cumulative signed dt V, not absolute slip or surface elastic energy |
| beta | 1 | `BulkTipCoupling::beta`; local `beta` | BE Maxwell relaxation, fixed during one attempt |
| h=c/d | 1 | analytic compliance, not independently stored | Physical cohesive compliance; endpoint limits evaluated without division by zero |
| H=c/(c+d) | 1 | `Response::fraction` | Bounded elimination fraction; differs from h and diagnostic H+ |
| F | Pa | `Response::force` | P:S+G beta e minus−f; older documents used z for this quantity |
| driving factor | 1 | `Response::driving_factor` | m(1+3phi)(1−phi)/(c+d)²; Y=factor F²/(2G) |
| Y | Pa | `Response::driving` | Cohesive energy derivative driving damage, not frictional or viscous heating |
| ds, dn | m | `Section::weight`, `normal_weights` | Fixed physical quadrature measures, with JxW=ds dn |

The accepted owner is [`PhaseFieldFault::BulkCohesiveState`](../../../../include/aspect/material_model/phase_field_fault/bulk_cohesive_internal.h).
The immutable parent and trial evaluation are separate objects. Spatial matrices, reference Bernstein values, fixed-parent phase samples and
section tangent actions are derived caches. The active route has no nodal compatibility/congruence map. Each `BulkTipCoupling` belongs to one fixed-parent attempt. Neither the
auxiliary stress compositions in the fixture nor mature separated T/R histories
are physical bulk-tip memory. The checkpoint signature remains
`bulk tip physical tau A DGQ2 sections BE v1`; the separate numerical basis identity rejects nodal/Bernstein mismatches before
deserializing these arrays. An unchanged physical law signature alone does not
make two phase coefficient bases restart-compatible. Moving quadrature/AMR conversion remains
unsupported.

## A point trace within an accepted-state attempt

Take G=10⁶ Pa, eta=10⁸ Pa s, m=80, dt=.005 s, phi=.2,
old tau_xy=300 Pa, old e=.001, imposed engineering strain increment=.0001,
and current section force f=200 Pa. This is a point-algebra illustration, not
an independently converged global solution. The accepted parent remains fixed
while the nonlinear solver changes the increment, phase and section force.

1. `Tip::update` forms beta=1/(1+.00005), predictor P, and old_e=2 Aminus:S.
2. It computes c=19.2, d=.64, denominator=19.84 and H=30/31. Thus intact
   and fully damaged limits require no singular compliance evaluation.
3. Force F=P:S+G beta old_e−200 is about 1199.9300035 Pa. The point returns
   e=H F/G, q=(1−H)F, tau=P−2G(e−beta old_e)S and w=(e/beta−old_e)/dt.
4. The section sums w with dn, resolves the common force, and repeats the point
   response with that force. A trial force is not accepted merely because this
   individual point response is finite.
5. `Tip::derivative` propagates changes through H, F, tau, w and Y. Cell
   assembly integrates them against native strain/phase test functions. The
   section tangent supplies the nonlocal feedback; fracture adds cell **and**
   face terms; the constrained solve evaluates the final KKT multipliers.
6. After all physical gates pass, the candidate keeps every point tau/A and
   updates each section's age/slip once. Rejected candidates do neither.

The numerical example below comes from the earlier nodal integration run. The
[native-basis report](../NATIVE_DG_BERNSTEIN_REPORT.md) records the current
fixed-parent comparison; this arithmetic example alone is not qualification.

## Historical accepted-step trace (nodal phase)

The unchanged original fine checkpoint at t=.0475 is the parent of
`integration-fine02`, step 20 at t=.05, dt=.0025. At cell `9_1:1`, Gauss point
11, (x,y)=(.4957793446,.5211744133) m, the accepted parent has
phi=.00170344009, tau_xy=837.2268836 Pa and A_xy=1.661523264e-5.
The converged candidate has phi=.00208504698, tau_xy=867.0377627 Pa,
A_xy=2.253424054e-5 and q=268.5032632 Pa. The stable point intermediates
are c=.1671515522, d=.9958342535, denominator=1.162985806,
H=.1437262186 and F=313.5717443 Pa. In this frame e=2 A_xy;
beta=1/1.000025. Substitution into `(e/beta−e_minus)/dt` gives
w=.004735657008 /s. The intact-reaction/section balance is directly visible:
tau_xy−q=598.5344994 Pa, exactly the section resistance to printed accuracy.

Section 23 uses ds=.01070778077 m and the actual normal quadrature weights.
Its integrated V=.0008519897765 m/s differs from the scalar constitutive root
by 7.59e-19 m/s. Its accepted age is 10.0427013746 s and cumulative signed slip
7.269135425e-6 m. These are one section's state, not the local w or local
cohesive strain. The candidate keeps the entire point tensor field and publishes
all section states only after the mechanical/KKT/energy gates pass. This step
rescues the original failure; it does not by itself qualify later evolution.
The exact values are retained in
[accepted-point-trace.json](../../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_solver/accepted-point-trace.json).

The [native Bernstein guide](../NATIVE_DG_BERNSTEIN.md) defines coefficient and
sampled-value meanings, checkpoint identity and remaining compatibility
conversions. The historical point trace above retains its original executable
and initialization; it is not relabeled as a native Bernstein result.
