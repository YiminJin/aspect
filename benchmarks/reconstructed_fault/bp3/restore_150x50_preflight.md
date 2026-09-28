# BP3 150 x 50 km restoration: shear-sense preflight

Historical preflight, 2026-09-24. The proposed shear-sense correction was
subsequently approved and implemented. See `restore_150x50_report.md` for
current qualification and resource status; the original findings below are
retained as the rationale, not a current request for approval.
Source base: `3ce447a17e14ac21cb24abf8fbf679d77b4f3f0e`, plus the
pre-existing working changes. Those changes and the BP5 cases are preserved.

## Confirmed incompatibility

The requested coordinates reflect the maintained BP3 chart. The maintained
`bp3_model.h` deliberately uses a proper 180-degree rotation to retain positive
thrust slip with the existing source convention. A reflection does not retain
that convention.

For the requested down-dip tangent s=(1/2,-sqrt(3)/2) and normal
n=(sqrt(3)/2,1/2), positive thrust has
u(r<0)-u(r>0)=Vp*s. Thus the requested smooth loading must use Vsrc=-Vp:

    sym grad u_load = -Vp F'(r) sym(s tensor n).

The existing continuation requires bottom-to-top vertices. The production
geometry cache uses t=-s and m=(-t_y,t_x)=-n, and therefore constructs

    S_current = sym(t tensor m) = sym(s tensor n).

Positive manager V consequently produces the opposite shear sense. Reversing
vertex ordering reverses both t and m and leaves S unchanged. Allowing negative
V instead would violate the specified V >= Vmin > 0 friction/state and active-set
contract. Flipping only the boundary velocity would reverse the physical
faulting sense; flipping only a diagnostic tensor would leave mechanics wrong.

The algebraic check gives:

    S_current = [[+0.4330127018922193, -0.25],
                 [-0.25,             -0.4330127018922193]]
    S_required = -S_current

Their relative Frobenius difference is 2, not a numerical discrepancy.
The correct far-side velocities in the symmetric thrust frame are
(+2.5e-10,-4.330127018922193e-10) m/s for r<0 and the opposite for r>0.

Source evidence: `manager.cc` constructs the Stokes QP frame;
`reconstructed_fault_stokes.cc::slip_tensor` constructs its source tensor;
`surface_system.cc` independently constructs the same tensor for the surface
work; `PhaseFieldFault::evaluate_reconstructed_fault_point` uses it in both
stress subtraction and driving traction. Particle stress publication also
constructs this tensor in `phase_field_fault.cc`. All consumers must agree.

## Smallest proposed capability (not implemented)

Add an explicit immutable per-fault shear-sense selector, default +1 to preserve
all qualified cases, with -1 for this reflected thrust fixture. Keep V and
Theta positive. Use the signed slip tensor consistently in bulk source,
working/particle stress, surface driving traction, B/G/K and diagnostics;
normal contraction N remains unchanged. Persist/validate the selector across
restart and invalidate affected coefficient caches when configured. Document
this explicitly rather than silently changing the established frame convention.

Focused qualification would check the reflected loading/source identity,
virtual-work and derivative signs, unchanged +1 behavior, and restart identity
before the requested startup runs.

## Other requested choices retained for subsequent implementation

- Unlimited native linear least squares for the compositional fields, with
  continuous Q2 (not the previous stress-only LLS/DWA routing).
- Separate raw, filter20 and filter40 inputs; true normal traction; filter
  active at initialization, never written into stress history.
- Endpoint/interior diagnostics must compare raw and actual filtered normal
  input, slip-rate variation and its growth at matched accepted times; the
  whole-domain transfer metric is not a qualification criterion.
- The requested fine mesh is not a cheap startup: the instructions estimate
  about 487,389 leaves and 4.39 million particles before balancing. Actual
  mesh/DoF/memory inventory is still required before a mechanical launch.

No fixture, production algorithm, loading, tolerance or simulation was changed
in this preflight. The only new file is this record. Restoration/startup and
the first-event workflow remain unimplemented pending the shear-sense decision.
