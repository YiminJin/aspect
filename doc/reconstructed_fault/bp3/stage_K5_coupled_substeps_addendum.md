# Bounded coupled-state timestep check

User-authorized extension of the noncommitting within-step A/B experiment:
reuse its coupled single-interval result, and run only two half-steps and four
quarter-steps from the same accepted step-9 checkpoint (15.30625726 yr), ending
at 29.24190894 yr. This is a benchmark-selected alternative time discretization,
not a replacement of the authoritative production split algorithm.

During mechanics, old nodal Theta is immutable and the exact candidate
T_i(V_i;Theta_old,dt) is interpolated at QPs. The already verified nonsymmetric
Jacobian is retained. Upon acceptance the ordinary terminal history publication
computes this same candidate **once** from old Theta and accepted V. No extra
update is added. Maxwell stress and every other history use the normal accepted
publication, transfer and advection; inert mature H and zero C remain unchanged.
Compare all committed nodes against the candidate and actual final mechanical
QP state against the committed interpolation. Retain independent aging checks.

Each run uses a disposable checkpoint copy. ASPECT saved an already advanced
pending clock; change only its pending time and dt doubles, verifying every
other uncompressed archive byte and mesh/history file is identical. Old dt,
solution vectors, benchmark slip, preceding Theta, physical background and
geometry are retained. Reuse the earlier diagnostic-only restoration of exact
saved frozen I_h, requiring roundoff-level agreement before substitution.
No general committing restart qualification is claimed.

Explicit `ASPECT_BP3_COUPLED_STATE_REPLAY=2` or `4`, together with the existing
candidate-state switch and committing work-measure benchmark parameter, permits
this narrow replay. The accepted-clock termination guard limits the runs to
the requested substeps. All spatial settings, pressure/boundary treatment,
controllers, tolerances and nonlinear criteria are unchanged. The first pending
step is the requested subdivision; later steps remain subject to the existing
strict replay/controller check. Failures are preserved, not retuned.

Compare final deficit 1-V39950/Vp, contrast (V39900-V39950)/Vp, and accumulated
last-element gradient increment sum(dt*(Vp-V39950))/50 m. The old slip gradient
is common to all three levels. Report successive differences and their ratios;
three levels at one interval do not establish full-trajectory convergence.
