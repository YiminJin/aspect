# Bounded notch-mechanism experiments

Baseline: commit `3335d3d26c`, mature revised-work 50-m junction fixture.
No first-event continuation or production formulation change is selected.

First distinguish local minima from a broad frictional slowdown using all
saved accepted states. Reconstruct actual work-quadrature friction, shear and
normal loads with **preceding** Theta, and verify against saved weak loads.
Keep current constitutive stress separate from published particle history.

The new mechanical experiment uses the identical accepted step-9 checkpoint,
real dt and old histories as the verified noncommitting lagged-state A solve.
Increase only the prescribed value at the exact 40-km node by 1% of Vp. This
is a localized boundary-basis impulse, not a replacement BP3 trajectory or
an approved physical loading change. All other prescribed nodes and all
friction/history data remain unchanged. Re-equilibrate bulk and free V;
do not commit. The sign and spatial structure of neighbouring rate changes
distinguish an oscillatory constrained discrete response from a state-update
artifact. Compare with the actual frozen-bulk surface tangent and consistent
Q1 mass response, but do not mistake either for a full coupled solution.

One four-rank solve is expected to cost 2–4 minutes and about 1.5 GiB/rank;
hard cap 600 seconds. A second, smaller amplitude is justified only to check
linearity if needed. Require unchanged incoming fingerprints, fresh linear
checks, genuine nonlinear convergence and complete rollback. Preserve failed
attempts. Use the saved A result as control; no repeat baseline is needed.

The first junction result is non-monotone: the adjacent response is -0.592
times the imposed rate increment. In addition to its half-amplitude check,
one neutral 25-km control is justified: pin that formerly free node at its
saved converged rate plus 0.01 Vp, leaving all other free nodes unconstrained.
The unperturbed saved solution satisfies this extra constraint exactly. The
response tests whether a localized rate-gradient disturbance excites the
same structure away from the friction transition and the 40-km junction.
It is a diagnostic point constraint, not a changed physical loading or mesh.
Same checkpoint, frozen histories, rollback and 600-s cap; no trajectory.

The 25-km control also produces alternating signs (-0.45/-0.46 at the two
neighbours, +0.12 one node farther out). One final operator-column separation
therefore holds all other V nodes at the saved converged values and applies
the same 1% junction impulse. Re-equilibrate only bulk u,p; observe unreplaced
surface residuals instead of enforcing them at the diagnostically prescribed
nodes. The unperturbed saved solution satisfies every added constraint. This
isolates direct Q1 Maxwell/friction terms from bulk shear and normal-stress
redistribution. It is not evidence of free-row convergence for this all-V-
prescribed experiment. No additional trajectory or altered state update.

The 15/18-km analysis must report whether a local notch is present in the
current formulation/time, rather than assigning the 40-km mechanism to every
frictional transition. Older volume-rule runs are labelled historical.
