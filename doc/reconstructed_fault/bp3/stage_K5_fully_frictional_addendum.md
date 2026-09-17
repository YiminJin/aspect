# Fully frictional seven-step replay

The user authorizes removal of all deep prescribed V rows, including the
bottom endpoint, for a fresh seven-step comparison. This deliberately changes
the BP3 deep boundary condition; it is not the official prescribed-deep-slip
benchmark and does not change the default fixture.

Use `ASPECT_BP3_FULLY_FRICTIONAL_REPLAY=1` only with the corrected mature
committing work-measure fixture. Retain the original continuous Q1 velocity,
state and slip everywhere; no independent-trace selector is enabled. Keep
the existing mesh, geometry, frozen phase, full Ih, support, both boundary
completions/source continuations, initial background and external loading.
The continued source and work equation use the current free endpoint rate
and its derivative, exactly as at the already free top endpoint.

For xd >= 40 km, a=.025, b=.015 and Vinit=Vp=1e-9 m/s. The existing supplied
Theta0=Dc/Vinit=8e6 s satisfies the initial continuum background balance
tau0=mu(Vinit,Theta0)*50MPa+damping*Vinit (mature C=0). Retain that state and
the captured initial background; report the actual initialized rates and
remaining discrete stress/balance discrepancies rather than refitting them.
Particle Maxwell history initially remains zero. Apply the ordinary lagged
state mechanics and one accepted update per step.

Run the original seven saved timesteps to 132230424.76671731 s. Compare with
`work-replay-50-local4` steps 0–7, not the later split-trace trajectory.
Require genuine convergence, unchanged timestep selection, history audits,
and no prescribed surface rows. Inspect continuous behaviour across 40 km,
shallow slip deficit and increasing elastic/shear loading, deep creep, and
current raw/weak top and bottom stress. A free bottom changes its mechanical
condition, not the outer zero-perturbation-traction condition. Stop on an
invariant/solver failure, or after seven steps; do not change tolerances.
