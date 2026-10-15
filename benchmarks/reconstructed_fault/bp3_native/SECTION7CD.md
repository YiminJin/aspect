# Selected 7C–7D: bounded refinement and MPI/rollback

Reuse the perturbed FE 7B trajectory and all physical inputs. Preserve its
executable, maintained plugin, observer and outputs. Fresh builds/outputs are
under `build-bp3-native/section7cd/` and separately named `runs/7cd-*`.
No unfiltered comparison, full-BP3 preparation, transport or restart.

1. dt 100000 to T2000000, twenty positive steps; keep Initial time step=200000
   so step-zero kinematics/initial constitutive model remain the 7B convention.
2. Same original mesh/surface/dt 200000 with q6, retaining the general consistent
   I_h integration. q8 only if a material ambiguity remains.
3. Halve actual surface spacing20→10 m with selected quadrature; then raise
   startup band/endpoint refinement by one level, retaining surface 10. The same
   prescribed physical phase profile is represented on the new bulk mesh.
4. One initialization/material check with the unmodified BP3 friction transition
   and its native state/background assignment; no coupled first-event run.
5. One selected stable discretization, clean one/two-rank runs through at least
   three positive steps; one-owner candidate-stage failure and identical-dt retry.
   Test accepted FE, Z, state, slip, all derived particles and coordinated rejection.

Reuse the existing observer/analysis, adding only the required rank-safe compact
outputs and targeted checks. Original nonlinear/linear tolerances 1e-8/1e-9,
filter 20/full bottom/frozen phase/stationary particles and physics remain fixed.
Aim at3600 s aggregate local simulation time excluding compilation. Each run
has a predeclared accepted-step and external wall cap; no silent schedule change.
Measure sensitivity/roughness and MPI correctness separately from physical accuracy.

After the spatial result, 7D selects the passing q6/surface 10/refined-band fixture at dt 200000 to 600000 s.
Physical-coordinate FE coefficients avoid distributed ordering comparisons. Before
launch, set comparison allowances to ten times the larger unchanged nonlinear/
linear criterion (1e-7), scaled by Vp, theta 0, unit Omega, Vp*T, measured stress
perturbations, Vp*T for d_R, and measured Z. Use the configured I_h integration
criterion1e-10 for I_h comparison and 1e-13 absolute for the frozen phase.
These are reproducibility allowances, not physical-error estimates or new solver
tolerances. Rejection checks require exact accepted-history/property preservation.
The last rank alone arms a fixed-sample registry mismatch; actual particles do
not move. Its native collective guard is reached after history candidates are
constructed, before FE vector collectives/publication. MPI run caps are 600 s each. Restore only that injected
registry entry, verify physical accepted data, roll back scratch mechanics and
allow the ordinary driver to retry the identical physical interval once.
