# Boundary/profile isolation protocol (2026-11-03)

Selected by the user after the coupled Stage C report. Preserve native source,
core and existing checker libraries; add an alternative nonpublishing checker.
No constitutive, solver, interpolation, inflow or acceptance changes to the
coupled path. Local tests use two MPI ranks sequentially, at most four workers.

1. Native DGQ2 SSPRK3, prescribed constant x velocity +/-0.4 m/s, [-2,2]^2,
   0.4 s: a tensor polynomial exactly representable in Q2 with correct analytic
   inflow, the same initial field with deliberately zero inflow, and a compact
   C3 polynomial bump that never touches a boundary. All components and implied
   zz are retained. Compare both the correct boundary-value reference and the
   smooth continuation; those are different physical problems for zero inflow.
   Meshes8/16/32 cells per side. dt=.00625 s across meshes; additionally
   .025/.0125 s on the16-cell mesh to separate SSPRK3 time errors. Native CFL
   is below .125. Correct-inflow polynomial error must be <1e-10 relative.
2. Independently integrate native volume tensor energy and face flux/jumps.
   E=integral |tau|_3^2/(4G), G=10 Pa. Verify instantaneous semidiscrete
   dE/dt+outward physical boundary flux+upwind loss=0 to1e-10 absolute.
   Report accumulated physical flux, upwind loss and endpoint-trapezoidal
   integration remainder separately, with timestep scaling. Do not infer a
   complete moving-fault energy qualification from a pure transport test.
3. Compact physical profile: repeated Q1 nodal pullback versus direct pullback
   of the initial Q1 interpolant and endpoint sampling of the analytic profile.
   Use exact translations, three mesh sizes and three physical timestep sizes.
   Quantify mass, maximum, second moment, L2 errors and the signed decomposition
   into repeated remapping and representation errors. Derive and check the
   interpolation variance increment n*C*(1-C)*h^2 for a one-dimensional control.
4. Two fresh coupled compact-profile cases at256cells, dt=.1/.05 s, same
   initial mechanics interval .1 s and final t=.4 s. Retain all other existing
   fixture inputs. Independently replay the actual native Q1 nodal mapping
   using exported accepted velocity/profile, and separate boundary-band and
   interior errors. No prescribed characteristic tensor is substituted into
   the solved mechanical state.

Expectations are hypotheses: compact support should eliminate physical inflow
flux; exact polynomial inflow should reproduce translation to arithmetic
precision. Fixed-h timestep reduction need not decrease repeated-Q1 error.
Retain failures. No limiter, damping, tolerance tuning, physical campaign or
production checkpoint conversion is part of this experiment.
