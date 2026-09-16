# K5: fault-grid sensitivity at the 40 km junction

Use the consolidated history/resolution report and the supplied final_profiles.csv and node_history.csv. Both existing runs retain the same 100 m fault grid and a sharp minimum at 39.9 km. Test whether the junction layer depends on that discretization.

1. **Change only the fault grid.** Keep the qualified refined bulk mesh (42,880 cells), its particle discretization, frozen phase profile, physical fault line, full support, production particle-history formulation and all physical parameters. Bisect existing fault elements in 36–44 km, giving approximately 50 m spacing there. Preserve existing vertices and the exact 40 km prescribed-slip junction; leave the remaining fault grid unchanged.

2. **Evolve comparable histories.** Initialize the new grid from the same physical initial data and run from t=0 through 2232176379.2516127 s only. Do not interpolate the late 100 m fault history onto the new grid. Reuse the existing 100 m trajectory if every recorded step is admissible for the new run. Otherwise use the existing matched-clock controller for one replacement 100 m/50 m pair, retaining all timestep restrictions. Report initial differences in I_h, cohesion and background traction at common coordinates; do not retune them to force agreement.

3. **Export only decisive evidence.** At common times, compare V/Vp, Theta, accumulated slip, cohesion, shear driving, physical R_i/m_i and bound reactions across 39–40.5 km. Track the last free node (now approximately 39.95 km), its neighbours, the old 39.9 km location, and one unchanged interior control. Report the physical width/location of the rate depression and total bound reaction over the same physical junction window. Keep projected normal stress separate from raw pressure/normal-stress extrema and all-sample tensile weights.

4. **Answer the decision.** Does the depression/bound follow the last free node and narrow with element length, or persist over a comparable physical interval? Do changes come mainly through cohesion, friction or shear driving? Does improved fault resolution reduce the raw pressure dipole, or only alter its projection? Interpret this as a fault-grid sensitivity test; two resolutions and the known temporal uncertainty do not establish continuum convergence.

**Budget:** at most one new 50 m trajectory, plus one replacement 100 m trajectory only if a common admissible clock requires it. Reuse existing runners and checks. No further bulk refinement, timestep-halving campaign, FE-history substitution, smoothing, clipping, cutoff change or first-event continuation.

Stop after this comparison. Deliver one compact profile figure, decisive numbers and one next recommendation; preserve original outputs and production defaults.
