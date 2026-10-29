# Fixed-surface front evidence index

[Report](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_FRONT_REPORT.md)
and [architecture proposal](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_ARCHITECTURE_PROPOSAL.md).
The figures show surface a on a prescribed Gamma, not bulk phi or free geometry.

| Item | Content |
|---|---|
| `space_time.png` | Tied/cohesive/decohered regions, geometric jump and physical weak shear versus x/time; initial seed marked |
| `profiles.png` | Damage, slip, traction at selected times; tied multipliers distinguished from physical traction |
| `comparison.png` | Spatial/timestep controls, separate process-zone and full-decohesion fronts, admissible/release curvature |
| `energy.png` | Reaction, storage, fracture/loss and numerical work remainders |
| `front_evolution.gif` | Equal-scale surface profiles; 82 frames, including every saved fine step near full decohesion |
| `physical_histories.csv` | Every accepted time: physical region lengths, contours, work/storage, reaction, certificates |
| `profiles.csv` | Selected actual Gauss profiles, with x/time/units and each mesh's own initial-field increment |
| `animation_frames.csv` | Actual data used by the GIF, including pressure and signed rate |
| `comparison.csv` | Common-time spatial/temporal errors, including early failures; growth L2 uses the same physical 1/32 m box average |
| `weak_admission_examples.csv` | Nodal-cutoff counterexample and admissible multiplier redistribution |
| `qualification_summary.json` | Independent reference, activation lifecycle, derivatives and bitwise restart |
| `growth_summary.json` | Measured mechanism, event brackets, refinement and energy errors |
| `runs-*.json`, `provenance.json` | Inputs, bounded sequential timings, exact source/binary/preservation hashes |

Coordinates in m, times and Delta t in s, stress/traction in Pa, jump in m,
velocity in m/s; energies are J/m per unit out-of-plane thickness. Damage and
Theta are dimensionless. Delta x=.03125/.015625 m, Delta y=.00078125 m. The same
physical seed [.375,.625] has a=1 and initial inventory .25 J/m on both grids;
all increments are relative to that initial state. New fracture is not counted
as dissipation a second time. No state interpolation is used in constitutive
checks. Diagnostic box averages use Gauss weights and do not smear across the
initial material-region boundaries. Raw Gauss peak differences remain separate.

The contour thresholds are .01,.05,.1,.5,.9 and exactly 1. Seed-connected extents
start at the physical seed endpoints and expand only when neighboring exterior
Gauss records reach the threshold. Sub-unit thresholds interpolate between
exterior sample points; a=1 extents are bounded by saved Gauss locations and their
neighbors. Full-decohesion event times are brackets, not exact rupture times.
Disconnected lengths are reported separately. Curves at the periodic cell edges
must not be interpreted as isolated cracks in an unbounded domain.

A tied multiplier is an admissible dual reaction and may be nonunique; only its
weak balance and bound are physical discrete requirements. The separately plotted
`shear_Pa` is the energy-conjugate projected physical traction. `q_Pa` is bond q
in damaged material and one certified multiplier in a tied record. Gauss records
use the legacy column name `node`; actual surface nodal memories/rates are in
`nodal_history.csv`. The full bulk tensor is represented by the fixed native
mechanical basis; no arbitrary old history or production checkpoint is converted.

The compact archive includes summaries, selected profiles/animation data, native
operator/check/energy histories and the full short activation/restart Gauss
streams. Long growth Gauss streams and nodal output are hashed and retained in
this directory and under `build-particle-free/cohesive-interface-front/runs/`;
they are omitted from the compact archive, as are host binaries and checkpoints.
Rerunning the supplied scripts regenerates them. The archive is a worktree overlay,
not a standalone ASPECT distribution. Failed semismooth reference drafts/logs and
the original edited source are preserved under the task build directory.
