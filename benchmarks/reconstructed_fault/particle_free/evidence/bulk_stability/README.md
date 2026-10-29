# Bounded bulk stability remedy

Read doc/reconstructed_fault/particle_free/BULK_STABILITY_REPORT.md in the worktree
or archive. verification.json contains measured native gates and bitwise physical
owner/restart comparisons; provenance.json contains exact inputs, commands/build
locations and source identities. changed-files.json lists this task alone.

publication-recovery.png/csv show actual first-hold rejected/published physical
phase and the relaxed negative phase mode in physical metres. Contours are
.01/.05/.1/.3; dashed x=1 m is the original seed endpoint. Phase and mode are
dimensionless; the two growth panels use identical scales.

CSV times are accepted seconds, dt is seconds, energy/work are J/m (unit
out-of-plane thickness), stress is Pa. Curvatures use Kf times lumped phase mass.
A stable candidate may still be discarded by the outer lifecycle test; only
bulk-accepted.csv records publication. Profile times are local seconds, include
intentional rejected work, and are not server estimates. A zero curvature entry
with zero critical masters means strict obstacle exclusion, not a measured zero
eigenvalue. Full history records, checkpoints and original logs remain separately
in the indicated build/log directories. No duplicate PDF/SVG/MP4 or full solver
archive is included. The P4 evidence archive and BP3 artifacts are unchanged.
