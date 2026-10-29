# Activated-append temporal work evidence

Start with `COHESIVE_INTERFACE_ENERGY_REPORT.md` (included in the archive) and:

- **energy.png**: raw absolute/signed work errors, independent decomposition and
  exact tied-Maxwell control. Replay curves begin at .58 s; they do not contain
  the earlier loading error. Dashed vertical lines mark the .65 s hold.
- **response.png**: damage, sliding loss and common-time half-step differences.
  The geometry and loading were prescribed; the calculation advances Maxwell,
  cohesive and RSF histories. It does not select a path or demonstrate complete
  decohesion. Spatial accuracy remains unqualified.
- **comparison.csv**, **verification.json**: selected acceptance results.
- **loading-hold.csv**: sums separately before/after motion stops; signed and
  absolute remainders make cancellation visible.
- **analytic-maxwell.csv**: exact exponential ramp/hold integrals compared with
  backward Euler. The intact tied control does not replace the damaged solution.
- **response-comparison.csv**, **profiles.csv**, **profile-errors.csv**: matched
  physical times, native surface Gauss positions and actual arclength weights.
  Damage a and age Theta are dimensionless; jump is metres. Weighted L2 uses
  sqrt(sum(w*(difference)^2)/sum(w*half_step^2)); no spatial interpolation.
- **coarse/fine/selected/replay-history.csv**: every accepted step. Energies are
  J/m of out-of-plane thickness; time seconds, pressure Pa, tangent Pa/m,
  integrated damage m, integrated absolute jump m². `physical_remainder` uses
  trapezoidal work; `ledger_error` uses endpoint work plus explicit BE numerical
  terms. No numerical remainder is treated as physical dissipation.
- **selected/replay-summary.json**: measured timings, numerical event brackets,
  maxima and checks. Final bulk/bond entries are storage, not increments.
- **selected/replay-snapshots.json**, **preactivation.json**, **final.json**:
  accepted histories for checkpoint/replay. Geometry generation/publication is
  distinct from physical accepted-step publication. The replay starts from the
  selected .58 s state, not from fresh histories.
- **preservation.json**, **run-manifest.json**, **SHA256SUMS**: provenance and
  checksums. `entry-protection.json` hashes all protected source/include/BP3
  files on entry. Previous admission archive and evidence are not overwritten.

Reproduce from the worktree root, sequentially (no overlapping timings):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/cohesive_interface_energy.py audit
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout 600s python3 benchmarks/reconstructed_fault/particle_free/cohesive_interface_energy.py fine
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout 300s python3 benchmarks/reconstructed_fault/particle_free/cohesive_interface_energy.py replay
MPLCONFIGDIR=/tmp/pf-energy-matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/analyze_cohesive_interface_energy.py
```

Python, NumPy, SciPy/HiGHS and matplotlib are needed. The immutable native fitted
kernel is at `build-particle-free/cohesive-interface-kink/native-16-axial`.
The archive includes the eight small files needed by `Kernel` under `inputs/`;
restore them to that path if absent. Archived .005/.0025 s states/CSV for `audit`
remain in the prior `cohesive_interface_admission.zip` evidence package.
The archive includes scripts/dependencies in `reproduction/`; their canonical
location is `benchmarks/reconstructed_fault/particle_free/` so relative ROOT
resolution works. This task does not compile or run a new native ASPECT solve.
