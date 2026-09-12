# K3 periodic-domain correction: verification record

## Implementation and limits

The approved global-period image construction and full-domain fragment contract
are in `stage_K3_periodic_domain_addendum.md`. Production changes are limited to:

- `source/particle/particle_domain.cc`: periodic-equivalent vertex neighbors,
  translated geometry-only particles with distinct temporary IDs, global Box
  extent, periodically mapped CPDI samples, and physical-box fragments.
- `include/aspect/particle/particle_domain.h`: retains unwrapped `vertices()`;
  adds `periodic_fragments()` (empty for unsplit polygons).
- `source/simulator/phase_field.cc`: completes phase sparsity across periodic
  cell neighbors. The phase weak residual/Jacobian formulas are unchanged.
- `source/reconstructed_fault/manager.cc`: integrates every physical fragment
  once into the existing cached open-fault domain quadrature. All surface
  projections, R_Gamma/K_V/G, norms and diagnostics reuse that cache.

Independent endpoints are not identified. Parent-center admission, full domain
measure, P0 bulk/history evaluation, history ownership/commit timing, I_h,
support, pressure and solver criteria are unchanged. The already deferred
polygon-gradient orientation issue is not changed. The optional face-based
Voronoi reconstruction interpolator still lacks image offsets and is now
explicitly rejected with periodic domains; the accepted cell-average fixture
requests no face data. Three-dimensional/multiple-axis periodic support is
not claimed by this bounded correction.

## Focused nonlinear phase and fragment regressions

`tests/phase_field_periodic_domains.cc`, built/run with the harness in
`benchmarks/reconstructed_fault/uniform_shear/evolving/periodic-domains`,
evaluates production phase assembly with

    phi(y) = 0.2 + 0.15 cos(2 pi y),
    H(y)   = 1 + 0.2 cos(4 pi y),
    dx(y)  = rho * particle_pitch * (0.7 + 0.3 sin(2 pi y)),
    rho    = 0, 0.12, 0.18, 0.8.

These are frozen diagnostic inputs, not changes to K3 loading. Owned and
periodic constrained DOFs are counted once, H is restored after assembly,
and live state fingerprints agree. The last displacement wraps particles.

The 4x16 case tests normal partitioning on one/two ranks. A separate 32x8
case uses x-ordered coarse roots and verifies **16 remote periodic faces**
on two ranks, so opposite-boundary image particles really cross MPI ownership.
That case additionally checks the complete physical-strip surface Q1 mass
against its independent analytic matrix, with exactly zero first/last-node
coupling. This exposes lost or double-counted periodic fragments without
requiring exact reproduction of a nonconstant field from cell averaging.

| Check | Maximum observed error |
| --- | ---: |
| Seam/interior production weak-load column difference | 4.705e-15 Pa m^2 |
| One/two-rank weak-load difference, 4x16 | 1.666e-16 Pa m^2 |
| One/two-rank weak-load difference, remote-image 32x8 | 4.164e-17 Pa m^2 |
| Constant weights | 4.441e-16 |
| Summed gradient norm | 1.351e-13 1/m |
| Unwrapped first-moment error | 2.238e-16 m |
| Physical-fragment / parent measure - 1 | 5.552e-15 |
| Analytic open-fault mass entry error | 1.341e-16 m^2 |

Total measure is 0.25 m^2 to within 6.1e-15 m^2. The wrapped state exercises
28 wraps/48 split parents in the 4x16 case and 14 wraps/24 split parents in
the 32x8 case. No endpoint DoF joining or history duplication is involved.
Each invocation takes only a few seconds; exact resources and per-state loads
are in `{one,two,one-remote,two-remote}.resources.json` and their output folders.

The initial test overlay accidentally inherited the reconstructed-fault
timestep selector while deliberately disabling mechanics. Its missing-V
failure occurred before the new checks. The test-only overlay now selects
the convection timestep; failed evidence is retained under
`one-before-timestep-fixture-fix`. No production parameter meaning changed.
MPI launch failures under the sandbox are retained separately; MPI-enabled
invocations pass. No expected numerical output was loosened or refreshed.

## Saved frozen K3 state audit

The old wall-domain evidence under `evolving/seam-audit/output` is preserved.
`output-corrected` uses the same saved real-parent positions, the same
homogeneous phi/H inputs, and the revised production domains. Changed volume
relative to the old wall construction is recorded rather than asserted equal.
The physical phase profiles, loading and particle/history data are not evolved
in this audit. Geometry setup remains distinct from the noncommitting probes.

| Saved domains | Original seam density RMS | Corrected seam density RMS | Corrected fixed-interior RMS |
| --- | ---: | ---: | ---: |
| Initial | 1.094e-12 Pa | 1.427e-12 Pa | 1.105e-12 Pa |
| Step 2, 0.75 s | 0.147951 Pa | 1.13743e-6 Pa | 2.63279e-7 Pa |
| Step 3, 1.125 s | 0.233182 Pa | 9.83795e-7 Pa | 2.02668e-7 Pa |

The seam component decreases by about 130,000 / 237,000 times. Whole-domain
nonuniform residual-density RMS becomes 3.462e-7 / 2.855e-7 Pa. These actual
saved states have small x-dependent particle positions from their previous
wall-domain coupled trajectory; they are not the ideal translated lattice.
The small residual is retained, not smoothed or asserted to vanish exactly.
Total measure relative error remains about -7.85e-9 / -5.38e-9, comparable
to the prior geometry's small measure error and separately recorded.

Production versus split residual agreement is <=6.576e-16. Normal and forced
exception restoration pass. The audit takes 19.520 s, peak 598,544 KiB. Its
intentional `K3_SEAM_AUDIT_COMPLETE` exception stops before mechanics and is
not a claimed nonlinear convergence result. See `summary-corrected.json`,
`columns-corrected.csv`, and `seam-audit-corrected.png` in the same directory.

## Other verification

The updated Debug executable passes 888 assertions in 21 domain/geometry and
Stage-I unit cases on one rank, and the same 888 assertions per rank on two
ranks. The filter is
`[particle_domain_constants],[particle_domain_area],[fault_domain_quadrature],Stage-I*`.
The domain moment tests retain their nonconstant manufactured fields and
independent references; failure/active-set/rollback unit coverage remains.

The focused integration logs are `focused-actions.log` and
`focused-mpi-lifecycle.log`. A sandbox MPI-launch failure is distinguished
from numerical failure and is retained. No full ASPECT suite is run.

| Focused integration fixture | Result | Elapsed |
| --- | --- | ---: |
| `phase_field_fault_changed_loading` | Pass | 76.05 s |
| `phase_field_fault_condensed_adiabatic` | Pass | 135.17 s |
| `phase_field_fault_condensed_adiabatic_mpi` | Pass | 104.53 s |
| `phase_field_fault_stage_i_rollback` | Pass | 62.71 s |
| `phase_field_fault_stage_j` (two ranks) | Pass | 359.44 s |

The first Stage-J launch was blocked by sandbox MPI sockets before numerical
execution; the MPI-enabled invocation above passes initialization and two
real feedback steps. Both Debug and Release targets were rebuilt with `-j4`.

The unchanged guard test passes, including its deliberately perturbed history
contribution: unchanged h containment cannot hide failed total supported
crack-strain normalization. The periodic replay's new output/library paths are
recognized by the existing common-dt guard; its numerical thresholds and
predeclared reference signal remain exactly unchanged.

## Coupled replay

The completed case is `evolving/spatial0375_n128_f32_periodic.prm`, inheriting
the exact existing 32x128/fault32, dt=0.375 s fixture through 3 s. Only output
and rebuilt plugin paths differ, verified against the fully resolved old
fault32 parameters. It uses fresh initialization, not transferred old nodal
history. Runtime is **120.403 s**, peak RSS **740,440 KiB (723 MiB)**, below
the 300-s process cap. Opt-in timing records 10 domain/CPDI generations taking
9.15 s; no separate performance campaign was run.

Initialization and all eight real steps pass every benchmark guard, including
phase/mechanical convergence, fixed geometry, lifecycle/state restoration,
admissibility, containment, complete supported normalization and whole-fault
homogeneity including the seam. All **37 fresh-linear checks pass**; maximum
fresh residual/requested target is 0.952751. There are 33 free and zero active
nodes throughout. The trajectory records 350 particle wraps without a seam
failure. Maximum surface balance RMS is 5.787e-6 Pa; at 3 s it is 9.418e-13 Pa.
The independent Theta update check differs by at most 4.441e-16 s.

| Along-fault range at 3 s | Old wall-domain run | Corrected periodic domains |
| --- | ---: | ---: |
| H [Pa] | 1.306964e-3 | 5.926657e-9 |
| phi | 1.643541e-5 | 1.487689e-10 |
| I_h [m] | 4.756832e-3 | 9.538184e-10 |
| C [Pa] | 8.844695e-3 | 3.883486e-9 |
| V [m/s] | 5.098687e-7 | 6.637442e-14 |

The largest omitted h fraction over all accepted states is **5.836624e-5**;
the largest actual supported total crack-strain normalization error is
**5.335760e-5**, both below their unchanged 1e-4 limits. At 3 s the measured
instantaneous/history/total integrals are respectively approximately
2.243333322e-3, -6.45835e-10 and 2.243332676e-3 m/s; the history term is
retained with its sign. Maximum phi is 0.599921925, below 0.8.

Initial particle IDs, positions, H and retained Maxwell stresses are exactly
unchanged. Initial surface differences are roundoff: V <=5.476e-18 m/s,
Theta <=7.106e-13 s, C <=1.706e-12 Pa, I_h <=7.106e-14 m, and fault y
<=2.962e-18 m (x unchanged). These differences are recorded rather than
reset. No computational/history timeline is changed by the geometry fix.

## Independent reference and remaining convergence question

The existing independent continuum reference
`evolving/spatial0375_n128_f32-reference` is reused only after verifying the
exact accepted time, dt and loading at every state. Reference histories remain
initialized once, never reset from later production values. The comparison is
`evolving/spatial0375_n128_f32_periodic-comparison.json`.

| Final quantity | Production | Independent reference | Relative difference |
| --- | ---: | ---: | ---: |
| V [m/s] | 0.002243450929 | 0.002243185919 | +0.011814% |
| Theta [s] | 1.583863240 | 1.583478199 | +0.024316% |
| C [Pa] | 356.6070803 | 355.6267695 | +0.275657% |
| I_h [m] | 108.1037147 | 108.1382272 | -0.031915% |
| Accumulated slip [m] | 0.005178829508 | 0.005179219256 | -0.007525% |

Maximum transverse phi error is 2.172693e-4 (initially 2.074680e-4).
Maximum H-profile error is 0.0230953 Pa, already present initially. Final-step
H/phi increment-profile errors are 3.880361e-4 Pa / 1.769252e-6. Raw stress
RMS/max errors are 1.327016/4.106648 Pa, versus initial 1.089703/3.187990 Pa.
The final I_h increment is about 0.00561957 m versus 0.00330470 m in the
reference: the small total-field error does not establish convergence of this
small feedback signal. Initial representation and normal-resolution errors
remain visible, not subtracted from the physical trajectory.

The periodic-domain mechanism and coupled homogeneity failure are resolved
for this tested configuration. **K3 spatial/temporal convergence is not
established; Gate K2 remains unmet.** A subsequent common-dt normal-resolution
comparison must use the corrected domain rule and fresh physical initialization
on both levels. No additional grid level or trajectory is run in this task.

## Recoverable evidence

`evolving/periodic-domains/summary.json` contains resolved-parameter checks,
per-state gates, rank comparisons, initial differences, exact resources and
SHA256 hashes of the tested production files, test source, executable and
coupled plugin. The Release executable SHA256 is
`02aa66fe96898b60c36913f41471fd113cca63dad5155c173af05551c3a56c70`.
`summarize.py` regenerates the summary from preserved outputs without another
simulation. The replay output, log and resources use the distinct
`spatial0375_n128_f32_periodic` name; previous wall-domain and failed evidence
are untouched. Changes remain uncommitted alongside the preexisting worktree.
