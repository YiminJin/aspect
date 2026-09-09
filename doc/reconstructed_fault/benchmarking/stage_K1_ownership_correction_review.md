# K1: deduplication audit and approved CPDI ownership correction

## Decision and provenance

Keep the current roundoff-scale Voronoi vertex deduplication. Correct only
2-D CPDI sampling-point ownership. H, phase initialization, physical parameters,
constraints, solver tolerances and geometry algorithms are unchanged. No commit
has been made for this task; earlier K1 and unrelated worktree changes remain.

**Documented history:** commit `f4032b1824ef4892020af8c58ef981aca03dc0ec`
introduced `64*epsilon*coordinate_scale`, 64-bit spatial-bin indices and
`[particle_domain_area]`. The Stage-J recovery report, section “Advection exposed
a separate particle-domain defect”, records raw Voronoi area
0.99999999999998557, old-cleanup area 0.99999068326658858 and corrected-cleanup
area 0.99999999999998568 for deterministic 1e-6 particle perturbations.
The recorded reason is that independent merging of genuine short edges opens
gaps between neighboring polygons. This rationale predates the present K1 issue;
it is not inferred solely from the source comment.

**New controlled comparison:** before editing production source, GDB changed
only the local deduplication tolerance to the old domain-diameter formula.
It retained the current 64-bit bins and all other code. On the existing perturbed
unit-square test, the old tolerance produced area **0.99999068326554508** and
triggered the unchanged production volume assertion. The debugger itself exits
zero after reporting SIGABRT; this is a **failed target test**, not a pass.
The small difference from the historical area does not affect this conclusion.
The current tolerance passed the original 1e-10 area assertion before and after
the ownership correction.

On the unchanged K1 initial layout, the old-tolerance capture stopped before
the first phase residual. Its **37,765 CPDI rows and 9,216 particle rows are
byte-identical** to the current-tolerance baseline, including all 68 defective
stencils. Thus the tolerance change is **not the trigger of this confirmed
ownership failure**. Reverting it would not fix K1 and demonstrably reinstates
the separate area failure. No claim is made that the tolerances produce the
same polygons for every other layout; the perturbed-grid test proves otherwise.

The exact short-edge mechanism remains supported by the historical raw/cleaned
comparison; this task independently reproduced the area loss, not a new
edge-by-edge geometric derivation. The ownership cause itself is directly
captured in `stage_K1_cpdi_cause_and_correction_plan.md`: a physical point below
y=0 maps to reference y=1 in the lower cell and a tiny negative y in the upper
cell. Both independent half-open tests reject it.

## Production correction and invariants

`source/particle/particle_domain.cc` now determines one owner per polygon corner
and, when used, centroid before cell-wise CPDI integration. Existing half-open
acceptance takes precedence. Otherwise a cell within the existing 1e-12 reference
tolerance is chosen by minimum reference-cell max-distance, then stable CellId.
Only that fallback coordinate is projected onto its selected unit cell. If no
candidate supplies support within tolerance, construction reports a numerical
support failure instead of silently dropping the sample. Multiple accepted
candidates also contribute only once, with a stable CellId choice.

Deduplication, polygon vertices, centroids, volumes and triangle integrals are
untouched. The generic 3-D path is untouched. The helper resides in the existing
source-only internal namespace; no production header/API was added. Unit tests
forward-declare it to exercise production ownership, not a duplicated formula.
No global-weight renormalization or gradient clipping was introduced.

`unit_tests/particles.cc` adds two tagged tests: exact/next-representable face
points plus the captured gap, and whole production particle-domain construction
on the periodic K1-sized regular layout and a nonuniform refined layout. The
tests require constant reproduction, zero constant gradient and area conservation;
the face test also checks the affine Q1 trace and rejects genuinely missing support.
The MPI test reduces diagnostics before assertions so both ranks share the
collective lifecycle. This checks constant reproduction on hanging-node meshes,
not a general affine-gradient or hanging-node transfer certification.

## Verification and exact results

Commands below run from the repository root unless indicated. Logs are under
`benchmarks/reconstructed_fault/uniform_shear/diagnostics/results/`.

| Command/check | Result |
| --- | --- |
| `cmake --build build-pf-cpdi --target aspect -j4` | Passed; existing unused-parameter warnings in the unimplemented 3-D Voronoi specialization |
| `build-pf-cpdi/aspect --test '[particle_domain_constants],[particle_domain_area]'` | Passed: 73 assertions, 3 test cases |
| `mpirun -np 2 build-pf-cpdi/aspect --test '[particle_domain_constants],[particle_domain_area]'` | Passed: 73 assertions, 3 test cases on each rank |
| `ctest --test-dir build-pf-cpdi/tests --output-on-failure -R '^phase_field_particle_domains$' -j1` | Existing two-rank regeneration integration test passed, 53.86 s (CTest total 53.90 s); no expected-output update |
| `python3 benchmarks/reconstructed_fault/uniform_shear/diagnostics/check_cpdi_cause.py` | Three saved pre-correction polygon/stencil causal replays pass |
| `python3 benchmarks/reconstructed_fault/uniform_shear/diagnostics/check_stationary_equation.py` | Analytic identity and five three-spacing finite-difference checks pass; activation mismatch confirmed, unchanged |
| `python3 -m unittest discover -s benchmarks/reconstructed_fault/uniform_shear/diagnostics -p test_packet.py` | Three baseline integrity tests pass, 0.018 s; existing VTK/NumPy deprecation warning |
| `git diff --check` | Passed |

The production initialization-only run uses `../corrected_initialization.prm`
from `diagnostics/build`, with `timeout 300` and the same debug executable.
Only the output directory differs from the saved effective parameters.
It completes in **29.71 s**, user 27.86 s, system 0.24 s, maximum RSS
**472872 KiB** (zsh built-in timing). It exits **1 intentionally**, with
`K1_INITIALIZATION_DIAGNOSTIC_COMPLETE`, after exporting initialized fields and
before the first mechanical residual. This is neither solver failure nor an
accepted mechanical timestep. There are 14 Newton updates, each one CG iteration
and zero line-search reductions; final relative residual is 1.206e-9. Independent
reassembly gives 1.205790477980762e-9. The original requested accuracy is unchanged.

`compare_ownership.py` verifies byte-identical initial H/positions/volumes,
mesh and starting phi, unchanged effective parameters except output destination,
and **37,473 exactly unchanged stencil rows on all previously unaffected particles**.
All 68 defective particles now reproduce constants. Particles 3836 and 3986 change
from sums 2/3 and 5/6 to 1; control 3983 remains 1. Worst new weight-sum error is
4.440892098500626e-16; worst summed-gradient norm is about 5.00e-14 m^-1
(slightly dependent on summation order). H and stencils are also byte-identical
before/after the phase solve within the corrected run.

## Unchanged-parameter initialization comparison

| Quantity | Before ownership fix | After ownership fix |
| --- | ---: | ---: |
| phi(x,0) minimum | 0.1680575641 | 0.6005826557110254 |
| phi(x,0) maximum | 0.2277336449 | 0.6005826557110306 |
| Global maximum phi | 0.4021619885 | 0.6005826557110306 |
| Maximum along-x range over all transverse rows | 0.05967608077 | 6.8833827527e-15 |
| Along-fault vertex phi range | 0.05297211037 | 5.2180482157e-15 |
| Maximum reconstructed abs(y), m | 6.2203617339e-4 | 8.1082713761e-18 |
| Particle-domain volume sum, m² | 0.25000000000000006 | same |

Representative transverse values at x=0.125 m (all native FE y coordinates):

| y (m) | Intended stationary profile | Old FE phi | Corrected FE phi |
| --- | ---: | ---: | ---: |
| 0 | 0.6 | 0.1907592705 | 0.6005826557 |
| 0.0625 | 0.4898276666 | 0.3806158639 | 0.4897723900 |
| 0.125 | 0.2962362531 | 0.2426836058 | 0.2949353631 |
| 0.1875 | 0.1333037726 | 0.1032098201 | 0.1303320245 |
| 0.203125 | 0.1016536049 | 0.0753786899 | 0.0981700281 |
| 0.25 | 0.0317706543 | 0.0189720369 | 0.0299103606 |
| 0.3125 | 0 | 0.0004897443 | 0.0012146165 |

The intended profile is **not an FE constraint**. Its untruncated positive-support
H/profile formulas satisfy the same phase equation to 2.84e-14 Pa in the analytic
audit. Actual H retains 0.5 Pa where the intended phi is <=0.1; substituting the
intended profile there leaves a nonzero residual (11.0751591 Pa at the inactive
side of phi=0.1). The corrected center differs from 0.6 by 0.000582656, and the
maximum sampled transverse discrepancy is 0.0036125943. This task does not
separate discretization, tabulation and activation effects on that remaining
discrepancy or claim continuum convergence from one mesh. It does establish
that the large center trough and unintended x-dependence were caused by the
confirmed CPDI ownership omission.

## Recoverable artifacts and reproduction

Base directory: `benchmarks/reconstructed_fault/uniform_shear/diagnostics/`.

- `results/`: retained original raw/visualization packet; new build, regression,
  old-tolerance debugger and corrected initialization logs have distinct names.
- `results-old-dedup/`: old-tolerance initial particle/stencil CSVs.
- `results-corrected/`: corrected raw and native-Q1 VTU data, fault/normal VTU,
  particle VTU, profiles, two PNG previews, measurements and ownership comparison.
- `old_dedup.gdb`, `stop_before_phase.gdb`, `old_dedup_initialization.prm`:
  pre-correction debugger comparison inputs. Line-dependent scripts must be used
  with the pre-correction source/executable, not blindly replayed after refactoring.
- `corrected_initialization.prm`: output-directory-only overlay.
- `compare_ownership.py`: preservation and corrected-stencil checks.

The old-tolerance area replay used
`gdb -q -batch -ex 'set $old_tol = 1.414213562373095e-6' -x .../old_dedup.gdb -ex run --args build-pf-cpdi/aspect --test '[particle_domain_area]'`
under `timeout 180`. The K1 replay used `1.030776406404415e-6`,
`-x ../old_dedup.gdb -x ../stop_before_phase.gdb`, and
`--args /home/ein/repository/aspect/build-pf-cpdi/aspect ../old_dedup_initialization.prm`
from the diagnostic build directory under `timeout 240`.

After rebuilding, reproduce corrected initialization and comparison with:

```sh
cd benchmarks/reconstructed_fault/uniform_shear/diagnostics/build
timeout 300 /home/ein/repository/aspect/build-pf-cpdi/aspect ../corrected_initialization.prm
cd /home/ein/repository/aspect
MPLCONFIGDIR=/tmp/k1-matplotlib python3 benchmarks/reconstructed_fault/uniform_shear/diagnostics/visualize.py \
  benchmarks/reconstructed_fault/uniform_shear/diagnostics/build/output-corrected \
  benchmarks/reconstructed_fault/uniform_shear/diagnostics/results-corrected
python3 benchmarks/reconstructed_fault/uniform_shear/diagnostics/compare_ownership.py \
  benchmarks/reconstructed_fault/uniform_shear/diagnostics/results-old-dedup \
  benchmarks/reconstructed_fault/uniform_shear/diagnostics/results-corrected
```

Preserve existing outputs before rerunning. Original executable SHA256:
`1144c79341f60facac2365791ebdf21f10a2c331cb742aa43beb1f61dfa9311a`.
Corrected executable SHA256:
`5022898bddbc9762595547433056e3d09e8fe06bb9bbb812038d80b3ceca33cd`.

No additional issue discovered in this comparison required expanding the patch.
The previously documented polygon-orientation sign of affine gradients remains
unchanged and deferred; a common per-particle gradient sign cancels in this
phase gradient bilinear form. This is not a general CPDI gradient certification.
No full suite, full K1 mechanics pilot, convergence matrix, 3-D or K2 was run.
