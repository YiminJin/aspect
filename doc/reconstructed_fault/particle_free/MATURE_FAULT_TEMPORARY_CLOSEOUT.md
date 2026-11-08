# Mature moving fault: temporary closure, 7 November 2026

The selected mature stage is closed around **four completed runs**, not declared
converged or production-ready. Propagation formulation recovery is the next
selected task. The proposed additional h=1/16 mature run remains deferred.
The [complete numerical report](DGQ2_COUPLED_VALIDATION_REPORT.md) and its four
figures remain the evidence; this note defines the preserved baseline and limits.
[Profile/balance](../../../benchmarks/reconstructed_fault/particle_free/evidence/dgq2_overnight/profile-balance-shape.png),
[traction/slip/state](../../../benchmarks/reconstructed_fault/particle_free/evidence/dgq2_overnight/coupled-traction-slip-state.png),
[moving-region fields](../../../benchmarks/reconstructed_fault/particle_free/evidence/dgq2_overnight/moving-region-fields.png),
[energy/cost](../../../benchmarks/reconstructed_fault/particle_free/evidence/dgq2_overnight/energy-runtime-subcycles.png).

The actual root is `/home/ein/repository/aspect-pf-rsf-particle-free`, branch
`pf-rsf-particle-free`, HEAD `36de6bcf3633dce96f904d27b1256c37b6675ccc`.
Seven incoming tracked documentation/analysis edits and the untracked overnight
support/evidence were preserved. All 950 recorded numerical source hashes match
the tested native-integration baseline `84f57503c`. Normal revert `319b10b94`
remains in place; shared-assembler `3a21d59cf` remains on its review tag.
No numerical source, quadrature, constitutive law or default changed in this closure.
The existing commit and source-hash manifest are the immutable source record;
no additional numerical patch or full repository archive is needed.

| Run | Drift | Mesh | Physical dt | End / accepted steps | Wall minutes |
|---|---:|---|---:|---|---:|
| A | +2 | 96×32 | .0125 | .8 / 64 | 33.15 |
| B | +2 | 96×32 | .00625 | .8 / 128 | 57.43 |
| D | +2 | 144×48 | .00625 | .8 / 128 | 123.25 |
| C | −2 | 96×32 | .0125 | .8 / 64 | 32.90 |

All 384 physical steps accepted without retries and published the joint material
state once. These are fresh, growth-disabled, layered-viscosity, direct physical
stress calculations, with bulk-slip-subtracted accepted spin, compatible exterior
stress/top loading, ordinary DGQ2 tensor history and DGQ2 phase composition.
Twenty material labels stay fixed across the comparisons. Accumulated slip spans
about 9.9–19 Dc: substantial stable sliding, not earthquake nucleation. SSPRK3 uses three
substeps per A/C step and two per B/D step; physical dt halving is not internal
phase dt halving. These are not four independent constant/layered constitutive
qualifications or propagation calculations.

Preserved files are indexed by
`benchmarks/reconstructed_fault/particle_free/evidence/propagation_recovery/entry-preservation.json`:

- Exact inputs, fault files, commands, logs, compression ledgers and all retained
  rotating checkpoints under `build-particle-free/transported-objective/stage-c/`
  `material-runs/layered-phase-integration-overnight-{a,b,d,c}/`.
  Last-good checkpoint slot markers are respectively 2, 3, 3, 2. Preserve each
  entire `restart/` directory with its marker; checkpoint cadence was 16 steps
  plus termination, not an archive of every historical checkpoint.
- Every file of the compact `evidence/dgq2_overnight/` record, including input
  identities, norms, energy reductions, frozen quadrature and PNG figures.
  Accepted bulk CSVs remain losslessly compressed in place, with plaintext
  hashes in each compression ledger. No accepted output was deleted here.
- Seven retained executable/plugin hashes, independently rechecked. Active core
  `build-particle-free/native-phase-integration/native/aspect-release` is SHA256
  `6ea374c5b9b50a20af00fa6e2c365df61d7ac59ed33ad26c7b338d7bb7ba67d9`.
  The full checker/provider identities are in the manifest. An embedded old Git
  banner is not the source identity. This closure does not rebuild these files.

This is in-place preservation, not an off-machine backup or a new Git commit.
Existing shared-assembler review artifacts and older protected binaries remain
untouched. Inherited cleanup losses remain **39 of 45 native-pilot frames and
earlier temporary logs**; the closure does not reconstruct missing evidence.

The native phase balance is excellent (absolute accumulated defects ≤3.22e−15),
but does not establish coupled accuracy. B/D final central/end stress differs
5.48%/3.71%, normal traction 2.52%/4.95%, slip rate .57%/2.73%, and age
.98%/4.13%. Temporal A/B final age differs .44%/1.92%, with selected-time
transient differences reaching 9.67%/12.32%. The mesh comparison refines bulk
mechanics, pressure and history together; it does not isolate phase accuracy.
Initial projected fields differ across meshes, and two meshes give no asymptotic
order. Fixed label resolution remains a separate uncertainty.

Endpoint absolute work remainder / active work is .269/.137/.140/.246% for
A/B/D/C. Signed totals are retained separately, as are backward-Euler and
projection terms. Localized-work Gauss8/12/16 sensitivity is not monotone in
every quantity. C has a larger trapezoidal remainder; finite-box direction
comparison is not an exact reversal symmetry test. Finite-end stress approximation,
early temporal state accuracy, normal traction, label resolution and energy/flux
accuracy remain open. The same tests cannot certify a new cohesive fracture
inventory, moving cohesive shape work, or a changed spin law.

The phase-integral changes (−.00530/−.00757/−.00114/−.00464%) are accounted for
by the discrete phi div(v) source, with negligible boundary flux. In this
incompressible model, nonzero computed divergence is a discretization
contribution, not physical compressibility or exact continuum conservation.
Phase transport takes less than .72% of runtime. The following are measured
comparisons, not errors against an exact coupled solution:

| Conclusion | Supporting evidence | Remaining limit |
|---|---|---|
| Joint moving-state algorithm runs in both directions | 384 accepted steps, native phase balance ≤3.22e−15 | Growth disabled; no propagation claim |
| Bounded phase representation is useful and inexpensive | Phase mesh difference .414%; transport <.72% of cost | Does not certify mechanical stress or surface labels |
| Mechanical accuracy still needs refinement | B/D stress 5.48%/3.71%; endpoint age 4.13% | Two meshes, traction ripples, fixed 20 labels and localized quadrature |
| Final temporal differences understate the transient | A/B early age differences about 10–12% | No absolute solution or temporal convergence order established |

Stress differences refer to the full physical tensor. The report separately
compares **total driving shear**, including the common background loading 3.7
(B/D .943%/.657%), and work-projected total shear (.0418%/.1368%). Raw
perturbation-shear percentages are not total driving-traction errors. Effective
compression includes the specified background pressure; pressure-inclusive
normal traction is also reported without that denominator convention.

Reproduction uses the existing `dgq2_phase_field/run_overnight.py` and
`analyze_overnight.py`; each run's `command.json` is the exact resolved launch
record, and `evidence/dgq2_overnight/inputs/` contains the frozen inputs. Reproduce
only into fresh output directories after explicit selection. The h=1/16 run,
surface/quadrature refinement, parameter sweeps, architecture work and production
paper runs are **parked**, not discarded or automatically authorized.

See [propagation recovery](PROPAGATION_FORMULATION_RESTART.md) for the formulation
separation, recovered pause point, short checks and the selective retest matrix.
