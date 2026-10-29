# Temporary candidates and energy-accounted formation

Baseline `f1f9f29cf40c32d1ddcea000b724375b64853532` plus preserved work.
[Formulation](COHESIVE_INTERFACE_CANDIDATE_ADMISSION_FORMULATION.md) ·
[declared tests](COHESIVE_INTERFACE_CANDIDATE_ADMISSION_SELECTION.md) ·
[PNG/GIF/data index](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_candidate_admission/INDEX.md).

**Implemented and qualified the ownership distinction in the bounded serial
native experiment.** Unused zero-history geometry no longer commits direction.
New material commits only with an accepted physical update that includes its
cohesive storage, fracture work and dissipation. Two versus eight temporary
cells produce identical persistent geometry and material histories within
roundoff. Production/BP3 and the previous permanent-reserve experiment remain unchanged.

## What changed

The initial state is the original accepted .410 s checkpoint **before** the
second append. Both existing pieces carry history; they are retained verbatim.
Their nonzero bulk Maxwell field, six Gauss histories, reference slip, clocks
and physical quadrature ownership are preserved. There is no reinterpretation
or conversion of an old wholly virgin committed piece.

New temporary cells must have exactly zero damage, signed/absolute slip,
reference and relaxed memory, plus exactly zero associated native nodal modes.
Their age follows the existing virgin-clock prescription. Bulk tensor history
in their physical footprint is generally nonzero and is **not** reset.
Any physical history makes a cell persistent, without a magnitude threshold.
One new Q1 cell (.09375 m) is the geometric commit unit; partly formed cells
retain their direction even where some of their Gauss points remain intact.

The optional native solver eliminates Q1 nodal coordinates analytically implied
by tied Gauss constraints before the remaining SVD. This replaces numerical
null-space shadows with exact independent masters; it does not clip an increment
or discard old history. Default behavior remains off. Retained physical response
columns/pressure and Gauss coordinates/weights match bitwise across view changes.
All measured replacement stress and storage defects in the trajectories are zero.

Two propagation/numerical definitions are explicit. The same R=.75 m averaging
disk is now anchored at the **formed endpoint**, rather than an unused reserve
endpoint. Fixed Gauss8 sampling removes dependence on candidate cut partitions;
Gauss12 differs by 5.89e−8 in full-tensor relative norm and 5.75e−8 degrees.
Mechanics stays Gauss6, surface history Gauss3, and numerical lifting/normal
half-width .75 m. Physical R, numerical support and candidate extent remain separate.
The accepted-stress maximum-shear/persistence hypothesis is retained, not replaced
by Coulomb-angle initiation or a global energy minimization rule.

Candidates remain fixed within each nonlinear attempt. A cached trial that
would form material is rejected and rebuilt from the actual incoming accepted
stress before publication. The same original histories feed every retry.
If growth reaches the candidate edge, enlarge the temporary view and re-solve
from that same input. Only the final endpoint publishes geometry and history once.

## Native evolution and numerical controls

Use the unchanged physical loading turn over .410–.425 s, the same evolved
incoming state, homogeneous cohesive–RSF calibration, current-pressure coupling,
confinement, radiation and fixed 64² Q2/P1disc mesh.

| Control | Physical steps | New formed cells | Growth in new-region ∫a ds, m | Absolute physical remainder / sliding | / added fracture |
|---|---:|---:|---:|---:|---:|
| 2 temporary cells, dt=.00025 s | 60 | 8 | .310009318 | .4877% | **1.4757%: fail** |
| 8 temporary cells, same dt | 60 | 8 | .310009318 | .4877% | **1.4757%: fail** |
| 2 cells, dt=.000125 s | 120 | 8 | .308321398 | .2440% | **.7371%: pass** |
| Half refresh cadence, .420–.425 suffix | 20 | 4 | .199360577 | .6118% | **1.6894%: fail** |

The last row retains the coarse timestep and excludes the common prefix; its
material response matches the corresponding coarse window exactly. It does not
repair time-integration error. No screen was loosened. The half step changes
growth by **−.5445%** and signed-slip increment by less than 1%. Its process front
remains in the same Gauss bracket; a 1.15e−7 m geometric difference is not sub-grid
front accuracy. There is no claim of a convergence order from two timesteps.

For 2 versus 8 candidate cells, persistent geometry and record IDs match exactly;
maximum damage/age differences are 3.33e−16/6.66e−16, and the maximum physical
stress-or-pressure component difference is **1.36e−12 Pa**. Regional work-conjugate
fault loads are exported on common tests. Full-tensor maximum difference is
1.82e−12 Pa, including zz and both symmetric xy contributions. The .0025→.00125 s
cadence checks have zero measured response change over both the active suffix
and .410–.41675 s onset control. The first two admission clocks remain .41525
and .41675 s; the first is bracketed by [.415,.41525] s, not an exact physical
onset time. Cached predictor state is
still numerical algorithm state; arbitrary cache changes between refresh clocks
are not assumed to be exact identities.

Eight cells, .75 m of geometric records, are admitted. Their damage is real, but
they are a **cohesive process zone**: no new fully decohered front advances in this
short window. The new-region damage integral ends at .347424 m. Accepted sampled
preference changes approximately 19.67°→10.24° through physical boundary loading
and ordinary Maxwell advancement. Old curves and tensors are never rotated.

## Admission energy and lifecycle

The original late failed state is retained. Its legacy pure-tied birth guard
still rejects on a private copy of the original operator/evidence, without
overwriting that evidence. A fresh temporary view has virtual peak **3065.76 Pa**,
above the unchanged 3000 Pa strength. It is not published as a tied surface.
The native positive-time joint solve admits one cell and converges at
5.99e−9 Pa weak defect, with unchanged incoming physical stress/history.

The admitted region receives **.000317655 J/m fracture work**, .000316220 J/m
cohesive storage, and .000001752 J/m friction plus radiation loss. Its endpoint
work is .000634193 J/m, with the explicitly retained −.000001435 J/m surface
endpoint remainder. The whole-step exact BE defect is 2.89e−13 J/m. These are
physical-update terms, not a zero-time registration fee.

An independent promotion inventory gives, before/after restricting the solved
endpoint to persistent material, exactly the same bulk **468.680116297 J/m**,
cohesive **.097560148167 J/m** and fracture **3.021311472141 J/m** storage.
The remaining temporary contribution is zero. No second Gc times registered
length is charged. The CSV first-admission increments are timestep-dependent;
the separate new-material inventory includes all subsequent damage work too.

Actual tests cover a residual-valid stale-direction predictor rejected before
publication, deterministic fresh retry, immutable original histories, one commit,
duplicate/false-zero-cell promotion rejection, and native-context JSON restart
with nonzero formed histories. Changed zero caches preserve physical histories
when both controls force the same fresh selection. A one-cell trial reaching its
edge enlarges and retries once; all six histories match direct two-cell padding
exactly. This is not a one-cell-per-step growth cap.

Coupled tangent FD errors are **6.27e−10** initially and **5.52e−10** during
new-cell admission against 1e−6; independent Q1 projector
error **4.54e−16**. Eight default operator/trace/probe tables remain bitwise
identical. A candidate with even 1e−30 m stored reference or nodal memory is
refused retirement. Native weak, contact, irreversibility and exact ledger gates
remain unchanged. Rebuilt-context restart is experimental JSON ownership, not
production Simulator/MPI checkpoint qualification.

## Scope, cost and next task

The coarse physical-energy failure is preserved. Half stepping meets the
declared energy screen within this fixture; this does not close historical
spatial/branch findings. Prior point-traction defects (~520 Pa versus 16.6 Pa
on the retained fitted comparison), the 2.85%/2.29 Pa ahead-normal failure and
the small old-region normal failure remain explicitly open. Preservation of a
numerical stress field is not proof of its point accuracy.

Timed trajectories ran sequentially: 389/473 s for 2/8 cells, 399 s for half
stepping, 42 s for the cadence suffix and 54 s for its onset control, including on-demand kernels and cache
reuse. Rebuilding/validating fixed-bulk kernels dominates cost; these are not
production speed estimates. Raw operators/checkpoints stay separate from the
compact figures/CSVs under the declared 1.2 GiB budget.

The bounded ownership/admission task is complete. Before a longer directional
claim, repeat a short part of the full loading turn at the qualified half step,
with one refinement of **new material-cell commit spacing**, preserving the
existing formed registry, physical R and numerical support. Retain a common-field
fitted check of relevant weak loads. This tests the remaining one-cell formation
extent and later developed advance. It is recommended, not started. General
topology, secondary fracture, AMR/transport, MPI and production integration remain later work.
