# Birth identity and production boundary buffer — review

Birth identification is corrected, and the **actual 60° production mesh passes
boundary-completion admission** with its retained endpoint buffer. Full production
normalization and mechanics startup are **not yet qualified**: the bounded runs
hit the geometry-preparation time limit and the matrix-allocation memory limit.
The old unbuffered 45° failure remains separate and does not invalidate the 60°
boundary result. No solver tolerance, physical parameter or core completion rule
was changed. Stop here for review; no server run, later cleanup section or R7.

Starting point: `1678949c2` plus the incoming Section-5 changes, committed
separately as `f935bd94b`. This corrective task is committed after review. The
incoming tracked patch and original local observer remain in `evidence/`. The
accepted core/plugin artifacts remain byte-identical; current hashes are in
`results/artifacts.json`.

## Changes and ownership

- `Particle::Manager::post_particle_creation` observes all three native insertion
  branches after initialized properties are installed. It does not change ID
  allocation, placement, fitting, RNG, property storage or checkpoint layout.
  The movable manager moves its signal connections. Slots must not mutate
  particles or call MPI collectives.
- BP3 records pending birth IDs, checks their H with the shared initializer and
  publishes their baselines at the **existing post-management audit**. An actual
  insertion replaces a retired ID's old baseline. Migrated/restored survivors
  must retain theirs. The current-attempt birth flag travels as one extra byte
  in the existing H transfer payload; no MPI collective is added. Native backup
  clears attempt flags; rollback restores H baselines and clears rejected births.
  Removed/old ghost entries are pruned as before. Version-5/6 archives still load;
  the legacy next-ID field is consumed but no longer used to infer birth identity.
- Accepted population output counts surviving event-identified births. The local
  coupled observer also starts a reused-ID newborn's path at zero. Timestep
  acceptance and physical history remain in their existing owners.
- The maintained `BP3 fault support` mesh now retains the **same endpoint strip
  as the successful local A/B meshes**, alongside A's original exterior slope.
  With profile radius R and configured coarse upper bound h_c, its boundary
  half-width is `(R+h_c*(abs(n_x)+abs(n_y)))/sin(dip)`, plus one fine edge; depth
  is two fine edges. Native smoothing and all resolution/coverage checks remain.
  Core support, associations, ordering, overlap checks and ghost-Q1 admission
  are unchanged. Resolved model identity includes the buffer; old unbuffered
  checkpoint identities are not silently accepted as the new mesh policy.
- The production PRM points to `bp3/build-maintained-buffered/`, containing the
  byte-identical tested default-build library. It depends on no test observer or
  fixture at runtime. The old `bp3/build-maintained/` library is preserved.
  Rebuild plugins against the corrected core C++ interface.

## Focused verification

Release builds pass for core and maintained/local/test plugins with GCC 12.4,
OpenMPI 5.0.6 and deal.II 9.6.2, including native 2D/3D template instantiations.
Runtime cases are 2D. Original numerical tolerances and solver gates are retained.

- **961/961** matched lifecycle checks against Section 3: serial retry, two-rank
  nonzero-stress births, checkpoint continuation, and two-rank retry. Particle,
  bulk, weak-row and history replay is bitwise exact; solver decisions, fresh
  residual gates and RNG replay agree (`results/comparison.json`).
- **424/424** birth/control checks: actual event flags agree with an independent
  known-flow ancestry check on one/two ranks; retained H is exact, newborn H
  matches the shared initializer, and restarted particle/history/birth flags
  match uninterrupted steps 3–4 exactly. With identical initialization, enabling
  the audit changes **none** of the transport diagnostic columns, including H,
  tensor inheritance, native LLS and continuous-Q2 errors.
- **6/6** dedicated reuse-step rollback checks: restore the native backup and
  repeat step 3, including its outflow and ID reuse. Full particle properties,
  reference positions, membership, manager/generator RNG and audit identities
  agree exactly, on one/two ranks. Later accepted particle snapshots agree too.
- **44/44** short A/B comparisons: unchanged meshes, velocity, fault fields,
  friction inputs, particle counters and solver decisions at steps 0–2, with
  fresh residual gates. The new birth-path guard is inert for retained particles.
- The standalone maintained production PRM validates with the corrected binary
  and buffered library. No Debug, 3D runtime, changed-rank restart or long-event
  qualification is claimed. Random/histogram insertion notifications compile;
  this regression exercises the selected point-density policy.

The real reused ID is **202495**. After leaving near `(156.56171875,
999.99171875)`, its new particle appears near `(-155.9244791667, .3255208333)`.
Active IDs remain globally unique. On both rank counts:

| Step | Current particles | Actual surviving births | Old threshold count | Losses |
|---|---:|---:|---:|---:|
| 3 | 201443 | **72** | 71 | 1125 |
| 4 | 201304 | 582 | 582 | 721 |

The original standalone transport fixture omitted BP3's post-manager startup-H
initializer. Attaching the production audit exposed that mismatch; the dedicated
fixture now invokes the **existing shared initializer** before the audit, just as
production does. This is a fixture correction, not a change to production H.
Against the old transport output, all non-H columns and survivor-H-change checks
pass; 15 H-minimum comparisons differ (largest cell-minimum difference
12254.4161). Those failed historical comparisons remain recorded in
`results/historical_transport_comparison.json`. The matched-initialization
control above separates this difference from the birth correction. No H
comparator was loosened and no accepted particle's H is rewritten during moves.

## Actual production boundary qualification and limits

The input keeps the 150×50 km Box, native fault.txt, 60° dip, peak .6, ell=20 m,
levels 9/1/0, 20 m structural spacing, frozen mature material, A slope, native
4×4/12–24 particles, LLS/Q2, filter20 and production physical/solver settings.

| Full production mesh quantity | Result |
|---|---:|
| Cells / support cells | 552084 / 319324 |
| Cell-edge range | 3.90625–1000 m |
| Total / Stokes DoFs | 19498528 / 5161624 |
| Expected initial population, 16 per cell | 8833344 |
| R / actual global projected-width padding | 39.5291640 / 1366.0254038 m |
| Required boundary half-width, each endpoint | **1622.9946162 m** |
| Aligned fine boundary faces covering each footprint | **832** |
| Core contact influence length, each endpoint | 811.4973081 m |

The full-mesh footprint CSV is **byte-identical on one and two ranks**. Every
intersecting boundary face has the required fine spacing/alignment and the
entire core footprint is enclosed. Ordinary mesh checks enforce the buffer.

The two-rank material probe initializes the full native mesh, DoFs and particles,
then invokes the exact native mature phase entry (essential-data lift, with no
matrix access), native reconstruction, and public material preparation. The
core's automatic contact/enclosure/source checks and collective prescribed-Q1
compatibility checks return successfully. A read-only debugger backtrace shows
execution in the subsequent `prepare_cell_normalization_geometry()` R-tree query;
`compute_normalization_integrals()` calls all automatic compatibility checks
before reaching that operation. This is direct evidence of **boundary admission**,
not a completed normalization or coupled solve.

The material probe reaches its **600 s** cap there (peak aggregate RSS about
10.54 GiB), before full Ih publication and the final pass marker. The normal
production callback attempt instead reaches the **20 GiB** aggregate RSS guard
during matrix allocation. Neither is reported as a successful full startup.
The unchanged query encloses each entire clipped normal line by an axis-aligned
box, then filters candidates by the existing slab clip. The backtrace identifies
this cost; no new geometric cutoff, cache rule or numerical correction is included.
No matched full-production pre-change timing run was made, so this report does
not label the measured resource cost a demonstrated performance regression.

Earlier probes called preparation before normal velocity-constraint/phase entry;
their failures are retained. The final probe uses the native phase operation.
Other retained harness failures are missing restart-branch preparation, the old
observer expecting post-management publication, and a sandbox MPI-socket denial.
The publication timing was preserved, branches were prepared from fresh tested
checkpoints, and MPI runs used the authorized socket-capable runner. An initial
wrong-toolchain build was canceled and every affected unit rebuilt with GCC 12.4.

## Reproduction and next bounded task

Build core with the matching PATH, then configure maintained sources with
`Aspect_DIR=build-refactor-r6b`; the `local` target also sets
`BP3_LOCAL_OSCILLATION_TEST=ON`. Build the three test-only CMake directories here
(`transport`, `observer`, `qualification`) after the maintained library.
Exact build logs, input copies and runtime commands are in `evidence/`.
All attempts total 726.8 s of focused local simulation and 941.2 s of production
probes (including failures and the 600 s timeout); builds are separate.

`prepare_inputs.py` prepares final fresh inputs. Run `create`, `staggered`,
`direct`, `retry`, and the two transport parents before `prepare_branches.py`;
then run `resume`, `direct2`, `retry2`, and transport resumes. The runner accepts
`CASE RANKS [UNIQUE_LABEL]`, refuses log replacement, and bounds runtime/RSS.
Run the three `compare_*.py` scripts and `verify_retry.py`. Production mesh-only
qualification stops intentionally; production material preparation currently
hits its recorded bound and must not be treated as a final-pass test.

**Recommended next task:** bound and profile the existing cell-normalization
candidate query on this actual mesh, then propose the smallest exact search
improvement preserving slab clipping, shared-face ownership, interval ordering
and cache criteria. Full normalization and a memory-suitable mechanics startup
remain prerequisites for a production trajectory. The 45° unbuffered case can
remain a separate limitation.
