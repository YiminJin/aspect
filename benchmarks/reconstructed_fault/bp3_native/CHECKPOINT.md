# Worktree checkpoint — 2026-10-15, task remains open

This checkpoint records the current implementation and measured evidence so a
new worktree can continue from branch `fault-propagation`. It is not scientific
closeout or authorization for another simulation. The parent revision is
`c33edaaeb145e5d3dce155bb815552ef80efc5f9`; the actual checkpoint includes the
previously uncommitted source, tests, fixtures, reports and design amendments.

## What follows into a new worktree

- Native mature-fault stationary FE history, general dipping normalization and
  the selected mechanical quadrature; the earlier P2 option remains available.
- The optional consistent-deviator particle convention, actual moving-particle
  transport, CPDI orientation/coordinate-conditioning corrections and unit tests.
- Maintained BP3 plugin changes and isolated local diagnostic controls; production
  defaults are preserved. The ordinary server handoff stays a stationary-history
  candidate, not a qualified moving-particle first-event configuration.
- All BP3 inputs and observers here, including strengthening/population controls,
  the original 5×5 density control, the weakening pulse and its tighter controller.
- [Reports and actual results](STATUS.md), the
  [ownership/update-sequence derivation](moving_particle_formulation.md), and
  [portable compact evidence](evidence/). The 59 copied files are byte-identical
  to their original build-tree artifacts; [checkpoint.json](checkpoint.json)
  records their origins, sizes and hashes.
- Manual [server build/environment instructions](server/README.md), fine-mesh
  input/fault data, original plugin snapshot and the separate deal.II 9.7 patch.
  The snapshot intentionally retains its original identity; it is not replaced
  with the subsequently changed diagnostic plugin.

## What stays in this existing worktree

Build trees, matching historical executables/shared libraries, complete raw
trajectories, particle/FE/RNG population-event snapshots, old source snapshots,
preservation backups and the original server tarball remain under
`/home/ein/repository/aspect-fault-propagation`. They are not copied by Git
worktree creation. [local-only-files.json](local-only-files.json) identifies the
2333 previously untracked local files now excluded from Git, with their hashes;
build artifacts were already excluded. No local evidence was deleted for this
checkpoint. The earlier disk-cleanup archive and restore instructions remain in
[CLEANUP.md](../CLEANUP.md).

Create a separate build/output tree for the new worktree. Reuse installed
dependencies and the recorded compiler/CMake settings; do not reconfigure this
worktree's builds or mix historical plugins with a newly compiled core. Local
diagnostic PRMs use `$ASPECT_SOURCE_DIR` and name specific plugin/output paths:
rebuild the appropriate plugins and use fresh output directories before running
them. Full-field assessment scripts require the original raw artifacts; compact
committed assessments do not substitute for those inputs. The earlier curated
[P2 checkpoint](../propagation/p2_checkpoint/README.md) has the same distinction.

## Outstanding scientific work

The latest accelerating pulse passes coupled recurrence, state, slip and
publication checks, but its accuracy exit remains unmet. Tightening the state
controller changes the peak time by about 1.02e6 s and the common-time velocity
by up to 43% of the primary peak. The lower-end pressure/traction feature remains
spatially unqualified. Longer managed-particle controls also retain unresolved
population-resampling and near-face trace sensitivities. These findings remain
in the [slip-front](slip_front_report.md), [fixed-state event](managed_event_report.md)
and [managed-particle](managed_particles_report.md) reports.

The next proposed investigation is the consistently tighter active timestep
restriction for this diagnostic pulse, retaining sparse accepted states for
endpoint analysis. The user must select further scientific scope. No new
trajectory, full first-event run, cluster submission, restart/cutback or physical
rotation-law change was performed or qualified for this checkpoint.

## Verification for committing

This checkpoint changes documentation/artifact curation only beyond the retained
incoming source. Checks cover incoming-source preservation, byte-identical
evidence copies, Python/JSON syntax, parameter includes and staged whitespace.
Existing numerical/build results are retained in the reports; no scientific
test or compilation is repeated merely to create this checkpoint. Actual
checkpoint checks are recorded in `checkpoint-checks.json`.
Native CSV line endings are retained. Existing empty PRM assignments and the
deal.II compatibility patch retain their original literal whitespace; full-diff
warnings for those inputs are recorded separately from source whitespace checks.
