# Reconstructed-fault reading guide

Start with the [current design](current_design.md) and [specification](specification.tex) for model ownership, history and acceptance. The [refactoring guide](refactoring.md) covers code conventions.

| Question | Starting point |
|---|---|
| How does the mature moving-fault model step? | [Mature fault algorithm](particle_free/MATURE_MOVING_FAULT.md) and [native phase discretization](particle_free/NATIVE_DG_BERNSTEIN.md) |
| How do phase, sections and RSF interact at a tip? | [Tip treatment](particle_free/TIP_TREATMENT.md), [localized direction](particle_free/LOCALIZED_TIP_DIRECTION.md) and [midpoint update](particle_free/MIDPOINT_UPDATE_CLOSEOUT.md) |
| Which numerical comparisons and limitations survive? | [Selected historical evidence](particle_free/README.md) and [particle-free status](particle_free/STATUS.md) |
| Where are the paper examples and older propagation baseline? | [Benchmark index](../../benchmarks/reconstructed_fault/README.md), [BP3 first-event review](particle_free/BP3_REVIEW.md) and [P2 closeout](propagation/P2_CLOSEOUT.md) |
| Where are replayable runtimes, checkpoints and archived inputs? | [Worktree cleanup and recovery](WORKTREE_CLEANUP.md) |

The earlier [project status](CURRENT_STATUS.md) and [cleanup record](../../benchmarks/reconstructed_fault/CLEANUP.md) retain the provenance of the original reconstructed-fault work.
