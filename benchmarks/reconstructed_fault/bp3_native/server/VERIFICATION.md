# Server-package verification — 2026-10-14

| Check actually performed locally | Result |
|---|---|
| Complete ASPECT snapshot, including current tracked edits and required untracked BP3 files | Copied with a per-file hash inventory and original symlinks |
| Configure that snapshot with bundled World Builder, deal.II 9.6.2 and Voro++ 0.4.6 | Pass, Release, GCC 12.4; no full ASPECT rebuild |
| Build the delivered ordinary BP3 plugin against the retained matching local ASPECT build | Pass; GNU 16.2.1 MPI wrapper; no local-test definitions |
| CMake-generated plugin/build/aspect-release link | Resolves to the matching Release executable |
| Standalone input in a fresh directory, through that relative executable/library layout | --validate passes; --output-json succeeds |
| Resolved input | Fine level 9/minimum 1, surface 10, q6, filter 20/full bottom, native FE/mature/frozen/stationary, no births, .2 state limiter, first/max 4e6/4e7, no checkpoint/resume/cutback |
| Environment file | bash/zsh syntax passes; diagnostic selectors cleared, dependency paths retained, threads=1 |
| Package input data and plugin C++/headers | Match maintained source hashes; only the plugin CMake floating-point flags differ |

The first source-copy attempt stopped at a directory symlink. Its incomplete
configure failed on missing World Builder files. The corrected complete copy
preserves all links and configures successfully. No source algorithm or
dependency default was changed to obtain that result.

The JSON query announces its loaded library on stdout before the JSON object;
the README's `sed` command removes that prefix. Parameter-query tests do not
initialize the mesh, particles, FE history or mechanics. No output directory
was created by these checks.

Existing physical evidence is retained, not rerun: the section-8 coarse full
geometry accepted step zero and two physical steps in 696.36 s, with native
Maxwell/state and once-only publication checks. Its active state limiter chose
1772.48/2166.15 s. The refined local filtered/raw comparison and profile-only
halving remain described in `section8_report.md` in the source snapshot.

Not performed: a fresh full ASPECT compilation, Intel/server build, fine full
mesh initialization, long first-event evolution or server MPI/resource test.
No server connection, job submission, mechanical trajectory, checkpoint or
automatic retry was launched for this packaging task.
