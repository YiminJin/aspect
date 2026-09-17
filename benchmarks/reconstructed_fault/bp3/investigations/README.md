# Retired K5 investigation code

These files preserve the frozen-cohesion, functional/candidate-Theta,
independent-trace and noncommitting work/junction comparisons. They are **not
included by the maintained plugin**. Their old environment selectors are
rejected by that plugin; they are not research model choices.

The core experimental overrides and the forced noncommitting solver stop have
also been removed. Do not include these headers against the cleaned core and
assume the old experiment has been reproduced. Use an isolated checkout of the
recorded HEAD plus the pre-cleanup snapshot when historical reproduction is
needed. The historical drivers/analyses and saved outputs remain in this
workspace; this cleanup does not delete their evidence.

`../cleanup-checkpoint-20260916/` contains:

- `manifest.json`: original HEAD, archive contents and SHA256 of every file in
  the successful `fully-frictional-seven-local4/` evidence directory;
- `source.patch`, `index.patch`: unstaged/staged pre-cleanup changes;
- `source-and-binaries.tar.gz`: changed source, benchmark source/scripts and
  reports, plus the tested executable and plugin;
- `core-diagnostic-helpers.tar.gz`: the two retired file-local constitutive
  diagnostic headers, at their original paths.

Restore only in a separate checkout/worktree; do not overwrite the present
research configuration or unrelated working changes. The preserved original
run also has its own `provenance.json` and `source.patch`. The archive is a
recoverable local snapshot, not a substitute for a future reviewed commit.

The maintained `../work_checks.h` keeps the qualified mature-law K/G finite
differences; the exact aging audit and stable-ID history checks remain in
normal accepted-state verification. The general surface direct solver and
its mathematical tests are retained, including indefinite-block support.
