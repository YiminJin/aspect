# Local context-probe verification, September 26

`before/` ran the new three-case context fixture against the preserved core
executable from the previous diagnostic task. `after/` ran the same plugin
against the rebuilt core containing the four-stage diagonal observer. Both
runners exited zero and all three cases passed. `cmp` found their full
`coarse_diagonal_probe.txt` files byte-identical (96 lines).

The new stderr records all four stages for all three cases. Owned diagonal
entries start at zero; both free entries finish at the analytic reciprocal.
This is GCC 12.4/OpenMPI/deal.II 9.6.2 with four SIMD lanes, not an Intel server
reproduction. Only the new operator fixtures were run; no coupled Stage-I or
BP3/BP5 trajectory was replayed.

| Artifact | SHA256 |
|---|---|
| Core before | `eef974905d3cca3063c5ae3937ca6cdd27bda3d8d3a5a6ba7dfa2fb199ac6743` |
| Core after | `25552f3528d95031f405096ee3771da4e2780c38eb84d03506a0e01c918e3ece` |
| Context plugin | `6770cb1b729ff010362afab18bdff6403b4c7b039b48e2df3bcaf36b6f903366` |

Build logs, caches, plugin compile commands, run logs, provenance and statuses
are adjacent. The 29 files listed in `server-inputs.sha256` were verified
unchanged after this task, including the copied server outputs, archives and
build/environment records. Shell syntax and whitespace checks passed. No
Python scripts were used.
