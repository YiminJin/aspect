# Preserved pre-cleanup research implementation

This source/header snapshot is the implementation before the clean 300-km
long-run plugin. It retains the constrained/reference, uniform-sliding,
saved-clock and diagnostic selectors for reproducibility, not as defaults of
the maintained `../bp3.cc`.

Existing results/checkpoints remain in their original directories. The copied
PRM and launcher are provenance snapshots, not relocatable launchers.
The maintained research launcher selects an explicitly separate library:

```sh
cmake -S benchmarks/reconstructed_fault/bp3 -B benchmarks/reconstructed_fault/bp3/build -DBP3_BUILD_RESEARCH_REFERENCE=ON
cmake --build benchmarks/reconstructed_fault/bp3/build --target bp3_research -j4
```

`run_research.py` uses `libbp3_research.release.so`; it never feeds the old
selectors to the clean long-run library. The reference target compiles, but
the historical simulation campaign was not repeated in the cleanup task.
The standalone Airy and replay-stop tests use these retained diagnostic
headers. Airy prestress remains excluded from production.
