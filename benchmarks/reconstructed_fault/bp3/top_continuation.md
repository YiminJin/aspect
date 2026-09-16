# Bounded top-boundary experiment

This is a frozen, straight, mature, **uniform prescribed-Vp** experiment, not
the supported free-RSF top treatment. The previously qualified bottom pairing
remains enabled. No surface particle admission or endpoint connectivity changes.

`run_top_source.py` preserves the bottom control's mesh, fault, physical data,
clock, constraints, solver settings and history timing. `paired` adds the
outside-top Q1 integral to the same immutable preprojection completion table and
sets `ASPECT_BP3_TOP_SOURCE_EXPERIMENT=1`. The existing completion-file parameter
is reused; despite its historical `Bottom` name, the paired table contains both
ends. The script verifies the top virtual Q1 profile against saved physical FE
samples before preparing it. The experiment switch rejects non-uniform RSF use.

The extended coordinate is the last segment at xi=1. Assembly and particle
Maxwell updates therefore use the current endpoint field, not a new constant
bulk-only rate. The first accepted state's independent endpoint probe tests
the production B action against the absolute bulk-residual derivative and the
reference quadrature action. It does not change the accepted V or histories.

```sh
cmake --build build-pf-cpdi --target aspect.exe.release -j4
cmake --build benchmarks/reconstructed_fault/bp3/build --target bp3.release -j4
python3 benchmarks/reconstructed_fault/bp3/run_top_source.py control --prepare-only
python3 benchmarks/reconstructed_fault/bp3/run_top_source.py paired --prepare-only
python3 benchmarks/reconstructed_fault/bp3/run_top_source.py control --run-prepared
python3 benchmarks/reconstructed_fault/bp3/run_top_source.py paired --run-prepared
MPLCONFIGDIR=/tmp/top-source-mpl python3 benchmarks/reconstructed_fault/bp3/analyze_top_source.py
```

The two output directories are retained and must not be overwritten. Runs use
four ranks, fresh initialization and exactly two real steps, with a 600-s hard
cap each. `run.log`, `execution.json`, and `provenance.json` identify the actual
commands, hashes, runtime and solver checks. All additional observations are
separate CSVs; ordinary graphical quantities are unchanged.

The all-QP comparison includes inactive positive-phase points, not only those
already associated. `top_source_parents_*` records source-only parents and their
zero surface weight. The comparison's virtual-work diagnostic uses the smooth
velocity test `w = Vp*s*X*(1-X)*Y`, which is in Q2 and vanishes on the prescribed
lateral boundaries. This tests a missing weak contribution, not B=transpose(G):
the current particle/domain G has a distinct measure and normal/friction terms.
The free-endpoint coupling gate must be resolved before mature-RSF adoption.
