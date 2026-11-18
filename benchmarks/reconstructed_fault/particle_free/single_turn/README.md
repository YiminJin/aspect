# Single-turn constitutive reference

See [formulation](../../../../doc/reconstructed_fault/particle_free/SINGLE_TURN_RSF_FORMULATION.md)
and [results](../../../../doc/reconstructed_fault/particle_free/SINGLE_TURN_RSF_REPORT.md).
This is a fixed-chart material/section calculation with prescribed strain and
phase. It is not a simulator timestep driver or a free-phase turning solver.

From the repository root:

```sh
cmake -S benchmarks/reconstructed_fault/particle_free/single_turn -B build-particle-free/single-turn-reference -DCMAKE_BUILD_TYPE=Release
cmake --build build-particle-free/single-turn-reference --target tip_point_oracle -j1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/single_turn/check.py --oracle build-particle-free/single-turn-reference/tip_point_oracle --output benchmarks/reconstructed_fault/particle_free/evidence/single_turn
MPLCONFIGDIR=/tmp/single-turn-mpl OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/single_turn/plot.py benchmarks/reconstructed_fault/particle_free/evidence/single_turn
```

`reference.py` contains immutable-parent point/section response, geometric
integration, aging and the candidate virgin-plane criterion. `check.py` owns
bounded paths and independent checks. `point_oracle.cc` calls the unchanged
native material source and reports its physical midpoint endpoint and ledger.
`plot.py` renders only measurements and explicitly prescribed geometry.

Dependencies: deal.II for the native oracle; NumPy, SciPy and Matplotlib for the
reference/figure. Check `checks.json`, not merely process completion. The retained
`failed_absolute_rate_root` output is intentionally a failed earlier reference,
not native solver evidence. Reproduction replaces current reference tables in
the requested output directory; choose another directory to retain a comparison.
