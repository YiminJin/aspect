# Bounded P2 direct irreversibility

[Scientific report](../../../doc/reconstructed_fault/propagation/P2_irreversibility_report.md)
leads with the model change, completed trajectories and remaining accuracy limits.
The selected physical lower bound is `phi_last_accepted`; fresh initialization
still relaxes on [0,1]. The recoverable driver, unloaded fixture, surface Z,
bulk-conjugate work, consistent normalization and common a=.256 ramp are retained.

Production selector (default false):

```text
subsection Solver parameters
 subsection Phase field solver parameters
  set Use bound constraints = true
  set Enforce phase irreversibility = true
 end
end
```

The fixture uses `Nonlinear solver failure strategy = abort program`.
The new selector rejects the native failed-iterate continuation mode. Physical
rejected-step rollback, coupled restart and transported/birth histories remain
outside this bounded qualification.

The plugin inherits existing P2 observation rather than duplicating mechanics.
With `--replay`, production remains box-only until the first retained healing
entry. The callback solves the same incoming H with an immutable accepted lower
bound, records full fields/KKT/fracture/I/source, probes failed publication and
normal/exceptional restoration, and stops before mechanics/history advances.
Its exit 1 is qualified by the explicit completed-replay marker, not alone.
All old failures are preserved. Reproduction labels include `replay_guard` and
use the same final executable as the complete physical trajectories.

```sh
cmake --build build-propagation-integration --target aspect -j3
cmake --build build-propagation-integration/plugin --target irreversibility_feedback -j2
python3 benchmarks/reconstructed_fault/propagation/run_irreversibility.py \
  irreversibility_replay_guard_dt125 --replay --dt .125 --work --amplitude .256
python3 benchmarks/reconstructed_fault/propagation/run_irreversibility.py \
  irreversibility_dt125 --dt .125 --work --amplitude .256
```

Corresponding `dt0625` and `dt03125` labels use .0625 and .03125. The runner
refuses existing output/log paths. All fresh loaded paths end at t=1; material,
mesh, support and loading are identical. Runtime cap is 240 s per serial run;
final replays take 67–167 s and fresh loaded runs 69–119 s in the bounded local
concurrent batch. The zero-load control takes 18 s. No cluster work.

Read-only analysis:

```sh
python3 benchmarks/reconstructed_fault/propagation/analyze_irreversibility_replays.py
python3 benchmarks/reconstructed_fault/propagation/analyze_irreversibility.py \
  irreversibility_dt125 irreversibility_dt0625 irreversibility_dt03125 --name comparison
python3 benchmarks/reconstructed_fault/propagation/finish_irreversibility_analysis.py
python3 benchmarks/reconstructed_fault/propagation/match_irreversibility_work.py
python3 benchmarks/reconstructed_fault/propagation/check_irreversibility_evidence.py
# Fresh mature default/enabled regression used run_irreversibility_compat.py.
```

`irreversibility_evidence/` contains numerical JSON, plots, exact production/spec
revision diffs, invocation logs/hashes and the manifest. Native compact CSVs and
expanded parameters remain in `build-propagation-integration/output-<label>/`;
key replay and accepted fields are also copied into evidence. The immutable
incoming snapshot contains all 3,154 prior pinned artifacts.

The check separates completed solver/admissibility outcomes from temporal,
normalization and energetic qualification. It does not turn the original
box-only failures into passing physical trajectories. Common-time comparisons
use identical physical times; separately named lag interpretations use identical
consumed-history times. Common-slip values interpolate accepted records and are
not new mechanical solves. The source spatial error of 6.2–6.4% and signed
profile work are retained in all conclusions. No material optimization/P3.
