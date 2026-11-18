# Overnight free-turn validation

Selected scope: `CODEX_OVERNIGHT_FREE_TURN_VALIDATION.txt`. The single, persisted
`evidence/overnight_free_turn/budget.json` fixes the deadline and reporting reserve.
Do not reset it when resuming. `run.py` bounds each selected launch by that cutoff,
uses four bound physical cores and one thread/rank, checkpoints every accepted
interval, and losslessly compresses closed exports after MPI exits. It refuses
an existing output label. All cases use the protected physical executable and
checker; there is no C++ rebuild or solver modification.

A uses the retained material-strip parent15/slot01 at .075 and .00125 s midpoint.
B uses `fixed-offaxis.prm`, a separate fixed-direction history with the complete
positive off-axis velocity. C uses `fine-normal.prm`, retaining128 x cells and
changing only normal cells128→256. Restart never changes the closure/mesh/law.
The executable and actual input hashes are in each run's command manifest;
current source hashes do not identify a retained executable.

Representative launches, through the existing `refactoring_boundary/run_logged.py`
wrapper (its timeout includes a short allowance for post-MPI compression):

```sh
python3 benchmarks/reconstructed_fault/particle_free/overnight_free_turn/run.py a --seconds 5400
python3 benchmarks/reconstructed_fault/particle_free/overnight_free_turn/run.py b-prefix --seconds 4200
python3 benchmarks/reconstructed_fault/particle_free/overnight_free_turn/run.py b-half --seconds 2100
python3 benchmarks/reconstructed_fault/particle_free/overnight_free_turn/run.py c-pilot --seconds 10000 --end-step 15 --point-stride 9999
```

Actual labels, timeouts, checkpoint parents, environment and partial/completed
endpoints are retained in launch/compression manifests. The nominal timeouts in
this README are examples; the common remaining budget always governs.
The fine pilot was handed off after its complete14/.070 checkpoint to avoid
interrupting a long15/.075 solve at its earlier launch cap. The guarded
`handoff_fine_prefix.py` records and stops only its four named native ranks;
`C-boundary-handoff.json` distinguishes this planned stop from solver failure.
Its ordinary `c-prefix` continuation keeps dt=.005 and copies only that own-grid
accepted parent. The common deadline is unchanged. Actual continuation command:

```sh
python3 benchmarks/reconstructed_fault/particle_free/overnight_free_turn/run.py c-prefix --seconds 8000 --resume-from build-particle-free/bulk-tip/runs/overnight-free-turn-c-pilot/output --snapshot-id 3 --point-stride 9999
```

Closeout: A accepts two quarter steps through .0775 before its wall limit; B
completes .080; C completes .0775 and its pending .080 is interrupted by the
common wall limit. The last good fine checkpoint is `c-half/output/restart/02`
under the bulk-tip runs directory. No job remains. Contour steering survives
the completed comparisons, while first registration shifts for 12 sections that
subsequently carry slip. The report separates this history sensitivity from
the close phase contours and energy totals.

These commands document the completed/selected execution; existing labels and
the persisted deadline are protected. Reproducing a numerical run later needs
an explicitly selected new task/budget and fresh labels. Postprocessing the
retained evidence needs neither a budget reset nor another simulation.

`observations.py` reuses the retained native Bernstein, fracture and contour
utilities. Exact vertical roots give width and midpoint, with independent DG
face traces respected. A contour piece is reported connected only when an exact
horizontal superlevel interval reaches an interior point of the original seed.
This is a sufficient connection check, not a graph continuation assumption;
multiple/unverified pieces are retained explicitly. Common x stations are cell
interiors on both grids, including fixed1/2 mm cuts beyond6 for an early subcell
advance and the usual6.03125+n*.0625 cuts. Dense contour observations do not
establish spatial resolution. Local native-Q2 root refinement measures the foremost
absolute contour, separately from structural-plane angles or tiny registration.
Native quadrature integrates field errors; a common subdivision preserves both
meshes' separate polynomials for C. Rates use accepted slip increments over the
same physical interval; endpoint stage rates at different stage times are not
silently compared.

Run analyses only on closed exports after the corresponding MPI process and
compression finish. `compare_a.py` compares common .0775/.080 endpoints and phase
differences relative to the increment since the shared .075 parent. Both paths
inherit the earlier .005 history.

`compare_bc.py b` compares the completed horizontal-plane control with the
retained free-plane history. `check_b_reduction.py` tests the discrete Maxwell
background prediction against the straight control **only through.075**, where
the timestep histories match. `compare_bc.py c` uses the fine run's last accepted
common time; it never substitutes an unaccepted trial or extrapolates a partial
history to.080. `registrations.py` takes an explicit common `--end-time` for each
grid; `compare_registrations.py` preserves unpaired registrations and relates
first formation to eventual slip without any phase/rate cutoff. `integrity.py`
checks bounds, accepted irreversibility, original formed-plane lifetime and
virgin age. `visualize.py` makes the equal-axis native contour overlay,
centerline/width figure and CSV comparison table after those analyses finish.
The C comparison also reports endpoint differences normalized by the coarse
.075–.0775 increment, and differences between the increments after subtracting
each path's own .075 field. These retain inherited spatial differences and
their cancellation explicitly, rather than comparing a seed-normalized number
directly with A's late-window norm.
Registration times mean the first **accepted export** with a formed plane,
not a resolved physical nucleation time. The accompanying phase maximum uses
the six normal Gauss points per cell at that section, matching its integration
sample locations; it is not an interpolated contour threshold or a global peak.

Disk preservation records are `repacked-exports.json` (identical decompressed
bytes at existing gzip paths) and `cold-export-compression.json` (raw paths
archived as gzip, with exact restoration commands and SHA256). The old64 broken
CSV aliases listed at entry remain unavailable; this task neither discards nor
claims to restore their historical data. Never rerun a one-time archiver over
its canonical manifest without preserving the prior record.

Final observation commands (one numerical thread per process; at most four
independent postprocessing processes, after MPI exit):

```sh
python3 benchmarks/reconstructed_fault/particle_free/overnight_free_turn/compare_a.py
python3 benchmarks/reconstructed_fault/particle_free/overnight_free_turn/compare_bc.py b
python3 benchmarks/reconstructed_fault/particle_free/overnight_free_turn/compare_bc.py c
python3 benchmarks/reconstructed_fault/particle_free/overnight_free_turn/registrations.py first-turn-positive-prefix first-turn-positive-end first-turn-positive-fourcore-end recover-free-turn-positive-half --hy .03125 --name coarse --end-time .0775
python3 benchmarks/reconstructed_fault/particle_free/overnight_free_turn/registrations.py overnight-free-turn-c-pilot overnight-free-turn-c-prefix overnight-free-turn-c-half --hy .015625 --name fine --end-time .0775
python3 benchmarks/reconstructed_fault/particle_free/overnight_free_turn/compare_registrations.py --time .0775
python3 benchmarks/reconstructed_fault/particle_free/overnight_free_turn/verify_observations.py
python3 benchmarks/reconstructed_fault/particle_free/overnight_free_turn/verify_restores.py
python3 benchmarks/reconstructed_fault/particle_free/overnight_free_turn/visualize.py
```

`integrity.py` is run separately on A, B prefix/half, and C pilot/prefix/half
(the latter with `--hy .015625`), writing `A/B/C-integrity.json`.
`closeout.py` checks the unchanged incoming source, original failure and binary
identities, hashes inputs/logs/checkpoints, and verifies copied parent bytes.
The existing coarse-only .080 follow-up in the registration JSON is labeled
separately from the common .0775 comparison; no fine value is extrapolated.
