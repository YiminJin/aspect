# Stage 4 loaded-seed preparation and conditional growth comparison

Completed 14 November 2026 within the original eight-hour allocation. The
authoritative outcome is in
[the overnight report](../../../../doc/reconstructed_fault/particle_free/STAGE4_OVERNIGHT_PREPARATION_REPORT.md).
Both fixed-phase histories reach 2.56 ms, both free histories reach 6.08 ms,
and both local half-step histories cover 4.8..5.44 ms (sixteen intervals each,
stopped normally at their wall caps). No job is active and this package does
not authorize repeating or extending those numerical cases.

The selected overnight task starts from the retained near-tip rate profile,
gamma_e=.000365 and 256×128 native mesh. At held displacement it minimizes the
native fracture functional plus mechanically equilibrated elastic/cohesive
storage and prescribed section-force work. The inherited phase is its lower
obstacle and one its upper obstacle. Native fracture assembly, the native
distributed mechanical operator/Stokes inverse and the existing Bernstein
obstacle solver supply the work. No artificial slip, aging or Maxwell time is
accumulated during preparation. A separate checker preserves all old runtimes.

The section target resistance is evaluated at the prescribed unperturbed
V0(x)/theta0(x). The matched b and mu choices make it common to both laws.
After preparation the same smooth 1% age perturbation is applied and each owned
law is inverted at that common force. Full bulk stress and cohesive strain are
equilibrated with phase; section age, rate target, zero cumulative slip and
stored directions are prescribed. Physical stepping retains the existing
midpoint equations and criteria, with free native virgin-plane selection.

For displacement u, plane S:S=1/2 and H=m phi(1+phi)/[m phi(1+phi)+(1-phi)^2],
the static eliminated cohesive scalar is

    e = H (2G dev(sym grad u):S - f) / G,
    tau = 2G dev(sym grad u) - 2G e S.

The regular reaction is q=(1-H)(2G E:S-f), with cohesive energy e q/2.
It gives e=0, finite q and zero cohesive energy at exactly intact phase;
the expression G e²/(2C) uses that continuous extension when C=0.

The minimized functional is integral [tau:tau/(4G) + G e²/(2 C(phi)) + f e]
plus the native full AT1/SIPG fracture energy, with C=H/(1-H). Its phase
derivative is Gamma' - integral N_i H' (2G E:S-f)²/(2G). The implementation
uses the native point response with u/beta, so its predictor equals 2G E;
beta cancels exactly and the algebraic stage duration advances no physical
clock. Existing physical midpoint update is separate from this static problem.

`prepare.py` builds the independent checker. `run.py --settle --input
input-inherited --perturbation 0` performs native held-displacement preparation.
Its final coefficients are retained as the exact common input for subsequent
fresh law runs, which verify physical phase KKT before applying the perturbation.
`static_equilibrium.h` supplies only the benchmark preparation; it does not add
a production solver or change the physical acceptance tolerances. Iteration
logs distinguish preparation from physical timesteps. At most one purposeful
specimen revision is allowed by the selected task; no preload sweep is selected.

The run manifest and report identify the tested executable independently of HEAD.
Do not rebuild over a protected executable or rerun preparation into an occupied
input directory. Exact frozen checker sources, commands and inputs accompany the
retained evidence; native output directories and checkpoint payloads remain under
`build-particle-free/bulk-tip/runs/stage4-preparation-*`.

`batch.py FAMILY` executes only `stage4-preparation/FAMILY-cases.json` sequentially,
with four ranks per case. It starts a durable supervisor and returns. An existing
ASPECT job prevents launch. Each completed case losslessly compresses its own
dense observations and verifies the decompressed SHA256 before unlinking the
uncompressed duplicate. Checkpoints are untouched. Both native wall-time and
wrapper limits apply; `run.py` also reserves ten minutes before the task deadline.

After a batch exits, reduce the pair with `analyze.py FAMILY` and, for released
phase, `analyze_growth.py FAMILY`. These distinguish midpoint rates/driving from
endpoint slip/phase, compare native initial tau/A, retain positive and negative
deformation, and check the physical work error before numerical corrections.
The optional dense margin cache belongs in the build tree; compact CSV/JSON and
PNG/PDF figures go to `evidence/stage4_preparation/`.

`compose.py NEW_FAMILY SOURCE_FAMILY ...` joins successive native restart runs
into a labelled analysis view under `stage4-preparation/analysis_views/`. It
checks accepted-step continuity, preserves original commands and frames, and
writes source-command hashes. Use `--view VIEW_PATH` on both analysis tools.
This view is not a numerical run or restart checkpoint. `compare_time.py` compares
a local half-step pair with that view at equal physical endpoints, averages two
fine midpoint rates over each coarse interval, and checks restored native phase.
It explicitly retains the shared coarse prefix as an accuracy limitation.

Terminal continuation uses the retained benchmark scheduling hook only when the
saved pending interval is clipped to roundoff. Accepted material time/fields and
last physical timestep remain unchanged. Ordinary periodic parents use the
existing restart-timestep callback instead. Read-only restore dumps verify
physical phase and section endpoint memory before any new trial.

`compress_closed_frames.py LABEL --through-step N` is optional disk maintenance
for explicitly listed cases of this task. During an active run,N must be at
least two accepted steps behind the current state. It uses low process priority,
compresses only closed diagnostic CSVs, verifies every decompressed SHA256, and
keeps `early-compression.json` alongside the batch's final compression manifest.
It changes no checkpoints or numerical state. Both manifests count toward the
retention and cost records.

`audit_restore.py` checks the fine parent after the numerical pair exits. It
reconstructs the input's actual6x6 Gauss rule and affine cell measures, compares
positive/negative volume rates with the native section observations, verifies
phase and section endpoint memory, and hashes the copied parent checkpoint.
It preserves the distinction between endpoint stress/A and stored last-stage
rate/reaction. The signed local force-work contributions are not a pointwise
friction law.

The final analysis uses `free76` (native `free` plus `extendplain`) and `half`
(native `half` only). Native accepted step numbers after restart are not equal
physical times between coarse and fine runs. These are the final commands;
run from the repository root with BLAS/OpenMP threads limited to one:

```sh
python3 benchmarks/reconstructed_fault/particle_free/stage4_preparation/analyze.py free76 --view build-particle-free/stage4-preparation/analysis_views/free76
python3 benchmarks/reconstructed_fault/particle_free/stage4_preparation/analyze_growth.py free76 --view build-particle-free/stage4-preparation/analysis_views/free76 --reference free76
python3 benchmarks/reconstructed_fault/particle_free/stage4_preparation/compare_time.py --coarse-view build-particle-free/stage4-preparation/analysis_views/free76 --fine-view build-particle-free/stage4-preparation/analysis_views/half --parent-step 60
python3 benchmarks/reconstructed_fault/particle_free/stage4_preparation/audit_restore.py --coarse-view build-particle-free/stage4-preparation/analysis_views/free76 --parent-step 60
python3 benchmarks/reconstructed_fault/particle_free/stage4_preparation/analyze_growth.py half --view build-particle-free/stage4-preparation/analysis_views/half --reference free76 --reference-root build-particle-free/stage4-preparation/analysis_views/free76
```

The exact executed argv are retained in `evidence/stage4_preparation/reduction-commands.json`.
Do not run more than four reductions at once. `cost.py` reports actual
completed intervals, failures, numerical time and compression. `retain.py`
verifies incoming manual source and protected binary hashes, checks runtime
libraries, copies compact inputs/scalars/frozen checker source, and records
checkpoint identities. It leaves all raw observations and checkpoint payloads
in the native run directories. This checker is evidence for its frozen runtime;
it is not a build/test claim for all current manual production changes.
