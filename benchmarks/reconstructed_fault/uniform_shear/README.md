# K1 uniform-shear pilot and independent reference

Current status (2026-09-09): Gate K1 passes under the reviewed K1-only
containment allowance. See
`doc/reconstructed_fault/benchmarking/stage_K1_verification.md` for the
completed five-case cross, raw stress errors, MPI/restart and regressions.
`nonuniform/` contains the bounded K2.1 prescribed-pressure pilot; its
containment gate is assessed independently and receives no K1 relaxation.

Residual-floor follow-up:
`doc/reconstructed_fault/benchmarking/stage_K1_residual_floor_review.md`
and `residual-floor/` contain the represented-increment diagnosis, local
gradient-evaluation correction, documented mixed bulk precision criterion,
and completed one-/two-rank dt=.5 trajectories through 6 s. The K1 restart
tail agrees bit-for-bit. The existing spatial/time cross and remaining
regressions now pass as documented above. Earlier failure records below
are historical.

Latest verification: the corrected pilot completes through 6 s; see
`doc/reconstructed_fault/benchmarking/stage_K1_coupled_corrections_review.md`.
The bounded spatial/timestep cross is recorded in
`doc/reconstructed_fault/benchmarking/stage_K1_convergence_review.md`, with
case inputs/results under `convergence/`. It retains the reviewed K1-only
1e-4 containment allowance and the unchanged 1e-4 actual slip-normalization
requirement. Earlier failure records below are historical, not current status.
K2 remains gated on completion of the outstanding K1 checks.
The pre-correction convergence attempt stopped at dt=0.5, t=4.5 s,
and the coupled solver was confirmed to omit configured volume-pressure
normalization. The approved correction and focused verification are recorded in
`doc/reconstructed_fault/benchmarking/stage_K1_pressure_normalization_review.md`,
with artifacts under `pressure-normalization/`. This prerequisite correction
passes its focused gauge tests, but the unchanged dt=.5 reproducer still fails
at 4.5 s. It does not complete the remaining convergence checks. No K2 run has begun.
The bounded replay in `linearization-diagnosis/` identifies false convergence
of the condensed linear solve; see
`doc/reconstructed_fault/benchmarking/stage_K1_linearization_diagnosis.md`
for true residuals, directional checks and the correction proposal awaiting review.

Historical follow-up: `doc/reconstructed_fault/benchmarking/stage_K1_support_resolution.md`
records the authorized corrected pilot and bounded resolution proposal. Corrected
t=0 mechanics is accepted, but the first real timestep fails its line search.
The unchanged 1e-6 containment gate still fails. The proposed K1-only budget
revision is **not applied**, and K2 has not started. Accepted initial ParaView
data are in `diagnostics/mechanical-pilot-view/`; prospective scalar predictions
are explicitly separated in `diagnostics/support-resolution/`.

Earlier pre-pilot gate status: **K1 is not satisfied; K2.1 has not started.** The corrected
initialization removes the CPDI/symmetry defect and passes the independent I_h
check, but its association strip omits 5.64918e-5 of the in-domain profile
integral, above the existing 1e-6 target. Corrected accepted mechanics and the
K1 convergence matrix remain unmeasured. See
`doc/reconstructed_fault/benchmarking/stage_K1_gate_review.md`.

Reproduce the missing initialization checks from the saved corrected packet:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 benchmarks/reconstructed_fault/uniform_shear/audit_gate_k1.py
```

Exit 2 denotes the unsatisfied gate, not a Python failure. Results are in
`diagnostics/gate-k1/`; existing corrected ParaView files/profiles are in
`diagnostics/results-corrected/`. No accepted mechanics are present there.
The following pilot status and commands describe the earlier, pre-ownership
correction run, not a new validated trajectory.

Status: the approved initial previous-profile correction is implemented and
its focused lifecycle/boundary tests pass. The one-rank pilot accepted t=0
and t=2 s, but hit its 600 s wall-clock cap during t=4 s. Its measured geometry,
containment and along-fault uniformity fail the benchmark assumptions. This
is **not a K1 verification pass**. See
`doc/reconstructed_fault/benchmarking/stage_K1_pilot_report.md`.

Run from the repository root, using Python with SciPy:

```sh
python3 -m unittest discover -s benchmarks/reconstructed_fault/uniform_shear -p test_reference.py -v
python3 benchmarks/reconstructed_fault/uniform_shear/reference.py
```

The second command uses the approved initialization rule and finds an
interior root without physically evolving the retained initial histories.

For comparison only:

```sh
python3 benchmarks/reconstructed_fault/uniform_shear/reference.py --previous-profile zero
```

This historical-defect reproducer exits **2**: a zero previous phase profile
gives no interior root. It is not clipped to the lower bound. Both commands
use an independently integrated stationary AT1 profile, not the measured
production Q1/CPDI profile or projected initial cohesive state.

`solve_step()` separates evaluated stress/cohesive responses from retained
history at timestep zero. For real steps it advances its own stress, cohesive
traction and ageing-law state. Its caller must initialize histories once,
then pass the returned history forward using the actual accepted timestep
and loading sequence. Later ASPECT histories must never reset the reference.

The production pilot uses the normal reconstruction activation 0.1 and the
converged initial Q1 phase field. The plugin then constrains only independent
phase DoFs, preserving periodic relations and leaving mechanics unconstrained.
`Evolve phase field=false` retains its existing meaning (freeze H); it is not
used as a substitute for these benchmark-only phase constraints.

Build and run (repository-root commands; adjust the absolute checkout path):

```sh
cmake --build build-pf-cpdi --target aspect -j4
cmake -S benchmarks/reconstructed_fault/uniform_shear -B /tmp/aspect-k1-pilot \
  -DAspect_DIR=/home/ein/repository/aspect/build-pf-cpdi -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/aspect-k1-pilot -j4
cd /tmp/aspect-k1-pilot
python3 /home/ein/repository/aspect/benchmarks/reconstructed_fault/uniform_shear/run_pilot.py \
  /home/ein/repository/aspect/build-pf-cpdi/aspect \
  /home/ein/repository/aspect/benchmarks/reconstructed_fault/uniform_shear/pilot.prm
```

The runner limits this single process to 600 s, reports Linux child peak RSS
in KiB and elapsed time, and returns 124 on timeout. Preserve existing output
before rerunning. Do not increase the cap to hide a failed feasibility gate.

Analyze completed accepted states, from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 benchmarks/reconstructed_fault/uniform_shear/analyze.py \
  /tmp/aspect-k1-pilot/output-pilot
```

The CSV files contain accepted times/loading, Q1 phase support values, fault
geometry/histories, actual segment support, particle histories/volumes and
Stokes-quadrature velocities/gradients/constitutive diagnostics. The analyzer
integrates actual Q1 normal profiles independently, reproduces the consistent
surface projections, measures assumptions and advances its own scalar history.
It returns 2 for an incomplete run or failed assumption checks. Conditional
averages are reported for diagnosis, never promoted to a trajectory pass.

No full convergence matrix or K2 case is supplied here. Pilot data are not
automatically added to the standard ASPECT integration suite.
