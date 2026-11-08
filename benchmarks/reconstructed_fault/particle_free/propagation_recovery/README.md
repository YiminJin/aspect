# Mature closure and propagation recovery

Current findings and selective retest plan:
[PROPAGATION_FORMULATION_RESTART.md](../../../../doc/reconstructed_fault/particle_free/PROPAGATION_FORMULATION_RESTART.md).
The exact short-check launches are recorded under
`benchmarks/reconstructed_fault/refactoring_boundary/evidence/propagation-recovery-*.json`,
and each cohesive run has `execution.json` and its resolved input under
`build-particle-free/propagation-recovery/runs/`. None of these generated records
or checkpoints belongs in a source commit. Original outputs are never overwritten.

The current numerical core was reused. Only its benchmark plugin was built:

```sh
cmake -S benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native \
  -B build-particle-free/propagation-recovery/checks \
  -DAspect_DIR="$PWD/build-particle-free/native-phase-integration/native" \
  -DCMAKE_CXX_COMPILER=/opt/gcc/12.4.0/bin/g++
cmake --build build-particle-free/propagation-recovery/checks -j2
```

Use the existing `cohesive_rsf_native/qualify.py` for new labels. Current-source
checks select `active`, `--history-records`, two ranks, the native-phase-integration
core and newly matched plugin. Historical episode checks instead select the
preserved `objective-stress/baseline/1-aspect-release` and
`2-libcohesive_rsf_native_checks.release.so`, four ranks, and copy the original
`cohesive-rsf-instability/runs/episode-local/output/restart`. Do not mix these
identities or resume that checkpoint into a newly selected stress law.

## Prepared next experiment — not launched

Hypothesis: the retained finite-wall, radiation-damped small-strain law can
complete its first RSF acceleration/decay episode while resolving whether the
adjacent cohesive-zone damage becomes fully decohered material. Acceleration,
partial damage, a=1 front advance and new persistent geometry are separate
observables. This does not predict a resolved wave or physical rupture speed.

Start from the accepted step-752 checkpoint produced by the four-step recovery.
The control/reference is the preserved step-748 checkpoint and its exact
zero-advance export, the earlier neutral-state-feedback control, and the
short continuation measured in this recovery. Keep G=1e6 Pa, eta=1e12 Pa s,
Gc=3 J/m², peak=2000 Pa, confinement=10000 Pa, RSF a=.005/b=.02,
mu0=.1, V0=1e−6 m/s, Dc=6e−5 m, radiation=25836.0214 Pa s/m,
R=.75 m, lifting width=.75 m, 128×96 bulk cells and .0625 m surface spacing.
Retain the half-event increment settings and residual-resolved history budget
5e−4. No phase fracture charge, new spin, loading change or surface reset.

After selection, run from the repository root with one numerical thread/rank:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DEAL_II_NUM_THREADS=1
python3 benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native/qualify.py active \
  --label selected-episode-01 \
  --work-dir build-particle-free/propagation-recovery \
  --input benchmarks/reconstructed_fault/particle_free/propagation_recovery/next-episode.prm \
  --aspect build-particle-free/objective-stress/baseline/1-aspect-release \
  --plugin build-particle-free/objective-stress/baseline/2-libcohesive_rsf_native_checks.release.so \
  --resume-from build-particle-free/propagation-recovery/runs/episode-short/output \
  --mpiexec /opt/openmpi/5.0.6/bin/mpiexec --mpi-args='--bind-to none' \
  --ranks 4 --timeout 5400 --graceful-stop-after 5100
```

The native wall limit is 1.4 hours; the runner margins allow accepted-state
output/checkpointing. Stop at the first of: a violated unchanged physical or
numerical guard; the reviewed aggregate budget (at most four such segments,
six wall hours); the explicit physical horizon t=374.68003334330067; or the
scientific decay diagnostic below. A wall/horizon stop is not event completion.
Any continuation uses a new label and `--resume-from` the last completed segment,
never restarts step 748 repeatedly or overwrites it. Before any later segment,
read the current diagnostic and decide whether further continuation is justified.
No server submission is included in this command or present task.

Scientific diagnostic: after exceeding 1e−3 m/s, global speed must remain below
10% of the observed maximum for max(.005 s, 5 Dc/V_peak). Preserve the historical
peak across segments. This is an **analysis diagnostic**, not an implemented
native termination plugin. The supervising task should request `terminate-aspect`
on a completed accepted state when it is met, and verify the terminal record.
The existing `cohesive_rsf_instability/event_window.py` combines one `--run`
with a preceding `--prefix` gzip/history pair, preserving the prior peak:

```sh
python3 benchmarks/reconstructed_fault/particle_free/cohesive_rsf_instability/event_window.py \
  --run build-particle-free/propagation-recovery/runs/selected-episode-01/output \
  --prefix benchmarks/reconstructed_fault/particle_free/evidence/propagation_recovery/episode/event-gauss.csv.gz \
  --output build-particle-free/propagation-recovery/selected-episode-01-analysis
```

For a later segment, use the preceding analysis's `event-gauss.csv.gz` and its
sibling `event-history.csv` as the prefix.

Export signed/absolute slip and age, partial-damage integrals, bracketed a=.01/.1/
.5/1 fronts, new persistent material identities, temporary candidate length,
full stress and current pressure/normal traction. Retain signed and absolute
physical work errors versus both sliding loss and new fracture work; separate
BE bulk, surface endpoint, projection/repartition and exact ledger errors.
Keep endpoint/contact/weak/source guards unchanged. A successful decay does not
erase the existing 1% fracture-relative or temporal/spatial accuracy failures.
Use matched-time and progress-aligned views separately; do not qualify onset
timing from a shared-history replay.

Historical warm throughput was 8.11 s per accepted step, about 5.9 local hours
per .1 physical seconds. Setup/restart has extra cost; new recovery timing is
reported separately. The .03184 s radiative low-mode scale suggests order .07 s
for a tenfold decay only if geometry/stiffness remain comparable. This is a
budget estimate, not a guarantee of completion. Do not spend more time or change
the law merely to force a completed episode.
