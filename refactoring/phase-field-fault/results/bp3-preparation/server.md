# Preparing the 60° DG-stress donor on Stampede3

## Physical parent and executable

The shortest continuation starts from the snapshot written by Stampede3 job
3616642 at accepted step **4229**, time **5172072500.6710749 s**. The job
completed eight more steps, the last at **5172072500.6796865 s**, and entered
4238 before Slurm killed it at 48 hours. There is no evidence of a checkpoint
at 4237. The application reported the write of slot 03, then successful
completion, at 4229. Only an inspection of the selected server files will
establish that they survived. Server access, replay and long allocations are
pending. The local copy has logs and sampled fields.

The run used the particle-free **donor's** optimized stationary mature DG stress
solver with frozen prescribed phase and 10 m fault spacing, a 150 × 50 km
graded box, 552,081 active cells, 19,498,460 DoFs, 5,776 fault nodes and
8,833,296 audited particles. The output identifies an ASPECT 3.1.0-pre
optimized AVX512 build with 64-bit deal.II 9.7.0, Trilinos 16.2.1, p4est 2.8.7
and World Builder 1.0.0, on two nodes and 160 MPI ranks. The stderr module list
records Intel 26.0, Intel MPI 21.18, Boost 1.86.0 and parallel HDF5 1.14.6.
The release executable loaded the ordinary restored-BP3 DSO and the coupled
checks DSO. The repaired work observer printed `guard_uses_matching_frame=1`
and a 1e−5 Pa bound. The local archive contains no SHA-256 for the actual
Stampede3 ELF files. Identify all three from the writer installation and any
retained server artifact manifest before treating a continuation as identical.

| Retained artifact | SHA-256 |
|---|---|
| Original input copied by job 3616642, including the effective last overrides | `1c8c078ba8e949787cb5985b35ae1c286516b456bf018cfdbefe9477053c4660` |
| Ordered five-anchor fault.txt from the package, matching every anchor printed at startup | `5460e34da479ef16e7dcc95f62669250a891ee8c969df5064dd3e4320bea4a24` |
| Initial job's original.prm, in output_start | `2b23fadbd3dc5a49c8fda75f1f1ac0a59d79966654662eb81aab65000e1c78c9` |
| Job 3616642 stdout | `e064c08b03595a96a50198aa856b20d0d80dfde9d2f0e712875919c7f2962020` |
| Job 3616642 stderr | `0b7b6d34e5075c01d689facbaf8720c7481c5ddf96477b27ce854eab5b2a6c97` |

Jobs 3610007 and 3613912 both reached 815 and failed the original
work-consistency guard at 816. The latter replayed 810–815 from 809.
Job 3616642 used the matching-frame repair and restored 815. The older two
accepted labels are retained locally as metadata without matching native
payloads. The search included the neighboring ASPECT trees, build/install
directories and locally held server archives under the user's build, temporary
and Downloads directories. One earlier complete 160-rank BP3 snapshot
in the neighboring ASPECT tree holds step 42 at 35593207.990566462 s. Its
particle-stress run has 542,958 cells and 19,149,424 DoFs. It is a different
experiment. BP5 and the accelerated oscillation fixtures also use different
initial states or driving. All three candidate local *repaired small-mesh*
binary hashes are recorded in repair-local/work-repair-control/record.json.
None of the inspected installed BP3 DSOs matched the recorded small-mesh
repaired writer pair. Neither those hashes nor a workstation executable
identify the full-grid Intel binaries.

The immutable donor revision
`612b75d3570d81763407875f3a12c12999b9fa8e` contains the optimized source
and work-guard repair. All runtime source/fixture entries in the archived
35-entry manifest agree with this revision after accounting for the recorded
single-file repair. The two differences are evolving project records. The
post-repair work_audit.cc SHA-256 is
`dbf790e01a1940c5759cab96514d90bc0d53e6ee552ab1421afd3741a6dbc5a0`.
The archived continuation input differs from the pinned post-repair continuation
only in the graceful wall budget, changed from 82,800 to 172,800 s. The current
donor checkout and the later branch tip should not supply this build by default.

## Launcher and startup identity

The two-node launcher accepts either input overlay, both of which include the
original parameters and select the five-anchor file relative to the run directory.
For a new Slurm submission supply the usual site account, compatible partition and
walltime. For example, from the staged directory in the instructions below:

```bash
export BP3_RUN="$PWD"
sbatch -A "$ACCOUNT" -p "$PARTITION" -t 02:00:00 donor-allocation.sh donor-replay.prm
# After the eight-step comparison and inspection of the 4237 checkpoint:
sbatch -A "$ACCOUNT" -p "$PARTITION" -t 48:00:00 donor-allocation.sh donor-cycles.prm
```

Job 3616642 reported loading both libraries under `./plugin/` and printed the
five (x, y, peak) triples from `./fault.txt`. Its repair instructions invoke
`ibrun ./plugin/aspect-release continue-3613912.prm`. In the preceding job
3613912 the stack names the executable under
`/scratch/11463/yiminjin/bp3_dg/plugin/build/aspect-release` and the plugin
source under `/work2/11463/yiminjin/stampede3/build/aspect/pf-rsf-dg/source`.
The repair rebuilt the plugin in that scratch build directory. The archived
3616642 output does not record the expanded SDK path or the final ELF hashes.
The launcher records the resolved executable, DSOs, input, ordered fault file,
and the plugin build cache's `Aspect_DIR` when available, and verifies the
executable against that SDK. For a separate build directory, export
`BP3_PLUGIN_BUILD` to its absolute path. If the original cache is unavailable,
the identity record explicitly marks its absence. Check the archived startup:

```bash
python3 donor_startup.py BP3.o3616642 fault.txt
```

The checker also runs on the new solver log. It verifies the two loaded DSOs
and the signed, five-anchor geometry the model actually reports. It accepted
the archived job, and rejected a changed anchor or a missing library line.

## Stage the exact inputs and restart

The prepared archives are in the worktree's `build-bp3-preparation/launcher` directory.
With the two archives and `server-archives.sha256` in a fresh staging
directory on the server, the following recovers the input kit and identifies
the fallback source archive.

```bash
sha256sum -c server-archives.sha256
tar -xzf donor-inputs.tar.gz
KIT="$PWD/donor-inputs"
BP3_SOURCE_ARCHIVE="$PWD/donor-source-612b75d.tar.gz"
(cd "$KIT" && sha256sum -c SHA256SUMS)
```

Run from the pf-rsf worktree. These commands create a new transportable
directory if regenerating the kit. The checkpoint continuation needs the three
original Stampede3 binaries or a replay-qualified rebuild and the native
checkpoint on shared storage.

```bash
set -euo pipefail
H="$PWD/refactoring/phase-field-fault/results/bp3-preparation"
D="$PWD/../aspect-pf-rsf-particle-free"
P=benchmarks/reconstructed_fault/particle_free/bp3_dg_first_event
R=612b75d3570d81763407875f3a12c12999b9fa8e
KIT=$(mktemp -d -t bp3-server-inputs.XXXXXXXX)
cp "$D/$P/output_resumed/original.prm" "$KIT/job3616642.prm"
cp "$D/$P/output_resumed/BP3.o3616642" "$KIT/"
for f in fault.txt environment.sh; do git -C "$D" show "$R:$P/$f" > "$KIT/$f"; done
git -C "$D" show "$R:benchmarks/reconstructed_fault/bp3/branch_output.sh" > "$KIT/branch_output.sh"
cp "$H"/{donor-allocation.sh,donor_parent.py,donor_replay.py,donor_startup.py,donor-replay.prm,donor-cycles.prm,history_events.py} "$KIT/"
sha256sum "$KIT"/{job3616642.prm,fault.txt,BP3.o3616642}
```

On the server point `KIT` to the transported directory. Substitute the actual
parent path if it was archived elsewhere. Load the dependency modules used by the
writer. Use a new destination. The donor's branch helper copies the selected
native slot, accepted metadata and the referenced profiles. It also carries
forward the available native visualization trees. Allow space for these copies.
If an indexed profile is missing, locate its original payload in the earlier
run. Native restart serializes accepted FE, particle, fault and output history.
A VTK profile or a table with the correct clock cannot reconstruct it.

Before budgeting further allocations, run the catalogue on the checkpoint's
own `bp3_output_metadata`. If its station and accepted-step files start partway
through the run, supply their earlier matching prefixes too. If the prefixes
cannot be recovered, a fresh initial run is the reproducible route to three
fully observed events.

```bash
set -euo pipefail
parent="$SCRATCH/bp3_dg/work_repair_3613912"
writer="$SCRATCH/bp3_dg"
export BP3_RUN="$SCRATCH/bp3-dg-three-events"
python3 "$KIT/donor_parent.py" "$KIT/job3616642.prm" "$KIT/fault.txt" "$parent/restart"
mkdir "$BP3_RUN"
cp -a "$KIT/." "$BP3_RUN/"
ln -s "$writer/plugin" "$BP3_RUN/plugin"
bash "$BP3_RUN/branch_output.sh" "$parent" 3 "$BP3_RUN/output"
cd "$BP3_RUN"
python3 ./donor_parent.py job3616642.prm fault.txt output/restart
source ./environment.sh
./plugin/aspect-release --validate donor-replay.prm
./plugin/aspect-release --validate donor-cycles.prm
```

The gate checks the published marker, the 4229 physical clock, writer rank
count, expected fixed/variable attachments from mesh.info, mesh, resume.z,
and the tail of each checkpointed observer series. The MPI mesh and attachments
are shared files. Testing for 160 separate native files would be incorrect.
If the marker points elsewhere, keep that generation intact and identify its
accepted parent from the logs before attempting a different replay.

The historical production command is `ibrun ./plugin/aspect-release input.prm`
under the matching two-node, 160-task stack, one thread per rank. The in-allocation
wrapper below records binary and input hashes, chosen input, module list and
stdout. Its log names prevent replacing the first launch's evidence. The archived
coarse single-rank first-cycle scheduler script targets a different experiment
and explicitly stops after its first event. An earlier 32-rank, one-node ICX
example and an SPR template also cannot identify the allocation or executable
of job 3616642. Its actual job header records the node names and task count. The
three-cycle setup uses its own stop controls.

Read the donor launchers and job header directly from the workstation:

```bash
git -C "$D" show "$R:benchmarks/reconstructed_fault/bp3/stampede3_first_cycle.sh"
git -C "$D" show "$R:benchmarks/reconstructed_fault/bp3/first_cycle_readiness.json"
cat ../aspect/benchmarks/reconstructed_fault/bp3/server_info/bp3-icx-24h.slurm
head -n 30 "$D/$P/output_resumed/BP3.o3616642"
cat "$D/$P/output_resumed/BP3.e3616642"
```

```bash
# Within a 160-task allocation, allow time for the bounded hour and terminal checkpoint.
bash ./donor-allocation.sh donor-replay.prm
python3 ./donor_replay.py BP3.o3616642 "launch-${SLURM_JOB_ID}-donor-replay.prm.out"
slot=$(cat output/restart/last_good_checkpoint.txt)
printf -v selected '%02d' "$slot"
cat "output/restart/$selected/bp3_accepted_state.txt"
```

Require the terminal label **4237 5172072500.6796865**. The eight accepted
steps 4230–4237 must reproduce each archived clock, duration, matching-frame
shear and normal work defect, work bound/guard and the independent native stress
check (error, allowance fraction, stress scale). The comparison uses exact
printed values as a same-stack replay test. A rebuild with a different
reduction order can need a separate quantified comparison if this test fails.
If the parent's live accepted_steps.csv and stations.csv are available through
4237, also compare the full eight rows and their station records by accepted
step and station coordinate. Only the logs of this part of the original run
were available locally. Preserve the joined parent output through this clock.
The maximum speed, material station slip, age, shear and normal traction
columns should accompany every subsequent restart. Any difference in the
input ledger, output clocks, native stress, fault response or duplicate
publication requires investigation before a long run.

```bash
python3 ./donor_replay.py BP3.o3616642 "launch-${SLURM_JOB_ID}-donor-replay.prm.out" \
        --tables "$parent" output
```

```bash
# In each later 160-task allocation with the writer stack and BP3_RUN set.
cd "$BP3_RUN"
bash ./donor-allocation.sh donor-cycles.prm
python3 ./history_events.py output > "events-${SLURM_JOB_ID}.json"
```

The original 47,336,400,000 s end, physical parameters, adaptive step controls
and ten-step checkpoint cadence remain in effect. The continuation turns off
the boolean that gates termination at the **first** completion. The native
first-event tracker is serialized and latches. It does not start counting again.
Its stop flag affects termination, not the surface solve or cumulative slip.
The per-accepted-step tables continue after its one-shot milestone. Analyze
those tables after each segment as described in history.md. If the reconstructed
branch's tables start at a later accepted step, supply the ordered, overlapping
earlier complete output directories to history_events.py before `output`.
Back up the selected checkpoint, checksums, complete tables and indexed
profiles to durable storage after each graceful stop. The 158,400 s budget
leaves four hours for a terminal checkpoint in a 48-hour reservation. Adjust
it downward for a shorter reservation. Retain the prefix manifest and create
a fresh branch from the selected slot after any kill. Schedule the next
allocation only while fewer than three independently recovered events have
been measured. A failed guard, insufficient slip or missing return to sliding
does not count as a recovered event.

## If the saved parent or writer is unavailable

Rebuild in a new directory from the same immutable source. Preserve the old
writer. Generate the complete source archive on the workstation, then transport
it and the inputs to the server. Use the actual matching server deal.II and
Voro++ installations. The source archive is approximately 458 MB uncompressed
(sum of tracked blob sizes). The compiler link stack and `CMakeCache.txt` from
the writer installation are valuable when reproducing the binary.

```bash
git -C "$D" archive --format=tar.gz --output="$KIT/donor-source.tar.gz" "$R"
```

```bash
set -euo pipefail
release="$WORK/bp3-612b75d-rebuild"
mkdir "$release"
mkdir "$release/source"
tar -xzf "${BP3_SOURCE_ARCHIVE:-$KIT/donor-source.tar.gz}" -C "$release/source"
P=benchmarks/reconstructed_fault/particle_free/bp3_dg_first_event
fresh="$SCRATCH/bp3-612b75d-fresh"
mkdir "$fresh"
cp -a "$release/source/$P/." "$fresh/"
cd "$fresh"
source ./environment.sh
export CXX=icpx
: "${DEAL_II_DIR:?Set the matching deal.II 9.7 installation}"
: "${VORO_DIR:?Set the matching Voro++ installation}"
cmake -S "$release/source" -B "$release/build" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER="$CXX" -DDEAL_II_DIR="$DEAL_II_DIR" \
      -DASPECT_WITH_VORO=ON -DVORO_DIR="$VORO_DIR" -DASPECT_WITH_WORLD_BUILDER=ON \
      -DASPECT_ADDITIONAL_CXX_FLAGS="-fno-finite-math-only -fp-model=precise -ffp-contract=off"
cmake --build "$release/build" -j 4
cmake -S "$fresh/plugin" -B "$release/plugin-build" -DAspect_DIR="$release/build" \
      -DBP3_ASPECT_SOURCE="$release/source" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$CXX"
cmake --build "$release/plugin-build" -j 4
./plugin/aspect-release --test
./plugin/aspect-release --validate startup.prm
./plugin/aspect-release --validate startup-resume.prm
```

If the native 4229 parent exists, point the new replay branch's `plugin`
symlink at this newly built trio and qualify the eight observed steps. If
the parent cannot be recovered, a full-grid fresh run is the alternative.
The pinned package supplies a separate bounded 0–5 startup and 5–8 restart
with no change to physics. Use separate suitable allocations to verify a
graceful 5 and 8 with native restart and inspect their accepted clocks,
material state, solved fault and work diagnostics. Then continue from the
accepted generation using the archived physical controls and the cycle stop
override. The three-event horizon and full-grid time remain unmeasured.

```bash
# Work in $fresh in an allocation. Keep the startup and resume logs separately.
ibrun ./plugin/aspect-release startup.prm > startup-0-to-5.out 2>&1
ibrun ./plugin/aspect-release startup-resume.prm > startup-5-to-8.out 2>&1
# Only after inspecting the 5 and 8 restart outputs.
cp "$KIT"/{job3616642.prm,donor-cycles.prm,donor-allocation.sh,donor_startup.py} "$fresh/"
export BP3_PLUGIN_BUILD="$release/plugin-build"
export BP3_RUN="$fresh"
bash ./donor-allocation.sh donor-cycles.prm
```

## Local preparation checks

The replay checker passed against an excerpt starting at 4230 and ending
at 4237 from the archived 3616642 stdout. Final accepted diagnostic stress
scale was 5085295.6541678132 Pa. Adding 1 Pa at 4230 to that step's stress
scale failed at that quantity. Removing its accepted-state marker failed at
4230. For the optional two-table comparison, eight test frames were numbered
4230–4237 using real station and accepted-step table layouts from the older
particle run. All eight summary rows and 96 station rows passed unchanged.
Adding 0.01 m to one first-station slip failed at its step, distance and column.
The metadata preflight accepted a synthetic 160-rank slot with the
correct 4229 clock, rejected a missing native variable-data attachment and
rejected either a station table ending at 4228 or the 4237 accepted clock
substituted for the checkpoint. An actual
older 160-rank step-42 snapshot failed its slot check. The launcher passed
Bash syntax and rejected a 32-rank or missing allocation. These are local
input and evidence checks. The original 4229 state, Stampede3 executable
and live scheduler were unavailable for a numerical restart here.
