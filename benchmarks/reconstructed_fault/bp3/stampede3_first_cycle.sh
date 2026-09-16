#!/bin/bash
# Prepared only. Submit manually after the corrected-baseline review.
# Supply your allocation with sbatch -A ACCOUNT; no automatic resubmission.
#SBATCH --job-name=bp3-coarse-first
#SBATCH --partition=skx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --output=bp3-first-%j.out
#SBATCH --error=bp3-first-%j.err

set -euo pipefail
: "${BP3_SOURCE_DIR:?Set the server ASPECT source directory}"
: "${BP3_ASPECT_BINARY:?Set the matching server Release executable}"
: "${BP3_OUTPUT_DIR:?Set a dedicated output directory under SCRATCH}"
: "${SLURM_JOB_ID:?Submit with Slurm, not on a login node}"

# Review gate comes before executable invocation or output-directory mutation.
python3 - "$BP3_SOURCE_DIR/benchmarks/reconstructed_fault/bp3/first_cycle_readiness.json" <<'PY'
import json, sys
r=json.load(open(sys.argv[1]))
if not (r['server_ready'] and r['filesystem_restart_equivalence_passed']):
    raise SystemExit('BP3 launch HELD FOR REVIEW: '+r['review_reason'])
PY

# Do not inherit development flags, including presence-based flags set to zero.
# BP3-specific paths above are retained. Reinstall the explicit source-path
# override used by ASPECT's existing input/library expansion infrastructure.
while IFS= read -r flag; do unset "$flag"; done < <(compgen -v ASPECT_)
export ASPECT_SOURCE_DIR="$BP3_SOURCE_DIR"
export ASPECT_FAULT_SURFACE_SOLVER=tridiagonal
export ASPECT_FAULT_EXPLICIT_B=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DEAL_II_NUM_THREADS=1

# Use the already loaded, consistent compiler/MPI/deal.II environment.
# Do not mix the locally built OpenMPI binary with a server Intel-MPI library.
module list
mkdir -p "$BP3_OUTPUT_DIR"
cd "$BP3_OUTPUT_DIR"

python3 - "$BP3_SOURCE_DIR" "$BP3_OUTPUT_DIR" "$SLURM_JOB_ID" <<'PY'
import csv, pathlib, sys
source, output=map(lambda s:pathlib.Path(s).resolve(), sys.argv[1:3])
event=output/'first_event.csv'
if event.exists() and next(csv.DictReader(event.open()))['complete']=='1':
    raise SystemExit('BP3 first event already completed; no additional solve is allowed.')
steps=output/'accepted_steps.csv'
if steps.exists():
    last=None
    for row in csv.DictReader(steps.open()): last=row
    if last and float(last['time'])>=47336400000:
        raise SystemExit('BP3 safety end already reached; inspect event status rather than resuming.')
resume=(output/'restart/last_good_checkpoint.txt').exists()
with (output/('launch-'+sys.argv[3]+'.prm')).open('x') as f:
    f.write('include '+str(source/'benchmarks/reconstructed_fault/bp3/bp3_first_cycle_coarse.prm')+'\n')
    f.write('set Output directory = '+str(output)+'\n')
    f.write('set Resume computation = '+str(resume).lower()+'\n')
PY

sha256sum "$BP3_ASPECT_BINARY" "$BP3_SOURCE_DIR/benchmarks/reconstructed_fault/bp3/build/libbp3.release.so"
env | sort | sed -n '/^ASPECT_/p; /^OMP_/p; /^OPENBLAS_/p; /^MKL_/p; /^DEAL_II_NUM_THREADS/p'
# No profiling/comparison callback, interface correction, sparse G or GMG.
ibrun "$BP3_ASPECT_BINARY" "launch-$SLURM_JOB_ID.prm" 2>&1 | tee "aspect-$SLURM_JOB_ID.log"

python3 - <<'PY'
import csv
r=next(csv.DictReader(open('first_event.csv')))
if r['complete']=='1': print('BP3 FIRST EVENT COMPLETE; see first_event.csv and event_states/termination.')
elif r['started']=='0': print('BP3 NO FIRST SEISMIC EVENT: normal program exit is not event success.')
else: print('BP3 FIRST EVENT INCOMPLETE: onset occurred, but the five-state termination criterion did not pass.')
PY
