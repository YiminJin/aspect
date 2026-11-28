"""Check raw repeated frames and the 2 ms restart inventory, and measure interval cost.

Run from the worktree root after the two-ms analysis.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import tarfile
import numpy as np
sys.path.insert(0,"benchmarks/granite_fault_growth")
from compare_drive import step_times
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("runs",type=Path,help="Root with coarse (superseded) and coarse2 (bounded) run directories")
parser.add_argument("prior",type=Path,help="Original compact output through 1 ms")
parser.add_argument("results",type=Path)
args=parser.parse_args()
w,d=args.runs,args.results
folder="output-granite-driven-continuation"
before=w/"coarse"/folder
end=w/"coarse2"/folder
historical=args.prior
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
overlap=sorted(before.glob("granite-*.csv"))
assert len(overlap)%3==0
for old in overlap: assert digest(old)==digest(end/old.name), old
changed=[]
for p in end.glob("granite-phase-*.csv"):
    rank=p.stem.split('-')[-1]
    if digest(p)!=digest(historical/f"granite-phase-0134-{rank}.csv"): changed.append(p.name)
with (d/"two-ms.csv").open() as f: rows=list(csv.DictReader(f))
times=step_times([w/"coarse2/timing/wall.log"])
assert [int(r['step']) for r in times]==[int(r['step']) for r in rows if float(r['time_s'])>.001]
metadata=end/'restart'/f"{int((end/'restart/last_good_checkpoint.txt').read_text()):02}"/'metadata.txt'
with tarfile.open(d/'continuation-2ms.tar.gz') as saved:
    for member in saved.getmembers():
        if member.isfile(): assert saved.extractfile(member).read()==(end/member.name).read_bytes()
with Path("refactoring/phase-field-fault/example2/steady-drive/drive-cost.csv").open() as f:
    first=[float(r['elapsed_s']) for r in csv.DictReader(f)]
r=dict(raw_overlap_files=len(overlap),raw_overlap_end_s=float(np.loadtxt(before/'statistics')[-1,1]),
       earlier_seconds_per_step=np.mean(first),continued_seconds_per_step=np.mean([r['elapsed_s'] for r in times]),
       continued_dt_range_s=[min(np.diff([float(r['time_s']) for r in rows if float(r['time_s'])>=.001])),
                             max(np.diff([float(r['time_s']) for r in rows if float(r['time_s'])>=.001]))],
       continued_repeats=(w/'coarse2/timing/solver.log').read_text().count('Repeating the current time step'),
       changed_sparse_phase_shards=len(changed),phase_shards=len(list(end.glob("granite-phase-*.csv"))),
       max_abs_ahead_phi=max(float(r['ahead_phi_abs_bound']) for r in rows),
       max_front_envelope_gap_m={c:max(abs(float(r[f'global_upper{c}_m'])-float(r[f'x{c}_m'])) for r in rows) for c in ('001','005')},
       continued_wall_s=json.loads((w/'coarse2/timing/exit.json').read_text())['wall_s'],
       archive_sha256=digest(d/'continuation-2ms.tar.gz'),resume_sha256=digest(metadata.parent/'resume.z'),
       restart_metadata=metadata.read_text().strip().splitlines())
(d/'replay-results.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
