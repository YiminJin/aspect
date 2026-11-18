#!/usr/bin/env python3
"""Small retained-log cost summary; no tensor/phase-array reduction or new run."""
from pathlib import Path
import csv,json,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
BUILD=ROOT/'build-particle-free/stage4-preparation';OUT=HERE.parent/'evidence/stage4_preparation'
def rows(path):
    return list(csv.DictReader(path.open())) if path.exists() else []
record={};active=[]
for run in sorted((ROOT/'build-particle-free/bulk-tip/runs').glob('stage4-preparation-*')):
    command=json.loads((run/'command.json').read_text());label=run.name.removeprefix('stage4-preparation-');out=run/'output'
    if 'exit_code' not in command:active.append(label);continue
    profile=rows(out/'tip-profile.csv');solver=rows(out/'tip-solver.csv');accepted=rows(out/'bulk-accepted.csv')
    compression=json.loads((out/'lossless-compression.json').read_text()) if (out/'lossless-compression.json').exists() else {}
    if (out/'early-compression.json').exists():compression.update(json.loads((out/'early-compression.json').read_text()))
    record[label]=dict(exit_code=command['exit_code'],wall_seconds=command['seconds'],ranks=command['arguments']['ranks'],requested_end_step=command['arguments']['end_step'],accepted_intervals=len(accepted),last_accepted_step=None if not accepted else int(accepted[-1]['step']),last_accepted_time=None if not accepted else float(accepted[-1]['time']),mean_physical_step_seconds=None if not profile else sum(float(r['step_seconds']) for r in profile)/len(profile),maximum_physical_step_seconds=None if not profile else max(float(r['step_seconds']) for r in profile),nonlinear_iterations=sum(int(r['nonlinear_iterations']) for r in solver),outer_krylov_iterations=sum(int(r['krylov_iterations']) for r in solver),stokes_inverse_solves=sum(int(r['stokes_solves']) for r in solver),original_observation_bytes=sum(r['bytes'] for r in compression.values()),compressed_observation_bytes=sum(r['gzip_bytes'] for r in compression.values()))
entry=json.loads((BUILD/'entry.json').read_text())
result=dict(cases=record,active_cases=active,completed_numerical_wall_seconds=sum(r['wall_seconds'] for r in record.values()),completed_rank_hours=sum(r['wall_seconds']*r['ranks']/3600 for r in record.values()),total_elapsed_seconds=time.time()-entry['started_unix'],deadline_unix=entry['deadline_unix'],note='Numerical wall includes failed preparation; total elapsed also includes build, analysis and orchestration. A zero exit does not imply the requested step was reached.')
OUT.mkdir(exist_ok=True);(OUT/'cost.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
