#!/usr/bin/env python3
from pathlib import Path
import json,hashlib,csv,shutil
r=Path(__file__).resolve().parent;repo=r.parents[2]
text=(r/'evidence/production-json-np1.log').read_text();start=text.index('{\n');data,_=json.JSONDecoder().raw_decode(text[start:])
def values(d):
 if not isinstance(d,dict):return d
 if 'value' in d and 'default_value' in d:return d['value']
 return {k:values(v) for k,v in d.items()}
d=values(data)
keys=['Dimension','Resume computation','Maximum time step','Maximum first time step','Maximum relative increase in time step','Nonlinear solver tolerance','Max nonlinear iterations','Nonlinear solver failure strategy','CFL number','Material model','Phase field model','Geometry model','Fault reconstruction','Mesh refinement','Time stepping','Particles','Solver parameters','Postprocess','Checkpointing','Discretization']
selected={k:d[k] for k in keys}
selected['Material model']={'Model name':d['Material model']['Model name'],'Phase field fault':d['Material model']['Phase field fault']}
selected['Geometry model']={'Model name':'box','Box':d['Geometry model']['Box']}
selected['Postprocess']={k:d['Postprocess'][k] for k in ['List of postprocessors','BP3','BP3 restored monitor','Particles','Visualization','Reconstructed faults']}
(r/'results/production_resolved.json').write_text(json.dumps(selected,indent=2)+'\n')
paths=['build-refactor-r6b/aspect-filter-derivative-qualified','benchmarks/reconstructed_fault/bp3_runtime/build/libbp3_restore_150x50.release.so','benchmarks/reconstructed_fault/bp3/build-maintained/libbp3_restore_150x50.release.so','benchmarks/reconstructed_fault/bp3_packaging/build/observer/libfrozen_birth_observer.release.so']
manifest={s:hashlib.sha256((repo/s).read_bytes()).hexdigest() for s in paths}
# Record the inspected scientific-worktree copies; no writes are made there.
for name in ['bp3_150x50_filter20.prm','bp3_150x50_first_event.prm']:
 p=repo.parent/'aspect/benchmarks/reconstructed_fault/bp3'/name
 manifest[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
checks=json.loads((r/'results/checks.json').read_text())
runs=[]
for p in sorted((r/'evidence').glob('*-np*.json')):
 record=json.loads(p.read_text());runs.append({'case':p.stem,'exit_code':record['exit_code'],'seconds':record['seconds'],'rss_kib':record['child_max_rss_kib']})
 input_file=Path(record['command'][-1]);shutil.copyfile(input_file,r/'evidence'/(p.stem+'.prm'))
for case in ['model-only','create','staggered','quiet','resume','reschedule','branch','direct','retry']:
 src=r/('output-'+case);dest=r/'results'/case;dest.mkdir(exist_ok=True)
 for pattern in ['accepted_steps.csv','restored_growth.csv','particle_summary.csv','profiles.csv','heavy_outputs.csv','lifecycle_rank*.csv','birth_H_rank*.csv']:
  for p in src.glob(pattern):shutil.copyfile(p,dest/p.name)
 for checkpoint in (src/'restart').glob('*/bp3_output_metadata/particle_summary.csv'):
  shutil.copyfile(checkpoint,dest/('checkpoint-'+checkpoint.parent.parent.name+'-particle-summary.csv'))
summary={'accepted_baseline':'e2ff248fb','passed_checks':sum(x['passed'] for x in checks),'checks':len(checks),'artifacts':manifest,'runs':runs,'total_seconds':sum(x['seconds'] for x in runs),'production_mesh_run':False,'server_input_confirmed':False,'graded_completion_blocker':'unchanged; Section-3 evidence reused'}
(r/'results/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['artifacts','runs']},indent=2))
print({k:selected[k] for k in keys[:9]})
