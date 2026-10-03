#!/usr/bin/env python3
"""Bounded isolated runs; no server or production input writes."""
from pathlib import Path
import os,subprocess,sys,json
r=Path(__file__).resolve().parent;repo=r.parents[2]
if sys.argv[1]=='--filter-unit':
 env=dict(os.environ,ASPECT_SOURCE_DIR=str(repo))
 result=subprocess.run([sys.executable,str(r/'run_logged.py'),'filter-unit-np2','120',
  'mpirun','-np','2',str(repo/'build-refactor-r6b/aspect-filter-derivative-qualified'),'--test',
  '[fault_normal_filter],[fault_normal_filter_traces],[fault_boundary_contact],[fault_bottom_source],[phase_field_profile_bounds],[bp3_restore_profile]'],cwd=repo,env=env)
 raise SystemExit(result.returncode)
if sys.argv[1]=='--endpoint-unit':
 env=dict(os.environ,ASPECT_SOURCE_DIR=str(repo))
 result=subprocess.run([sys.executable,str(r/'run_logged.py'),'endpoint-unit-np2','120',
  'mpirun','-np','2',str(repo/'build-refactor-r6b/aspect-endpoint-plane-qualified'),'--test',
  '[fault_boundary_contact],[fault_bottom_source],[phase_field_profile_bounds],[bp3_restore_profile]'],cwd=repo,env=env)
 raise SystemExit(result.returncode)
if sys.argv[1]=='--unit':
 ranks=sys.argv[2] if len(sys.argv)>2 else '1'
 env=dict(os.environ,ASPECT_SOURCE_DIR=str(repo))
 result=subprocess.run([sys.executable,str(r/'run_logged.py'),'unit-profile-bounds-np'+ranks,'120',
  'mpirun','-np',ranks,str(repo/'build-refactor-r6b/aspect-profile-bounds-qualified'),'--test',
  '[phase_field_profile_bounds],[bp3_restore_profile]'],cwd=repo,env=env)
 raise SystemExit(result.returncode)
if sys.argv[1]=='--batch':
 for item in sys.argv[2:]:
  case,ranks=item.split(':')
  result=subprocess.run([sys.executable,__file__,case,ranks])
  assert result.returncode==0, (case,result.returncode)
 raise SystemExit(0)
case=sys.argv[1];ranks=sys.argv[2] if len(sys.argv)>2 else '1'
env={k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')};env['ASPECT_SOURCE_DIR']=str(repo)
label=case+(('-'+sys.argv[3]) if len(sys.argv)>3 else '')+'-np'+ranks
used=sum(json.loads(p.read_text())['seconds'] for p in (r/'evidence').glob('*-np*.json'))
assert used<1200,'Local simulation budget exhausted'
result=subprocess.run([sys.executable,str(r/'run_logged.py'),label,str(min(180,1200-used)),'mpirun','-np',ranks,str(repo/('build-refactor-r6b/aspect-profile-bounds-qualified' if case.startswith('qualified-') else 'build-refactor-r6b/aspect-endpoint-plane-qualified' if case.startswith('endpoint-') else 'build-refactor-r6b/aspect-filter-derivative-qualified' if case.startswith('filter-') else 'build-refactor-r6b/aspect-particle-lifecycle-qualified')),*(['--validate'] if case.endswith('-parse') else []),str(r/'inputs'/f'{case}.prm')],cwd=repo,env=env,check=False)

if case.startswith('guard-'):
 log=(r/'evidence'/f'{label}.log').read_text(errors='replace')
 assert result.returncode!=0 and 'BP3 GEOMETRY PASS:' in log, 'Missing explicit geometry pass marker'
 assert all(f'BP3 GEOMETRY CHECK COMPLETE rank {i}' in log for i in range(int(ranks))), 'Not every rank completed the geometry checks'
elif case in ('qualified-old-resume','qualified-changed-resume','qualified-evolving'):
 log=(r/'evidence'/f'{label}.log').read_text(errors='replace')
 marker=('EVOLVING H TRANSFER PASS:' if case=='qualified-evolving' else 'BP3 restart requires the same geometry identity')
 assert result.returncode!=0 and marker in log, 'Missing expected qualification/rejection marker'
else:
 raise SystemExit(result.returncode)
