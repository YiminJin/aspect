#!/usr/bin/env python3
"""Compact independent fixture, geometry and qualification evidence."""
from pathlib import Path
import csv,json,math,hashlib,shutil,re
import numpy as np
r=Path(__file__).resolve().parent;repo=r.parents[2];old=r.with_name('bp3_geometry')
result={}
# Compare the new loading primitive to the preserved table and its original
# Gauss8 partial-panel convention. This table is diagnostic input only.
p=repo/'benchmarks/reconstructed_fault/refactoring_r1/fixtures/profile.txt'
with p.open() as f:
 n,scale=map(float,f.readline().split());table=np.loadtxt(f)
x,phi,integral=table.T
q,w=np.polynomial.legendre.leggauss(8);q=(q+1)/2;w=w/2
errors=[];phase_errors=[]
for row in csv.DictReader((r/'output-dip-60-matched/loading_profile.csv').open()):
 d=float(row['r']);a=abs(d);expected=0. if d<0 else 1.;expected_phi=0.
 if a<x[-1]:
  i=np.searchsorted(x,a,side='right')-1
  y=x[i]+(a-x[i])*q;ph=phi[i]+(phi[i+1]-phi[i])*(y-x[i])/(x[i+1]-x[i])
  value=integral[i]+(a-x[i])*np.dot(w,scale*ph*(1+ph)/(1-ph)**2)
  expected=.5+math.copysign(.5,d)*value/integral[-1]
  expected_phi=np.interp(a,x,phi)
 errors.append(abs(expected-float(row['C'])));phase_errors.append(abs(expected_phi-float(row['phi'])))
result['legacy_loading']={'samples':len(errors),'max_C_difference':max(errors),'max_phase_difference':max(phase_errors),'passed':bool(max(errors)<2e-13 and max(phase_errors)<2e-11)}
# Resolution check independent of the plugin's distance helper.
meshes={}
for case,dip in [('small-60-startup',60),('small-45-startup',45),('small-45-reverse-startup',45),('baseline-graded',60)]:
 data=[]
 for p in (r/('output-'+case)).glob('initial_mesh_*.csv'):data+=list(csv.DictReader(p.open()))
 assert data,case
 normal=np.array([math.sin(math.radians(dip)),math.cos(math.radians(dip))])
 upper=np.array([-500/math.tan(math.radians(dip)),1000.])
 fine=500/128;support=x[-1]/200;band=max(40,support+2*fine)
 invalid=0;support_cells=0
 for row in data:
  h=float(row['h']);center=np.array([float(row['x']),float(row['y'])])
  d=max(0,abs(np.dot(normal,center-upper))-.5*h*sum(abs(normal)))
  target=fine if d<=band else min(250,2*fine+(d-band)/4)
  invalid+=h>target*(1+1e-12) or h<fine*(1-1e-12)
  support_cells+=d<=support
 count=len(data);area=sum(float(row['h'])**2 for row in data)
 meshes[case]={'cells':count,'support_cells':int(support_cells),'h_min':min(float(row['h']) for row in data),'h_max':max(float(row['h']) for row in data),'area':area,'invalid':int(invalid),'unique_cells':len({row['cell'] for row in data})}
 assert invalid==0 and area==2000000 and len({row['cell'] for row in data})==count
 (r/'results'/(case+'-mesh.json')).write_text(json.dumps(meshes[case],indent=2)+'\n')
 if case=='small-60-startup':reference=sorted(data,key=lambda x:x['cell'])
 if case=='baseline-graded':assert reference==sorted(data,key=lambda x:x['cell'])
result['meshes']=meshes
result['graded_limitation']={'profile_radius':support,'fine_band':band,'maximum_projected_cell_width_60':125*(math.sqrt(3)/2+.5),'required_uniform_boundary_half_width_60':(support+125*(math.sqrt(3)/2+.5))/(math.sqrt(3)/2),'preserved_refinement_band_boundary_half_width_60':band/(math.sqrt(3)/2),'accepted_plugin_reproduces':True}
# Explicit rejection markers and clean runtime only model.
for case,marker in [('heterogeneous','BP3 requires identical stationary profiles in all materials.'),('legacy-rejected','BP3 requires frozen mature mechanics and automatic prescribed boundary completion.'),('baseline-graded','Nonuniform or misaligned boundary ghost-Q1 lattice')]:
 log=next((r/'evidence').glob(case+'-np*.log')).read_text(errors='replace')
 result[case]=marker in log
 assert result[case],case
assert json.loads((r/'evidence/model-only-np1.json').read_text())['exit_code']==0
result['model_only_runtime']=True
# Complete run inventory, and immutable copies of the actual expanded inputs.
runs=[]
for p in sorted((r/'evidence').glob('*-np*.json')):
 d=json.loads(p.read_text());runs.append({'case':p.stem,'exit_code':d['exit_code'],'seconds':d['seconds'],'rss_kib':d['child_max_rss_kib']})
 case=Path(d['command'][-1]).stem;params=r/'inputs'/(case+'.prm')
 original=r/('output-'+case)/'original.prm'
 if original.exists():params=original
 if params.exists():shutil.copyfile(params,r/'evidence'/(p.stem+'.prm'))
result['runs']=runs;result['total_simulation_seconds']=sum(x['seconds'] for x in runs)
result['source_commit']='48f23492d'
paths=['build-refactor-r6b/aspect-filter-derivative-qualified','benchmarks/reconstructed_fault/bp3_geometry/build/bp3/libbp3_restore_150x50.release.so','benchmarks/reconstructed_fault/bp3_runtime/build/libbp3_restore_150x50.release.so','benchmarks/reconstructed_fault/bp3_runtime/build/observer/libfrozen_birth_observer.release.so']
result['artifacts']={s:hashlib.sha256((repo/s).read_bytes()).hexdigest() for s in paths}
(r/'results/summary.json').write_text(json.dumps(result,indent=2)+'\n')
# Keep compact physical/history evidence, not all bulk/particle dumps.
for case in ['dip-60-matched','dip-45-serial','dip-45-reverse-serial','staggered','create','resume','direct','retry','direct2','retry2','model-only']:
 source=r/('output-'+case);dest=r/'results'/case;dest.mkdir(exist_ok=True)
 for pattern in ['accepted_steps.csv','restored_growth.csv','lifecycle_rank*.csv','birth_H_rank*.csv','loading_profile.csv','rng-*.txt','profiles/fault_*.csv']:
  for p in source.glob(pattern):shutil.copyfile(p,dest/p.name)
print(json.dumps({k:v for k,v in result.items() if k not in ['runs','artifacts','meshes']},indent=2))
