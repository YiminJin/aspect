#!/usr/bin/env python3
"""Report exact lifecycle checks and scaled field differences, including failures."""
from pathlib import Path
import csv,json,shutil
import numpy as np
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
from vtkmodules.util.numpy_support import vtk_to_numpy
r=Path(__file__).resolve().parent;o=r/'results';o.mkdir(exist_ok=True)
def rows(d,name):return list(csv.DictReader((r/('output-'+d)/name).open()))
def accepted(d):return {x['step']:x for x in rows(d,'lifecycle_rank0.csv') if x['stage']=='accepted'}
checks=[];differences=[];states=[]
for case in ['coupled-regular-Hlimited','coupled-random-5433-Hlimited','crossing-serial','crossing-mpi','crossing-create','crossing-resume','crossing-after-create','crossing-after-resume','crossing-direct','crossing-retry']:
 d=r/('output-'+case)
 for p in sorted(d.glob('lifecycle_rank*.csv')):
  for x in csv.DictReader(p.open()):
   if x['stage'] in ('accepted','rejected','restored_before_advection'):
    states.append(dict(case=case,rank=p.stem.split('rank')[1],**x))
 for name in ['accepted_steps.csv','restored_growth.csv']:
  if (d/name).exists():shutil.copyfile(d/name,o/(case+'-'+name))
for ref,trial,steps in [('crossing-serial','crossing-resume',['1','2']),('crossing-serial','crossing-after-resume',['2']),('crossing-direct','crossing-retry',['1'])]:
 a,b=accepted(ref),accepted(trial)
 for step in steps:
  for field in ['n','next_id','position_hash','property_hash','integrator_hash']:
   checks.append(dict(reference=ref,candidate=trial,step=step,field=field,equal=a[step][field]==b[step][field],reference_value=a[step][field],candidate_value=b[step][field]))
retry=rows('crossing-retry','lifecycle_rank0.csv');initial=next(x for x in retry if x['stage']=='accepted' and x['step']=='0');restored=next(x for x in retry if x['stage']=='restored_before_advection' and x['time']=='50')
for field in ['n','next_id','position_hash','property_hash','integrator_hash']:
 checks.append(dict(reference='pre-attempt',candidate='restored-before-retry',step='1',field=field,equal=initial[field]==restored[field],reference_value=initial[field],candidate_value=restored[field]))
# Numeric comparisons use common fault nodes and accepted times. Store absolute
# differences; no automatic relaxation of a scientific acceptance tolerance.
for ref,trial in [('coupled-regular-Hlimited','coupled-random-5433-Hlimited'),('crossing-serial','crossing-mpi'),('crossing-serial','crossing-resume'),('crossing-serial','crossing-after-resume'),('crossing-direct','crossing-retry')]:
 for file in ['accepted_steps.csv','restored_growth.csv']:
  a={x['step']:x for x in rows(ref,file)};b={x['step']:x for x in rows(trial,file)}
  common=a.keys()&b.keys()
  for col in a[next(iter(a))]:
   aa=np.array([float(a[k][col]) for k in common]);bb=np.array([float(b[k][col]) for k in common]);
   differences.append(dict(reference=ref,candidate=trial,source=file,field=col,max_abs=float(np.max(np.abs(aa-bb))),max_abs_reference=float(np.max(np.abs(aa)))))
 for step in ['0','1','2']:
  paths=[r/('output-'+c)/f'profiles/fault_{step}.csv' for c in [ref,trial]]
  if not all(p.exists() for p in paths):continue
  a,b=[list(csv.DictReader(p.open())) for p in paths]
  assert len(a)==len(b)
  for col in a[0]:
   aa=np.array([float(x[col]) for x in a]);bb=np.array([float(x[col]) for x in b]);differences.append(dict(reference=ref,candidate=trial,source=f'profile-step{step}',field=col,max_abs=float(np.max(np.abs(aa-bb))),max_abs_reference=float(np.max(np.abs(aa)))))
# Q2 visualization includes each cell's 3x3 support grid. Inspect duplicates
# across actual MPI piece boundaries and match coordinates against serial.
def vtk_fields(case):
 data={};cross={};shared=0
 for rank,p in enumerate(sorted((r/('output-'+case)/'solution').glob('solution-00001.*.vtu'))):
  reader=vtkXMLUnstructuredGridReader();reader.SetFileName(str(p));reader.Update();mesh=reader.GetOutput();xyz=vtk_to_numpy(mesh.GetPoints().GetData());pd=mesh.GetPointData();names=[pd.GetArrayName(k) for k in range(pd.GetNumberOfArrays())];arrays=[vtk_to_numpy(pd.GetArray(k)).reshape(len(xyz),-1) for k in range(len(names))]
  for i,point in enumerate(xyz):
   key=tuple(point);v={n:a[i] for n,a in zip(names,arrays)}
   if key in data and data[key][0]!=rank:
    shared+=1
    for n in names:cross[n]=max(cross.get(n,0),float(np.max(np.abs(data[key][1][n]-v[n]))))
   data[key]=(rank,v)
 return {k:v[1] for k,v in data.items()},shared,cross
a,_,_=vtk_fields('crossing-serial');b,shared,cross=vtk_fields('crossing-mpi');assert a.keys()==b.keys()
vtk_summary={'common_support_positions':len(a),'shared_piece_occurrences':shared,'shared_max_abs':cross,'serial_mpi_max_abs':{n:max(float(np.max(np.abs(a[k][n]-b[k][n]))) for k in a) for n in next(iter(a.values()))},'precision':'Float32 visualization, not full-precision DoF dumps'}
(o/'mpi_supports.json').write_text(json.dumps(vtk_summary,indent=2)+'\n')
for name,data in [('lifecycle_checks.csv',checks),('lifecycle_states.csv',states),('coupled_differences.csv',differences)]:
 with (o/name).open('w') as f:w=csv.DictWriter(f,lineterminator="\n",fieldnames=data[0].keys());w.writeheader();w.writerows(data)
# Preserve the observed H reproducer and its first failed proposal.
d=r/'output-coupled-random-5433';shutil.copyfile(d/'negative_H_cloud_rank0.csv',o/'negative_H_cloud.csv')
a=list(csv.DictReader((d/'insertion_rank0.csv').open()));unique={tuple(x[k] for k in ['step','cell','x','y','property']):x for x in a if x['property']=='0' and float(x['proposal'])<0}
with (o/'negative_H_proposals.csv').open('w') as f:w=csv.DictWriter(f,lineterminator="\n",fieldnames=next(iter(unique.values())).keys());w.writeheader();w.writerows(unique.values())
print('Failed exact checks:',[x for x in checks if not x['equal']]);print(vtk_summary)
