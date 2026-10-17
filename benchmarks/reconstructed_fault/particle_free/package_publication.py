#!/usr/bin/env python3
"""Freeze compact replay inputs without modifying any native output or prior evidence."""
import argparse,hashlib,json,tarfile
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args()
root=Path(__file__).resolve().parents[3];runs=root/'build-particle-free/runs';diag=root/'build-particle-free/publication-diagnosis'
labels=['pf6a-publication-reproduce','pf6a-publication-controls512','pf6a-publication-history-controls512','pf6a-publication-cutback512','pf6a-publication-native512','pf6a-publication-terminal-controls512','pf6a-publication-verified-controls512','pf6a-publication-terminal-native512','pf6a-publication-batched-controls512','pf6a-publication-batched-native512','pf6a-publication-final-fe','pf6a-publication-final-dg','pf6a-publication-final-operators']
a.output.mkdir(parents=True,exist_ok=True);archive=a.output/'reference.tar.xz';assert not archive.exists()
files={}
def add(path,name):
 if path.is_file():files[name]=path
for label in labels:
 run=runs/label
 for name in ['input.prm','record.json','run.log','source.patch','observer-source.cc']:add(run/name,f'{label}/{name}')
 for path in (run/'run').glob('*.csv'):add(path,f'{label}/run/{path.name}')
 states=sorted((run/'run').glob('state-*'),key=lambda p:int(p.name.split('-')[-1]))
 # Metrics are written last: avoid treating a timeout-interrupted export as a
 # complete snapshot. Keep its existing raw files in the original run directory.
 import numpy as np
 metrics_path=run/'run/metrics.csv'
 metrics=np.atleast_1d(np.genfromtxt(metrics_path,delimiter=',',names=True)) if metrics_path.exists() else None
 complete={int(r['step'])for r in metrics} if metrics is not None and 'step' in metrics.dtype.names else set()
 for state in states:
  k=int(state.name.split('-')[-1])
  if k not in complete:continue
  names=['nodes.csv','cells.csv','constants.csv','profiles.csv','rule.csv','tensor.csv','surface.csv','accepted-surface.csv','gradient.csv','velocity.csv','native-loads.csv','mechanical.csv','clock.csv']
  if label not in ['pf6a-publication-native512','pf6a-publication-terminal-native512','pf6a-publication-batched-native512']:names+=['persistent.bin','rng.bin','manager.bin','shared.csv']
  if label=='pf6a-publication-batched-native512' and k in [1,68,max(complete)]:names+=['physical.csv']
  for name in names:add(state/name,f'{label}/run/{state.name}/{name}')
for path in diag.iterdir():
 if path.suffix in ['.json','.log','.prm']:add(path,'diagnosis/'+path.name)
for name in ['replay_publication.py','assess_publication_confirmation.py','plot_publication_confirmation.py','assess_pf6a.py','assess_phase_energy.py','assess_pf6a_reference.py','pf6a_spatial_reference.py','pf6a_checks.cc','run_pf6a_reference.py','run_fixed_profile.py','reference_checks.py','package_publication.py','verify_publication.py','diagnose_publication_phase.py']:add(Path(__file__).with_name(name),'source/benchmarks/reconstructed_fault/particle_free/'+name)
for path in (root/'benchmarks/reconstructed_fault/propagation').glob('*.py'):
 add(path,'source/'+str(path.relative_to(root)))
for name in ['pf6a-initial-phase.txt','CMakeLists.txt','fixed_profile_checks.cc','transport_checks.cc','assess_fixed_profile.py']:
 add(Path(__file__).with_name(name),'source/benchmarks/reconstructed_fault/particle_free/'+name)
asset=root/'benchmarks/reconstructed_fault/propagation/matched_initialization_evidence/reference.npz'
add(asset,'source/'+str(asset.relative_to(root)))
for case in ['p512-m512-d128-q12','p1024-mexact-d128-q12','p512-m512-d256-q12','p1024-mexact-d256-q12']:
 for name in ['fields.npz','history.json','summary.json']:
  add(root/'build-particle-free/pf6a-reference/continuum'/case/name,'reference/'+case+'/'+name)
for name in ['PF6A_PUBLICATION_REPORT.md']:add(root/'doc/reconstructed_fault/particle_free'/name,name)
manifest={name:hashlib.sha256(path.read_bytes()).hexdigest()for name,path in files.items()}
with tarfile.open(archive,'w:xz',preset=2) as out:
 for name,path in sorted(files.items()):out.add(path,arcname=name,recursive=False)
(a.output/'archive-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
records={l:json.loads((runs/l/'record.json').read_text())for l in labels}
(a.output/'provenance.json').write_text(json.dumps(dict(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),records=records),indent=2)+'\n')
print(json.dumps(dict(files=len(files),bytes=archive.stat().st_size,archive=str(archive))))
