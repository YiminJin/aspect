#!/usr/bin/env python3
"""Read-only native row margins and fixed connected-profile advance direction."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,numpy as np
R=Path(__file__).resolve().parents[4];O=R/'benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot';B=R/'build-particle-free/stage4-propagation-pilot'
H=1/32

def scan(p):
 files=sorted(p.glob('preload-scan-0000-r*.csv*'));assert len(files)==4
 a=np.concatenate([np.loadtxt(f,delimiter=',',skiprows=1) for f in files]);return a[np.lexsort((a[:,2],a[:,0],a[:,1]))]

def measures(a):
 b=a[:,3].reshape(128,256,9);shift=np.zeros_like(b);shift[:,1:]=b[:,:-1]
 chi=np.maximum(0,shift-b).ravel()*np.maximum(0,1-((a[:,0]-6.002054379936584)/.5)**2)**2
 mass=chi@a[:,4];net=(a[:,7]-a[:,5])/a[:,4];envelope=(a[:,8]-a[:,5])/a[:,4]
 intact=np.repeat(np.max(abs(b),axis=2).ravel()==0,9)
 masks=dict(right_tip=(abs(a[:,0]-6.00205)<.0625)&(abs(a[:,1]-2.00936)<.125),right_ahead_intact=intact&(a[:,0]>6)&(abs(a[:,1]-2)<.5),all_intact=intact,remote=(abs(a[:,0]-4)>3)|(abs(a[:,1]-2)>1))
 result=dict(profile_net=float(chi@(a[:,7]-a[:,5])/mass),profile_driving=float(chi@a[:,7]/mass),profile_cost=float(chi@a[:,5]/mass),regions={})
 for name,mask in masks.items():
  ids=np.flatnonzero(mask);i=ids[np.argmax(net[mask])];j=ids[np.argmax(envelope[mask])]
  result['regions'][name]=dict(max_net=float(net[i]),location=a[i,:3].tolist(),max_envelope=float(envelope[j]),envelope_location=a[j,:3].tolist(),positive_rows=int(sum(net[mask]>2e-7)))
 return result

if __name__=='__main__':
 for label in sys.argv[1:]:
  run=R/'build-particle-free/bulk-tip/runs'/('stage4-propagation-'+label)
  c=json.loads((run/'command.json').read_text());assert c['exit_code']==0
  a=scan(run/'output');q=measures(a);q['seconds']=c['seconds'];(O/(label+'-margins.json')).write_text(json.dumps(q,indent=2)+'\n');np.savez_compressed(B/(label+'-rows.npz'),rows=a);print(label,json.dumps(q,indent=2))
