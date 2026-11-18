#!/usr/bin/env python3
"""Classify held-displacement preparation and freeze its exact common input."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-preparation-mpl')
from pathlib import Path
import argparse,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from phase import coefficients,fronts,value,read,H
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];BUILD=ROOT/'build-particle-free/stage4-preparation'
OUT=HERE.parent/'evidence/stage4_preparation';OUT.mkdir(exist_ok=True)
p=argparse.ArgumentParser();p.add_argument('label',default='settle',nargs='?');p.add_argument('--freeze',action='store_true');a=p.parse_args()
run=ROOT/'build-particle-free/bulk-tip/runs'/('stage4-preparation-'+a.label);out=run/'output'
cmd=json.loads((run/'command.json').read_text());assert cmd['exit_code']==0
b=coefficients(out,0);raw=np.loadtxt(BUILD/'input-inherited/phase.txt',skiprows=1).reshape(128,256,3,3)
f=fronts(b,8);f4=fronts(b,4);f0=fronts(raw,8)
scans=sorted(out.glob('preload-scan-0000-r*.csv*'));assert len(scans)==4
s=np.concatenate([np.loadtxt(p,delimiter=',',skiprows=1) for p in scans]);s=s[np.lexsort((s[:,2],s[:,0],s[:,1]))]
net=(s[:,7]-s[:,5])/s[:,4];envelope=(s[:,8]-s[:,5])/s[:,4]
intact=np.repeat(np.max(abs(s[:,3].reshape(-1,9)),axis=1)==0,9)
x,y=s[:,:2].T
masks=dict(right_tip=(x==(int(f['right_x']/H)+.5)*H)&(abs(y-f['right_y'])<H),left_tip=(x==(int(f['left_x']/H)+.5)*H)&(abs(y-f['left_y'])<H),right_cap=(abs(x-f['right_x'])<.15625)&(abs(y-f['right_y'])<.125),left_cap=(abs(x-f['left_x'])<.15625)&(abs(y-f['left_y'])<.125),intact=intact,remote=(abs(x-4)>3)|(abs(y-2)>1))
margin={}
for name,mask in masks.items():
    margin[name]=dict(maximum_net_Pa=float(max(net[mask])),positive_rows=int(sum(net[mask]>2e-7)),maximum_envelope_net_Pa=float(max(envelope[mask])))
rates=read(out/'pilot-sections-0000.csv');before=read(ROOT/'build-particle-free/bulk-tip/runs/stage4-near-tip-initial/output/pilot-sections-0000.csv')
positions=[2.,5.5,5.75,5.9,6.,f['right_x']]
record=dict(command=cmd['arguments'],seconds=cmd['seconds'],inherited_front=f0,prepared_front=f,right_preparation_advance=f['right_x']-f0['right_x'],left_preparation_advance=f0['left_x']-f['left_x'],front_sampling_uncertainty=max(abs(f[k]-f4[k]) for k in ['left_x','right_x']),minimum_coefficient_increment=float(np.min(b-raw)),maximum_coefficient_increment=float(np.max(b-raw)),initially_intact_changed_cells=int(sum(np.max(b-raw,axis=(2,3))[np.max(raw,axis=(2,3))==0]>1e-12)),margins=margin,rate_samples={str(v):dict(old=float(np.interp(v,before['x'],before['rate_stage'])),prepared=float(np.interp(v,rates['x'],rates['rate_stage']))) for v in positions},static_checks={k:float(v) for k,v in zip(read(out/'preload-checks.csv').dtype.names,read(out/'preload-checks.csv')[0])})
np.savez_compressed(OUT/'prepared-phase.npz',inherited=raw,prepared=b)
(OUT/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')
with (OUT/'prepared-row-margins.npz').open('wb') as file:np.savez_compressed(file,locations=s[:,:4],net=net,envelope=envelope)
if a.freeze:
    assert record['minimum_coefficient_increment']>=-1e-14
    target=BUILD/'input-prepared';target.mkdir(exist_ok=False)
    for p in (BUILD/'input-inherited').iterdir():shutil.copy2(p,target/p.name)
    np.savetxt(target/'phase.txt',b.reshape(-1,9),header='256 128',comments='',fmt='%.17g')
    # Formed ownership is irreversible; include any newly nonzero sections.
    old=read(out/'pilot-sections-0000.csv')
    planes=np.loadtxt(target/'sections.txt',skiprows=1)
    occupied=np.max(b,axis=(0,2,3))>0
    planes[:,2]=np.maximum(planes[:,2],occupied[np.minimum((planes[:,0]/H).astype(int),255)])
    np.savetxt(target/'sections.txt',planes,header=str(len(planes)),comments='',fmt=['%.17g','%.17g','%d'])
fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
for side,ax in [('left',axs[0]),('right',axs[1])]:
    center=f0[side+'_x'];xx=np.linspace(center-.3,center+.3,1601)
    ax.plot(xx,value(raw,xx,2.),label='Inherited φ');ax.plot(xx,value(b,xx,2.),label='Prepared φ')
    twin=ax.twinx();twin.plot(xx,value(b,xx,2.)-value(raw,xx,2.),color='tab:green',label='Preparation Δφ, right axis');twin.set_ylabel('Δφ',color='tab:green');twin.tick_params(axis='y',colors='tab:green');twin.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    ax.axhline(.05,color='k',ls=':',label='Absolute .05');ax.set(xlabel='x (m)',ylabel='Phase',title=side+' original tip')
    handles,labels=ax.get_legend_handles_labels();extra,names=twin.get_legend_handles_labels();ax.legend(handles+extra,labels+names,fontsize=7)
axs[2].plot(before['x'],before['rate_stage']*1e3,label='Previous, perturbed');axs[2].plot(rates['x'],rates['rate_stage']*1e3,label='Prepared, unperturbed');axs[2].axvspan(5.5,6,color='gray',alpha=.15);axs[2].set(xlim=(5.4,6.2),xlabel='x (m)',ylabel='V (mm/s)',title='Actual initial right-taper sliding');axs[2].legend(fontsize=8)
for ext in ('png','pdf'):fig.savefig(OUT/f'preparation.{ext}',dpi=170)
print(json.dumps(record,indent=2))
