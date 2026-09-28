"""Unsmoothed endpoint/interior growth on accepted restored-BP3 profiles.

Raw sigma_Q1 is a consistent projection for observation, NOT the raw friction
input. The filtered column is the actual represented friction input. Keep the
native row diagnostic alongside both in the original CSV files.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('output',type=Path);args=p.parse_args()
files=sorted(args.output.glob('restored_fault_*.csv'),key=lambda x:int(x.stem.split('_')[-1]))
assert files, 'No accepted restored-BP3 profiles'
rows=[];fig,axes=plt.subplots(3,3,figsize=(15,10))
for path in files:
    d=np.genfromtxt(path,delimiter=',',names=True);step=int(path.stem.split('_')[-1]);s=d['s_m']
    length=max(s);time=float(d['time_s'][0])
    regions={'top':s<=1000.,'transition':(s>=13000)&(s<=20000),'bottom':s>=length-1000.}
    for col,(region,mask) in enumerate(regions.items()):
        order=np.argsort(s[mask]);x=s[mask][order]/1000
        record={'step':step,'time_s':time,'region':region}
        for field in ('V','raw_sigma_Q1','friction_sigma_Q1','V_chord','raw_chord','filtered_chord'):
            v=d[field][mask]
            record[field+'_mean']=float(np.mean(v));record[field+'_ptp']=float(np.ptp(v))
            record[field+'_rms_centered']=float(np.std(v))
            record[field+'_max_abs']=float(max(abs(v)))
            mass=d['work_mass'][mask]
            mean=float(np.sum(mass*v)/sum(mass))
            record[field+'_work_mean']=mean
            record[field+'_work_rms_centered']=float(np.sqrt(np.sum(mass*(v-mean)**2)/sum(mass)))
        rows.append(record)
        for row,(field,scale) in enumerate([('V',1e-9),('raw_sigma_Q1',1e6),('friction_sigma_Q1',1e6)]):
            y=d[field][mask][order]/scale
            if row:y-=50.
            axes[row,col].plot(x,y,label=f'{step}: {time:.6g} s',lw=.8)
            axes[row,col].set_title(region);axes[row,col].set_xlabel('down-dip km')
            axes[row,col].set_ylabel(['V/Vp','raw normal - 50 MPa','friction normal - 50 MPa'][row])
            for marker in (15,18,40):
                if x[0]<=marker<=x[-1]:axes[row,col].axvline(marker,color='grey',ls=':',lw=.6)
axes[0,0].legend(fontsize=6)
fig.tight_layout();fig.savefig(args.output/'restored_endpoint_growth.png',dpi=180)
with (args.output/'restored_region_growth.csv').open('w') as out:
    w=csv.DictWriter(out,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print('Saved unsmoothed spatial and work-weighted shape/growth diagnostics.')
