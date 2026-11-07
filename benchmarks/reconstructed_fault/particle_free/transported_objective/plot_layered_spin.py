#!/usr/bin/env python3
"""Actual localization-weighted spin and stress response, with material-label traces."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-layered-mpl')
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from assess_layered_campaign import OUT

def main():
 p=argparse.ArgumentParser();p.add_argument('cases',nargs='+');a=p.parse_args()
 fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
 fig.suptitle('Total-spin law uses omega; bulk-spin law uses omega_b = omega - omega_f',fontsize=11)
 for index,name in enumerate(a.cases):
  color=f'C{index%10}'
  c=json.loads((OUT/(name+'.json')).read_text());rs=c['records'];t=[r['time'] for r in rs];tag=name.replace('layered-bulk-','')
  sd=[r['spin_diagnostics'] for r in rs];middle=len(sd[0]['label_chi'])//2
  axes[0,0].plot(t,[d['sampled_max_envelope'] for d in sd],color=color,label=tag+' envelope')
  axes[0,0].plot(t,[abs(d['fault_spin_exposure'][middle]) for d in sd],'--',color=color,label=tag+' middle label')
  for key,style in [('label_omega','-'),('label_omega_f','--'),('label_omega_b',':')]:
   axes[0,1].plot(t,[d[key][middle] for d in sd],style,color=color,label=tag+' '+key.removeprefix('label_'))
  axes[0,2].plot(t,[d['stress_weighted_tendency'][middle] for d in sd],color=color,label=tag)
  axes[1,0].plot(t,[d['label_stress'][middle][0] for d in sd],color=color,label=tag+' xx')
  axes[1,0].plot(t,[d['label_stress'][middle][2] for d in sd],'--',color=color,label=tag+' xy')
  angle=np.array([d['principal_angle'][middle] for d in sd]);gap=np.array([d['principal_gap'][middle] for d in sd])
  angle[gap<1e-12*max(1.,gap.max())]=np.nan
  axes[1,1].plot(t,angle*180/np.pi,color=color,label=tag)
  axes[1,2].plot(t,[r['energy']['sampled_max_chi'] for r in rs],color=color,label=tag)
 titles=['Slip-spin exposure (rad)','Middle-label spin rates (rad/s)','Middle-label commutator norm (Pa/s)',
         'Middle-label stress (Pa)','In-plane principal direction (degrees)\nzero stress gap omitted','Sampled maximum chi (1/m)']
 for ax,title in zip(axes.flat,titles):
  ax.set(title=title,xlabel='time (s)');ax.grid(alpha=.2);ax.legend(fontsize=6)
 fig.savefig(OUT/'layered-spin.png',dpi=160);fig.savefig(OUT/'layered-spin.pdf')
if __name__=='__main__':main()
