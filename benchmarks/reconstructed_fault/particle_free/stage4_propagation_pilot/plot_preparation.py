#!/usr/bin/env python3
"""Selection evidence for the single propagation preparation."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-propagation-pilot-mpl')
from pathlib import Path
import sys,json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from screen import R,B,O,scan
sys.path.insert(0,str(Path(__file__).resolve().parent))
from phase import coefficients,value,read
initial=np.loadtxt(B/'input/phase.txt',skiprows=1).reshape(128,256,3,3)
b=np.loadtxt(B/'input-candidate/phase.txt',skiprows=1).reshape(128,256,3,3)
out=R/'build-particle-free/bulk-tip/runs/stage4-propagation-candidate-settle/output'
a=scan(out);net=(a[:,7]-a[:,5])/a[:,4]
f=json.loads((O/'candidate.json').read_text());cal=json.loads((O/'calibration.json').read_text())
fig,axs=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
for side,ax in [('left',axs[0,0]),('right',axs[0,1])]:
 center=f['inherited_front'][side+'_x'];y=f['inherited_front'][side+'_y'];x=np.linspace(center-.3,center+.3,1601)
 ax.plot(x,value(initial,x,y),color='k',label='Inherited φ');ax.plot(x,value(b,x,y),color='tab:red',label='Prepared φ');ax.axhline(.05,color='gray',ls=':');twin=ax.twinx();twin.plot(x,value(b,x,y)-value(initial,x,y),color='tab:green',ls='--');twin.set_ylabel('Preparation Δφ',color='tab:green');ax.set(xlabel='x (m)',ylabel='φ',title=side+' prepared cap');ax.legend(fontsize=8)
reg=(abs(a[:,0]-6)<.4)&(abs(a[:,1]-2)<.3);sc=axs[1,0].scatter(a[reg,0]+(a[reg,2]%3-1)/64,a[reg,1]+(a[reg,2]//3-1)/64,c=net[reg],s=6,cmap='coolwarm',vmin=-2,vmax=2);fig.colorbar(sc,ax=axs[1,0],label='Native row net driving (Pa)');axs[1,0].set(xlabel='x (m)',ylabel='y (m)',title='Prepared right-tip neighborhood')
x=np.array(cal['response_amplitudes']);ys=cal['native_response']
axs[1,1].plot(x,[q['profile_net'] for q in ys],'o-',label='Broad connected-profile direction');axs[1,1].plot(x,[q['regions']['all_intact']['max_envelope'] for q in ys],'o-',label='Strongest intact envelope');axs[1,1].axhline(0,color='k',lw=.6);axs[1,1].axvline(.00024,color='gray',ls='--',label='Candidate before settlement');axs[1,1].set(xlabel='Added shear at fixed normal preload',ylabel='Mean / row net driving (Pa)',title='Native fixed-phase estimates');axs[1,1].legend(fontsize=7)
for ext in ['png','pdf']:fig.savefig(O/('preparation.'+ext),dpi=180)
