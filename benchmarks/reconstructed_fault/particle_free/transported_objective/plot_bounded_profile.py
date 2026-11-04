#!/usr/bin/env python3
"""Compact coupled profile/error and energy/cost figures from measured outputs."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
import json
import numpy as np
from compare_bounded_profile import OUT,load
os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

def main():
    rows=json.loads((OUT/'comparison.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    energy,ea=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for col,c in enumerate(['c','l']):
        selected=[r for r in rows if r['arguments']['speed']>0 and (r['arguments']['eta']=='1')==(c=='c')]
        ref=load(selected[0]['reference_case'],final_only=True)[3];ref=ref[max(ref)]
        X=ref.surface[:,1:3];center=.5*(X[0]+X[-1]);t=X[-1]-X[0];t/=np.linalg.norm(t);n=np.array([-t[1],t[0]])
        line=np.linspace(-.65,.65,400);points=center+line[:,None]*n
        axes[0,col].plot(line,ref.at(points)[:,0],'k',lw=2,label='64² bounded reference')
        for method,color in [('Q1 remap','#d55e00'),('bounded MacCormack','#0072b2')]:
            for level,dt,style in [(4,.1,':'),(5,.05,'-')]:
                r=next(r for r in selected if r['arguments']['profile_method']==method and r['arguments']['level']==level and r['arguments']['dt']==dt)
                s=load(r['case'],final_only=True)[3];s=s[max(s)]
                axes[0,col].plot(line,s.at(points)[:,0],style,color=color,label=f'{method}, {2**level}²')
            joint=sorted([r for r in selected if r['arguments']['profile_method']==method and (r['arguments']['level'],r['arguments']['dt']) in [(4,.1),(5,.05)]],key=lambda r:r['arguments']['level'])
            h=[4/2**r['arguments']['level'] for r in joint]
            for metric,style,mark in [('profile_relative_L2','-','o'),('stress_relative_L2','--','s')]:
                axes[1,col].loglog(h,[100*r['fine_reference'][metric] for r in joint],style+mark,color=color,label=f'{method}: {metric.split("_")[0]}')
            times=sorted([r for r in selected if r['arguments']['profile_method']==method and r['arguments']['level']==5],key=lambda r:r['arguments']['dt'])
            ea[0,col].plot([r['arguments']['dt'] for r in times],[100*r['endpoint_relative'] for r in times],'o-',color=color,label=method+' endpoint')
            ea[0,col].plot([r['arguments']['dt'] for r in times],[100*r['trapezoid_relative'] for r in times],'s--',color=color,label=method+' trapezoid')
            offset=-.15 if method=='Q1 remap' else .15
            ea[1,col].bar(np.arange(len(joint))+offset,[r['wall_s'] for r in joint],width=.3,color=color,label=method)
        title='Constant viscosity' if c=='c' else 'Layered viscosity'
        axes[0,col].set(title=title,xlabel='Normal distance from reference curve (m)',ylabel='Physical phase at t=0.2 s')
        axes[1,col].set(xlabel='Cell width h (m), dt=0.4 h',ylabel='Relative L2 error to finer reference (%)',xticks=[.125,.25],xticklabels=['0.125','0.25'])
        ea[0,col].set(title=title,xlabel='Timestep (s), 32² cells',ylabel='Energy remainder / active work (%)')
        ea[1,col].set(xticks=[0,1],xticklabels=['16², dt=.1','32², dt=.05'],ylabel='Wall time (s), two MPI ranks')
        axes[1,col].xaxis.set_minor_formatter(NullFormatter())
        for ax in [axes[0,col],axes[1,col],ea[0,col],ea[1,col]]:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(OUT/'coupled-profile-comparison.png',dpi=180)
    energy.savefig(OUT/'coupled-energy-cost.png',dpi=180)
    plt.close('all')

if __name__=='__main__':main()
