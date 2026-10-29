#!/usr/bin/env python3
"""Publication artifact for the bounded prescribed-transport comparison."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from assess_methods import Data,exact

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dg',type=Path,required=True)
    p.add_argument('--ev',type=Path,required=True)
    p.add_argument('--supg',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    x=np.linspace(0,1,10001); y=np.full_like(x,.5)
    fig,axes=plt.subplots(1,2,figsize=(11,3.8))
    for ax,case,title in zip(axes,['signed_jump','finite_width'],['Sharp signed pulse','Fixed finite-width pulse']):
        ax.plot(x,exact(case,x,y,.2,32)[:,0]/1000,'k-',lw=1.4,label='Analytic')
        dg=Data(a.dg)
        projection=dg.project(case,.2,'DG')
        ax.plot(x,dg.evaluate(projection,x,y)[:,0]/1000,color='.55',ls=':',lw=1,label='Exact DG L2 projection')
        for directory,label,legend in [(a.dg,case+'-unlimited','Unlimited DGQ2'),(a.dg,case+'-scaling','Scaling reference'),(a.ev,case,'Native scalar EV'),(a.supg,case,'Native scalar SUPG')]:
            d=Data(directory)
            ax.plot(x,d.evaluate(d.load('field-'+label),x,y)[:,0]/1000,lw=1,label=legend)
        ax.set(xlabel='x (m)',ylabel='xx / 1000 Pa',title=title,xlim=(.15,.98))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7,loc='lower left')
    fig.suptitle('Prescribed transport, 32 x 64 cells, t = 0.2 s; no coupled evolution',fontsize=10)
    fig.tight_layout()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output,dpi=180)
if __name__=='__main__': main()
