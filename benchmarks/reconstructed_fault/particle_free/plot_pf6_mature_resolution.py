#!/usr/bin/env python3
"""Plot measured short-horizon backend differences; no accuracy truth is implied."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_pf6_mature import read
from analyze_pf6_mature_resolution import L,TAU0,SIGMA0,VP


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('summary',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    summary=json.loads(a.summary.read_text());fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    colors={'cc':'#ba372d','fc':'#25814d','ff':'#285daa'}
    for backend,style in [('dg','-'),('particle','--')]:
        for kind in ['cc','fc','ff']:
            label=backend+'-'+kind;folder=a.directory/('dg-fc-final' if label=='dg-fc' else label)/'run'
            final=read(folder/'coupled-profile-3.csv');s=final['s_m'];color=colors[kind]
            for ax,key,bg,scale in [(axes[0,0],'V_m_s',VP,VP/100),
                                     (axes[0,1],'theta_s',8e6,1e3),
                                     (axes[1,0],'raw_normal_Pa',SIGMA0,1e3),
                                     (axes[1,1],'shear_Pa',summary['tau0_Pa'],1e3)]:
                ax.plot(s,(final[key]-bg)/scale,style,color=color,lw=1.25,label=label.upper())
    for ax,ylabel in zip(axes.flat,['(V − Vp) / Vp, %','Theta − steady state, 1000 s','Raw normal projection − 50 MPa, kPa','Shear projection − background, kPa']):
        ax.axvspan(0,125,color='.92',zorder=-2);ax.axvspan(L-125,L,color='.92',zorder=-2)
        ax.set(xlabel='Down-dip distance, m',ylabel=ylabel,xlim=(0,L));ax.grid(alpha=.2)
    axes[0,0].legend(ncol=2,fontsize=8)
    fig.suptitle('Mature stationary fault, T = 600000 s, Gauss6\nCC: coarse/coarse; FC: fine mechanics/coarse phase; FF: fine/fine')
    fig.savefig(a.output)
    fig.savefig(a.output.with_suffix(".png"),dpi=150)
if __name__=='__main__':main()
