#!/usr/bin/env python3
"""Three actual-data figures separating seed geometry and driving geometry."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from assess_angular import RADII,SMOOTH_R,SEED_R,SEED_P,smooth_profile,smooth_force,smooth_weak
from compare_native import interpolate

plt.rcParams.update({'font.size':10,'axes.titlesize':11,'figure.constrained_layout.use':True})


def save(fig,path):
    for suffix in ['.png','.pdf']:fig.savefig(path.with_suffix(suffix),dpi=190)
    plt.close(fig)


def nodal(file,step,key):return file[f'{step}_{key}']


def arc(data,step,radius):
    a=data[(data['step']==step)&np.isclose(data['radius'],radius,rtol=0,atol=1e-14)]
    return np.sort(a,order='angle')


def seed_figure(out):
    a=np.genfromtxt(out/'wide-arcs.csv',names=True,delimiter=',');d=np.load(out/'wide-nodal.npz')
    summary=json.loads((out/'assessment.json').read_text());r=np.linspace(0,.38,1200);x=d['0_x'];y=d['0_y']
    fig,ax=plt.subplots(2,2,figsize=(12,8))
    ax[0,0].plot(r,smooth_profile(r),color='black',label='Smooth defining profile P(r)')
    ax[0,0].plot(r,np.interp(r,SEED_R,SEED_P,right=0),color='tab:blue',ls='--',label='Actual 5000-point table (overlaps)')
    for angle in [0,45,90]:
        t=np.deg2rad(angle);p=interpolate((x,y,d['0_phi']),.5+r*np.cos(t),.5+r*np.sin(t))
        ax[0,0].plot(r,p,lw=.8,label=f'Q1 ray θ={angle}°')
    ax[0,0].set(xlabel='Distance from seed [m]',ylabel='φ [1]',title='Compact cohesive seed; peak 0.6, no flat core');ax[0,0].legend(fontsize=8)
    band,cap=smooth_force(r)
    ax[0,1].plot(r,band,label='Smooth straight-band point force',color='tab:blue')
    ax[0,1].plot(r,cap,label='Smooth cap point force (forward limit)',color='tab:orange')
    ax[0,1].plot(r,interpolate((x,y,d['0_Fweak']),np.full_like(r,.25),.5+r),color='tab:blue',ls='--',label='Q1 weak band density')
    ax[0,1].plot(r,interpolate((x,y,d['0_Fweak']),.5+r,np.full_like(r,.5)),color='black',ls='--',label='Q1 weak centerline-cap density')
    ax[0,1].set(xlabel='Normal / radial distance [m]',ylabel='Fracture force / density [Pa]',title='Point force versus weak Q1 averages');ax[0,1].legend(fontsize=8)
    for aa in ax[0]:aa.axvline(SMOOTH_R[-1],color='.5',ls=':',lw=1)
    image=ax[1,0].pcolormesh(x,y,d['0_Fweak'],shading='gouraud',vmin=0,vmax=400,cmap='viridis')
    ax[1,0].contour(x,y,d['0_phi'],levels=[.001,.01,.1,.3,.5],colors='white',linewidths=.7)
    t=np.linspace(-np.pi/2,np.pi/2,361)
    ax[1,0].plot(.5+RADII[2]*np.cos(t),.5+RADII[2]*np.sin(t),color='cyan',lw=1)
    ax[1,0].plot([.25,.5],[.5,.5],color='white',lw=2);ax[1,0].plot(.5,.5,'wx',ms=7)
    ax[1,0].set(xlim=(.25,.87),ylim=(.15,.85),aspect='equal',xlabel='x [m]',ylabel='y [m]',title='Initial weak fracture density; phase contours white')
    fig.colorbar(image,ax=ax[1,0],label='g_f / nodal mass [Pa]',shrink=.8)
    q=arc(a,0,RADII[2]);ax[1,1].plot(q['angle'],q['F'],color='black',label='Actual Q1 seed / native weak test')
    ax[1,1].plot(q['angle'],q['F_smooth_weak'],color='tab:blue',label='Smooth seed / SAME weak test')
    ax[1,1].plot(q['angle'],q['F_point_cap'],color='.5',ls=':',label='Smooth cap point force')
    for n in summary['wide']['nodes']:
        if abs(n['radius']-RADII[2])<1e-12:ax[1,1].plot(n['angle'],n['F0'],'rx',ms=7)
    ax[1,1].set(xlabel='θ from +x [degrees]',ylabel='Fracture force / density [Pa]',title='r = 0.288111 m: angular variation before loading');ax[1,1].legend(fontsize=8)
    for aa in ax.flat:aa.grid(alpha=.15)
    fig.suptitle('The smooth cap is radial; compact support and grid-based weak averaging matter')
    save(fig,out/'seed-force')


def arcs_figure(out):
    a=np.genfromtxt(out/'wide-arcs.csv',names=True,delimiter=',');summary=json.loads((out/'assessment.json').read_text())
    fig,ax=plt.subplots(5,len(RADII),figsize=(16,12))
    for col,radius in enumerate(RADII):
        b=arc(a,0,radius);c=arc(a,54,radius)
        for row,(key,label) in enumerate([('phi','Physical Q1 φ [1]'),('qbar','Weak average q [Pa]'),('Ybar','Weak average Y [Pa]'),('F','Fracture density [Pa]'),('R','Total residual density [Pa]')]):
            aa=ax[row,col];aa.plot(b['angle'],b[key],color='tab:blue',ls='--',lw=1.1,label='Initialization')
            aa.plot(c['angle'],c[key],color='tab:orange',lw=1.2,label='Pre-onset: t=0.135 s')
            if key=='phi':aa.plot(b['angle'],b['phi_smooth'],color='.5',ls=':',lw=1,label='Smooth profile')
            if key=='F':
                aa.plot(b['angle'],b['F_smooth_weak'],color='.5',lw=.8,label='Smooth seed / same weak test')
                aa.plot(b['angle'],b['F_point_cap'],color='.5',ls=':',lw=.8,label='Smooth point force')
            for n in summary['wide']['nodes']:
                if abs(n['radius']-radius)<1e-12:aa.axvline(n['angle'],color='red',ls=':',lw=.7)
            if col==0:aa.set_ylabel(label)
            if row==0:aa.set_title(f'r = {radius:.6f} m')
            if row==4:aa.set_xlabel('θ from +x [degrees]')
            aa.set_xlim(-90,90);aa.grid(alpha=.2)
    handles,labels=ax[0,0].get_legend_handles_labels()
    handles.append(plt.Line2D([],[],color='.5',lw=.8));labels.append('Smooth seed / same weak test (fracture row)')
    fig.legend(handles,labels,loc='outside lower center',ncol=4)
    fig.suptitle('Forward-cap arcs: compatible weak averages; red guides mark actual first-growing nodes at t=0.1375 s')
    save(fig,out/'angular-arcs')


def driving_figure(out):
    d=np.load(out/'wide-nodal.npz');summary=json.loads((out/'assessment.json').read_text())
    fig,ax=plt.subplots(2,3,figsize=(14,9));qps=[np.load(out/f'wide-qps-{step}.npz') for step in [0,54,55]]
    selected=[(p['x']>=.42)&(p['x']<=.87)&(p['y']>=.15)&(p['y']<=.85) for p in qps]
    maximum=max(np.max(p['Y'][m]) for p,m in zip(qps,selected))
    for col,(step,p,mask) in enumerate(zip([0,54,55],qps,selected)):
        aa=ax[0,col];im=aa.scatter(p['x'][mask],p['y'][mask],c=p['Y'][mask],s=3,vmin=0,vmax=maximum,cmap='viridis',rasterized=True)
        contours=aa.contour(d[f'{step}_x'],d[f'{step}_y'],d[f'{step}_phi'],levels=[.001,.01,.1,.3,.5],colors='white',linewidths=.8)
        aa.plot([.42,.5],[.5,.5],color='white',lw=2)
        aa.plot(.5,.5,marker='o',markerfacecolor='white',markeredgecolor='red',ms=6)
        if step>0:
            for n in summary['wide']['nodes']:aa.plot(n['x'],n['y'],marker='x',c='red',ms=6)
        aa.set(xlim=(.42,.87),ylim=(.15,.85),aspect='equal',xlabel='x [m]',ylabel='y [m]',title=f'Actual Gauss Y, t={step*.0025:g} s')
    fig.colorbar(im,ax=list(ax[0,:]),label='Mechanical driving Y [Pa]',shrink=.8)
    for name,color,style in [('square','tab:orange','-'),('fine','black','--'),('wide','tab:green','-')]:
        a=np.genfromtxt(out/(name+'-arcs.csv'),names=True,delimiter=',');b=arc(a,0,RADII[2]);c=arc(a,54,RADII[2])
        for col,key,data in [(0,'F',b),(1,'Ybar',c),(2,'R',c)]:ax[1,col].plot(data['angle'],data[key],color=color,ls=style,label=name)
    for col,title in enumerate(['Initial fracture weak density','Driving at common time 0.135 s','Total weak density at common time 0.135 s']):
        ax[1,col].set(title=title,xlabel='θ from +x [degrees]',ylabel='Density [Pa]',xlim=(-90,90));ax[1,col].grid(alpha=.2);ax[1,col].legend()
    fig.suptitle('White φ contours (.001, .01, .1, .3, .5) show geometry; color shows mechanical driving Y\nRing: original endpoint. Red crosses: actual first-growing nodes (upcoming in precursor panel).')
    save(fig,out/'driving-geometry')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args()
    seed_figure(a.output);arcs_figure(a.output);driving_figure(a.output)
