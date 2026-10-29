#!/usr/bin/env python3
"""Fixed-scale physical fields, phase geometry, work and contour comparison."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,FFMpegWriter,PillowWriter
from matplotlib.colors import Normalize
from assess_native import csv
from assess_spatial import q1
from assess_continued import LEVELS
from assess_boundary import NEW,OLD,EVIDENCE
from recover_remote import load_layout,reconstruct,MAP


def prepare(out):
    m=dict(np.load(out/'mixed-fields.npz'));old=dict(np.load(EVIDENCE/'fields.npz'))
    L=load_layout(MAP);centers=L['points'].mean(axis=1)
    ix=np.rint(centers[:,0]/L['dx']-.5).astype(int);iy=np.rint(centers[:,1]/L['dy']-.5).astype(int)
    shape=(len(L['y'])-1,len(L['x'])-1);oldstress=[];oldrate=[];oldY=[]
    for n in range(120,149):
        _,tau,Y,h,_=reconstruct(L,n)
        def grid(a):
            g=np.empty(shape);g[iy,ix]=np.sum(L['weights']*a,axis=1)/L['weights'].sum(axis=1);return g
        oldstress.append(grid(tau[:,:,2]));oldrate.append(grid(abs(h[:,:,4])));oldY.append(grid(Y))
    stress=[];rate=[];drive=[]
    for n in range(149):
        a=np.loadtxt(NEW/f'means-{n:04d}.csv',delimiter=',',skiprows=1,usecols=(1,2,4,5,6))
        ii=np.rint(a[:,0]/L['dx']-.5).astype(int);jj=np.rint(a[:,1]/L['dy']-.5).astype(int)
        for target,col in [(stress,2),(drive,3),(rate,4)]:
            g=np.empty(shape);g[jj,ii]=a[:,col];target.append(g)
    np.savez_compressed(out/'movie-data.npz',old_stress=oldstress,old_rate=oldrate,old_Y=oldY,
        stress=stress,rate=rate,Y=drive,old_phase=old['phase'],old_grid_ids=old['grid_ids'],**m)


def main(out,movies):
    if not (out/'movie-data.npz').exists():prepare(out)
    d=dict(np.load(out/'movie-data.npz'));assessment=json.load(open(out/'boundary-assessment.json'))
    mh=json.load(open(out/'mixed-history.json'));oh=json.load(open(EVIDENCE/'history.json'))
    mc=json.load(open(out/'mixed-contours.json'));oc=json.load(open(EVIDENCE/'contours.json'))
    me=csv(out/'mixed-energy.csv');oe=csv(EVIDENCE/'energy.csv')
    for n in range(149):
        if str(n) not in oc:
            assert np.array_equal(d['old_phase'][n],d['old_phase'][0])
            oc[str(n)]=oc['0']
    L={k:d[k] for k in ['x','y','grid_ids']};L.update(dx=L['x'][1]-L['x'][0],dy=L['y'][1]-L['y'][0]);OL={**L,'grid_ids':d['old_grid_ids']}
    X,Y=np.meshgrid(np.linspace(0,1.5,385),np.linspace(0,1,257));xy=np.stack([X,Y],axis=-1)
    initial=q1(L,d['phase'][0],xy)[0];times=d['time']
    normstress=Normalize(min(0.,float(d['stress'].min()),float(d['old_stress'].min())),max(float(d['stress'].max()),float(d['old_stress'].max())))
    normdamage=Normalize(0,max(float((d['phase']-d['phase'][0]).max()),float((d['old_phase']-d['old_phase'][0]).max())))
    normrate=Normalize(0,max(float(d['rate'].max()),float(d['old_rate'].max())))
    normY=Normalize(0,max(float(d['Y'].max()),float(d['old_Y'].max())))
    labels=[r'Physical $\tau_{xy}$ [Pa]',r'Phase $\phi$ [1]',r'Damage $\phi-\phi_0$ [1]',r'Cell mean $|w|$ [s$^{-1}$]']
    norms=[normstress,Normalize(0,.6),normdamage,normrate]
    def phase(case,n):return q1(OL if case==0 else L,d['old_phase' if case==0 else 'phase'][n],xy)[0]
    def values(case,n):
        p=phase(case,n)
        return [d['old_stress' if case==0 else 'stress'][n-120 if case==0 else n],p,p-initial,d['old_rate' if case==0 else 'rate'][n-120 if case==0 else n]]
    def marks(ax,p,case,n):
        cs=[]
        cs.append(ax.contour(X,Y,p,levels=LEVELS,colors='cyan',linewidths=.6))
        activated=(initial==0)&(p>1e-12)
        if activated.any():cs.append(ax.contour(X,Y,activated.astype(float),levels=[.5],colors='magenta',linewidths=.6))
        return cs
    def base_marks(ax):
        ax.contour(X,Y,initial,levels=LEVELS,colors='white',linestyles='--',linewidths=.45)
        ax.contour(X,Y,np.hypot(np.maximum(X-.5,0),Y-.5),levels=[.30882159],colors='lime',linestyles=':',linewidths=.7)
        ax.plot([0,.5],[.5,.5],color='red',lw=.8);ax.plot(.5,.5,'rx',ms=4)
        ax.set(xlim=(0,1.5),ylim=(0,1),aspect='equal',xlabel='x [m]',ylabel='y [m]')
    # Whole-domain, same-scale static comparison at loading stop.
    fig,axes=plt.subplots(2,3,figsize=(13,7.6),layout='constrained')
    for row in range(2):
        p=phase(row,140)
        for col,(a,norm,title) in enumerate([(values(row,140)[0],normstress,'Physical tau_xy [Pa]'),(p-initial,normdamage,'Damage phi - phi0 [1]'),(d['old_Y'][20] if row==0 else d['Y'][140],normY,'Native phase driving Y [Pa]')]):
            ax=axes[row,col]
            art=ax.pcolormesh(L['x'],L['y'],a,norm=norm,cmap='magma' if col==2 else 'viridis',rasterized=True) if col!=1 else ax.pcolormesh(X,Y,a,norm=norm,cmap='viridis',rasterized=True)
            base_marks(ax);marks(ax,p,row,140)
            if row==0:ax.set_title(title)
            if col==0:ax.set_ylabel(('All-boundary affine' if row==0 else 'Lateral ux released')+'\ny [m]')
            if row==1:fig.colorbar(art,ax=axes[:,col],location='bottom',shrink=.8,pad=.04)
    fig.suptitle('Same specimen and loading at 0.175 s: boundary-condition control\n'+r'$\Delta x=0.015625$ m, $\Delta y=0.0078125$ m; $\Delta t=0.00125$ s',fontsize=12)
    fig.supxlabel('White dashed: original phase; cyan: current phase; green dotted: continuum support; magenta: new damage in initially intact regions.\nRed: original seed/endpoint. Stress and Y are Gauss8-weighted cell means. Spatial accuracy and extended-interval temporal uncertainty remain unresolved.',fontsize=9)
    fig.savefig(out/'boundary-comparison.png',dpi=160);fig.savefig(out/'boundary-comparison.pdf');plt.close(fig)
    # Histories and energy: separate actual external-work measures.
    fig,axes=plt.subplots(2,3,figsize=(14,8.5),layout='constrained')
    height=1.0 # m; Q_gamma/height is conjugate to relative grip displacement.
    gamma=height*np.cumsum([r['strain_increment'] for r in mh]);oldreaction=d['old_reactions'];Q=np.array([r['Q'] for r in mh])
    axes[0,0].plot(gamma,oldreaction[:,0]/height,label='Affine: total prescribed reaction',color='tab:blue');axes[0,0].plot(gamma,oldreaction[:,1]/height,label='Affine: grip reaction',color='tab:blue',ls=':');axes[0,0].plot(gamma,Q/height,label='Released: grip reaction',color='tab:orange');axes[0,0].set(xlabel='Relative grip displacement [m]',ylabel='Generalized reaction [N/m]',title='Same prescribed grip motion');axes[0,0].legend(fontsize=7)
    oldE=np.array([r['bulk']+r['cohesive']+r['fracture']-oh[0]['fracture'] for r in oh]);newE=np.r_[0.,me['storage_change']]
    oldD=np.r_[0.,np.cumsum(oe['physical_dissipation'])];newD=np.r_[0.,np.cumsum(me['physical_dissipation'])]
    oldW=np.r_[0.,np.cumsum(oe['trapezoidal_work'])];newW=np.r_[0.,np.cumsum(me['trapezoidal_work'])]
    for color,E,D,W,case in [('tab:blue',oldE,oldD,oldW,'Affine'),('tab:orange',newE,newD,newW,'Released')]:
        axes[0,1].plot(times,W,color=color,label=case+' work');axes[0,1].plot(times,E+D,color=color,ls='--',label=case+' storage + dissipation')
    axes[0,1].set(xlabel='Time [s]',ylabel='Energy [J/m]',title='Trapezoidal external work');axes[0,1].legend(fontsize=7)
    for color,h,case in [('tab:blue',oh,'Affine'),('tab:orange',mh,'Released')]:
        cap=[r['cap_growth'] if case=='Affine' else r['integrals']['cap'] for r in h];intact=[r['continuum_intact_growth'] if case=='Affine' else r['integrals']['initially_intact'] for r in h]
        axes[0,2].plot(times,cap,color=color,label=case+' cap');axes[0,2].plot(times,intact,color=color,ls='--',label=case+' initially intact')
    axes[0,2].set(xlabel='Time [s]',ylabel='Integrated phase increment [m2]',title='Cap growth versus new damage');axes[0,2].legend(fontsize=7)
    for color,cont,case in [('tab:blue',oc,'Affine'),('tab:orange',mc,'Released')]:
        for level,ls in [('0.001','-'),('0.01','--')]:
            axes[1,0].plot(times,[1000*(cont[str(n)][level]['centerline']-cont['0'][level]['centerline']) if cont[str(n)][level]['centerline']<.9 else np.nan for n in range(149)],color=color,ls=ls,label=case+' center '+level)
            axes[1,0].plot(times,[1000*(cont[str(n)][level]['upper']['ray45_radius']-cont['0'][level]['upper']['ray45_radius']) for n in range(149)],color=color,ls=ls,lw=.9,alpha=.4)
    axes[1,0].set(xlabel='Time [s]',ylabel='First-crossing advance [mm]',title='Near cap: center (thick), upper 45 deg (thin)');axes[1,0].legend(fontsize=7)
    for color,e,case in [('tab:blue',oe,'Affine'),('tab:orange',me,'Released')]:
        axes[1,1].plot(times[1:],np.cumsum(e['endpoint_remainder']),color=color,label=case+' BE endpoint');axes[1,1].plot(times[1:],np.cumsum(abs(e['trapezoidal_remainder'])),color=color,ls='--',label=case+' sum |trapezoidal defect|')
    axes[1,1].set_yscale('symlog',linthresh=1e-6);axes[1,1].set(xlabel='Time [s]',ylabel='Remainder [J/m]',title='BE accounting and physical work are separate');axes[1,1].legend(fontsize=7)
    for color,h,case in [('tab:blue',oh,'Affine'),('tab:orange',mh,'Released')]:
        cap=[r['cap_peak'] if case=='Affine' else r['peaks']['cap'] for r in h];remote=[r['remote_peak'] if case=='Affine' else r['peaks']['remote'] for r in h]
        axes[1,2].plot(times,cap,color=color,label=case+' cap');axes[1,2].plot(times,remote,color=color,ls='--',label=case+' remote (x>.9)')
    axes[1,2].set(xlabel='Time [s]',ylabel='Peak phase increment [1]',title='Remote activation kept distinct');axes[1,2].legend(fontsize=7)
    for ax in axes.flat:
        ax.grid(alpha=.25)
        if ax!=axes[0,0]:ax.axvline(.175,color='.5',ls=':',lw=.8)
    fig.supxlabel('Grip displacement equals engineering shear strain numerically because height is 1 m. Corner reactions are assigned to grips in the nodal work decomposition.\nCenter curves stop at remote joining (.155/.1675 s for released phi=.001/.01); domain-spanning connections are not tip travel. No growth speed or spatial convergence is claimed.',fontsize=9)
    fig.savefig(out/'boundary-histories.png',dpi=160);fig.savefig(out/'boundary-histories.pdf');plt.close(fig)
    # Common kernel profiles, before/during remote response.
    remote=dict(np.load(out/'remote-fields.npz'));fig,axes=plt.subplots(3,3,figsize=(13,10),layout='constrained')
    sl=[slice(0,193),slice(193,314),slice(314,None)]
    for row,name in enumerate(['Centerline','Upper 45 deg','Lower 45 deg']):
        xx=d['xx'] if row==0 else d['rr']
        for n,color in zip([120,124,140],['tab:blue','tab:orange','tab:green']):
            oldp=remote['profiles'][list(remote['steps']).index(n),sl[row]];newp=d['profiles'][list(d['profile_steps']).index(n),sl[row]]
            for col in [0,1]:
                axes[row,col].plot(xx,oldp[:,col],color=color,ls='--',label=f'Affine {times[n]:.3f} s');axes[row,col].plot(xx,newp[:,col],color=color,label=f'Released {times[n]:.3f} s')
            axes[row,2].plot(xx,oldp[:,2]-oldp[:,3],color=color,ls='--');axes[row,2].plot(xx,newp[:,2]-newp[:,3],color=color)
        for col,label in enumerate(['Physical tau_xy [Pa]','Local Y [Pa]','Weak driving - resistance [Pa]']):
            axes[row,col].set(xlabel='x [m]' if row==0 else 'Radius from original endpoint [m]',ylabel=name+'\n'+label);axes[row,col].grid(alpha=.25)
    axes[0,0].legend(fontsize=7);fig.suptitle('Identical physical compact averaging kernel, radius .0625 m')
    fig.supxlabel('Stress/Y: original Gauss8 points with positive JxW. Weak forces: native nodal integrals with matching mass.\nNo constitutive history is interpolated into a new physical state. Solid = horizontal-side release; dashed = retained affine loading.',fontsize=9)
    fig.savefig(out/'boundary-profiles.png',dpi=160);fig.savefig(out/'boundary-profiles.pdf');plt.close(fig)
    if not movies:return
    # Native means give a full NEW trajectory; the direct stress comparison is
    # restricted to the saved affine full-history window, .15-.185 s.
    def movie(cases,frames,name):
        fig,axes=plt.subplots(len(cases),4,figsize=(16,4*len(cases)),squeeze=False)
        fig.subplots_adjust(left=.05,right=.98,bottom=.15,top=.88,hspace=.42,wspace=.3)
        arts=[];dynamic=[]
        for i,case in enumerate(cases):
            n=frames[0];p=phase(case,n);vals=values(case,n)
            for col,ax in enumerate(axes[i]):
                art=ax.pcolormesh(L['x'],L['y'],vals[col],norm=norms[col],cmap='magma' if col==3 else 'viridis') if col in [0,3] else ax.pcolormesh(X,Y,vals[col],norm=norms[col],cmap='viridis')
                base_marks(ax);dynamic.extend(marks(ax,p,case,n));arts.append(art)
                if i==0:ax.set_title(labels[col],fontsize=10)
                if col==0:ax.set_ylabel(('All-boundary affine' if case==0 else 'Lateral ux free; uy=0')+'\ny [m]')
                if i==len(cases)-1:fig.colorbar(art,ax=axes[:,col],location='bottom',pad=.1,shrink=.8,fraction=.04)
        title=fig.suptitle('');fig.text(.5,.02,'Same physical axes and fixed colour scales. Stress/rate are original-Gauss8-weighted cell means; w is bulk slip strain rate, not surface velocity.\nWhite dashed: initial phase; cyan: current; green dotted: original support; magenta: initially intact damage. Red: original seed/end.\nPhysical accuracy and growth speed remain unqualified.',ha='center',fontsize=9)
        def draw(n):
            nonlocal dynamic
            for c in dynamic:c.remove()
            dynamic=[]
            for i,case in enumerate(cases):
                p=phase(case,n)
                for col,a in enumerate(values(case,n)):arts[4*i+col].set_array(a.ravel());dynamic.extend(marks(axes[i,col],p,case,n))
            title.set_text(f'Boundary-loading control: t={times[n]:.5f} s; '+r'$\Delta t=0.00125$ s; '+('loading' if times[n]<=.175+1e-12 else 'fixed-grip hold'))
            return arts
        ani=FuncAnimation(fig,draw,frames=frames,interval=125,blit=False)
        ani.save(out/(name+'.mp4'),writer=FFMpegWriter(fps=8,codec='libx264',extra_args=['-pix_fmt','yuv420p','-crf','23']),dpi=100)
        if len(cases)==2:
            ani=FuncAnimation(fig,draw,frames=frames[::2]+[frames[-1]],interval=220,blit=False)
            ani.save(out/(name+'.gif'),writer=PillowWriter(fps=5),dpi=65)
        draw(frames[-1]);fig.savefig(out/(name+'-final.png'),dpi=140);plt.close(fig)
    movie([0,1],list(range(120,149)),'boundary-comparison')
    movie([1],list(range(149)),'released-full-trajectory')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--movies',action='store_true');a=p.parse_args();main(a.output,a.movies)
