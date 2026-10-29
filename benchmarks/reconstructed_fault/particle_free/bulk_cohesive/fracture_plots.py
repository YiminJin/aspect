#!/usr/bin/env python3
"""PNG/GIF only; plotted numerical arrays are exported separately as CSV."""
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,Normalize
from matplotlib.animation import FuncAnimation,PillowWriter
from fracture_analysis import OUT,BUILD,LEVELS,read
from fracture_strip import seed_table

plt.rcParams.update({'font.size':9,'axes.titlesize':10})

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=170);plt.close(fig)

def strip():
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
    summary=json.loads((OUT/'strip-summary.json').read_text())
    for row,name in enumerate(['retained','candidate']):
        a=read(OUT/f'{name}-strip-envelope.csv');p=summary[name]
        ax=axes[row,0];ax.plot(a['elastic_slip_m']*1000,a['traction_Pa'],label='Stationary strip envelope')
        actual=read(OUT/f'{name}-strip-history.csv')
        ax.plot(actual['elastic_slip_m']*1000,actual['traction_Pa'],color='tab:orange',label='Actual BE strip to .185 s')
        ax.axhline(1000,color='.4',ls='--',label='Residual friction');ax.set(xlabel='Elastic strip slip [mm]',ylabel='Traction [Pa]',title=name.capitalize()+': cohesive-scale response')
        ax.axvline(p['final_elastic_slip_limit_m']*1000,color='tab:red',ls=':',label='Analytic finite-slip limit')
        if row==0:ax.legend(fontsize=7)
        ax=axes[row,1]
        for key,label_,style in [('fracture_J_m2','Fracture storage','-'),('cohesive_J_m2','Recoverable cohesive storage','--'),('excess_work_J_m2','Work above residual friction',':')]:
            ax.plot(a['peak_phi'],a[key]/p['Gc'],style,label=label_)
        ax.axhline(p['seed_fracture_fraction'],color='.5',ls='-.',label='Pre-existing seed storage')
        ax.set(xlabel='Profile peak phase [1]',ylabel='Energy / Gc [1]',title='Stationary family: no viscosity; virgin reference')
        if row==0:ax.legend(fontsize=7)
    fig.suptitle('Gc is the AT1 full-band fracture normalization, not frictional work\nActual finite-time BE strip histories, viscosity and seed storage are separate CSV records',fontsize=12)
    save(fig,'strip-scales')

def xtime(names):
    fig,axes=plt.subplots(len(names),2,figsize=(12,4.1*len(names)),squeeze=False,layout='constrained')
    arrays=[read(OUT/f'{name}-centerline.csv') for name in names]
    vmax=max(float(a['phase_increment'].max()) for a in arrays)
    rate_max=max(float(a['cross_fault_integral_abs_w_m_s'].max()) for a in arrays)
    for row,(name,a) in enumerate(zip(names,arrays)):
        t=np.unique(a['time_s']);x=np.unique(a['x_m']);s=json.loads((OUT/f'{name}-summary.json').read_text());tip=s['tip_x_m'];support=tip+seed_table(s['ell_m'])[0][-1]
        for col,(key,label_,limit,low) in enumerate([('phase_increment','Centerline phase increment [1]',vmax,1e-6),('cross_fault_integral_abs_w_m_s','Cross-fault integral of |w| [m/s]',rate_max,1e-7)]):
            ax=axes[row,col];v=a[key].reshape(len(t),len(x))
            art=ax.pcolormesh(x,t,np.ma.masked_less_equal(v,0),norm=LogNorm(low,max(limit,low*10)),cmap='magma',shading='auto')
            ax.axvline(tip,color='cyan',lw=1,label='Original endpoint');ax.axvline(support,color='lime',ls=':',lw=1,label='Initial support end')
            for thresh in [1e-5,1e-4]:
                if v.min()<thresh<v.max() and col==0:ax.contour(x,t,v,levels=[thresh],colors='white',linewidths=.6)
            ax.set(xlabel='x [m]',ylabel='Accepted physical time [s]',ylim=(.11,t[-1]),title=name.capitalize()+': '+label_)
            fig.colorbar(art,ax=ax,pad=.02)
            bracket=s['onset']['remote_1e-12']
            if bracket:
                ax.axhspan(*bracket,color='deepskyblue',alpha=.3)
                ax.text(.02,.97,f'Far-ahead onset in ({bracket[0]:.5f}, {bracket[1]:.5f}] s',transform=ax.transAxes,va='top',color='tab:blue',bbox=dict(fc='white',alpha=.8,ec='none'))
            if row==0 and col==0:ax.legend(loc='lower right',fontsize=7)
    fig.suptitle('Activation in saved accepted states; Δt = 0.00125 s\nRate is the integral of absolute bulk shear rate, not signed surface velocity',fontsize=12)
    save(fig,'activation-x-time' if len(names)>1 else 'released-activation-x-time')

def early(names):
    fig,axes=plt.subplots(len(names),3,figsize=(14,3.8*len(names)),squeeze=False,layout='constrained')
    for row,name in enumerate(names):
        a=read(OUT/f'{name}-early-profiles.csv');steps=np.unique(a['step']);s=json.loads((OUT/f'{name}-summary.json').read_text())
        colors=plt.cm.viridis(np.linspace(0,1,len(steps)))
        for step,c in zip(steps,colors):
            b=a[a['step']==step];label_=f'{b["time_s"][0]:.5f} s'
            axes[row,0].plot(b['x_m'],b['physical_tau_xy_Pa'],color=c,label=label_)
            axes[row,1].plot(b['x_m'],b['cell_mean_Y_Pa'],color=c)
            axes[row,2].plot(b['x_m'],b['weak_mechanical_drive_Pa'],color=c,label=label_+' drive')
            axes[row,2].plot(b['x_m'],b['weak_fracture_resistance_Pa'],color=c,ls='--')
        for col,ax in enumerate(axes[row]):
            ax.axvline(s['tip_x_m'],color='red',ls=':',lw=.8)
            ax.axvline(s['tip_x_m']+seed_table(s['ell_m'])[0][-1],color='green',ls=':',lw=.8)
            ax.set(xlabel='x [m]',ylabel=['Physical tau_xy [Pa]','Cell-mean Y [Pa]','Compatible weak force / mass [Pa]'][col],title=name.capitalize());ax.grid(alpha=.2)
        axes[row,0].legend(fontsize=7,ncol=2)
        axes[row,0].axhline(2000 if name=='released' else 3000,color='tab:red',ls='--',lw=.7)
    fig.suptitle('Before and during initially intact activation, along y = 0.5 m\nSame compact kernel radius 0.03125 m; solid weak drive, dashed weak fracture resistance',fontsize=12)
    fig.supxlabel('Stress/Y: positive native-quadrature cell means, followed by area-weighted physical averaging. Weak forces: same kernel with native nodal masses.\nNo local constitutive history is interpolated. Red: original endpoint; green: original cap support. Phase geometry is shown separately.',fontsize=8)
    save(fig,'early-physical-profiles' if len(names)>1 else 'released-early-physical-profiles')

def marks(ax,d,s,n):
    x,y=d['x'],d['y'];p=d['phase'][n];p0=d['phase'][0];tip=s['tip_x_m'];radius=seed_table(s['ell_m'])[0][-1]
    ax.contour(x,y,p0,levels=LEVELS,colors='white',linestyles='--',linewidths=.5)
    cs=ax.contour(x,y,p,levels=LEVELS,colors='cyan',linewidths=.65)
    xx,yy=np.meshgrid(x,y);support=np.hypot(np.maximum(xx-tip,0),yy-.5)
    ax.contour(x,y,support,levels=[radius],colors='lime',linestyles=':',linewidths=.7)
    born=(p0==0)&(p-p0>1e-5)
    if born.any():ax.contour(x,y,born.astype(float),levels=[.5],colors='magenta',linewidths=.7)
    ax.plot([0,tip],[.5,.5],color='red',lw=1);ax.plot(tip,.5,'rx',ms=5)
    ax.set(xlabel='x [m]',ylabel='y [m]',xlim=(0,1.5),ylim=(0,1),aspect='equal')

def maps(names):
    data=[np.load(BUILD/f'{n}-plot-data.npz') for n in names]
    maximum=max(float((d['phase']-d['phase'][0]).max()) for d in data)
    maximumstress=max(float(d['stress'].max()) for d in data)
    maximumY=max(float(d['Y'].max()) for d in data)
    fig,axes=plt.subplots(len(names),3,figsize=(13.5,3.9*len(names)),squeeze=False,layout='constrained')
    for row,(name,d) in enumerate(zip(names,data)):
        s=json.loads((OUT/f'{name}-summary.json').read_text());n=min(np.searchsorted(d['time'],.175),len(d['time'])-1)
        fields=[d['stress'][n],d['phase'][n]-d['phase'][0],d['Y'][n]]
        for col,(field,label_) in enumerate(zip(fields,['Physical tau_xy [Pa]','Phase increment [1]','Mechanical Y [Pa]'])):
            ax=axes[row,col];art=ax.pcolormesh(d['x'],d['y'],field,shading='auto',cmap='magma' if col==2 else 'viridis',vmin=0,vmax=[maximumstress,maximum,maximumY][col])
            marks(ax,d,s,n);ax.set_title(f'{name}, t={d["time"][n]:.5f} s: '+label_);fig.colorbar(art,ax=ax,pad=.02,location='bottom',shrink=.85)
    fig.suptitle('Whole-domain response: contours distinguish geometry from driving\nWhite dashed: initial phase; cyan: accepted phase; lime: original support; magenta: initially-zero damage > 1e-5\nPhase levels: .001, .01, .05, .1, .3, .5',fontsize=10)
    save(fig,'physical-mechanism')
    fig,axes=plt.subplots(1,len(names),figsize=(6*len(names),4),squeeze=False,layout='constrained')
    for ax,name,d in zip(axes[0],names,data):
        a=read(OUT/f'{name}-activation-nodes.csv');s=json.loads((OUT/f'{name}-summary.json').read_text())
        art=ax.scatter(a['x_m'],a['y_m'],c=a['activation_upper_s'],vmin=.12,vmax=.185,s=7,cmap='viridis',marker='s')
        marks(ax,d,s,0);fig.colorbar(art,ax=ax,label='Upper end of activation bracket [s]',location='bottom');ax.set_title(name)
    fig.suptitle('First numerical phase increment > 1e-12: accepted-state brackets, not exact times')
    save(fig,'first-activation-locations')

def energy(names):
    fig,axes=plt.subplots(2,3,figsize=(13.5,8),layout='constrained')
    for name in names:
        a=read(OUT/f'{name}-history.csv');t=a['time_s'];s=json.loads((OUT/f'{name}-summary.json').read_text())
        axes[0,0].plot(a['prescribed_strain'],a['reaction_N_m'],label=name)
        axes[0,1].plot(t,a['cap_growth_m2'],label=name+' cap')
        axes[0,1].plot(t,a['outside_seed_growth_m2'],ls='--',label=name+' outside support')
        axes[0,2].plot(t,a['max_phase_increment'],label=name)
        W=np.cumsum(a['trapezoidal_work_J_m']);E=a['bulk_J_m']+a['cohesive_J_m']+a['added_fracture_J_m'];D=np.cumsum(a['Db_J_m']+a['Dc_J_m']+a['Df_J_m'])
        axes[1,0].plot(t,W,label=name+' W_tr');axes[1,0].plot(t,E+D,ls='--',label=name+' storage + D')
        axes[1,1].plot(t,np.cumsum(a['endpoint_remainder_J_m']),label=name+' endpoint')
        axes[1,1].plot(t,np.cumsum(abs(a['trapezoidal_remainder_J_m'])),ls='--',label=name+' sum |trapezoidal|')
        b=read(OUT/f'{name}-contours.csv')
        for level_,ls in [(.01,'-'),(.1,'--'),(.5,':')]:
            c=b[b['phase_threshold']==level_];axes[1,2].plot(c['time_s'],1000*(c['centerline_first_exit_x_m']-c['centerline_first_exit_x_m'][0]),ls=ls,label=name+f' phi={level_:g}')
    settings=[('Prescribed shear strain [1]','Generalized reaction [N/m]','Measured boundary response'),('Time [s]','Integrated phase increment [m2]','Growth location'),('Time [s]','Peak phase increment [1]','Growth, excluding initial seed'),('Time [s]','Work / energy [J/m]','Initial fracture storage subtracted'),('Time [s]','Numerical remainder [J/m]','Energy accuracy is not ledger closure'),('Time [s]','Centerline first-exit change [mm]','Connected contour diagnostic')]
    for ax,(xl,yl,title) in zip(axes.ravel(),settings):ax.set(xlabel=xl,ylabel=yl,title=title);ax.grid(alpha=.2);ax.legend(fontsize=6)
    axes[1,1].set_yscale('symlog',linthresh=1e-7)
    fig.suptitle('New physical scales and seed length form a new specimen; these are mechanism comparisons, not convergence errors')
    save(fig,'response-and-work')

def gif():
    name='candidate';d=np.load(BUILD/f'{name}-plot-data.npz');s=json.loads((OUT/f'{name}-summary.json').read_text());nlast=len(d['time'])-1
    selected=list(range(96 if nlast>=96 else 0,nlast+1))
    maxdamage=max(float((d['phase']-d['phase'][0]).max()),1e-6);maxrate=max(float(d['rate'].max()),1e-6)
    fields=[d['phase'][0],np.zeros_like(d['phase'][0]),d['stress'][0],d['rate'][0]]
    norms=[Normalize(0,max(.6,float(d['phase'].max()))),Normalize(0,maxdamage),Normalize(0,float(d['stress'].max())),Normalize(0,maxrate)]
    labels=['Phase phi [1]','Phase increment [1]','Physical tau_xy [Pa]','Native cell mean |w| [1/s]']
    fig,axes=plt.subplots(2,2,figsize=(10,7.2),layout='constrained');artists=[];dynamic=[]
    for ax,field,norm,label_ in zip(axes.ravel(),fields,norms,labels):
        artists.append(ax.pcolormesh(d['x'],d['y'],field,shading='auto',norm=norm,cmap='viridis'))
        ax.set_title(label_);fig.colorbar(artists[-1],ax=ax,pad=.01,location='bottom',shrink=.8)
    title=fig.suptitle('')
    fig.supxlabel('White dashed: initial phase; cyan: accepted contours (.001,.01,.05,.1,.3,.5); lime: original support; red: original seed.\nPrescribed released-side loading; fixed mesh and geometry. Growth speed and branch selection remain unqualified.',fontsize=8)
    def frame(n):
        nonlocal dynamic
        for c in dynamic:c.remove()
        dynamic=[];p=d['phase'][n];values=[p,p-d['phase'][0],d['stress'][n],d['rate'][n]]
        for ax,artist,value in zip(axes.ravel(),artists,values):
            artist.set_array(value.ravel())
            # Constant annotations are reset with the contours to avoid accumulation.
            for line in list(ax.lines):line.remove()
            tip=s['tip_x_m'];xx,yy=np.meshgrid(d['x'],d['y']);radius=seed_table(s['ell_m'])[0][-1]
            dynamic.extend([ax.contour(d['x'],d['y'],d['phase'][0],levels=LEVELS,colors='white',linestyles='--',linewidths=.45),ax.contour(d['x'],d['y'],p,levels=LEVELS,colors='cyan',linewidths=.6),ax.contour(d['x'],d['y'],np.hypot(np.maximum(xx-tip,0),yy-.5),levels=[radius],colors='lime',linestyles=':',linewidths=.6)])
            ax.plot([0,tip],[.5,.5],color='red',lw=.8);ax.plot(tip,.5,'rx',ms=4)
            ax.set(xlim=(0,1.5),ylim=(0,1),aspect='equal',xlabel='x [m]',ylabel='y [m]')
        title.set_text(f'One fresh scale-selected specimen, accepted t={d["time"][n]:.5f} s\nEvery accepted frame, Δt=.00125 s; growth speed unqualified')
        return artists
    FuncAnimation(fig,frame,frames=selected,interval=170,blit=False).save(OUT/'propagation-focused.gif',writer=PillowWriter(fps=6));plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--released-only',action='store_true');p.add_argument('--gif',action='store_true');a=p.parse_args()
    names=['released'] if a.released_only else ['released','candidate']
    strip();xtime(names);early(names)
    if not a.released_only:maps(names);energy(names)
    if a.gif:gif()
