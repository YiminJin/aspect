#!/usr/bin/env python3
"""Actual accepted geometry, front, slip, stress and contact-reaction views."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Circle
from PIL import Image

from cohesive_interface_path import WORK, OUT, graph, functionals, save, table
from analyze_cohesive_interface_path import context, state, snapshots, rows, summary

plt.rcParams.update({'font.size':10,'figure.dpi':140,'axes.grid':True,'grid.alpha':.15})
CACHE={}


def view(s):
    key=(s['kernel_folder'],s.get('pressure_mode','point'))
    if key not in CACHE:CACHE[key]=context(s)
    return CACHE[key]


def values(k,s):
    c=np.r_[s['wall_memory'],s['memory']]
    return np.einsum('ijk,j->ik',k.mean,c),10000+k.N@c,10000+k.raw_Nm@c,10000+k.raw_Np@c


def map_state(ax,s,stress=True,whole=False,arrows=True):
    k=view(s);F,p,pm,pp=values(k,s);h=12/k.meta['nx'];centers=np.c_[k.centers['x_m'],k.centers['y_m']]
    field=None
    if stress:
        polygon=centers[:,None,:]+h*np.array([[[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]])
        field=PolyCollection(polygon,array=F[:,3]/1000,cmap='RdBu_r',clim=(-4,4),edgecolors='none',rasterized=True);ax.add_collection(field)
    for arc in s['geometry']:
        z=np.linspace(0,1,60);yy,_=graph(arc,z);xx=arc['x']+(arc['end']-arc['x'])*z
        ax.plot(xx,yy,'--',color='.5',lw=1.4,zorder=3)
    xy=np.c_[k.s['x_m'],k.s['y_m']];a=np.array(s['histories']['damage'])
    segments=np.stack([xy[:-1],xy[1:]],axis=1);colors=.5*(a[:-1]+a[1:])
    surf=LineCollection(segments,array=colors,cmap='viridis',clim=(0,1),linewidths=3,zorder=4);ax.add_collection(surf)
    ax.plot([3,6],[0,0],'k--',lw=.8,zorder=5);ax.plot(6,0,'ko',ms=4,zorder=6)
    f=functionals(k,s)
    for label,marker,color in [('process','o','C3'),('half','s','C1'),('full','*','k')]:
        if label=='full' and not f['new_full_front_available']:continue
        if f[label+'_front_x_m']>6:ax.plot(f[label+'_front_x_m'],f[label+'_front_y_m'],marker=marker,color=color,ms=6,zorder=8)
    if arrows:
        ids=np.arange(4,len(a),10);jump=np.array(s['histories']['jump'])
        ax.quiver(xy[ids,0],xy[ids,1]+.12,jump[ids]*k.s['tx'][ids],jump[ids]*k.s['ty'][ids],
                  color='k',scale=.004,scale_units='xy',angles='xy',width=.004,zorder=7)
    ax.set_aspect('equal');ax.set(xlabel='x [m]',ylabel='y [m]')
    ax.set(xlim=(0,12) if whole else (5.7,9.2),ylim=(-6,6) if whole else (-.25,1.3))
    return field,surf


def export_surface(label,s,frame):
    k=view(s);_,pn,pm,pp=values(k,s);vel=k.Q@np.array(s.get('nodal_rate',np.zeros(len(k.nodes))))
    h=s['histories'];data=[];arc=np.cumsum(k.w)-.5*k.w-3
    for q in range(len(k.w)):
        active=abs(vel[q])>1e-9;angle=np.rad2deg(np.arctan2(k.s['ty'][q],k.s['tx'][q]))+(180 if vel[q]<0 else 0)
        data.append(dict(case=label,frame=frame,time_s=s['clock_s'],gamma=s['gamma'],generation=s['generation'],
            x_m=float(k.s['x_m'][q]),y_m=float(k.s['y_m'][q]),arclength_from_old_tip_m=float(arc[q]),weight_m=float(k.w[q]),
            damage=h['damage'][q],signed_jump_m=h['jump'][q],jump_x_m=h['jump'][q]*k.s['tx'][q],jump_y_m=h['jump'][q]*k.s['ty'][q],
            velocity_m_s=float(vel[q]),flow_angle_deg=float(angle) if active else 'undefined',
            weak_contact_pressure_Pa=float(pn[q]),raw_normal_minus_Pa=float(pm[q]),raw_normal_plus_Pa=float(pp[q]),theta=h['theta'][q]))
    return data


def main():
    plotted=[];primary='work96';s=state(primary)
    fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    analytic=json.loads((OUT/'analytic-checks.json').read_text())
    for angle,color,style,label in [(0,'k','-','old tangent 0°'),(-25,'.35','-','principal compression −25°'),
             (20,'C0','-','maximum shear +20°'),(-70,'C0','-','conjugate −70°'),
             (analytic['diagnostic_Coulomb_deg'][0],'C1','--','constant-Coulomb diagnostic'),
             (analytic['diagnostic_Coulomb_deg'][1],'C1','--',None)]:
        p=np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))]);axs[0].plot([-p[0],p[0]],[-p[1],p[1]],color=color,ls=style,label=label)
    axs[0].set_aspect('equal');axs[0].set(xlim=(-1.2,1.2),ylim=(-1.2,1.2),xlabel='+x reference',ylabel='+y reference',title='Homogeneous physical-stress limit')
    axs[0].legend(fontsize=8,loc='upper left')
    map_state(axs[1],s,stress=False);axs[1].set_title('Computed Γ and actual fronts at 0.445 s')
    f=functionals(view(s),s)
    axs[1].text(.02,.96,f'○ a=.05: {f["process_front_tangent_deg"]:.2f}°\n□ a=.5: {f["half_front_tangent_deg"]:.2f}°\n★ a=1: {f["full_front_tangent_deg"]:.2f}°',transform=axs[1].transAxes,va='top')
    end=s['geometry'][-1];yy,pp=graph(end,1.);terminal=np.rad2deg(np.arctan(pp));tt=np.array([1.,pp])/np.hypot(1,pp)
    axs[1].annotate('',xy=np.array([end['end'],yy])+.28*tt,xytext=(end['end'],yy),arrowprops=dict(arrowstyle='->',color='C4'))
    axs[1].text(end['end']-.4,yy+.2,f'terminal {terminal:.2f}°',color='C4',fontsize=8)
    fig.suptitle('Formation preference is maximum incipient shear work; μ₀=0.1 does not change intact strength\nTerminal candidate angle, actual front tangent and signed slip direction are distinct')
    fig.savefig(OUT/'angular-conventions.png');plt.close(fig)

    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    clocks=[0,.15,.40025,.44,.445]
    for ax,clock in zip(axs.flat[:5],clocks):
        ss=state(primary,clock);field,surf=map_state(ax,ss,stress=True)
        ax.set_title(f't={clock:g} s, γ={ss["gamma"]:.5f}');plotted+=export_surface(primary,ss,clock)
    k=view(s);F,p,pm,pp=values(k,s);x=k.s['x_m']
    axs[1,2].plot(x,p/1000,label='weak normal reaction');axs[1,2].plot(x,pm/1000,'--',color='.5',lw=.7,label='raw lower trace');axs[1,2].plot(x,pp/1000,':',color='.25',lw=.7,label='raw upper trace')
    axs[1,2].set(xlabel='x [m]',ylabel='compression [kPa]',title='Current contact and\nunfiltered Cauchy diagnostics');axs[1,2].legend(fontsize=8)
    fig.colorbar(field,ax=list(axs.flat[:5]),label='physical cell-mean τxy [kPa]',shrink=.65);fig.colorbar(surf,ax=list(axs.flat[:5]),label='surface damage a',shrink=.65,pad=.01)
    fig.suptitle('Successive accepted-state selections preserve the old straight seed\nDashed tails are tied computational support; colors show actual damage, arrows amplified signed slip')
    fig.savefig(OUT/'successive-extensions.png');plt.close(fig)

    labels=['work64','work96','work-step96','work-q3','work-surface96','work-time96']
    titles=['64² Q2 bulk','96² Q2 bulk','96² Q2, Δs=.375 m','64² Q3 bulk',
            '96² Q2, surface spacing .046875 m','96² Q2, late Δt=.000125 s']
    fig,axs=plt.subplots(2,3,figsize=(15,7),layout='constrained')
    for ax,label,title in zip(axs.flat,labels,titles):
        ss=state(label);field,surf=map_state(ax,ss,stress=False);ff=functionals(view(ss),ss)
        ax.set_title(f'{title}\nnew ∫a ds={ff["new_damage_integral_m"]:.4f} m');plotted+=export_surface(label,ss,'final')
    fig.colorbar(surf,ax=list(axs.flat),label='surface damage a',shrink=.75)
    fig.suptitle('Same physical R=.75 m, calibration, continuum initialization and loading; t=.445 s\n○ process front, □ half damage, ★ fully decohered front; grey tail is not fracture advance')
    fig.savefig(OUT/'path-controls.png');plt.close(fig)

    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for label in ['work64','work96','work-step96','work-q3']:
        r=rows(label);axs[0,0].plot(r['time_s'],r['new_damage_integral_m'],label=label)
        axs[0,1].plot(r['time_s'],r['full_front_x_m']-6,label=label)
    axs[0,0].set(xlabel='time [s]',ylabel='new ∫a ds [m]');axs[0,0].legend(fontsize=8)
    axs[0,1].set(xlabel='time [s]',ylabel='developed advance beyond old tip [m]')
    for label in ['work96','work-time96']:
        r=rows(label);r=r[r['time_s']>.42025];axs[1,0].plot(r['time_s'],np.cumsum(abs(r['physical_remainder_J_m']))*1000,label=label)
    axs[1,0].set(xlabel='time [s]',ylabel='late Σ|physical remainder| [mJ/m]');axs[1,0].legend(fontsize=8)
    r=rows(primary);initial=np.genfromtxt(OUT/'initialization.csv',delimiter=',',names=True,dtype=None,encoding=None)
    init=initial[initial['label']==primary][0]
    available=np.cumsum(r['trapezoid_work_J_m'])-(r['bulk_J_m']+r['bond_J_m']-init['bulk_J_m']-init['bond_J_m'])
    axs[1,1].plot(r['time_s'],available,label='work minus storage change')
    for name,key in [('friction + radiation','sliding_loss_J_m'),('Maxwell loss','maxwell_loss_J_m'),('additional fracture','fracture_increment_J_m')]:
        axs[1,1].plot(r['time_s'],np.cumsum(r[key]),label=name)
    axs[1,1].set(xlabel='time [s]',ylabel='energy [J/m thickness]');axs[1,1].legend(fontsize=8)
    fig.suptitle('Actual response and physical work; small backward-Euler ledger closure is not the energy criterion')
    fig.savefig(OUT/'growth-and-work.png');plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    c=np.genfromtxt(OUT/'independent-interpolation-ODE.csv',delimiter=',',names=True)
    axs[0].loglog(c['extension_m'],c['curvature_L2_relative'],'o-');axs[0].set(xlabel='numerical extension length Δs [m]',ylabel='curvature error / exact curvature',title='Same physical dθ/ds=(θ*−θ)/R')
    p=np.genfromtxt(OUT/'frozen-pressure-comparison.csv',delimiter=',',names=True)
    axs[1].bar(np.arange(len(p))-.18,p['normal_difference_RMS_Pa'],width=.36,label='weak normal reaction')
    axs[1].bar(np.arange(len(p))+.18,p['raw_projected_difference_RMS_Pa'],width=.36,label='raw mean, same bubble projection')
    axs[1].set(xticks=np.arange(len(p)),xticklabels=[f'{n}²' for n in p['mesh']],ylabel='normal-pressure comparison RMS [Pa]',title='Independent fitted curved-geometry comparison');axs[1].legend(fontsize=8)
    fig.suptitle('Numerical defects are measured separately from path assumptions and physical dissipation')
    fig.savefig(OUT/'numerical-controls.png');plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(12,6),layout='constrained')
    field,surf=map_state(axs[0],s,stress=True,whole=True);fig.colorbar(field,ax=axs[0],label='physical cell-mean τxy [kPa]',shrink=.7)
    axs[0].set_title('Finite confined 12 × 12 m specimen')
    axs[0].text(.02,.97,'p₀=10 kPa; all-boundary affine strain\nOld seed (3,0)→(6,0), no old direction changes',transform=axs[0].transAxes,va='top',fontsize=8)
    before=json.loads((WORK/primary/'before-extension-3.json').read_text());kk=view(before)
    end=before['geometry'][-1];yy,pp=graph(end,1.);tt=np.array([1.,pp])/np.hypot(1,pp);center=np.array([end['end'],yy])+.75*tt
    map_state(axs[1],before,stress=False);axs[1].add_patch(Circle(center,.75,facecolor='C0',edgecolor='C0',alpha=.12))
    axs[1].plot(*center,'+',color='C0');axs[1].set_title(f'Accepted-state selection at t={before["clock_s"]:g} s')
    axs[1].text(.02,.95,'Disk: physical stress-sampling radius R=.75 m\nTied reserve is prospective geometry, not damage',transform=axs[1].transAxes,va='top',fontsize=8)
    fig.suptitle('What is imposed and what is selected\nThe formation/persistence hypothesis uses accepted stress; no future damage score or Coulomb-angle prescription')
    fig.savefig(OUT/'specimen-and-selection.png');plt.close(fig)

    a={round(s['clock_s'],10):s for s in snapshots('work96')};b={round(s['clock_s'],10):s for s in snapshots('work-step96')}
    a[0.]=state('work96',0);b[0.]=state('work-step96',0)
    common=sorted(set(a)&set(b));late=[t for t in common if t>=.28]
    chosen=sorted(set([0.,.15]+[late[i] for i in np.linspace(0,len(late)-1,min(23,len(late))).astype(int)]));images=[]
    for i,t in enumerate(chosen):
        fig,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained')
        for j,(label,ss) in enumerate([('work96',a[t]),('work-step96',b[t])]):
            field,surf=map_state(axs[0,j],ss,stress=True);axs[0,j].set_title(f'96² Q2 bulk: Δs={summary(label)["extension_m"]:g} m')
            kk=view(ss);_,p,pm,pp=values(kk,ss);arc=np.cumsum(kk.w)-.5*kk.w-3
            axs[1,j].plot(arc,p/1000,label='normal reaction');axs[1,j].plot(arc,pm/1000,'--',color='.6',lw=.6,label='raw traces');axs[1,j].plot(arc,pp/1000,':',color='.4',lw=.6)
            axs[1,j].set(xlim=(-3,3),ylim=(5,14),xlabel='arclength from old tip [m]',ylabel='compression [kPa]');axs[1,j].legend(fontsize=8)
            plotted+=export_surface(label,ss,i)
        fig.colorbar(field,ax=list(axs[0]),label='cell-mean τxy [kPa]',shrink=.75);fig.colorbar(surf,ax=list(axs[0]),label='a',shrink=.75,pad=.01)
        fig.suptitle(f't={t:.5f} s | γ={a[t]["gamma"]:.6f} | native Δt=.00025 s\nNew tied records cost no fracture energy; only actual damage contributes Gc a')
        fig.canvas.draw();im=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3]);images.append(im.quantize(colors=160));plt.close(fig)
    images[0].save(OUT/'successive-deflection.gif',save_all=True,append_images=images[1:],duration=180,loop=0,optimize=False)
    table(OUT/'plotted-profiles.csv',plotted)
    save(OUT/'plot-definitions.json',dict(frames=len(images),spatial_axes='physical metres; equal aspect',
        bulk_stress='Native higher-order cut-cell mean physical deviatoric tau_xy; no auxiliary T or visual filtering.',
        contact='Authoritative DG0 current normal reaction; raw sided Cauchy traces remain unfiltered diagnostics.',
        surface='Actual Gauss-owned a, signed slip/rate and theta. Flow angle printed only for |V|>1e-9 m/s.',
        fronts='Beyond old endpoint only; Q1 slip thresholds used after verification against every new actual damage history. No new full front before a=1.',
        arrows='Signed t*delta, length amplified for readability. Flow/crack/front/candidate angles are distinct.',
        units=dict(stress='Pa',slip='m',velocity='m/s',time='s',energy='J/m out-of-plane thickness'),
        animation='Actual accepted snapshots, no interpolated histories; playback speed is not rupture speed. At equal clocks, show the latest accepted zero-time geometry.'))
    for k in CACHE.values():k.close()
    print(json.dumps(dict(pngs=6,gif_frames=len(images),surface_rows=len(plotted))))


if __name__=='__main__':main()
