#!/usr/bin/env python3
"""Plots use accepted surface histories; no interpolated constitutive states."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-cohesive-surface')
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from analyze_cohesive_interface_surface import (OUT, WORK, PREVIOUS, CASES, TIMES,
    history, fields, state, contour, operators)

LABELS = {'surface-256':'Q1 surface Δs=62.5 mm',
          'surface-512':'Q1 surface Δs=31.25 mm',
          'surface-512-half':'31.25 mm, half Δt'}
COLORS = dict(zip(CASES,['#2166ac','#b2182b','#e69f00']))
STYLES = dict(zip(CASES,['-','--',':']))


def finish(fig,name):
    fig.savefig(OUT/name,dpi=160,bbox_inches='tight')
    plt.close(fig)


def main():
    result=json.loads((OUT/'verification.json').read_text())
    summary=result['summary']
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for name in CASES:
        rows=[r for r in summary if r['case']==name]
        x=[r['time_s'] for r in rows]
        for ax,key in zip(axes.flat,['left_full_advance_m','integrated_new_damage_m','right_integrated_growth_m','integrated_abs_jump_m2']):
            ax.plot(x,[r[key]*(1000 if key=='integrated_abs_jump_m2' else 1) for r in rows],
                    STYLES[name],color=COLORS[name],marker='o',ms=3,label=LABELS[name])
    for ax,title,y in zip(axes.flat,['Connected full-decohesion advance','New fracture/damage inventory','Right cohesive flank','Integrated absolute jump'],
                         ['Left advance (m)','New Gc∫a (J/m)','Right ∫(a−a₀) ds (m)','∫|jump| ds (mm m)']):
        ax.set(title=title,xlabel='Time (s)',ylabel=y)
        ax.grid(alpha=.25)
        ax.axvline(2.55,color='.4',ls=':',label='Loading → hold' if ax is axes[0,0] else None)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Fresh surface-space comparison; same analytical bulk, law and physical loading\nInitial mature seed inventory 3 J/m is excluded from growth')
    finish(fig,'comparison.png')

    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for name in CASES:
        f=fields(name,2.58)
        values=[f['s']['a'],1000*f['s']['jump'],f['pn']/1000,f['shear']/1000]
        for ax,z in zip(axes.flat,values):
            ax.plot(f['x'],z,STYLES[name],color=COLORS[name],label=LABELS[name])
    for ax,title,y in zip(axes.flat,['Surface damage; bulk φ is not evolved','Physical signed displacement jump',
                                   'Current normal traction entering friction','Work-adjoint shear load'],
                         ['a','Jump (mm)','Normal pressure (kPa)','Projected load (kPa)']):
        ax.set(title=title,xlabel='Physical x (m)',ylabel=y,xlim=(-2.5,8.))
        ax.axvspan(2.5,5.5,color='.8',alpha=.3)
        ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8)
    axes[0,0].set_ylim(-.02,1.03)
    fig.suptitle('Common physical time 2.58 s; grey band is the pre-existing mature seed\nTractions use the established Q1 projection; these are not raw point-trace accuracy estimates')
    finish(fig,'surface-profiles.png')

    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for kind,color,style in [('surface refinement','#b2182b','-'),('fine temporal','#2166ac','--')]:
        rows=[r for r in result['comparisons'] if r['kind']==kind and r['time_s']>=2.45]
        for ax,key,scale in zip(axes.flat,['growth_relative','left_front_absolute_m','growth_box_L2_relative','normal_box_RMS_Pa'],[100,1000,100,1]):
            ax.plot([r['time_s'] for r in rows],[r[key]*scale for r in rows],style+'o',color=color,label=kind)
    for ax,title,y in zip(axes.flat,['Integrated growth difference','Continuous full-front difference',
                                   'Common physical .125 m box averages','Common-box normal traction'],
                         ['Relative difference (%)','Absolute difference (mm)','Growth-relative L² difference (%)','RMS difference (Pa)']):
        ax.set(title=title,xlabel='Time (s)',ylabel=y)
        ax.grid(alpha=.25)
    axes[0,0].legend()
    fig.suptitle('Space and time changes measured separately\nDifferences below temporal uncertainty do not establish a spatial convergence order')
    finish(fig,'space-time-errors.png')

    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for name in CASES[:2]:
        data=[r for r in history(name) if 1.35<=r['time_s']<=2.30]
        for ax,limits in zip(axes,[(1.35,1.60),(1.5,2.30)]):
            ax.semilogy([r['time_s'] for r in data],[max(r['integrated_new_damage_m'],1e-9) for r in data],
                        STYLES[name],color=COLORS[name],label=LABELS[name])
            ax.set(xlim=limits,xlabel='Time (s)',ylabel='Integrated new damage (m)',ylim=(5e-7,.3))
            ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle('Early admission remains resolution-sensitive: first-growth brackets differ by 72.5 ms\nCommon early Δt=2.5 ms; no new early-time halving, so precise onset is unqualified')
    finish(fig,'early-response.png')

    # One fine trajectory, actual saved accepted times, including restart/hold.
    # The animation never interpolates damage, rate or friction histories.
    name='surface-512'
    paths=sorted((WORK/name).glob('state-*.npz'),key=lambda p:int(p.stem.split('-')[1]))
    snapshots=[]
    op=operators(512)
    for path in paths:
        t=int(path.stem.split('-')[1])/1e6
        if t>=2.30-1e-8:
            f=fields(name,t,op)
            snapshots.append((float(f['s']['clock_s']),f))
    fig,axes=plt.subplots(3,1,figsize=(12,7),sharex=True,layout='constrained')
    lines=[ax.plot([],[],color=COLORS[name],lw=1.8)[0] for ax in axes]
    for ax,y,limits in zip(axes,['Surface damage a','Physical jump (mm)','Normal pressure (kPa)'],[(-.02,1.03),(-.02,1.4),(7.,14.)]):
        ax.set(ylabel=y,ylim=limits,xlim=(-4,12))
        ax.axvspan(2.5,5.5,color='.8',alpha=.35)
        for endpoint in [-3.5,11.5]:ax.axvline(endpoint,color='.5',ls=':')
        ax.grid(alpha=.2)
    axes[-1].set_xlabel('Physical x (m); grey = initial mature seed, dotted = fixed Γ ends')
    text=fig.suptitle('')
    def update(i):
        t,f=snapshots[i]
        for line,z in zip(lines,[f['s']['a'],1000*f['s']['jump'],f['pn']/1000]):line.set_data(f['x'],z)
        text.set_text(f'Accepted fine-surface solution: t={t:.6f} s, U={min(t,2.55):.4f} mm; '+('loading' if t<=2.55+1e-9 else 'displacement hold'))
        return lines+[text]
    animation=FuncAnimation(fig,update,frames=len(snapshots),interval=150,blit=False)
    animation.save(OUT/'surface-evolution.gif',writer=PillowWriter(fps=7))
    plt.close(fig)
    (OUT/'animation-times.json').write_text(json.dumps(dict(case=name,times_s=[t for t,_ in snapshots],interpolated_histories=False),indent=2)+'\n')


if __name__=='__main__':
    main()
