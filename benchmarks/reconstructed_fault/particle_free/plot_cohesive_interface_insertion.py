#!/usr/bin/env python3
"""Physical axes; distinguish candidate geometry, damage, jump and traction."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from matplotlib.collections import LineCollection
from analyze_cohesive_interface_insertion import read
from run_cohesive_interface_insertion import OUT
plt.rcParams.update({'font.size':10,'figure.dpi':145})

def save(fig,name):fig.savefig(OUT/name,bbox_inches='tight');plt.close(fig)
def geometry(ax):
    ax.axvspan(2.5,5.5,color='0.85',alpha=.35,label='initial mature seed')
    for x in [1,7]:ax.axvline(x,color='0.45',ls=':',lw=1)
    for x in [2.5,5.5]:ax.axvline(x,color='0.45',ls='--',lw=.8)
    ax.set_xlim(0,8);ax.grid(alpha=.15)
def comparison(h,cases):
    # All curves end at the same physical time, including the coarse control.
    endpoint=2.42
    fig,axes=plt.subplots(2,2,figsize=(12,8.5),layout='constrained')
    series={}
    for case,style,color in zip(cases,['--','-.','-'],['#0072b2','#d55e00','#009e73']):
        z=h[(h['case']==case)&(h['time_s']<=endpoint+1e-10)]
        series[case]=z
        label={'growth-64':'64 × 40; Δt=.005 s','growth-128':'128 × 80; Δt=.005 s','growth-128-half':'128 × 80; Δt=.0025 s'}[case]
        axes[0,0].plot(z['time_s'],2.5-z['left_0p05_m'],style,color=color,label=label)
        axes[0,0].plot(z['time_s'],2.5-z['left_jump_threshold_m'],style,color=color,lw=2,alpha=.65)
        axes[0,1].plot(z['time_s'],z['integrated_growth_m'],style,color=color,label=label)
    axes[0,0].set(ylabel='left advance from seed tip (m)',title='Process zone a≥.05 (upper); full decohesion (lower)')
    axes[0,0].legend(fontsize=7)
    axes[0,1].set(ylabel='integrated damage growth ∫(a−a₀) ds (m)',title='Common endpoint: 2.42 s; early spatial failure retained')
    inset=axes[0,1].inset_axes([.13,.45,.48,.47])
    for case,style,color in zip(cases,['--','-.','-'],['#0072b2','#d55e00','#009e73']):
        z=series[case];inset.plot(z['time_s'],z['integrated_growth_m'],style,color=color)
    inset.set(xlim=(1.45,1.85),ylim=(-.001,.025),title='Early-time inset');inset.tick_params(labelsize=7);inset.grid(alpha=.2)
    rows=[]
    for name,left,right,color in [('spatial',cases[0],cases[1],'#d55e00'),('temporal',cases[1],cases[2],'#009e73')]:
        a,b=series[left],series[right]
        # Existing accepted common times only; interpolation matches dt-multiple endpoints.
        ts=np.unique(a['time_s']);ts=ts[(ts>=1.5)&(ts<=endpoint+1e-10)]
        aa=np.interp(ts,a['time_s'],a['integrated_growth_m']);bb=np.interp(ts,b['time_s'],b['integrated_growth_m'])
        absolute=abs(aa-bb);relative=np.divide(absolute,bb,out=np.full_like(bb,np.nan),where=bb>1e-12)
        axes[1,0].plot(ts,absolute,color=color,label=name)
        axes[1,1].plot(ts,relative*100,color=color,label=name)
        rows.extend(zip([name]*len(ts),ts,aa,bb,absolute,relative))
    axes[1,0].set(ylabel='absolute integrated-growth difference (m)',title='Absolute difference is small early, but consequential')
    axes[1,1].set(ylabel='difference / finer growth (%)',title='Relative difference undefined before reference growth',ylim=(0,100))
    axes[1,1].axhline(5,color='#d55e00',ls=':',lw=1);axes[1,1].annotate('65.4% at 1.8 s\n|Δgrowth| = 0.0055521 m',(1.8,65.4245),xytext=(1.88,79),arrowprops={'arrowstyle':'->'},fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel('time (s)');ax.set_xlim(1.45,endpoint);ax.grid(alpha=.2);ax.axvline(endpoint,color='0.3',ls=':',lw=1)
    axes[1,0].legend();axes[1,1].legend(loc='upper left');save(fig,'comparison.png')
    import csv
    with (OUT/'comparison_growth_differences.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['comparison','time_s','coarser_growth_m','finer_growth_m','absolute_difference_m','relative_difference']);writer.writerows(rows)

def main():
    fig,ax=plt.subplots(figsize=(11,3.4));ax.plot([0,8,8,0,0],[0,0,1,1,0],color='black')
    ax.plot([0,8],[.4,.4],color='0.5',ls=':',label='uninstalled candidate continuation')
    ax.plot([1,7],[.4,.4],lw=5,color='#999999',label='finite installed interface; initially tied outside seed')
    ax.plot([2.5,5.5],[.4,.4],lw=5,color='#2b6cb0',label='mature seed, a=1')
    for x in np.linspace(.5,7.5,8):
        ax.annotate('',(x+.25,1.08),(x-.25,1.08),arrowprops={'arrowstyle':'->','color':'#b91c1c'})
    ax.text(4,1.19,'Top: U̇ = 0.001 m/s; bottom fixed. Background confinement p₀ = 10 kPa',ha='center')
    ax.text(.05,.74,'periodic',rotation=90);ax.text(7.8,.74,'periodic',rotation=90)
    ax.set(xlabel='x (m)',ylabel='y (m)',ylim=(-.15,1.35),xlim=(-.1,8.1),title='Selected finite-interface specimen — φ is provisional geometry only')
    ax.set_aspect('equal');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.35),ncol=1,frameon=False);save(fig,'specimen.png')
    a=read(OUT/'insert-loaded/insertion-before-fields.csv');b=read(OUT/'insert-loaded/insertion-after-fields.csv')
    fig,axes=plt.subplots(2,1,figsize=(9,6),layout='constrained');v=np.abs(a['y_m']-.05);row=v==min(v)
    order=np.argsort(a['x_m'][row]);xx=a['x_m'][row][order]
    axes[0].plot(xx,a['tau_xy_Pa'][row][order]/1000,label='before');axes[0].plot(xx,b['tau_xy_Pa'][row][order]/1000,'--',label='after insertion')
    axes[0].set(ylabel='physical τxy (kPa)',title='Nonzero loaded-state transfer: physical stress is unchanged');axes[0].legend()
    q=read(OUT/'insert-loaded/states.csv');t=.28;z=q[abs(q['time_s']-t)<1e-12]
    axes[1].plot(z['x_m'],z['jump_m']*1000,color='#2b6cb0');axes[1].set(ylabel='accepted jump (mm)',xlabel='x (m)')
    for ax in axes:
        ax.axvspan(.125,.25,color='#fbbf24',alpha=.3);ax.axvspan(.75,.875,color='#fbbf24',alpha=.3)
        for x in [.25,.75]:ax.axvline(x,color='gray',ls='--')
        ax.grid(alpha=.2)
    axes[1].text(.5,.8*max(z['jump_m'])*1000,'New tied records shaded; old endpoints dashed',ha='center',fontsize=9)
    save(fig,'transfer.png')
    profiles=np.genfromtxt(OUT/'profiles.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
    cases=['growth-64','growth-128','growth-128-half'];final=cases[-1] if np.any(profiles['case']==cases[-1]) else cases[0]
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True,layout='constrained')
    p=profiles[profiles['case']==final];times=np.unique(p['time_s']);selected=times[-min(6,len(times)):]
    for index,t in enumerate(selected):
        q=p[p['time_s']==t];color=plt.cm.tab10(index)
        for ax,key,scale in zip(axes,['damage','jump_m','shear_Pa'],[1,1000,.001]):ax.plot(q['x_m'],q[key]*scale,color=color,label=f'{t:.4f} s')
    axes[0].set(ylabel='surface damage a',ylim=(-.03,1.05),title='Cohesive-zone expansion and full decohesion are different fronts')
    axes[1].set_ylabel('geometric jump δ (mm)');axes[2].set_ylabel('weak shear traction (kPa)');axes[2].set_xlabel('x (m)');axes[2].axhline(3,color='r',ls=':',label='intact strength')
    for ax in axes:geometry(ax)
    axes[0].legend(ncol=3,fontsize=8);save(fig,'evolution.png')
    h=np.genfromtxt(OUT/'physical_histories.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
    comparison(h,cases)
    z=h[h['case']==final];fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained')
    for key,label in [('trapezoid_work_J_m','trapezoidal work'),('bulk_J_m','bulk storage'),('bond_J_m','bond storage'),('added_fracture_J_m','added fracture (initial 3 J/m excluded)'),('sliding_loss_J_m','contact + radiation loss')]:axes[0].plot(z['time_s'],z[key],label=label)
    axes[0].set_ylabel('energy per out-of-plane length (J/m)');axes[0].legend(fontsize=8)
    axes[1].plot(z['time_s'],z['signed_physical_remainder_J_m'],label='signed physical remainder');axes[1].plot(z['time_s'],z['absolute_physical_remainder_J_m'],label='sum of absolute step remainders');axes[1].plot(z['time_s'],z['ledger_error_J_m'],label='BE ledger error')
    axes[1].set(xlabel='time (s)',ylabel='remainder (J/m)');axes[1].legend();save(fig,'energy.png')
    if (OUT/'animation_frames.csv').exists():
        a=read(OUT/'animation_frames.csv');times=np.unique(a['time_s']);fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True,layout='constrained');lines=[]
        for ax,ylabel in zip(axes,['surface damage a','jump δ (mm)','weak shear (kPa)']):
            geometry(ax);line,=ax.plot([],[],color='#2b6cb0',lw=2);lines.append(line);ax.set_ylabel(ylabel)
        axes[0].set_ylim(-.03,1.15);strip=axes[0].scatter([],[],s=12,marker='s');axes[1].set_ylim(0,1.05*max(a['jump_m'])*1000);axes[2].set_ylim(0,max(3.2,1.05*max(a['shear_Pa'])*.001));axes[2].axhline(3,color='r',ls=':');axes[2].set_xlabel('x (m)');title=axes[0].set_title('')
        def update(k):
            q=a[a['time_s']==times[k]]
            for line,key,scale in zip(lines,['damage','jump_m','shear_Pa'],[1,1000,.001]):line.set_data(q['x_m'],q[key]*scale)
            strip.set_offsets(np.c_[q['x_m'],np.full(len(q),1.08)]);strip.set_color(np.where((q['x_m']<1)|(q['x_m']>7),'white',np.where(q['damage']==0,'#999999',np.where(q['damage']==1,'#2b6cb0','#e69f00'))))
            title.set_text(f't = {times[k]:.4f} s; imposed U = {q["U_m"][0]*1000:.4f} mm — fixed straight candidate')
            return lines+[title,strip]
        movie=FuncAnimation(fig,update,frames=len(times));movie.save(OUT/'evolution.gif',writer=PillowWriter(fps=12));plt.close(fig)
if __name__=='__main__':main()
