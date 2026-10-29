#!/usr/bin/env python3
"""One spatial and one boundary-distance comparison of native physical fields."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from continuation_analysis import BASE, OUT, LEVELS, compare, read, save_csv, crossing
from continuation_plots import load, COLORS, save

NAMES = ['reference-continuation', 'spatial-complete', 'domain-complete']
LABELS = ['96 x 128; width 1.5 m', '192 x 128; width 1.5 m', '112 x 128; width 1.75 m']


def state(d, t):
    k = np.argmin(abs(d['times']-t))
    assert abs(d['times'][k]-t) < 1e-10
    return k


def main():
    results = {}
    table = []
    for name, label in zip(NAMES, LABELS):
        d = load(name)
        h = read(OUT/(name+'-history.csv'))
        c = read(OUT/(name+'-contours.csv'))
        b = read(OUT/(name+'-background.csv'))
        # A resumed original case starts late; its complete original prefix
        # supplies the onset bracket without fabricating an earlier state.
        onset_h = read(OUT/'original-history.csv') if name == NAMES[0] else h
        first = np.flatnonzero(onset_h['max_total_phase_increment'] > 1e-10)[0]
        onset = [float(onset_h['time_s'][first-1]), float(onset_h['time_s'][first])]
        ids = np.flatnonzero((h['time_s'] > .15875+1e-12) & (h['time_s'] <= .17875+1e-12))
        k = state(d, .17875)
        # Lumped Q1 nodal mass integrates Q1 phase exactly on these meshes.
        common = d['x'] <= 1.5+1e-12
        # Integrate the added strip separately, including its half boundary
        # masses, rather than cutting a nodal mass vector in half incorrectly.
        extra = d['x'] >= 1.5-1e-12
        def integral(values, xmask):
            a = values[:, xmask]
            if a.shape[1] < 2:
                return 0.
            dx = d['x'][1]-d['x'][0]; dy = d['y'][1]-d['y'][0]
            m = np.full_like(a, dx*dy); m[[0,-1]] *= .5; m[:,[0,-1]] *= .5
            return float(np.sum(m*a))
        center = c[(abs(c['time_s']-.17875) < 1e-10) & (c['ray_angle_deg'] == 0)]
        front = {str(v): float(center['x_m'][center['phi_contour'] == v][0]) for v in LEVELS}
        mid=np.argmin(abs(d['y']-.5))
        initial_front={str(v):crossing(d['x'],d['initial'][mid],v) for v in LEVELS}
        initial_name={'reference-continuation':'original','spatial-complete':'spatial-load','domain-complete':'domain-corrected-load'}[name]
        initial = json.load(open(OUT/(initial_name+'-summary.json')))
        res = dict(label=label, onset_bracket_s=onset,
                   cells=(len(d['x'])-1)*(len(d['y'])-1),
                   dx_m=float(d['x'][1]-d['x'][0]), dy_m=float(d['y'][1]-d['y'][0]),
                   initial_representation=initial['initial_representation'],
                   unloaded_phase_drift=initial['unloaded_phase_drift'],
                   final_centerline_fronts_m=front,
                   initial_centerline_fronts_m=initial_front,
                   final_centerline_advances_m={v:front[v]-initial_front[v] for v in front},
                   final_common_domain_growth_m2=integral(d['growth'][k], common),
                   final_added_strip_growth_m2=integral(d['growth'][k], extra),
                   max_step_phase_change=float(h['max_step_phase_increment'][ids].max()),
                   hold_max_phase_change=float(h['max_step_phase_increment'][h['time_s']>.175+1e-12].max()),
                   final_reaction_N_m=float(h['shear_reaction_N_m'][-1]),
                   final_reaction_per_width_Pa=float(b['actual_reaction_per_width_Pa'][-1]),
                   final_seed_excess_reaction_N_m=float(b['seed_excess_reaction_N_m'][-1]),
                   final_bulk_J_m=float(h['bulk_J_m'][-1]),
                   final_seed_excess_bulk_J_m=float(b['seed_excess_bulk_J_m'][-1]),
                   final_cohesive_J_m=float(h['cohesive_J_m'][-1]),
                   final_added_fracture_J_m=float(h['fracture_J_m'][-1]-read(OUT/('original-history.csv' if name == NAMES[0] else name+'-history.csv'))['fracture_J_m'][0]))
        results[name] = res
        table.append([name, res['cells'], res['dx_m'], res['dy_m'], *onset,
                      res['max_step_phase_change'], res['hold_max_phase_change'],
                      res['final_common_domain_growth_m2'], res['final_added_strip_growth_m2'],
                      *front.values(), res['final_reaction_N_m'], res['final_seed_excess_reaction_N_m'],
                      res['final_added_fracture_J_m']])
    for other, label in [('spatial-complete','spatial-comparison'), ('domain-complete','domain-comparison')]:
        compare(NAMES[0], other, label)
        a = read(OUT/(label+'.csv')); a = a[(a['time_s']>=.15875-1e-12) & (a['time_s']<=.17875+1e-12)]
        ca = read(OUT/(NAMES[0]+'-contours.csv')); cb = read(OUT/(other+'-contours.csv'))
        rows=[]
        for x in ca:
            if not .15875-1e-12 <= x['time_s'] <= .17875+1e-12:
                continue
            hit=cb[(abs(cb['time_s']-x['time_s'])<1e-10) & (cb['phi_contour']==x['phi_contour']) & (cb['ray_angle_deg']==x['ray_angle_deg'])]
            if len(hit):
                rows.append([x['time_s'],x['imposed_relative_displacement_m'],x['ray_angle_deg'],x['phi_contour'],x['x_m'],hit['x_m'][0],x['x_m']-hit['x_m'][0]])
        save_csv(OUT/(label+'-fronts.csv'), ['time_s','imposed_relative_displacement_m','ray_angle_deg','phi_contour','reference_x_m','control_x_m','difference_m'], rows)
        rows=np.asarray(rows); primary=rows[(rows[:,2]==0)&(rows[:,3]==.1)]
        integral=abs(a['growth_integral_a_m2']-a['growth_integral_b_m2'])/np.maximum(1e-100,abs(a['growth_integral_b_m2']))
        results[label]=dict(final={key:float(a[key][-1]) for key in a.dtype.names},
                            maximum_growth_relative_L2=float(a['growth_relative_L2'].max()),
                            maximum_growth_relative_peak=float(a['growth_relative_peak_difference'].max()),
                            maximum_growth_integral_relative_difference=float(integral.max()),
                            maximum_primary_centerline_front_difference_m=float(abs(primary[:,6]).max()),
                            practical_targets_passed=bool(a['growth_relative_L2'].max()<=.05 and a['growth_relative_peak_difference'].max()<=.05 and integral.max()<=.05 and abs(primary[:,6]).max()<=.015625))
    save_csv(OUT/'spatial-domain-table.csv', ['case','cells','dx_m','dy_m','onset_lower_s','onset_upper_s','maximum_step_dphi','hold_maximum_dphi','common_domain_growth_m2','added_strip_growth_m2',*[f'centerline_phi_{v:g}_x_m' for v in LEVELS],'reaction_N_m','seed_excess_reaction_N_m','added_fracture_J_m'],table)
    (OUT/'spatial-domain-results.json').write_text(json.dumps(results,indent=2)+'\n')

    ds=[load(n) for n in NAMES]
    fig,axs=plt.subplots(3,3,figsize=(15,10),layout='constrained')
    k=[state(d,.175) for d in ds]
    maxgrowth=max(d['growth'][i].max() for d,i in zip(ds,k))
    maxstress=max(d['stress'][i].max() for d,i in zip(ds,k))
    minstress=min(0.,min(d['stress'][i].min() for d,i in zip(ds,k)))
    for j,(d,i,label) in enumerate(zip(ds,k,LABELS)):
        for row,(value,title,hi,cmap) in enumerate([(d['initial'],'Fresh continuum-seed representation',.6,'viridis'),(d['growth'][i],'Phase change from own initial field',maxgrowth,'magma'),(d['stress'][i],'Physical cell-mean tau_xy [Pa]',maxstress,'viridis')]):
            ax=axs[row,j];m=ax.pcolormesh(d['x'],d['y'],value,shading='gouraud' if row<2 else 'flat',vmin=0 if row<2 else minstress,vmax=hi,cmap=cmap)
            ax.contour(d['x'],d['y'],d['initial'] if row==0 else d['phase'][i],levels=[.01,.1,.3,.5],colors='white',linewidths=.5)
            ax.plot([0,1],[.5,.5],color='cyan',lw=.8);ax.axvline(1.5,color='.5',ls=':',lw=.7)
            ax.set(xlim=(0,1.75),ylim=(0,1),xlabel='x [m]',ylabel='y [m]',title=label+'\n'+title);ax.set_aspect('equal')
            fig.colorbar(m,ax=ax,shrink=.75)
    save(fig,'spatial-domain-fields')

    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for j,(name,label) in enumerate(zip(NAMES,LABELS)):
        h=read(OUT/(name+'-history.csv'));c=read(OUT/(name+'-contours.csv'));b=read(OUT/(name+'-background.csv'))
        mask=(h['time_s']>=.15875-1e-12)&(h['time_s']<=.17875+1e-12);h=h[mask];b=b[mask]
        for level,style in [(.1,'-'),(.3,'--'),(.5,':')]:
            a=c[(c['ray_angle_deg']==0)&(c['phi_contour']==level)&(c['time_s']>=.15875-1e-12)&(c['time_s']<=.17875+1e-12)]
            axs[0,0].plot(a['time_s'],a['x_m'],ls=style,color=COLORS[j],label=label+f'; phi={level:g}')
            axs[0,1].plot(1000*a['imposed_relative_displacement_m'],a['x_m'],ls=style,color=COLORS[j])
        axs[0,2].plot(h['time_s'],h['integrated_phase_growth_m2'],color=COLORS[j],label=label)
        axs[1,0].plot(h['time_s'],h['max_step_phase_increment'],color=COLORS[j],label=label)
        axs[1,1].plot(1000*h['imposed_relative_displacement_m'],b['actual_reaction_per_width_Pa'],color=COLORS[j],label=label)
        axs[1,2].plot(1000*h['imposed_relative_displacement_m'],b['seed_excess_reaction_N_m'],color=COLORS[j],label=label)
    for ax,x,y in zip(axs.flat,['Time [s]','Relative grip displacement [mm]','Time [s]','Time [s]','Relative grip displacement [mm]','Relative grip displacement [mm]'],['Centerline first exit x [m]','Centerline first exit x [m]','Integrated phase growth [m2]','Maximum accepted-step phase increment','Reaction / specimen width [Pa]','Reaction minus intact background [N/m]']):
        ax.set(xlabel=x,ylabel=y);ax.grid(alpha=.2)
        if x=='Time [s]':
            ax.xaxis.set_major_locator(MaxNLocator(5));ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        if ax!=axs[0,1]:ax.legend(fontsize=7)
    save(fig,'spatial-domain-response')

    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for j,(name,label) in enumerate(zip(NAMES,LABELS)):
        p=read(OUT/(name+'-profiles.csv'));p=p[abs(p['time_s']-.175)<1e-10]
        for ax,key,title in zip(axs.flat,['phase_increment','kernel_tau_xy_Pa','kernel_Y_Pa','compatible_weak_fracture_Pa','compatible_weak_mechanical_Pa','compatible_weak_residual_Pa'],['Centerline phase increment','Physical shear stress [Pa]','Point mechanical driving Y [Pa]','Compatible fracture force [Pa]','Compatible mechanical force [Pa]','Compatible total weak residual [Pa]']):
            ax.plot(p['x_m'],p[key],color=COLORS[j],label=label);ax.set(xlabel='x [m]',ylabel=title);ax.grid(alpha=.2);ax.axvline(1,color='.6',ls=':');ax.legend(fontsize=7)
    fig.suptitle('t=.175 s; common physical kernel radius .03125 m; forces use common nodal-mass averaging')
    save(fig,'spatial-domain-profiles')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
