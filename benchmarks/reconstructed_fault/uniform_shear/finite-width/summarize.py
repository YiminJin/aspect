"""Read saved K4.1 evidence; do not perform additional reference solves."""
import json
from pathlib import Path

import numpy as np

from reference_check import Y, compare, dump

HERE=Path(__file__).resolve().parent
DATA=HERE/'k41'


def load(name,cells,dt):
    directory=DATA/f'{name}-{cells}'
    return json.loads((directory/f'dt{dt:g}.json').read_text()), np.load(directory/f'dt{dt:g}.npz')['velocity']


def main():
    initial={name:{n:json.loads((DATA/f'{name}-{n}/initialization.json').read_text())
                   for n in (2048,4096)} for name in ('ell0','half')}
    initial_comparison={}
    for key in ('Ih','support_half_width_m','activation_distance_m','H0_max',
                'localization_second_moment_m2','localization_rms_width_m'):
        values={name:initial[name][4096][key] for name in initial}
        initial_comparison[key]=dict(**values,half_minus_ell0=values['half']-values['ell0'],
            reference_grid_uncertainty=sum(abs(initial[name][4096][key]-initial[name][2048][key]) for name in initial))
    for key in ('phi_center','C0'):
        value=lambda name,n: (initial[name][n]['phi_range'][1] if key=='phi_center'
                              else initial[name][n]['retained_initial']['C'])
        initial_comparison[key]=dict(ell0=value('ell0',4096),half=value('half',4096),
            half_minus_ell0=value('half',4096)-value('ell0',4096),
            reference_grid_uncertainty=sum(abs(value(name,4096)-value(name,2048)) for name in initial))
    widths={}
    for dt in (.5,.25,.125):
        a,ua=load('ell0',4096,dt); b,ub=load('half',4096,dt)
        # The comparison helper supplies physical max/L2 velocity differences;
        # its quarter-allowance ratios are NOT a test that width effects vanish.
        widths[str(dt)]=compare(a,b,ua,ub)
    fine={name:load(name,4096,.125) for name in initial}
    coarse={name:load(name,4096,.25) for name in initial}
    rows=[]
    for t in (0.,.5,1.,2.,4.,6.):
        fi=round(t/.125); ci=round(t/.25)
        row=dict(time_s=t,metrics={})
        for key in ('V','q','C','Theta','slip','crack_integral'):
            diff=fine['half'][0][fi][key]-fine['ell0'][0][fi][key]
            previous=coarse['half'][0][ci][key]-coarse['ell0'][0][ci][key]
            conservative=sum(abs(fine[n][0][fi][key]-coarse[n][0][ci][key]) for n in initial)
            row['metrics'][key]=dict(half_minus_ell0=diff,
                temporal_change_of_width_difference=abs(diff-previous),
                conservative_temporal_change_sum=conservative)
        du=fine['half'][1][fi]-fine['ell0'][1][fi]
        du_old=coarse['half'][1][ci]-coarse['ell0'][1][ci]
        row['metrics']['velocity']=dict(max_difference=float(np.max(abs(du))),
            L2_difference=float(np.sqrt(np.trapezoid(du**2,Y))),
            temporal_change_of_width_difference=float(np.max(abs(du-du_old))),
            conservative_temporal_change_sum=sum(float(np.max(abs(fine[n][1][fi]-coarse[n][1][ci]))) for n in initial))
        rows.append(row)
        np.savetxt(DATA/f'velocity-t{t:g}.csv',np.c_[Y,fine['ell0'][1][fi],fine['half'][1][fi],du],
                   delimiter=',',header='y,u_ell0,u_half,half_minus_ell0',comments='')
    report=dict(initial=initial_comparison,width_differences=widths,
                finest_tested_dt_s=.125,common_times=rows,
                caution='Temporal changes are empirical uncertainty indicators, not asymptotic error bounds. No common timestep passed.')
    dump(DATA/'width-comparison.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for name,label in (('ell0','ell = 0.15625 m'),('half','ell = 0.078125 m')):
        fields=np.load(DATA/f'{name}-4096/profile.npz')
        ell=initial[name][4096]['ell']
        axes[0,0].plot(Y,fields['phi'],label=label)
        axes[0,1].plot(Y/ell,fields['phi'],label=label)
        axes[1,0].semilogy(Y,fields['H'],label=label)
        axes[1,1].plot(Y,fine[name][1][-1],label=label)
    for ax in axes.flat:
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    axes[0,0].set(xlabel='y [m]',ylabel='initial phi')
    axes[0,1].set(xlabel='y / ell',ylabel='initial phi',xlim=(-3,3))
    axes[1,0].set(xlabel='y [m]',ylabel='prescribed H0 [Pa]')
    axes[1,1].set(xlabel='y [m]',ylabel='u_x at 6 s [m/s]',title='dt = 0.125 s; not temporally qualified')
    fig.savefig(DATA/'width-profiles.png',dpi=160)
    plt.close(fig)
    print(json.dumps(dict(initial=initial_comparison,final=rows[-1]),indent=2))


if __name__=='__main__': main()
