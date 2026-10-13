#!/usr/bin/env python3
"""Unshifted endpoint-equivalence and independent residual-work/cost gates."""
import json

import numpy as np

import cheaper_coupling as test
import assess_single_pass as comparison
from assess_coupled_branch import totals


def assess():
    p = test.Measured()
    ref = {d:comparison.load(test.native.E/f'dt-1-{d}') for d in [1024,2048]}
    candidate = {d:comparison.load(test.E/f'dt-1-{d}') for d in [1024,2048]}
    slip_scale = ref[2048]['history'][-1]['slip']
    rate_scale = max(row['V'] for row in ref[2048]['history'])
    same, controls, energies = {}, {}, {}
    for d in [1024,2048]:
        for key,value in ref[d]['initial'].items():
            assert np.array_equal(value,candidate[d]['initial'][key]),key
        controls[d] = comparison.controls(candidate[d])
        same[d] = comparison.comparison(p.r,candidate[d],ref[d],slip_scale,rate_scale)
        energies[d] = dict(candidate=totals(comparison.read(candidate[d]['path']/'energy.json')),
                           endpoint=totals(comparison.read(ref[d]['path']/'energy.json')))
    temporal = comparison.comparison(p.r,candidate[1024],candidate[2048],slip_scale,rate_scale)
    for result in [*same.values(),temporal]:
        assert all(result['passes'].values()),result['maximum']
    result = dict(same_schedule=same,temporal=temporal,controls=controls,energy=energies,
                  initial_fields_identical=True,global_reflection_applied=False,
                  stress_scale_Pa=comparison.STRESS,storage_scale_J_per_m=comparison.STORAGE,
                  slip_scale_m=slip_scale,peak_rate_m_per_s=rate_scale)
    test.core.write(test.E/'assessment.json',result)
    print(json.dumps(dict(same={d:x['maximum'] for d,x in same.items()},
                          temporal=temporal['maximum'],passed=True)),flush=True)
    plot(candidate,ref)


def plot(cases,refs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    fig2,energy_axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for d in [1024,2048]:
        for data,name,style in [(cases[d],'quasi-Newton','-'),(refs[d],'endpoint','--')]:
            f,h=data['fields'],data['history'];t=f['time'];label=f'{name} 1/{d}'
            axes[0,0].plot(t,f['phase'][:,256],style,label=label)
            axes[0,1].plot([v['slip'] for v in h],[v['q'] for v in h],style,label=label)
            axes[0,2].plot(t[1:],[1000*(v['centroid']['phase']-.5) for v in h],style,label=label)
            if d == 2048:
                for time,color in zip([.53125,.59375,.625],['tab:blue','tab:orange','tab:green']):
                    j=int(np.flatnonzero(t==time)[0]);axes[1,0].plot(f['y'],f['phase'][j],style,
                         color=color,label=f't={time:g}' if name=='quasi-Newton' else None)
            axes[1,1].plot(t[1:],[v['bulk'] for v in h],style,label=label)
            if 'counts' in h[0]:
                axes[1,2].plot(t[1:],[v['counts']['mechanical_evaluations'] for v in h],style,label=label)
            ledger=comparison.read(data['path']/'energy.json')
            for ax,key in zip(energy_axes.flat,['external_remainder','mechanical_power','friction_residual_power',
                                              'phase_power','mechanical_power_absolute','phase_power_absolute']):
                ax.plot(t[1:],np.cumsum([v[key] for v in ledger]),style,label=label)
                ax.set(title=key,xlabel='t (s)',ylabel='J/m')
    for ax,title,xlabel,ylabel in zip(axes.flat,['Core phase','Traction–slip','Centroid','Unaligned profiles',
                                               'Bulk storage','Mechanical evaluations per step'],
           ['t (s)','slip (m)','t (s)','y (m)','t (s)','t (s)'],
           ['phase','q (Pa)','shift (mm)','phase','J/m','calls']):
        ax.set(title=title,xlabel=xlabel,ylabel=ylabel)
    for a in [*axes.flat,*energy_axes.flat]:
        a.grid(alpha=.2);a.legend(fontsize=6)
    axes[1,0].set_title('Profiles, 1/2048: QN solid, reference dashed')
    fig.savefig(test.E/'physical-comparison.png',dpi=150)
    fig2.savefig(test.E/'separate-energy.png',dpi=150)
    plt.close(fig);plt.close(fig2)


if __name__ == '__main__':
    assess()
