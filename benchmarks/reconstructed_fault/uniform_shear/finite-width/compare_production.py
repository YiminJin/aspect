"""Conservative K4.2 width-versus-discretization accounting on [4,6] s."""
import json
import numpy as np
from reference_check import HERE,Y,dump
from production import read


def main():
    cases={c:json.loads((HERE/f'k42_{c}-analysis.json').read_text()) for c in 'ABC'}
    assert all(c['completed'] and c['residuals']['passed'] for c in cases.values())
    profiles={c:np.load(HERE/f'k42_{c}-profiles.npz')['velocity'] for c in 'ABC'}
    ref={n:json.loads((HERE/'k41'/f'{n}-4096'/'dt0.125.json').read_text()) for n in ('ell0','half')}
    refu={n:np.load(HERE/'k41'/f'{n}-4096'/'dt0.125.npz')['velocity'] for n in ref}
    widths=json.loads((HERE/'k41b/comparison.json').read_text())['new_pair']['rows']
    spatial=json.loads((HERE/'k41/decision.json').read_text())['spatial']
    rows=[]
    for step in range(32,49):
        data={c:cases[c]['rows'][step] for c in cases}
        metrics={}
        for key in ('V','q','C','Theta','slip','Ih','supported_integral','velocity_max','velocity_L2'):
            velocity=key.startswith('velocity')
            norm=lambda v:float(np.sqrt(np.trapezoid(np.asarray(v)**2,Y))) if key=='velocity_L2' else float(np.max(abs(v)))
            obs={c:profiles[c][step] if velocity else data[c]['observed'][key] for c in cases}
            exact={n:refu[n][step] if velocity else ref[n][step][key] for n in ref}
            target=exact['half']-exact['ell0']
            width_B=obs['B']-obs['A']; width_C=obs['C']-obs['A']
            errA=norm(obs['A']-exact['ell0'])
            errB=norm(obs['B']-exact['half'])
            errC=norm(obs['C']-exact['half'])
            width_time=widths[step]['metrics'][key]['matched_temporal_change']
            rk='velocity' if key=='velocity_max' else key
            if key=='Ih':
                ref_grid=sum(abs(json.loads((HERE/'k41'/f'{n}-2048'/'initialization.json').read_text())['Ih']
                                 -json.loads((HERE/'k41'/f'{n}-4096'/'initialization.json').read_text())['Ih']) for n in ref)
            elif key=='supported_integral':
                ref_grid=0.
                for n in ref:
                    coarse=json.loads((HERE/'k41'/f'{n}-2048'/'dt0.5.json').read_text())
                    fine=json.loads((HERE/'k41'/f'{n}-4096'/'dt0.5.json').read_text())
                    ref_grid+=max(abs(a[key]-b[key]) for a,b in zip(coarse,fine))
            else:
                ref_grid=sum(spatial[n]['maxima'][rk] for n in ref)
            # No favorable cancellation is assumed between production errors.
            uncertainty_B=errA+errB+width_time+ref_grid
            uncertainty_C=errA+errC+width_time+ref_grid
            metrics[key]=dict(reference_width_signal=norm(target),
                finer_reference_width_signal=abs(widths[step]['metrics'][key]['fine_width_difference']),
                production_same_mesh_width=norm(width_B) if velocity else width_B,
                production_refined_half_width=norm(width_C) if velocity else width_C,
                same_mesh_width_error=norm(width_B-target),refined_width_error=norm(width_C-target),
                production_errors=dict(A=errA,B=errB,C=errC),
                half_width_normal_refinement=norm(obs['C']-obs['B']),
                reference_grid_uncertainty=ref_grid,width_temporal_uncertainty=width_time,
                conservative_uncertainty_B=uncertainty_B,conservative_uncertainty_C=uncertainty_C,
                signal_over_uncertainty_B=norm(target)/uncertainty_B,
                signal_over_uncertainty_C=norm(target)/uncertainty_C,
                distinguishable_B=norm(target)>=4*uncertainty_B,
                distinguishable_C=norm(target)>=4*uncertainty_C)
        rows.append(dict(time_s=step*.125,metrics=metrics))
    summary={key:dict(
        min_signal_over_uncertainty_B=min(r['metrics'][key]['signal_over_uncertainty_B'] for r in rows),
        min_signal_over_uncertainty_C=min(r['metrics'][key]['signal_over_uncertainty_C'] for r in rows),
        distinguishable_C_all_times=all(r['metrics'][key]['distinguishable_C'] for r in rows),
        max_errors={c:max(r['metrics'][key]['production_errors'][c] for r in rows) for c in 'ABC'},
        max_BC_change=max(r['metrics'][key]['half_width_normal_refinement'] for r in rows))
        for key in rows[0]['metrics']}
    initial={}
    for c in cases:
        particles=read(HERE/f'k42_{c}','particles',0)
        initial[c]={**cases[c]['initial'],
                    'sampled_H0_range':[float(min(particles['H'])),float(max(particles['H']))]}
    report=dict(accuracy_interval=[4,6],original_K41_all_time_pass=False,
                empirical_uncertainty_not_rigorous_bound=True,summary=summary,rows=rows,
                initial=initial,
                individual_checks={c:cases[c]['post_transient_checks'] for c in cases},
                individual_fine_reference_checks={c:cases[c]['post_transient_fine_reference_checks'] for c in cases})
    dump(HERE/'k42-comparison.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__': main()
