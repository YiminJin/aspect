#!/usr/bin/env python3
"""Reduce existing cohesive endpoint diagnostics; never run a simulation."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
RUNS=ROOT/'build-particle-free/transported-objective/stage-c/runs'
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_c_coupled'
CASES=['cut-native-phase8','cut-phase16','cut-phase32']
TERMS=['endpoint_minus_trapezoid','BE_bulk','surface_endpoint','quadrature_repartition',
       'objective_projection','ledger','physical_corrected']


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cases=[];inputs={}
    for name in CASES:
        directory=RUNS/name/'output';series=[]
        for path in sorted(directory.glob('cohesive-*-diagnostics.csv')):
            d={r['quantity']:float(r['value']) for r in csv.DictReader(path.open())}
            inputs[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
            if d.get('dt_s',0.)<=0:continue
            row=dict(time_s=d['time_s'],dt_s=d['dt_s'],
              endpoint_minus_trapezoid=d['endpoint_work_J_m']-d['trapezoid_work_J_m'],
              BE_bulk=d['BE_bulk_J_m'],surface_endpoint=d['surface_remainder_J_m'],
              quadrature_repartition=d['quadrature_repartition_J_m'],
              objective_projection=d.get('objective_projection_J_m',0.),ledger=d['ledger_error_J_m'],
              physical_corrected=d['physical_remainder_with_objective_J_m'],
              fracture=d['fracture_increment_J_m'],endpoint_work=d['endpoint_work_J_m'],
              friction=d['friction_J_m'],radiation=d['radiation_J_m'],maxwell=d['maxwell_J_m'],
              transfer_stress_Pa=d.get('transfer_stress_Pa',0.))
            row['decomposition_defect']=row['physical_corrected']-(row['BE_bulk']+row['surface_endpoint']
                -row['endpoint_minus_trapezoid']+row['ledger']-row['quadrature_repartition']-row['objective_projection'])
            series.append(row)
        sums={key:dict(signed=sum(r[key] for r in series),absolute=sum(abs(r[key]) for r in series))
              for key in TERMS+['fracture','endpoint_work','friction','radiation','maxwell']}
        command=json.loads((RUNS/name/'command.json').read_text())
        last={r['quantity']:float(r['value']) for r in csv.DictReader(sorted(directory.glob('cohesive-*-diagnostics.csv'))[-1].open())}
        c=dict(case=name,steps=len(series),dt_s=series[0]['dt_s'],sums_J_m=sums,series=series,
          max_decomposition_defect_J_m=max(abs(r['decomposition_defect']) for r in series),
          max_reported_transfer_stress_Pa=max(r['transfer_stress_Pa'] for r in series),
          fracture_relative_signed=sums['physical_corrected']['signed']/sums['fracture']['signed'],
          fracture_relative_absolute=sums['physical_corrected']['absolute']/sums['fracture']['signed'],
          endpoint_work_relative_signed=sums['physical_corrected']['signed']/sums['endpoint_work']['signed'],
          cost=dict(wall_s=command['elapsed_s'],bulk_solves=last['bulk_solves_cumulative'],
                    setup_s=last['setup_seconds_cumulative'],history_s=last['history_seconds_cumulative'],
                    surface_s=last['surface_seconds_cumulative']))
        assert c['max_decomposition_defect_J_m']<1e-11
        cases.append(c)
    orders={key:[math.log(abs(cases[i]['sums_J_m'][key]['signed']/cases[i+1]['sums_J_m'][key]['signed']),2)
                 for i in (0,1)] for key in ['BE_bulk','surface_endpoint','endpoint_minus_trapezoid','physical_corrected']}
    result=dict(source='existing outputs only; unchanged endpoint law, loads and tolerances',
      units='energy per out-of-plane length J/m; transfer_stress_Pa is not an energy',cases=cases,
      cumulative_timestep_orders=orders,input_sha256=inputs,
      identity='physical_corrected = BE_bulk + surface_endpoint - (endpoint_work-trapezoid_work) + ledger - quadrature_repartition - objective_projection',
      cost_extrapolation=dict(method='first-order continuation of finest observed error; not a measured qualification',
        target_fracture_relative=.01,estimated_steps=math.ceil(32*cases[-1]['fracture_relative_absolute']/.01)))
    (OUT/'cohesive-energy-decomposition.json').write_text(json.dumps(result,indent=2)+'\n')
    with (OUT/'cohesive-energy-decomposition.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['case','steps','dt_s','term','signed_J_m','absolute_J_m'])
        for c in cases:
            for key,v in c['sums_J_m'].items():writer.writerow([c['case'],c['steps'],c['dt_s'],key,v['signed'],v['absolute']])
    os.environ.setdefault('MPLCONFIGDIR',str(OUT/'matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for key in ['BE_bulk','surface_endpoint','endpoint_minus_trapezoid','physical_corrected']:
        axes[0,0].loglog([c['dt_s'] for c in cases],[abs(c['sums_J_m'][key]['signed']) for c in cases],'o-',label=key)
    axes[0,0].set(xlabel='Timestep (s)',ylabel='Absolute signed cumulative energy (J/m)',title='Endpoint contributions')
    c=cases[-1]
    for key in ['BE_bulk','surface_endpoint','endpoint_minus_trapezoid','physical_corrected']:
        axes[0,1].plot([r['time_s'] for r in c['series']],
          [sum(z[key] for z in c['series'][:i+1]) for i in range(len(c['series']))],label=key)
    axes[0,1].set(xlabel='Physical time (s)',ylabel='Signed cumulative energy (J/m)',title='Finest retained run')
    for kind in ['signed','absolute']:
        axes[1,0].loglog([c['dt_s'] for c in cases],
          [abs(c['sums_J_m']['physical_corrected'][kind])/c['sums_J_m']['fracture']['signed'] for c in cases],'o-',label=kind)
    axes[1,0].axhline(.01,color='0.5',ls=':',label='retained 1% target')
    axes[1,0].set(xlabel='Timestep (s)',ylabel='Remainder / new fracture work',title='Cancellation check')
    for key in ['quadrature_repartition','objective_projection','ledger']:
        axes[1,1].semilogy([c['steps'] for c in cases],[max(c['sums_J_m'][key]['absolute'],1e-30) for c in cases],'o-',label=key)
    axes[1,1].set(xlabel='Accepted steps over 0.001 s',ylabel='Sum of absolute defects (J/m)',title='Projection and arithmetic defects')
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(OUT/'cohesive-energy-decomposition.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(orders=orders,rows=[dict(case=c['case'],steps=c['steps'],signed=c['sums_J_m']['physical_corrected']['signed'],absolute=c['sums_J_m']['physical_corrected']['absolute'],fracture_relative=c['fracture_relative_absolute'],cost=c['cost']) for c in cases],extrapolation=result['cost_extrapolation']),indent=2))


if __name__=='__main__':main()
