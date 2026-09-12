"""Preserve the failed K3 spatial gate and summarize accepted/reference evidence."""
import csv
import json
from pathlib import Path
import re

import numpy as np

from reference import read_parameters


here = Path(__file__).resolve().parent
out = here/'periodic-convergence'
spatial = []
for normal in (128,256):
    name = f'spatial0375_n{normal}_f32_periodic'
    data = json.loads((here/f'{name}-comparison.json').read_text())
    for state in data['states'][:2]:
        k = state['step']
        h = np.genfromtxt(here/name/f'comparison_H_{k}.csv',delimiter=',',names=True)
        row = dict(normal_cells=normal,step=k,time=state['time_s'],
            phi_error_max=state['phi_max_absolute_error'],
            H_error_max=state['H_profile_max_absolute_error'],
            H_error_rms=float(np.sqrt(np.trapezoid(h['difference']**2,h['y'])/(h['y'][-1]-h['y'][0]))),
            H_max_error_y=float(h['y'][np.argmax(abs(h['difference']))]),
            H_feedback_max=state.get('H_increment_max_Pa',0.),
            phi_increment_error=state['phi_increment_profile_max_error'],
            Ih=state['surface_means']['Ih'],
            Ih_error=state['surface_means']['Ih']-state['reference']['Ih'],
            Ih_increment=state['surface_means']['Ih']-data['states'][0]['surface_means']['Ih'],
            q_error_rms=state['raw_stress_error_rms_Pa'],q_error_max=state['raw_stress_error_max_Pa'],
            slip=state['accumulated_slip_m'],slip_reference=state['reference_slip_m'],
            containment=state['guard']['max_omitted_fraction'],
            normalization=state['guard']['max_supported_normalization_error'],
            all_state_guards=state['guard']['passed'])
        for field in ('V','Theta','C'):
            row[field]=state['surface_means'][field]
            row[field+'_error']=state['surface_means'][field]-state['reference'][field]
        spatial.append(row)
with (out/'accepted-spatial.csv').open('w') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(spatial[0]))
    writer.writeheader()
    writer.writerows(spatial)

reference=[]
for dt,tag in ((.375,'0375'),(.1875,'01875'),(.09375,'009375')):
    directory=here/'spatial0375_n128_f32-reference' if dt==.375 else out/f'reference{tag}'
    coarse=json.loads((directory/'report.json').read_text())
    fine_dir=out/f'reference{tag}-4096'
    fine=json.loads((fine_dir/'report.json').read_text())
    k=coarse['completed_real_steps']
    phi0=np.loadtxt(directory/'phase-0.csv',delimiter=',',skiprows=1)
    phi=np.loadtxt(directory/f'phase-{k}.csv',delimiter=',',skiprows=1)
    H0=np.loadtxt(directory/'history-0.csv',delimiter=',',skiprows=1)
    H=np.loadtxt(directory/f'history-{k}.csv',delimiter=',',skiprows=1)
    feedback=dict(H=float(np.max(H[:,1]-H0[:,1])),phi=float(np.max(abs(phi[:,1]-phi0[:,1]))),
                  Ih=coarse['steps'][-1]['Ih']-coarse['initial']['Ih'])
    row=dict(dt=dt,steps=k,feedback=feedback,fine_feedback=fine['cumulative_feedback'],
        feedback_reference_resolution_difference={key:abs(value-fine['cumulative_feedback'][key]) for key,value in feedback.items()},
        max_candidate_ratio_4096=max(r['max_candidate_over_old_H'] for r in fine['maximum_rule_audit']),
        max_normalization=max(r['supported_slip_normalization_error'] for r in coarse['steps']),
        max_omission=max(r['omitted_fraction'] for r in coarse['steps']),
        max_phi=max(r['phi_max'] for r in coarse['steps']),
        final={field:coarse['steps'][-1][field] for field in ('V','Theta','C','q','Ih')},
        slip=sum(s['dt_s']*s['V'] for s in coarse['steps']))
    reference.append(row)

case='spatial0375_n256_f32_periodic'
log=(here/f'{case}.log').read_text()
failed=log.split('*** Timestep 2:')[1]
iterations=[dict(iteration=int(k),linear_iterations=int(l),line_search=int(s),relative=float(r))
    for k,l,s,r in re.findall(r'Iteration\s+(\d+)\s*: linear solver iterations =\s*(\d+)\s*, line search iterations = (\d+), relative residual = (\S+)',failed)]
entry=np.genfromtxt(here/case/'phase_probe_2.csv',delimiter=',',names=True)
initial=float(entry['R_old_phi_Hprevious'])
old_parameters=read_parameters(here/'spatial0375_n128_f32_periodic/parameters.prm')
new_parameters=read_parameters(here/case/'parameters.prm')
diff={k:[old_parameters.get(k),new_parameters.get(k)] for k in old_parameters.keys()|new_parameters.keys()
      if old_parameters.get(k)!=new_parameters.get(k)}
assert set(diff)=={'Geometry model/Box/Y repetitions','Output directory'}
summary=dict(gate_K3='unmet: new phase nonlinear exhaustion before spatial/temporal convergence can be established',
    production_parameter_differences=diff,spatial_accepted_prefix=spatial,independent_temporal_reference=reference,
    failed_phase=dict(step=2,time=.75,entry_residual=initial,tolerance=1e-8,
        implied_absolute_target=initial*1e-8,final_logged_relative=iterations[-1]['relative'],
        implied_final_absolute_from_rounded_log=initial*iterations[-1]['relative'],iterations=iterations,
        forced_failure_restoration=bool(entry['forced_failure_restored']),stable_ids=bool(entry['stable_ids']),
        cause='unresolved: absolute residual is consistent with numerical floor, but cancellation versus represented-update/linear-operator error was not measured'),
    resources=json.loads((here/f'{case}.resources.json').read_text()),
    unperformed=['32x512 production','production temporal refinement','production correction'])
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
# Plot only accepted production states. Missing later fine states must remain
# absent, not be replaced by the reference or a failed Newton candidate.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2,2,figsize=(10,7))
for normal in (128,256):
    case=here/f'spatial0375_n{normal}_f32_periodic'
    phi=np.genfromtxt(case/'comparison_phase_1.csv',delimiter=',',names=True)
    H=np.genfromtxt(case/'comparison_H_1.csv',delimiter=',',names=True)
    axes[0,0].plot(phi['y'],phi['difference'],label=f'32x{normal}')
    axes[0,1].plot(phi['y'],phi['increment_error'],label=f'32x{normal}')
    axes[1,0].plot(H['y'],H['difference'],label=f'32x{normal}')
for row,tag in zip(reference,('0375','01875','009375')):
    directory=here/'spatial0375_n128_f32-reference' if tag=='0375' else out/f'reference{tag}'
    data=json.loads((directory/'report.json').read_text())
    axes[1,1].plot([0]+[s['time_s'] for s in data['steps']],
        [0]+[s['Ih']-data['initial']['Ih'] for s in data['steps']],
        label=f"reference dt={row['dt']}")
for ax,title,ylabel in zip(axes.ravel(),
        ('Accepted phi error, t=.375 s','Accepted phi increment error, t=.375 s',
         'Accepted H input error, t=.375 s','Independent temporal feedback'),
        ('phi - reference','Delta phi - reference','H - reference [Pa]','Ih - Ih0 [m]')):
    ax.set_title(title); ax.set_ylabel(ylabel); ax.legend(); ax.grid(alpha=.3)
for ax in (axes[0,0],axes[0,1],axes[1,0]): ax.set_xlabel('y [m]')
axes[1,1].set_xlabel('time [s]')
fig.tight_layout()
fig.savefig(out/'accepted-prefix-and-reference.png',dpi=150)
print(json.dumps(summary,indent=2))
