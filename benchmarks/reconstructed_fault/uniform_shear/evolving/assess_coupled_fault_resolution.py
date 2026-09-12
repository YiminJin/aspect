"""Summarize completed coupled fault-resolution cases without rerunning them."""
import csv
import json
from pathlib import Path

import numpy as np


here = Path(__file__).resolve().parent
summaries, rows = [], []
for normal in (128,256):
    name = f'spatial0375_n{normal}_f32'
    path = here/name
    comparison = json.loads((here/f'{name}-comparison.json').read_text())
    identities = json.loads((path/'identity_audit.json').read_text())
    initial = np.genfromtxt(path/'surface_0.csv',delimiter=',',names=True)
    old_initial = np.genfromtxt(here/f'spatial0375_n{normal}/surface_0.csv',delimiter=',',names=True)
    initial_differences = {field:float(max(abs(initial[field]-np.interp(initial['x'],old_initial['x'],old_initial[field]))))
                           for field in ('V','Theta','C','Ih')}
    particles = np.sort(np.genfromtxt(path/'particles_0.csv',delimiter=',',names=True),order='id')
    old_particles = np.sort(np.genfromtxt(here/f'spatial0375_n{normal}/particles_0.csv',delimiter=',',names=True),order='id')
    final_step = len(comparison['states'])-1
    for state, identity in zip(comparison['states'],identities):
        k = state['step']
        surface = np.genfromtxt(path/f'surface_{k}.csv',delimiter=',',names=True)
        audit = np.genfromtxt(path/f'identity_audit_{k}.csv',delimiter=',',names=True)
        increment = surface['Ih']-initial['Ih']
        interior = (surface['x']>=.046875)&(surface['x']<=.203125)
        rows.append(dict(case=name,step=k,time_s=state['time_s'],
            all_gates_pass=state['guard']['passed'],
            omitted_h_fraction=state['guard']['max_omitted_fraction'],
            actual_normalization=state['guard']['max_supported_normalization_error'],
            full_profile_defect=identity['max_full_defect'],
            max_predicted_measured_difference=float(max(abs(audit['predicted_full_integral_minus_V_over_V']
                                                         -audit['measured_full_integral_minus_V_over_V']))),
            max_relative_Ih_error=float(max(abs(audit['relative_Ih_error']))),
            max_signed_tail_magnitude=identity['max_tail'],
            max_bulk_quadrature_defect=identity['max_bulk_quadrature_defect'],
            H_spread=state['guard']['along_fault_ranges']['H'],
            phi_spread=state['guard']['along_fault_ranges']['phi'],
            Ih_spread=state['guard']['along_fault_ranges']['Ih'],
            Ih_interior_spread=float(np.ptp(surface['Ih'][interior])),
            Ih_increment_center=float(np.interp(.125,surface['x'],increment)),
            Ih_increment_endpoint=float(increment[0]),
            phi_profile_max_error=state['phi_max_absolute_error'],
            H_profile_max_error=state['H_profile_max_absolute_error'],
            raw_q_rms_error=state['raw_stress_error_rms_Pa'],
            weak_surface_balance=state['surface_balance_rms_Pa'],
            **{f'mean_{key}':value for key,value in state['surface_means'].items()},
            slip=state['accumulated_slip_m']))
    case_rows = rows[-len(identities):]
    summaries.append(dict(case=name,final_time=comparison['states'][-1]['time_s'],
        complete_pass=comparison['complete_smoke'],initial_surface_difference_vs_f16=initial_differences,
        initial_particle_ids_and_H_identical=bool(np.array_equal(particles['id'],old_particles['id'])
                                                 and np.array_equal(particles['H'],old_particles['H'])),
        maximum_normalization=max(r['actual_normalization'] for r in case_rows),
        maximum_full_profile_defect=max(r['full_profile_defect'] for r in case_rows),
        maximum_prediction_difference=max(r['max_predicted_measured_difference'] for r in case_rows),
        fresh_linear_checks=len(comparison['fresh_linear_checks']),
        max_fresh_over_target=max(v['fresh']/v['target'] for v in comparison['fresh_linear_checks']),
        resources=json.loads((here/f'{name}.resources.json').read_text()),
        final=case_rows[-1],reference=comparison['states'][-1]['reference']))
with (here/'coupled-fault-resolution.csv').open('w') as stream:
    writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(here/'coupled-fault-resolution.json').write_text(json.dumps(summaries,indent=2)+'\n')
print(json.dumps(summaries,indent=2))
