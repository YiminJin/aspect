#!/usr/bin/env python3
"""Collect completed K1 cases without replacing their individual references."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    cases = {}
    for path in sorted(args.directory.glob('space*-errors.json')):
        name = path.name.removesuffix('-errors.json')
        data = json.loads(path.read_text())
        if abs(data['steps'][-1]['time_s']-6.) > 1e-12:
            raise ValueError(f'Incomplete trajectory: {name}')
        cases[name] = data
    if not cases:
        raise ValueError('No completed analyzed cases')
    summary, common = [], []
    for name, data in cases.items():
        steps = data['steps']
        row = dict(case=name, **data['initialization'],
                   omitted_fraction=data['omitted_fraction'],
                   max_slip_normalization_error=max(abs(r['normalization_ratio']-1) for r in steps),
                   max_V_error_m_s=max(r['V']['max_error'] for r in steps),
                   max_Theta_error_s=max(r['Theta']['max_error'] for r in steps),
                   max_C_error_Pa=max(r['C']['max_error'] for r in steps),
                   max_raw_q_rms_error_Pa=max(r['raw_q_rms_error_Pa'] for r in steps),
                   max_raw_q_point_error_Pa=max(r['raw_q_max_error_Pa'] for r in steps),
                   initial_raw_q_rms_error_Pa=steps[0]['raw_q_rms_error_Pa'],
                   initial_raw_q_point_error_Pa=steps[0]['raw_q_max_error_Pa'],
                   max_velocity_rms_error_m_s=max(r['velocity_rms_error_m_s'] for r in steps),
                   max_velocity_point_error_m_s=max(r['velocity_max_error_m_s'] for r in steps),
                   max_slip_error_m=max(abs(r['accumulated_slip_m']-r['reference_slip_m']) for r in steps),
                   resolved_conditional_pass=data['resolved_conditional_pass'])
        summary.append(row)
        for r in steps:
            if not any(abs(r['time_s']-t)<1e-12 for t in [2.,4.,6.]):
                continue
            limit = next(v for v in data['temporal_reference'] if v['time_s']==r['time_s'])
            sample = dict(case=name, time_s=r['time_s'], dt_s=r['dt_s'])
            for field in ['V','Theta','C']:
                sample[field+'_aspect'] = r[field]['mean']
                sample[field+'_discrete_reference'] = r[field]['reference']
                sample[field+'_small_dt_reference'] = limit[field]
                sample[field+'_temporal_error'] = r[field]['reference']-limit[field]
            sample.update(slip_aspect=r['accumulated_slip_m'], slip_discrete_reference=r['reference_slip_m'],
                          slip_small_dt_reference=limit['slip'], slip_temporal_error=r['reference_slip_m']-limit['slip'])
            common.append(sample)
    for filename, rows in [('case_summary.csv', summary), ('common_time_comparison.csv', common)]:
        with (args.directory/filename).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    (args.directory/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    fig, axes = plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for label, key, axis in [('Raw stress RMS [Pa]','max_raw_q_rms_error_Pa',axes[0]),
                             ('Raw stress maximum [Pa]','max_raw_q_point_error_Pa',axes[1]),
                             ('V maximum error [m/s]','max_V_error_m_s',axes[2])]:
        selected=[r for r in summary if r['case'].endswith('dt05')]
        axis.loglog([1/r['mesh'][1] for r in selected],[r[key] for r in selected],'o-')
        axis.set(xlabel='Bulk cell size [m]',ylabel=label)
        axis.grid(True,which='both',alpha=.3)
    fig.savefig(args.directory/'spatial_errors.png',dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for name,d in cases.items():
        if not name.startswith('space64_'):
            continue
        for field,axis in zip(['V','Theta','C'],axes):
            axis.plot([r['time_s'] for r in d['steps']],[r[field]['mean'] for r in d['steps']],
                      'o-',ms=3,label=name)
            axis.plot([r['time_s'] for r in d['steps']],[r[field]['reference'] for r in d['steps']],
                      '--',lw=1)
            axis.set(xlabel='Time [s]',ylabel=field)
    axes[0].legend(fontsize=7)
    fig.suptitle('ASPECT (solid) and each independently advanced discrete reference (dashed)')
    fig.savefig(args.directory/'histories.png',dpi=160); plt.close(fig)
    # Initial-profile changes are a separate axis of evidence, not mechanical
    # errors against a scalar reference with different initial data.
    profiles = []
    for name in cases:
        if not name.endswith('dt05'):
            continue
        nodes = np.genfromtxt(args.directory/name/'phase_0.csv', names=True, delimiter=',')
        center = np.unique(nodes['x'])[len(np.unique(nodes['x']))//2]
        column = np.sort(np.unique(nodes[nodes['x']==center][['y','phi']]),order='y')
        profiles.append((name,column))
    profiles.sort(key=lambda item: len(item[1]))
    if profiles:
        finest = profiles[-1][1]
        initialization = []
        fig,axes = plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for name,column in profiles:
            grid = np.unique(np.r_[column['y'],finest['y']])
            error = np.interp(grid,column['y'],column['phi'])-np.interp(grid,finest['y'],finest['phi'])
            # Exact squared difference integral of the two piecewise-Q1 fields.
            rms = np.sqrt(np.sum(np.diff(grid)*(error[:-1]**2+error[:-1]*error[1:]+error[1:]**2)/3))
            initialization.append(dict(case=name,phi_L2_difference_to_finest=float(rms),
                                       phi_max_difference_to_finest=float(max(abs(error)))))
            axes[0].plot(column['y'],column['phi'],label=name)
            axes[1].semilogy(column['y'],np.maximum(column['phi'],1e-12),label=name)
        axes[0].set(xlabel='y [m]',ylabel='Initialized Q1 phase field')
        axes[1].set(xlabel='y [m]',ylabel='Initialized Q1 phase field (log scale)')
        axes[0].legend(fontsize=8)
        fig.savefig(args.directory/'initial_profiles.png',dpi=160); plt.close(fig)
        (args.directory/'initialization_differences.json').write_text(json.dumps(initialization,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
