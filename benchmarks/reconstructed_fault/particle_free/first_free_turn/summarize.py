#!/usr/bin/env python3
"""Join accepted windows without duplicating steps; report work and execution cost.

Run after analyze.py and after MPI has exited. Does not infer missing endpoints
or reuse an interrupted trial as an accepted state.
"""
import json
from analyze import OUT, RUNS

WINDOWS = {
    'positive': ['first-turn-positive-prefix', 'first-turn-positive-end',
                 'first-turn-positive-fourcore-end'],
    'reflected': ['first-turn-reflected-prefix', 'first-turn-reflected-end'],
}


def combined(labels):
    steps = []
    costs = []
    for label in labels:
        data = json.loads((OUT / (label + '-analysis.json')).read_text())
        steps += data['steps']
        command = json.loads((RUNS / label / 'command.json').read_text())
        costs.append(dict(label=label, ranks=command['arguments']['ranks'],
                          process_seconds=command.get('seconds'),
                          exit_code=command.get('exit_code'), **data['cost']))
    assert [s['step'] for s in steps] == list(range(1, steps[-1]['step']+1))
    errors = [s['physical_work_error'] for s in steps]
    late = [s for s in steps if s['time'] > .0400000001]
    late_fracture = steps[-1]['added_fracture'] - steps[7]['added_fracture']
    return dict(steps=steps, cost=costs, last_accepted_time=steps[-1]['time'],
                target_time=.08, target_reached=steps[-1]['time']>=.08-1e-12,
                work=dict(signed=sum(errors), accumulated_absolute=sum(map(abs, errors)),
                          added_fracture=steps[-1]['added_fracture'],
                          signed_fraction=sum(errors)/steps[-1]['added_fracture'],
                          absolute_fraction=sum(map(abs, errors))/steps[-1]['added_fracture']),
                after_load_change=dict(
                    signed=sum(s['physical_work_error'] for s in late),
                    accumulated_absolute=sum(abs(s['physical_work_error']) for s in late),
                    added_fracture=late_fracture),
                accepted_solver_seconds=sum(c['solver_seconds'] for c in costs),
                process_seconds_including_interrupted_trial=sum(c['process_seconds'] for c in costs),
                rank_seconds_including_interrupted_trial=sum(c['ranks']*c['process_seconds'] for c in costs))


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-reflected',action='store_true')
    args=parser.parse_args()
    selected=WINDOWS if args.include_reflected else {'positive':WINDOWS['positive']}
    data = {name: combined(labels) for name, labels in selected.items()}
    (OUT / 'accepted-history-summary.json').write_text(json.dumps(data, indent=2) + '\n')
    import csv
    for name,item in data.items():
        with (OUT/(name+'-accepted-history.csv')).open('w') as stream:
            writer=csv.writer(stream)
            writer.writerow(['step','time','added_fracture','work_remainder','W_over_hn',
                             'central_fracture','taper_fracture','ahead_fracture','remote_fracture',
                             'rightmost_x_32','rightmost_y_32','new_sections','phase_KKT','mechanical_residual'])
            for s in item['steps']:
                writer.writerow([s['step'],s['time'],s['added_fracture'],s['physical_work_error'],s['width'][0]['W_over_hn'],
                                 *[s['regions'][r]['added_fracture'] for r in ['central_band','old_tip_taper','axial_ahead','remote']],
                                 *s['contours']['32']['rightmost'],s['formation']['new_sections'],s['accepted']['phase_KKT'],s['accepted']['mechanical_residual']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    fig.suptitle('Accepted history; contour positions sampled at 32 subdivisions per cell',fontsize=11)
    for name, color in [('positive', '#c44e52'), ('reflected', '#4c72b0')]:
        if name not in data: continue
        steps = data[name]['steps']
        time = [s['time'] for s in steps]
        axes[0, 0].plot(time, [s['added_fracture'] for s in steps], '.-', color=color, label=name)
        axes[0, 1].plot(time, [s['contours']['32']['rightmost'][0]-6 for s in steps], '.-', color=color)
        errors = np.array([s['physical_work_error'] for s in steps])
        axes[1, 0].plot(time, np.cumsum(errors), '-', color=color, label=name+' signed')
        axes[1, 0].plot(time, np.cumsum(abs(errors)), '--', color=color, label=name+' absolute')
        axes[1, 1].plot(time, [s['contours']['32']['rightmost_lateral'] for s in steps], '.-', color=color)
    for ax, title in zip(axes.flat, ['Added fracture energy (J/m)',
                                   r'Rightmost $\phi=.05$ point: x − 6 (m)',
                                   'Accumulated physical-work remainder (J/m)',
                                   r'Rightmost $\phi=.05$ point: y − 2 (m)']):
        ax.set(xlabel='Time (s)', title=title)
        ax.grid(alpha=.2)
        ax.axvline(.04, ls=':', color='.5', lw=.8)
    axes[0, 0].legend(); axes[1, 0].legend(fontsize=8)
    axes[0, 1].axhline(0, color='.5', lw=.8)
    for suffix in ('png', 'pdf'):
        fig.savefig(OUT / ('free-turn-evolution.'+suffix), dpi=180)
    plt.close(fig)
    for name, item in data.items():
        print(name, json.dumps({k:v for k,v in item.items() if k not in ('steps','cost')}))
