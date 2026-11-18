#!/usr/bin/env python3
"""Local half-step check from retained, law-specific accepted parents.

Compare endpoint phase, geometry, slip/age and physical work at equal times.
Two fine midpoint rates are averaged over each coarse interval; they are not
mislabelled as coincident stages. The earlier trajectory is shared, not refined.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/aspect-preparation-mpl')
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from phase import H, coefficients, fronts, read, value

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUNS = ROOT / 'build-particle-free/bulk-tip/runs'
OUT = HERE.parent / 'evidence/stage4_preparation'
BUILD = ROOT / 'build-particle-free/stage4-preparation'
GRAM = np.array([[1/5, 1/10, 1/30], [1/10, 2/15, 1/10], [1/30, 1/10, 1/5]])


def l2(field):
    return float(np.sqrt(max(0., H**2 * np.einsum('yxij,ik,jl,yxkl->', field, GRAM, GRAM, field))))


def native_gradient(output, step):
    paths = sorted(output.glob(f'tip-balance-{step:04d}-r*.csv*'))
    assert len(paths) == 4
    return np.concatenate([np.loadtxt(path, delimiter=',', skiprows=1, usecols=[1, 2, 3, 4, 7]) for path in paths])


def work(rows):
    return rows['Wext'] - rows['energy_change'] - rows['Db'] - rows['Dc'] - rows['Df']


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--coarse-view', type=Path, required=True)
parser.add_argument('--coarse-family', default='free76')
parser.add_argument('--fine-family', default='half')
parser.add_argument('--fine-view', type=Path)
parser.add_argument('--parent-step', type=int, default=60)
args = parser.parse_args()
prepared = np.loadtxt(BUILD / 'input-prepared/phase.txt', skiprows=1).reshape(128, 256, 3, 3)
initial = fronts(prepared, 8)
records = {}
histories = {}
for law in ('weakening', 'neutral'):
    coarse = args.coarse_view / f'stage4-preparation-{args.coarse_family}-{law}' / 'output'
    fine = (args.fine_view.resolve() if args.fine_view else RUNS) / f'stage4-preparation-{args.fine_family}-{law}' / 'output'
    cs = read(coarse / 'pilot-summary.csv')
    fs = read(fine / 'pilot-summary.csv')
    ce = read(coarse / 'bulk-accepted.csv')
    fe = read(fine / 'bulk-accepted.csv')
    parent = cs[cs['step'] == args.parent_step][0]
    parent_b = coefficients(coarse, args.parent_step)
    parent_g = native_gradient(coarse, args.parent_step)
    parent_energy = ce[ce['step'] == args.parent_step][0]
    restore = read(fine / 'restore-clock.csv.gz' if (fine / 'restore-clock.csv.gz').exists() else fine / 'restore-clock.csv')
    assert int(restore['accepted_step'][0]) == args.parent_step
    assert abs(restore['accepted_time'][0] - parent['time']) < 1e-14
    assert int(restore['publications'][0]) == args.parent_step
    # The dump is emitted before any trial or changed interval scheduling.
    restored_parts = sorted(fine.glob('restore-phase-r*.csv*'))
    assert len(restored_parts) == 4
    restored = np.concatenate([np.loadtxt(path, delimiter=',', skiprows=1, usecols=[1, 2, 3]) for path in restored_parts])
    assert np.max(abs(restored[:, 1] - restored[:, 2])) < 1e-14
    assert np.array_equal(restored[:, 1], parent_g[:, 3])
    rows = []
    for coarse_state in cs[cs['step'] > args.parent_step]:
        fine_states = fs[np.abs(fs['time'] - coarse_state['time']) < 1e-13]
        if not len(fine_states):
            continue
        fine_state = fine_states[0]
        coarse_step, fine_step = int(coarse_state['step']), int(fine_state['step'])
        bc, bf = coefficients(coarse, coarse_step), coefficients(fine, fine_step)
        fc, ff = fronts(bc, 8), fronts(bf, 8)
        ec = ce[ce['step'] == coarse_step][0]
        ef = fe[fe['step'] == fine_step][0]
        sc = read(coarse / f'pilot-sections-{coarse_step:04d}.csv')
        sf = read(fine / f'pilot-sections-{fine_step:04d}.csv')
        assert np.array_equal(sc['x'], sf['x'])
        fine_stages = fs[(fs['time'] > coarse_state['time'] - coarse_state['dt'] + 1e-13) & (fs['time'] <= coarse_state['time'] + 1e-13)]
        assert len(fine_stages) == 2
        stage_sections = [read(fine / f'pilot-sections-{int(state["step"]):04d}.csv') for state in fine_stages]
        average_rate = sum(stage['rate_stage'] for stage in stage_sections) / 2
        row = dict(time=float(coarse_state['time']), coarse_step=coarse_step, fine_step=fine_step,
                   maximum_coefficient_difference=float(np.max(abs(bf-bc))),
                   phase_L2_difference=l2(bf-bc), phase_increment_L2_coarse=l2(bc-prepared),
                   phase_increment_L2_fine=l2(bf-prepared),
                   right_advance_coarse=fc['right_x']-initial['right_x'], right_advance_fine=ff['right_x']-initial['right_x'],
                   left_advance_coarse=initial['left_x']-fc['left_x'], left_advance_fine=initial['left_x']-ff['left_x'],
                   right_contour_y_coarse=fc['right_y'], right_contour_y_fine=ff['right_y'],
                   right_front_time_check_error=ff['right_x']-fc['right_x'],
                   fracture_coarse=float(ec['fracture']), fracture_fine=float(ef['fracture']),
                   local_added_fracture_coarse=float(ec['fracture']-parent_energy['fracture']),
                   local_added_fracture_fine=float(ef['fracture']-parent_energy['fracture']),
                   integrated_slip_coarse=float(coarse_state['integrated_slip']),
                   integrated_slip_fine=float(fine_state['integrated_slip']),
                   maximum_endpoint_section_slip_difference=float(max(abs(sf['slip_endpoint']-sc['slip_endpoint']))),
                   maximum_endpoint_section_age_difference=float(max(abs(sf['age_endpoint']-sc['age_endpoint']))),
                   maximum_section_measure_difference=float(max(abs(sf['ds']-sc['ds']))),
                   maximum_section_angle_difference_rad=float(max(abs(sf['angle']-sc['angle']))),
                   maximum_interval_average_rate_difference=float(max(abs(average_rate-sc['rate_stage']))))
        for station in (5.75, initial['right_x'], initial['left_x']):
            for key in ('slip_endpoint', 'age_endpoint'):
                if key in sc.dtype.names:
                    row[f'{key}_x{station:g}_coarse'] = float(np.interp(station, sc['x'], sc[key]))
                    row[f'{key}_x{station:g}_fine'] = float(np.interp(station, sf['x'], sf[key]))
            for quantity, key in [('rate', 'rate_stage'), ('resistance', 'resistance_stage')]:
                row[f'interval_average_{quantity}_x{station:g}_coarse'] = float(np.interp(station, sc['x'], sc[key]))
                average = sum(stage[key] for stage in stage_sections)/2
                row[f'interval_average_{quantity}_x{station:g}_fine'] = float(np.interp(station, sf['x'], average))
        for label, energy, upper_step in [('coarse', ce, coarse_step), ('fine', fe, fine_step)]:
            local = energy[(energy['step'] > args.parent_step) & (energy['step'] <= upper_step)]
            row['local_signed_work_error_'+label] = float(sum(work(local)))
            row['local_absolute_work_error_'+label] = float(sum(abs(work(local))))
        for label, output, step in [('coarse', coarse, coarse_step), ('fine', fine, fine_step)]:
            endpoint = native_gradient(output, step)
            assert np.array_equal(endpoint[:, :3], parent_g[:, :3])
            partition = .5*(endpoint[:, 3]-parent_g[:, 3])*(endpoint[:, 4]+parent_g[:, 4])
            row['local_fracture_partition_identity_error_'+label] = float(sum(partition)-row['local_added_fracture_'+label])
            for side, mask in [('right', parent_g[:, 1]+H/2 >= initial['right_x']), ('left', parent_g[:, 1]-H/2 <= initial['left_x'])]:
                row[side+'_cap_and_ahead_fracture_'+label] = float(sum(partition[mask]))
        rows.append(row)
    assert rows
    histories[law] = rows
    prefix = ce[ce['step'] <= args.parent_step]
    records[law] = dict(parent_step=args.parent_step, parent_time=float(parent['time']), final=rows[-1],
                        prefix_signed_work_error=float(sum(work(prefix))), prefix_absolute_work_error=float(sum(abs(work(prefix)))),
                        restored_native_phase_exact=True,
                        note='Earlier prefix uses the retained coarse solution; this is a local, not from-zero, temporal check.')
    with (OUT / f'{args.fine_family}-{law}-time-check.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
by_time={law:{round(row['time'],13):row for row in rows} for law,rows in histories.items()}
common=sorted(set(by_time['weakening']) & set(by_time['neutral']));assert common
last=common[-1]
for law in histories:
    records[law]['last_available']=records[law]['final']
    records[law]['final']=by_time[law][last]
    # Retain every row in the CSV; figures and paired metrics use only the
    # common accepted interval if a native wall-time limit ends a case early.
    histories[law]=[row for row in histories[law] if round(row['time'],13)<=last]
w, n = (by_time[law][last] for law in ('weakening', 'neutral'))
result = dict(laws=records, final_common_time=last,
              coarse_right_front_law_difference=w['right_advance_coarse']-n['right_advance_coarse'],
              fine_right_front_law_difference=w['right_advance_fine']-n['right_advance_fine'],
              maximum_right_front_local_time_check_error=max(abs(w['right_front_time_check_error']), abs(n['right_front_time_check_error'])),
              rate_comparison='Two fine midpoint samples averaged per coarse interval; endpoints compared at exactly common physical times.')
result['change_in_right_front_law_difference'] = result['fine_right_front_law_difference']-result['coarse_right_front_law_difference']
result['summed_individual_local_front_changes'] = abs(w['right_front_time_check_error'])+abs(n['right_front_time_check_error'])
result['uncertainty_note'] = 'Measured local refinement changes, not a rigorous error bound; the earlier coarse prefix and spatial discretization are shared.'
(OUT / f'{args.fine_family}-time-check.json').write_text(json.dumps(result, indent=2)+'\n')
fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
for law, rows in histories.items():
    time = [row['time']*1e3 for row in rows]
    color = 'tab:red' if law == 'weakening' else 'tab:blue'
    for scheme, style in [('coarse', '--'), ('fine', '-')]:
        axes[0].plot(time, [row['right_advance_'+scheme]/H for row in rows], style, color=color, label=law+', '+scheme)
        axes[1].plot(time, [row['local_added_fracture_'+scheme] for row in rows], style, color=color, label=law+', '+scheme)
        axes[2].plot(time, [row['local_absolute_work_error_'+scheme] for row in rows], style, color=color, label=law+', '+scheme)
for ax, title, label in zip(axes, ['Connected right-front advance', 'Fracture over checked interval', 'Accumulated absolute physical-work error'], ['Advance / cell width', 'Energy (J/m)', 'Error (J/m)']):
    ax.set(xlabel='Endpoint time (ms)', ylabel=label, title=title); ax.legend(fontsize=7)
for extension in ('png', 'pdf'):
    fig.savefig(OUT / f'{args.fine_family}-time-check.{extension}', dpi=170)
print(json.dumps(result, indent=2))
