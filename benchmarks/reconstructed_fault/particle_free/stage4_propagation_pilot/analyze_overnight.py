"""Plot published fields and feasible forward variations on one pilot trajectory.

Run in the worktree with --dest DIRECTORY and, optionally, successive native
run directories continuing step 78. A finer-step branch must be reduced
separately. All stress/velocity/phase work comes from the native observers.
"""
import argparse
import csv
import os
from pathlib import Path

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/aspect-propagation-pilot-mpl')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np

from measure_propagation_fronts import observe
from phase import coefficients, H, read, value


def records(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def record(path, step):
    row, = (r for r in records(path) if int(r['step']) == step)
    return {k: float(v) for k, v in row.items()}


def write_csv(path, data):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=data[-1])
        writer.writeheader()
        writer.writerows(data)


def history(folder, step, sections):
    stage = record(folder / 'pilot-summary.csv', step)
    row = dict(step=step, endpoint_ms=stage['time'] * 1000, stage_ms=stage['stage_time'] * 1000,
               dt=stage['dt'], peak_rate=stage['peak_rate'], integrated_slip=stage['integrated_slip'])
    x, rate = sections['x'], sections['rate_stage']
    row['peak_x'] = x[np.argmax(abs(rate))]
    assert np.all(np.diff(x) > 0)
    for station in (3.001, 5.8, 6., 6.03125):
        # At the patch edge the two adjacent material sections have different
        # initial ages. Track the peak section itself throughout the record.
        probe = x[np.argmin(abs(x-station))] if station == 3.001 else station
        row[f'x_{station:g}'] = probe
        row[f'rate_{station:g}'] = np.interp(probe, x, rate)
        row[f'age_{station:g}'] = np.interp(probe, x, sections['age_endpoint'])
    energy = record(folder / 'bulk-accepted.csv', step)
    assert energy['publications'] == step
    assert energy['mechanical_residual'] <= 1e-9 and energy['phase_KKT'] <= 1e-7
    row.update(fracture_energy=energy['fracture'], mechanics=energy['mechanical_residual'],
               kkt=energy['phase_KKT'], ledger_error=energy['ledger_error'],
               work_error=energy['Wext']-energy['energy_change']-energy['Db']-energy['Dc']-energy['Df'])
    if step >= 68:
        probes = [r for r in records(folder / 'release-driving.csv') if int(r['step']) == step]
        assert {r['probe'] for r in probes} == {'0', '1'}
        for r in probes:
            assert abs(float(r['stage_time'])*1000-row['stage_ms']) < 1e-10
            for key, source in [('net', 'mean_net_driving'), ('mechanical', 'mechanical_driving'),
                                ('fracture', 'fracture_derivative'), ('admissible', 'admissible_stage_epsilon')]:
                row[key+r['probe']] = float(r[source])
    return row


def plot(path, stations, rates, rows, field, seed, intact):
    fig, ax = plt.subplots(2, 2, figsize=(11.4, 7.3), layout='constrained')
    stage = [r['stage_ms'] for r in rows]
    end = [r['endpoint_ms'] for r in rows]
    rates = np.array(rates)*1000
    rate_image = ax[0, 0].pcolormesh(stage, stations, rates.T, shading='nearest', cmap='RdBu_r',
                                     norm=SymLogNorm(linthresh=.5, vmin=-rates.max(), vmax=rates.max()))
    ax[0, 0].axhline(6, color='.2', ls=':', lw=.8)
    fig.colorbar(rate_image, ax=ax[0, 0], label='Signed slip rate (mm/s)', ticks=[-10, -1, 0, 1, 10])
    ax[0, 0].set(xlabel='Midpoint time (ms)', ylabel='Section x (m)', title='Slip rate along the fault')
    for st in (3.001, 5.8, 6.):
        ax[0, 1].semilogy(end, [r[f'age_{st:g}']*1000 for r in rows], label=f'x = {st:g} m')
    ax[0, 1].legend()
    ax[0, 1].set(xlabel='Endpoint time (ms)', ylabel='Section age (ms)', title='Patch, leading flank and original tip')

    x, y = np.meshgrid(np.linspace(5.95, 6.34, 320), np.linspace(1.90, 2.15, 230))
    phase = value(field, x, y)
    damage = ax[1, 0].pcolormesh(x, y, value(field-seed, x, y), shading='auto', cmap='magma')
    levels = ax[1, 0].contour(x, y, phase, levels=[.001, .01, .05], colors='white', linewidths=[.8, 1.1, 1.5])
    ax[1, 0].clabel(levels, fmt='%g', fontsize=7)
    ax[1, 0].contour(x, y, intact[(y/H).astype(int), (x/H).astype(int)].astype(float),
                     levels=[.5], colors='#55e6db', linewidths=1.3)
    fig.colorbar(damage, ax=ax[1, 0], label='Phase increment from seed')
    ax[1, 0].set(xlabel='x (m)', ylabel='y (m)', title=f'Damage at {end[-1]:.2f} ms and initial support (cyan)', aspect='equal')

    measured = [r for r in rows if 'net0' in r]
    for k, label in [('net0', 'Near tip'), ('net1', 'Fully intact ahead')]:
        ax[1, 1].plot([r['stage_ms'] for r in measured], [r[k] for r in measured], '.-', label=label)
    ax[1, 1].axhline(0, color='.4', lw=.7)
    ax[1, 1].legend()
    ax[1, 1].set(xlabel='Midpoint time (ms)', ylabel='Net driving / probe area (J/m³)',
                 title='Feasible positive phase increments')
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest', type=Path, required=True)
    parser.add_argument('new_runs', nargs='*', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    build = root/'build-particle-free'
    original = build/'stage4-propagation-pilot/analysis_views/through70'
    coarse = build/'guard80/through80/output'
    paths = [(original, step) for step in range(1, 69)]
    for run in [build/'guard80/replay', build/'guard80/continuation', coarse.parent, *args.new_runs]:
        folder = run/'output'
        paths.extend((folder, int(r['step'])) for r in records(folder/'pilot-summary.csv'))
    assert [n for _, n in paths] == list(range(1, paths[-1][1]+1))
    args.dest.mkdir(exist_ok=True, parents=True)

    seed = coefficients(original, 0)
    intact = np.max(abs(seed), axis=(2, 3)) == 0
    y, x = np.indices(seed.shape[:2])
    remote = (abs((y+.5)*H-2) > .5) | ((x+.5)*H < 1.5) | ((x+.5)*H > 6.5)
    previous = coefficients(coarse, 78)
    reference = root/'benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot'
    previous_tip = float(records(reference/'through80/threshold-fronts.csv')[-1]['tip_end_m'])
    old_planes = read(coarse/'tip-planes-0078.csv')
    predecessor_planes = old_planes
    rows, rates, fronts = [], [], []
    for folder, step in paths:
        sections = read(folder/f'pilot-sections-{step:04d}.csv')
        if rows:
            assert np.array_equal(stations, sections['x'])
        stations = sections['x']
        rows.append(history(folder, step, sections))
        rates.append(sections['rate_stage'])
        if step > 78:
            front = observe(folder, step, intact, previous_tip)
            previous_tip = front['tip_end_m']
            field = coefficients(folder, step)
            front.update(min_B=field.min(), max_B=field.max(), minimum_delta_B=(field-previous).min(),
                         remote_max_delta_B=np.max(abs(field-seed)[remote]),
                         fresh_cells_mean_gt_1e_12=np.count_nonzero(intact & (field.mean(axis=(2, 3)) > 1e-12)))
            assert field.min() >= -1e-14 and field.max() <= 1+1e-14 and front['minimum_delta_B'] >= -1e-14
            planes = read(folder/f'tip-planes-{step:04d}.csv')
            formed = predecessor_planes['formed'] == 1
            assert np.array_equal(planes['parent_formed'], predecessor_planes['formed'])
            assert np.array_equal(old_planes['x'], planes['x'])
            assert (planes['formed'][formed] == 1).all()
            assert np.array_equal(planes['angle'][formed], predecessor_planes['angle'][formed])
            new = (planes['formed'] == 1) & (old_planes['formed'] == 0)
            front.update(new_this_step=np.count_nonzero((planes['formed'] == 1) & ~formed), new_sections=new.sum(), new_angles_deg=np.rad2deg(planes['angle'][new]).tolist())
            fronts.append(front)
            previous = field
            predecessor_planes = planes
    end = np.array([r['endpoint_ms'] for r in rows])
    np.testing.assert_allclose(np.diff(end, prepend=0), [r['dt']*1000 for r in rows], atol=1e-11, rtol=0)
    write_csv(args.dest/'accepted-history.csv', rows)
    if fronts:
        write_csv(args.dest/'new-fronts.csv', fronts)
    plot(args.dest/'accepted-response.png', stations, rates, rows, previous, seed, intact)


if __name__ == '__main__':
    main()
