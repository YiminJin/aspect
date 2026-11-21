"""Measure the connected phase and slip-rate fronts for a coarse continuation."""
import argparse
import csv
import os
from pathlib import Path

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/aspect-propagation-pilot-mpl')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label
from phase import H, NX, NY, coefficients, connected_front, read, sample


def connected_intact_cells(field, intact, level, points):
    labels, _ = label(sample(field, points) >= level)
    origin = labels[NY * points // 2, NX * points // 2]
    assert origin > 0
    connected = (labels == origin).reshape(NY, points, NX, points).any(axis=(1, 3))
    return int(np.count_nonzero(intact & connected))


def observe(out, step, intact, previous_tip):
    field = coefficients(out, step)
    time_row = read(out / 'pilot-summary.csv')
    row, = time_row[time_row['step'] == step]
    assert abs(row['time'] - row['stage_time'] - row['dt'] / 2) < 1e-13
    sections = read(out / f'pilot-sections-{step:04d}.csv')
    x, v = sections['x'], sections['rate_stage']
    assert (np.diff(x) > 0).all()
    threshold = .005
    # Bracket the rightmost crossing of the signed positive-rate threshold.
    i = np.flatnonzero(v >= threshold)[-1]
    assert i + 1 < len(x) and v[i] >= threshold > v[i+1]
    crossing = x[i] + (x[i+1] - x[i]) * (v[i] - threshold) / (v[i] - v[i+1])
    current_tip = connected_front(field, 16, .05)['right_x']
    assert current_tip >= previous_tip - 1e-12
    weak_front = connected_front(field, 16, .01)['right_x']
    upper_bound = float(np.max(field[intact]))
    result = dict(step=step, endpoint_ms=1000 * row['time'], midpoint_ms=1000 * row['stage_time'],
                  intact_phase_upper_bound=upper_bound, intact_phase_integral_m2=H**2 * field[intact].mean(axis=(1, 2)).sum(),
                  phi0p01_right_x_m=weak_front, phi0p05_right_x_m=current_tip,
                  rate_bracket_left_m=x[i], rate_bracket_right_m=x[i+1],
                  rate_crossing_m=crossing, rate_crossing_dx_m=x[i+1]-x[i],
                  tip_start_m=previous_tip, tip_end_m=current_tip,
                  min_gap_from_brackets_m=previous_tip-x[i+1], max_gap_from_brackets_m=current_tip-x[i],
                  endpoint_tip_minus_midpoint_crossing_m=current_tip-crossing,
                  peak_rate_m_s=v.max(), peak_x=x[v.argmax()],
                  rightmost_crossing_connected_to_peak=bool(np.all(v[v.argmax():i+1] >= threshold)))
    for points in (8, 16):
        for level in (.001, .01):
            result[f'connected_intact_cells_{level:g}_n{points}'] = connected_intact_cells(field, intact, level, points)
    if upper_bound < .01:
        assert result['connected_intact_cells_0.01_n16'] == 0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('initial_output', 'parent_output', 'continuation_output', 'destination'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    # A cell is intact only when its entire initial DG polynomial is zero.
    intact = np.max(abs(coefficients(args.initial_output, 0)), axis=(2, 3)) == 0.
    steps = read(args.continuation_output / 'pilot-summary.csv')['step'].astype(int)
    assert np.array_equal(steps, np.arange(steps[0], steps[-1]+1))
    parent = steps[0] - 1
    prior_tip = connected_front(coefficients(args.parent_output, parent-1), 16)['right_x']
    rows = []
    for step in [parent, *steps]:
        out = args.parent_output if step == parent else args.continuation_output
        row = observe(out, step, intact, prior_tip)
        rows.append(row)
        prior_tip = row['tip_end_m']
    with (args.destination / 'threshold-fronts.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    def series(key):
        return np.array([row[key] for row in rows])
    fig, (phase_ax, gap_ax) = plt.subplots(1, 2, figsize=(9, 3.6), layout='constrained')
    phase_ax.plot(series('endpoint_ms'), series('intact_phase_upper_bound'), 'o-', label='Maximum Bernstein coefficient')
    phase_ax.axhline(.01, color='.3', ls='--', label='Phase level 0.01')
    phase_ax.set(xlabel='Endpoint time (ms)', ylabel='Phase', title='Initially intact material', ylim=(0,None))
    phase_ax.legend(fontsize=8)
    t = series('midpoint_ms')
    gap_ax.fill_between(t, 1000*series('min_gap_from_brackets_m'), 1000*series('max_gap_from_brackets_m'),
                        alpha=.25, label='Station and tip-position brackets')
    gap_ax.plot(t, 1000*series('endpoint_tip_minus_midpoint_crossing_m'), 'o-',label='Interpolated crossing to endpoint tip')
    gap_ax.set(xlabel='Slip-rate midpoint time (ms)',ylabel='Distance behind connected φ = 0.05 tip (mm)',
               title='Rightmost 5 mm/s crossing')
    gap_ax.legend(fontsize=8)
    for extension in ('png','pdf'):
        fig.savefig(args.destination / f'threshold-fronts.{extension}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
