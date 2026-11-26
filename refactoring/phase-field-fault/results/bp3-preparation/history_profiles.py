"""Plot the archived 60-degree BP3 accepted speed and held-station slip.

The argument is the donor's benchmarks/reconstructed_fault/particle_free directory.
Outputs are written in the destination. The duplicate six-step replay from job
3613912 is checked against the original and never counted a second time.
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from history_events import ONSET, RECOVERY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('particle_free', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)

    archives = args.particle_free/'evidence'
    accepted = np.genfromtxt(archives/'server_3610007/accepted_steps.csv', delimiter=',', names=True)
    repeat = np.genfromtxt(archives/'server_3613912/accepted_steps.csv', delimiter=',', names=True)
    np.testing.assert_array_equal(accepted['step'], np.arange(816))
    np.testing.assert_array_equal(repeat['step'], np.arange(810, 816))
    for column in ('step', 'time', 'dt', 'max_V', 'xd_at_max_V'):
        np.testing.assert_array_equal(accepted[column][810:], repeat[column])
    if not np.isfinite(accepted['max_V']).all():
        raise ValueError('Nonfinite archived speed')
    onset = np.flatnonzero(accepted['max_V'] >= ONSET)[0]
    t_on = accepted['time'][onset]

    starts = args.particle_free/'bp3_dg_first_event'
    records = []
    distances = [5000, 10000, 15000]
    all_xd = None
    for part in ('output_start', 'output_resumed'):
        with (starts/part/'profiles.csv').open() as stream:
            for row in csv.DictReader(stream):
                step, time = int(row['step']), float(row['time_s'])
                if records and (step <= records[-1][0] or time <= records[-1][1]):
                    raise ValueError(f'Duplicate or unordered indexed profile {step}')
                fields = np.loadtxt(starts/part/row['file'], delimiter=',', skiprows=1, ndmin=2)
                if not np.isfinite(fields).all():
                    raise ValueError(f'Nonfinite profile at {step}')
                np.testing.assert_array_equal(fields[:, 0], 0)
                np.testing.assert_array_equal(fields[:, 1], np.arange(5776))
                np.testing.assert_array_equal(fields[:, 2], step)
                np.testing.assert_array_equal(fields[:, 3], time)
                xd, slip, v = fields[:, 4], fields[:, 7], fields[:, 8]
                if all_xd is None:
                    all_xd = xd.copy()
                    if not np.all(np.diff(xd) < 0) or min(distances) < xd.min() or max(distances) > xd.max():
                        raise ValueError('Expected ordered, in-box physical stations')
                np.testing.assert_array_equal(xd, all_xd)
                peak = np.argmax(v)
                if step < len(accepted):
                    np.testing.assert_equal(time, accepted['time'][step])
                    np.testing.assert_equal(v[peak], accepted['max_V'][step])
                fixed_slip = np.interp(distances, xd[::-1], slip[::-1])
                records.append((step, time, v[peak], xd[peak], *fixed_slip))
    with (args.destination/'history-profiles.csv').open('w') as out:
        writer = csv.writer(out, lineterminator='\n')
        writer.writerow(['step', 'time_s', 'vmax_m_s', 'xd_at_max_m', *[f'slip_{x}m' for x in distances]])
        writer.writerows(records)

    profiles = np.array(records)
    time, maximum = profiles[:, 1:3].T
    first, = profiles[profiles[:, 0] == onset]
    colors = ['#166884', '#a33325', '#696034']
    fig, panels = plt.subplots(1, 3, figsize=(13.3, 4), layout='constrained')
    ax, event, stations = panels
    ax.semilogy(accepted['time']/31557600., accepted['max_V'], color=colors[0], lw=1.6)
    ax.axhline(ONSET, ls='--', color='0.5', lw=.8)
    ax.axhline(RECOVERY, ls=':', color='0.5', lw=.8)
    ax.set(xlabel='Time since initialization (yr)', ylabel='Maximum fault slip rate (m/s)',
           title='Loading to the first onset', ylim=(5e-10, 3), xlim=(-3, 168))
    event.semilogy(accepted['time']-t_on, accepted['max_V'], '.', color=colors[0], ms=3,
                   label='Every accepted step to 815')
    after = profiles[:, 0] > 815
    event.semilogy(time[after]-t_on, maximum[after], '.', color=colors[1], ms=4,
                   label='Saved full-fault profiles')
    event.axhline(ONSET, ls='--', color='.45', lw=.8)
    event.axvline(0, color='.55', ls=':', lw=1)
    event.set(xlabel='Seconds from first onset', ylabel='Maximum fault slip rate (m/s)',
              xlim=(-15, 18.5), ylim=(3e-4, 3), title='Acceleration within the episode')
    event.legend(loc='lower right', fontsize=8)
    shown = profiles[:, 0] >= onset
    for j, xd in enumerate(distances):
        stations.plot(time[shown]-t_on, profiles[shown, 4+j]-first[4+j], '.-', ms=2, lw=.9,
                      label=f'{xd/1000:g} km down dip', color=colors[j])
    stations.set(xlabel='Seconds from first onset', ylabel='Accumulated slip since onset (m)',
                 title='Slip at fixed material locations', xlim=(-.45, 18.2))
    stations.legend(loc='upper left', fontsize=8)
    for ax in panels:
        ax.grid(visible=True, which='major', alpha=.17)
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(args.destination/'history-first-acceleration.png', dpi=220)
    plt.close(fig)
    print('Before / onset:', accepted[onset-1: onset+1][['step', 'time', 'dt', 'max_V', 'xd_at_max_V']])
    print('Indexed profiles:', len(profiles), 'last:', records[-1])
    print('Largest observed whole-fault rate in the profiles:', records[int(np.argmax(maximum))][:4])
    print('Fixed-station slip since onset (m):', profiles[-1, 4:]-first[4:])
    print('Slowest archived post-onset profile:', min(maximum[shown]), 'm/s')


if __name__ == '__main__':
    main()
