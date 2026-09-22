"""Fig.-8-style slip contours selected by slip increment, not elapsed time.

Reads the every-accepted-state cumulative_slip.csv, one profile at a time.
Only recorded profiles are drawn: no time interpolation or rate integration.
"""
import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import warnings

import numpy as np

YEAR = 31557600.0
HEADER = ['step', 'time_s', 'fault', 'node', 's_m', 'xd_m', 'slip_m']


@dataclass
class Profile:
    step: int
    time: float
    nodes: np.ndarray
    xd: np.ndarray
    slip: np.ndarray


def read_profiles(path, fault=0):
    """Validate fixed geometry and complete, consecutive accepted states.

    A partially copied final profile fails explicitly, rather than becoming
    a misleading final contour. Copy a complete file from the server first.
    """
    previous = None
    key = None
    rows = []

    def finish(current, values):
        nonlocal previous
        if not values:
            raise ValueError(f'Fault {fault} is missing at step {current[0]}')
        # An appended fresh start may repeat initialization. Discard only
        # identical complete blocks, never choose between different histories.
        starts = [i for i, value in enumerate(values) if value[0] == 0]
        if len(starts) > 1:
            width = starts[1]
            if (starts != list(range(0, len(values), width)) or len(values) % width
                    or any(values[i:i+width] != values[:width] for i in starts[1:])):
                raise ValueError(f'Conflicting duplicate profiles at step {current[0]}')
            warnings.warn(f'Skipped {len(starts)-1} identical duplicate profile(s) at step {current[0]}')
            values = values[:width]
        nodes = np.array([v[0] for v in values], dtype=int)
        data = np.array([v[1:] for v in values])
        if not np.array_equal(nodes, np.arange(len(nodes))) or len(nodes) < 2:
            raise ValueError(f'Incomplete/duplicate/unordered vertices at step {current[0]}')
        if not np.isfinite(data).all() or not np.isfinite(current[1]):
            raise ValueError(f'Nonfinite slip/geometry/time at step {current[0]}')
        profile = Profile(*current, nodes, data[:, 0], data[:, 1])
        if previous is not None:
            if profile.step != previous.step + 1 or profile.time <= previous.time:
                raise ValueError('Expected consecutive steps with increasing physical time; '
                                 'rate classification requires the every-step table')
            if (not np.array_equal(profile.nodes, previous.nodes)
                    or not np.array_equal(profile.xd, previous.xd)):
                raise ValueError(f'Incomplete profile or changed fault geometry at step {profile.step}')
        previous = profile
        return profile

    with Path(path).open(newline='') as stream:
        reader = csv.reader(stream)
        if next(reader, None) != HEADER:
            raise ValueError('Expected the BP3 cumulative_slip.csv header')
        for line, row in enumerate(reader, 2):
            if len(row) != len(HEADER):
                raise ValueError(f'Malformed or partially copied CSV row {line}')
            current = (int(row[0]), float(row[1]))
            if key is not None and current != key:
                yield finish(key, rows)
                rows = []
            key = current
            if int(row[2]) == fault:
                rows.append((int(row[3]), float(row[5]), float(row[6]), float(row[4])))
        if key is None:
            raise ValueError('No profiles in cumulative slip file')
        yield finish(key, rows)


def select_profiles(profiles, increment=0.1, xd_min=0., xd_max=40000.,
                    seismic_rate=1e-3, start_time=0., end_time=float('inf')):
    """Use the infinity norm of slip change in the displayed down-dip window.

    Regime changes retain both adjoining accepted states even below the plotting
    increment. Colour uses the entire fault, not only the displayed window.
    In this benchmark slip[k]-slip[k-1] = dt[k]*V[k]. These inferred accepted
    rates cannot detect unresolved between-step peaks.
    """
    if increment <= 0 or seismic_rate <= 0 or xd_max <= xd_min:
        raise ValueError('Require positive increment/rate and an increasing down-dip window')
    selected = []
    mask = None
    last_plotted = None
    previous = None
    previous_record = None
    all_count = 0
    in_window = 0
    seismic_count = 0
    max_rate = 0.
    dt_min = float('inf')
    dt_max = 0.
    largest_step_slip = 0.

    def retain(record):
        nonlocal last_plotted
        if not selected or selected[-1]['step'] != record['step']:
            selected.append(record)
            last_plotted = record['slip']

    for profile in profiles:
        all_count += 1
        if mask is None:
            mask = (profile.xd >= xd_min) & (profile.xd <= xd_max)
            if np.count_nonzero(mask) < 2:
                raise ValueError('Down-dip window contains fewer than two vertices')
            order = np.argsort(profile.xd[mask])
            xd = profile.xd[mask][order]
        rate = None
        step_slip = 0.
        dt = None
        if previous is not None:
            dt = profile.time - previous.time
            change = profile.slip - previous.slip
            rate = float(np.max(np.abs(change)) / dt)
            step_slip = float(np.max(np.abs(change[mask])))
        regime = 'unknown' if rate is None else ('coseismic' if rate >= seismic_rate else 'interseismic')
        record = dict(step=profile.step, time_s=profile.time, time_yr=profile.time/YEAR,
                      regime=regime, max_rate_m_s=rate,
                      slip=profile.slip[mask][order].copy())
        if start_time <= profile.time <= end_time:
            in_window += 1
            if rate is not None:
                seismic_count += regime == 'coseismic'
                max_rate = max(max_rate, rate)
                dt_min, dt_max = min(dt_min, dt), max(dt_max, dt)
                largest_step_slip = max(largest_step_slip, step_slip)
            if last_plotted is None:
                retain(record)
            elif previous_record['regime'] != regime:
                retain(previous_record)
                retain(record)
            elif np.max(np.abs(record['slip'] - last_plotted)) >= increment:
                retain(record)
            previous_record = record
        previous = profile

    if not selected:
        raise ValueError('No accepted states in the requested time window')
    retain(previous_record)
    summary = dict(accepted_profiles_read=all_count, accepted_profiles_in_window=in_window,
                   plotted_profiles=len(selected), coseismic_accepted_profiles=seismic_count,
                   first_time_yr=selected[0]['time_yr'], final_time_yr=selected[-1]['time_yr'],
                   maximum_recovered_rate_m_s=max_rate,
                   minimum_dt_s=dt_min if dt_min != float('inf') else None,
                   maximum_dt_s=dt_max if dt_max else None,
                   largest_single_step_slip_change_m=largest_step_slip,
                   slip_increment_m=increment, seismic_threshold_m_s=seismic_rate,
                   xd_window_km=[xd_min/1000, xd_max/1000],
                   rate_definition='max over entire selected fault of abs(delta slip)/delta time; '
                                   'accepted nodal V for the BP3 dt*V slip update, not substep peaks',
                   selection_definition='max absolute slip change in displayed window since last '
                                        'plotted state; also retain first/last and regime transitions')
    return xd, selected, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Run directory or cumulative_slip.csv')
    parser.add_argument('--slip-increment', type=float, default=0.1, metavar='METRES',
                        help='Maximum slip change triggering a new recorded contour (default: 0.1 m)')
    parser.add_argument('--xd-min-km', type=float, default=0.)
    parser.add_argument('--xd-max-km', type=float, default=40.)
    parser.add_argument('--fault', type=int, default=0)
    parser.add_argument('--seismic-rate', type=float, default=1e-3, metavar='M/S')
    parser.add_argument('--start-years', type=float, default=0.)
    parser.add_argument('--end-years', type=float, default=float('inf'))
    parser.add_argument('--output', type=Path, help='PNG/PDF/SVG filename (default: cumulative_slip_fig8.png)')
    parser.add_argument('--title', default='Modified BP3', help='Benchmark label in the plot title')
    args = parser.parse_args()
    if (not np.isfinite([args.slip_increment, args.xd_min_km, args.xd_max_km,
                         args.seismic_rate, args.start_years]).all()
            or np.isnan(args.end_years) or args.end_years < args.start_years):
        parser.error('Invalid plotting limits/increment')
    source = args.input/'cumulative_slip.csv' if args.input.is_dir() else args.input
    try:
        xd, selected, summary = select_profiles(
            read_profiles(source, args.fault), args.slip_increment,
            args.xd_min_km*1000, args.xd_max_km*1000, args.seismic_rate,
            args.start_years*YEAR, args.end_years*YEAR)
    except ValueError as error:
        parser.error(str(error))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = dict(interseismic='#2459bb', coseismic='#d62728', unknown='0.5')
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for record in selected:
        ax.plot(record['slip'], xd/1000, color=colors[record['regime']], lw=0.55, alpha=0.85)
    ax.set(xlabel='Cumulative slip (m)', ylabel='Distance down-dip (km)',
           ylim=(args.xd_max_km, args.xd_min_km),
           title=f'{args.title}: accepted cumulative-slip profiles\n'
                 f'{summary["first_time_yr"]:.2f}–{summary["final_time_yr"]:.2f} yr; '
                 f'slip-increment selection {args.slip_increment:g} m')
    ax.margins(x=0.015)
    handles = [Line2D([0], [0], color=colors[name], lw=1.2, label=label)
               for name, label in [('interseismic', f'Interseismic: max |V| < {args.seismic_rate:g} m/s'),
                                   ('coseismic', f'Coseismic: max |V| ≥ {args.seismic_rate:g} m/s')]
               if any(r['regime'] == name for r in selected)]
    ax.legend(handles=handles, loc='upper right', fontsize=9)
    ax.grid(alpha=0.12)
    fig.text(0.5, 0.01, 'Recorded states only; colours from Δslip/Δt over the entire fault. '
             'No temporal interpolation.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    output = args.output or source.parent/'cumulative_slip_fig8.png'
    fig.savefig(output, dpi=220)
    plt.close(fig)

    # Selection provenance makes contour spacing auditable without rereading the figure.
    columns = ['step', 'time_s', 'time_yr', 'regime', 'max_rate_m_s']
    with output.with_suffix('.profiles.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: r[k] for k in columns} for r in selected)
    summary['input'] = str(source)
    summary['fault'] = args.fault
    output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    print(f'Wrote {output}')
    if summary['largest_single_step_slip_change_m'] > args.slip_increment:
        print('Note: some accepted steps exceed the plotting increment; '
              'no intermediate profiles were invented.')


if __name__ == '__main__':
    main()
