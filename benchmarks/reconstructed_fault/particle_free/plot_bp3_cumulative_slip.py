#!/usr/bin/env python3
"""Fig. 8-style cumulative slip from saved BP3 server profiles.

Defaults: combine output_start/ and output_resumed/, use 5-year interseismic
and 1-second coseismic target times, and draw only recorded profiles. A target
inside a saved-data gap wider than its plotting interval is omitted. No slip
rate integration, temporal interpolation, slip offset or inferred event arrest.

Dependencies: NumPy, Matplotlib and the repository's bp3/plot_cumulative_slip.py
reader. Only the explicitly chosen output prefix is written.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bp3'))
from plot_cumulative_slip import YEAR, read_profiles

SERVER = ROOT / 'benchmarks/reconstructed_fault/particle_free/bp3_dg_first_event'
DEFAULT_INPUTS = [SERVER / 'output_start', SERVER / 'output_resumed']
DEFAULT_OUTPUT = ROOT / 'build-particle-free/bp3-cumulative-slip/cumulative_slip'


def load_history(inputs, fault=0):
    """Merge chronological restart segments, rejecting conflicting overlaps.

    Reuse the existing scheduled-profile reader's complete-node, finite-value,
    fixed-geometry and index/payload checks. Cross-segment geometry is checked
    here as well. An identical copied checkpoint profile is retained only once.
    """
    by_step, by_time, provenance = {}, {}, {}
    geometry = None
    indexes, duplicates = [], 0
    for source in inputs:
        index = Path(source).resolve()
        if index.is_dir():
            index /= 'profiles.csv'
        if index.name != 'profiles.csv':
            raise ValueError(f'Expected a run directory or profiles.csv: {source}')
        with index.open(newline='') as stream:
            entries = list(csv.DictReader(stream))
        if not entries:
            raise ValueError(f'Empty profile index: {index}')
        paths = {}
        for entry in entries:
            key = (int(entry['step']), float(entry['time_s']))
            path = (index.parent / entry['file']).resolve()
            if key in paths and paths[key] != path:
                raise ValueError(f'Conflicting payload references in {index}: {key}')
            paths[key] = path
        first_path = paths[int(entries[0]['step']), float(entries[0]['time_s'])]
        first = np.atleast_1d(np.genfromtxt(first_path, delimiter=',', names=True))
        first = first[first['fault'] == fault]
        current_geometry = np.column_stack([first[k] for k in ('node', 'xd_m', 'x_m', 'y_m')])
        if geometry is not None and not np.array_equal(current_geometry, geometry):
            raise ValueError(f'Fault geometry differs across input segments: {index}')
        geometry = current_geometry
        indexes.append(dict(path=str(index), sha256=hashlib.sha256(index.read_bytes()).hexdigest()))
        for profile in read_profiles(index, fault):
            path = paths[profile.step, profile.time]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if profile.step in by_step:
                old = by_step[profile.step]
                if old.time != profile.time:
                    raise ValueError(f'Conflicting time for repeated step {profile.step}')
                if digest != provenance[profile.step]['sha256']:
                    # Different text formatting is harmless, different histories
                    # in any exported column are not. Do not silently pick one.
                    left = np.atleast_1d(np.genfromtxt(provenance[profile.step]['path'], delimiter=',', names=True))
                    right = np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True))
                    if left.dtype.names != right.dtype.names or not np.array_equal(left, right):
                        raise ValueError(f'Conflicting overlapping profile at step {profile.step}')
                duplicates += 1
                continue
            if profile.time in by_time:
                raise ValueError(f'Different steps share physical time {profile.time:.17g}')
            by_step[profile.step] = profile
            by_time[profile.time] = profile.step
            provenance[profile.step] = dict(path=str(path), index=str(index), sha256=digest)
    profiles = sorted(by_step.values(), key=lambda p: p.time)
    if not profiles:
        raise ValueError('No profiles loaded')
    for previous, current in zip(profiles[:-1], profiles[1:]):
        if current.step <= previous.step:
            raise ValueError('Merged step numbers do not increase with physical time')
        if not (np.array_equal(current.nodes, previous.nodes)
                and np.array_equal(current.xd, previous.xd)):
            raise ValueError('Incomplete profile or changed geometry across segments')
    return profiles, provenance, indexes, duplicates


def select_timed_profiles(profiles, interseismic_years=5., coseismic_seconds=1., seismic_rate=1e-3):
    """Select actual saved states at supported nominal times within each regime.

    Interseismic targets are multiples of the interval from simulation time0.
    Coseismic targets start at the first saved seismic state of each contiguous
    sampled episode; that anchor is not an exact onset estimate. A nonexact
    target requires two saved states of the same regime at most one interval
    apart, then uses the nearer state (ties go to the earlier state).
    First/last states of each regime are kept, with dashed extra contours.
    """
    if not profiles:
        raise ValueError('No saved profiles to select')
    if not np.isfinite([interseismic_years, coseismic_seconds, seismic_rate]).all() or min(
            interseismic_years, coseismic_seconds, seismic_rate) <= 0:
        raise ValueError('Require finite positive intervals and seismic threshold')
    if any(p.velocity is None for p in profiles):
        raise ValueError('Instantaneous V is required; sparse slip differences cannot classify an event')
    times = np.array([p.time for p in profiles])
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Profile times must be finite and strictly increasing')
    rates = np.array([np.max(np.abs(p.velocity)) for p in profiles])
    if not np.isfinite(rates).all():
        raise ValueError('Nonfinite instantaneous slip rate')
    seismic = rates >= seismic_rate
    boundaries = [0] + (np.flatnonzero(seismic[1:] != seismic[:-1])+1).tolist() + [len(profiles)]
    selected, targets, segments, transitions = {}, [], [], []
    # At century-scale clocks a one-second subtraction must retain the original
    # full-precision seconds, not rounded years. This tolerance is only clock ULPs.
    clock_tolerance = 8*np.spacing(max(1., float(np.max(np.abs(times)))))

    def retain(index, segment, target=None):
        record = selected.setdefault(index, dict(profile_index=index, step=profiles[index].step,
            time_s=profiles[index].time, time_yr=profiles[index].time/YEAR,
            regime='coseismic' if seismic[index] else 'interseismic',
            segment=segment, max_V_m_per_s=float(rates[index]), target_times_s=[]))
        if target is not None:
            record['target_times_s'].append(float(target))

    for segment, (begin, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        regime = 'coseismic' if seismic[begin] else 'interseismic'
        interval = coseismic_seconds if seismic[begin] else interseismic_years*YEAR
        anchor = times[begin] if seismic[begin] else 0.
        relative = times[begin:end]-anchor
        first_target = max(0, int(np.ceil((relative[0]-clock_tolerance)/interval)))
        last_target = int(np.floor((relative[-1]+clock_tolerance)/interval))
        if last_target-first_target > 100000:
            raise ValueError('More than100000 target contours in one regime; increase the plotting interval')
        retain(begin, segment)
        retain(end-1, segment)
        requested, omitted, errors = 0, 0, []
        for k in range(first_target, last_target+1):
            requested += 1
            offset = k*interval
            target = anchor+offset
            position = int(np.searchsorted(relative, offset))
            candidates = [i for i in (position-1, position) if 0 <= i < len(relative)]
            nearest = min(candidates, key=lambda i: abs(relative[i]-offset))
            row = dict(regime=regime, segment=segment, target_time_s=float(target),
                       target_offset_s=float(offset), anchor_time_s=float(anchor),
                       interval_s=float(interval), selected_step=None, actual_time_s=None,
                       time_error_s=None, bracketing_gap_s=None, status='omitted')
            exact = abs(relative[nearest]-offset) <= clock_tolerance
            if exact:
                row['bracketing_gap_s'] = 0.
            elif 0 < position < len(relative):
                row['bracketing_gap_s'] = float(relative[position]-relative[position-1])
            if exact or (row['bracketing_gap_s'] is not None
                         and row['bracketing_gap_s'] <= interval+clock_tolerance):
                index = begin+nearest
                retain(index, segment, target)
                error = float(relative[nearest]-offset)
                errors.append(abs(error))
                row.update(selected_step=profiles[index].step, actual_time_s=float(times[index]),
                           time_error_s=error, status='exact' if exact else 'nearest_saved')
            else:
                omitted += 1
            targets.append(row)
        gaps = np.diff(times[begin:end])
        segments.append(dict(segment=segment, regime=regime, first_step=profiles[begin].step,
            last_step=profiles[end-1].step, start_time_s=float(times[begin]),
            end_time_s=float(times[end-1]), saved_profiles=end-begin, interval_s=float(interval),
            anchor_time_s=float(anchor), target_count=requested, omitted_targets=omitted,
            maximum_target_time_error_s=max(errors, default=0.),
            minimum_saved_gap_s=float(min(gaps)) if len(gaps) else None,
            median_saved_gap_s=float(np.median(gaps)) if len(gaps) else None,
            maximum_saved_gap_s=float(max(gaps)) if len(gaps) else None))
        if begin:
            transitions.append(dict(previous_step=profiles[begin-1].step, next_step=profiles[begin].step,
                previous_time_s=float(times[begin-1]), next_time_s=float(times[begin]),
                unsampled_gap_s=float(times[begin]-times[begin-1])))
    chosen = []
    for index, record in sorted(selected.items()):
        record['selection_kind'] = 'target' if record['target_times_s'] else 'boundary'
        record['nearest_target_error_s'] = (record['time_s']-record['target_times_s'][0]
                                             if record['target_times_s'] else None)
        chosen.append(record)
    summary = dict(saved_profiles=len(profiles), plotted_profiles=len(chosen),
        plotted_interseismic=sum(r['regime']=='interseismic' for r in chosen),
        plotted_coseismic=sum(r['regime']=='coseismic' for r in chosen),
        interseismic_interval_years=interseismic_years, coseismic_interval_s=coseismic_seconds,
        seismic_threshold_m_per_s=seismic_rate, first_saved_time_s=float(times[0]),
        last_saved_time_s=float(times[-1]), last_saved_step=profiles[-1].step,
        last_saved_regime='coseismic' if seismic[-1] else 'interseismic',
        sampled_maximum_V_m_per_s=float(max(rates)),
        omitted_targets=sum(s['omitted_targets'] for s in segments),
        segments=segments, sampled_regime_transitions=transitions,
        selection='Nearest recorded state; require a same-regime bracketing gap no wider than the target interval. '
                  'Also retain each regime first/last state. No temporal interpolation, rate integration or extrapolation.',
        slip_reference='Original stored cumulative signed slip; no offset/subtraction.',
        event_completion='Not inferred from sparse profiles; final export may be incomplete.')
    return chosen, targets, summary


def write_outputs(profiles, chosen, targets, summary, prefix, xd_min_km, xd_max_km, title):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(prefix.parent / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    mask = (profiles[0].xd >= 1000*xd_min_km) & (profiles[0].xd <= 1000*xd_max_km)
    if np.count_nonzero(mask) < 2:
        raise ValueError('Down-dip window contains fewer than two saved nodes')
    order = np.argsort(profiles[0].xd[mask])
    colors = {'interseismic': '#215ac4', 'coseismic': '#d52228'}
#    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    fig, ax = plt.subplots(figsize=(5, 8))
    for record in chosen:
        p = profiles[record['profile_index']]
        ax.plot(p.slip[mask][order], p.xd[mask][order]/1000,
                color=colors[record['regime']], linewidth=1.5, #linewidth=.65,
                linestyle='--' if record['selection_kind']=='boundary' else '-', alpha=.95)
    ax.set(xlabel='Cumulative slip (m)', ylabel='Distance down-dip (km)',
           ylim=(xd_max_km, xd_min_km),
           title=f'{title}\n{profiles[0].time/YEAR:.2f}–{profiles[-1].time/YEAR:.2f} yr; saved profiles')
    ax.margins(x=.02)
    handles = [Line2D([0], [0], color=colors[regime], lw=1.2, label=label)
               for regime, label in [('interseismic', f'Interseismic: {summary["interseismic_interval_years"]:g} yr targets'),
                                     ('coseismic', f'Coseismic: {summary["coseismic_interval_s"]:g} s targets')]
               if any(r['regime']==regime for r in chosen)]
    if any(r['selection_kind']=='boundary' for r in chosen):
        handles.append(Line2D([0], [0], color='0.35', lw=1., ls='--', label='Extra transition/final saved states'))
    ax.legend(handles=handles, loc='upper right', fontsize=9, framealpha=.95)
    ax.tick_params(direction='in', top=True, right=True)
    omitted = summary['omitted_targets']
    note = f'Nearest recorded states; {omitted} unsupported target time(s) omitted. No time interpolation.'
    if summary['last_saved_regime']=='coseismic':
        note += '\nExport ends in the coseismic regime; event completion is not demonstrated.'
    fig.text(.5, .015, note, ha='center', va='bottom', fontsize=8)
    fig.tight_layout(rect=(0, .075, 1, 1))
    for extension in ('.png', '.pdf'):
        fig.savefig(str(prefix)+extension, dpi=220)
    plt.close(fig)

    # A separate coverage plot reveals missing early event data without changing
    # the contour plot or implying a completed first-event peak/decay.
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    seismic = [s for s in summary['segments'] if s['regime']=='coseismic']
    if seismic:
        anchor = seismic[0]['anchor_time_s']
        times = np.array([p.time-anchor for p in profiles])
        rates = np.array([max(abs(p.velocity)) for p in profiles])
        ax.semilogy(times, rates, '.', color='0.4', ms=3, label='Saved profiles')
        for record in chosen:
            if record['regime']=='coseismic':
                ax.plot(record['time_s']-anchor, record['max_V_m_per_s'], 'o', color=colors['coseismic'], ms=4)
        for s in seismic:
            begin = next(i for i, p in enumerate(profiles) if p.step==s['first_step'])
            end = next(i for i, p in enumerate(profiles) if p.step==s['last_step'])
            for left, right in zip(profiles[begin:end], profiles[begin+1:end+1]):
                if right.time-left.time > s['interval_s']:
                    ax.axvspan(left.time-anchor, right.time-anchor, color='#e7bd61', alpha=.3)
        duration = max(s['end_time_s'] for s in seismic)-anchor
        ax.set_xlim(-max(1., .05*duration), duration+max(.5, .02*duration))
        ax.set_xlabel('Time from first saved coseismic profile (s; not exact onset)')
        ax.plot([], [], 'o', color=colors['coseismic'], ms=4, label='Drawn coseismic contours')
        ax.axvspan(0, 0, color='#e7bd61', alpha=.3, label='Saved spacing exceeds coseismic interval')
    else:
        ax.semilogy([p.time/YEAR for p in profiles], [max(abs(p.velocity)) for p in profiles], '.-', color=colors['interseismic'])
        ax.set_xlabel('Simulation time (yr)')
    ax.axhline(summary['seismic_threshold_m_per_s'], color='0.25', ls='--', lw=.8, label='Seismic threshold')
    ax.set_ylabel('Saved max |V| over full fault (m/s)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(prefix)+'.sampling.png', dpi=180)
    plt.close(fig)

    selection_columns = ['step', 'time_s', 'time_yr', 'regime', 'segment', 'max_V_m_per_s',
                         'selection_kind', 'target_times_s', 'nearest_target_error_s', 'path', 'sha256']
    with Path(str(prefix)+'.selected.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=selection_columns)
        writer.writeheader()
        for r in chosen:
            row = {k: r.get(k) for k in selection_columns}
            row['target_times_s'] = ';'.join(format(t, '.17g') for t in r['target_times_s'])
            writer.writerow(row)
    with Path(str(prefix)+'.targets.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(targets[0]) if targets else [
            'regime', 'segment', 'target_time_s', 'status'])
        writer.writeheader(); writer.writerows(targets)
    Path(str(prefix)+'.summary.json').write_text(json.dumps(summary, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inputs', nargs='*', type=Path, help='Run directories or profiles.csv indexes in one continuous history')
    parser.add_argument('--interseismic-years', type=float, default=5.)
    parser.add_argument('--coseismic-seconds', type=float, default=1.)
    parser.add_argument('--seismic-rate', type=float, default=1e-3, metavar='M/S')
    parser.add_argument('--fault', type=int, default=0)
    parser.add_argument('--xd-min-km', type=float, default=0.)
    parser.add_argument('--xd-max-km', type=float, default=40.)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='Output prefix; creates PNG, PDF, sampling plot, selected/target CSVs and JSON')
    parser.add_argument('--title', default='BP3-derived server run')
    args = parser.parse_args()
    if not np.isfinite([args.xd_min_km, args.xd_max_km]).all() or args.xd_min_km >= args.xd_max_km:
        parser.error('Require a finite increasing down-dip window')
    try:
        profiles, provenance, indexes, duplicates = load_history(args.inputs or DEFAULT_INPUTS, args.fault)
        chosen, targets, summary = select_timed_profiles(profiles, args.interseismic_years,
                                                       args.coseismic_seconds, args.seismic_rate)
        for r in chosen:
            r.update(provenance[r['step']])
        summary.update(inputs=indexes, duplicate_profiles_discarded=duplicates, fault=args.fault,
                       input_profiles=[dict(step=p.step, time_s=p.time, **provenance[p.step]) for p in profiles],
                       xd_window_km=[args.xd_min_km, args.xd_max_km],
                       figure_reference='Erickson et al. (2023), Fig. 8 caption, doi:10.1785/0120220066; 5 yr blue / 1 s red.',
                       command=[sys.executable]+sys.argv,
                       script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        write_outputs(profiles, chosen, targets, summary, args.output, args.xd_min_km, args.xd_max_km, args.title)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    brief = {k: summary[k] for k in ['saved_profiles', 'plotted_profiles', 'plotted_interseismic',
             'plotted_coseismic', 'interseismic_interval_years', 'coseismic_interval_s', 'omitted_targets',
             'last_saved_step', 'last_saved_regime']}
    print(json.dumps(brief, indent=2))
    print(f'Wrote {args.output}.png and .pdf; see .selected.csv / .targets.csv / .summary.json for actual times.')
    if summary['omitted_targets']:
        print('Some target times are unsupported by the saved spacing; consult .sampling.png before interpreting contour gaps.')


if __name__ == '__main__':
    main()
