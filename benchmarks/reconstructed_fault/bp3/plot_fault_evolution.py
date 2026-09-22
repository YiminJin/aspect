"""Plot saved BP3 fault profiles and native properties, without rerunning ASPECT."""
import argparse
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import warnings

import numpy as np

YEAR = 31557600.
FIELDS = ['slip_m', 'V_m_per_s', 'Theta_s', 'q_weak_Pa', 'sigma_n_weak_Pa']


def read_profiles(run, skip_missing=False):
    with (run/'profiles.csv').open() as stream:
        entries = list(csv.DictReader(stream))
    unique = []
    for entry in entries:
        if unique and entry == unique[-1]:
            continue
        if unique and (int(entry['step']) <= int(unique[-1]['step'])
                       or float(entry['time_s']) <= float(unique[-1]['time_s'])):
            raise ValueError('Profile index has conflicting duplicate or nonincreasing states')
        unique.append(entry)
    missing = [entry['file'] for entry in unique if not (run/entry['file']).is_file()]
    if missing:
        if not skip_missing:
            raise ValueError(f'Missing {len(missing)} indexed profiles; --skip-missing permits '
                             'plotting only the available copied data')
        warnings.warn(f'Omitting missing profiles: {missing}')
    duplicates = len(entries)-len(unique)
    unique = [entry for entry in unique if (run/entry['file']).is_file()]
    if len(unique) < 2:
        raise ValueError('Need at least two saved profiles')
    history = {name: [] for name in FIELDS}
    geometry = None
    for entry in unique:
        path = run/entry['file']
        with path.open() as stream:
            header = next(csv.reader(stream))
        data = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
        columns = {name: data[:, i] for i, name in enumerate(header)}
        if not np.isfinite(data).all() or np.any(columns['fault'] != 0):
            raise ValueError(f'Nonfinite data or multiple faults in {path}')
        if (np.any(columns['step'] != int(entry['step']))
                or np.any(columns['time_s'] != float(entry['time_s']))):
            raise ValueError(f'Profile/index timestamp mismatch: {path}')
        current = np.column_stack([columns[k] for k in ['node', 'xd_m', 'x_m', 'y_m']])
        if geometry is None:
            geometry = current
            order = np.argsort(columns['xd_m'])
            if (len(np.unique(columns['node'])) != len(data)
                    or np.any(np.diff(columns['xd_m'][order]) <= 0)):
                raise ValueError('Duplicate fault nodes or down-dip coordinates')
        elif not np.array_equal(current, geometry):
            raise ValueError(f'Changed geometry or incomplete profile: {path}')
        for name in FIELDS:
            history[name].append(columns[name][order])
    return (geometry[order], np.array([float(e['time_s'])/YEAR for e in unique]),
            np.array([int(e['step']) for e in unique]),
            {name: np.array(values) for name, values in history.items()},
            duplicates, missing)


def native_properties(run, geometry, skip_missing=False):
    entries = ET.parse(run/'reconstructed_faults.pvd').findall('.//DataSet')
    fields = {}
    count = 0
    missing = []
    times = []
    for entry in entries:
        path = run/entry.attrib['file']
        if not path.is_file() and skip_missing:
            missing.append(entry.attrib['file'])
            continue
        root = ET.parse(path)
        xyz = np.fromstring(root.find('.//Points/DataArray').text, sep=' ').reshape(-1, 3)
        nodes = geometry[:, 0].astype(int)
        if not np.allclose(xyz[nodes, :2], geometry[:, 2:], rtol=0, atol=1e-8):
            raise ValueError('Native fault geometry differs from profile geometry')
        for array in root.findall('.//PointData/DataArray'):
            name = array.attrib['Name']
            if name not in {'previous_I_h', 'cohesive_traction', 'composition_strengthening',
                            'background_tractions'}:
                continue
            if array.attrib.get('format') != 'ascii':
                raise ValueError('This reader expects the native ASCII fault VTU format')
            components = int(array.attrib.get('NumberOfComponents', 1))
            values = np.fromstring(array.text, sep=' ').reshape(-1, components)[nodes]
            if not np.isfinite(values).all():
                raise ValueError(f'Nonfinite native property {name}')
            if name not in fields:
                fields[name] = dict(initial=values.copy(), final=values.copy(),
                                    minimum=values.copy(), maximum=values.copy())
            fields[name]['final'] = values
            fields[name]['minimum'] = np.minimum(fields[name]['minimum'], values)
            fields[name]['maximum'] = np.maximum(fields[name]['maximum'], values)
        count += 1
        times.append(float(entry.attrib['timestep'])/YEAR)
    if missing:
        warnings.warn(f'Omitting missing native files: {missing}')
    return fields, count, missing, times


def edges(values):
    """Display samples on nearest-time/node bins, without smoothing/interpolation."""
    return np.r_[values[0], (values[:-1]+values[1:])/2, values[-1]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, help='Figure directory (default: run/fault_evolution)')
    parser.add_argument('--xd-max-km', type=float, default=45.)
    parser.add_argument('--skip-missing', action='store_true',
                        help='Explicitly omit missing indexed files from an incomplete copy; report all gaps')
    parser.add_argument('--times-years', type=float, nargs='+', default=[0, 1, 5, 20, 50, 100, 200, 300],
                        help='Nearest saved states; final state always included')
    args = parser.parse_args()
    geometry, times, steps, history, duplicates, missing = read_profiles(args.run, args.skip_missing)
    fixed, native_count, missing_native, native_times = native_properties(args.run, geometry, args.skip_missing)
    xd = geometry[:, 1]/1000
    if args.xd_max_km <= xd[0] or np.count_nonzero(xd <= args.xd_max_km) < 2:
        parser.error('Down-dip window must contain at least two vertices')
    selected = sorted(set([int(np.argmin(np.abs(times-t))) for t in args.times_years
                           if times[0] <= t <= times[-1]] + [0, len(times)-1]))
    output = args.output or args.run/'fault_evolution'
    output.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.cm import ScalarMappable

    # Mechanics tractions are projected accepted weak loads; Theta is updated state.
    q = history['q_weak_Pa']/1e6
    normal = history['sigma_n_weak_Pa']/1e6
    panels = [('Slip rate (m/s)', history['V_m_per_s'], True),
              ('Committed slip state Θ (yr)', history['Theta_s']/YEAR, True),
              ('Accumulated slip (m)', history['slip_m'], False),
              ('Total weak shear traction q (MPa)', q, False),
              ('Total weak normal traction σₙ (MPa)', normal, False),
              ('Weak normal change σₙ − σₙ,₀ (MPa)', normal-normal[0], False)]
    cmap = plt.get_cmap('viridis')
    norm = Normalize(times[0], times[-1])
    for limit, suffix in [(args.xd_max_km, 'shallow'), (xd[-1], 'full_fault')]:
        mask = xd <= limit
        fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True, layout='constrained')
        for ax, (label, values, logarithmic) in zip(axes.flat, panels):
            for i in selected:
                if logarithmic and np.any(values[i, mask] <= 0):
                    raise ValueError(f'Cannot logarithmically plot nonpositive {label}')
                ax.plot(xd[mask], values[i, mask], color=cmap(norm(times[i])), lw=.9)
            if logarithmic:
                ax.set_yscale('log')
            ax.set_ylabel(label)
            ax.set_xlim(0, limit)
            ax.grid(alpha=.2)
            for location in (15, 18, 40):
                ax.axvline(location, ls=':', color='0.6', lw=.65)
        for ax in axes[-1]:
            ax.set_xlabel('Down-dip distance (km)')
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, label='Saved physical time (yr)', shrink=.8)
        fig.suptitle('Modified BP3 — fault-property profiles\n'
                     'Actual saved times (yr): '+', '.join(f'{times[i]:.2f}' for i in selected), fontsize=12)
        fig.savefig(output/f'profiles_{suffix}.png', dpi=180)
        plt.close(fig)

    # Irregular saved times retain their physical spacing. Colour rectangles
    # represent nearest saved samples, not newly solved intermediate states.
    mask = xd <= args.xd_max_km
    maps = panels[:3] + [('Weak shear change q − q₀ (MPa)', q-q[0], False), panels[-1],
                        ('Element slip gradient (m/km)',
                         np.diff(history['slip_m'], axis=1)/np.diff(xd), False)]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True, layout='constrained')
    for index, (ax, (label, values, logarithmic)) in enumerate(zip(axes.flat, maps)):
        if index == 5:
            n = np.count_nonzero(mask)
            z = values[:, :n-1].T
            yedges = xd[:n]
        else:
            z = values[:, mask].T
            yedges = edges(xd[mask])
        scale = None
        color = 'viridis'
        if logarithmic:
            if np.any(z <= 0):
                raise ValueError(f'Nonpositive {label}')
            scale = LogNorm(vmin=z.min(), vmax=z.max())
        elif z.min() < 0 < z.max():
            limit = float(np.max(np.abs(z)))
            scale = Normalize(vmin=-limit, vmax=limit)
            color = 'RdBu_r'
        mesh = ax.pcolormesh(edges(times), yedges, z, shading='flat', cmap=color, norm=scale,
                             rasterized=True)
        ax.set_title(label, fontsize=10)
        ax.set_ylim(args.xd_max_km, 0)
        ax.set_ylabel('Down-dip distance (km)')
        for location in (15, 18, 40):
            ax.axhline(location, color='0.5', ls=':', lw=.6)
        fig.colorbar(mesh, ax=ax, shrink=.85)
    for ax in axes[-1]:
        ax.set_xlabel('Physical time (yr)')
    fig.suptitle('Saved fault evolution — no smoothing\n'
                 'Midpoint time bins; traction changes referenced to initialization', fontsize=12)
    fig.savefig(output/'space_time_shallow.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, layout='constrained')
    for ax, name, label, factor in zip(axes.flat,
            ['previous_I_h', 'cohesive_traction', 'composition_strengthening', 'background_tractions'],
            ['Retained previous I_h (m)', 'Cohesive history (MPa)',
             'Strengthening fraction', 'Background tractions (MPa)'], [1., 1e-6, 1., 1e-6]):
        field = fixed[name]
        for c in range(field['initial'].shape[1]):
            component = ('shear' if c == 0 else 'normal') if name == 'background_tractions' else name
            ax.fill_between(xd, factor*field['minimum'][:, c], factor*field['maximum'][:, c], alpha=.2)
            ax.plot(xd, factor*field['initial'][:, c], label=f'{component}: initial', lw=1)
            ax.plot(xd, factor*field['final'][:, c], '--', label=f'{component}: last VTU', lw=.9)
        ax.set_ylabel(label)
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    for ax in axes[-1]:
        ax.set_xlabel('Down-dip distance (km)')
    fig.suptitle(f'Native fault properties ({native_count} snapshots, '
                 f'{native_times[0]:.2f}–{native_times[-1]:.2f} yr)\n'
                 'Shading is min–max over saved snapshots, not uncertainty', fontsize=12)
    fig.savefig(output/'native_property_ranges.png', dpi=180)
    plt.close(fig)

    summary = dict(saved_profiles=len(times), skipped_duplicate_index_rows=duplicates,
                   first_year=float(times[0]), last_year=float(times[-1]),
                   first_step=int(steps[0]), last_step=int(steps[-1]), vertices=len(xd),
                   native_snapshots=native_count,
                   native_time_range_years=[native_times[0], native_times[-1]],
                   missing_indexed_profiles=missing, missing_indexed_native_files=missing_native,
                   selected_profiles=[dict(step=int(steps[i]), year=float(times[i])) for i in selected],
                   native_max_change={name: float(np.max(field['maximum']-field['minimum']))
                                      for name, field in fixed.items()},
                   ranges={name: dict(minimum=float(value.min()), maximum=float(value.max()),
                                      final_minimum=float(value[-1].min()), final_maximum=float(value[-1].max()))
                           for name, value in history.items()})
    accepted = args.run/'accepted_steps.csv'
    if accepted.is_file():
        with accepted.open() as stream:
            last = list(csv.DictReader(stream))[-1]
        summary['accepted_log_last_step'] = int(last['step'])
        summary['accepted_log_last_year'] = float(last['time'])/YEAR
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (output/'README.md').write_text(
        '# Saved reconstructed-fault evolution\n\n'
        'All plots use saved accepted-state data only; no simulation was run.\n\n'
        '- `profiles_shallow.png`: 0–45 km (or requested window).\n'
        '- `profiles_full_fault.png`: full fault, including both boundary endpoints.\n'
        '- `space_time_shallow.png`: all saved profiles at their irregular physical times; '
        'rectangles use midpoint boundaries, without smoothing. Element slip gradients '
        'are differences of neighbouring Q1 nodal slip values divided by element length.\n'
        '- `native_property_ranges.png`: initial/final native fields with min–max bands.\n\n'
        'V and slip are accepted kinematic fields. Theta is the committed post-update state; '
        'the split mechanical solve used the preceding Theta. q and sigma_n are the '
        'consistent-Q1 representations M^-1 times the accepted mechanical weak traction '
        'loads, including background; these are not raw constitutive samples or bulk '
        'stress-history fields. Differences subtract each node’s initialized projected '
        'traction. Positive sigma_n means compression. `previous_I_h` retains its history '
        'meaning; it is not relabelled as a separately measured current integral.\n\n'
        '15/18 km mark friction transitions; 40 km is only a historical landmark in the '
        'fully frictional configuration, not a prescribed/free interface. '
        'Exact repeated index rows are skipped; original outputs are unchanged.\n')
    if missing or missing_native:
        with (output/'README.md').open('a') as stream:
            stream.write('\n## Incomplete copied output\n\n'
                         f'{len(missing)} indexed profile files and {len(missing_native)} native '
                         'files are absent and were explicitly omitted using `--skip-missing`. '
                         'See summary.json for their names. These plots do not establish a '
                         'coherent restart state or cover the absent final profiles.\n')
    if summary.get('accepted_log_last_step', int(steps[-1])) != int(steps[-1]):
        with (output/'README.md').open('a') as stream:
            stream.write(f'\nThe copied accepted-state log ends at step '
                         f'{summary["accepted_log_last_step"]}, whereas available profile CSVs '
                         f'reach {steps[-1]}. Trailing profiles are plotted as exported, '
                         'but are not independently cross-validated against that incomplete log.\n')
    print(json.dumps(summary, indent=2))
    print(f'Figures written to {output}')


if __name__ == '__main__':
    main()
