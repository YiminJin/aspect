#!/usr/bin/env python3
"""Read the supplied BP3 exports; write compact review evidence, never a run."""
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'benchmarks/reconstructed_fault/particle_free/bp3_dg_first_event'
OUT = ROOT / 'benchmarks/reconstructed_fault/particle_free/evidence/objective_stress'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    records, snapshots, manifest, logs = [], [], {}, {}
    for name in ('output_start', 'output_resumed'):
        directory = BASE / name
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                manifest[str(path.relative_to(ROOT))] = dict(bytes=path.stat().st_size, sha256=digest(path))
        log = (directory / 'log.txt').read_text()
        clocks = {int(s): (float(t), float(dt)) for s, t, dt in re.findall(
            r'\*\*\* Timestep (\d+):\s+t=([^ ]+) seconds, dt=([^ ]+) seconds', log)}
        accepted = [int(s) for s in re.findall(r'BP3 accepted state\s+(\d+)', log)]
        work = [tuple(map(float, row)) for row in re.findall(
            r'matching_shear_Pa=([^,]+).*?matching_raw_normal_Pa=([^,]+)', log)]
        iterations = [int(i) for i in re.findall(r'after nonlinear iteration\s+(\d+):', log)]
        logs[name] = dict(timestep_entries=len(clocks), accepted_count=len(accepted),
                          first_accepted=min(accepted), last_accepted=max(accepted),
                          last_accepted_time_s=clocks[max(accepted)][0],
                          last_begun=max(clocks), last_begun_clock=clocks[max(clocks)],
                          accepted_dt_range_s=[min(clocks[i][1] for i in accepted if i in clocks),
                                               max(clocks[i][1] for i in accepted if i in clocks)],
                          matching_work_max_Pa=np.max(work, axis=0).tolist() if work else None,
                          max_nonlinear_iteration=max(iterations),
                          publication_ratio_max=max(map(float, re.findall(r'maximum error/allowance=([^ ]+)', log)), default=None),
                          line_search_rejections=[int(i) for i in re.findall(r'accepted after (\d+) rejected candidates', log)],
                          cutback_lines=[line for line in log.splitlines() if re.search(r'repeat|cut.?back|reject.*timestep', line, re.I)])
        # The sampling index is authoritative for saved profiles, not evidence
        # that every accepted physical step has a field export.
        with (directory / 'profiles.csv').open() as stream:
            index = list(csv.DictReader(stream))
        for entry in index:
            path = directory / entry['file']
            a = np.genfromtxt(path, delimiter=',', names=True)
            assert np.all(np.isfinite(np.column_stack([a[c] for c in a.dtype.names])))
            assert np.all(a['Theta_s'] > 0) and np.all(a['V_m_per_s'] >= 0)
            assert len(np.unique(a['node'])) == len(a)
            imax = np.argmax(a['V_m_per_s'])
            row = dict(segment=name, step=int(entry['step']), time_s=float(entry['time_s']),
                       max_V=float(a['V_m_per_s'][imax]), peak_xd_m=float(a['xd_m'][imax]),
                       max_slip=float(np.max(a['slip_m'])), min_theta=float(np.min(a['Theta_s'])),
                       min_q=float(np.min(a['q_weak_Pa'])), max_q=float(np.max(a['q_weak_Pa'])),
                       min_sigma=float(np.min(a['sigma_n_weak_Pa'])), max_sigma=float(np.max(a['sigma_n_weak_Pa'])),
                       nodes=len(a))
            records.append(row)
            snapshots.append((row, a))
        logs[name]['profiles'] = len(index)
        logs[name]['saved_profile_last_step'] = max(int(e['step']) for e in index)
        logs[name]['line_search_rejections'] = dict(total=sum(logs[name]['line_search_rejections']),
                                                   solves=len(logs[name]['line_search_rejections']))
    records.sort(key=lambda r: (r['time_s'], r['step']))
    by_step = {}
    for record in records:
        if record['step'] in by_step:
            assert record['max_V'] == by_step[record['step']]['max_V']
        by_step[record['step']] = record
    unique = sorted(by_step.values(), key=lambda r: r['step'])
    peak = max(unique, key=lambda r: r['max_V'])
    onset_after = next(r for r in unique if r['max_V'] >= 1e-3)
    onset_before = unique[unique.index(onset_after)-1]
    resumed = [r for r in unique if r['segment'] == 'output_resumed']
    late = resumed[-1]
    # Differences over the saved event window are reported without extrapolating
    # to the unsaved accepted endpoint or inferring an event termination.
    first_data = next(a for r, a in snapshots if r['step'] == onset_before['step'])
    last_data = next(a for r, a in snapshots if r['step'] == late['step'])
    assert np.array_equal(first_data['node'], last_data['node'])
    summary = dict(logs=logs, saved_profiles=len(unique), onset_threshold_m_per_s=1e-3,
                   onset_bracket=[onset_before, onset_after], sampled_peak=peak,
                   last_saved_profile=late, last_to_sampled_peak_ratio=late['max_V']/peak['max_V'],
                   event_window_max_added_slip_m=float(np.max(last_data['slip_m']-first_data['slip_m'])),
                   first_event_complete=False,
                   completion_reason='No post-peak arrest criterion met or terminal event export; scheduler time limit during step 4238.',
                   source_identity='Exported runtime labels/inputs only; exact remote source and binary hash are absent.')
    (OUT / 'bp3-input-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (OUT / 'bp3-review.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (OUT / 'bp3-profile-history.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(unique)
    # Compact publication figure: saved samples only; no interpolated event peak.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    t0 = onset_after['time_s']
    event = [(r, a) for r, a in snapshots if r['time_s'] >= onset_before['time_s']]
    event.sort(key=lambda x: x[0]['time_s'])
    seismic = [(r, a) for r, a in event if r['time_s'] >= t0]
    event_t = np.array([r['time_s']-t0 for r, a in seismic])
    x = seismic[0][1]['xd_m']/1000
    order = np.argsort(x)
    axes[0, 0].semilogy([r['time_s']-t0 for r in unique], [r['max_V'] for r in unique], '.-')
    axes[0, 0].axhline(1e-3, color='gray', linewidth=.6)
    axes[0, 0].set_xlim(-2, event_t[-1]+.2)
    axes[0, 0].set(ylabel='Saved max V (m/s)', xlabel='Time from first saved seismic sample (s)')
    c = axes[0, 1].pcolormesh(event_t, x[order], np.array([np.log10(a['V_m_per_s'][order]) for r, a in seismic]).T,
                              shading='auto', vmin=-9, vmax=0)
    axes[0, 1].set_xlim(0, event_t[-1])
    fig.colorbar(c, ax=axes[0, 1], label='log10 V (m/s)')
    axes[0, 1].set(xlabel='Event time (s)', ylabel='Down-dip distance (km)')
    for row, a in (event[0], seismic[len(seismic)//2], event[-1]):
        label=f"step {row['step']}"
        axes[1, 0].plot(a['xd_m']/1000, a['slip_m']-first_data['slip_m'], label=label)
        axes[1, 1].plot(a['xd_m']/1000, a['sigma_n_weak_Pa']/1e6, label=label)
    axes[1, 0].set(xlabel='Down-dip distance (km)', ylabel='Added slip (m)')
    axes[1, 1].set(xlabel='Down-dip distance (km)', ylabel='Weak normal traction (MPa)')
    for ax in axes[1]: ax.legend(fontsize=8)
    fig.savefig(OUT / 'bp3-review.png', dpi=160)
    plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='logs'}, indent=2))


if __name__ == '__main__':
    main()
