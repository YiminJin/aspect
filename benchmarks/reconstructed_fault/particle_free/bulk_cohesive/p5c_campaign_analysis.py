#!/usr/bin/env python3
"""Compact physical-field analysis for the selected P5c continuation."""
from pathlib import Path
import argparse
import csv
import io
import json
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image
from p5c_analysis import read
from p5c_specimen_analysis import grid, history
from p5c_axial_analysis import q1_norm
from p5c_campaign import ROOT, WORK, AXIAL, OUT, save
from p5c_campaign_events import Events, write_rows, DX, DY, ELL

LABELS = {'fixed': 'Fixed-direction cohesive flow', 'radial': 'Radial tensor flow (direction free)'}


def stage_summary(record):
    run = WORK/'runs'/record['name']
    rows = list(csv.DictReader((run/'events.csv').open())) if (run/'events.csv').exists() else []
    a = read(run/'bulk-accepted.csv') if (run/'bulk-accepted.csv').exists() else []
    p = read(run/'bulk-profile.csv') if (run/'bulk-profile.csv').exists() else []
    s = read(run/'bulk-stability.csv') if (run/'bulk-stability.csv').exists() else []
    result = dict(record)
    result['last_events'] = rows[-1] if rows else None
    result['accepted_steps'] = len(a)
    resources = run/'resources.csv'
    if resources.exists():
        resource_rows = list(csv.DictReader(resources.open()))
        assert all(r['freeze'] == '1' and r['particles'] == '221184'
                   and r['particle_fingerprint'] == '13847991597444700578'
                   for r in resource_rows)
        if resource_rows:
            result['maximum_rank_peak_RSS_GiB'] = max(int(r['max_rank_KiB']) for r in resource_rows)/1024**2
            result['maximum_sum_rank_peaks_GiB'] = max(int(r['sum_rank_peaks_KiB']) for r in resource_rows)/1024**2
    if len(a):
        assert np.all(a['publications'] == a['step'])
        assert max(a['mechanical_residual']) <= 1e-9 and max(a['phase_KKT']) <= 1e-7 and max(a['projection_error']) <= 1e-12
        bound = 1e-10*np.maximum.reduce([np.ones(len(a)), abs(a['Wext']), a['bulk']+a['cohesive']+a['fracture']])
        assert max(abs(a['ledger_error'])/bound) <= 1
        result['maximum_checks'] = {key: float(max(abs(a[key]))) for key in ['mechanical_residual', 'phase_KKT', 'projection_error', 'ledger_error']}
    if len(p):
        result['cost_s'] = {key: float(sum(p[key])) for key in ['assembly', 'communication', 'linear', 'history', 'stability', 'total']}
        result['linear_solves'] = int(sum(p['linear_solves']))
    if len(s):
        result['max_attempted_critical_masters'] = int(max(s['critical_masters']))
        result['unstable_or_inconclusive_attempts'] = int(sum(s['stable'] == 0))
        result['stability_records'] = [{key: float(row[key]) for key in s.dtype.names} for row in s]
    log = ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence'/f'p5c-campaign-{record["name"]}.log'
    command_record = log.with_suffix('.json')
    if command_record.exists():
        result['wall_seconds'] = json.loads(command_record.read_text())['seconds']
    if log.exists():
        match = re.search(r'\| Create snapshot\s*\|\s*(\d+)\s*\|\s*([\d.e+]+)s', log.read_text())
        if match:
            result['checkpoint_timer'] = {'count': int(match[1]), 'seconds': float(match[2])}
    return result


def intermediate():
    state = json.loads((WORK/'state.json').read_text())
    results = {r['name']: stage_summary(r) for r in state['runs']}
    save(OUT/'intermediate-summaries.json', results)
    activity = []
    for record in state['runs']:
        run = WORK/'runs'/record['name']
        if not (run/'bulk-profile.csv').exists():
            continue
        profile = read(run/'bulk-profile.csv')
        stable = read(run/'bulk-stability.csv')
        for row in profile:
            step = int(row['step']); path = run/f'phase-forces-{step:04}.csv'
            if not path.exists():
                continue
            force = np.loadtxt(path, delimiter=',', skiprows=1)
            certificates = stable[stable['step'] == step]
            certificate = certificates[-1]
            assert certificate['stable'] == 1
            activity.append({'run': record['name'], 'mode': record['mode'], 'step': step, 'time_s': float(row['time']),
                'free_phase_nodes_exact': int(np.count_nonzero(force[:, 1] > force[:, 2])),
                'lower_obstacle_nodes_exact': int(np.count_nonzero(force[:, 1] == force[:, 2])),
                'critical_masters': int(certificate['critical_masters']), 'minimum_relaxed_curvature': float(certificate['minimum_curvature']),
                **{key+'_s': float(row[key]) for key in ['assembly', 'communication', 'linear', 'history', 'stability', 'total']},
                'linear_solves': int(row['linear_solves'])})
    if activity: write_rows(OUT/'activity-cost.csv', activity)
    return results


def view(mode):
    state = json.loads((WORK/'state.json').read_text())
    stages = [r for r in state['runs'] if r['mode'] == mode and not r['name'].startswith(('replay-', 'onset-'))]
    if not stages:
        return None
    destination = WORK/'analysis'/mode
    destination.mkdir(parents=True, exist_ok=True)
    sources = [(AXIAL/'runs'/f'{mode}-prefix', 1, 95), (AXIAL/'runs'/f'{mode}-active', 96, 127)]
    sources += [(WORK/'runs'/r['name'], 128, 100000) for r in stages]
    ledger = []; header = None
    for run, lower, upper in sources:
        path = run/'bulk-accepted.csv'
        if not path.exists():
            continue
        lines = path.read_text().splitlines()
        header = header or lines[0]
        chosen = [line for line in lines[1:] if lower <= int(line.split(',')[0]) <= upper]
        ledger += chosen
        for line in chosen:
            step = int(line.split(',')[0])
            for prefix in ['phase', 'means', 'boundary-reactions', 'phase-forces']:
                source = run/f'{prefix}-{step:04}.csv'; target = destination/source.name
                if source.exists() and not target.exists():
                    os.link(source, target)
    assert len({int(line.split(',')[0]) for line in ledger}) == len(ledger)
    (destination/'bulk-accepted.csv').write_text('\n'.join([header, *ledger])+'\n')
    prefix = AXIAL/'runs'/f'{mode}-prefix'
    for name in ['phase-0000.csv', 'means-0000.csv', 'boundary-reactions-0000.csv', 'seed-prescription.csv']:
        if not (destination/name).exists():
            os.link(prefix/name, destination/name)
    return destination


def physical_history(mode, run):
    state = json.loads((WORK/'state.json').read_text())
    load = state.get('loading_endpoints', {}).get(mode, .2)
    summary, rows = history(run, OUT, mode)
    detector = Events(WORK/f'{mode}-initial.csv')
    previous = detector.initial; events = []
    for row in rows:
        # The inherited helper called all growth outside initial support
        # "remote". Connected extension is not disconnected nucleation.
        row['growth_outside_initial_nodal_support_m2'] = row.pop('remote_growth_m2')
        phi = detector.phase(run/f'phase-{row["step"]:04}.csv')
        e = detector.measure(phi, previous, row['time_s'], row['step'])
        e['loading_displacement_m'] = .018*max(0, min(row['time_s'], load)-.05)
        row['imposed_relative_displacement_m'] = e['loading_displacement_m']
        row['growth_L2_m'] = q1_norm(detector.x, detector.y, phi-detector.initial)
        events.append(e); previous = phi
    write_rows(OUT/f'{mode}-history.csv', rows)
    write_rows(OUT/f'{mode}-fronts.csv', events)
    summary['loading_endpoint_s'] = load
    summary['initial_fracture_J_m'] = float(read(run/'bulk-accepted.csv')['fracture'][0])
    summary['added_fracture_J_m'] = rows[-1]['fracture']-summary['initial_fracture_J_m']
    summary['final_event'] = events[-1]
    summary['first_intact_bracket_s'] = next(([events[i-1]['time_s'], e['time_s']] for i, e in enumerate(events) if e['intact_activation']), None)
    summary['first_rapid_bracket_s'] = next(([events[i-1]['time_s'], e['time_s']] for i, e in enumerate(events) if e['rapid_growth']), None)
    initial_event = detector.measure(detector.initial, detector.initial, 0., 0)
    summary['initial_axial_fronts_m'] = {str(level): initial_event[f'axial_front_phi{level}_m'] for level in [.01, .05, .1]}
    summary['final_growth_regions'] = {}
    delta = phi-detector.initial
    for cutoff in [1e-8, .01, .05, .1]:
        mask = delta > cutoff
        summary['final_growth_regions'][str(cutoff)] = {'nodes': int(mask.sum()),
            'x_range_m': [float(detector.X[mask].min()), float(detector.X[mask].max())] if mask.any() else None,
            'y_range_m': [float(detector.Y[mask].min()), float(detector.Y[mask].max())] if mask.any() else None}
    summary['higher_damage_fronts_m'] = {}
    for level in [.2, .4, .6]:
        _, old = detector.components(detector.initial, level)
        _, current = detector.components(phi, level)
        summary['higher_damage_fronts_m'][str(level)] = {
            'initial': detector.front(detector.initial, old, level, True),
            'final': detector.front(phi, current, level, True)}
    return summary, rows, events


def support_segments(detector, cells=None):
    cells = detector.support if cells is None else cells
    result = []
    for j, i in zip(*np.where(cells)):
        x0, x1 = detector.x[i:i+2]; y0, y1 = detector.y[j:j+2]
        if i == 0 or not cells[j, i-1]: result.append([(x0, y0), (x0, y1)])
        if i+1 == cells.shape[1] or not cells[j, i+1]: result.append([(x1, y0), (x1, y1)])
        if j == 0 or not cells[j-1, i]: result.append([(x0, y0), (x1, y0)])
        if j+1 == cells.shape[0] or not cells[j+1, i]: result.append([(x0, y1), (x1, y1)])
    return result


def figures(runs, histories, summaries):
    radial = runs['radial']; ledger = read(radial/'bulk-accepted.csv')
    times = {0: 0., **{int(r['step']): float(r['time']) for r in ledger}}
    step_at = {round(t, 10): step for step, t in times.items()}
    start = 127; last = max(times)
    steps = sorted(set([start, last, *range(129, last+1, 4), *[step_at[t] for t in [.155, .175, .2] if t in step_at]]))
    endpoints = [s['loading_endpoint_s'] for s in summaries.values()]
    endpoints += [s['final']['time_s'] for s in summaries.values()]
    endpoint_steps = [step_at[round(t, 10)] for t in endpoints if round(t, 10) in step_at]
    steps.extend(endpoint_steps)
    for key in ['first_intact_bracket_s', 'first_rapid_bracket_s']:
        bracket = summaries['radial'][key]
        if bracket:
            steps.extend([step_at[round(t, 10)] for t in bracket])
    steps = sorted(set(steps))
    detector = Events(WORK/'radial-initial.csv'); cache = {}; field_rows = []; phase_rows = []
    # Select and preserve exact plotting values rather than re-evaluate laws.
    for mode, run in runs.items():
        for step in sorted(set([s for s in steps if (run/f'phase-{s:04}.csv').exists()] + [int(read(run/'bulk-accepted.csv')['step'][-1])])):
            phi = detector.phase(run/f'phase-{step:04}.csv')
            forces = read(run/f'phase-forces-{step:04}.csv')
            assert np.array_equal(forces['dof'].astype(int), detector.ids)
            weak = {}
            for key in ['mechanical_gradient', 'fracture_gradient', 'residual']:
                values = np.empty_like(phi)
                values[detector.iy, detector.ix] = forces[key]/forces['mass']
                weak[key] = values
            a = read(run/f'means-{step:04}.csv'); x, y, tau = grid(a, 'tauxy')
            drive = grid(a, 'Y')[2]; speed = grid(a, 'abs_rate')[2]
            cache[mode, step] = (phi, x, y, tau, drive, speed)
            physical_time = times[step] if step in times else float(read(run/'bulk-accepted.csv')['time'][read(run/'bulk-accepted.csv')['step']==step][0])
            for yy in [.5-ELL, .5, .5+ELL]:
                j = int(np.argmin(abs(detector.y-yy)))
                for i in range(len(detector.x)):
                    phase_rows.append({'mode': mode, 'time_s': physical_time,
                                       'x_m': detector.x[i], 'y_m': detector.y[j], 'initial_phi': detector.initial[j, i],
                                       'phi': phi[j, i], 'increment': phi[j, i]-detector.initial[j, i],
                                       'native_mechanical_driving_Pa': -weak['mechanical_gradient'][j, i],
                                       'native_fracture_resistance_Pa': weak['fracture_gradient'][j, i],
                                       'native_total_weak_residual_Pa': weak['residual'][j, i]})
                for xx in x:
                    keep = (abs(a['x']-xx)<1e-12)&(abs(a['y']-yy)<DY+1e-12)
                    weight = a['area'][keep]; weight = weight/weight.sum()
                    average = lambda key: float(weight@a[key][keep])
                    field_rows.append({'mode': mode, 'time_s': physical_time, 'x_m': xx, 'kernel_y_m': yy,
                        'physical_tauxy_Pa': average('tauxy'), 'physical_tauxx_Pa': average('tauxx'), 'physical_tauyy_Pa': average('tauyy'),
                        'physical_tauzz_Pa': -average('tauxx')-average('tauyy'),
                        'driving_Y_Pa': average('Y'), 'engineering_speed_per_s': average('abs_rate'),
                        'Axx_strain': average('Axx'), 'Axy_strain': average('Axy'),
                        'Kxx_strain': average('Kxx'), 'Kxy_strain': average('Kxy'),
                        'L_accumulated_strain': average('L'),
                        'Wxx_per_s': average('Wxx'), 'Wxy_per_s': average('Wxy')})
    tau_scale = max(float(abs(v[3]).max()) for v in cache.values())
    drive_scale = max(float(v[4].max()) for v in cache.values())
    speed_scale = max(float(v[5].max()) for v in cache.values())
    delta_scale = max(float((v[0]-detector.initial).max()) for v in cache.values())
    segments = support_segments(detector)
    def draw(items, zoom=False):
        fig, axes = plt.subplots(len(items), 5, figsize=(17, 3.4*len(items)), constrained_layout=True, squeeze=False)
        for row, (mode, step, t) in enumerate(items):
            phi, x, y, tau, drive, speed = cache[mode, step]
            cellphi = (phi[:-1, :-1]+phi[1:, :-1]+phi[:-1, 1:]+phi[1:, 1:])/4
            new = detector.virgin_cells & (cellphi > 1e-8)
            fields = [(detector.x, detector.y, phi, 'Phase φ', 0, 1, 'viridis'),
                      (detector.x, detector.y, phi-detector.initial, 'Increment from initial seed', 0, delta_scale, 'magma'),
                      (x, y, tau, 'Physical τxy [Pa]', -tau_scale, tau_scale, 'RdBu_r'),
                      (x, y, drive, 'Mechanical driving Y [Pa]', 0, drive_scale, 'inferno'),
                      (x, y, speed, 'Mean absolute slip-strain rate [1/s]', 0, speed_scale, 'plasma')]
            load = summaries[mode]['loading_endpoint_s']; displacement = .018*max(0, min(t, load)-.05)
            for ax, (xx, yy, values, title, lo, hi, cmap) in zip(axes[row], fields):
                im = ax.pcolormesh(xx, yy, values, shading='nearest', cmap=cmap, vmin=lo, vmax=hi)
                ax.contour(detector.x, detector.y, phi, levels=[.01, .05, .1, .3, .6, .9], colors='white', linewidths=.6)
                ax.add_collection(LineCollection(segments, colors='cyan', linewidths=.8, linestyles='dashed'))
                if new.any(): ax.add_collection(LineCollection(support_segments(detector, new), colors='lime', linewidths=.65))
                ax.plot(1, .5, 'c+', ms=5); ax.axvline(detector.support_front+ELL, color='.5', linewidth=.6, linestyle=':')
                ax.set_aspect('equal'); ax.set_xlim(.9 if zoom else 0, 1.35 if zoom else 1.5); ax.set_ylim(.3 if zoom else 0, .7 if zoom else 1)
                ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_title(f'{LABELS[mode]}\nt={t:.5f} s; u={displacement*1000:.4f} mm\n{title}', fontsize=9)
                fig.colorbar(im, ax=ax, shrink=.75)
        fig.suptitle('Cyan dashed: initial represented support; lime: newly damaged initially-zero cells; white: phase contours\nConstitutive flow direction is distinct from crack geometry; stress is physical τ, not auxiliary T.', fontsize=10)
        return fig
    selected = sorted(set([start, last, *endpoint_steps, *[step_at[t] for t in [.155, .175, .2] if t in step_at]]))
    for step in selected:
        items = [(mode, step, times[step]) for mode in ['radial', 'fixed'] if (mode, step) in cache]
        for zoom in [False, True]:
            fig = draw(items, zoom); fig.savefig(OUT/f'fields-{step:04}{"-tip" if zoom else ""}.png', dpi=150); plt.close(fig)
    images = []
    for step in steps:
        fig = draw([('radial', step, times[step])]); stream = io.BytesIO(); fig.savefig(stream, format='png', dpi=90); plt.close(fig); stream.seek(0)
        images.append(Image.open(stream).convert('P', palette=Image.Palette.ADAPTIVE))
    images[0].save(OUT/'radial-continuation.gif', save_all=True, append_images=images[1:], duration=450, loop=0)
    write_rows(OUT/'plotted-phase.csv', phase_rows); write_rows(OUT/'plotted-physical-fields.csv', field_rows)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for mode, rows in histories.items():
        t = [r['time_s'] for r in rows]; u = [r['imposed_relative_displacement_m']*1000 for r in rows]
        axes[0].plot(t, [r['integral_growth_m2'] for r in rows], label=LABELS[mode])
        axes[1].plot(u, [r['top_horizontal_reaction_N_per_m'] for r in rows], label=LABELS[mode])
        axes[2].plot(t[1:], [r['trapezoid_physical_remainder_J_m'] for r in rows[1:]], label=LABELS[mode])
    for ax, title in zip(axes, ['Integrated phase increment [m²]', 'Top horizontal reaction [N/m]', 'Trapezoidal physical remainder [J/m]']):
        ax.set_ylabel(title); ax.set_xlabel('time [s]'); ax.legend(fontsize=7); ax.grid(alpha=.2)
    axes[1].set_xlabel('Relative grip displacement [mm]')
    fig.savefig(OUT/'reaction-growth-work.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(2, len(histories), figsize=(6*len(histories), 6),
                             constrained_layout=True, squeeze=False)
    for column, (mode, rows) in enumerate(histories.items()):
        selected_rows = [r for r in rows if r['step'] >= 127]
        incoming = selected_rows[0]
        t = np.array([r['time_s'] for r in selected_rows])
        storage = np.array([r['bulk']+r['cohesive']+r['fracture'] for r in selected_rows])
        storage -= storage[0]
        work = np.array([r['trapezoidal_reaction_work_J_m']-incoming['trapezoidal_reaction_work_J_m'] for r in selected_rows])
        dissipated = np.array([r['cumulative_dissipation_J_m']-incoming['cumulative_dissipation_J_m'] for r in selected_rows])
        remainder = work-storage-dissipated
        axes[0,column].plot(t,work,label='Trapezoidal boundary work')
        axes[0,column].plot(t,storage,label='Storage change')
        axes[0,column].plot(t,dissipated,label='Physical dissipation')
        axes[0,column].set_ylabel('Energy per thickness [J/m]')
        axes[0,column].set_title(LABELS[mode],fontsize=10)
        axes[1,column].plot(t,remainder,label='Signed physical remainder')
        axes[1,column].plot(t,np.cumsum(abs(np.diff(remainder,prepend=0.))),label='Sum of absolute step remainders')
        axes[1,column].set_ylabel('Physical work remainder [J/m]')
        for ax in axes[:,column]:
            ax.axvline(summaries[mode]['loading_endpoint_s'],color='.5',linestyle='--',linewidth=.8)
            ax.set_xlabel('time [s]');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Continuation accounting relative to accepted .13875 s state; dashed line: loading ends.\nBackward-Euler ledger closure and its separate numerical terms remain in the CSV histories.',fontsize=10)
    fig.savefig(OUT/'energy-accounting.png',dpi=170);plt.close(fig)
    save(OUT/'plot-definitions.json', {'phase': 'Q1 phase and its change from each mesh own initial represented seed',
         'fields': 'actual Gauss8 physical cell means; no constitutive-law evaluation after averaging',
         'initial_support': 'exact cell-union boundary of nonzero initial Q1 field, cyan dashed',
         'new_cells': 'initially all-zero Q1 cells with current mean phi>1e-8, green boundary',
         'profile_kernel': 'cell-width DX by 2DY=.015625 m, centered at y=.5,.5±ell. Average actual physical cell integrals, never re-evaluate the constitutive law on averages.',
         'native_weak_profiles': 'Complete native mechanical/fracture/total phase gradients divided by the same Q1 lumped test mass, in Pa. These nodal tests differ from cell-mean Y; driving is minus the mechanical gradient.',
         'growth_outside_initial_nodal_support': 'May be seed-connected extension. Disconnected/remote nucleation is measured separately by the declared component detector.',
         'contours': [.01, .05, .1, .3, .6, .9], 'GIF': 'primary radial trajectory at actual accepted times, whole domain, no synthetic frozen continuation',
         'GIF_playback': 'Equal playback duration per saved frame, with nonuniform selected time spacing. Read the actual time/displacement labels; playback is not a validated propagation speed.',
         'scales': {'delta_phi': delta_scale, 'absolute_tau_Pa': tau_scale, 'Y_Pa': drive_scale, 'slip_strain_rate_per_s': speed_scale},
         'time_step_s': .000625, 'cell_sizes_m': [DX, DY], 'target_x_m': detector.support_front+ELL})


def final(plots=True):
    intermediate(); runs = {}; histories = {}; summaries = {}
    for mode in ['radial', 'fixed']:
        run = view(mode)
        if run:
            runs[mode] = run
            summaries[mode], histories[mode], events = physical_history(mode, run)
    save(OUT/'physical-summary.json', summaries)
    if len(histories) == 2:
        by_time = {mode: {round(row['time_s'], 10): row for row in rows}
                   for mode, rows in histories.items()}
        common_loading_end = min(s['loading_endpoint_s'] for s in summaries.values())
        comparison = []
        for t in sorted(set(by_time['radial']) & set(by_time['fixed'])):
            r, f = by_time['radial'][t], by_time['fixed'][t]
            comparison.append({'time_s': t, 'same_loading_history': t <= common_loading_end+1e-12,
                'radial_integrated_increment_m2': r['integral_growth_m2'],
                'fixed_integrated_increment_m2': f['integral_growth_m2'],
                'radial_peak_increment': r['peak_increment'], 'fixed_peak_increment': f['peak_increment'],
                'integrated_radial_reduction': 1-r['integral_growth_m2']/f['integral_growth_m2'] if f['integral_growth_m2'] > 0 else None,
                'radial_top_reaction_N_m': r['top_horizontal_reaction_N_per_m'],
                'fixed_top_reaction_N_m': f['top_horizontal_reaction_N_per_m']})
        write_rows(OUT/'matched-mode-comparison.csv', comparison)
    if plots: figures(runs, histories, summaries)
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['intermediate', 'final', 'data'])
    args = parser.parse_args()
    intermediate() if args.action == 'intermediate' else final(args.action == 'final')
