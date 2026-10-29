#!/usr/bin/env python3
"""Focused, same-incoming-history temporal comparisons; no branch alignment."""
from pathlib import Path
import json
import numpy as np
from p5c_analysis import read
from p5c_axial_analysis import q1_norm
from p5c_campaign import ROOT, WORK, OUT, save
from p5c_campaign_events import Events, write_rows


def reaction(path):
    a = read(path)
    return a[np.lexsort((a['component'], a['y'], a['x']))]


def work_history(run, parent_phase, parent_run, parent_step, allowed_steps=None):
    ledger = read(run/'bulk-accepted.csv')
    incoming = read(parent_run/'bulk-accepted.csv')
    incoming = incoming[incoming['step'] == parent_step][0]
    baseline = float(incoming['bulk']+incoming['cohesive']+incoming['fracture'])
    old = reaction(parent_run/f'boundary-reactions-{parent_step:04}.csv')
    cumulative_work = endpoint = dissipated = absolute_work = absolute_error = 0.
    rows = {}; prior_storage = baseline
    for row in ledger:
        step = int(row['step'])
        if allowed_steps is not None and step not in allowed_steps:
            continue
        r = reaction(run/f'boundary-reactions-{step:04}.csv')
        assert np.array_equal(r[['component', 'x', 'y']], old[['component', 'x', 'y']])
        w = float(.5*(r['force']+old['force'])@r['increment'])
        wb = float(r['force']@r['increment'])
        energy = float(row['bulk']+row['cohesive']+row['fracture'])
        d = float(row['Db']+row['Dc']+row['Df'])
        remainder = w-(energy-prior_storage)-d
        absolute_error += abs(remainder); absolute_work += abs(w)
        cumulative_work += w; endpoint += wb; dissipated += d
        rows[round(float(row['time']), 10)] = {'time_s': float(row['time']), 'step': step,
            'trapezoid_work_J_m': cumulative_work, 'endpoint_work_J_m': endpoint,
            'storage_change_J_m': energy-baseline, 'physical_dissipation_J_m': dissipated,
            'signed_physical_remainder_J_m': cumulative_work-(energy-baseline)-dissipated,
            'sum_absolute_step_physical_remainders_J_m': absolute_error,
            'absolute_trapezoid_work_J_m': absolute_work,
            'top_reaction_N_m': float(np.sum(r['force'][(r['component'] == 0)&(abs(r['y']-1) < 1e-12)]))}
        old = r; prior_storage = energy
    return rows


def main():
    state = json.loads((WORK/'state.json').read_text()); results = {}; maps = []
    for record in [r for r in state['runs'] if r['name'].startswith(('replay-', 'onset-'))]:
        name, mode = record['name'], record['mode']
        run = WORK/'runs'/name; base = WORK/'analysis'/mode
        parent = ROOT/record['parent']; meta = json.loads((parent/'metadata.json').read_text())
        if not (run/'bulk-accepted.csv').exists():
            results[name] = {'exit_code': record['exit_code'], 'accepted_replay_steps': 0,
                             'scope': 'native limit before first publication; verification not completed'}
            continue
        detector = Events(WORK/f'{mode}-initial.csv')
        initial = detector.phase(parent/'phase.csv')
        parent_step, start = meta['step'], meta['time_s']
        end = record['actual_end_s']
        a, b = read(base/'bulk-accepted.csv'), read(run/'bulk-accepted.csv')
        selected = a[(a['time'] > start+1e-12)&(a['time'] <= end+1e-12)]
        base_work = work_history(base, initial, base, parent_step, set(selected['step']))
        fine_work = work_history(run, initial, base, parent_step)
        write_rows(OUT/f'{name}-work.csv', list(fine_work.values()))
        at = {round(float(r['time']), 10): int(r['step']) for r in selected}
        bt = {round(float(r['time']), 10): int(r['step']) for r in b}
        comparisons = []; previous_coarse = initial
        for t in sorted(set(at)&set(bt)):
            coarse = detector.phase(base/f'phase-{at[t]:04}.csv')
            fine = detector.phase(run/f'phase-{bt[t]:04}.csv')
            norm = q1_norm(detector.x, detector.y, fine-initial)
            peak = float(np.max(fine-initial))
            co = detector.measure(coarse, initial, t, at[t]); fi = detector.measure(fine, initial, t, bt[t])
            coarse_fields = read(base/f'means-{at[t]:04}.csv')
            fine_fields = read(run/f'means-{bt[t]:04}.csv')
            assert np.array_equal(coarse_fields['cell'], fine_fields['cell'])
            stress_norm2 = lambda xx, yy, xy: xx*xx+yy*yy+(xx+yy)**2+2*xy*xy
            area = fine_fields['area']
            stress_difference = sum(area*stress_norm2(*[coarse_fields[key]-fine_fields[key]
                                                       for key in ['tauxx', 'tauyy', 'tauxy']]))
            stress_scale = sum(area*stress_norm2(*[fine_fields[key] for key in ['tauxx', 'tauyy', 'tauxy']]))
            cw, fw = base_work[t], fine_work[t]
            row = {'time_s': t, 'coarse_step': at[t], 'half_step': bt[t],
                   'coarse_accepted_step_peak_increment': float(np.max(coarse-previous_coarse)),
                   'window_growth_L2_m': norm, 'window_peak_increment': peak,
                   'growth_relative_L2_difference': q1_norm(detector.x, detector.y, coarse-fine)/norm if peak > 1e-8 else None,
                   'peak_field_difference_over_window_growth': float(np.max(abs(coarse-fine)))/peak if peak > 1e-8 else None,
                   'relative_cell_mean_full_tensor_stress_difference': float(np.sqrt(stress_difference/stress_scale)),
                   'relative_cell_mean_Y_difference': float(np.sqrt(np.sum(area*(coarse_fields['Y']-fine_fields['Y'])**2)/np.sum(area*fine_fields['Y']**2))),
                   'reaction_relative_difference': abs(cw['top_reaction_N_m']-fw['top_reaction_N_m'])/max(abs(fw['top_reaction_N_m']), 1e-300),
                   'work_difference_J_m': abs(cw['trapezoid_work_J_m']-fw['trapezoid_work_J_m']),
                   'storage_difference_J_m': abs(cw['storage_change_J_m']-fw['storage_change_J_m']),
                   'dissipation_difference_J_m': abs(cw['physical_dissipation_J_m']-fw['physical_dissipation_J_m'])}
            for level in [.01, .05, .1]:
                key = f'axial_front_phi{level}_m'
                row[f'coarse_front_phi{level}_m'] = co[key]; row[f'half_front_phi{level}_m'] = fi[key]
                row[f'front_phi{level}_difference_m'] = co[key]-fi[key]
            comparisons.append(row)
            previous_coarse = coarse
        write_rows(OUT/f'{name}-temporal.csv', comparisons)
        fine_events = []; previous = initial
        for row in b:
            step = int(row['step']); phi = detector.phase(run/f'phase-{step:04}.csv')
            fine_events.append(detector.measure(phi, previous, float(row['time']), step)); previous = phi
        write_rows(OUT/f'{name}-events.csv', fine_events)
        first = next((i for i, row in enumerate(fine_events) if row['intact_activation']), None)
        bracket = ([start if first == 0 else fine_events[first-1]['time_s'], fine_events[first]['time_s']]
                   if first is not None and detector.measure(initial, initial, start, parent_step)['initially_zero_cell_peak'] <= 1e-8 else None)
        last = fine_work[max(fine_work)]
        scale = last['absolute_trapezoid_work_J_m']
        c_e = last['sum_absolute_step_physical_remainders_J_m']/scale if scale else None
        c_t = max(r['work_difference_J_m']+r['storage_difference_J_m']+r['dissipation_difference_J_m'] for r in comparisons)/scale if scale and comparisons else None
        results[name] = {'mode': mode, 'incoming_time_s': start, 'end_time_s': end,
            'loading_endpoint_s': record['loading_end_s'], 'exit_code': record['exit_code'],
            'first_intact_bracket_s': bracket,
            'maximum_coarse_step_phase_increment': max((r['coarse_accepted_step_peak_increment'] for r in comparisons), default=None),
            'maximum_half_step_phase_increment': max((r['max_step_increment'] for r in fine_events), default=None),
            'maximum_growth_relative_L2_difference': max((r['growth_relative_L2_difference'] or 0) for r in comparisons) if comparisons else None,
            'maximum_peak_relative_difference': max((r['peak_field_difference_over_window_growth'] or 0) for r in comparisons) if comparisons else None,
            'final': comparisons[-1] if comparisons else None,
            'work_scale_J_m': scale, 'C_E': c_e, 'C_T': c_t,
            'B_E': c_e+c_t if c_e is not None and c_t is not None else None,
            'full_temporal_qualification': False,
            'scope': 'same accepted incoming histories; local two-step Cauchy check, no mirror alignment, no global temporal/order claim'}
        if comparisons and name.startswith('replay-'):
            maps.append((mode, start, comparisons[-1]['time_s'], coarse-initial, fine-initial,
                         coarse-fine, coarse.copy(), fine.copy()))
    save(OUT/'temporal-summary.json', results)
    if maps:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(maps), 3, figsize=(10, 3.4*len(maps)),
                                 constrained_layout=True, squeeze=False)
        growth_scale = max(float(max(m[3].max(), m[4].max())) for m in maps)
        difference_scale = max(float(abs(m[5]).max()) for m in maps) or 1e-16
        for row, (mode, start, t, coarse_growth, half_growth, difference, coarse_phi, half_phi) in enumerate(maps):
            fields = [(coarse_growth, coarse_phi, 'Δt=.000625 s', 'magma', 0, growth_scale),
                      (half_growth, half_phi, 'Δt=.0003125 s', 'magma', 0, growth_scale),
                      (difference, half_phi, 'coarse − half-step', 'RdBu_r', -difference_scale, difference_scale)]
            for ax, (values, phi, title, cmap, lo, hi) in zip(axes[row], fields):
                im = ax.pcolormesh(detector.x, detector.y, values, shading='nearest', cmap=cmap, vmin=lo, vmax=hi)
                ax.contour(detector.x, detector.y, phi, levels=[.01,.05,.1,.2,.4,.6], colors='white', linewidths=.5)
                ax.axvline(detector.support_front, color='cyan', linewidth=.7, linestyle='--')
                ax.plot(1,.5,'c+',ms=5); ax.set_aspect('equal')
                ax.set_xlim(.95,1.25); ax.set_ylim(.37,.63)
                ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
                label = 'Radial tensor flow' if mode == 'radial' else 'Fixed-direction flow'
                ax.set_title(f'{label}: {title}\nt={t:.6f} s\nincoming t={start:.6f} s', fontsize=8)
                fig.colorbar(im,ax=ax,shrink=.8,label='Δφ')
        fig.suptitle('Same incoming histories; no reflection or branch alignment. White: phase contours.',fontsize=10)
        fig.savefig(OUT/'temporal-damage-comparison.png',dpi=170);plt.close(fig)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
