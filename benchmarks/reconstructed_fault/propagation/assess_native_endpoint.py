#!/usr/bin/env python3
"""Unaligned same-spatial-model temporal gate; no trajectory transformation."""
import numpy as np
import native_endpoint as native
import asymmetric_branch as old
from assess_coupled_branch import totals

E = native.E
STRESS = 1332.521975
STORAGE = .887807407


def assess():
    paths = [E / f'dt-1-{d}' for d in [1024, 2048]]
    summaries = [old.read(p / 'summary.json') for p in paths]
    assert all(s['completed'] and s['last_accepted'] == .625 for s in summaries)
    f, fine = [dict(np.load(p / 'fields.npz')) for p in paths]
    h, hf = [old.read(p / 'history.json') for p in paths]
    init, initf = [dict(np.load(p / 'initialization.npz')) for p in paths]
    for key in init:
        assert np.array_equal(init[key], initf[key]), key
    dt = 1 / 1024
    r = native.SharedFE().r
    z, w = np.polynomial.legendre.leggauss(24)
    s = (z + 1) / 2
    weight = np.tile(w / (2 * r.n), r.n)
    def stress(fields, index, integral):
        phase = fields['phase'][index]
        phi = (1-s)*phase[:-1, None] + s*phase[1:, None]
        R = fields['R'][index]
        derivative = r.n*((-3+4*s)*R[:-2:2, None] +
                            (4-8*s)*R[1::2, None] + (-1+4*s)*R[2::2, None])
        chi = old.core.h(phi).ravel() / integral
        return old.G*derivative.ravel() - fields['Z'][index]*chi, chi
    peak = max(row['V'] for row in hf)
    slip_scale = hf[-1]['slip']
    rows = []
    for j, (a, b) in enumerate(zip(h, hf[1::2]), 1):
        assert a['time'] == b['time']
        ta, _ = stress(f, j, a['I'])
        tb, chi = stress(fine, 2*j, b['I'])
        delta = ta-tb
        slip_previous = hf[2*j-3]['slip'] if j > 1 else 0.
        rate = (b['slip']-slip_previous) / dt
        rows.append(dict(time=a['time'],
            phase=float(max(abs(f['phase'][j]-fine['phase'][2*j]))),
            q=abs(a['q']-b['q']), slip_fraction=abs(a['slip']-b['slip'])/slip_scale,
            interval_rate_fraction=abs(a['V']-rate)/peak,
            whole_stress_fraction=float(np.sqrt(weight@(delta*delta)))/STRESS,
            localized_stress_fraction=float(np.sqrt((weight*chi)@(delta*delta)/(weight@chi)))/STRESS,
            bulk_storage_fraction=abs(a['bulk']-b['bulk'])/STORAGE,
            centroid_difference=a['centroid']['phase']-b['centroid']['phase']))
    targets = dict(phase=.001, q=1., slip_fraction=.001, interval_rate_fraction=.005,
                   whole_stress_fraction=.01, localized_stress_fraction=.01,
                   bulk_storage_fraction=.01)
    maximum = {k:max(row[k] for row in rows) for k in targets}
    worst = {k:max(rows, key=lambda row:row[k])['time'] for k in targets}
    controls = []
    for path, fields, hist, summary in zip(paths, [f, fine], [h, hf], summaries):
        growth = np.max(fields['phase']-fields['phase'][0], axis=1)
        hit = np.flatnonzero(growth > 1e-6)
        step = summary['dt']
        beta = np.exp(-old.G*step/old.ETA)
        kappa = -old.ETA*np.expm1(-old.G*step/old.ETA)
        recurrence = float(max(abs((fields['R'][1:]-beta*fields['R'][:-1]-
                                   kappa/old.G*fields['velocity']).ravel())))
        assert recurrence < 1e-16
        assert max(row['KKT'] for row in hist) <= old.TARGET
        assert max(row['force'] for row in hist) < 1e-7
        assert max(row['contact'] for row in hist) < 1e-7
        assert min(np.diff(fields['phase'], axis=0).ravel()) >= -4.922e-12
        controls.append(dict(summary=summary, onset_bracket=None if not len(hit) else
            [(int(hit[0])-1)*step, int(hit[0])*step],
            maximum_KKT=max(row['KKT'] for row in hist),
            maximum_force=max(row['force'] for row in hist),
            maximum_contact=max(row['contact'] for row in hist), recurrence=recurrence,
            maximum_phase_increment=max(row['phase_increment'] for row in hist),
            minimum_healing=float(min(np.diff(fields['phase'], axis=0).ravel())),
            maximum_centroid_excursion=max(abs(row['centroid']['phase']-.5) for row in hist),
            energy=totals(old.read(path/'energy.json'))))
    answer = dict(initial_fields_identical=True, global_reflection_applied=False,
                  targets=targets, maximum=maximum, worst_time=worst,
                  passes={k:maximum[k] <= targets[k] for k in targets}, controls=controls,
                  stress_scale_Pa=STRESS, storage_scale_J_per_m=STORAGE,
                  slip_scale_m=slip_scale, peak_rate_scale_m_per_s=peak, by_time=rows)
    old.write(E/'temporal-comparison.json', answer)
    print(old.json.dumps({k:answer[k] for k in ['maximum', 'passes', 'worst_time']}))
    plot(f, fine, h, hf, rows)


def plot(f, fine, h, hf, errors):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for fields, hist, label, denominator in [(f,h,'1/1024 s',1024), (fine,hf,'1/2048 s',2048)]:
        t = np.r_[0., [row['time'] for row in hist]]
        slip = np.r_[0., [row['slip'] for row in hist]]
        q = np.r_[0., [row['q'] for row in hist]]
        axes[0,0].plot(t, fields['phase'][:,256], label=label)
        axes[0,1].plot(slip, q, label=label)
        axes[0,2].plot(t[1:], [1000*(row['centroid']['phase']-.5) for row in hist], label=label)
        for time in [.53125, .59375, .625]:
            axes[1,0].plot(fields['y'], fields['phase'][round(time*denominator)],
                           label=f'{label}, t={time:g}')
        ledger = old.read(E/f'dt-1-{denominator}/energy.json')
        for key, style in [('mechanical_power','-'), ('friction_residual_power','--'), ('phase_power',':')]:
            axes[1,2].plot(t[1:], np.cumsum([row[key] for row in ledger]),
                           style, label=f'{label}: {key}')
    axes[1,1].plot([r['time'] for r in errors], [r['phase'] for r in errors])
    axes[1,1].axhline(.001, color='black', ls='--', label='temporal target')
    for ax, title, xlabel, ylabel in zip(axes.flat,
        ['Core phase', 'Traction–slip', 'Unaligned centroid', 'Physical profiles',
         'Full-field halving difference', 'Separate signed residual works'],
        ['t (s)','slip (m)','t (s)','y (m)','t (s)','t (s)'],
        ['phase','q (Pa)','centroid shift (mm)','phase','max phase difference','work (J/m)']):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.savefig(E/'temporal-comparison.png', dpi=160)
    fig.savefig(E/'temporal-comparison.pdf')
    plt.close(fig)


if __name__ == '__main__':
    assess()
