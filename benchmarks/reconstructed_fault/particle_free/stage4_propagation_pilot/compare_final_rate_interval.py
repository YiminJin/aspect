"""Reduce the matched 80/40 µs fourfold-rate window from physical step 79."""
import argparse
import json
import os
from pathlib import Path

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/aspect-propagation-pilot-mpl')

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np

from phase import H, NY, coefficients, read, value
from measure_propagation_fronts import connected_intact_cells


def groups(stations):
    """All contiguous positive 5 mm/s groups strictly ahead of x=6."""
    x, v = stations['x'], stations['rate_stage']
    fast = np.flatnonzero(v >= .005)
    return [g for g in np.split(fast, np.flatnonzero(np.diff(fast) > 1)+1)
            if len(g) and x[g[0]] > 6.]


def crossing(x, v, i, j):
    return x[i] + (.005-v[i])*(x[j]-x[i])/(v[j]-v[i])


def plot(path, stations, dt, fields, parent, support, fixed):
    x = stations[0]['x']
    visible = (x >= 5.80) & (x <= 6.24)
    rates = [s['rate_stage']*1000 for s in stations]
    labels = ['80 µs · 6.36 ms', '40 µs · 6.34 ms', '40 µs · 6.38 ms']
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    colors = ['#30363b', '#1e77a5', '#b63f30']
    for r, label, color in zip(rates, labels, colors):
        ax[0, 0].plot(x, r, label=label, color=color)
    ax[0, 0].axhline(5, ls=':', color='.4')
    ax[0, 0].axvline(6, ls=':', color='.4')
    ax[0, 0].set(xlim=(5.80, 6.24), ylim=(-.3, 1.1*max(r[visible].max() for r in rates)),
                 xlabel='Section x (m)', ylabel='Midpoint slip rate (mm/s)')
    ax[0, 0].legend(fontsize=8)
    slip = [rates[0]*dt[0]*1000, (rates[1]*dt[1]+rates[2]*dt[2])*1000]
    ax[0, 1].plot(x, slip[0], color=colors[0], linewidth=2.5, label='1 × 80 µs')
    ax[0, 1].plot(x, slip[1], color=colors[1], linestyle='--', label='2 × 40 µs')
    ax[0, 1].axvspan(x[fixed[0]], x[fixed[-1]], color='.85', label='Coarse secondary group')
    ax[0, 1].set(xlim=(5.80, 6.24), ylim=(-.025, 1.1*max(d[visible].max() for d in slip)),
                 xlabel='Section x (m)', ylabel='Slip added over 6.32–6.40 ms (µm)')
    ax[0, 1].legend(fontsize=8)
    xx, yy = np.meshgrid(np.linspace(5.94, 6.36, 380), np.linspace(1.82, 2.18, 300))
    delta = [value(f-parent, xx, yy) for f in fields]
    norm = PowerNorm(.5, vmin=0., vmax=max(d.max() for d in delta))
    for panel, d, f, caption in zip(ax[1], delta, fields, ['1 × 80 µs', '2 × 40 µs']):
        im = panel.pcolormesh(xx, yy, d, norm=norm, cmap='magma', shading='auto')
        panel.contour(xx, yy, support[(yy/H).astype(int), (xx/H).astype(int)],
                      levels=[.5], colors='cyan', linewidths=1)
        cs = panel.contour(xx, yy, value(f, xx, yy), levels=[.001,.01,.05], colors='white', linewidths=.7)
        panel.clabel(cs, fmt='%g', fontsize=7)
        panel.set(title=caption+' at 6.40 ms', aspect='equal', xlabel='x (m)', ylabel='y (m)')
    fig.colorbar(im, ax=ax[1].tolist(), label='Phase increment since 6.32 ms')
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('coarse', 'fine', 'initial', 'destination'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    out = args.destination
    out.mkdir(exist_ok=True, parents=True)
    seed, parent = coefficients(args.initial, 0), coefficients(args.coarse, 79)
    intact = np.max(abs(seed), axis=(2,3)) == 0
    parent_planes = read(args.coarse/'tip-planes-0079.csv')
    source = [(args.coarse, 80), (args.fine, 80), (args.fine, 81)]
    rates, dt, measures, stages, bands, registered, final = [], [], [], [], [], [], []
    fixed = None
    initial_tip = read(args.coarse/'pilot-sections-0079.csv')
    previous_planes = parent_planes
    for index, (folder, step) in enumerate(source):
        s = read(folder/f'pilot-sections-{step:04d}.csv')
        summary = read(folder/'pilot-summary.csv')
        row, = summary[summary['step'] == step]
        e = read(folder/'bulk-accepted.csv')
        energy, = e[e['step'] == step]
        assert energy['publications'] == step and energy['phase_KKT'] <= 1e-7 and energy['mechanical_residual'] <= 1e-9
        t, delta = row['time'], row['dt']
        assert abs(row['stage_time']-t+delta/2) < 1e-13
        assert abs(t-[.0064,.00636,.0064][index]) < 1e-13
        assert np.array_equal(s['x'], initial_tip['x'])
        x, v, ds = s['x'], s['rate_stage'], s['ds']
        active = groups(s)
        if index == 0:
            fixed, = active
        dt.append(delta)
        rates.append(s)
        dose = delta*(ds*v)
        measures.append(dose)
        assert abs(dose.sum()-row['integrated_rate']*delta) < 1e-16
        for g in active:
            i,j = g[0], g[-1]
            left, right = crossing(x,v,i-1,i), crossing(x,v,j,j+1)
            bands.append(dict(stage_ms=1000*row['stage_time'], first_station=x[i], last_station=x[j],
                              left_m=left, right_m=right, width_m=right-left,
                              left_bracket_m=[x[i-1],x[i]], right_bracket_m=[x[j],x[j+1]],
                              count=len(g), peak_mm_s=1000*v[g].max(), signed_slip_m2=dose[g].sum()))
        field = coefficients(folder, step)
        increment = field-parent
        planes = read(folder/f'tip-planes-{step:04d}.csv')
        prior = previous_planes if index == 2 else parent_planes
        assert np.array_equal(planes['parent_formed'], prior['formed'])
        old = prior['formed'] == 1
        assert (planes['formed'][old] == 1).all() and np.array_equal(planes['angle'][old],prior['angle'][old])
        previous_planes = planes
        y = ((np.arange(NY)[:, None]+(np.polynomial.legendre.leggauss(6)[0]+1)/2)*H).ravel()
        for k in np.flatnonzero((planes['formed'] == 1) & ~old):
            registered.append(dict(dt_us=round(delta*1e6), end_ms=t*1000, section=int(k), x=x[k],
                                   angle_deg=np.rad2deg(planes['angle'][k]), rate_m_s=v[k],
                                   max_native_phase=value(field,np.full(y.shape,x[k]),y).max()))
        stages.append(dict(dt_us=round(delta*1e6), stage_ms=1000*row['stage_time'], end_ms=1000*t,
                           right_groups=len(active), fixed_peak_mm_s=1000*v[fixed].max(), fixed_slip_m2=dose[fixed].sum(),
                           fast_slip_m2=sum(dose[g].sum() for g in active), total_slip_m2=dose.sum(),
                           tip_rate_mm_s=np.interp(6.,x,v)*1000, tip_age_ms=np.interp(6.,x,s['age_endpoint'])*1000,
                           fresh_increment_integral_m2=H**2*increment[intact].mean(axis=(1,2)).sum(),
                           max_fresh_increment=increment[intact].max(), max_fresh_coefficient=field[intact].max(),
                           min_B=field.min(), max_B=field.max(), new_since_parent=int(np.count_nonzero(planes['formed']>parent_planes['formed'])),
                           work_error=energy['Wext']-energy['energy_change']-energy['Db']-energy['Dc']-energy['Df'],
                           mechanics=energy['mechanical_residual'], kkt=energy['phase_KKT']))
        if index != 1:
            for level in (.001,.01):
                for points in (8,16):
                    stages[-1][f'fresh_connected_{level}_n{points}'] = connected_intact_cells(field,intact,level,points)
            final.append(field)
    for i in (0,2):
        section_delta = rates[i]['slip_endpoint']-initial_tip['slip_endpoint']
        expected = dt[0]*rates[0]['rate_stage'] if i == 0 else sum(t*s['rate_stage'] for t,s in zip(dt[1:],rates[1:]))
        np.testing.assert_allclose(section_delta,expected,rtol=1e-10,atol=1e-16)
    averaged = rates[1].copy()
    averaged['rate_stage'] = (dt[1]*rates[1]['rate_stage']+dt[2]*rates[2]['rate_stage'])/dt[0]
    mean_groups = [[float(x[g[0]]), float(x[g[-1]])] for g in groups(averaged)]
    result = dict(fine_mean_secondary_station_spans=mean_groups, stages=stages, bands=bands, registered=registered,
                  fixed_stations=[int(fixed[0]),int(fixed[-1])],
                  fixed_slip_m2=[measures[0][fixed].sum(),(measures[1]+measures[2])[fixed].sum()])
    (out/'secondary-group.json').write_text(json.dumps(result,indent=2)+'\n')
    plot(out/'last-interval.png', rates, dt, final, parent, intact, fixed)


if __name__ == '__main__':
    main()
