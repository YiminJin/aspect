"""Offline accepted-state BP3 junction evidence; no constitutive reevaluation."""
import argparse
from decimal import Decimal, localcontext
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_normal_stress import rows, write


def detail_profiles(root):
    """Resolve small projected changes separately from large raw extrema."""
    widths = []
    for step in (11, 12):
        folder = root/'analysis'/f'step{step}'
        profile = sorted(rows(folder/'projected_full.csv'), key=lambda r: float(r['xd']))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, (lo, hi) in zip(axes, [(13000, 20000), (37000, 43000)]):
            region = [r for r in profile if lo-1e-7 <= float(r['xd']) <= hi+1e-7]
            x = np.array([float(r['xd']) for r in region])
            for field, label, offset in [('delta_p', 'delta p', 0),
                                          ('minus_delta_tau_N', '-delta tau:N', 0),
                                          ('sigma_n', 'sigma_n - 50 MPa', 50e6)]:
                y = np.array([float(r[field]) for r in region])
                ax.plot(x/1000, (y-offset)/1e6, '.-', markersize=2, label=label)
                # Width of the largest detrended nodal feature. Linear endpoint
                # detrending is diagnostic only; original fields stay in CSV.
                a = abs(y-np.interp(x, [x[0], x[-1]], [y[0], y[-1]]))
                peak = int(a.argmax())
                threshold = .5*a[peak]
                left = right = peak
                while left > 0 and a[left] >= threshold:
                    left -= 1
                while right < len(x)-1 and a[right] >= threshold:
                    right += 1
                xl = x[left]+(x[left+1]-x[left])*(threshold-a[left])/(a[left+1]-a[left])
                xr = x[right-1]+(x[right]-x[right-1])*(threshold-a[right-1])/(a[right]-a[right-1])
                widths.append(dict(step=step, region=f'{lo}-{hi}', field=field,
                                   peak_xd=x[peak], detrended_peak=a[peak],
                                   half_amplitude_left=xl, half_amplitude_right=xr, width=xr-xl))
            ax.set(xlabel='down-dip distance [km]', ylabel='stress perturbation [MPa]')
            for mark in (15, 18, 40):
                if lo/1000 <= mark <= hi/1000:
                    ax.axvline(mark, color='grey', ls=':')
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
        fig.suptitle(f'Step {step}: consistent Q1 projection, not raw extrema')
        fig.tight_layout()
        fig.savefig(folder/'projected_stress_detail.png', dpi=150)
        plt.close(fig)
    write(root/'analysis'/'feature_widths.csv', widths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    out = root/'analysis'
    accepted = rows(root/'accepted_steps.csv')
    log = (root/'run.log').read_text()
    history = []
    theta_checks = []
    for a in accepted:
        step = int(a['step'])
        folder = out/f'step{step}'
        if not folder.exists():
            continue
        profile = rows(folder/'projected_full.csv')
        if step > 0:
            previous = rows(out/f'step{step-1}'/'projected_full.csv')
            with localcontext() as context:
                context.prec = 50
                dc = Decimal.from_float(.008)
                dt = Decimal.from_float(float(a['dt']))
                worst = None
                for i, r in enumerate(profile):
                    velocity = Decimal.from_float(float(r['V']))
                    old = Decimal.from_float(float(previous[i]['Theta_committed']))
                    actual = Decimal.from_float(float(r['Theta_committed']))
                    x = velocity*dt/dc
                    decay = (-x).exp()
                    reference = old*decay + dc/velocity*(1-decay)
                    relative = float(abs(actual/reference-1))
                    if worst is None or relative > worst['relative_error']:
                        worst = dict(step=step, node=i, xd=float(r['xd']), V=float(velocity),
                                     x=float(x), old_Theta=float(old), committed_Theta=float(actual),
                                     reference_50_digit=str(reference), relative_error=relative)
                assert worst['relative_error'] < 1e-12
                theta_checks.append(worst)
        summary = json.loads((folder/'summary.json').read_text())
        free = [r for r in profile if not int(r['prescribed'])]
        last = max(free, key=lambda r: float(r['xd']))
        deep = [r for r in profile if int(r['prescribed'])]
        first = min(deep, key=lambda r: float(r['xd']))
        tensile = [r for r in profile if float(r['tensile_weight']) > 0]
        entry = dict(step=step, time=float(a['time']),
                     free_xd=float(last['xd']), free_V=float(last['V']),
                     prescribed_xd=float(first['xd']), Vp=float(first['V']),
                     V_neighbor_difference=float(last['V'])-float(first['V']),
                     raw_min=float(summary['minimum']['sigma_n']),
                     raw_max=float(summary['maximum']['sigma_n']),
                     raw_min_xd=float(summary['minimum']['xd']),
                     tensile_weight=summary['all']['tensile_weight'],
                     free_tensile_weight=summary['free']['tensile_weight'],
                     free_tensile_weak_ratio=summary['free']['tensile_to_total_weak_friction_l2'],
                     tensile_node_xd_min=min((float(r['xd']) for r in tensile), default=None),
                     tensile_node_xd_max=max((float(r['xd']) for r in tensile), default=None))
        for label, bounds in [('transition', (13000, 20000)), ('junction', (37000, 43000))]:
            region = [r for r in profile if bounds[0]-1e-7 <= float(r['xd']) <= bounds[1]+1e-7]
            for field in ['delta_p', 'minus_delta_tau_N', 'sigma_n', 'q']:
                values = np.array([float(r[field]) for r in region])
                entry[label+'_'+field+'_ptp'] = float(np.ptp(values))
                entry[label+'_'+field+'_min'] = float(values.min())
                entry[label+'_'+field+'_max'] = float(values.max())
            # A diagnostic width, not a fitted singularity exponent: the span
            # above half of the largest absolute nodal departure from 50 MPa.
            amplitude = np.array([abs(float(r['sigma_n'])-50e6) for r in region])
            xd = np.array([float(r['xd']) for r in region])
            selected = xd[amplitude >= .5*amplitude.max()]
            entry[label+'_sigma_half_amplitude_span'] = float(np.ptp(selected))
        history.append(entry)

        if step not in (11, 12):
            continue
        write(folder/'profile_10_45km.csv', [r for r in profile if 1e4-1e-7 <= float(r['xd']) <= 4.5e4+1e-7])
        raw = rows(folder/'raw_selected.csv')
        support_extrema = {}
        for code, name in [('0', 'unprescribed'), ('1', 'prescribed'), ('2', 'mixed')]:
            selected = [r for r in raw if r['support'] == code]
            if selected:
                support_extrema[name] = {
                    'minimum': min(selected, key=lambda r: float(r['sigma_n'])),
                    'maximum': max(selected, key=lambda r: float(r['sigma_n']))}
        bulk = rows(folder/'bulk_transfer.csv')
        deep_bulk = [r for r in bulk if min(float(r['xd0']), float(r['xd1'])) >= 40000-1e-7]
        ratio = sum(float(r['instantaneous_integral']) for r in deep_bulk)/sum(float(r['chi_integral']) for r in deep_bulk)
        transfer = dict(deep_localization_weighted_V=ratio,
                        max_absolute_history_integral=max(abs(float(r['history_integral'])) for r in bulk),
                        max_crack_identity_error=max(abs(float(r['total_integral'])-float(r['instantaneous_integral'])
                                                         -float(r['history_integral'])) for r in bulk),
                        bottom_segment=max(bulk, key=lambda r: max(float(r['xd0']), float(r['xd1']))))
        (folder/'support_and_transfer.json').write_text(json.dumps(
            dict(support_extrema=support_extrema, transfer=transfer), indent=2)+'\n')
        previous = rows(out/f'step{step-1}'/'projected_full.csv')
        for r in raw:
            i, xi = int(r['segment']), float(r['xi'])
            r['Theta_used_by_mechanics'] = ((1-xi)*float(previous[i]['Theta_committed'])
                                            + xi*float(previous[i+1]['Theta_committed']))
            r['Theta_committed_Q1_at_sample'] = ((1-xi)*float(profile[i]['Theta_committed'])
                                                 + xi*float(profile[i+1]['Theta_committed']))
        write(folder/'raw_selected_with_theta.csv', raw)
        fig, axes = plt.subplots(3, 3, figsize=(16, 10))
        xd = np.array([float(r['xd'])/1000 for r in profile])
        for col, limits in enumerate([(13, 20), (37, 43), (113, 115.5)]):
            for f, label in [('delta_p', 'delta p'), ('minus_delta_tau_N', '-delta tau:N'),
                             ('sigma_n', 'total sigma_n')]:
                axes[0, col].plot(xd, np.array([float(r[f]) for r in profile])/1e6, label=label)
            selected = [r for r in raw if limits[0] <= float(r['xd'])/1000 <= limits[1]]
            axes[0, col].scatter([float(r['xd'])/1000 for r in selected],
                                 [float(r['sigma_n'])/1e6 for r in selected],
                                 s=8, c='black', label='selected raw sigma_n')
            axes[1, col].semilogy(xd, [float(r['V']) for r in profile])
            axes[2, col].plot(xd, np.array([float(r['q']) for r in profile])/1e6)
            for ax in axes[:, col]:
                ax.set_xlim(*limits)
                for mark in (15, 18, 40):
                    if limits[0] <= mark <= limits[1]:
                        ax.axvline(mark, color='grey', ls=':')
                ax.grid(alpha=.25)
            axes[0, col].legend(fontsize=8)
            axes[2, col].set_xlabel('down-dip distance [km]')
        axes[0, 0].set_ylabel('stress [MPa]')
        axes[1, 0].set_ylabel('V [m/s]')
        axes[2, 0].set_ylabel('total shear q [MPa]')
        fig.suptitle(f'Accepted step {step}: consistent Q1 profiles and selected raw extrema')
        fig.tight_layout()
        fig.savefig(folder/'junction_profiles.png', dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 4))
        sc = ax.scatter([float(r['xd'])/1000 for r in raw],
                        [float(r['signed_normal_distance']) for r in raw],
                        c=[float(r['sigma_n'])/1e6 for r in raw], cmap='coolwarm', s=10)
        ax.axvline(40, color='black', ls=':')
        ax.set(xlabel='surface down-dip coordinate [km]', ylabel='parent normal offset [m]',
               title=f'Step {step}: selected extrema, not a full band sampling')
        fig.colorbar(sc, ax=ax, label='sigma_n [MPa]')
        fig.tight_layout()
        fig.savefig(folder/'raw_locations.png', dpi=150)
        plt.close(fig)
    write(out/'junction_history.csv', history)
    write(out/'theta_independent.csv', theta_checks)
    linear = []
    nonlinear = {}
    step = None
    for line in log.splitlines():
        match = re.search(r'\*\*\* Timestep (\d+):', line)
        if match:
            step = int(match[1])
        if 'Fault linear solve:' in line:
            data = {k.strip(): float(v) for k, v in re.findall(r'([a-zA-Z_ ]+)=([-+\deE.]+)', line)}
            linear.append(dict(step=step, **data))
        if 'Fault nonlinear residual:' in line:
            nonlinear[step] = {k.strip(): float(v) for k, v in re.findall(r'([a-zA-Z_ ]+)=([-+\deE.]+)', line)}
    checks = dict(linear_checks=len(linear),
                  fresh_failures=sum(r['fresh'] > r['target'] for r in linear),
                  max_fresh_target_ratio=max(r['fresh']/r['target'] for r in linear),
                  final_nonlinear=nonlinear)
    assert checks['fresh_failures'] == 0, 'A returned direction failed the fresh residual check.'
    for a in accepted:
        r = nonlinear[int(a['step'])]
        assert r['bulk'] < r['bulk target'] and r['surface']/r['surface scale'] < 1e-8
    (out/'convergence.json').write_text(json.dumps(checks, indent=2)+'\n')
    write(out/'linear_checks.csv', linear)
    detail_profiles(root)
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
