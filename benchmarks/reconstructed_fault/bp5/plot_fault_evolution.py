"""Plot accepted BP5 station histories, full-fault kinematics, and solver progress.

Tractions in stations.csv are interpolated consistent-Q1 weak projections, not
raw constitutive samples. State is committed output; mechanics k used state
k-1. Nodal V at real steps is recovered from the production slip += dt*V record.
"""
import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bp3'))
from plot_cumulative_slip import read_profiles


def table(path):
    return np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True))


def save(fig, output, name, note):
    fig.text(0.02, 0.015, note, fontsize=8, va='bottom')
    fig.tight_layout(rect=(0, 0.075, 1, 0.96))
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'{name}.{extension}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.run / 'plots'
    output.mkdir(parents=True, exist_ok=True)
    accepted = table(args.run / 'accepted_steps.csv')
    stations = table(args.run / 'stations.csv')
    if not np.array_equal(accepted['step'], np.arange(len(accepted))):
        raise ValueError('Expected a consecutive accepted trajectory from initialization')
    positions = np.unique(stations['xd'])
    series = {}
    for xd in positions:
        rows = np.sort(stations[stations['xd'] == xd], order='step')
        if (not np.array_equal(rows['step'], accepted['step'])
                or not np.array_equal(rows['time'], accepted['time'])):
            raise ValueError('Station and accepted-state timelines differ')
        series[xd] = rows
    days = accepted['time'] / 86400
    vp, vmin, dc = 1e-9, 1e-20, 0.1
    # Read the actual configured constants; refuse silent reuse for another model.
    parameters = (args.run / 'parameters.prm').read_text()
    for name, expected in [('Minimum slip rate', vmin), ('Characteristic slip distance', dc)]:
        values = re.findall(r'^\s*set ' + name + r'\s*=\s*([^#\n]+)', parameters, re.M)
        if len(values) != 1 or float(values[0]) != expected:
            raise ValueError(f'This BP5 plot expects {name} = {expected}')

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    shown = [x for x in (0, 10000, 20000, 25000, 30000, 35000) if x in series]
    colors = plt.get_cmap('tab10').colors
    for color, xd in zip(colors, shown):
        r = series[xd]
        incoming = np.r_[r['Theta'][0], r['Theta'][:-1]]
        values = [r['V'] / vp, r['Theta'], r['slip'] * 1e6,
                  (r['tau_total'] - r['tau_total'][0]) / 1000,
                  (r['sigma_n_total'] - r['sigma_n_total'][0]) / 1000,
                  r['V'] * incoming / dc]
        for ax, value in zip(axes.flat, values):
            ax.plot(days, value, color=color, lw=1.3, label=f'{xd / 1000:g} km')
    labels = ['Slip rate / plate rate', 'Committed state Θ (s)', 'Accumulated slip (µm)',
              'Weak shear change from initialization (kPa)',
              'Weak compression change from initialization (kPa)', 'V × incoming Θ / Dc']
    for ax, label in zip(axes.flat, labels):
        ax.set(xlabel='Physical time (days)', ylabel=label, xlim=(0, days[-1]))
        ax.grid(alpha=0.2)
    for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 2]):
        ax.set_yscale('log')
    axes[0, 0].axhline(vmin / vp, color='black', ls='--', lw=0.8)
    axes[0, 0].text(0.03, 0.05, 'Dashed: Vmin / Vp = 10⁻¹¹', transform=axes[0, 0].transAxes, fontsize=8)
    axes[0, 1].legend(fontsize=8, ncol=2)
    axes[0, 2].set_title('Log scale; zero initial slip omitted')
    fig.suptitle(f'BP5 fault station evolution — accepted steps 0–{int(accepted["step"][-1])}')
    save(fig, output, 'fault_station_evolution',
         'Tractions: consistent-Q1 projections of current production weak loads, sampled at stations; not raw point stress.\n'
         'Mechanics uses preceding committed Θ; the state panel shows newly committed Θ. Positive normal traction denotes compression.')

    # Stream the large file and verify recovered nodal velocities against the
    # independently saved station rates and global maximum before plotting them.
    selected = sorted(set([0, len(accepted) - 1] +
                          [int(np.argmin(abs(days - t))) for t in (.1, .5, 1, 2, 4, 6)]))
    initial = table(args.run / 'profiles' / 'fault_0.csv')
    snapshots = []
    previous = None
    maximum_station_relative_error = 0.
    maximum_peak_relative_error = 0.
    for profile in read_profiles(args.run / 'cumulative_slip.csv'):
        k = profile.step
        if k >= len(accepted) or profile.time != accepted['time'][k]:
            raise ValueError('Slip and accepted-state timelines differ')
        order = np.argsort(profile.xd)
        xd = profile.xd[order]
        if previous is None:
            initial = np.sort(initial, order='node')
            if not np.array_equal(initial['xd_m'], profile.xd):
                raise ValueError('Initial profile and cumulative-slip geometry differ')
            velocity = initial['V_m_per_s'][order]
        else:
            velocity = ((profile.slip - previous.slip) / (profile.time - previous.time))[order]
        if np.any(velocity <= 0):
            raise ValueError('Nonpositive recovered slip rate')
        for station, rows in series.items():
            measured = rows['V'][k]
            reconstructed = np.interp(station, xd, velocity)
            maximum_station_relative_error = max(maximum_station_relative_error,
                                                 abs(reconstructed / measured - 1))
        maximum_peak_relative_error = max(maximum_peak_relative_error,
                                          abs(velocity.max() / accepted['max_V'][k] - 1))
        if k in selected:
            snapshots.append((k, xd.copy(), velocity.copy(), profile.slip[order].copy()))
        previous = profile
    if previous is None or previous.step != len(accepted) - 1:
        raise ValueError('Cumulative-slip file does not reach the final accepted state')
    # Cancellation at ~1e-20 m/s loses digits in differences of cumulative slip.
    if maximum_station_relative_error > 1e-4 or maximum_peak_relative_error > 1e-9:
        raise ValueError('Recovered velocity fails independent output cross-check')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for color, (k, xd, velocity, slip) in zip(plt.get_cmap('viridis')(np.linspace(0, 1, len(snapshots))), snapshots):
        for column in range(2):
            axes[0, column].plot(xd / 1000, velocity / vp, color=color, label=f'{days[k]:.3f} d')
            axes[1, column].plot(xd / 1000, slip * 1000, color=color)
    for j, hi in enumerate((35, snapshots[-1][1][-1] / 1000)):
        axes[0, j].set_yscale('log')
        axes[0, j].axhline(vmin / vp, color='black', ls='--', lw=.8)
        axes[0, j].set_ylabel('V / Vp (log scale)')
        axes[1, j].set_ylabel('Accumulated slip (mm)')
        for ax in axes[:, j]:
            ax.set(xlim=(0, hi), xlabel='Down-dip distance (km)')
            for boundary in (30, 33):
                ax.axvline(boundary, color='0.5', ls=':', lw=.8)
            ax.grid(alpha=.2)
    axes[0, 0].legend(ncol=2, fontsize=8)
    axes[0, 0].set_title('Shallow fault; dotted lines at 30 and 33 km')
    axes[0, 1].set_title('Entire fault')
    fig.suptitle('Full-fault kinematic evolution — recorded nodal data')
    save(fig, output, 'fault_kinematic_evolution',
         'Real-step V recovered from Δslip/Δt and checked against saved station rates and global maxima.\n'
         'Initialization uses directly saved V. No spatial smoothing; connected lines show the Q1 nodal representation.')

    log = (args.run / 'log.txt').read_text()
    match = list(re.finditer(r'\*\*\* Timestep (\d+):\s+t=([\d.eE+-]+)', log))[-1]
    last_attempt = int(match[1])
    failed = log[match.start():]
    residuals = np.array(re.findall(r'after nonlinear iteration\s+(\d+):\s+([\d.eE+-]+),\s+([\d.eE+-]+)', failed), float)
    alpha = np.array(re.findall(
        r'line search accepted after (\d+) rejected candidates; alpha=(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\.',
        failed), float)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes[0, 0].semilogy(days, accepted['min_free_V'] / vmin)
    axes[0, 0].axhline(1, color='black', ls='--', lw=.8)
    axes[0, 0].set_ylabel('Minimum free V / Vmin')
    axes[0, 1].plot(days[1:], accepted['dt'][1:])
    axes[0, 1].set_ylabel('Accepted physical dt (s)')
    axes[0, 2].semilogy(days, accepted['surface_RMS_Pa'])
    axes[0, 2].set_ylabel('Accepted surface residual RMS (Pa)')
    axes[1, 0].step(days, accepted['lower_active'], where='post')
    axes[1, 0].set(ylabel='Accepted lower-active nodes', ylim=(-.5, max(1, accepted['lower_active'].max() + .5)))
    for ax in (*axes[0], axes[1, 0]):
        ax.set(xlabel='Physical time (days)', xlim=(0, days[-1]))
    if last_attempt <= int(accepted['step'][-1]) or 'failed to converge' not in failed:
        raise ValueError('Expected a separate final failed solve in the supplied log')
    axes[1, 1].semilogy(residuals[:, 0], residuals[:, 1], label='Bulk')
    axes[1, 1].semilogy(residuals[:, 0], residuals[:, 2], label='Surface')
    axes[1, 1].set(xlabel='Newton iteration', ylabel='Relative residual', title=f'FAILED step {last_attempt}; not an accepted state')
    axes[1, 1].legend(fontsize=8)
    axes[1, 2].semilogy(np.arange(len(alpha)), alpha[:, 1], 'o-', ms=3)
    axes[1, 2].set(xlabel='Newton update', ylabel='Accepted line-search α', title=f'Failed solve: {int(alpha[:, 0].sum())} rejected trials')
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle('Approach to the slip-rate bound and nonlinear termination')
    save(fig, output, 'fault_solver_evolution',
         'Physical histories end at the last converged state. Bottom-right panels describe only the unsuccessful next solve.\n'
         'A line-search-accepted Newton iterate is not a converged physical state.')
    summary = dict(accepted_states=len(accepted), final_step=int(accepted['step'][-1]),
                   final_time_days=float(days[-1]), station_count=len(positions),
                   fault_nodes=len(previous.nodes), selected_profile_steps=selected,
                   maximum_recovered_station_V_relative_error=maximum_station_relative_error,
                   maximum_recovered_global_max_V_relative_error=maximum_peak_relative_error,
                   failed_step=last_attempt,
                   traction_definition='Consistent Q1 projection of current production weak traction, interpolated at station',
                   state_definition='Theta output is committed; mechanics uses preceding accepted Theta')
    (output / 'fault_evolution_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
