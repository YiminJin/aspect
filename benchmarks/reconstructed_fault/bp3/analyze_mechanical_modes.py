"""Reduce the frozen production-weighted probes, without launching ASPECT."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from run_mechanical_modes import parameters


def table(path):
    return np.genfromtxt(path, names=True, delimiter=',', dtype=None, encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--reference', type=Path, default=Path(__file__).parent/'first_long_run/output')
    args = parser.parse_args()
    run = args.run
    modes, nodes = table(run/'mechanical_modes.csv'), table(run/'mechanical_mode_nodes.csv')
    names = list(modes['mode'])
    by_mode = [nodes[nodes['mode'] == name] for name in names]
    assert all(np.array_equal(rows['node'], by_mode[0]['node']) for rows in by_mode)
    result = {'modal_coefficients_units': 'Pa/(m/s)', 'modes': [], 'cross_actions': {}}
    raw = np.concatenate([table(path) for path in sorted(run.glob('mechanical_mode_qp_rank*.csv'))])
    assert len({(r['mode'], r['cell'], int(r['qp'])) for r in raw}) == len(raw), 'Duplicate owned quadrature points'
    for row, data in zip(modes, by_mode):
        d = {key: row[key].item() for key in modes.dtype.names}
        mass_row = data['mass_row']
        # These are explicitly mass-row averages, not consistent nodal point values.
        window = (data['xd'] >= 15000.) & (data['xd'] <= 18000.)
        def rms(field):
            return float(np.sqrt(np.sum(field[window]**2/mass_row[window])/np.sum(mass_row[window])))
        d.update(weak_average_RMS_Pa={key: rms(data[key]) for key in (
            'delta_q', 'delta_p', 'minus_delta_tau_N', 'minus_mu_delta_p',
            'mu_delta_tau_N', 'minus_sigma_muV_deltaV', 'delta_R', 'delta_R_frozen_normal')})
        d['normal_feedback_weak_RMS_Pa'] = rms(data['delta_R']-data['delta_R_frozen_normal'])
        d['feedback_RMS_over_frozen_RMS'] = d['normal_feedback_weak_RMS_Pa']/rms(data['delta_R_frozen_normal'])
        d['normal_feedback_modal_fraction'] = (d['full_restoring']-d['frozen_normal_restoring'])/d['frozen_normal_restoring']
        d['mechanical_fraction_of_frozen'] = d['mechanical_shear']/d['frozen_normal_restoring']
        selected_raw = raw['mode'] == row['mode']
        sample = raw[selected_raw]
        mass_export = np.sum(sample['weight']*sample['deltaV']**2)
        md = data['mass_diagonal']*data['deltaV']
        md[:-1] += data['mass_upper'][:-1]*data['deltaV'][1:]
        md[1:] += data['mass_upper'][:-1]*data['deltaV'][:-1]
        assert abs(np.dot(data['deltaV'], md)/row['mass_norm']-1) < 1e-12
        projected_coordinate = ((1-sample['xi'])*data['xd'][sample['segment']]
                                + sample['xi']*data['xd'][sample['segment']+1])
        inside = (projected_coordinate >= 15000.) & (projected_coordinate <= 18000.)
        weight = sample['weight'][inside]
        def qp_stats(values):
            values = values[inside]
            return dict(weighted_mean=float(np.dot(weight, values)/weight.sum()),
                        weighted_RMS=float(np.sqrt(np.dot(weight, values**2)/weight.sum())),
                        minimum=float(values.min()), maximum=float(values.max()))
        d['raw_QP_change_statistics_Pa'] = {key: qp_stats(sample[key]) for key in (
            'delta_q', 'delta_p', 'minus_delta_tau_N', 'minus_sigma_muV_deltaV', 'delta_R', 'delta_R_frozen_normal')}
        d['raw_QP_change_statistics_Pa']['delta_sigma_n'] = qp_stats(sample['delta_p']+sample['minus_delta_tau_N'])
        d['baseline_QP_normal_Pa'] = qp_stats(sample['sigma'])
        frozen_bulk = np.dot(data['deltaV'], data['K_deltaV'])/row['mass_norm']
        direct_shear = frozen_bulk-d['instantaneous_friction']-d['damping']
        d['exported_fraction_of_modal_mass'] = float(mass_export/row['mass_norm'])
        d['direct_frozen_bulk_shear_restoring'] = float(direct_shear)
        d['frozen_bulk_total'] = float(frozen_bulk)
        d['bulk_relaxation_fraction_of_direct_shear'] = float(1-d['mechanical_shear']/direct_shear)
        result['modes'].append(d)
    # Off-diagonal modal actions expose normal feedback that may be invisible
    # to a same-mode energy quotient of a nonsymmetric operator.
    for field in ('delta_q', 'delta_R', 'delta_R_frozen_normal'):
        result['cross_actions'][field] = [
            [float(-np.dot(a['deltaV'], b[field])/np.sqrt(modes[i]['mass_norm']*modes[j]['mass_norm']))
             for j, b in enumerate(by_mode)] for i, a in enumerate(by_mode)]

    prefix = []
    for file in sorted((run/'profiles').glob('fault_*.csv'), key=lambda p: int(p.stem.split('_')[-1])):
        reference = args.reference/'profiles'/file.name
        if not reference.exists():
            continue
        actual, ref = table(file), table(reference)
        assert np.array_equal(actual['node'], ref['node'])
        assert np.max(np.abs(actual['xd_m']-ref['xd_m'])) < 1e-8
        p = dict(step=int(actual['step'][0]), time_s=float(actual['time_s'][0]),
                 time_difference_s=float(actual['time_s'][0]-ref['time_s'][0]))
        for key in ('V_m_per_s', 'Theta_s', 'slip_m', 'q_weak_Pa', 'sigma_n_weak_Pa'):
            difference = actual[key]-ref[key]
            p[key+'_max_absolute_difference'] = float(np.max(np.abs(difference)))
            p[key+'_relative_L2_difference'] = float(np.linalg.norm(difference)/max(np.linalg.norm(ref[key]), 1e-300))
        prefix.append(p)
    result['prefix_comparison'] = prefix
    control = table(args.reference/'profiles/fault_11.csv')
    assert np.array_equal(control['node'], by_mode[0]['node'])
    result['probe_base_V_relative_L2_difference_from_server_step11'] = float(
        np.linalg.norm(by_mode[0]['V']-control['V_m_per_s'])/np.linalg.norm(control['V_m_per_s']))
    patch = (control['xd_m'] >= 15000.) & (control['xd_m'] <= 18000.)
    prm = parameters((run/'parameters.prm').read_text())
    dc = float(prm[('Material model', 'Phase field fault', 'Characteristic slip distance')])
    aging_number = control['V_m_per_s'][patch]*float(modes['dt'][0])/dc
    result['nodal_aging_number_in_patch'] = dict(minimum=float(aging_number.min()), maximum=float(aging_number.max()),
                                               meaning='accepted V*dt/Dc; timing diagnostic, not a weak friction balance')
    (run/'mechanical_analysis.json').write_text(json.dumps(result, indent=2)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 3, figsize=(14, 11), sharex=True)
    for col, (name, data) in enumerate(zip(names, by_mode)):
        order = np.argsort(data['xd']); data = data[order]
        selected = (data['xd'] >= 14000.) & (data['xd'] <= 19000.)
        x = data['xd'][selected]/1000
        axes[0, col].plot(x, data['deltaV'][selected]/1e-12)
        axes[0, col].set_title(name)
        for key, label in [('delta_q', 'shear'), ('delta_p', 'pressure'), ('minus_delta_tau_N', '-deviatoric normal')]:
            axes[1, col].plot(x, (data[key]/data['mass_row'])[selected]/1000, label=label)
        for key, label in [('minus_mu_delta_p', 'pressure feedback'), ('mu_delta_tau_N', 'deviatoric feedback'), ('minus_sigma_muV_deltaV', 'instantaneous friction')]:
            axes[2, col].plot(x, (data[key]/data['mass_row'])[selected]/1000, label=label)
        for key, label in [('delta_R', 'full'), ('delta_R_frozen_normal', 'frozen normal')]:
            axes[3, col].plot(x, (data[key]/data['mass_row'])[selected]/1000, label=label)
        for ax in axes[:, col]:
            ax.grid(alpha=.25)
            for transition in (15., 18.): ax.axvline(transition, color='grey', lw=.7, ls=':')
        axes[3, col].set_xlabel('Down-dip distance (km)')
    axes[0, 0].set_ylabel('Velocity variation / 1e-12 m/s')
    for i in (1, 2, 3):
        axes[i, 0].set_ylabel('Mass-row weak average (kPa)')
        axes[i, 0].legend(fontsize=8)
    fig.suptitle('Frozen incoming state; bulk-relaxed probes with production work weights')
    fig.tight_layout();fig.savefig(run/'mechanical_modes.png', dpi=180);plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
