"""Reuse verified uniform-Maxwell bulk responses with actual step-11 QP friction.

This is an offline linear-response calculation, not a new accepted trajectory.
Geometry/weight coverage and the independently captured target broad response
must pass before the other two responses may be reused.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from analyze_mechanical_modes import table
from run_mechanical_modes import parameters


def raw_rows(directory, allow_truncated_last=False):
    rows, truncated, paths = {}, 0, sorted(directory.glob('mechanical_mode_qp_rank*.csv'))
    for path in paths:
        if not path.stat().st_size:
            continue
        with path.open() as stream:
            reader = csv.DictReader(stream)
            entries = list(reader)
            for i, row in enumerate(entries):
                if None in row or any(value is None for value in row.values()):
                    assert allow_truncated_last and i == len(entries)-1
                    truncated += 1
                    continue
                key = row['mode'], row['cell'], int(row['qp'])
                assert key not in rows, 'Duplicate owned quadrature point'
                rows[key] = {k: v if k in ('mode', 'cell') else float(v)
                             for k, v in row.items()}
    return rows, truncated, paths


def relative(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-300))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert 'MECHANICAL MODES VERIFIED' in (args.initial/'probe.log').read_text()
    initial, _, files = raw_rows(args.initial)
    target, truncated, target_files = raw_rows(args.target, True)
    modes = table(args.initial/'mechanical_modes.csv')
    nodes = table(args.initial/'mechanical_mode_nodes.csv')
    control = table(args.control/'profiles/fault_11.csv')
    names = list(modes['mode'])
    native = [nodes[nodes['mode'] == name] for name in names]
    assert np.array_equal(native[0]['node'], control['node'])
    assert np.max(np.abs(native[0]['xd']-control['xd_m'])) < 1e-8
    prm = parameters((args.target/'parameters.prm').read_text())
    for key in ('Reference viscosities', 'Elastic shear moduli'):
        assert ',' not in prm['Material model', 'Phase field fault', key], 'Reuse requires uniform bulk coefficients'
    damping = float(prm['Material model', 'Phase field fault', 'Radiation damping coefficients'])
    key_list = sorted((cell, q) for mode, cell, q in target if mode == names[0])
    base = {key: np.array([target[names[0], cell, q][key] for cell, q in key_list])
            for key in next(iter(target.values())) if key not in ('mode', 'cell')}
    segment, xi, weight = base['segment'].astype(int), base['xi'], base['weight']
    initial_broad = {key: np.array([initial[names[0], cell, q][key] for cell, q in key_list])
                     for key in base}
    ratio = base['kappa']/initial_broad['kappa']
    assert np.ptp(ratio)/ratio.mean() < 1e-14
    required = {key[1:] for key, row in initial.items()
                if key[0] == names[0] and row['deltaV'] != 0.}
    assert required <= set(key_list), 'Missing a QP on the modal support'
    geometry_checks = {key: relative(base[key], initial_broad[key])
                       for key in ('xi', 'x', 'y', 'weight', 'JxW', 'chi', 'deltaV')}
    assert max(geometry_checks.values()) < 1e-13
    scaling_checks = {key: relative(base[key], initial_broad[key]*ratio)
                      for key in ('delta_q', 'delta_p', 'minus_delta_tau_N')}
    assert max(scaling_checks.values()) < 1e-10
    n = len(native[0])
    window = (native[0]['xd'] >= 15000.-1e-7) & (native[0]['xd'] <= 18000.+1e-7)

    def weak(values):
        result = np.zeros(n)
        np.add.at(result, segment, weight*(1-xi)*values)
        np.add.at(result, segment+1, weight*xi*values)
        return result

    # Require full support for every reported weak row, not merely global
    # modal contractions. Missing off-window records are not filled as data.
    mass_row = native[0]['mass_row']
    row_coverage = float(np.max(np.abs(weak(np.ones_like(weight))[window]/mass_row[window]-1)))
    assert row_coverage < 1e-12
    trial_v = (1-xi)*control['V_m_per_s'][segment]+xi*control['V_m_per_s'][segment+1]
    velocity_check = relative(base['V'], trial_v)
    assert velocity_check < 1e-6, 'Captured base differs materially from accepted control'
    nonzero = base['deltaV'] != 0.
    results, loads, output_rows = [], [], []
    for mode, data in zip(modes, native):
        sample = {key: np.array([initial[mode['mode'], cell, q][key] for cell, q in key_list])
                  for key in ('deltaV', 'delta_q', 'delta_p', 'minus_delta_tau_N')}
        dv = sample['deltaV']
        assert np.all(dv[~nonzero] == 0.)
        change = {key: sample[key]*ratio for key in ('delta_q', 'delta_p', 'minus_delta_tau_N')}
        change['minus_mu_delta_p'] = -base['mu']*change['delta_p']
        change['mu_delta_tau_N'] = -base['mu']*change['minus_delta_tau_N']
        # The production derivative includes the actual incoming Theta, V and
        # total normal traction. Never substitute initialization friction here.
        friction = np.zeros_like(dv)
        friction[nonzero] = base['minus_sigma_muV_deltaV'][nonzero]*(dv[nonzero]/base['deltaV'][nonzero])
        change['instantaneous_friction'] = friction
        change['damping'] = -damping*dv
        change['frozen_normal'] = change['delta_q']+friction-damping*dv
        change['full'] = change['frozen_normal']+change['minus_mu_delta_p']+change['mu_delta_tau_N']
        mass = float(np.dot(weight, dv*dv))
        assert abs(mass/mode['mass_norm']-1) < 1e-12
        load = {key: weak(value) for key, value in change.items()}
        coefficients = {key: float(-np.dot(weight*dv, value)/mass) for key, value in change.items()}
        def rms(value):
            return float(np.sqrt(np.sum(value[window]**2/mass_row[window])/mass_row[window].sum()))
        projected_x = (1-xi)*data['xd'][segment]+xi*data['xd'][segment+1]
        inside = (projected_x >= 15000.) & (projected_x <= 18000.)
        def qp_stats(value):
            w, v = weight[inside], value[inside]
            return dict(mean=float(np.dot(w, v)/w.sum()),
                        RMS=float(np.sqrt(np.dot(w, v*v)/w.sum())),
                        minimum=float(v.min()), maximum=float(v.max()))
        raw_statistics = {key: qp_stats(value) for key, value in change.items()}
        raw_statistics['delta_sigma_n'] = qp_stats(change['delta_p']+change['minus_delta_tau_N'])
        direct = (float(np.dot(data['deltaV'], data['K_deltaV'])/mode['mass_norm'])
                  -mode['instantaneous_friction']-mode['damping'])*float(ratio.mean())
        results.append(dict(mode=mode['mode'], mass_norm=mass, coefficients=coefficients,
                            weak_average_RMS_Pa={key: rms(value) for key, value in load.items()},
                            normal_feedback_weak_RMS_Pa=rms(load['full']-load['frozen_normal']),
                            normal_feedback_over_frozen_RMS=rms(load['full']-load['frozen_normal'])/rms(load['frozen_normal']),
                            normal_feedback_modal_fraction=(coefficients['full']/coefficients['frozen_normal']-1),
                            direct_frozen_bulk_shear=direct,
                            mechanical_remaining_fraction=coefficients['delta_q']/direct,
                            raw_QP_change_Pa=raw_statistics))
        # The nearly locked shallow part has a large friction derivative.
        # Report the actual onset neighbourhood separately so it cannot mask
        # a locally important normal-feedback response in a whole-patch norm.
        regional = {}
        for lo, hi in ((16000., 18000.), (16500., 17500.)):
            selected = (data['xd'] >= lo-1e-7) & (data['xd'] <= hi+1e-7)
            def regional_rms(value):
                return float(np.sqrt(np.sum(value[selected]**2/mass_row[selected])/mass_row[selected].sum()))
            regional[f'{lo:g}-{hi:g}_m'] = {key: regional_rms(load[key]) for key in
                                          ('delta_q', 'delta_p', 'minus_delta_tau_N', 'instantaneous_friction', 'full', 'frozen_normal')}
            regional[f'{lo:g}-{hi:g}_m']['normal_feedback'] = regional_rms(load['full']-load['frozen_normal'])
        results[-1]['onset_region_weak_average_RMS_Pa'] = regional
        if mode['mode'] == names[0]:
            for key, source in (('full', 'delta_R'), ('frozen_normal', 'delta_R_frozen_normal')):
                scaling_checks[key] = relative(change[key], base[source])
                assert scaling_checks[key] < 1e-10
        loads.append(load)
        for i in np.flatnonzero(window):
            output_rows.append(dict(mode=mode['mode'], node=int(data['node'][i]), xd_m=float(data['xd'][i]),
                                    deltaV_m_per_s=float(data['deltaV'][i]),
                                    **{key+'_weak_average_Pa': float(value[i]/mass_row[i]) for key, value in load.items()}))
    cross = {key: [[float(-np.dot(a['deltaV'], b[key])/np.sqrt(modes[i]['mass_norm']*modes[j]['mass_norm']))
                    for j, b in enumerate(loads)] for i, a in enumerate(native)]
             for key in ('delta_q', 'full', 'frozen_normal')}
    # The initial outputs hold the native solve/action/finite-difference checks;
    # do not relabel these as three fresh step-11 solves.
    evidence_files = files+target_files+[args.initial/'mechanical_modes.csv', args.initial/'mechanical_mode_nodes.csv',
                                       args.control/'profiles/fault_11.csv', args.target/'parameters.prm']
    result = dict(method='Exact uniform-Maxwell response scaling; actual target QP friction, no history update',
                  units='Pa/(m/s) for modal coefficients; Pa for response to 1e-12 m/s nodal amplitude',
                  amplitude_caution='Linear tangent normalization only: 1e-12 m/s is not an admissible finite trial at every nearly locked node',
                  target_step=11, target_time_s=float(control['time_s'][0]), kappa_ratio=float(ratio.mean()),
                  captured_QPs=len(key_list), required_nonzero_mode_QPs=len(required),
                  skipped_truncated_last_rows=truncated, geometry_relative_errors=geometry_checks,
                  broad_target_reuse_relative_errors=scaling_checks, weak_row_coverage_error=row_coverage,
                  captured_V_vs_accepted_control_relative_L2=velocity_check,
                  baseline_QP_sigma_Pa=qp_stats(base['sigma']), modes=results, cross_actions=cross,
                  initial_native_checks=[{key: row[key].item() for key in modes.dtype.names} for row in modes],
                  provenance={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in evidence_files})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'mechanical_response.json').write_text(json.dumps(result, indent=2)+'\n')
    with (args.output/'step11_response_weak_profiles.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader(); writer.writerows(output_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 3, figsize=(14, 11), sharex=True)
    for col, (data, load) in enumerate(zip(native, loads)):
        order = np.flatnonzero(window)[np.argsort(data['xd'][window])]
        x = data['xd'][order]/1000
        axes[0, col].plot(x, data['deltaV'][order]/1e-12)
        axes[0, col].set_title(names[col])
        for row, keys in enumerate((('delta_q', 'delta_p', 'minus_delta_tau_N'),
                                    ('minus_mu_delta_p', 'mu_delta_tau_N', 'instantaneous_friction'),
                                    ('full', 'frozen_normal')), start=1):
            for key in keys:
                axes[row, col].plot(x, load[key][order]/mass_row[order]/1000, label=key)
            axes[row, col].legend(fontsize=7)
        axes[3, col].set_xlabel('Down-dip distance (km)')
        for ax in axes[:, col]: ax.grid(alpha=.25)
    axes[0, 0].set_ylabel('Velocity variation / 1e-12 m/s')
    for row in (1, 2, 3): axes[row, 0].set_ylabel('Mass-row weak average (kPa)')
    fig.suptitle('Step 11: exact reused bulk responses with actual incoming-state friction; no history evolution')
    fig.tight_layout(); fig.savefig(args.output/'step11_mechanical_response.png', dpi=180); plt.close(fig)
    print(json.dumps({key: value for key, value in result.items() if key not in ('provenance', 'initial_native_checks')}, indent=2))


if __name__ == '__main__':
    main()
