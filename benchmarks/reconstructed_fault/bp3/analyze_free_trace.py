"""Compare a noncommitting independent trace with the saved lagged BP3 state."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/aspect-trace-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_uniform_sliding import read, cat, select, records, write

HERE = Path(__file__).resolve().parent
BASE = HERE/'work-replay-50-local4'
VP = 1e-9


def weak(raw, values, n, native=True):
    j = raw['segment'].astype(int)
    right = raw['shape1'] if native else raw['xi']
    weight = raw['weight']*values
    return (np.bincount(j, weights=(1-right)*weight, minlength=n)+
            np.bincount(j+1, weights=right*weight, minlength=n))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default='independent-local4-mpi')
    args = parser.parse_args()
    root = HERE/'free-trace'/args.case
    execution = json.loads((root/'execution.json').read_text())
    assert all(execution[k] for k in ('converged', 'rollback', 'trace_checks', 'fresh_linear_passed',
                                    'incoming_identical', 'original_unchanged', 'copy_unchanged'))
    assert not execution['accepted_output_written']
    output = root/'analysis'
    output.mkdir(exist_ok=True)
    baseline, old = read(BASE/'fault_10.csv'), read(BASE/'fault_9.csv')
    result = read(root/'noncommitting_surface.csv')
    trial = cat(root.glob('state_qp_rank*.csv'), ('cell',))
    j = trial['segment'].astype(int)
    trial['xd'] = (1-trial['xi'])*baseline['xd'][j]+trial['xi']*baseline['xd'][j+1]
    expected_rate = trial['shape0']*result['V'][j]+trial['shape1']*result['V'][j+1]
    np.testing.assert_allclose(trial['V'], expected_rate, rtol=3e-14, atol=1e-28)
    expected_theta = (1-trial['xi'])*old['Theta'][j]+trial['xi']*old['Theta'][j+1]
    np.testing.assert_allclose(trial['Theta'], expected_theta, rtol=3e-14)
    deep = j == 794
    assert np.any(deep)
    np.testing.assert_array_equal(trial['shape1'][deep], 0.)
    np.testing.assert_array_equal(trial['V'][deep], VP)
    assert result['prescribed'][795] == 0 and result['prescribed'][794] == 1

    raw_base = cat(BASE.glob('work_qp_10_rank*.csv'), ('cell',))
    raw_base = select(raw_base, (raw_base['source_active'] == 1)&(raw_base['chi'] > 0))
    raw_base['weight'] = raw_base['JxW']*raw_base['chi']
    raw_base['shape1'] = raw_base['xi']
    raw_base['sigma'] = raw_base['sigma_n']
    raw_base['minus_tauN'] = -raw_base['tauN']
    trial['minus_tauN'] = trial['sigma']-50e6-trial['p']
    # Check the displayed tensor against the actual production normal traction.
    normal_tensor = .75*trial['tau_xx']+.25*trial['tau_yy']-np.sqrt(3)/2*trial['tau_xy']
    np.testing.assert_allclose(trial['sigma'], 50e6+trial['p']-normal_tensor, rtol=3e-13, atol=2e-6)
    theta_error = float(max(abs(trial['Theta']/expected_theta-1)))

    n = len(result['V'])
    mass = weak(trial, np.ones(len(j)), n)
    reconstructed = {name: weak(trial, trial[column], n) for name, column in
                     [('weak_q', 'q'), ('weak_sigma', 'sigma'), ('weak_R', 'R')]}
    reproduction = {name: float(max(abs(value-result[name])/mass)) for name, value in reconstructed.items()}
    assert max(reproduction.values()) < 2e-4, reproduction
    predicted_base = old['slip']+baseline['dt'][0]*baseline['V']
    predicted_trial = old['slip']+baseline['dt'][0]*result['V']
    np.testing.assert_allclose(predicted_base, baseline['slip'], rtol=1e-13, atol=1e-14)
    node_rows = []
    for i in range(792, 802):
        node_rows.append(dict(node=i, xd=baseline['xd'][i], V_base=baseline['V'][i], V_free_trace=result['V'][i],
                             V_ratio_change=(result['V'][i]-baseline['V'][i])/VP, old_Theta=old['Theta'][i],
                             old_slip=old['slip'][i], predicted_slip_base=predicted_base[i],
                             predicted_slip_free_trace=predicted_trial[i], prescribed=result['prescribed'][i],
                             lower_active=result['lower_active'][i], native_mass=mass[i],
                             native_residual_Pa=result['weak_R'][i]/mass[i]))
    records(output/'nodes.csv', node_rows)

    budgets = []
    control = cat((HERE/'within-step-50-local4/A').glob('state_qp_rank*.csv'), ('cell',))
    control['shape1'] = control['xi']
    for name, data, nodes in [('baseline', control, (796,)), ('independent_trace', trial, (795, 796, 797))]:
        for i in nodes:
            for segment in (i-1, i):
                mask = data['segment'] == segment
                shape = data['shape1'][mask] if i == segment+1 else 1-data['shape1'][mask]
                weight = data['weight'][mask]*shape
                budgets.append(dict(case=name, node=i, segment=segment, measure=float(sum(weight)),
                                    **{key: float(weight@data[key][mask]) for key in ('q', 'friction', 'damping', 'sigma', 'R')}))
    records(output/'element_budgets.csv', budgets)

    # Compare the same actual bulk QPs, not differently weighted nodal averages.
    stats = []
    differences = []
    common_agreement = {}
    for region, lo, hi in [('junction', 37000., 43000.), ('last_elements', 39900., 40050.),
                           ('transition', 13000., 20000.)]:
        p = select(raw_base, (raw_base['xd'] >= lo)&(raw_base['xd'] <= hi))
        r = select(trial, (trial['xd'] >= lo)&(trial['xd'] <= hi))
        p = select(p, np.lexsort((p['x'], p['y'])))
        r = select(r, np.lexsort((r['x'], r['y'])))
        np.testing.assert_array_equal(p['x'], r['x'])
        np.testing.assert_array_equal(p['y'], r['y'])
        np.testing.assert_allclose(p['weight'], r['weight'], rtol=2e-12, atol=1e-16)
        np.testing.assert_allclose(p['chi'], r['chi'], rtol=2e-12, atol=1e-20)
        common_agreement[region] = dict(samples=len(p['x']),
                                        max_relative_chi_difference=float(max(abs(r['chi']/p['chi']-1))))
        for name, data in [('baseline', p), ('independent_trace', r)]:
            for field in ('p', 'minus_tauN', 'sigma', 'q', 'tau_xx', 'tau_yy', 'tau_xy'):
                low, high = np.argmin(data[field]), np.argmax(data[field])
                stats.append(dict(region=region, case=name, field=field, minimum=data[field][low], maximum=data[field][high],
                                  minimum_xd=data['xd'][low], maximum_xd=data['xd'][high],
                                  weighted_mean=float(data['weight']@data[field]/sum(data['weight'])),
                                  peak_to_peak=float(np.ptp(data[field]))))
        paired = dict(xd=p['xd'], x=p['x'], y=p['y'], weight=p['weight'])
        for field in ('V', 'p', 'minus_tauN', 'sigma', 'q'):
            paired[field+'_base'] = p[field]
            paired[field+'_trace'] = r[field]
            delta = r[field]-p[field]
            differences.append(dict(region=region, field=field, mean=float(p['weight']@delta/sum(p['weight'])),
                                    rms=float(np.sqrt(p['weight']@(delta*delta)/sum(p['weight']))),
                                    maximum_absolute=float(max(abs(delta)))))
        write(output/f'{region}_common_qp.csv', paired)
    records(output/'raw_extrema.csv', stats)
    records(output/'common_qp_differences.csv', differences)

    native_sigma = weak(trial, trial['sigma'], n)/mass
    common_mass = weak(trial, np.ones(len(j)), n, native=False)
    common_sigma = weak(trial, trial['sigma'], n, native=False)/common_mass
    # The baseline saved native surface values reconstruct their weak average.
    bmass = baseline['mass_diagonal'].copy()
    bmass[:-1] += baseline['mass_upper'][:-1]
    bmass[1:] += baseline['mass_upper'][:-1]
    junction_raw = select(raw_base, (raw_base['xd'] > 37000)&(raw_base['xd'] < 43000))
    old_sigma_load = weak(junction_raw, junction_raw['sigma'], n)
    common_base_sigma = old_sigma_load/bmass
    # The saved raw window does not cover every node. Do not present absent
    # samples or partly covered boundary rows as a zero physical traction.
    covered_mass = weak(junction_raw, np.ones(len(junction_raw['xi'])), n)
    complete = np.isclose(covered_mass, bmass, rtol=2e-12, atol=1e-12)
    common_base_sigma[~complete] = np.nan
    order = np.argsort(baseline['xd'])
    write(output/'nodal_profile.csv', dict(xd=baseline['xd'][order], V_base=baseline['V'][order],
          V_trace=result['V'][order], predicted_slip_base=predicted_base[order], predicted_slip_trace=predicted_trial[order],
          native_sigma_trace=native_sigma[order], common_sigma_trace=common_sigma[order],
          common_sigma_base=common_base_sigma[order]))
    spacing = baseline['xd'][795]-baseline['xd'][796]
    old_gradient = (old['slip'][795]-old['slip'][796])/spacing
    summary = dict(time=baseline['time'][0], dt=baseline['dt'][0], old_Theta_error=theta_error,
                   weak_load_reproduction_Pa=reproduction, common_QPs=common_agreement,
                   free_trace_ratio=result['V'][795]/VP, deep_trace_ratio=1.,
                   V39950_base=baseline['V'][796]/VP, V39950_trace=result['V'][796]/VP,
                   V39900_base=baseline['V'][797]/VP, V39900_trace=result['V'][797]/VP,
                   neighbor_contrast_base=(baseline['V'][797]-baseline['V'][796])/VP,
                   neighbor_contrast_trace=(result['V'][797]-result['V'][796])/VP,
                   last_element_predicted_gradient_base=(predicted_base[795]-predicted_base[796])/spacing,
                   last_element_predicted_gradient_trace=(predicted_trial[795]-predicted_trial[796])/spacing,
                   old_last_element_gradient=old_gradient,
                   last_element_gradient_increment_base=baseline['dt'][0]*(baseline['V'][795]-baseline['V'][796])/spacing,
                   last_element_gradient_increment_trace=baseline['dt'][0]*(result['V'][795]-result['V'][796])/spacing,
                   predicted_slip_jump_free_minus_deep=predicted_trial[795]-predicted_base[795],
                   free_nodes=int(sum(result['prescribed'] == 0)), lower_active=int(sum(result['lower_active'])),
                   seconds=execution['seconds'], krylov=execution['krylov'], fresh_checks=execution['fresh_linear_checks'])
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')

    order = order[(baseline['xd'][order] >= 39600.) & (baseline['xd'][order] <= 40400.)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(baseline['xd'][order]/1000, baseline['V'][order]/VP, '.-', label='baseline')
    free = baseline['xd'][order] <= 40000.000001
    axes[0].plot(baseline['xd'][order][free]/1000, result['V'][order][free]/VP, '.-', label='independent free trace')
    axes[0].plot([40, 40.5], [1, 1], '--', label='prescribed deep trace')
    axes[1].plot(baseline['xd'][order]/1000, predicted_base[order], '.-', label='baseline')
    axes[1].plot(baseline['xd'][order][free]/1000, predicted_trial[order][free], '.-', label='independent free trace')
    axes[1].plot([40, 40.5], [predicted_base[795]]*2, '--', label='deep trace')
    axes[2].plot(baseline['xd'][order]/1000, common_base_sigma[order]/1e6, '.-', label='baseline, common weights')
    axes[2].plot(baseline['xd'][order]/1000, common_sigma[order]/1e6, '.-', label='trial, common weights')
    axes[2].plot(baseline['xd'][order]/1000, native_sigma[order]/1e6, ':', label='trial, native weights')
    for ax, title in zip(axes, ('V/Vp', 'Predicted slip (m), not committed', 'Weak normal traction (MPa)')):
        ax.set_xlim(39.6, 40.4);ax.set_xlabel('Down-dip distance (km)');ax.set_title(title)
        ax.grid(True, alpha=.3);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/'junction_comparison.png', dpi=160);plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
