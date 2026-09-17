"""Compare the wide/narrow saved-clock trajectories without rerunning a control.

Native weak traction = sum(JxW chi N_i traction)/sum(JxW chi N_i).
Spatial statistics use physical along-fault length, not node counts. Initial
offsets and each run's subsequent increments are recorded separately.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

os.environ.setdefault('MPLCONFIGDIR', '/tmp/aspect-box-width-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_uniform_sliding import read, records
from check_first_cycle_restart import convergence

HERE = Path(__file__).resolve().parent
VP = 1e-9
YEAR = 31557600.
REGIONS = {'0-40km': (0., 40000.), '0-15km': (0., 15000.),
           '15-18km': (15000., 18000.), '18-40km': (18000., 40000.)}


def stats(x, values, lo, hi):
    """Exact mean/RMS of the linearly interpolated reporting profile."""
    xx = np.r_[lo, x[(x > lo) & (x < hi)], hi]
    v = np.interp(xx, x, values)
    h = np.diff(xx)
    mean = np.sum(h*(v[:-1]+v[1:])/2)/(hi-lo)
    rms = np.sqrt(np.sum(h*(v[:-1]**2+v[:-1]*v[1:]+v[1:]**2)/3)/(hi-lo))
    centered = v-mean
    variation = np.sqrt(np.sum(h*(centered[:-1]**2+centered[:-1]*centered[1:]+centered[1:]**2)/3)/(hi-lo))
    i = np.argmax(abs(v))
    return dict(mean=float(mean), rms=float(rms), variation_rms=float(variation), max_abs=float(abs(v[i])),
                xd_at_max_m=float(xx[i]), minimum=float(min(v)), maximum=float(max(v)))


def fields(root, step):
    f = read(root/f'fault_{step}.csv')
    w = read(root/f'work_weak_{step}.csv')
    np.testing.assert_array_equal(f['xd'], w['xd'])
    assert np.all(w['weight'] > 0)
    values = dict(V_over_Vp=f['V']/VP, slip_m=f['slip'], Theta_s=f['Theta'],
                  shear_Pa=w['q']/w['weight'], normal_Pa=w['sigma']/w['weight'],
                  pressure_Pa=w['p']/w['weight'], minus_tauN_Pa=-w['tauN']/w['weight'])
    np.testing.assert_allclose(values['normal_Pa'], 50e6+values['pressure_Pa']+values['minus_tauN_Pa'],
                               rtol=2e-14, atol=1e-7)
    return f, w, values


def parameters(root):
    def flatten(tree, path=''):
        result = {}
        for key, value in tree.items():
            if isinstance(value, dict) and 'value' in value:
                result[path+'/'+key] = value['value']
            elif isinstance(value, dict):
                result.update(flatten(value, path+'/'+key))
        return result
    return flatten(json.loads((root/'parameters.json').read_text()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=HERE/'fully-frictional-cleanup-local4')
    parser.add_argument('--run', type=Path, default=HERE/'wide-seven-local4')
    args = parser.parse_args()
    out = args.run/'width-comparison'
    out.mkdir(exist_ok=True)
    roots = dict(narrow=args.reference, wide=args.run)
    summary = {'definitions': {'difference': 'wide minus narrow',
        'increment_difference': '(wide_k-wide_0)-(narrow_k-narrow_0)',
        'traction': 'native mechanical weak load divided by native JxW*chi*N_i weight',
        'spatial_norm': 'length-normalized integral of squared linear reporting profile; no 50 MPa scaling'},
        'checks': {}, 'execution': {}}
    old, new = [parameters(root) for root in roots.values()]
    changes = {k: [old.get(k), new.get(k)] for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
    allowed = {'/Output directory', '/Geometry model/Box/Box origin X coordinate',
               '/Geometry model/Box/X extent', '/Geometry model/Box/X repetitions',
               '/Geometry model/Box/Y repetitions', '/Mesh refinement/Initial global refinement',
               '/Mesh refinement/Initial adaptive refinement',
               '/Termination criteria/Termination criteria', '/Termination criteria/End step',
               '/Termination criteria/Wall time'}
    # The maintained fixture relocated immutable inputs after the saved run.
    # Allow path changes only when the actual bytes are identical.
    relocated = {}
    for key in ('/Fault reconstruction/Prescribed faults file',
                '/Postprocess/BP3/Mature prestress file',
                '/Postprocess/BP3/Bottom normalization completion file'):
        if key in changes:
            hashes = [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in changes[key]]
            assert hashes[0] == hashes[1], key
            relocated[key] = hashes[0]
            allowed.add(key)
    assert set(changes) <= allowed, changes
    summary['effective_parameter_changes'] = changes
    summary['relocated_identical_inputs_sha256'] = relocated
    clocks = {}
    for tag, root in roots.items():
        ex = json.loads((root/'execution.json').read_text())
        assert ex['status'] == 0 and ex['fresh_linear_passed'] and ex['first_update_passed']
        summary['execution'][tag] = ex
        summary['checks'][tag] = {'convergence': convergence(root/'run.log')}
        log = (root/'run.log').read_text()
        weak_checks = re.findall(r'Work replay accepted observation: step=(\d+), frozen weak stress discrepancy=([^ ]+) Pa', log)
        assert [int(k) for k, _ in weak_checks] == list(range(8))
        assert max(float(error) for _, error in weak_checks) < 1e-5
        summary['checks'][tag]['max_weak_observer_error_Pa'] = max(float(error) for _, error in weak_checks)
        histories = [read(root/f'history_{k}.csv') for k in range(8)]
        summary['checks'][tag]['max_Theta_audit_relative_error'] = max(float(h['Theta_reference_relative_error'][0]) for h in histories)
        assert summary['checks'][tag]['max_Theta_audit_relative_error'] <= 1e-12
        assert histories[0]['committed_particle_stress_max_Pa'][0] == 0.
        clocks[tag] = read(root/'accepted_steps.csv')
        np.testing.assert_array_equal(clocks[tag]['step'], np.arange(8))
        np.testing.assert_array_equal(clocks[tag]['free'], 1236)
        np.testing.assert_array_equal(clocks[tag]['lower_active'], 0)
    for key in ('time', 'dt'):
        np.testing.assert_array_equal(clocks['narrow'][key], clocks['wide'][key])

    initial = {}; previous = {}; metrics = []; profiles = []; stations = []; invariant = []
    final = {}
    for k in range(8):
        pair = {tag: fields(root, k) for tag, root in roots.items()}
        for key in ('xd', 'x', 'y', 'time', 'dt', 'sigma_n_bg', 'prescribed'):
            np.testing.assert_array_equal(pair['narrow'][0][key], pair['wide'][0][key])
        # tau_bg in fault_k.csv is the observer's mass-inverted weak background,
        # not the immutable input table. Its projection has MPI roundoff.
        np.testing.assert_allclose(pair['narrow'][0]['tau_bg'], pair['wide'][0]['tau_bg'], rtol=1e-8, atol=0.)
        for tag, (f, w, values) in pair.items():
            if k == 0:
                initial[tag] = (f, w, values)
                np.testing.assert_array_equal(f['slip'], 0.)
            else:
                old = previous[tag]
                x = f['V']*f['dt']/.008
                theta = old['Theta']*np.exp(-x)-.008/f['V']*np.expm1(-x)
                np.testing.assert_allclose(f['Theta'], theta, rtol=1e-12)
                np.testing.assert_allclose(f['slip'], old['slip']+f['dt']*f['V'], rtol=3e-15, atol=1e-17)
            for key in ('x', 'y', 'Ih'):
                np.testing.assert_array_equal(f[key], initial[tag][0][key])
            np.testing.assert_allclose(f['tau_bg'], initial[tag][0]['tau_bg'], rtol=1e-8, atol=0.)
            np.testing.assert_array_equal(f['C'], 0.)
            np.testing.assert_array_equal(f['prescribed'], 0.)
            previous[tag] = f
        if k == 0:
            np.testing.assert_array_equal(pair['narrow'][0]['Theta'], pair['wide'][0]['Theta'])
        a, b = pair['narrow'], pair['wide']
        invariant.append(dict(step=k, Ih_max_abs_m=float(max(abs(b[0]['Ih']-a[0]['Ih']))),
            Ih_max_relative=float(max(abs(b[0]['Ih']/a[0]['Ih']-1))),
            reported_background_max_abs_Pa=float(max(abs(b[0]['tau_bg']-a[0]['tau_bg']))),
            weight_max_relative=float(max(abs(b[1]['weight']/a[1]['weight']-1)))))
        order = np.argsort(a[0]['xd']); xd = a[0]['xd'][order]
        time = float(a[0]['time'][0]); time_years = time/YEAR
        for field in a[2]:
            va, vb = a[2][field][order], b[2][field][order]
            va0, vb0 = initial['narrow'][2][field][order], initial['wide'][2][field][order]
            ea, eb = va-va0, vb-vb0
            for region, (lo, hi) in REGIONS.items():
                row = dict(step=k, time_years=time_years, region=region, field=field)
                for label, v in [('narrow', va), ('wide', vb), ('difference', vb-va),
                                 ('narrow_increment', ea), ('wide_increment', eb),
                                 ('increment_difference', eb-ea), ('initial_difference', vb0-va0)]:
                    row.update({label+'_'+key: value for key, value in stats(xd, v, lo, hi).items()})
                scale = row['narrow_increment_rms']
                row['increment_difference_over_narrow_evolution_rms'] = row['increment_difference_rms']/scale if scale else None
                row['total_difference_over_narrow_evolution_rms'] = row['difference_rms']/scale if scale else None
                metrics.append(row)
        for i in order:
            if -1e-7 <= a[0]['xd'][i] <= 40000+1e-7:
                row = dict(step=k, time_years=time_years, xd_m=float(a[0]['xd'][i]))
                for field in a[2]:
                    for tag in roots:
                        value = pair[tag][2][field][i]
                        row[tag+'_'+field] = value
                        row[tag+'_increment_'+field] = value-initial[tag][2][field][i]
                    row['difference_'+field] = row['wide_'+field]-row['narrow_'+field]
                    row['increment_difference_'+field] = row['wide_increment_'+field]-row['narrow_increment_'+field]
                profiles.append(row)
        for target in (0., 5000., 10000., 15000., 18000., 25000., 35000., 39900., 39950., 40000.):
            i = int(np.argmin(abs(a[0]['xd']-target)))
            row = dict(step=k, time_years=time_years, target_m=target, xd_m=float(a[0]['xd'][i]))
            for tag in roots:
                for field, values in pair[tag][2].items(): row[tag+'_'+field] = values[i]
            stations.append(row)
        if k == 7: final = pair
    records(out/'metrics.csv', metrics)
    records(out/'profiles_0_40km.csv', profiles)
    records(out/'stations.csv', stations)
    records(out/'geometry_localization_checks.csv', invariant)
    summary.update(final_time_years=time_years, initialization=[r for r in metrics if r['step']==0],
                   final=[r for r in metrics if r['step']==7], localization=invariant,
                   checks_passed=True)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for tag, (f, w, values) in final.items():
        ix = np.argsort(f['xd']); xx = f['xd'][ix]/1000
        for ax, field, scale, inc in [(axes[0,0], 'V_over_Vp', 1., False),
            (axes[0,1], 'slip_m', 1000., False), (axes[1,0], 'shear_Pa', .001, True),
            (axes[1,1], 'normal_Pa', .001, True)]:
            v = values[field]-(initial[tag][2][field] if inc else 0.)
            ax.plot(xx, v[ix]*scale, label=tag)
        v0 = initial[tag][2]
        axes[0,2].plot(xx, (v0['shear_Pa']-initial[tag][1]['bg']/initial[tag][1]['weight'])[ix]/1000, label=tag+' shear perturbation')
        axes[0,2].plot(xx, (v0['normal_Pa']-50e6)[ix]/1000, '--', label=tag+' normal perturbation')
    for field in ('shear_Pa', 'normal_Pa'):
        delta = (final['wide'][2][field]-initial['wide'][2][field])-(final['narrow'][2][field]-initial['narrow'][2][field])
        axes[1,2].plot(xx, delta[ix]/1000, label=field.replace('_Pa',''))
    titles = ['Final V/Vp', 'Final accumulated slip [mm]', 'Initialization: weak stress perturbations [kPa]',
              'Weak shear change since own initialization [kPa]', 'Weak normal change since own initialization [kPa]',
              'Wide minus narrow evolution [kPa]']
    for ax, title in zip(axes.flat, titles):
        ax.set(xlim=(0,40), xlabel='Down-dip distance [km]', title=title)
        for x in (15,18,40): ax.axvline(x, color='gray', lw=.6, ls=':')
        ax.grid(); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'comparison.png', dpi=170); plt.close(fig)
    print('Width comparison passed clock, convergence and history checks:', out)
    for row in summary['final']:
        if row['region']=='0-40km' and row['field'] in ('V_over_Vp','slip_m','shear_Pa','normal_Pa'):
            print(row['field'], 'difference RMS/max:', row['difference_rms'], row['difference_max_abs'],
                  'evolution difference / narrow evolution:', row['increment_difference_over_narrow_evolution_rms'])


if __name__ == '__main__':
    main()
