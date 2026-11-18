"""Reduce the bounded step-89 search comparison; unfinished runs stay explicit."""
import json
import os
from pathlib import Path
import re

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
from analyze_time_update import read
from analyze_material_overnight import fast_frame, fast_balance

RUNS = Path('build-particle-free/bulk-tip/runs')
OUT = Path('benchmarks/reconstructed_fault/particle_free/evidence/tip_step89_search')


def log_summary(order):
    run = RUNS / ('step89-search-' + order)
    text = (run / 'run.log').read_text()
    steps = [dict(iteration=int(n), mechanics=float(m), phase=float(k))
             for n, m, k in re.findall(r'coupled Newton (\d+): mechanics=([^,]+), phase KKT=([^,]+),', text)]
    linear = [dict(pass_index=int(p), iterations=int(n), residual=float(r), seconds=float(t))
              for p, n, r, t in re.findall(r'Tip bound linear solve: pass=(\d+), iterations=(\d+), residual=([^,]+), seconds=(\S+)', text)]
    vi = [dict(pass_index=int(p), feasible=int(f), phase=float(k), mechanics=float(m), continuity=float(c))
          for p, f, k, m, c in re.findall(r'Tip linearized VI: pass=(\d+), feasible=(\d+), phase=([^,]+), mechanics=([^,]+), continuity=(\S+)', text)]
    trials = [dict(shift=float(s), alpha=float(a), merit=float(m), functional_change=float(f), slope=float(d))
              for s, a, m, f, d in re.findall(r'Tip search trial: shift=([^,]+), alpha=([^,]+), merit=([^,]+), functional change=([^,]+), slope=(\S+)', text)]
    command = json.loads((run / 'command.json').read_text())
    result = dict(command=command, nonlinear=steps, linear=linear, linearized_VI=vi, trials=trials,
                  successful_exact_Krylov=sum(r['iterations'] for r in linear),
                  successful_exact_seconds=sum(r['seconds'] for r in linear))
    result['accepted_searches'] = [dict(shift=float(s), alpha=float(a)) for s, a in
                                  re.findall(r'accepted search: shift=([^,]+), alpha=(\S+)', text)]
    groups = []
    for line in text.splitlines():
        m = re.search(r'Tip linearized VI: pass=(\d+), feasible=(\d+), phase=([^,]+)', line)
        if m:
            if int(m[1]) == 0:
                groups.append(dict(checks=0, linear_solves=0, krylov=0, seconds=0., max_VI=0.))
            g = groups[-1]
            g['checks'] += 1
            g['max_VI'] = max(g['max_VI'], float(m[3]))
            g['last_VI'] = float(m[3])
            g['last_feasible'] = bool(int(m[2]))
            g['certified'] = int(m[1]) > 0 and int(m[2]) == 1 and float(m[3]) < 1e-9
        m = re.search(r'Tip bound linear solve: pass=(\d+), iterations=(\d+), residual=([^,]+), seconds=(\S+)', line)
        if m:
            g = groups[-1]
            g['linear_solves'] += 1
            g['krylov'] += int(m[2])
            g['seconds'] += float(m[4])
    result['exact_searches'] = groups
    accepted = run / 'output/bulk-accepted.csv'
    if accepted.exists():
        rows = read(accepted)
        assert len(rows) == 1 and rows.step[0] == 89
        result['accepted'] = {n: float(rows[n][0]) for n in rows.dtype.names}
        r = result['accepted']
        result['physical_work_error'] = r['Wext'] - r['energy_change'] - r['Db'] - r['Dc'] - r['Df']
        parent = read(RUNS / 'fine-growth-warm01/output/bulk-accepted.csv')[-1]
        assert parent.step == 88
        result['added_fracture'] = r['fracture'] - float(parent.fracture)
        result['physical_error_over_added_fracture'] = result['physical_work_error'] / result['added_fracture']
        coefficients = fast_balance(run / 'output', 'tip-balance-0089-r*.csv')
        result['native_bounds'] = dict(minimum=float(min(coefficients.B)), maximum=float(max(coefficients.B)),
                                     minimum_increment=float(min(coefficients.B - coefficients.lower)),
                                     maximum_increment=float(max(coefficients.B - coefficients.lower)))
        assert result['native_bounds']['minimum_increment'] >= 0 and result['native_bounds']['maximum'] <= 1
        for table in ('tip-profile', 'tip-solver', 'tip-power'):
            a = read(run / ('output/' + table + '.csv'))
            result[table] = {n: float(a[n][-1]) for n in a.dtype.names}
    return result


def error(a, b, weight):
    difference = a - b
    l2 = float(np.sqrt(np.sum(weight * difference**2)))
    norm = float(np.sqrt(np.sum(weight * b**2)))
    return dict(maximum=float(np.max(abs(difference))), L2=l2,
                relative_L2=l2 / norm if norm else None)


def endpoint_comparison():
    dirs = [RUNS / ('step89-search-' + o) / 'output' for o in ('positive', 'default')]
    a, b = [fast_frame(d, 89) for d in dirs]
    assert np.array_equal(a.cell, b.cell) and np.array_equal(a.q, b.q)
    assert np.array_equal(a.weight, b.weight)
    fields = {n: error(a[n], b[n], b.weight) for n in ('phi', 'tauxx', 'tauyy', 'tauzz', 'tauxy', 'Axx', 'Ayy', 'Azz', 'Axy', 'qcoh', 'w')}
    for name, columns in [('stress', ['tauxx', 'tauyy', 'tauzz', 'tauxy']), ('A', ['Axx', 'Ayy', 'Azz', 'Axy'])]:
        av = np.column_stack([a[n] for n in columns]); bv = np.column_stack([b[n] for n in columns])
        fields[name] = error(av, bv, b.weight[:, None] * np.array([1, 1, 1, 2]))
    a, b = [fast_balance(d, 'tip-balance-0089-r*.csv') for d in dirs]
    assert np.array_equal(a.cell, b.cell) and np.array_equal(a.i, b.i)
    fields['phase_coefficients'] = error(a.B, b.B, b.mass)
    a, b = [read(d / 'tip-sections-0089.csv') for d in dirs]
    assert np.array_equal(a.s, b.s) and np.array_equal(a.ds, b.ds)
    fields['sections'] = {n: error(a[n], b[n], b.ds) for n in ('age', 'slip', 'V', 'force')}
    return fields


def figures(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    for order in ('positive', 'default'):
        if order not in result:
            continue
        r = result[order]
        label = 'Positive metric first' if order == 'positive' else 'Adaptive exact first'
        label += ' (accepted)' if 'accepted' in r else ' (no accepted endpoint)'
        n = r['nonlinear']
        axes[0, 0].semilogy([p['iteration'] for p in n], [p['phase'] / 1e-7 for p in n], '.-', label=label)
        axes[0, 1].semilogy([p['iteration'] for p in n], [p['mechanics'] / 1e-9 for p in n], '.-', label=label)
        a = r['accepted_searches']
        axes[1, 0].semilogy(range(len(a)), [p['alpha'] for p in a], '.-', label=label)
        v = r['linearized_VI']
        if v:
            axes[1, 1].semilogy(range(len(v)), [max(p['phase'], 1e-18) for p in v], '.-', label=label)
            for i, p in enumerate(v):
                if i and p['pass_index'] == 0:
                    axes[1, 1].axvline(i - .5, color='grey', alpha=.3, linewidth=.6)
    for ax, title, ylabel in zip(axes.flat, ['Physical phase KKT', 'Mechanical residual', 'Accepted line fraction', 'Exact-search true linearized VI'],
                                 ['Residual / 1e-7', 'Residual / 1e-9', 'alpha', 'Phase natural-map residual']):
        ax.set(title=title, xlabel='Nonlinear iteration' if ax != axes[1, 1] else 'Logged exact active-set pass', ylabel=ylabel)
        ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    for ax in axes[0]:
        ax.axhline(1., color='k', linestyle=':', linewidth=.8)
    axes[1, 1].axhline(1e-9, color='k', linestyle=':', linewidth=.8)
    fig.suptitle('Single interval 88 → 89; identical parent, equations and acceptance gates')
    for extension in ('png', 'pdf'):
        fig.savefig(OUT / ('search-comparison.' + extension), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    result = {order: log_summary(order) for order in ('positive', 'default')
              if (RUNS / ('step89-search-' + order) / 'run.log').exists()}
    if len(result) == 2 and all('accepted' in r for r in result.values()):
        result['endpoint_difference'] = endpoint_comparison()
    OUT.mkdir(exist_ok=True)
    (OUT / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    figures(result)
    print(json.dumps({order: {k: r[k] for k in ('successful_exact_Krylov', 'successful_exact_seconds', 'accepted', 'physical_work_error') if k in r}
                      | {'last_nonlinear': r['nonlinear'][-1:]}
                      for order, r in result.items() if order != 'endpoint_difference'}, indent=2))
