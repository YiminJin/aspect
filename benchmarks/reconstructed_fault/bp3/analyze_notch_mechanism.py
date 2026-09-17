"""Saved-state force audit and frozen prescribed-node impulse comparisons.

No stresses/histories are advanced. Actual Stokes QP weights are reused; the
frozen-bulk inverse and the fully coupled probe are labelled separately.
"""
import json
import hashlib
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/aspect-notch-mpl')
import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_uniform_sliding import read, cat, select, records, write

HERE = Path(__file__).resolve().parent
OUT = HERE / 'notch-mechanism'
BASE = HERE / 'work-replay-50-local4'
VP, DC, VREF, B, F0, DAMP = 1e-9, .008, 1e-6, .015, .6, 4624440.


def mu(v, theta, a):
    return a*np.arcsinh(v/(2*VREF)*np.exp((F0+B*np.log(theta*VREF/DC))/a))


def assemble(raw, values, n):
    j = raw['segment'].astype(int)
    z = raw['xi']
    w = raw['JxW']*raw['chi']
    return (np.bincount(j, weights=w*(1-z)*values, minlength=n) +
            np.bincount(j+1, weights=w*z*values, minlength=n))


def band(raw, coefficient, n):
    j = raw['segment'].astype(int)
    z = raw['xi']
    w = raw['JxW']*raw['chi']*coefficient
    diag = np.bincount(j, weights=w*(1-z)**2, minlength=n)+np.bincount(j+1, weights=w*z*z, minlength=n)
    upper = np.bincount(j, weights=w*z*(1-z), minlength=n)
    result = np.zeros((3, n))
    result[1] = diag
    result[0, 1:] = upper[:-1]
    result[2, :-1] = upper[:-1]
    return result


def inverse_boundary_impulse(matrix, lo, hi, boundary):
    # Zero perturbations beyond this frozen-bulk patch, prescribed impulse at
    # its deep end. This is NOT the Schur-complement/bulk-relaxed response.
    rhs = np.zeros(hi-lo)
    rhs[0] = -matrix[2, boundary]
    return solve_banded((1, 1), matrix[:, lo:hi], rhs)


def check_reaction_stencil():
    # Exact uniform Q1 reaction-matrix solution on a long finite chain.
    n = 100
    matrix = np.ones((3, n)); matrix[1] = 4.
    rhs = np.zeros(n); rhs[0] = -1.
    response = solve_banded((1, 1), matrix, rhs)
    expected = (-2+np.sqrt(3.))**np.arange(1, n+1)
    np.testing.assert_allclose(response[:20], expected[:20], rtol=2e-14, atol=1e-24)
    np.testing.assert_array_equal(solve_banded((1, 1), matrix, np.zeros(n)), np.zeros(n))


def inventory():
    rows = []
    profiles = []
    for case, steps in [('work-replay-50-local4', [0, 1, 2, 3, 5, 7, 9, 10]),
                        ('work-replay-halfdt-50-local4', [20]),
                        ('mature-fault-50-local4', [10])]:
        for step in steps:
            f = read(HERE/case/f'fault_{step}.csv')
            order = np.argsort(f['xd'])
            x, v = f['xd'][order], f['V'][order]
            for name, lo, hi in [('15', 14000, 16000), ('18', 17000, 19000), ('40', 39000, 40050)]:
                indices = np.flatnonzero((x >= lo)&(x <= hi))
                minima = [i for i in indices if 0 < i < len(v)-1 and v[i] < min(v[i-1], v[i+1])]
                rows.append(dict(case=case, step=step, time_yr=f['time'][0]/31557600, region=name,
                                 minimum_V_over_Vp=min(v[indices])/VP,
                                 local_minima_m=';'.join(str(x[i]) for i in minima),
                                 local_minima_V_over_Vp=';'.join(str(v[i]/VP) for i in minima)))
            if step in (0, 3, 5, 10, 20):
                for i in range(len(x)):
                    if 13000 <= x[i] <= 20000 or 39000 <= x[i] <= 40200:
                        j = order[i]
                        profiles.append(dict(case=case, step=step, time_yr=f['time'][0]/31557600,
                                             xd=x[i], V=f['V'][j], Theta=f['Theta'][j], slip=f['slip'][j]))
    records(OUT/'extrema_inventory.csv', rows)
    records(OUT/'saved_profiles.csv', profiles)
    # The initial VW state is far below steady state, even though initial V
    # is Vp. This homogeneous-material diagnostic isolates the first aging
    # kick, not the subsequent coupled stress response.
    f0, f1, f2 = [read(BASE/f'fault_{k}.csv') for k in (0, 1, 2)]
    controls = []
    for xd in (14000., 15000., 16500., 18000., 25000., 39950., 40000.):
        i = int(np.argmin(abs(f0['xd']-xd)))
        a = .01+.015*np.clip((xd-15000)/3000, 0, 1)
        added = 50e6*(mu(VP, f1['Theta'][i], a)-mu(VP, f0['Theta'][i], a))
        controls.append(dict(xd=xd, theta_initial=f0['Theta'][i], theta_after_first_update=f1['Theta'][i],
                             theta_initial_over_steady=f0['Theta'][i]/(DC/VP),
                             state_only_friction_increment_Pa=added,
                             homogeneous_constant_stress_V_over_Vp=(f0['Theta'][i]/f1['Theta'][i])**(B/a),
                             actual_V1_over_Vp=f1['V'][i]/VP, actual_V2_over_Vp=f2['V'][i]/VP))
    records(OUT/'initial_aging_control.csv', controls)

    gradients = []
    for k in (3, 5, 10):
        f = read(BASE/f'fault_{k}.csv')
        order = np.argsort(f['xd']); x = f['xd'][order]
        for i in range(1, len(x)-1):
            if not (13000 <= x[i] <= 20000 or 39000 <= x[i] <= 40200):
                continue
            row = dict(step=k, xd=x[i], h_left=x[i]-x[i-1], h_right=x[i+1]-x[i],
                       V=f['V'][order[i]], slip=f['slip'][order[i]])
            for field in ('V', 'slip'):
                y = f[field][order]
                row[field+'_gradient_left'] = (y[i]-y[i-1])/(x[i]-x[i-1])
                row[field+'_gradient_right'] = (y[i+1]-y[i])/(x[i+1]-x[i])
                row[field+'_gradient_jump'] = row[field+'_gradient_right']-row[field+'_gradient_left']
            gradients.append(row)
    records(OUT/'gradient_changes.csv', gradients)


def final_audit():
    f = read(BASE/'fault_10.csv')
    old = read(BASE/'fault_9.csv')
    props = read(OUT/'impulse-0.01-local4/notch_input_properties.csv')
    strengthening = [k for k in props if 'strengthening' in k]
    assert len(strengthening) == 1, strengthening
    raw_fraction = props[strengthening[0]]
    fraction = np.clip(raw_fraction, 0, 1)
    a = .01+.015*fraction
    raw = cat(BASE.glob('work_qp_10_rank*.csv'), ('cell',))
    raw = select(raw, (raw['source_active'] == 1)&(raw['chi'] > 0))
    j = raw['segment'].astype(int)
    z = raw['xi']
    theta_q = (1-z)*old['Theta'][j]+z*old['Theta'][j+1]
    # Match production: interpolate the stored chemical field FIRST, then
    # convert to bounded fractions. Clipping the nodes first is a different law.
    a_q = .01+.015*np.clip((1-z)*raw_fraction[j]+z*raw_fraction[j+1], 0, 1)
    v_q = (1-z)*f['V'][j]+z*f['V'][j+1]
    np.testing.assert_allclose(raw['V'], v_q, rtol=1e-13, atol=1e-25)
    friction = mu(v_q, theta_q, a_q)*raw['sigma_n']
    residual = raw['q']-friction-DAMP*v_q
    n = len(f['V'])
    weak = read(BASE/'work_weak_10.csv')
    mass = band(raw, np.ones(len(j)), n)
    load = {key: assemble(raw, value, n) for key, value in
            [('measure', np.ones(len(j))), ('q', raw['q']), ('sigma', raw['sigma_n']),
             ('friction', friction), ('R', residual)]}
    full_friction = f['mass_diagonal']*f['friction_traction']
    full_friction[1:] += f['mass_upper'][:-1]*f['friction_traction'][:-1]
    full_friction[:-1] += f['mass_upper'][:-1]*f['friction_traction'][1:]
    complete = ((f['xd'] >= 13200)&(f['xd'] <= 19800)) | ((f['xd'] >= 37200)&(f['xd'] <= 42800))
    errors = {}
    for key, reference in [('measure', weak['weight']), ('q', weak['q']), ('sigma', weak['sigma']),
                           ('friction', full_friction), ('R', f['weak_residual'])]:
        errors[key] = float(max(abs((load[key][complete]-reference[complete])/weak['weight'][complete])))
        assert errors[key] < (1e-8 if key == 'measure' else 2e-5), (key, errors[key])

    # Split the exact local tangent into the Maxwell crack-rate mass and the
    # friction/damping reaction. Bulk relaxation is intentionally absent here.
    kappa = -1e26*np.expm1(-f['dt'][0]*32038120320/1e26)
    Z = v_q/(2*VREF)*np.exp((F0+B*np.log(theta_q*VREF/DC))/a_q)
    mu_v = a_q/v_q*np.where(Z < 1e6, Z/np.sqrt(1+Z*Z), 1.)
    elastic = band(raw, kappa*raw['chi'], n)
    reaction = band(raw, raw['sigma_n']*mu_v+DAMP, n)
    tangent = elastic+reaction
    rates = {}
    for name, matrix in [('mass', mass), ('elastic', elastic), ('reaction', reaction), ('fixed_bulk', tangent)]:
        rates[name] = inverse_boundary_impulse(matrix, 796, 835, 795)
    nodal = []
    for i in np.flatnonzero(complete):
        if f['prescribed'][i]:
            continue
        w = weak['weight'][i]
        qbar, sigbar = weak['q'][i]/w, weak['sigma'][i]/w
        # A labelled lumped/point-inversion diagnostic, not an alternative
        # production law. It exposes averaging bias; it is not a bulk solve.
        root = brentq(lambda logv: mu(np.exp(logv), old['Theta'][i], a[i])*sigbar+
                      DAMP*np.exp(logv)-qbar, np.log(1e-25), np.log(1e-4))
        nodal.append(dict(node=i, xd=f['xd'][i], V=f['V'][i], point_inversion_V=np.exp(root),
                          theta_used=old['Theta'][i], theta_committed=f['Theta'][i], a=a[i],
                          q=qbar, sigma=sigbar, R=load['R'][i]/w,
                          K_elastic_diag=elastic[1, i], K_friction_diag=reaction[1, i]))
    records(OUT/'force_and_point_inversion.csv', nodal)
    rows = []
    for i in range(796, 807):
        rows.append(dict(node=i, xd=f['xd'][i], **{key: value[i-796] for key, value in rates.items()}))
    records(OUT/'frozen_bulk_boundary_response.csv', rows)
    write(OUT/'surface_material.csv', dict(node=np.arange(n), xd=f['xd'],
                                         projected_fraction=fraction, a=a,
                                         exact_sharp_a=.01+.015*np.clip((f['xd']-15000)/3000, 0, 1)))
    (OUT/'force_checks.json').write_text(json.dumps(dict(kappa=kappa, errors_per_row_measure=errors), indent=2)+'\n')


def impulse():
    baseline = read(HERE/'within-step-50-local4/A/noncommitting_surface.csv')
    f = read(BASE/'fault_10.csv')
    rows = []
    summaries = []
    for folder in sorted(OUT.glob('*-local4')):
        metadata = folder/'execution.json'
        if not metadata.exists():
            continue
        check = json.loads(metadata.read_text())
        assert check['converged'] and check['rollback'] and check['incoming_identical']
        if check.get('clamp_others', False):
            continue
        amplitude = check['amplitude']
        location = check.get('location', 40000.)
        node = int(np.argmin(abs(f['xd']-location)))
        d = read(folder/'noncommitting_surface.csv')
        delta = (d['V']-baseline['V'])/(amplitude*VP)
        for i in np.flatnonzero(abs(f['xd']-location) <= 2000):
            rows.append(dict(amplitude=amplitude, location=location, node=i, xd=f['xd'][i], V_baseline=baseline['V'][i],
                             V_probe=d['V'][i], normalized_response=delta[i], prescribed=d['prescribed'][i]))
        summaries.append(dict(amplitude=amplitude, location=location,
                              shallow_adjacent_response=delta[node+1], shallow_next_response=delta[node+2],
                              deep_adjacent_response=delta[node-1], deep_next_response=delta[node-2],
                              prescribed_response=delta[node], active_nodes=int(sum(d['lower_active'])),
                              seconds=check['seconds'], fresh_checks=check['fresh_linear_checks']))
    if rows:
        records(OUT/'coupled_boundary_response.csv', rows)
        (OUT/'impulse_summary.json').write_text(json.dumps(summaries, indent=2)+'\n')

    # Exact last-row incremental force budget at identical production QPs.
    # Use only nonempty rank files: the patch has one owner on this partition.
    def samples(folder):
        return cat([p for p in folder.glob('state_qp_rank*.csv') if p.stat().st_size > 200], ('cell',))
    a = samples(HERE/'within-step-50-local4/A')
    budgets = []
    for folder in sorted(OUT.glob('impulse-*-local4')):
        if not (folder/'execution.json').exists():
            continue
        b = samples(folder)
        for field in ('qp', 'segment', 'xi', 'weight', 'Theta'):
            np.testing.assert_array_equal(a[field], b[field])
        w = a['weight']*np.where(a['segment'] == 795, a['xi'], 1-a['xi'])
        dv, ds = b['V']-a['V'], b['sigma']-a['sigma']
        mu0 = a['friction']/a['sigma']
        dm = b['friction']/b['sigma']-mu0
        kel = a['Kfixed']-a['sigma']*.025/a['V']-DAMP
        contributions = dict(q=b['q']-a['q'], q_bulk=b['q']-a['q']+kel*dv, q_source=-kel*dv,
                             friction_rate=a['sigma']*dm, friction_normal=mu0*ds,
                             friction_cross=dm*ds, damping=DAMP*dv, residual=b['R']-a['R'])
        row = dict(case=folder.name, row_measure=sum(w))
        row.update({key: float(np.dot(w, value)/sum(w)) for key, value in contributions.items()})
        assert abs(row['q_bulk']+row['q_source']-row['friction_rate']-row['friction_normal']-
                   row['friction_cross']-row['damping']-row['residual']) < 1e-7
        budgets.append(row)
    records(OUT/'impulse_force_budget_Pa.csv', budgets)


def operator_column():
    folder = OUT/'operator-column-local4'
    if not (folder/'execution.json').exists():
        return
    check = json.loads((folder/'execution.json').read_text())
    assert check['converged'] and check['rollback'] and check['clamp_others']
    a = read(HERE/'within-step-50-local4/A/noncommitting_surface.csv')
    d = read(folder/'noncommitting_surface.csv')
    f = read(BASE/'fault_10.csv')
    expected = a['V'].copy(); expected[795] += .01*VP
    np.testing.assert_allclose(d['V'], expected, rtol=2e-14, atol=1e-25)
    assert np.all(d['prescribed'])
    measure = a['mass_diagonal'].copy()
    measure[1:] += a['mass_upper'][:-1]
    measure[:-1] += a['mass_upper'][:-1]
    rows = []
    for i in np.flatnonzero(abs(f['xd']-40000) < 2000):
        rows.append(dict(node=i, xd=f['xd'][i],
                         minus_dR_times_Vp_over_measure=-(d['weak_R'][i]-a['weak_R'][i])/.01/measure[i],
                         minus_dq_times_Vp_over_measure=-(d['weak_q'][i]-a['weak_q'][i])/.01/measure[i],
                         dsigma_times_Vp_over_measure=(d['weak_sigma'][i]-a['weak_sigma'][i])/.01/measure[i]))
    records(OUT/'bulk_relaxed_operator_column.csv', rows)

    def samples(root):
        return cat([p for p in root.glob('state_qp_rank*.csv') if p.stat().st_size > 200], ('cell',))
    before = samples(HERE/'within-step-50-local4/A')
    after = samples(folder)
    for key in ('qp', 'segment', 'xi', 'weight', 'Theta'):
        np.testing.assert_array_equal(before[key], after[key])
    w = before['weight']*np.where(before['segment'] == 795, before['xi'], 1-before['xi'])
    shape = np.where(before['segment'] == 795, 1-before['xi'], 0.)
    mu0 = before['friction']/before['sigma']
    muV = .025/before['V']  # Regularization multiplier differs from one below roundoff here.
    kel = before['Kfixed']-before['sigma']*muV-DAMP
    ds = (after['sigma']-before['sigma'])/.01
    dq = (after['q']-before['q'])/.01
    parts = dict(direct_elastic=kel*shape*VP,
                 bulk_shear=-dq-kel*shape*VP,
                 friction_rate=before['sigma']*muV*shape*VP,
                 friction_normal=mu0*ds, damping=DAMP*shape*VP)
    summary = {key: float(np.dot(w, value)/sum(w)) for key, value in parts.items()}
    summary['total_exact_tangent'] = sum(summary.values())
    summary['finite_difference_total'] = float(-np.dot(w, after['R']-before['R'])/sum(w)/.01)
    delta_mu = after['friction']/after['sigma']-mu0
    remainder = (before['sigma']*(delta_mu-muV*(after['V']-before['V'])) +
                 delta_mu*(after['sigma']-before['sigma']))/.01
    summary['finite_increment_friction_remainder'] = float(np.dot(w, remainder)/sum(w))
    assert abs(summary['finite_difference_total']-summary['total_exact_tangent']-
               summary['finite_increment_friction_remainder']) < 1e-6
    summary['row_measure'] = float(sum(w))
    summary['units'] = 'Pa for S_Gamma[796,795] * Vp / row_measure, at baseline'
    summary['seconds'] = check['seconds']
    summary['fresh_checks'] = check['fresh_linear_checks']
    assert abs(sum(w)-measure[796]) < 1e-9
    (OUT/'operator_column_decomposition.json').write_text(json.dumps(summary, indent=2)+'\n')


def plots():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for k in (3, 5, 10):
        f = read(BASE/f'fault_{k}.csv')
        for ax, (lo, hi) in zip(axes, [(14000, 16000), (17500, 18500), (39500, 40100)]):
            ix = np.flatnonzero((f['xd'] >= lo)&(f['xd'] <= hi))[::-1]
            ax.plot(f['xd'][ix]/1000, f['V'][ix]/VP, '.-', label=f'step {k}')
            ax.set(xlabel='Down dip (km)', ylabel='V / Vp')
    axes[0].set_yscale('log')
    for ax in axes:
        ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(OUT/'notches_by_time.png', dpi=180); plt.close(fig)
    path = OUT/'coupled_boundary_response.csv'
    if path.exists():
        d = read(path)
        frozen = read(OUT/'frozen_bulk_boundary_response.csv')
        fig, ax = plt.subplots(figsize=(7, 4))
        for amplitude in np.unique(d['amplitude']):
            m = (d['location'] == 40000)&(d['amplitude'] == amplitude)&(d['xd'] >= 39600)&(d['xd'] <= 40000)
            ax.plot((40000-d['xd'][m])/50, d['normalized_response'][m], 'o-', label=f'coupled {amplitude:.3g}')
        ax.plot((40000-frozen['xd'])/50, frozen['fixed_bulk'], '.--', label='fixed bulk tangent')
        ax.plot((40000-frozen['xd'])/50, frozen['mass'], ':', label='actual work mass only')
        ax.axhline(0, color='k', lw=.6)
        ax.set(xlabel='Free-side distance / 50 m', ylabel='delta V / imposed delta V', xlim=(0, 8))
        ax.legend(); fig.tight_layout(); fig.savefig(OUT/'boundary_impulse.png', dpi=180); plt.close(fig)
        if np.any(d['location'] == 25000):
            fig, ax = plt.subplots(figsize=(7, 4))
            m = (d['location'] == 25000)&(abs(d['xd']-25000) <= 800)
            ax.plot((d['xd'][m]-25000)/100, d['normalized_response'][m], 'o-')
            ax.axhline(0, color='k', lw=.6)
            ax.set(xlabel='Distance from interior probe / 100 m', ylabel='delta V / imposed delta V')
            fig.tight_layout(); fig.savefig(OUT/'interior_impulse.png', dpi=180); plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    check_reaction_stencil()
    inventory()
    if (OUT/'impulse-0.01-local4/notch_input_properties.csv').exists():
        final_audit()
        impulse()
        operator_column()
    plots()
    sources = [Path(__file__), HERE/'run_notch_probe.py', HERE/'bp3.cc', HERE/'within_step_diagnostic.h']
    sources += [BASE/f'fault_{k}.csv' for k in (0, 1, 2, 3, 5, 7, 9, 10)]
    sources += sorted(BASE.glob('work_qp_10_rank*.csv'))
    sources += [BASE/'work_weak_10.csv', HERE/'within-step-50-local4/A/noncommitting_surface.csv']
    (OUT/'analysis_provenance.json').write_text(json.dumps(dict(
        source_baseline='3335d3d26c298ff5aaeba77062b0a77c8d20f0b5',
        sha256={str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        uniform_reaction_stencil_test='passed',
        data_reproduction_assertions='passed'), indent=2)+'\n')
