"""Offline aging/weak-friction accounting; never runs or changes a trajectory.

Read archived QP exports directly, retaining only the 14--19 km window.
Nodal aging identities and production-QP force budgets are separate diagnostics.
"""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

import numpy as np
from analyze_disturbance import ROOT, table, infer_a, mu, quadratic

REPO = Path(__file__).resolve().parents[3]
ARCHIVE = REPO / '.benchmark-cleanup-20260917-reconstructed-alLfUb'
OUT = ROOT / 'transient-accounting'
YEAR, DC, B = 31557600., .008, .015
STEPS = range(12, 44)
BRANCHES = ('plus32', 'state32', 'normal32')


def write_csv(name, rows):
    with (OUT / name).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_qps(nodes):
    """Hash-check full archived records, then keep exact production samples."""
    cache = OUT / 'window_qp.npz'
    if cache.exists():
        with np.load(cache) as data:
            return {key: data[key] for key in data.files}
    manifest = json.loads((ARCHIVE / 'manifest.json').read_text())
    selected = {}
    for record in manifest['files']:
        path = Path(record['path'])
        if path.parent.name not in ('reference32', *BRANCHES):
            continue
        if not path.name.startswith('disturbance_qp_'):
            continue
        step = int(path.stem.split('_')[2])
        if step in STEPS:
            selected[record['path']] = record
    assert len(selected) == 4*32*4
    blocks, provenance = {}, []
    for archive in sorted({r['archive'] for r in selected.values()}):
        proc = subprocess.Popen(['zstd', '-q', '-dc', str(ARCHIVE / archive)],
                                stdout=subprocess.PIPE)
        with tarfile.open(fileobj=proc.stdout, mode='r|') as tar:
            for member in tar:
                if member.name not in selected:
                    continue
                record = selected[member.name]
                raw = tar.extractfile(member).read()
                assert len(raw) == record['bytes']
                assert hashlib.sha256(raw).hexdigest() == record['sha256']
                path = Path(member.name)
                step = int(path.stem.split('_')[2])
                rank = int(path.stem.split('rank')[-1])
                q = np.frombuffer(raw, dtype='<f8').reshape(-1, 12)
                j = q[:, 2].astype(int)
                xd = (1-q[:, 3])*nodes['xd'][j] + q[:, 3]*nodes['xd'][j+1]
                q = q[(xd >= 14000) & (xd <= 19000)].copy()
                q[:, 0] += rank*1e7
                blocks.setdefault(f'{path.parent.name}_{step}', []).append(q)
                provenance.append({k: record[k] for k in ('path', 'sha256', 'bytes')})
        while proc.stdout.read(1024*1024):
            pass
        proc.stdout.close()
        assert proc.wait() == 0
        print('Read and verified', archive, flush=True)
    result = {}
    for key, arrays in blocks.items():
        assert len(arrays) == 4
        q = np.concatenate(arrays)
        result[key] = q[np.lexsort((q[:, 1], q[:, 0]))]
    assert len(result) == 128
    np.savez_compressed(cache, **result)
    (OUT / 'qp_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    return result


def aging(theta, velocity, dt):
    theta, velocity = np.longdouble(theta), np.longdouble(velocity)
    x = velocity*np.longdouble(dt)/np.longdouble(DC)
    return theta*np.exp(-x)-np.longdouble(DC)/velocity*np.expm1(-x)


def main():
    OUT.mkdir(exist_ok=True)
    nodes = {branch: [table(ROOT/branch/f'disturbance_nodes_{step}.csv')
                      for step in STEPS] for branch in ('reference32', *BRANCHES)}
    r0 = nodes['reference32'][0]
    qps = read_qps(r0)
    n = len(r0)
    selected = np.flatnonzero((r0['xd'] >= 15000) & (r0['xd'] <= 18000))
    sites = [int(np.argmin(abs(r0['xd']-x))) for x in (15500,15900,16000,16200,16500,17000,17500)]
    initial = {branch: nodes[branch][0]['Theta_in']-r0['Theta_in'] for branch in BRANCHES}
    inherited = {branch: initial[branch].copy() for branch in BRANCHES}
    accumulated = {branch: np.zeros(n) for branch in BRANCHES}
    growth_integral = np.zeros(n)
    absolute_integral = np.zeros(n)
    rows, weakrows, globalrows = [], [], []
    checks = dict(aging_absolute_error_s=0., split_absolute_error_s=0.,
                  cumulative_absolute_error_s=0., weak_closure_Pa=0.,
                  qp_interpolation_relative_error=0., weak_linearization_relative_error=0.,
                  work_row_measure_relative_error=0., native_weak_row_error_Pa=0.,
                  window_boundary_influence_relative=0.)
    for k, step in enumerate(STEPS):
        r = nodes['reference32'][k]
        dt = float(r['dt'][0])
        qr = qps[f'reference32_{step}']
        j, xi, weight = qr[:,2].astype(int), qr[:,3], qr[:,4]
        left, right = 1-xi, xi
        def interpolate(field):
            return left*field[j]+right*field[j+1]
        def load(field):
            return (np.bincount(j, weights=weight*left*field, minlength=n)
                    + np.bincount(j+1, weights=weight*right*field, minlength=n))
        mass = load(np.ones(len(qr)))
        native_mass = r['mass_diagonal'].copy()
        native_mass[:-1] += r['mass_upper'][:-1]
        native_mass[1:] += r['mass_upper'][:-1]
        checks['work_row_measure_relative_error'] = max(checks['work_row_measure_relative_error'],
            float(np.max(abs(mass[selected]/native_mass[selected]-1))))
        valid = mass > 0
        a = infer_a(qr)
        z = np.log(qr[:,5]/2e-6)+(.6+B*np.log(qr[:,6]*1e-6/DC))/a
        multiplier = np.where(z < np.log(1e6), 1/np.sqrt(1+np.exp(-2*z)), 1.)
        mu_v = a/qr[:,5]*multiplier
        mu_t = mu_v*(B/a)*qr[:,5]/qr[:,6]
        local_R = qr[:,5]*qr[:,6]/DC
        lambda_abs = (B/a-1)*qr[:,5]/DC
        lambda_rel = B/a*qr[:,5]/DC-1/qr[:,6]
        # These weighted averages describe the local heuristic. They are not
        # replacements for the Q1 weak friction operator assembled below.
        avg = {key: np.divide(load(value), mass, out=np.zeros(n), where=valid)
               for key,value in dict(a=a,R=local_R,lambda_abs=lambda_abs,
                                     lambda_rel=lambda_rel).items()}
        growth_integral += avg['lambda_rel']*dt
        absolute_integral += avg['lambda_abs']*dt
        # Friction-only instantaneous closure on the same Q1/QP space. Fix only
        # the two outer window nodes to measured increments, well away from
        # the reported 15--18 km region. No state trajectory is integrated.
        ids = np.flatnonzero(valid)
        lo, hi = ids[0], ids[-1]
        coef = weight*qr[:,8]*mu_v
        diag = (np.bincount(j,weights=coef*left**2,minlength=n)
                +np.bincount(j+1,weights=coef*right**2,minlength=n))
        upper = np.bincount(j,weights=coef*left*right,minlength=n)
        matrix = np.diag(diag[lo+1:hi])+np.diag(upper[lo+1:hi-1],1)+np.diag(upper[lo+1:hi-1],-1)
        for branch in BRANCHES:
            p = nodes[branch][k]
            assert np.array_equal(p['xd'],r['xd']) and np.array_equal(p['dt'],r['dt'])
            assert np.array_equal(p['time'],r['time'])
            if k:
                assert np.array_equal(p['Theta_in'],nodes[branch][k-1]['Theta_out'])
                assert np.array_equal(r['Theta_in'],nodes['reference32'][k-1]['Theta_out'])
            old = p['Theta_in']-r['Theta_in']
            new = p['Theta_out']-r['Theta_out']
            dv = p['V']-r['V']
            used_v = r['V'] if branch == 'state32' else p['V']
            er = np.exp(-r['V']*dt/DC)
            # Exact telescoping with the reference-rate inherited decay:
            # F(Tp,Vp)-F(Tr,Vr)=exp(-xr)(Tp-Tr)+F(Tp,Vp)-F(Tp,Vr).
            velocity_term = np.asarray(aging(p['Theta_in'],used_v,dt)
                                      -aging(p['Theta_in'],r['V'],dt),dtype=float)
            decay = er*old
            inherited[branch] *= er
            accumulated[branch] = er*accumulated[branch]+velocity_term
            checks['aging_absolute_error_s'] = max(checks['aging_absolute_error_s'],float(np.max(abs(aging(p['Theta_in'],used_v,dt)-p['Theta_out']))))
            checks['split_absolute_error_s'] = max(checks['split_absolute_error_s'],float(np.max(abs(new-decay-velocity_term))))
            checks['cumulative_absolute_error_s'] = max(checks['cumulative_absolute_error_s'],float(np.max(abs(new-inherited[branch]-accumulated[branch]))))
            qp = qps[f'{branch}_{step}']
            assert np.array_equal(qp[:,:4],qr[:,:4])
            assert np.max(abs(qp[:,4]/weight-1)) < 1e-12
            for values, record in ((p,qp),(r,qr)):
                for field,column in (('V',5),('Theta_in',6)):
                    error = np.max(abs(interpolate(values[field])/record[:,column]-1))
                    checks['qp_interpolation_relative_error'] = max(checks['qp_interpolation_relative_error'],float(error))
            mu_rate = mu(qp[:,5],qr[:,6],a)
            mu_state = mu(qp[:,5],qp[:,6],a)
            sigma_used = qr[:,8] if branch == 'normal32' else qp[:,8]
            terms = dict(shear=qp[:,7]-qr[:,7], normal=-qp[:,9]*(sigma_used-qr[:,8]),
                         mixture=-qr[:,8]*(qp[:,9]-mu_state),
                         state=-qr[:,8]*(mu_state-mu_rate), rate=-qr[:,8]*(mu_rate-qr[:,9]),
                         damping=-(qp[:,10]-qr[:,10]))
            checks['weak_closure_Pa'] = max(checks['weak_closure_Pa'],float(np.max(abs(sum(terms.values())-(qp[:,11]-qr[:,11])))))
            loads = {key:load(value) for key,value in terms.items()}
            native_difference = p['residual_load']-r['residual_load']
            checks['native_weak_row_error_Pa'] = max(checks['native_weak_row_error_Pa'],
                float(np.max(abs((sum(loads.values())-native_difference)[selected]/mass[selected]))))
            state_linear = load(qr[:,8]*mu_t*interpolate(old))
            rate_linear = load(qr[:,8]*mu_v*interpolate(dv))
            nonlinear_linear_error = np.linalg.norm((loads['state']+loads['rate']+state_linear+rate_linear)[selected])/np.linalg.norm(loads['state'][selected])
            checks['weak_linearization_relative_error'] = max(checks['weak_linearization_relative_error'],float(nonlinear_linear_error))
            rhs = -state_linear[lo+1:hi].copy()
            rhs[0] -= upper[lo]*dv[lo]
            rhs[-1] -= upper[hi-1]*dv[hi]
            predicted_v = dv.copy()
            predicted_v[lo+1:hi] = np.linalg.solve(matrix,rhs)
            mask = np.zeros(n); mask[selected]=1
            norm = lambda value: float(np.sqrt(max(0.,quadratic(mask*value,mask*value,r))))
            zero_boundary = np.zeros(n)
            zero_boundary[lo+1:hi] = np.linalg.solve(matrix,-state_linear[lo+1:hi])
            checks['window_boundary_influence_relative'] = max(checks['window_boundary_influence_relative'],
                norm(predicted_v-zero_boundary)/norm(dv))
            whole_norm = lambda value: float(np.sqrt(max(0.,quadratic(value,value,r))))
            initial_relative = initial[branch]/r0['Theta_in']
            first_dv = nodes[branch][0]['V']-r0['V']
            globalrows.append(dict(branch=branch,step=step,elapsed_years=(k+1)/32,
                                   weak_friction_only_V_relative_difference=norm(predicted_v-dv)/norm(dv),
                                   absolute_state_norm_gain=np.sqrt(quadratic(new,new,r)/quadratic(initial[branch],initial[branch],r)),
                                   relative_state_norm_gain=whole_norm(new/r['Theta_out'])/whole_norm(initial_relative),
                                   frozen_initial_denominator_gain=whole_norm(new/r0['Theta_in'])/whole_norm(initial_relative),
                                   denominator_only_factor=whole_norm(new/r['Theta_out'])/whole_norm(new/r0['Theta_in']),
                                   absolute_velocity_norm_gain=whole_norm(dv)/whole_norm(first_dv),
                                   relative_velocity_norm_gain=whole_norm(dv/r['V'])/whole_norm(first_dv/r0['V'])))
            for i in selected:
                # The taper has exactly zero initial disturbance at its ends;
                # gains there are undefined, not infinities or evidence of growth.
                absolute_gain = new[i]/initial[branch][i] if initial[branch][i] != 0 else np.nan
                row = dict(branch=branch,step=step,node=int(i),xd_m=r['xd'][i],elapsed_years=(k+1)/32,
                           Theta_reference_in_s=r['Theta_in'][i],Theta_reference_out_s=r['Theta_out'][i],
                           delta_Theta_in_s=old[i],delta_Theta_out_s=new[i],
                           relative_Theta_in=old[i]/r['Theta_in'][i],relative_Theta_out=new[i]/r['Theta_out'][i],
                           V_reference=r['V'][i],delta_V=dv[i],relative_V=dv[i]/r['V'][i],
                           inherited_step_s=decay[i],decay_loss_s=decay[i]-old[i],velocity_feedback_s=velocity_term[i],
                           inherited_from_initial_s=inherited[branch][i],accumulated_velocity_feedback_s=accumulated[branch][i],
                           absolute_gain=absolute_gain,
                           denominator_gain=r0['Theta_in'][i]/r['Theta_out'][i],
                           relative_gain=absolute_gain*r0['Theta_in'][i]/r['Theta_out'][i],
                           absolute_growth_this_step=int(abs(new[i])>abs(old[i])),
                           nodal_R_in=r['V'][i]*r['Theta_in'][i]/DC,
                           qp_weighted_R=avg['R'][i],qp_weighted_a=avg['a'][i],
                           lambda_absolute_per_year=avg['lambda_abs'][i]*YEAR,
                           lambda_relative_per_year=avg['lambda_rel'][i]*YEAR,
                           heuristic_absolute_gain=np.exp(absolute_integral[i]),
                           heuristic_relative_gain=np.exp(growth_integral[i]),
                           pointwise_rate_estimate=-B/avg['a'][i]*r['V'][i]/r['Theta_in'][i]*old[i],
                           weak_friction_only_delta_V=predicted_v[i])
                rows.append(row)
                weakrows.append(dict(branch=branch,step=step,node=int(i),xd_m=r['xd'][i],
                                     **{name+'_Pa':value[i]/mass[i] for name,value in loads.items()},
                                     total_Pa=sum(value[i] for value in loads.values())/mass[i],
                                     state_linear_Pa=-state_linear[i]/mass[i],
                                     rate_linear_Pa=-rate_linear[i]/mass[i]))
        print('Analyzed accepted step',step,flush=True)
    assert checks['aging_absolute_error_s'] < 2e-6
    assert checks['split_absolute_error_s'] < 2e-6
    assert checks['cumulative_absolute_error_s'] < 1e-5
    assert checks['weak_closure_Pa'] < 1e-7
    assert checks['qp_interpolation_relative_error'] < 1e-12
    assert checks['weak_linearization_relative_error'] < 1e-3
    assert checks['work_row_measure_relative_error'] < 1e-12
    assert checks['native_weak_row_error_Pa'] < 1e-7
    assert checks['window_boundary_influence_relative'] < 1e-8
    write_csv('node_updates.csv',rows)
    write_csv('weak_friction_rows.csv',weakrows)
    write_csv('norms_and_instantaneous_closure.csv',globalrows)
    selected_rows = [row for row in rows if row['node'] in sites]
    write_csv('selected_locations.csv',selected_rows)
    (OUT/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))
    make_plots(selected_rows,rows)


def make_plots(sites, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    locations = sorted({r['xd_m'] for r in sites})
    fig,axes = plt.subplots(4,2,figsize=(12,13),sharex=True)
    axes.flat[-1].set_visible(False)
    for x,ax in zip(locations,axes.flat):
        data = [r for r in sites if r['xd_m']==x and r['branch']=='plus32']
        t = [r['elapsed_years'] for r in data]
        for key,label in [('absolute_gain','absolute state'),('relative_gain','relative state'),
                          ('denominator_gain','reference denominator'),('heuristic_relative_gain','local relative heuristic')]:
            ax.plot(t,[r[key] for r in data],label=label)
        ax.axhline(1,color='0.5',lw=.7);ax.grid(alpha=.3);ax.set_title(f'{x/1000:.2f} km')
    axes[0,0].legend(fontsize=8)
    for ax in axes[-1]:ax.set_xlabel('Years after accepted step 11')
    fig.tight_layout();fig.savefig(OUT/'state_and_denominator.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(12,13),sharex=True)
    axes.flat[-1].set_visible(False)
    for x,ax in zip(locations,axes.flat):
        for branch,label in [('plus32','full'),('state32','reference V in aging'),('normal32','reference normal')]:
            data=[r for r in sites if r['xd_m']==x and r['branch']==branch]
            ax.plot([r['elapsed_years'] for r in data],[r['delta_V']/1e-14 for r in data],label=label)
        ax.grid(alpha=.3);ax.set_title(f'{x/1000:.2f} km; delta V / 1e-14 m/s')
    axes[0,0].legend(fontsize=8)
    for ax in axes[-1]:ax.set_xlabel('Years after accepted step 11')
    fig.tight_layout();fig.savefig(OUT/'velocity_controls.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True)
    final=sorted([r for r in rows if r['branch']=='plus32' and r['step']==43],key=lambda r:r['xd_m'])
    x=[r['xd_m']/1000 for r in final]
    for key in ('absolute_gain','relative_gain','heuristic_relative_gain'):
        axes[0].plot(x,[r[key] for r in final],label=key)
    for key in ('nodal_R_in','qp_weighted_R'):
        axes[1].plot(x,[r[key] for r in final],label=key)
    for key in ('lambda_absolute_per_year','lambda_relative_per_year'):
        axes[2].plot(x,[r[key] for r in final],label=key)
    for ax in axes:ax.grid(alpha=.3);ax.legend(fontsize=8)
    axes[2].set_xlabel('Down dip (km)');fig.tight_layout()
    fig.savefig(OUT/'spatial_growth.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    main()
