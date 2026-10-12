#!/usr/bin/env python3
"""Continue the unchanged consistent-endpoint prototype to .625 s.

Each partition resumes its own accepted .5625-s FE state. Old files remain
immutable; combined evidence contains the exact original prefix. No physical
or numerical equation is implemented here: use Coupled and path_ledger.
"""
import argparse
import hashlib
import json
import resource
import shutil
import time

import numpy as np

import coupled_branch as core
import assess_coupled_branch as assessment

HERE = core.HERE
ROOT = HERE.parents[2]
PREFIX = core.E
E = HERE / 'coupled_branch_extension_evidence'
TAIL_START = core.FINAL
FINAL = .625
COMMON = .59375  # Last common accepted time; never align different endpoints.
LEVELS = (256, 512, 1024)
read = assessment.read
write = core.write


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    E.mkdir(exist_ok=False)
    files = set(ROOT/name for name in read(PREFIX/'preservation.json'))
    files.update(p for p in PREFIX.rglob('*') if p.is_file())
    files.update(HERE/name for name in ['coupled_branch.py',
                 'assess_coupled_branch.py', 'check_coupled_branch.py',
                 'localization_1d.py', 'total_energy_step.py'])
    files.update(ROOT/'doc/reconstructed_fault/propagation'/name for name in
                 ['P2_coupled_branch_report.md', 'P2_coupled_branch_equations.md'])
    write(E/'preservation.json', {str(p.relative_to(ROOT)): digest(p)
                                 for p in sorted(files)})
    shutil.copyfile(PREFIX/'start.npz', E/'start.npz')
    entries = []
    for d in LEVELS:
        old = PREFIX/f'dt-1-{d}'
        s = read(old/'summary.json'); hist = read(old/'history.json')
        assert s['completed'] and hist[-1]['time'] == TAIL_START
        f = np.load(old/'fields.npz')
        assert len(f['phase']) == round(TAIL_START*d)+1
        p = core.Coupled()
        check = core.state_check(p.r, f['phase'][-1], f['R'][-1],
                                 f['Z'][-1], hist[-1]['V'], f['phase'][-2])
        assert check['phase_KKT'] <= core.TARGET
        assert check['mechanical'] < 1e-7 and check['contact'] < 1e-7
        entries.append(dict(denominator=d, time=TAIL_START,
                            source=str(old.relative_to(ROOT)), conditions=check))
    write(E/'continuation.json', dict(entries=entries, final=FINAL,
          method='same-partition accepted FE state; no checkpoint conversion',
          original_initial_state='fresh unloaded seed, identical across partitions'))


def trajectory(d):
    out = E/f'dt-1-{d}'; out.mkdir(exist_ok=False)
    old = PREFIX/f'dt-1-{d}'; f = np.load(old/'fields.npz')
    rows = read(old/'history.json'); energies = read(old/'energy.json')
    statuses = read(old/'stability.json')
    phases = list(f['phase']); strains = list(f['R']); zs = list(f['Z'])
    us = list(f['velocity'])
    phi = phases[-1].copy(); R = strains[-1].copy(); Z = float(zs[-1])
    slip = rows[-1]['slip']; dt = 1/d; p = core.Coupled()
    started = time.perf_counter(); failures = []; publishes = 0
    for k in range(round((FINAL-TAIL_START)*d)):
        t = TAIL_START+(k+1)*dt
        p.incoming(phi, R, Z, slip, t, dt)
        final, kkt, message = p.solve()
        p.evaluate(final)
        data = p.last[3].copy(); Rb = p.last[4].copy(); Zb = p.last[5]
        u = p.r.velocity.copy()
        assert np.array_equal(phi, p.lower) and np.array_equal(R, p.Ra) and Z == p.Za
        status = core.stability(p, final); status['time'] = t
        if (kkt > core.TARGET or data['force'] > 1e-7 or data['contact'] > 1e-7
                or min(final-phi) < -4.922e-12 or max(final) >= 1
                or status['admissible_negative']):
            failures.append(dict(time=t, KKT=kkt, mechanical=data,
                                 stability=status, message=message))
            np.savez_compressed(out/'failed-trial.npz', phi=final, lower=phi,
                                R=Rb, Z=Zb, Ra=R, Za=Z, velocity=u)
            break
        ledger = core.path_ledger(p.r, phi, final, R, Z, Rb, Zb, u, data['V'], dt)
        ledger.update(time=t, phase_KKT=kkt, phase_increment=float(max(final-phi)))
        # Only this point publishes physical history, once per successful step.
        phi = final.copy(); R = Rb.copy(); Z = Zb; slip = data['slip']
        publishes += 1
        phases.append(phi.copy()); strains.append(R.copy()); zs.append(Z); us.append(u)
        rows.append(dict(time=t, phase_max=float(max(phi)),
                         phase_increment=ledger['phase_increment'], phase_KKT=kkt, **data))
        energies.append(ledger); statuses.append(status)
        write(out/'history.json', rows); write(out/'energy.json', energies)
        write(out/'stability.json', statuses)
        print(json.dumps(dict(d=d, time=t, phase=max(phi), increment=ledger['phase_increment'],
                              KKT=kkt, curvature=status['minimum'],
                              remainder=ledger['external_remainder'])), flush=True)
    np.savez_compressed(out/'fields.npz', phase=np.array(phases), R=np.array(strains),
                        Z=np.array(zs), velocity=np.array(us), y=f['y'], yq=f['yq'],
                        initial=f['initial'])
    prior = read(old/'summary.json')
    summary = dict(denominator=d, dt=dt, start=0., final=FINAL,
                   completed=not failures, accepted=len(rows), added_accepted=publishes,
                   failures=failures, onsets=prior['onsets'], evaluations=p.evaluations,
                   runtime_seconds=time.perf_counter()-started,
                   prefix_runtime_seconds=prior['runtime_seconds'],
                   peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(out/'summary.json', summary); print(json.dumps(summary), flush=True)


def late_verification(d, f, hist, energies):
    p = core.Coupled(); results = []
    # Late growth and worst measured work/phase steps, not just the endpoint.
    tail = [x for x in energies if x['time'] > TAIL_START]
    times = {.5703125, .578125, .5859375, COMMON, hist[-1]['time']}
    for key in ['phase_increment', 'external_remainder']:
        times.add(max(tail, key=lambda x: abs(x[key]))['time'])
    for t in sorted(times):
        j = round(t*d)-1
        pa = f['phase'][j]; pb = f['phase'][j+1]
        p.incoming(pa, f['R'][j], f['Z'][j], hist[j-1]['slip'], t, 1/d)
        value, g = p.evaluate(pb); H = p.hessian(pb)
        ids = np.flatnonzero(pb > pa+1e-13)
        eig, vec = np.linalg.eigh(H[np.ix_(ids, ids)])
        direction = np.zeros(len(pb)); direction[ids] = vec[:, 0]
        derivatives = []
        for epsilon in [1e-4, 1e-5, 1e-6]:
            vp, gp = p.evaluate(pb+epsilon*direction)
            vm, gm = p.evaluate(pb-epsilon*direction)
            fd = (gp-gm)/(2*epsilon)
            derivatives.append(dict(epsilon=epsilon,
                energy_error=float(abs((vp-vm)/(2*epsilon)-g@direction)),
                tangent_error=float(max(abs(fd-H@direction))),
                mode_error=float(abs(direction@fd-eig[0]))))
        assert min(x['tangent_error'] for x in derivatives) < 1e-7
        assert min(x['mode_error'] for x in derivatives) < .01*eig[0]
        correction = np.linalg.solve(H[np.ix_(ids, ids)], -g[ids])
        p.evaluate(pb); data = p.last[3]; u = p.r.velocity.copy()
        ledgers = [core.path_ledger(p.r, pa, pb, p.Ra, p.Za, p.last[4],
                   p.last[5], u, data['V'], 1/d, order) for order in [16, 32]]
        keys = ['external_work', 'bulk_viscous', 'cohesive_viscous',
                'mechanical_power', 'friction_residual_power', 'phase_power',
                'external_remainder', 'mechanical_power_absolute',
                'friction_residual_power_absolute', 'phase_power_absolute']
        uncertainty = max(abs(ledgers[0][key]-ledgers[1][key]) for key in keys)
        assert uncertainty < 1e-10
        results.append(dict(time=t, derivatives=derivatives, reduced_minimum=float(eig[0]),
                            Newton_correction=float(max(abs(correction))),
                            time_quadrature_uncertainty=uncertainty))
    return results


def assess():
    out = E/'assessment'; out.mkdir(exist_ok=False)
    tables = []; results = []; cases = {}
    for d in LEVELS:
        p = E/f'dt-1-{d}'; en = read(p/'energy.json'); hist = read(p/'history.json')
        f = np.load(p/'fields.npz'); old = np.load(PREFIX/f'dt-1-{d}'/'fields.npz')
        n = len(old['phase'])
        for key in ['phase', 'R', 'Z']:
            assert np.array_equal(f[key][:n], old[key]), key
        assert np.array_equal(f['velocity'][:n-1], old['velocity'])
        assert hist[:n-1] == read(PREFIX/f'dt-1-{d}'/'history.json')
        assert en[:n-1] == read(PREFIX/f'dt-1-{d}'/'energy.json')
        summary = read(p/'summary.json')
        assert not summary['completed'] and len(summary['failures']) == 1
        assert summary['failures'][0]['stability']['admissible_negative']
        common = [x for x in en if x['time'] <= COMMON]
        tail = [x for x in common if x['time'] > TAIL_START]
        stable = [x for x in read(p/'stability.json') if x['time'] > TAIL_START]
        assert len(stable) == summary['added_accepted']
        assert all(x['equilibrium'] and x['minimum'] > 0 for x in stable)
        assert max(x['phase_KKT'] for x in hist) <= core.TARGET
        assert max(x['force'] for x in hist) < 1e-7 and max(x['contact'] for x in hist) < 1e-7
        assert max(x['identity_error'] for x in en) < 1e-10
        table = dict(denominator=d, summary=summary, last_accepted=hist[-1],
                     common_state=hist[round(COMMON*d)-1],
                     common_totals=assessment.totals(common),tail_totals=assessment.totals(tail),
                     all_accepted_totals=assessment.totals(en),
                     phase_increment_max=max(x['phase_increment'] for x in hist),
                     endpoint_KKT_max=max(x['phase_KKT'] for x in hist),
                     endpoint_force_max=max(x['force'] for x in hist),
                     endpoint_contact_max=max(x['contact'] for x in hist))
        table['tail_minimum_curvature'] = min(x['minimum'] for x in stable)
        table['tail_weak_max'] = max(x['weak'] for x in stable)
        table['tail_intermediate_max'] = {k:max(x['intermediate_max'][k] for x in tail)
                                          for k in tail[0]['intermediate_max']}
        table['boundary_phase'] = f['phase'][-1][[0,-1]].tolist()
        table['max_phase_location'] = f['y'][np.argmax(f['phase'][-1])].item()
        results.append(dict(denominator=d, checks=late_verification(d,f,hist,en)))
        table['failed_entry'] = diagnose_failed(d)
        tables.append(table); cases[d] = (hist,f,en)
    write(E/'assessment/tables.json', tables)
    write(E/'assessment/late-verification.json', results)
    write(E/'assessment/comparison.json', compare(cases))
    write(E/'assessment/decision.json', dict(complete_interval=False,
          requested_final=FINAL,common_accepted_final=COMMON,failed_entry_time=.59765625,
          numerical_equilibrium_checks='pass',stability_gate='fail',native_adoption=False,P3=False,
          note='All three partitions reach an admissible antisymmetric saddle; no unstable trial published'))
    late_plots()
    for table in tables:
        print(json.dumps(dict(d=table['denominator'], final=table['last_accepted'],
                              tail=table['tail_totals'],
                              curvature=table['tail_minimum_curvature'])), flush=True)


def compare(cases):
    r = core.Reference(256,1.,12); results = []
    for da,db in [(256,512),(512,1024)]:
        ha,fa,_ = cases[da]; hb,fb,_ = cases[db]; rows = []
        previous_a = previous_b = 0.
        for j,a in enumerate(ha):
            t = a['time']
            if t > COMMON: break
            k = round(t*db)-1; b = hb[k]
            pa = fa['phase'][j+1]; pb = fb['phase'][k+1]
            delta = assessment.stress(r,pa,fa['R'][j+1],fa['Z'][j+1])-assessment.stress(r,pb,fb['R'][k+1],fb['Z'][k+1])
            chi = core.h(r.B@pb)/r.direct(pb)[0]
            rows.append(dict(time=t,phase=float(max(abs(pa-pb))),q=abs(a['q']-b['q']),
                slip=abs(a['slip']-b['slip']),bulk=abs(a['bulk']-b['bulk']),
                cohesive=abs(a['cohesive']-b['cohesive']),fracture=abs(a['fracture']-b['fracture']),
                stress_RMS=float(np.sqrt(r.w@(delta*delta))),
                localized_stress_RMS=float(np.sqrt((r.w*chi)@(delta*delta)/(r.w@chi))),
                stress_max=float(max(abs(delta))),
                common_interval_V=abs((a['slip']-previous_a-b['slip']+previous_b)*da)))
            previous_a = a['slip']; previous_b = b['slip']
        keys = list(rows[0])[1:]
        results.append(dict(coarse=da,fine=db,common_final=COMMON,
            maximum={key:max(x[key] for x in rows) for key in keys},
            tail_maximum={key:max(x[key] for x in rows if x['time']>TAIL_START) for key in keys},
            by_time=rows))
    return results


def diagnose_failed(d):
    out = E/f'dt-1-{d}'; s = read(out/'summary.json'); hist = read(out/'history.json')
    trial = np.load(out/'failed-trial.npz'); f = np.load(out/'fields.npz')
    t = s['failures'][0]['time']; p = core.Coupled()
    incoming = core.state_check(p.r,trial['lower'],trial['Ra'],trial['Za'],
                               hist[-1]['V'],f['phase'][-2])
    assert incoming['phase_KKT']<=core.TARGET and incoming['mechanical']<1e-7 and incoming['contact']<1e-7
    p.incoming(trial['lower'],trial['Ra'],trial['Za'],hist[-1]['slip'],t,1/d)
    x = trial['phi']; value,g = p.evaluate(x); H = p.hessian(x)
    assert min(x-p.lower)>=-4.922e-12 and min(x)>=0 and max(x)<1
    assert p.r.defect(g,x,p.lower)<=core.TARGET
    fixed = p.r.phase_hessian(x); ids = np.flatnonzero(x>p.lower+1e-13)
    eig,vec = np.linalg.eigh(H[np.ix_(ids,ids)])
    direction = np.zeros(len(x));direction[ids] = vec[:,0]
    nonzero = abs(direction[ids])>1e-12
    amplitude = min((x-p.lower)[ids][nonzero]/abs(direction[ids][nonzero]))
    derivatives = []
    for epsilon in [1e-5,2e-5,3e-5]:
        assert epsilon < amplitude
        assert min(x-epsilon*abs(direction)-p.lower)>=0 and max(x+epsilon*abs(direction))<1
        vp,gp = p.evaluate(x+epsilon*direction); vm,gm = p.evaluate(x-epsilon*direction)
        derivatives.append(dict(epsilon=epsilon,minimum_FD=float(direction@(gp-gm)/(2*epsilon)),
              mode_error=float(abs(direction@(gp-gm)/(2*epsilon)-eig[0])),
              potential_change_plus=float(vp-value),potential_change_minus=float(vm-value),
              predicted_quadratic_change=float(.5*eig[0]*epsilon**2)))
    uncertainty = min(x['mode_error'] for x in derivatives)
    assert eig[0]<0 and uncertainty<abs(eig[0])*.001
    assert derivatives[-1]['potential_change_plus']<0 and derivatives[-1]['potential_change_minus']<0
    p.evaluate(x); data = p.last[3].copy(); u = p.r.velocity.copy()
    ledgers = [core.path_ledger(p.r,p.lower,x,p.Ra,p.Za,p.last[4],p.last[5],u,data['V'],1/d,n) for n in [16,32]]
    keys = ['external_remainder','mechanical_power','friction_residual_power','phase_power',
            'mechanical_power_absolute','friction_residual_power_absolute','phase_power_absolute']
    quadrature = max(abs(ledgers[0][k]-ledgers[1][k]) for k in keys)
    assert quadrature<1e-10
    # An identical incoming state repeats the saddle: not merely an unlucky trial.
    repeated,kkt,_ = p.solve()
    repeat_error = float(max(abs(repeated-x)))
    assert repeat_error<1e-8 and kkt<=core.TARGET
    result = dict(time=t,incoming_conditions=incoming,endpoint=data,
           endpoint_KKT=s['failures'][0]['KKT'],phase_increment=float(max(x-p.lower)),
           reduced_minimum=float(eig[0]),second_eigenvalue=float(eig[1]),
           fixed_history_minimum=float(np.linalg.eigvalsh(fixed[np.ix_(ids,ids)])[0]),
           mode_uncertainty=uncertainty,derivatives=derivatives,
           antisymmetry_error=float(max(abs(direction+direction[::-1]))),
           mode_peak_y=float(p.r.y[np.argmax(abs(direction))]),
           admissible_amplitude=float(amplitude),identical_entry_repeat_error=repeat_error,
           lower_bound_violation=float(max(0.,max(p.lower-x))),
           strict_lower_gradient_min=float(min(g[x<=p.lower+1e-13])),
           newly_free_y=p.r.y[(x>p.lower+1e-13)&(p.lower<=f['phase'][-2]+1e-13)].tolist(),
           time_quadrature_uncertainty=quadrature,continuous_path=ledgers[-1],published=False)
    np.savez_compressed(out/'negative-mode.npz',y=p.r.y,mode=direction,phase=x,lower=p.lower)
    write(out/'failure-diagnosis.json',result)
    return result


def late_plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,3,figsize=(12,7))
    for d in LEVELS:
        p = E/f'dt-1-{d}'; hist = read(p/'history.json'); en = read(p/'energy.json')
        ids = [j for j,x in enumerate(hist) if x['time'] > TAIL_START]
        t = [hist[j]['time'] for j in ids]
        axes[0,0].plot(t,[hist[j]['phase_max'] for j in ids],label=f'1/{d} s')
        axes[0,1].plot(t,[hist[j]['q'] for j in ids])
        axes[0,2].plot([hist[j]['slip'] for j in ids],[hist[j]['q'] for j in ids])
        for a,key in zip(axes[1],['mechanical_power','friction_residual_power','phase_power']):
            a.plot(t,np.cumsum([en[j][key+'_absolute'] for j in ids]),label=f'1/{d} s')
    for ax,x,y in zip(axes.ravel(),['time [s]','time [s]','slip [m]']+['time [s]']*3,
        ['maximum phase','traction [Pa]','traction [Pa]','absolute mechanical residual work [J/m]',
         'absolute friction residual work [J/m]','absolute phase residual work [J/m]']):
        ax.set_xlabel(x); ax.set_ylabel(y); ax.grid(alpha=.2)
    axes[0,0].legend();fig.tight_layout();fig.savefig(E/'assessment/tail.png',dpi=160);plt.close(fig)
    f = np.load(E/'dt-1-1024/fields.npz'); r = core.Reference(256,1.,12)
    fig, axes = plt.subplots(1,3,figsize=(12,3.3))
    for t in [TAIL_START,.5703125,.578125,.5859375,COMMON]:
        j = round(t*1024); phi = f['phase'][j]
        axes[0].plot(f['y'],phi,label=f'{t:g} s')
        axes[1].plot(f['y'],core.h(phi)/r.direct(phi)[0])
        axes[2].plot(r.yq,assessment.stress(r,phi,f['R'][j],f['Z'][j]))
    for ax,y in zip(axes,['phase','chi [1/m]','stress [Pa]']):
        ax.set_xlabel('y [m]');ax.set_ylabel(y);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(E/'assessment/tail-profiles.png',dpi=160);plt.close(fig)
    fig, axes = plt.subplots(2,3,figsize=(12,7))
    en = read(E/'dt-1-1024/energy.json')
    t = [x['time'] for x in en]
    for key in ['bulk_change','cohesive_change','fracture_change']:
        axes[0,0].plot(t,np.cumsum([x[key] for x in en]),label=key)
    for key in ['bulk_viscous','cohesive_viscous','friction']:
        axes[0,1].plot(t,np.cumsum([x[key] for x in en]),label=key)
    for key in ['mechanical_power','friction_residual_power','phase_power']:
        axes[0,2].plot(t,np.cumsum([x[key] for x in en]),label=key)
    for d in LEVELS:
        p = E/f'dt-1-{d}'; ed = read(p/'energy.json')
        td = [x['time'] for x in ed]
        axes[1,0].plot(td,np.cumsum([x['external_remainder'] for x in ed]),label=f'1/{d} s')
        axes[1,1].plot(td,np.cumsum([abs(x['external_remainder']) for x in ed]),label=f'1/{d} s')
        sd = [x for x in read(p/'stability.json') if x['minimum'] is not None]
        axes[1,2].plot([x['time'] for x in sd],[x['minimum'] for x in sd],label=f'1/{d} s')
        fail = read(p/'failure-diagnosis.json')
        axes[1,2].plot(fail['time'],fail['reduced_minimum'],'rx')
    for ax,y in zip(axes.ravel(),['storage change [J/m]','loss [J/m]',
                  'signed residual work [J/m]','signed external remainder [J/m]',
                  'sum absolute step remainder [J/m]','reduced curvature']):
        ax.set_xlabel('time [s]');ax.set_ylabel(y);ax.grid(alpha=.2);ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(E/'assessment/energy-stability.png',dpi=160);plt.close(fig)
    f = np.load(E/'dt-1-1024/negative-mode.npz')
    fig, axes = plt.subplots(1,2,figsize=(9,3.5))
    axes[0].plot(f['y'],f['lower'],label='accepted .5966796875 s')
    axes[0].plot(f['y'],f['phase'],'--',label='uncommitted .59765625 s')
    axes[1].plot(f['y'],f['mode'],label='admissible unstable mode')
    for ax,y in zip(axes,['phase','unit coefficient eigenvector']):
        ax.set_xlabel('y [m]');ax.set_ylabel(y);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(E/'assessment/negative-mode.png',dpi=160);plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,3.5))
    times=np.arange(round(TAIL_START*256)+1,round(COMMON*256)+1)/256
    for d in LEVELS:
        hist=read(E/f'dt-1-{d}/history.json')
        rates=[(hist[round(t*d)-1]['slip']-hist[round((t-1/256)*d)-1]['slip'])*256 for t in times]
        ax.plot(times-1/512,rates,label=f'1/{d} s')
    ax.set_xlabel('common interval midpoint [s]');ax.set_ylabel('mean slip rate [m/s]')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    fig.savefig(E/'assessment/common-rate.png',dpi=160);plt.close(fig)


def check():
    manifest = read(E/'preservation.json')
    for name,value in manifest.items(): assert digest(ROOT/name)==value,name
    provenance = E/'provenance.json'
    if provenance.exists():
        recorded = read(provenance)
        for group in ['new_sources','dependencies']:
            for name,value in recorded[group].items(): assert digest(ROOT/name)==value,name
    assert not read(E/'assessment/decision.json')['complete_interval']
    normalization = []
    for d in LEVELS:
        p = E/f'dt-1-{d}'; f = np.load(p/'fields.npz'); hist = read(p/'history.json')
        old = np.load(PREFIX/f'dt-1-{d}'/'fields.npz')
        for key in ['phase','R','Z']:
            assert np.array_equal(f[key][:len(old[key])], old[key])
        s = read(p/'summary.json'); assert s['added_accepted']==len(hist)-round(TAIL_START*d)
        assert len(f['phase'])==len(hist)+1 and len(f['velocity'])==len(hist)
        beta = np.exp(-core.G/(d*core.ETA));kappa = -core.ETA*np.expm1(-core.G/(d*core.ETA))
        assert max(abs((f['R'][1:]-beta*f['R'][:-1]-kappa/core.G*f['velocity']).ravel()))<1e-16
        assert max(abs(f['Z'][1:]-beta*f['Z'][:-1]-kappa*np.array([x['V'] for x in hist])))<1e-10
        assert abs(hist[-1]['slip']-sum(x['V']/d for x in hist))<1e-15
        displacement = sum((u[-1]-u[0])/d for u in f['velocity'])
        assert abs(displacement-.128*hist[-1]['time']**2)<1e-15
        assert min(np.diff(f['phase'],axis=0).ravel())>=-4.922e-12
        assert min(f['phase'].ravel())>=0 and max(f['phase'].ravel())<1
        assert all(x['equilibrium'] and not x['admissible_negative'] for x in read(p/'stability.json'))
        failure = read(p/'failure-diagnosis.json')
        trial = np.load(p/'failed-trial.npz')
        assert np.array_equal(trial['Ra'],f['R'][-1])
        assert trial['Za']==f['Z'][-1] and np.array_equal(trial['lower'],f['phase'][-1])
        assert max(abs(trial['R']-beta*trial['Ra']-kappa/core.G*trial['velocity']))<1e-16
        assert abs(trial['Z']-beta*trial['Za']-kappa*failure['endpoint']['V'])<1e-10
        assert not failure['published'] and failure['reduced_minimum']<0
        assert failure['mode_uncertainty']<abs(failure['reduced_minimum'])*.001
        assert max(x['identity_error'] for x in read(p/'energy.json'))<1e-10
        assert failure['continuous_path']['identity_error']<1e-10
        r = core.Reference(256,1.,6); norm = []
        for phi in [f['phase'][0],f['phase'][-1],np.load(p/'failed-trial.npz')['phi']]:
            I,J = r.direct(phi); estimates = []
            for order in [12,24]:
                z,w = np.polynomial.legendre.leggauss(order);theta=(z+1)/2
                val=phi[:-1,None]*(1-theta)+phi[1:,None]*theta
                integral=float(np.sum(core.h(val)*w*r.dy/2))
                left=np.sum(core.dh(val)*(1-theta)*w*r.dy/2,axis=1)
                right=np.sum(core.dh(val)*theta*w*r.dy/2,axis=1)
                jac=np.r_[left,0.]+np.r_[0.,right]
                estimates.append(dict(order=order,I_relative_error=abs(integral-I)/I,
                    derivative_relative_error=float(max(abs(jac-J))/max(abs(J)))))
            assert I>0 and estimates[-1]['I_relative_error']<1e-11
            assert estimates[-1]['derivative_relative_error']<1e-11
            norm.append(dict(I=I,independent_integrals=estimates))
        normalization.append(dict(denominator=d,states=['initial','last accepted','failed entry'],checks=norm))
    for case in read(E/'assessment/late-verification.json'):
        for v in case['checks']:
            assert v['reduced_minimum'] > 0 and v['time_quadrature_uncertainty'] < 1e-10
    # Read-only verifier; the command log retains independent integral estimates.
    print(json.dumps(dict(accepted_numerical_checks='pass',stability_gate='fail',
                         requested_final=FINAL,common_accepted_final=COMMON,
                         normalization=normalization,preserved_files=len(manifest),
                         production_unchanged=True,native_adoption=False,P3=False)),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','run','assess','check'])
    parser.add_argument('--denominator',type=int,choices=LEVELS,default=256)
    args = parser.parse_args()
    if args.action == 'run': trajectory(args.denominator)
    else: globals()[args.action]()
