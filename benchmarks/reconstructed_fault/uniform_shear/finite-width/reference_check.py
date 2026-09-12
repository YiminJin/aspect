"""K4.1 only: independent fixed-profile references, no ASPECT execution.

Reuse K3's independent initialization and pointwise scalar mechanics. Freeze
the initialized H/phi thereafter (K1), retaining timestep-zero histories.
Reference, temporal and between-width differences are separate quantities.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REF_PATH = HERE.parent/'evolving/reference.py'
spec = importlib.util.spec_from_file_location('k3_reference', REF_PATH)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
PARAMETERS = HERE.parent/'residual-floor/convergence/space32_dt05/parameters.prm'
SCALES = dict(V=1e-4, q=1500., C=1500., Theta=200., slip=6e-4,
              velocity=1e-4, crack_integral=1e-4)
Y = np.linspace(-.5, .5, 8193)


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def initialize(ell, cells, output):
    start = time.monotonic()
    params = reference.read_parameters(PARAMETERS)
    params['Phase field model/Length scale'] = str(ell)
    model = reference.Reference(params, cells, 6)
    phi, iterations = model.phase(model.H0, model.stationary(model.y))
    g, h, integral = model.localization(phi)
    C = float(np.sum(model.w*model.admitted*g*np.sqrt(2*model.G*model.H0)) /
              np.sum(model.w*model.admitted))
    # Split at phase knots, profile/activation boundaries and diagnostic
    # sample locations. Compare 6/8-point integration of this SAME P1 field.
    nodes = np.unique(np.r_[Y, model.y])
    dy = np.diff(nodes)
    sums = []
    for order in (6, 8):
        points, weights = np.polynomial.legendre.leggauss(order)
        qy = nodes[:-1, None]+dy[:, None]*(1+points)/2
        values = np.maximum(np.interp(qy, model.y, phi), 0)
        _, qh = model.degradation(values)
        w = dy[:, None]*weights/2
        panel = np.sum(w*qh, axis=1)
        sums.append(float(np.sum(panel)))
    assert abs(sums[1]-sums[0]) <= 1e-9 and abs(sums[1]/sums[0]-1) <= 1e-10
    assert abs(sums[1]-integral) <= 1e-9
    primitive = np.r_[0., np.cumsum(panel)]
    normalized_primitive = np.interp(Y, nodes, primitive)/integral
    omitted = float(np.sum(w*qh*(abs(qy)>model.support))/integral)
    second_moment = float(np.sum(w*qh*qy*qy)/integral)
    initial = dict(q=1500., C=C, Theta=200., Ih=integral)
    response, _, _ = model.mechanics(initial, 2., 1e-4, integral)
    assert initial == dict(q=1500., C=C, Theta=200., Ih=integral)
    evaluation_phi = np.interp(Y, model.y, phi)
    prescribed = model.stationary(Y)
    init_g, _ = model.degradation(prescribed)
    _, hc = model.degradation(model.core)
    initial_H = np.where(prescribed>model.activation,
                         model.Ec*model.core/(hc*init_g**2), model.Hc)
    report = dict(ell=ell, nominal_cells=cells, actual_cells=len(model.dy),
                  activation=model.activation, m=model.m, Hc=model.Hc, Ec=model.Ec,
                  support_half_width_m=model.support, activation_distance_m=model.cutoff,
                  retained_initial=initial, evaluated_initial=response,
                  phi_range=[float(min(phi)),float(max(phi))], H0_max=float(max(initial_H)),
                  Ih=integral, omitted_fraction=omitted,
                  localization_second_moment_m2=second_moment,
                  localization_rms_width_m=math.sqrt(second_moment),
                  phase_residual_history=iterations,
                  fresh_phase_residual=float(np.linalg.norm(model.residual(phi,model.H0))),
                  same_profile_quadrature_difference_m=abs(sums[1]-sums[0]),
                  guards_pass=bool(min(phi)>=-1e-4 and max(phi)<.8 and omitted<=1e-4),
                  seconds=time.monotonic()-start,
                  parameters=params, parameters_source=str(PARAMETERS.relative_to(ROOT)),
                  parameters_sha256=hashlib.sha256(PARAMETERS.read_bytes()).hexdigest(),
                  reference_sha256=hashlib.sha256(REF_PATH.read_bytes()).hexdigest())
    output.mkdir()
    np.savez(output/'profile.npz', phi=evaluation_phi, H=initial_H,
             primitive=normalized_primitive, nodes=model.y, nodal_phi=phi)
    np.savetxt(output/'profile.csv', np.c_[Y,Y/ell,evaluation_phi,initial_H,normalized_primitive],
               delimiter=',',header='y,y_over_ell,phi,H0,integrated_h_over_Ih',comments='')
    dump(output/'initialization.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('parameters','phase_residual_history')}),flush=True)
    return report


def trajectory(directory, dt):
    data = json.loads((directory/'initialization.json').read_text())
    profile = np.load(directory/'profile.npz')
    # This creates the independent constitutive coefficients only; phase is
    # never re-solved in the fixed-profile timestep sequence.
    model = reference.Reference(data['parameters'], 16, 6)
    old = data['retained_initial'].copy()
    I = old['Ih']
    values, velocities = [], []
    slip = 0.
    for step in range(round(6/dt)+1):
        t = step*dt
        U = 1e-4*(1+.2*min(t/4,1))
        previous_q = old['q']
        r,beta,kappa = model.mechanics(old,dt if step else 2.,U,I)
        if step:
            decay = math.exp(-r['V']*dt/model.Dc)
            theta = old['Theta']*decay+model.Dc/r['V']*(-math.expm1(-r['V']*dt/model.Dc))
            slip += dt*r['V']
            old = dict(q=r['q'],C=r['C'],Theta=theta,Ih=I)
        else:
            theta = old['Theta']  # evaluated q/C are NOT retained histories
        bulk_shear = (r['q']-beta*previous_q)/kappa
        u = -U/2+bulk_shear*(Y+.5)+r['V']*profile['primitive']
        full_integral = r['V']*float(profile['primitive'][-1])
        supported = r['V']*(1-data['omitted_fraction'])
        assert abs(full_integral-r['V'])/max(abs(r['V']),1e-4) <= 1e-12
        assert abs(u[-1]-U/2)<1e-12 and abs(u[0]+U/2)<1e-12
        values.append(dict(step=step,time_s=t,U=U,**r,Theta=theta,slip=slip,
                           crack_integral=full_integral, supported_integral=supported,
                           history_integral=0.,
                           supported_normalization_error=abs(supported-r['V'])/max(abs(r['V']),1e-4)))
        velocities.append(u)
    assert all(s['supported_normalization_error']<=1e-4 for s in values)
    dump(directory/f'dt{dt:g}.json',values)
    np.savez(directory/f'dt{dt:g}.npz',velocity=velocities)
    return values, np.asarray(velocities)


def compare(a,b,ua,ub):
    rows = []
    lookup={s['time_s']:i for i,s in enumerate(b)}
    for i,s in enumerate(a):
        j=lookup[s['time_s']]
        ref=b[j]
        metrics={}
        for key,scale in SCALES.items():
            delta = float(np.max(abs(ua[i]-ub[j]))) if key=='velocity' else abs(s[key]-ref[key])
            size = float(np.max(abs(ub[j]))) if key=='velocity' else abs(ref[key])
            target=.25*(.002*size+1e-5*scale)
            metrics[key]=dict(difference=delta,quarter_allowance=target,ratio=delta/target)
        l2=float(np.sqrt(np.trapezoid((ua[i]-ub[j])**2,Y)))
        l2_ref=float(np.sqrt(np.trapezoid(ub[j]**2,Y)))
        target=.25*(.002*l2_ref+1e-5*SCALES['velocity'])
        metrics['velocity_L2']=dict(difference=l2,quarter_allowance=target,ratio=l2/target)
        rows.append(dict(time_s=s['time_s'],metrics=metrics))
    return dict(rows=rows,max_ratio=max(m['ratio'] for r in rows for m in r['metrics'].values()),
                maxima={k:max(r['metrics'][k]['difference'] for r in rows) for k in rows[0]['metrics']})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists(), 'Preserve prior evidence.'
    args.output.mkdir(parents=True)
    start=time.monotonic()
    initial={}
    for name,ell in [('ell0',.15625),('half',.078125)]:
        for cells in (2048,4096):
            path=args.output/f'{name}-{cells}'
            initial[name,cells]=initialize(ell,cells,path)
            if not initial[name,cells]['guards_pass']:
                dump(args.output/'decision.json',dict(ready=False,stop='initial support/admissibility',case=path.name))
                return
    spatial,temporal={},{}
    for name in ('ell0','half'):
        low=args.output/f'{name}-2048'; high=args.output/f'{name}-4096'
        a,ua=trajectory(low,.5); b,ub=trajectory(high,.5)
        spatial[name]=compare(a,b,ua,ub)
        spatial[name]['phi_max_difference']=float(np.max(abs(np.load(low/'profile.npz')['phi']-np.load(high/'profile.npz')['phi'])))
        c,uc=trajectory(high,.25)
        temporal[name]=dict(dt05=compare(b,c,ub,uc))
    selected=None
    if all(s['max_ratio']<=1 for s in spatial.values()):
        if all(t['dt05']['max_ratio']<=1 for t in temporal.values()):
            selected=.5
        else:
            for name in ('ell0','half'):
                high=args.output/f'{name}-4096'
                c=json.loads((high/'dt0.25.json').read_text())
                uc=np.load(high/'dt0.25.npz')['velocity']
                d,ud=trajectory(high,.125)
                temporal[name]['dt025']=compare(c,d,uc,ud)
            if all(t['dt025']['max_ratio']<=1 for t in temporal.values()): selected=.25
    report=dict(ready=selected is not None,selected_common_dt_s=selected,
                spatial=spatial,temporal=temporal,elapsed_seconds=time.monotonic()-start,
                stop=None if selected else 'Reference resolution or the approved timestep checks do not meet the quarter-allowance criterion.')
    dump(args.output/'decision.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('spatial','temporal')}),flush=True)


if __name__=='__main__': main()
