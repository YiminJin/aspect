"""Same-incoming-history A/B: velocity notch and exact incident-element work."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from analyze_uniform_sliding import read,cat,records

HERE=Path(__file__).resolve().parent
ROOT=HERE/'within-step-50-local4'
BASE=HERE/'work-replay-50-local4'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('case',choices=['A','B']);case=parser.parse_args().case
    out=ROOT/case
    assert json.loads((out/'execution.json').read_text())['verified']
    d=read(out/'noncommitting_surface.csv');old=read(BASE/'fault_9.csv');reference=read(BASE/'fault_10.csv')
    assert len(d['V'])==len(old['V'])==1236
    np.testing.assert_array_equal(d['x'],old['x']);np.testing.assert_array_equal(d['y'],old['y'])
    retained=read(out/'noncommitting_history.csv')
    np.testing.assert_array_equal(retained['Theta_retained'],old['Theta'])
    np.testing.assert_array_equal(retained['V_committed'],old['V'])
    if case=='A':
        assert not np.any(d['lower_active'])
        np.testing.assert_allclose(d['V'],reference['V'],rtol=1e-8,atol=1e-20)
    else:
        for rank in range(4):assert (ROOT/'A'/f'incoming_rank{rank}.txt').read_bytes()==(out/f'incoming_rank{rank}.txt').read_bytes()
    dt=float(reference['dt'][0]);predicted_slip=old['slip']+dt*d['V']
    x=d['V']*dt/.008
    candidate=old['Theta']*np.exp(-x)-.008/d['V']*np.expm1(-x)
    raw=cat(out.glob('state_qp_rank*.csv'),('cell',));j=raw['segment'].astype(int);z=raw['xi']
    theta=candidate if case=='B' else old['Theta']
    np.testing.assert_allclose(raw['V'],(1-z)*d['V'][j]+z*d['V'][j+1],rtol=2e-14,atol=1e-25)
    np.testing.assert_allclose(raw['Theta'],(1-z)*theta[j]+z*theta[j+1],rtol=2e-14,atol=1e-12)
    # Both incident elements are fully VS. Independently reproduce each raw
    # friction value and weak residual from the actual work quadrature.
    mu=.025*np.arcsinh(raw['V']/(2e-6)*np.exp((.6+.015*np.log(raw['Theta']*1e-6/.008))/.025))
    np.testing.assert_allclose(raw['friction'],mu*raw['sigma'],rtol=1e-13,atol=1e-7)
    np.testing.assert_allclose(raw['R'],raw['q']-raw['friction']-raw['damping'],rtol=0,atol=1e-7)
    opposite=candidate if case=='A' else old['Theta']
    opposite_theta=(1-z)*opposite[j]+z*opposite[j+1]
    opposite_mu=.025*np.arcsinh(raw['V']/(2e-6)*np.exp((.6+.015*np.log(opposite_theta*1e-6/.008))/.025))
    opposite_R=raw['q']-opposite_mu*raw['sigma']-raw['damping']
    rows=[]
    node=796
    # Frozen-bulk probe of a smooth last-free nodal rate, not another solve.
    # Recover the affine shear-rate coefficient from the exported exact fixed-
    # state tangent; S:N=0 leaves normal traction unchanged in this probe.
    no_notch=d['V'].copy();no_notch[node]=.5*(d['V'][795]+d['V'][797])
    probe_v=(1-z)*no_notch[j]+z*no_notch[j+1]
    probe_state=old['Theta'].copy()
    if case=='B':
        probe_x=no_notch*dt/.008
        probe_state=old['Theta']*np.exp(-probe_x)-.008/no_notch*np.expm1(-probe_x)
    probe_theta=(1-z)*probe_state[j]+z*probe_state[j+1]
    Z=raw['V']/(2e-6)*np.exp((.6+.015*np.log(raw['Theta']*1e-6/.008))/.025)
    mu_V=.025/raw['V']*np.where(Z<1e6,Z/np.sqrt(1+Z*Z),1.)
    damping=raw['damping']/raw['V']
    shear_slope=raw['Kfixed']-raw['sigma']*mu_V-damping
    probe_q=raw['q']-shear_slope*(probe_v-raw['V'])
    probe_mu=.025*np.arcsinh(probe_v/(2e-6)*np.exp((.6+.015*np.log(probe_theta*1e-6/.008))/.025))
    probe_R=probe_q-probe_mu*raw['sigma']-damping*probe_v
    for segment in (795,796):
        mask=j==segment;weight=raw['weight'][mask]*(z[mask] if node==segment+1 else 1-z[mask])
        r=dict(case=case,node=node,segment=segment,xd0=float(old['xd'][segment]),xd1=float(old['xd'][segment+1]),samples=int(sum(mask)),row_mass=float(sum(weight)))
        for key in ('q','friction','damping','sigma','R'):
            r[key+'_load']=float(np.dot(weight,raw[key][mask]));r[key+'_mean']=r[key+'_load']/r['row_mass']
        r['no_notch_R_load']=float(np.dot(weight,probe_R[mask]))
        r['no_notch_R_mean']=r['no_notch_R_load']/r['row_mass']
        r['opposite_state_R_load']=float(np.dot(weight,opposite_R[mask]))
        r['opposite_state_R_mean']=r['opposite_state_R_load']/r['row_mass']
        rows.append(r)
    for field,weak in [('q','weak_q'),('sigma','weak_sigma'),('R','weak_R')]:
        actual=sum(r[field+'_load'] for r in rows)
        assert abs(actual-d[weak][node])<1e-3,(field,actual,d[weak][node])
    records(out/'element_balance.csv',rows)
    metrics=[]
    for i in (795,796,797,798):
        metrics.append(dict(case=case,node=i,xd=float(old['xd'][i]),V=float(d['V'][i]),old_Theta=float(old['Theta'][i]),candidate_Theta=float(candidate[i]),
            old_slip=float(old['slip'][i]),predicted_slip=float(predicted_slip[i]),prescribed=int(d['prescribed'][i]),lower_active=int(d['lower_active'][i])))
    records(out/'nodes.csv',metrics)
    a,b=795,796;length=old['xd'][a]-old['xd'][b]
    result=dict(case=case,baseline_reproduced=case=='A',time=float(d['time'][0]),dt=dt,
        lower_active=int(sum(d['lower_active'])),
        contrast=float((d['V'][797]-d['V'][796])/1e-9),deficit=float(1-d['V'][796]/1e-9),
        gradient_increment=float(dt*(d['V'][a]-d['V'][b])/length),
        old_gradient=float((old['slip'][a]-old['slip'][b])/length),
        predicted_gradient=float((predicted_slip[a]-predicted_slip[b])/length),
        maximum_A_rate_relative_difference=float(max(abs(d['V']/reference['V']-1))) if case=='A' else None,
        weak_friction_mean=sum(r['friction_load'] for r in rows)/sum(r['row_mass'] for r in rows),
        weak_residual_mean=float(d['weak_R'][node]/sum(r['row_mass'] for r in rows)),
        no_notch_V=float(no_notch[node]),
        no_notch_residual_mean=sum(r['no_notch_R_load'] for r in rows)/sum(r['row_mass'] for r in rows),
        opposite_state_residual_mean=sum(r['opposite_state_R_load'] for r in rows)/sum(r['row_mass'] for r in rows),
        raw_sigma_min=float(min(raw['sigma'])),raw_sigma_max=float(max(raw['sigma'])),
        elements=rows,nodes=metrics)
    (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
