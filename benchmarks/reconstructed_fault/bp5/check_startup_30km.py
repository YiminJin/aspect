"""Accepted-state lifecycle, matched-timestep and same-rank restart checks."""
import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
import numpy as np
from startup_30km import STUDY
from analyze_length_coupled import raw_qps
from check_first_cycle_restart import convergence, difference, load_parts


def table(path):return np.atleast_1d(np.genfromtxt(path,names=True,delimiter=','))


def check(kind, accepted_prefix=False, preceding_state=None, initialized_path=None):
    path=STUDY/kind
    clock=table(path/'accepted_steps.csv')
    if accepted_prefix:
        # Never report a failed or still-running solve as converged. This
        # separately labelled evidence checks ONLY published accepted states.
        chunks=re.split(r'(\*\*\* Timestep \d+:)',(path/'run.log').read_text())
        accepted=set(clock['step'].astype(int));text=chunks[0]
        for i in range(1,len(chunks),2):
            if int(re.search(r'\d+',chunks[i]).group()) in accepted:text+=''.join(chunks[i:i+2])
        logfile=path/'accepted_prefix.log';logfile.write_text(text)
        conv=convergence(logfile)
    else:
        assert json.loads((path/'execution.json').read_text())['passed']
        conv=convergence(path/'run.log')
        assert clock[-1]['time']==16e6
    assert np.all(clock['fresh_linear_checks_passed']==1)
    assert np.all(clock['free']+clock['lower_active']==1156)
    assert max(clock['Theta_relative_error'])<1e-12
    rows={};previous=preceding_state
    if kind=='restart':previous=table(STUDY/'startup/state_work_2.csv')
    for k in conv:
        s=table(path/f'state_work_{k}.csv');w=table(path/f'work_weak_{k}.csv')
        assert np.min(s['V'])>=1e-20 and np.min(s['Theta_out'])>0
        assert len(s)==1156
        if k==0:
            init=table(path/'weak_initialization.csv')
            np.testing.assert_array_equal(s['Theta_in'],init['Theta_weak'])
            np.testing.assert_array_equal(s['Theta_in'],s['Theta_out'])
            assert max(abs(init['weight']/w['weight']-1))<1e-12
            assert max(abs(init['weak_friction_excess_Pa']))<1e-5
            assert clock[0]['max_committed_stress_Pa']==0
        if previous is not None:
            np.testing.assert_array_equal(s['Theta_in'],previous['Theta_out'])
            dt=float(s['dt'][0]);x=s['V']*dt/.1
            expected=s['Theta_in']*np.exp(-x)-.1/s['V']*np.expm1(-x)
            assert max(abs(s['Theta_out']/expected-1))<1e-12
            expected_slip=np.array([math.fma(dt,float(v),float(old)) for v,old in zip(s['V'],previous['slip'])])
            np.testing.assert_array_equal(s['slip'],expected_slip)
        previous=s
        raw=np.concatenate([raw_qps(p) for p in sorted(path.glob(f'work_qp_{k}_rank*.csv'))])
        active=raw['source_active']==1
        assert np.min(raw['V'][active])>=1e-20
        assert len(set(zip(raw['cell'],raw['qp'])))==len(raw)
        j=raw['segment'].astype(int);xi=raw['xi'];weights=raw['JxW']*raw['chi']
        assembled=np.zeros((len(w),3))
        for end,basis in [(0,1-xi),(1,xi)]:
            for c,field in enumerate([np.ones(len(raw)),raw['q'],raw['sigma_n']]):
                np.add.at(assembled[:,c],j+end,weights*basis*field)
        covered=((s['xd']>=20000)&(s['xd']<=42000))|(s['xd']<1500)|(s['xd']>100000/np.sqrt(.75)-1500)
        assert max(abs(assembled[covered,0]/w['weight'][covered]-1))<1e-11
        for c,key in [(1,'q'),(2,'sigma')]:
            assert max(abs(assembled[covered,c]-w[key][covered])/w['weight'][covered])<1e-5
        rows[k]=dict(time=float(s['time'][0]),min_V=float(min(s['V'])),max_V=float(max(s['V'])),
            minimum_QP_V=float(min(raw['V'][active])),nodes_at_bound=int(sum(s['V']==1e-20)),
            max_Vdt_Dc=float(max(s['V']*s['dt']/.1)),sigma_range=[float(min(raw['sigma_n'][active])),float(max(raw['sigma_n'][active]))])
    # The regenerated input must produce the intended projected plateau/transition.
    if kind!='restart':
        initial_path=path if initialized_path is None else initialized_path
        arrays={a.attrib['Name']:np.fromstring(a.text,sep=' ') for a in ET.parse(
            initial_path/'reconstructed_faults/reconstructed_faults-00000.vtu').findall('.//PointData/DataArray')}
        initial=table(initial_path/'state_work_0.csv');f=arrays['composition_strengthening']
        assert max(abs(f[(initial['xd']>1000)&(initial['xd']<29000)]))<1e-8
        assert max(abs(f[(initial['xd']>34000)&(initial['xd']<110000)]-1))<1e-8
    result=dict(passed=True,scope='accepted prefix only' if accepted_prefix else 'complete bounded run',convergence=conv,states=rows)
    (path/('accepted_prefix_checks.json' if accepted_prefix else 'checks.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result


def restart():
    check('startup');check('restart');a=STUDY/'startup';b=STUDY/'restart';result={}
    ca,cb=table(a/'accepted_steps.csv'),table(b/'accepted_steps.csv')
    for key in ['step','time','dt','free','lower_active']:np.testing.assert_array_equal(ca[key],cb[key])
    for k in [3,4]:
        entry={}
        for pattern in [f'audit_bulk_{k}_rank*.csv',f'audit_particles_{k}_rank*.csv']:
            x,y=load_parts(a,pattern),load_parts(b,pattern)
            np.testing.assert_array_equal(x[:,0],y[:,0])
            if 'bulk' in pattern:
                np.testing.assert_array_equal(x[:,1],y[:,1])
                entry[pattern]={str(int(c)):difference(x[x[:,1]==c,2],y[y[:,1]==c,2]) for c in np.unique(x[:,1])}
            else:entry[pattern]={str(c):difference(x[:,c],y[:,c]) for c in range(1,x.shape[1])}
        x,y=table(a/f'state_work_{k}.csv'),table(b/f'state_work_{k}.csv')
        entry['fault']={key:difference(x[key],y[key]) for key in ['V','Theta_in','Theta_out','slip','weak_q','weak_sigma','weak_friction']}
        entry['weak_residual_absolute_difference']=float(max(abs(x['weak_residual']-y['weak_residual'])))
        def raw(path):
            d=np.concatenate([raw_qps(p) for p in sorted(path.glob(f'work_qp_{k}_rank*.csv'))]);return np.sort(d,order=['cell','qp'])
        x,y=raw(a),raw(b)
        for key in ['cell','qp','x','y','source_active','segment','xi']:np.testing.assert_array_equal(x[key],y[key])
        entry['constitutive']={key:difference(x[key],y[key]) for key in ['phi','Ih','chi','V','p','tau_xx','tau_yy','tau_xy','sigma_n','q']}
        result[k]=entry
    (STUDY/'restart_comparison.json').write_text(json.dumps(dict(passed=True,steps=result),indent=2)+'\n')


def temporal():
    check('startup');check('half');a=STUDY/'startup';b=STUDY/'half'
    ca,cb=table(a/'accepted_steps.csv'),table(b/'accepted_steps.csv');results=[]
    for row in ca:
        matches=cb[abs(cb['time']-row['time'])<1e-7];assert len(matches)==1
        ka,kb=int(row['step']),int(matches[0]['step'])
        x,y=table(a/f'state_work_{ka}.csv'),table(b/f'state_work_{kb}.csv')
        wa,wb=table(a/f'work_weak_{ka}.csv'),table(b/f'work_weak_{kb}.csv')
        for name,lo,hi in [('weakening',0,30000),('transition',28000,35000),('strengthening',35000,110000)]:
            mask=(x['xd']>=lo)&(x['xd']<=hi)
            record=dict(time=float(row['time']),region=name)
            fields={k:(x[k],y[k]) for k in ['V','Theta_out','slip']}
            fields.update({k:(wa[k]/wa['weight'],wb[k]/wb['weight']) for k in ['q','sigma']})
            fields.update(delta_q=((wa['q']-wa['bg'])/wa['weight'],(wb['q']-wb['bg'])/wb['weight']),
                          delta_sigma=(wa['sigma']/wa['weight']-50e6,wb['sigma']/wb['weight']-50e6))
            for field,(u,v) in fields.items():
                error=u[mask]-v[mask];weight=wa['weight'][mask];scale=max(abs(v[mask]))
                record[field]=dict(max_absolute=float(max(abs(error))),relative_max=float(max(abs(error))/scale) if scale else 0.,
                    rms=float(np.sqrt(np.dot(weight,error**2)/sum(weight))))
            results.append(record)
    (STUDY/'timestep_comparison.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['startup','half','restart','temporal'])
    parser.add_argument('--accepted-prefix',action='store_true');a=parser.parse_args()
    if a.action=='restart':restart()
    elif a.action=='temporal':temporal()
    else:check(a.action,a.accepted_prefix)
