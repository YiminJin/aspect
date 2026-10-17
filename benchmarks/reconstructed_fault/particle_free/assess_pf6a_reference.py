#!/usr/bin/env python3
"""Common-time, common-functional evolving and frozen spatial measurements."""
import argparse
import json
from pathlib import Path
import numpy as np
from pf6a_spatial_reference import Endpoint, SEED, G, A, ELL, h, write, PROP, evaluate_reference

TARGETS = dict(whole_stress=.01, localized_stress=.02, phase_L2=.01,
               phase_Linf=.005, q=.01, slip=.01, energy=.01, driving=.02)


def pullback(y, n=64):
    j=np.minimum((y*n).astype(int),n-1);s=y*n-j
    P=np.zeros((len(y),n+1));P[np.arange(len(y)),j]=1-s;P[np.arange(len(y)),j+1]=s
    return P


def load(path):
    return dict(path=path,summary=json.loads((path/'summary.json').read_text()),
                fields=dict(np.load(path/'fields.npz')),history=json.loads((path/'history.json').read_text()))


def sample(case, y):
    f=case['fields'];phase=np.array([np.interp(y,f['y'],p) for p in f['phase']])
    I=np.r_[Endpoint(len(f['y'])-1).direct(f['phase'][0])[0], [r['I']for r in case['history']]]
    chi=h(phase)/I[:,None]
    lam=np.r_[0., [r['tau_mean']for r in case['history']]]
    nm=case['summary']['mechanical_cells']
    if nm:
        j=np.minimum((y*nm).astype(int),nm-1);xi=2*(y*nm-j)-1
        coef=f['coefficients'];T=coef[:,j,0]+coef[:,j,1]*xi
        stress=T-f['Z'][:,None]*chi
    else:stress=np.broadcast_to(lam[:,None],phase.shape)
    c=np.minimum((y*(len(f['y'])-1)).astype(int),len(f['y'])-2)
    gradient=np.diff(f['phase'],axis=1)[:,c]*(len(f['y'])-1)
    return phase,chi,stress,gradient


def comparison(a,b):
    fa,fb=a['fields'],b['fields'];ta,tb=fa['time'],fb['time']
    indices=np.rint(ta*b['summary']['divisor']).astype(int)
    assert max(abs(tb[indices]-ta))<1e-14
    z,w=np.polynomial.legendre.leggauss(8);y=(np.arange(1024)[:,None]+(z+1)/2).ravel()/1024;w=np.tile(w/(2*1024),1024)
    pa,ca,sa,da=sample(a,y);pb,cb,sb,db=sample(b,y)
    pb,cb,sb,db=[x[indices]for x in [pb,cb,sb,db]]
    stress_scale=max(1.,np.max(np.sqrt(2*np.sum(w*sb*sb,axis=1))))
    phase_scale=max(np.sqrt(np.sum(w*pb*pb,axis=1)))
    q1=np.r_[0.,[r['q']for r in a['history']]];q2=np.r_[0.,[r['q']for r in b['history']]][indices]
    e1=np.r_[A*np.sum(np.diff(fa['y'])*(fa['phase'][0,:-1]+fa['phase'][0,1:])/2)+A*ELL**2*np.sum(np.diff(fa['phase'][0])**2/np.diff(fa['y'])),[r['bulk']+r['cohesive']+r['fracture']for r in a['history']]]
    e2=np.r_[A*np.sum(np.diff(fb['y'])*(fb['phase'][0,:-1]+fb['phase'][0,1:])/2)+A*ELL**2*np.sum(np.diff(fb['phase'][0])**2/np.diff(fb['y'])),[r['bulk']+r['cohesive']+r['fracture']for r in b['history']]][indices]
    ga=fa['driving']@pullback(fa['y']);gb=(fb['driving']@pullback(fb['y']))[indices]
    driving_scale=max(1e-30,np.max(abs(gb[:,0]+gb[:,1])))
    mass=np.full(65,1/64);mass[[0,-1]]*=.5
    whole=np.sqrt(2*np.sum(w*(sa-sb)**2,axis=1));localized=np.sqrt(2*np.sum(w*cb*(sa-sb)**2,axis=1)/np.sum(w*cb,axis=1))
    curves=dict(whole_stress=whole/stress_scale,localized_stress=localized/stress_scale,
                phase_L2=np.sqrt(np.sum(w*(pa-pb)**2,axis=1))/phase_scale,
                phase_Linf=np.max(abs(np.array([np.interp(np.union1d(fa['y'],fb['y']),fa['y'],p) for p in fa['phase']])-np.array([np.interp(np.union1d(fa['y'],fb['y']),fb['y'],p) for p in fb['phase'][indices]])),axis=1),
                q=abs(q1-q2)/max(abs(q2)),slip=abs(fa['slip']-fb['slip'][indices])/fb['slip'][-1],
                energy=abs(e1-e2)/max(abs(e2)),
                driving=np.max(abs(ga[:,0]+ga[:,1]-gb[:,0]-gb[:,1]),axis=1)/driving_scale)
    rows=[]
    for k,t in enumerate(ta):
        row=dict(time=float(t),whole_Pa=float(whole[k]),localized_Pa=float(localized[k]),
                 phase_gradient_L2=float(np.sqrt(w@((da[k]-db[k])**2))),energy_J_per_m=float(abs(e1[k]-e2[k])),
                 q_Pa=float(abs(q1[k]-q2[k])),slip_m=float(abs(fa['slip'][k]-fb['slip'][indices[k]])))
        for name,c in curves.items():row[name]=float(c[k])
        for j,name in enumerate(['bulk','cohesive','fracture']):
            row[name+'_driving_Linf']=float(max(abs(ga[k,j]-gb[k,j])))
            row[name+'_driving_dual_L2']=float(np.sqrt(np.sum((ga[k,j]-gb[k,j])**2/mass)))
        rows.append(row)
    maximum={name:max(rows,key=lambda r:r[name])[name]for name in TARGETS}
    worst={name:max(rows,key=lambda r:r[name])['time']for name in TARGETS}
    onset_difference=abs(a['summary']['onset']-b['summary']['onset'])
    return dict(maximum=maximum,worst_time=worst,passes={**{k:maximum[k]<=v for k,v in TARGETS.items()},'onset':onset_difference<=1/128},onset_difference_seconds=onset_difference,onset_tolerance_seconds=1/128,
                reference_scales=dict(stress_tensor_Pa=stress_scale,phase_L2=phase_scale,q_Pa=float(max(abs(q2))),
                                      slip_m=float(fb['slip'][-1]),energy_J_per_m=float(max(abs(e2))),driving=driving_scale),
                onset_a=a['summary']['onset'],onset_b=b['summary']['onset'],by_time=rows)


def seed_errors():
    from scipy.interpolate import PPoly
    f=np.load(PROP/'matched_initialization_evidence/reference.npz')
    poly=PPoly.construct_fast(f['coefficients'],f['knots'],axis=1)
    cut=float(f['cut']);z,w=np.polynomial.legendre.leggauss(8)
    y=(np.arange(4096)[:,None]+(z+1)/2).ravel()/4096;w=np.tile(w/(2*4096),4096)
    exact=evaluate_reference(y);distance=abs(y-.5);inside=distance<=cut
    de=np.empty_like(y);de[inside]=poly.derivative()(distance[inside]/cut)[0]/cut
    de[~inside]=poly.derivative()((distance[~inside]-cut)/(.5-cut))[2]/(.5-cut)
    de*=np.sign(y-.5)
    rows=[]
    for n in [32,64,128,256,512,1024]:
        yn=np.linspace(0,1,n+1);p=evaluate_reference(yn);v=np.interp(y,yn,p)
        dp=np.diff(p)*n;d=dp[np.minimum((y*n).astype(int),n-1)]
        rows.append(dict(cells=n,relative_L2=float(np.sqrt(w@((v-exact)**2)/(w@(exact**2)))),
                         Linf=float(max(abs(v-exact))),gradient_L2_error=float(np.sqrt(w@((d-de)**2))),
                         fracture_energy_error_J_per_m=float(A*w@(v-exact+ELL**2*(d*d-de*de))),
                         unloaded_KKT=float(max(np.maximum(-(A*Endpoint(n).mass+Endpoint(n).F@p)/A,0.)))))
    return rows


def frozen(common):
    f=np.load(common);phase=f['phase'][0].mean(axis=1);R=f['R'][1,:,:,0].mean(axis=1)
    mean=G*(R[-1]-R[0]);Z=float(np.mean(f['Z'][1]));out=[]
    for q in [24,48]:
        exact=Endpoint(32,None,q);exact.incoming(phase,mean,Z,0.,.5625,1/128);exact.evaluate(phase);e=exact.last
        for n in [128,256]:
            p=Endpoint(32,n,q);p.incoming(phase,mean,Z,0.,.5625,1/128);p.evaluate(phase);d=p.last
            error=d['tau']-e['tau_mean'];whole=np.sqrt(2*(p.w@(error*error)));local=np.sqrt(2*((p.w*d['chi'])@(error*error))/(p.w@d['chi']))
            out.append(dict(cells=n,quadrature=q,whole_full_tensor_L2_Pa=float(whole),localized_full_tensor_RMS_Pa=float(local),
                            whole_relative=float(whole/(np.sqrt(2)*e['tau_mean'])),localized_relative=float(local/(np.sqrt(2)*e['tau_mean'])),
                            mean_tau=d['tau_mean'],Z=d['Z'],remainder_L2=float(d['Z']*np.sqrt(p.w@(p.integrals(phase)['residual']**2)))))
    return out


def remainder_history(case):
    n=case['summary']['phase_cells'];nm=case['summary']['mechanical_cells']
    if not nm:return []
    model=Endpoint(n,nm,24);f=case['fields'];rows=[]
    scale=max(1.,np.sqrt(2)*max(abs(r['tau_mean'])for r in case['history']))
    for k,row in enumerate(case['history'],1):
        a=model.integrals(f['phase'][k]);r=a['residual'];Z=f['Z'][k]
        whole=np.sqrt(2)*Z*np.sqrt(model.w@(r*r))
        local=np.sqrt(2)*Z*np.sqrt((model.w*a['chi'])@(r*r)/(model.w@a['chi']))
        rows.append(dict(time=row['time'],whole_remainder_Pa=float(whole),localized_remainder_Pa=float(local),
                         whole_fraction=float(whole/scale),localized_fraction=float(local/scale),
                         phase_peak=row['phi_max'],Z=float(Z)))
    return rows


def fixed_quadrature(case):
    f=case['fields'];models=[Endpoint(512,512,q)for q in [12,24]];errors=[]
    for k,row in enumerate(case['history'],1):
        states=[]
        for model in models:
            model.incoming(f['phase'][k-1],f['meanT'][k-1],f['Z'][k-1],f['slip'][k-1],row['time'],1/128)
            model.evaluate(f['phase'][k]);states.append(model.last)
        a,b=states
        errors.append(dict(time=row['time'],gradient_Linf=float(max(abs(a['gradient']-b['gradient']))),
                           energy_J_per_m=float(abs(sum(a[n]for n in ['bulk','cohesive','fracture'])-sum(b[n]for n in ['bulk','cohesive','fracture']))),
                           Z_Pa_m=float(abs(a['Z']-b['Z'])),tau_mean_Pa=float(abs(a['tau_mean']-b['tau_mean']))))
    return dict(maximum={name:max(r[name]for r in errors)for name in errors[0]if name!='time'},by_time=errors)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cases',type=Path);p.add_argument('output',type=Path);p.add_argument('--common',type=Path,required=True);a=p.parse_args()
    cases={p.name:load(p)for p in a.cases.iterdir()if p.is_dir() and (p/'summary.json').exists()}
    result=dict(physical_initialization='retained continuum seeded-H BVP; exact native table nodal identity at selected mesh',targets=TARGETS,seed_representation=seed_errors(),frozen=frozen(a.common),comparisons={})
    pairs=[(f'p{n}-m{n}-d128-q12',f'p{n}-mexact-d128-q12',f'mechanics_{n}')for n in [64,128,256,512]]
    pairs += [(f'p{n}-mexact-d128-q12','p1024-mexact-d128-q12',f'phase_{n}')for n in [64,128,256,512]]
    pairs += [('p512-m512-d128-q12','p1024-mexact-d128-q12','candidate_512'),('p512-m512-d128-q12','p512-m512-d256-q12','temporal_512'),('p256-m256-d128-q12','p1024-mexact-d128-q12','candidate_256'),('p256-m256-d128-q12','p256-m256-d256-q12','temporal_candidate'),('p1024-mexact-d128-q12','p1024-mexact-d256-q12','temporal_reference'),('p256-m256-d128-q12','p256-m256-d128-q24','quadrature_candidate')]
    for x,y,label in pairs:
        if x in cases and y in cases:result['comparisons'][label]=comparison(cases[x],cases[y])
    result['remainder_history']={n:remainder_history(c)for n,c in cases.items()if n in ['p256-m256-d128-q12','p512-m512-d128-q12']}
    if 'p512-m512-d128-q12' in cases:result['selected_fixed_quadrature']=fixed_quadrature(cases['p512-m512-d128-q12'])
    result['cases']={n:dict(summary=c['summary'],final=c['history'][-1],departures=sum(len(r['stability']['departures']) for r in c['history']))for n,c in cases.items()}
    write(a.output,result)
    print(json.dumps({k:v['maximum']for k,v in result['comparisons'].items()},indent=2))


if __name__=='__main__':main()
