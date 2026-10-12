#!/usr/bin/env python3
"""Common initialization, position/shape separation and full 512 FE assessment."""
import argparse,hashlib
import numpy as np
from scipy import sparse
import matched_initialization as m
import asymmetric_branch as a
from assess_asymmetric_branch import normalization_check
from assess_coupled_branch import totals
from normal_resolution import moments,extrema

E=m.E
DT=1/1024
SCALE_STRESS=1332.521975
SCALE_STORAGE=.887807407

def baseline(exact):
    path=a.E/('n512-d1024'+('-exact' if exact else ''))
    f=np.load(path/'fields.npz');c=np.load(path/'search-continue-+1/fields.npz')
    data={key:np.concatenate([f[key],c[key][1:]]) for key in ['phase','R','Z']}
    data.update(y=f['y'],yq=f['yq'])
    hist=a.read(path/'history.json')+a.read(path/'search-continue-+1/history.json')
    assert len(hist)==640 and len(data['phase'])==641 and hist[-1]['time']==.625
    assert max(row['KKT'] for row in hist)<=a.TARGET
    assert max(row['force'] for row in hist)<1e-7 and max(row['contact'] for row in hist)<1e-7
    assert min(data['phase'].ravel())>=0 and max(data['phase'].ravel())<1
    assert min(np.diff(data['phase'],axis=0).ravel())>=-4.922e-12
    return data,hist


def load(n):
    path=E/f'n{n}';summary=a.read(path/'summary.json');assert summary['completed']
    return dict(np.load(path/'fields.npz')),a.read(path/'history.json')


def compare(fa,ha,fb,hb,label):
    y=np.linspace(0,1,16385);rows=[]
    for i in range(641):
        pa=np.interp(y,fa['y'],fa['phase'][i]);pb=np.interp(y,fb['y'],fb['phase'][i])
        ca=float(np.trapezoid(y*pa,y)/np.trapezoid(pa,y));cb=float(np.trapezoid(y*pb,y)/np.trapezoid(pb,y))
        # Analysis-only coordinate pullback. Every physical history remains untouched.
        shifted=np.interp(y+ca-cb,fa['y'],fa['phase'][i]);Ia=ha[i-1]['I'] if i else normalization_check(a.core.Reference(len(fa['y'])-1,1.,6),fa['phase'][0])['analytic_I']
        Ib=hb[i-1]['I'] if i else normalization_check(a.core.Reference(len(fb['y'])-1,1.,6),fb['phase'][0])['analytic_I']
        localization_centroid_a=float(np.trapezoid(y*a.core.h(pa),y)/np.trapezoid(a.core.h(pa),y))
        localization_centroid_b=float(np.trapezoid(y*a.core.h(pb),y)/np.trapezoid(a.core.h(pb),y))
        row=dict(time=i*DT,phase=float(max(abs(pa-pb))),aligned_phase=float(max(abs(shifted-pb))),
          centroid=abs(ca-cb),signed_centroid=ca-cb,chi=float(max(abs(a.core.h(pa)/Ia-a.core.h(pb)/Ib))),
          localization_centroid=abs(localization_centroid_a-localization_centroid_b),
          aligned_chi=float(max(abs(a.core.h(shifted)/Ia-a.core.h(pb)/Ib))),I=abs(Ia-Ib),relative_I=abs(Ia/Ib-1))
        if i:
            row.update({key:abs(ha[i-1][key]-hb[i-1][key]) for key in ['q','slip','bulk','cohesive','fracture']})
        else:row.update(q=0.,slip=0.,bulk=0.,cohesive=0.,fracture=abs(fracture(fa['y'],fa['phase'][0])-fracture(fb['y'],fb['phase'][0])))
        rows.append(row)
    keys=['phase','aligned_phase','centroid','localization_centroid','chi','aligned_chi','I','relative_I','q','slip','bulk','cohesive','fracture']
    return dict(label=label,maximum={k:max(r[k] for r in rows) for k in keys},final=rows[-1],by_time=rows)


def fracture(y,phi):return a.A*(np.trapezoid(phi,y)+m.ELL**2*np.sum(np.diff(phi)**2/np.diff(y)))


def initial(f,hist):
    y=np.linspace(0,1,65537);phi=np.interp(y,f['y'],f['phase'][0]);reference=m.evaluate_reference(y)
    growth=np.max(f['phase']-f['phase'][0],axis=1);ids=np.flatnonzero(growth>1e-6)
    onset=None if not len(ids) else [(int(ids[0])-1)*DT,int(ids[0])*DT]
    return dict(Q1_representation_error=float(max(abs(phi-reference))),initial_core=float(np.interp(.5,f['y'],f['phase'][0])),
        initial_max=float(max(f['phase'][0])),initial_fracture=fracture(f['y'],f['phase'][0]),onset_bracket=onset,
        final_centroid=hist[-1]['centroid'],maximum_centroid_shift=max(abs(r['centroid']['phase']-.5) for r in hist),
        maximum_centroid_time=max(hist,key=lambda r:abs(r['centroid']['phase']-.5))['time'])


def independent_stress(f,hist,ref,order=24):
    n=len(f['y'])-1;z,w=np.polynomial.legendre.leggauss(order);s=(z+1)/2;weight=np.tile(w/(2*n),n);out=[]
    for j,row in enumerate(hist):
        phi=(1-s)*f['phase'][j+1,:-1,None]+s*f['phase'][j+1,1:,None]
        R=f['R'][j+1];derivative=n*((-3+4*s)*R[:-2:2,None]+(4-8*s)*R[1::2,None]+(-1+4*s)*R[2::2,None])
        chi=a.core.h(phi)/row['I'];tau=a.G*derivative-f['Z'][j+1]*chi
        result=moments(tau.ravel(),weight,chi.ravel(),ref[j]['stress_min'])
        result.update(time=row['time'],storage_error=abs(result['bulk_energy']-ref[j]['bulk']),
                      stress_min=float(min(tau.ravel())),stress_max=float(max(tau.ravel())))
        if j in [543,607,639]:result.update(extrema(n,f['phase'][j+1],R,f['Z'][j+1],row['I']))
        out.append(result)
    return out


def baseline_energy(exact):
    target=E/('baseline-exact-energy.json' if exact else 'baseline-FE-energy.json')
    if target.exists():return a.read(target)
    f,hist=baseline(exact);p=(m.Exact(512) if exact else a.MeshCoupled(512));energies=[]
    for j,row in enumerate(hist):
        pa=f['phase'][j];pb=f['phase'][j+1];p.incoming(pa,f['R'][j],f['Z'][j],hist[j-1]['slip'] if j else 0.,row['time'],DT)
        p.evaluate(pb);u=p.r.velocity.copy();assert max(abs(p.last[4]-f['R'][j+1]))<1e-12
        ledger=(a.exact_ledger(p,pa,pb,p.Ra,p.Za,f['R'][j+1],f['Z'][j+1],u,row['V'],DT) if exact else
                a.core.path_ledger(p.r,pa,pb,p.Ra,p.Za,f['R'][j+1],f['Z'][j+1],u,row['V'],DT))
        ledger['time']=row['time'];energies.append(ledger)
    a.write(target,energies);return energies


def assess():
    result=dict(reference=a.read(E/'reference.json'),operator=a.read(E/'operator-equivalence.json'),initialization={},controls={},comparisons=[])
    shared={n:load(n) for n in [511,512,1024]};oldexact=baseline(True);oldFE=baseline(False)
    for label,(f,history) in [('old-512-exact',oldexact),('old-512-FE',oldFE)]+[(f'shared-{n}',v) for n,v in shared.items()]:
        result['initialization'][label]=initial(f,history)
    for n,(f,hist) in shared.items():
        p=m.Exact(n);energies=a.read(E/f'n{n}/energy.json');checks=[]
        for t in [.53125,.5625,.59375,.625]:
            j=round(t/DT)-1;p.incoming(f['phase'][j],f['R'][j],f['Z'][j],hist[j-1]['slip'],t,DT)
            x=f['phase'][j+1];status,v=a.spectrum(p,x);p.evaluate(x)
            low=a.exact_ledger(p,p.lower,x,p.Ra,p.Za,p.last[4],p.last[5],p.r.velocity,hist[j]['V'],DT,16)
            high=a.exact_ledger(p,p.lower,x,p.Ra,p.Za,p.last[4],p.last[5],p.r.velocity,hist[j]['V'],DT,32)
            uncertainty=max(abs(low[k]-high[k]) for k in ['external_remainder','mechanical_power','friction_residual_power','phase_power'])
            assert uncertainty<1e-10
            checks.append(dict(time=t,stability=status,normalization=normalization_check(p.r,x),time_quadrature_uncertainty=uncertainty))
        assert max(r['KKT'] for r in hist)<=a.TARGET and max(r['force'] for r in hist)<1e-7 and max(r['contact'] for r in hist)<1e-7
        assert min(np.diff(f['phase'],axis=0).ravel())>=-4.922e-12
        assert max(e['identity_error'] for e in energies)<1e-10
        beta=np.exp(-a.G*DT/a.ETA);kappa=-a.ETA*np.expm1(-a.G*DT/a.ETA)
        recurrence=float(max(abs((f['R'][1:]-beta*f['R'][:-1]-kappa/a.G*f['velocity']).ravel())))
        assert recurrence<1e-16
        result['controls'][str(n)]=dict(summary=a.read(E/f'n{n}/summary.json'),initialization=a.read(E/f'n{n}/initialization.json'),
            max_KKT=max(r['KKT'] for r in hist),max_force=max(r['force'] for r in hist),max_contact=max(r['contact'] for r in hist),
            max_phase_increment=max(r['phase_increment'] for r in hist),recurrence=recurrence,energy=totals(energies),checks=checks,final=hist[-1])
    for fa,ha,fb,hb,label in [(*oldexact,*shared[512],'old vs shared initialization, 512 exact'),
                              (*shared[511],*shared[512],'shared 511 vs 512 exact'),
                              (*shared[512],*shared[1024],'shared 512 vs 1024 exact'),
                              (*oldFE,*shared[1024],'complete retained 512 FE vs shared 1024 exact'),
                              (*oldFE,*oldexact,'retained 512 FE vs same-initialization exact bulk')]:
        result['comparisons'].append(compare(fa,ha,fb,hb,label))
    stress=independent_stress(*oldFE,shared[1024][1]);a.write(E/'512-FE-independent-stress.json',stress)
    higher=independent_stress(*oldFE,shared[1024][1],48)
    integration_errors={key:max(abs(low[key]-high[key]) for low,high in zip(stress,higher)) for key in ['mean_stress_Pa','whole_error_RMS_Pa','localized_error_RMS_Pa','bulk_energy']}
    assert integration_errors['bulk_energy']<1e-11 and integration_errors['localized_error_RMS_Pa']<1e-7
    maximum={key:max(r[key] for r in stress) for key in ['whole_error_RMS_Pa','localized_error_RMS_Pa','oscillation_RMS_Pa','storage_error']}
    maximum['mean_error_Pa']=max(abs(r['mean_error_Pa']) for r in stress)
    comparison=result['comparisons'][3]['maximum'];slipref=shared[1024][1][-1]['slip']
    metrics=dict(phase=comparison['phase'],q=comparison['q'],slip_fraction=comparison['slip']/slipref,
                 whole_stress_fraction=maximum['whole_error_RMS_Pa']/SCALE_STRESS,localized_stress_fraction=maximum['localized_error_RMS_Pa']/SCALE_STRESS,
                 storage_fraction=maximum['storage_error']/SCALE_STORAGE)
    targets=dict(phase=.01,q=10.,slip_fraction=.005,whole_stress_fraction=.05,localized_stress_fraction=.05,storage_fraction=.05)
    result['512_FE_verification']=dict(metrics=metrics,targets=targets,passes={k:metrics[k]<=v for k,v in targets.items()},
       stress_scale_Pa=SCALE_STRESS,storage_scale_J_per_m=SCALE_STORAGE,independent_stress_maximum=maximum,stress_final=stress[-1],
       independent_stress_quadrature_difference_24_to_48=integration_errors,
       energy=totals(baseline_energy(False)),exact_baseline_energy=totals(baseline_energy(True)))
    result['preserved_files']=len(a.read(E/'preservation.json'))
    for name,digest in a.read(E/'preservation.json').items():assert hashlib.sha256((a.ROOT/name).read_bytes()).hexdigest()==digest,name
    a.write(E/'assessment.json',result)
    print(a.json.dumps(dict(initialization=result['initialization'],comparisons=[dict(label=x['label'],maximum=x['maximum']) for x in result['comparisons']],verification=result['512_FE_verification']['metrics'])),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--baseline-energy',choices=['FE','exact']);args=parser.parse_args()
    if args.baseline_energy:baseline_energy(args.baseline_energy=='exact')
    else:assess()
