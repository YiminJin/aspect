#!/usr/bin/env python3
"""Common-time and common-interval assessment of one fixed spatial model."""
import json
import numpy as np
from time_resolution import E,BASE,directory,load,Reference,G,h

def trajectory(d):
    f=np.load(directory(d)/'fields.npz');hist=load(directory(d)/'history.json');n=len(hist)
    energy=load((E/'baseline-diagnostics' if d==64 else directory(d))/'energy.json')
    return dict(d=d,steps=n,phi=np.vstack([f['initial'],f['phase']]),r=np.vstack([f['incoming_r'],f['final_r']]),Z=np.r_[f['incoming_Z'],f['final_Z']],q=np.r_[0.,[v['q'] for v in hist]],slip=np.r_[0.,[v['slip'] for v in hist]],bulk=np.r_[0.,[v['bulk'] for v in hist]],cohesive=np.r_[0.,[v['cohesive'] for v in hist]],fracture=np.r_[energy[0]['fracture']-energy[0]['fracture_change'],[v['fracture'] for v in hist]],history=hist,energy=energy)

def stress_fields(a,indices,r):
    phi=a['phi'][indices];I=np.array([r.direct(p)[0] for p in phi]);chi=h((r.B@phi.T).T)/I[:,None]
    return G*(r.D@a['r'][indices].T).T-a['Z'][indices,None]*chi,chi

def difference(a,b,r,common_only=False,mask=None):
    # All candidate accepted times also exist in the dyadic finer reference.
    base=64 if common_only else a['d'];times=np.arange(round(.625*base)+1)/base
    if mask is not None:times=times[mask(times)]
    ia=np.rint(times*a['d']).astype(int);ib=np.rint(times*b['d']).astype(int)
    ta,_=stress_fields(a,ia,r);tb,chi=stress_fields(b,ib,r);dtau=ta-tb
    whole=np.sqrt((dtau*dtau)@r.w);local=np.sqrt(np.sum(dtau*dtau*chi*r.w,axis=1)/(chi@r.w))
    onset_change=np.max(abs(a['phi'][ia]-b['phi'][ib]),axis=1)
    ks=np.arange(41);sa=a['slip'][ks*(a['d']//64)];sb=b['slip'][ks*(b['d']//64)]
    rates_a=np.diff(sa)*64;rates_b=np.diff(sb)*64
    peak=float(max(abs(rates_b)));cfg=load(E/'baseline.json')
    rate_mask=np.ones(40,dtype=bool) if mask is None else mask(ks[:-1]/64)&mask(ks[1:]/64)
    errors=dict(phi=float(max(onset_change)),q_Pa=float(max(abs(a['q'][ia]-b['q'][ib]))),slip_relative=float(max(abs(a['slip'][ia]-b['slip'][ib]))/b['slip'][-1]),V_common_relative=float(max(abs(rates_a[rate_mask]-rates_b[rate_mask]))/peak),whole_stress_relative=float(max(whole)/cfg['stress_scale_Pa']),localized_stress_relative=float(max(local)/cfg['stress_scale_Pa']),bulk_storage_relative=float(max(abs(a['bulk'][ia]-b['bulk'][ib]))/cfg['storage_scale_J_m']),cohesive_storage=float(max(abs(a['cohesive'][ia]-b['cohesive'][ib]))),fracture_storage=float(max(abs(a['fracture'][ia]-b['fracture'][ib]))),mean_stress_Pa=float(max(abs(dtau@r.w))),phase_error_time=float(times[np.argmax(onset_change)]),fixed_peak_common_rate=peak)
    return errors

def onset(a):
    result={}
    for key,values in [('full_field',np.max(a['phi']-a['phi'][0],axis=1)),('core',a['phi'][:,128]-a['phi'][0,128])]:
        active=np.flatnonzero(values>1e-5)
        if len(active):
            j=int(active[0]);result[key]=dict(time_bracket=[(j-1)/a['d'],j/a['d']],displacement_bracket=[.128*((j-1)/a['d'])**2,.128*(j/a['d'])**2])
        else:result[key]=None
    return result

def main():
    ds=[64,128,256,512]+([1024] if (directory(1024)/'fields.npz').exists() else [])
    paths={d:trajectory(d) for d in ds};r=Reference(256,1.,24);ref=paths[ds[-1]];cfg=load(E/'baseline.json');targets=cfg['targets']
    pairs=[]
    for a,b in zip(ds[:-1],ds[1:]):
        pairs.append(dict(coarse=a,fine=b,common=difference(paths[a],paths[b],r,True),all_coarse_times=difference(paths[a],paths[b],r),smooth_pre_onset=difference(paths[a],paths[b],r,True,lambda t:t<=.515625),onset=difference(paths[a],paths[b],r,True,lambda t:(t>.515625)&(t<=.5625)),late=difference(paths[a],paths[b],r,True,lambda t:t>.5625)))
    orders=[]
    for first,second in zip(pairs[:-1],pairs[1:]):
        row=dict(partitions=[first['coarse'],first['fine'],second['fine']])
        for window in ['common','smooth_pre_onset','onset','late']:
            row[window]={k:float(np.log2(first[window][k]/second[window][k])) if min(first[window][k],second[window][k])>1e-12 else None for k in targets}
        orders.append(row)
    # Richardson estimates are labeled conditional; unresolved onset orders
    # retain the full last halving difference instead of inventing an order.
    uncertainty={};last=pairs[-1]['all_coarse_times']
    for k in targets:
        p=orders[-1]['common'][k]
        uncertainty[k]=dict(last_halving_difference=last[k],observed_order=p,estimated_finest_error=last[k]/(2**p-1) if p is not None and .5<p<3 else last[k],conditional_on_smooth_asymptotic_behavior=p is not None and .5<p<3)
    rows={}
    for d,a in paths.items():
        errors=difference(a,ref,r);pass_raw=all(errors[k]<=v for k,v in targets.items())
        # The finest trajectory is not qualified by comparison with itself.
        pass_with_uncertainty=all(errors[k]+uncertainty[k]['estimated_finest_error']<=v for k,v in targets.items())
        rows[str(d)]=dict(steps=a['steps'],errors=errors,raw_target_pass=pass_raw,uncertainty_inclusive_pass=pass_with_uncertainty,onset=onset(a),final=dict(phi_max=float(max(a['phi'][-1])),q_Pa=float(a['q'][-1]),slip=float(a['slip'][-1]),bulk=float(a['bulk'][-1]),cohesive=float(a['cohesive'][-1]),fracture=float(a['fracture'][-1])),maximum_phase_increment=max(v['increment'] for v in a['history']))
    candidates=[d for d in ds if d<ds[-1] and rows[str(d)]['uncertainty_inclusive_pass']]
    selected=min(candidates) if candidates else None
    report=dict(reference_denominator=ds[-1],partitions=ds,targets=targets,errors=rows,successive_halvings=pairs,observed_orders=orders,finest_reference_uncertainty=uncertainty,selected_denominator=selected,common_interval=[1/64,'rates from slip differences on forty identical intervals'],stress_scale_Pa=cfg['stress_scale_Pa'],storage_scale_J_m=cfg['storage_scale_J_m'],need_optional_refinement=ds[-1]<1024 and (selected is None or any(v['estimated_finest_error']>targets[k]/4 for k,v in uncertainty.items())))
    (E/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(reference=ds[-1],selected=selected,need_optional_refinement=report['need_optional_refinement'],rows={k:dict(errors=v['errors'],increment=v['maximum_phase_increment']) for k,v in rows.items()},uncertainty=uncertainty),indent=2))

if __name__=='__main__':main()
