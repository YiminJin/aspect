#!/usr/bin/env python3
"""Crossed coupled boundary comparison and fixed-label shear refinement."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','DEAL_II_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-exterior-stress-mpl')
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from assess_sheared_moving_fault import diagnostics,compare
from assess_corner_isolation import regional
from compare_bounded_profile import ROOT,RUNS,load,tensor,quadrature,dot3
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/exterior_stress'
NEW=['exterior-zero025','exterior-compatible025','exterior-compatible050','exterior-compatible0125','exterior-fine0125']
OLD=['shear32-dt025','corner-coupled025']

def background(step,dt):
    # Real-component recurrence, independent of provider's complex closed form.
    a=np.zeros(3);c=np.cos(-.25*dt);s=np.sin(-.25*dt);beta=np.exp(-dt)
    for _ in range(step):
        x,y,z=a
        a=beta*np.array([c*c*x+s*s*y-2*c*s*z,s*s*x+c*c*y+2*c*s*z,c*s*(x-y)+(c*c-s*s)*z])
        a[2]+=5*(-np.expm1(-dt))
    return a

def supplied_flux(state,step,dt,order=8):
    g,gw=np.polynomial.legendre.leggauss(order);line=(-2+state.h*(np.arange(state.n)[:,None]+.5*(g+1))).ravel()
    w=np.tile(.5*state.h*gw,state.n);exterior=background(step,dt)
    out={}
    for axis in (0,1):
        for side in (-1,1):
            x=np.zeros((len(line),2));x[:,axis]=2*side;x[:,1-axis]=line
            v=state.at(x)[:,1:3];tau=tensor(state,x);vn=side*v[:,axis]
            ext=np.tile(exterior,(len(x),1));up=np.where((vn<0)[:,None],ext,tau)
            jump=tau-ext;incoming=vn<0
            out[['left','right','bottom','top'][2*axis+(side==1)]]=dict(
                supplied_physical_flux=float(w@(vn*dot3(up,up)/40)),
                interior_flux=float(w@(vn*dot3(tau,tau)/40)),
                incoming_penalty=float(w@(-np.minimum(vn,0)*dot3(jump,jump)/40)),
                incoming_measure=float(w@incoming),min_normal_velocity=float(vn.min()))
    return out

def stress_peaks(state):
    x,w=quadrature(state.n,4);tau=tensor(state,x);norm=np.sqrt(dot3(tau,tau));corner=(x[:,0]<-1.5)&(x[:,1]>1.5);i=int(norm.argmax())
    return dict(peak=float(norm[i]),position=x[i].tolist(),corner_peak=float(norm[corner].max()),
                whole_L2=float(np.sqrt(w@dot3(tau,tau))),corner_L2=float(np.sqrt(w[corner]@dot3(tau[corner],tau[corner]))))

def profile_shape(state):
    x,w=quadrature(state.n,8);phi=state.at(x)[:,0];mass=float(w@phi)
    center=np.sum((w*phi)[:,None]*x,axis=0)/mass
    tangent=state.surface[-1,1:3]-state.surface[0,1:3];tangent/=np.linalg.norm(tangent)
    normal=np.array([-tangent[1],tangent[0]])
    return dict(mass=mass,centroid=center.tolist(),normal_variance=float((w*phi)@((x-center)@normal)**2/mass))

def reduce_one(name):
    r,start,end=diagnostics(name)
    r['initial_profile_shape']=profile_shape(start)
    cmd,rows,meta,states=load(name);dt=cmd['arguments']['dt']
    records=list(csv.DictReader((RUNS/name/'output/exterior-stress.csv').open()))
    assert len(records)==len(rows)
    formula_error=0.;counts={key:0 for key in ['left_calls','right_calls','bottom_calls','top_calls']}
    for row in records:
        step=int(row['step']);parent=background(max(0,step-1),dt);endpoint=background(step,dt)
        formula_error=max(formula_error,abs(float(row['parent_xx'])-parent[0]),abs(float(row['parent_xy'])-parent[2]),
                          abs(float(row['endpoint_xx'])-endpoint[0]),abs(float(row['endpoint_xy'])-endpoint[2]))
        if step==1:assert float(row['parent_xx'])==float(row['parent_xy'])==0
        for key in counts:counts[key]+=int(row[key])
    assert formula_error<1e-12 and counts['left_calls']>0 and counts['right_calls']>0
    fluxes=[];peaks=[];signed=absolute=0.;previous=None
    for row in rows:
        step=int(row['step']);s=states[step];terms=supplied_flux(s,step,dt)
        peaks.append(dict(step=step,time=float(row['time']),**stress_peaks(s)))
        supplied=sum(v['supplied_physical_flux'] for v in terms.values())
        penalty=sum(v['incoming_penalty'] for v in terms.values())
        e=r['coupled_work']['steps'][step]
        item=dict(step=step,time=float(row['time']),boundaries=terms,supplied_physical_flux=supplied,incoming_penalty=penalty)
        if previous is not None:
            # Sensitivity of the endpoint continuum balance to prescribed physical
            # exterior. Not the SSPRK transport-stage balance or a fitted closure.
            defect=e['coupled_endpoint_remainder']+float(row['dt'])*(supplied-e['flux'])
            signed+=defect;absolute+=abs(defect);item['endpoint_remainder_supplied_flux']=defect
        fluxes.append(item);previous=e
    r['stress_evolution']=peaks
    r['exterior_audit']=dict(formula_max_error=formula_error,first_transport_exterior_zero=True,calls=counts,steps=fluxes,
                            endpoint_supplied_flux_signed=signed,endpoint_supplied_flux_absolute=absolute,
                            endpoint_supplied_flux_relative=absolute/r['coupled_work']['active_scale'],
                            interpretation='Endpoint physical-background upwind flux sensitivity, not the split transport substep ledger; retained interior-trace work balance is reported separately.')
    x,w=quadrature(end.n,4);a=tensor(end,x);norm=np.sqrt(dot3(a,a));i=int(norm.argmax())
    r['extrema']=dict(sampled_max_stress=float(norm[i]),position=x[i].tolist(),sampled_max_velocity=float(np.linalg.norm(end.at(x)[:,1:3],axis=1).max()))
    (OUT/(name+'.json')).write_text(json.dumps(r,indent=2)+'\n')
    print(name,'peak',r['extrema']['sampled_max_stress'],'endpoint',r['coupled_work']['relative_active'],'top calls',counts['top_calls'],flush=True)

def combine():
    previous=json.loads((OUT.parent/'corner_isolation/coupled.json').read_text())
    results={name:previous['cases'][name] for name in OLD}
    results.update({name:json.loads((OUT/(name+'.json')).read_text()) for name in NEW})
    starts={};firsts={};ends={}
    for name in OLD+NEW:
        _,rows,_,states=load(name);starts[name]=states[0];firsts[name]=states[1];ends[name]=states[max(states)]
        if name in OLD:results[name]['stress_evolution']=[dict(step=int(row['step']),time=float(row['time']),**stress_peaks(states[int(row['step'])])) for row in rows]
    # Temporal initial state must be identical, independent of exterior timestep.
    for name in NEW[2:4]:
        assert np.array_equal(starts[NEW[1]].coeff,starts[name].coeff)
        assert np.array_equal(starts[NEW[1]].surface,starts[name].surface)
    # Inflow is inactive at initialization, so compare to retained loading controls.
    for a,b in zip(NEW[:2],OLD):
        assert np.array_equal(starts[a].coeff,starts[b].coeff),(a,b,'initial bulk')
        assert np.array_equal(starts[a].surface,starts[b].surface),(a,b,'initial surface')
        # Both exterior options are exactly zero on the first pure-transport
        # stage. A mistakenly reacted first exterior would break this control.
        assert np.array_equal(firsts[a].coeff,firsts[b].coeff),(a,b,'first-step bulk')
        assert np.array_equal(firsts[a].surface,firsts[b].surface),(a,b,'first-step surface')
    assert all(np.array_equal(starts[NEW[1]].surface[:,:3],starts[n].surface[:,:3]) for n in NEW)
    pairs=[('zero_inflow',OLD[0],NEW[0]),('compatible_inflow',OLD[1],NEW[1]),
           ('prescribed_top',NEW[0],NEW[1]),('time050_025',NEW[2],NEW[1]),
           ('time025_0125',NEW[1],NEW[3]),('space32_64',NEW[3],NEW[4])]
    comparisons={}
    for key,a,b in pairs:
        comparisons[key]=dict(cases=[a,b],initial=compare(starts[a],starts[b]),
            final=compare(ends[a],ends[b],results[a]['endpoint_cumulative_slip'],results[b]['endpoint_cumulative_slip'],
                          region_reference=ends[NEW[3]] if key.startswith('time') else None),
            regional_stress=regional(ends[a],ends[b]))
    temporal_orders={}
    coarse=comparisons['time050_025']['final'];fine=comparisons['time025_0125']['final']
    for region in ['central','ends']:
        temporal_orders[region]={'stress':math.log2(coarse['stress'][region]['absolute_L2']/fine['stress'][region]['absolute_L2'])}
        for field in ['shear_traction','work_shear_traction','total_normal_traction','slip_rate','cumulative_slip','geometry','age','normalization']:
            temporal_orders[region][field]=math.log2(coarse['surface'][region][field]['absolute_L2']/fine['surface'][region][field]['absolute_L2'])
    result=dict(cases=results,comparisons=comparisons,temporal_orders=temporal_orders,retained_extrema=previous['extrema'],
                initialization_bitwise_equal=True,first_step_trace_controls_bitwise_equal=True,labels_fixed=True,
                interpretation='Crossed differences are boundary-model responses. Temporal differences include discrete background-loading changes. Spatial comparison holds dt and material labels fixed. Absolute continuum accuracy and production readiness are not established.')
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    for key,c in comparisons.items():
        print(key,'global',c['regional_stress']['whole']['relative_L2'],'corner',c['regional_stress']['corner']['relative_L2'],flush=True)
    return result

def plot(r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    names=[OLD[0],NEW[0],OLD[1],NEW[1]];labels=['Zero / trace','Zero / exterior','Compatible / trace','Compatible / exterior']
    peaks=[(r['retained_extrema'][n] if n in OLD else r['cases'][n]['extrema'])['sampled_max_stress'] for n in names]
    axes[0,0].bar(labels,peaks,color=['#9c3348','#dc8650','#5682a0','#36744d']);axes[0,0].tick_params(axis='x',rotation=20)
    axes[0,0].set(ylabel='Peak stress norm (Pa)',title='Crossed boundary comparison, 32² / dt=.025')
    sequence=[NEW[2],NEW[1],NEW[3]];dt=[r['cases'][n]['arguments']['dt'] for n in sequence]
    ax=axes[0,1]
    ax.plot(dt,[100*r['cases'][n]['coupled_work']['relative_active'] for n in sequence],'o-',label='Endpoint, interior-trace flux')
    ax.plot(dt,[100*r['cases'][n]['trapezoid_absolute']/r['cases'][n]['power_scale'] for n in sequence],'s-',label='Trapezoidal diagnostic')
    ax.set(xlabel='Timestep (s)',ylabel='Absolute remainder / active work (%)',title='Compatible / exterior temporal sequence');ax.legend(fontsize=8)
    for name in sequence+[NEW[4]]:
        c=r['cases'][name];p=c['final_profiles'];label=f"{2**c['arguments']['level']}², dt={c['arguments']['dt']}"
        axes[1,0].plot(p['labels'],p['slip_rate'],label=label)
        axes[1,1].plot(p['labels'],p['total_normal_traction'],label=label)
    for ax in axes[1]:
        ax.axvspan(0,.2,color='.8',alpha=.25);ax.axvspan(.8,1,color='.8',alpha=.25);ax.set_xlabel('Material label');ax.legend(fontsize=8)
    axes[1,0].set(ylabel='Slip rate (m/s)',title='Final slip rate (shaded endpoint regions)')
    axes[1,1].set(ylabel='Normal traction including pressure (Pa)',title='Final normal traction')
    for ax in axes.flat:ax.grid(alpha=.2)
    for ext in ('png','pdf'):fig.savefig(OUT/f'exterior-stress.{ext}',dpi=160)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('case',nargs='?',choices=NEW);a=p.parse_args()
    if a.case:reduce_one(a.case)
    else:plot(combine())
if __name__=='__main__':main()
