#!/usr/bin/env python3
"""Reduce prescribed and coupled corner controls; no new simulation or fitting."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','DEAL_II_NUM_THREADS'):os.environ[key]='1'
import csv
import json
import math
from pathlib import Path
import numpy as np
from assess_sheared_moving_fault import diagnostics, compare
from compare_bounded_profile import ROOT, RUNS, load, tensor, quadrature, dot3

OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/corner_isolation'
PRESCRIBED=['corner-prescribed32-r1','corner-prescribed64']
COUPLED=['corner-coupled050-r1','corner-coupled025']
OLD=['shear32-dt050','shear32-dt025']


def prescribed():
    cases=[]
    for name in PRESCRIBED:
        command=json.loads((RUNS/name/'command.json').read_text());assert command['exit_code']==0
        rows=[{k:float(v) for k,v in row.items()} for row in csv.DictReader((RUNS/name/'output/corner-isolation.csv').open())]
        groups={}
        for row in rows:groups.setdefault((row['reaction'],row['held_trace'],row['dt']),[]).append(row)
        for (reaction,held,dt),steps in groups.items():
            assert len(steps)==round(.4/dt) and abs(steps[-1]['time']-.4)<1e-14
            if not held:assert max(s['relative_discrete'] for s in steps)<1e-9
            last=steps[-1];cases.append(dict(case=name,reaction=bool(reaction),held_trace=bool(held),dt=dt,cells=int(last['cells']),
                final=last,steps=steps,maximum_identity=max(s['max_identity'] for s in steps),
                transport_balance_signed=sum(s['transport_balance'] for s in steps),
                transport_balance_absolute=sum(abs(s['transport_balance']) for s in steps),
                supplied_flux_integral=sum(s['flux_supplied_integral'] for s in steps),
                interior_flux_integral=sum(s['flux_interior_integral'] for s in steps),
                upwind_integral=sum(s['upwind_integral'] for s in steps),native_wall_s=command['wall_s']))
    orders=[]
    a=[c for c in cases if c['cells']==1024 and c['reaction'] and not c['held_trace']]
    for first,second in zip(a,a[1:]):orders.append(math.log2(first['final']['relative_continuous']/second['final']['relative_continuous']))
    result=dict(cases=cases,exact_inflow_continuous_temporal_orders=orders,
                interpretation='Errors against explicitly continued polynomial characteristic data. Held trace is a different inflow model; its differences are not quadrature error. The continuous/discrete split references distinguish first-order forcing order from boundary error. Energy balance uses native q8 face sign quadrature and upwind supplied data.')
    (OUT/'prescribed.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def regional(a,b):
    x,w=quadrature(max(a.n,b.n),8);aa,bb=tensor(a,x),tensor(b,x);delta=aa-bb
    masks=dict(whole=np.ones(len(x),bool),corner=(x[:,0]<-1.5)&(x[:,1]>1.5),
               rim=np.max(abs(x),axis=1)>1.75,interior=np.max(abs(x),axis=1)<=1.75)
    out={}
    for key,select in masks.items():
        num=float(w[select]@dot3(delta[select],delta[select]));den=float(w[select]@dot3(bb[select],bb[select]))
        out[key]=dict(absolute_L2=float(np.sqrt(num)),relative_L2=float(np.sqrt(num/den)) if den else None,
                      squared_error_fraction=num/float(w@dot3(delta,delta)))
    return out


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prescribed-only',action='store_true');args=p.parse_args()
    prescribed()
    if args.prescribed_only:return
    retained=json.loads((OUT.parent/'sheared_moving_fault/comparison.json').read_text())['cases']
    results={};starts={};ends={}
    for name in COUPLED:
        results[name],starts[name],ends[name]=diagnostics(name)
        print(name,'reduced',flush=True)
    for name in OLD:
        results[name]=retained[name];_,_,_,states=load(name);starts[name]=states[0];ends[name]=states[max(states)]
    assert np.array_equal(starts[COUPLED[0]].coeff,starts[COUPLED[1]].coeff)
    assert np.array_equal(starts[COUPLED[0]].surface,starts[COUPLED[1]].surface)
    comparisons={}
    for key,a,b in [('compatible_time',*COUPLED),('old_time',*OLD),('boundary050',OLD[0],COUPLED[0]),('boundary025',OLD[1],COUPLED[1])]:
        ca,cb=results[a]['endpoint_cumulative_slip'],results[b]['endpoint_cumulative_slip']
        comparisons[key]=dict(cases=[a,b],initial=compare(starts[a],starts[b]),
                             final=compare(ends[a],ends[b],ca,cb),regional_stress=regional(ends[a],ends[b]))
    extrema={};background_check={}
    for name,s in ends.items():
        x,w=quadrature(s.n,4);a=tensor(s,x);norm=np.sqrt(dot3(a,a));i=int(norm.argmax())
        extrema[name]=dict(sampled_max_stress=float(norm[i]),position=x[i].tolist(),
                           sampled_max_velocity=float(np.linalg.norm(s.at(x)[:,1:3],axis=1).max()))
        # Check the complex closed form used by the benchmark traction plugin
        # against explicit real component iteration, including initialization.
        dt=results[name]['arguments']['dt'];xx=yy=xy=0.;records=[]
        for step in range(1,round(.4/dt)+1):
            angle=-.25*dt;c=np.cos(angle);sn=np.sin(angle);beta=np.exp(-dt)
            xx,yy,xy=beta*(c*c*xx+sn*sn*yy-2*c*sn*xy),beta*(sn*sn*xx+c*c*yy+2*c*sn*xy),beta*(c*sn*(xx-yy)+(c*c-sn*sn)*xy)+5*(-np.expm1(-dt))
            r=np.exp(complex(-dt,-.5*dt));z=5j*(-np.expm1(-dt))*(1-r**step)/(1-r)
            error=max(abs(xx-z.real),abs(yy+z.real),abs(xy-z.imag));assert error<1e-13
            records.append(dict(step=step,time=step*dt,tx=xy,ty=yy,formula_difference=error))
        background_check[name]=records
    result=dict(cases=results,comparisons=comparisons,extrema=extrema,traction_formula_check=background_check,
                compatible_temporal_initialization_bitwise_equal=True,
                interpretation='The top-traction controls deliberately change the boundary-value problem. Material/fault inputs and held inflow are unchanged. Boundary-pair differences are responses to changed loading, not continuum errors. Within-compatible-boundary timestep sensitivity includes its discrete background traction approximation; two levels do not determine an order.')
    (OUT/'coupled.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Prescribed references, coupled comparison and discrete traction formula verified.',flush=True)


if __name__=='__main__':main()
