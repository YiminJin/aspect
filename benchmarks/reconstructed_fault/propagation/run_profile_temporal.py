#!/usr/bin/env python3
"""Two authorized endpoint refinements and smooth prescribed-profile controls."""
import argparse
import json
import resource
import time
import numpy as np

import time_resolution as existing
from profile_temporal_diagnosis import E, OLD, Reference, load, audit


def extended(d):
    assert d in [2048,4096]
    # Reuse the complete existing step, bounds, tolerances and publication.
    # Only the evidence destination differs; original files are immutable.
    existing.E = E
    existing.case(d)


def smooth(d):
    out=E/f'smooth-dt-1-{d}'
    out.mkdir(exist_ok=False)
    start=time.perf_counter();cpu=time.process_time();dt=1/d
    r=Reference(256,1.,6)
    reference=np.load(OLD/'baseline-diagnostics/initial.npz')
    assert np.array_equal(r.initial,reference['phi'])
    existing.initial_snapshot(r,out)
    # Smooth monotone prescribed nodal profile, C2 at activation time .5s.
    # This named diagnostic is independent of every evolving trajectory.
    shape=np.exp(-((r.y-.5)/.15)**2)
    def profile(t):
        s=np.clip((t-.5)/.125,0.,1.)
        return r.initial+.08*(10*s**3-15*s**4+6*s**5)*shape
    fields=[];anchors=[];zs=[];hist=[];energies=[]
    for j in range(1,round(.625*d)+1):
        t=j/d;lower=r.phi.copy();anchor=r.r.copy();Z=r.Z
        r.phi=profile(t)
        assert np.min(r.phi-lower)>=-1e-14 and max(r.phi)<1
        data=r.mechanical(t,dt)
        term=existing.energy(r,lower,r.phi,anchor,Z,r.r,r.Z,dt,data['V'],t)
        term.update(time=t,increment=float(max(r.phi-lower)))
        assert data['force']<1e-7 and abs(data['stick_slip'])<1e-7
        assert abs(term['bulk_work_error']-data['mechanical_error'])<1e-10
        fields.append(r.phi.copy());anchors.append(anchor);zs.append(Z)
        hist.append(dict(time=t,phi_max=float(max(r.phi)),increment=term['increment'],
                         I=float(r.direct(r.phi)[0]),**{k:float(v) for k,v in data.items()}))
        energies.append(term)
    np.savez_compressed(out/'fields.npz',y=r.y,phase=fields,initial=r.initial,
                        incoming_r=anchors,incoming_Z=zs,final_r=r.r,final_Z=r.Z)
    summary=dict(denominator=d,steps=len(hist),final_time=.625,
                 prescribed_phase='initial+.08*smoothstep5((t-.5)/.125)*exp(-((y-.5)/.15)^2)',
                 diagnostic_not_physical_model=True,energy_totals=existing.summarize(energies),
                 runtime_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
                 peak_memory_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    for name,values in [('history',hist),('energy',energies),('summary',summary)]:
        (out/(name+'.json')).write_text(json.dumps(values,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    audit(d,out,'smooth-diagnosis-'+str(d))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--denominator',type=int,required=True)
    parser.add_argument('--smooth',action='store_true')
    args=parser.parse_args()
    if args.smooth:smooth(args.denominator)
    else:extended(args.denominator)
