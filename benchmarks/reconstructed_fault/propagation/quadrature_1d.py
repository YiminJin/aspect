#!/usr/bin/env python3
"""Bounded matched Q2/Q3 normal-quadrature and Q2 mesh study."""
import argparse,json,time,resource
from pathlib import Path
import numpy as np
from localization_1d import Reference
from total_energy_step import G,ETA,FRACTURE_SCALE as A,h

HERE=Path(__file__).resolve().parent
E=HERE/'quadrature_evidence'
DT=.015625;FINAL=.625

def case(n,degree,rule):
    out=E/f'one-d-n{n}-d{degree}-q{rule}'
    out.mkdir(exist_ok=False)
    start=time.perf_counter();cpu=time.process_time()
    r=Reference(n,1.,rule,degree=degree)
    check=Reference(n,1.,24,degree=degree)
    higher=Reference(n,1.,48,degree=degree)
    assert np.array_equal(r.initial,check.initial) and np.array_equal(r.initial,higher.initial)
    initial=r.initial.copy();fields=[];anchors=[];zs=[];history=[];release=error=0.
    audits=[];onset=None
    try:
        for step in range(1,41):
            old=r.phi.copy();anchor=r.r.copy();Z=r.Z
            e0=r.objective(old)[0]*A
            r.phi,kkt=r.phase(old)
            en=r.objective(r.phi)[0]*A
            release+=e0-en
            if onset is None and max(r.phi-initial)>1e-5:onset=step*DT
            # Proper incoming reference anchor: no mechanical publication yet.
            for ref in [check,higher]:ref.r=anchor.copy();ref.Z=Z
            value,grad,parts=r.objective(r.phi)
            cv,cg,cp=check.objective(r.phi);hv,hg,hp=higher.objective(r.phi)
            H=r.phase_hessian(r.phi);CH=check.phase_hessian(r.phi)
            v=np.sin(np.arange(n+1)+.2);v[r.phi<.001]=0;v[r.phi>.999]=0
            eps=1e-5
            gp=r.objective(r.phi+eps*v)[1];gm=r.objective(r.phi-eps*v)[1]
            fd=r.objective(r.phi+eps*v)[0]-r.objective(r.phi-eps*v)[0]
            derivative_error=abs(fd/(2*eps)-grad@v)/max(1.,abs(grad@v))
            hessian_error=max(abs((gp-gm)/(2*eps)-H@v))/max(1.,max(abs(H@v)))
            # Identical positive weights in coupling and its transpose.
            I,_=r.direct(r.phi);chi=h(r.B@r.phi)/I
            u=np.sin(np.arange(len(r.r))+.3);B=-r.D.T@(r.w*chi)
            left=u@B;right=-np.dot(r.w*chi,r.D@u)
            assert abs(left-right)<=256*np.finfo(float).eps*max(1.,abs(left),abs(right))
            pherr=max(abs(grad-cg));herr=np.max(abs(H-CH))/max(1.,np.max(abs(CH)))
            audit=dict(time=step*DT,phase_gradient_error=float(pherr),phase_Hessian_relative_error=float(herr),bulk_energy_error=float(abs(parts[0]-cp[0])),higher_reference_energy_error=float(abs(cp[0]-hp[0])),higher_reference_gradient_error=float(max(abs(cg-hg))),derivative_relative_error=float(derivative_error),Hessian_derivative_relative_error=float(hessian_error),adjoint_error=float(abs(left-right)))
            # Mechanical solve uses the same rule; independent endpoint force
            # and bulk energy re-evaluation uses frozen FE coefficients.
            data=r.mechanical(step*DT,DT);error+=abs(data['mechanical_error'])
            check.r=r.r.copy();check.Z=r.Z
            _,_,end=check.objective(r.phi);tau=end[-1]
            force=check.D[:,1:-1].T@(check.w*tau)
            surface=np.dot(check.w*h(check.B@r.phi)/I,tau)-r.Z/I-1000
            audit.update(endpoint_bulk_force_error_Pa=float(max(abs(force))),endpoint_surface_force_error_Pa=float(abs(surface) if data['V']>0 else max(surface,0)),endpoint_bulk_energy_error=float(abs(data['bulk']-end[0])))
            audits.append(audit)
            history.append(dict(time=step*DT,phi_max=float(max(r.phi)),increment=float(max(r.phi-old)),healing=float(max(old-r.phi)),KKT=float(kkt),I=float(I),fracture=float(r.fracture(r.phi)),release=float(release),absolute_mechanical_error=float(error),**{k:float(v) for k,v in data.items()}))
            fields.append(r.phi.copy());anchors.append(anchor);zs.append(Z)
    except Exception as exc:
        np.savez_compressed(out/'failed.npz',phi=r.phi,lower=old,r=anchor,Z=Z)
        (out/'failure.json').write_text(json.dumps(dict(error=repr(exc),step=step),indent=2))
        raise
    elapsed=time.perf_counter()-start;used=time.process_time()-cpu
    summary=dict(cells=n,degree=degree,normal_points=rule,dt=DT,final=FINAL,onset=onset,initial_max=float(max(initial)),runtime_seconds=elapsed,cpu_seconds=used,peak_memory_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,audit_max={k:max(v[k] for v in audits) for k in audits[0] if k!='time'},initialization_separate=True,phase_release=release,absolute_mechanical_error=error)
    np.savez_compressed(out/'fields.npz',y=r.y,phase=np.array(fields),initial=initial,incoming_r=np.array(anchors),incoming_Z=np.array(zs),final_r=r.r,final_Z=r.Z)
    for name,data in [('history',history),('audits',audits),('summary',summary)]:
        (out/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cells',type=int,required=True);p.add_argument('--degree',type=int,default=2);p.add_argument('--quadrature',type=int,required=True);args=p.parse_args()
    case(args.cells,args.degree,args.quadrature)
