#!/usr/bin/env python3
"""Independent admissible perturbations for the saved endpoint replay."""
import json
import numpy as np
from profile_endpoint_replay import Endpoint
from profile_temporal_diagnosis import E


def main():
    p=Endpoint(1024)
    replay=np.load(E/'endpoint-replay-1024/trial.npz')
    report={}
    for label,x in [('sequential',p.sequential),('coupled_trial',replay['phase'])]:
        value,g=p.evaluate(x);H=p.hessian(x)
        v=np.cos(np.arange(len(x))*.23)*((x>.05)&(x<.95))
        v/=np.linalg.norm(v)
        checks=[]
        for eps in [1e-4,1e-5,1e-6]:
            xp=x+eps*v;xm=x-eps*v
            assert min(xp)>=0 and min(xm)>=0 and max(xp)<1 and max(xm)<1
            vp,gp=p.evaluate(xp);vm,gm=p.evaluate(xm)
            checks.append(dict(epsilon=eps,energy_derivative_error=float(abs((vp-vm)/(2*eps)-g@v)),
                               Hessian_direction_error=float(max(abs((gp-gm)/(2*eps)-H@v)))))
        free=np.flatnonzero((x>p.lower+4.922e-12)|(g< -1e-11))
        eigen=np.linalg.eigvalsh(H[np.ix_(free,free)])
        report[label]=dict(derivative_checks=checks,minimum_reduced_curvature=float(min(eigen)),
                           negative_eigenvalues=int(sum(eigen<0)),free_nodes=len(free),
                           endpoint_phase_KKT=float(p.r.defect(g,x,p.lower)))
        assert min(a['energy_derivative_error'] for a in checks)<1e-7
        assert min(a['Hessian_direction_error'] for a in checks)<1e-5
    assert report['coupled_trial']['endpoint_phase_KKT']<=1e-11
    assert np.array_equal(p.Ra,replay['incoming_R']) and p.Za==replay['incoming_Z']
    report['original_probe_note']='Initial replay diagnostic perturbed nearly-zero tail nodes outside [0,1], producing invalid finite differences. Retained in its summary; only these support-restricted finite differences qualify the derivative.'
    (E/'endpoint-replay-1024/derivative-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
