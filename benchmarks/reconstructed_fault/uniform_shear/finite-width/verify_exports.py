"""Final all-time export checks using native Q2 traces and resolved inputs."""
import json
import numpy as np
from production import HERE,read
from reference_check import PARAMETERS,reference,dump


def boundary_error(data,loading):
    result=0.
    ys=np.unique(data[:,1])
    for values,boundary in ((ys[:3],-.5),(ys[-3:],.5)):
        mask=np.isin(data[:,1],values)
        rows=data[mask]
        rows=rows[np.lexsort((rows[:,1],rows[:,0]))].reshape(-1,3,4)
        y=rows[:,:,1]
        basis=np.ones_like(y)
        for i in range(3):
            for j in range(3):
                if j!=i: basis[:,i]*=(boundary-y[:,j])/(y[:,i]-y[:,j])
        result=max(result,float(max(abs(np.sum(basis*rows[:,:,2],axis=1)-boundary*loading))),
                   float(max(abs(np.sum(basis*rows[:,:,3],axis=1)))))
    return result


def main():
    original=reference.read_parameters(PARAMETERS)
    prefixes=('Material model/Phase field fault/','Phase field model/',
              'Solver parameters/','Compositional fields/','Particles/',
              'Boundary velocity model/','Time stepping/')
    result={}
    for c in 'ABC':
        path=HERE/f'k42_{c}'
        params=reference.read_parameters(path/'parameters.prm')
        differences={k:[original.get(k),v] for k,v in params.items()
                     if k.startswith(prefixes) and original.get(k)!=v}
        allowed={} if c=='A' else {'Phase field model/Length scale':['0.15625','0.078125']}
        assert differences==allowed,(c,differences)
        rows=[]
        for step in range(49):
            s=read(path,'surface',step)
            guard=json.loads((path/f'k4_guard_{step}.json').read_text())
            data=np.loadtxt(path/f'bulk_{step}.csv',delimiter=',',skiprows=1,usecols=(0,1,3,4))
            t=step*.125
            error=boundary_error(data,1e-4*(1+.2*min(t/4,1)))
            # This homogeneous fixture's Ih is constant along x. Retain the
            # measured FE-column range when checking its consistent projection.
            low,high=guard['independent_Ih_range']
            Ih_error=float(max(max(abs(s['Ih']-low)/low),max(abs(s['Ih']-high)/high)))
            assert error<=1e-13 and Ih_error<=1e-6,(c,step,error,Ih_error)
            rows.append(dict(time_s=t,boundary_error_m_s=error,Ih_projection_relative_error=Ih_error))
        result[c]=dict(approved_physics_parameter_differences=differences,
                       max_boundary_error=max(r['boundary_error_m_s'] for r in rows),
                       max_Ih_projection_error=max(r['Ih_projection_relative_error'] for r in rows),
                       passed=True,rows=rows)
    dump(HERE/'k42-export-verification.json',result)
    print(json.dumps({c:{k:v for k,v in r.items() if k!='rows'} for c,r in result.items()},indent=2))


if __name__=='__main__': main()
