#!/usr/bin/env python3
"""Frozen FE operator inputs and a verified identical native timestep schedule."""
import numpy as np
import native_endpoint as native
import asymmetric_branch as old
from run_native_fe import flatten, ROOT

E=native.E


def prepare():
    target=E/'operator-input';target.mkdir(exist_ok=False)
    p=native.SharedFE();r=p.r
    data=dict(np.load(E/'dt-1-1024/fields.npz'))
    history=old.read(E/'dt-1-1024/history.json')
    np.savetxt(E/'shared-initial-phase.txt',np.c_[r.y,r.initial],fmt='%.17g')
    np.savetxt(E/'shared-initial-H.txt',np.c_[r.initial_particle_y,r.initial_particle_H],fmt='%.17g')
    manifest=[]
    for label,index in [('unloaded',0),('growing',560),('asymmetric',608)]:
        time=index/1024;dt=1/1024
        lower=data['phase'][max(0,index-1)]
        Ra=data['R'][max(0,index-1)]
        Za=float(data['Z'][max(0,index-1)])
        slip=history[index-2]['slip'] if index>1 else 0.
        p.incoming(lower,Ra,Za,slip,time if index else dt/2,dt)
        phi=data['phase'][index]
        direction=np.sin(np.pi*r.y)*phi
        direction/=max(abs(direction))
        expected_action=p.hessian(phi)@direction
        np.savetxt(target/f'{label}-action.txt',expected_action,fmt='%.17g')
        for offset in [0.,1e-5,-1e-5,5e-6,-5e-6]:
            x=phi+offset*direction
            potential,g=p.evaluate(x);info=p.last[3];Rb=p.last[4];Zb=p.last[5];u=r.velocity
            I,J=r.direct(x)
            name=f'{label}-{offset:+g}.txt'
            with (target/name).open('w') as out:
                out.write(' '.join('%.17g'%float(v) for v in [r.n,time,dt,Za,Zb,info['V'],I,
                    info['bulk'],info['cohesive'],info['fracture'],potential*old.A,J@direction])+'\n')
                for array in [x,lower,Ra,Rb,u,g,direction]:
                    out.write(' '.join('%.17g'%float(v) for v in array)+'\n')
            manifest.append(dict(file=name,label=label,offset=offset,index=index,
                                 mechanical_force=info['force'],contact=info['contact']))
    old.write(target/'manifest.json',manifest)
    (target/'files.txt').write_text('\n'.join(row['file'] for row in manifest)+'\n')
    base=ROOT/'benchmarks/reconstructed_fault/propagation/normal_resolution_evidence/native.prm'
    lines=flatten(base)
    lines=[line.replace('P2 native FE strain','P2 native endpoint') for line in lines]
    shared=f'''set Additional shared libraries = {ROOT}/build-propagation-integration/plugin/libnative_endpoint_verification.release.so
subsection Geometry model
 subsection Box
  set X repetitions = 1
  set Y repetitions = 32
 end
end
subsection Material model
 subsection Phase field fault
  set Consistent endpoint phase solve = true
  set FE initial phase profile file = {E}/shared-initial-phase.txt
 end
end
subsection Time stepping
 set List of model names = function
 subsection Function
  set Variable names = time
  set Function expression = if(time<0.5,0.015625,0.0009765625)
 end
end
subsection Postprocess
 set List of postprocessors = particles, P2 native endpoint
 subsection P2 native endpoint
  set Initial history file = {E}/shared-initial-H.txt
 end
end
'''
    operator=f'''set Output directory = {ROOT}/build-propagation-integration/runs/output-native-endpoint-operators
set End time = 0.0009765625
set Maximum time step = 0.0009765625
subsection Boundary velocity model
 subsection Function
  set Function expression = 0; 0
 end
end
subsection Time stepping
 subsection Function
  set Function expression = 0.0009765625
 end
end
subsection Postprocess
 subsection P2 native endpoint
  set Operator input directory = {target}
 end
end
'''
    (E/'operators.prm').write_text('\n'.join(lines)+'\n'+shared+operator)
    evolution=f'''set Output directory = {ROOT}/build-propagation-integration/runs/output-native-endpoint-prefix
set End time = 0.53125
set Maximum time step = 0.015625
subsection Boundary velocity model
 subsection Function
  set Function expression = 0.256*max(0,t-if(t<=0.5,0.0078125,0.00048828125))*(y-0.5); 0
 end
end
'''
    (E/'native-prefix.prm').write_text('\n'.join(lines)+'\n'+shared+evolution)
    onset='''set Maximum time step = 0.5
subsection Time stepping
 subsection Function
  set Function expression = if(time<0.5,0.5,if(time<0.52734375,0.02734375,0.0009765625))
 end
end
subsection Boundary velocity model
 subsection Function
  set Function expression = 0.256*max(0,t-if(t<=0.5,0.25,if(t<=0.52734375,0.013671875,0.00048828125)))*(y-0.5); 0
 end
end
'''
    (E/'native-onset.prm').write_text('\n'.join(lines)+'\n'+shared+evolution+onset)
    # These are operator inputs, not converted checkpoints or initial histories.
    old.write(E/'operator-input-definition.json',dict(material='unchanged reference',
              initialization='retained continuum seed, immutable previous physical phase',
              stage='fixed-state operator tests; no accepted history is imported',
              points=['unloaded','growing .546875 s','asymmetric .59375 s']))


if __name__=='__main__':prepare()
