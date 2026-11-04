#!/usr/bin/env python3
"""Independent full mature work balance from pressure-enriched native exports."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
import argparse
import json
import numpy as np
from compare_bounded_profile import load,quadrature,tensor,dot3,energy,OUT

def powers(state,old,initial,G,eta,order=8):
    assert state.values.shape[1]==18,'Use the separate pressure-export checker.'
    x,w=quadrature(state.n,order);v=state.at(x);tau=tensor(state,x)
    e=energy(state,G,eta,order)
    e['pressure_dilatation']=float(w@(v[:,17]*(v[:,12]+v[:,15])))
    X=state.surface[:,1:3];width=state.surface[:,7];I=state.surface[:,5]
    fields=np.zeros((len(x),5));best=np.full(len(x),np.inf)
    background=2.7+.15*np.cos(3*initial.surface[:,1])
    for j in range(len(X)-1):
        d=X[j+1]-X[j];t=d/np.linalg.norm(d);n=np.array([-t[1],t[0]])
        xi=((x-X[j])@d)/(d@d);dist=abs((x-X[j])@n)
        select=(xi>=0)&(xi<=1)&(dist<best)&(dist<=(1-xi)*width[j]+xi*width[j+1])
        q=xi[select];p=v[select,0];interp=lambda a:(1-q)*a[j]+q*a[j+1]
        chi=state.m*p*(1+p)/(1-p)**2/interp(I)
        V=interp(state.surface[:,6]);theta=interp(old.surface[:,3])
        # Native mechanics uses incoming committed age; the exact aging source
        # runs once after acceptance. Do not insert endpoint age in friction.
        mu=.025*np.arcsinh(V/.002*np.exp((.6+.015*np.log(theta*.001/.008))/.025))
        a=tau[select];normal=5+v[select,17]-(a[:,0]*n[0]**2+a[:,1]*n[1]**2+2*a[:,2]*n[0]*n[1])
        shear=t[0]*(a[:,0]*n[0]+a[:,2]*n[1])+t[1]*(a[:,2]*n[0]+a[:,1]*n[1])
        fields[select]=np.stack([chi*V*interp(background),chi*V*mu*normal,chi*V*V,chi*V*shear,chi],axis=1)
        best[select]=dist[select]
    names=['background_power','friction_power','radiation_power','surface_bulk_power','localization_measure']
    e.update({k:float(w@fields[:,i]) for i,k in enumerate(names)})
    e['surface_work_residual']=e['surface_bulk_power']+e['background_power']-e['friction_power']-e['radiation_power']
    e['slip_adjoint_defect']=e['surface_bulk_power']-e['slip_power']
    # Direct face traction power is a diagnostic. The weak boundary reaction
    # power is mechanical_power - pressure_dilatation and should be zero in
    # this pure-translation/no-body-force fixture.
    g,gw=np.polynomial.legendre.leggauss(order);h=state.h
    line=(-2+h*(np.arange(state.n)[:,None]+.5*(g+1))).ravel();weights=np.tile(.5*h*gw,state.n)
    face_power=0.;relative_power=0.;force=np.zeros(2)
    translation=initial.at(np.array([[-2.,0.]]))[0,1:3]
    for axis in [0,1]:
        for side in [-1,1]:
            points=np.zeros((len(line),2));points[:,axis]=2*side;points[:,1-axis]=line
            a=state.at(points);tau=tensor(state,points)
            traction=np.stack([tau[:,0]-a[:,17],tau[:,2]],axis=1) if axis==0 else np.stack([tau[:,2],tau[:,1]-a[:,17]],axis=1)
            face_power+=weights@(side*np.sum(a[:,1:3]*traction,axis=1))
            relative_power+=weights@(side*np.sum((a[:,1:3]-translation)*traction,axis=1))
            force+=np.sum((side*weights)[:,None]*traction,axis=0)
    e['face_mechanical_power']=float(face_power)
    e['weak_mechanical_power']=e['mechanical_power']-e['pressure_dilatation']
    e['mechanical_face_defect']=e['face_mechanical_power']-e['weak_mechanical_power']
    e['face_force_x']=float(force[0]);e['face_force_y']=float(force[1])
    e['rigid_translation_face_power']=float(translation@force)
    e['relative_velocity_face_power']=float(relative_power)
    return e

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cases',nargs='+');p.add_argument('--output',default='coupled-work.json');a=p.parse_args();out=[]
    for name in a.cases:
        command,rows,meta,states=load(name);assert command['arguments']['shear']==0
        initial=states[min(states)];terms=[]
        cumulative=absolute=active=gross=0.;previous=None
        for row in rows:
            step=int(row['step']);dt=float(row['dt']);state=states[step];old=states.get(step-1,state)
            e=powers(state,old,initial,meta['G'],command['arguments']['eta'])
            e.update(step=step,time=float(row['time']),dt=dt)
            if previous is not None:
                # Friction uses old age and endpoint V: retain that endpoint
                # convention. The bulk-only trapezoidal diagnostic is separate.
                rhs=e['mechanical_power']+e['background_power']-e['friction_power']-e['radiation_power']-e['maxwell_loss']+e['dilatation_power']-e['flux']
                e['coupled_endpoint_remainder']=e['storage']-previous['storage']-dt*rhs
                cumulative+=e['coupled_endpoint_remainder'];absolute+=abs(e['coupled_endpoint_remainder'])
                active+=.5*dt*sum(abs(z[k]) for z in [e,previous]
                    for k in ['mechanical_power','slip_power','maxwell_loss','dilatation_power','flux'])
                gross+=dt*sum(abs(e[k]) for k in ['background_power','friction_power','radiation_power','maxwell_loss','flux'])
            terms.append(e);previous=e
        end=max(states);q12=powers(states[end],states[end-1],initial,meta['G'],command['arguments']['eta'],12)
        out.append(dict(case=name,steps=terms,signed=cumulative,absolute=absolute,active_scale=active,gross_scale=gross,
                        relative_active=absolute/active,relative_gross=absolute/gross,
                        final_q12_minus_q8={k:q12[k]-terms[-1][k] for k in q12}))
        print(name,'signed',cumulative,'relative_active',absolute/active,flush=True)
    (OUT/a.output).write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':main()
