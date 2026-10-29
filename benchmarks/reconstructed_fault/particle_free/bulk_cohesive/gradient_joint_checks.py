#!/usr/bin/env python3
"""Bounded joint-field prototype, dispersion, tangent and history controls."""
import csv,json,time
import numpy as np
import scipy.linalg as la
from gradient_joint import Strip,account
from gradient_point import OUT,WORK,point,initial,project

def write(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def save(name,r):(OUT/name).write_text(json.dumps(r,indent=2)+'\n')

def state_from_point(s,n):
    return {'T':np.tile(s[:3],(n,1)),'A':np.tile(s[3:5],(n,1)),
            'K':np.tile(s[5:7],(n,1)),'Q':np.tile(s[7:9],(n,1)),'L':np.full((n,1),s[9])}

def dispersion():
    old=initial(.8)
    for step in range(1,8):r,_=point(old,[0,0,.00025],.0125);old=r[:11]
    de=np.array([0.,0.,.00025]);r,J=point(old,de,.0125);U=.0125*J[14:16];rows=[];homogeneous=0.
    for ls in [.003,.006]:
        mesh=Strip(65,ls);incoming=state_from_point(old,mesh.n);base=mesh.trial(incoming,np.tile(de,(mesh.n,1)),.0125)
        homogeneous=max(homogeneous,float(np.max(abs(base['K']-r[5:7])))/.001)
        for mode in [1,2,4,8,16]:
            wave=np.cos(mode*np.pi*mesh.x/.25);lambda_h=4*np.sin(mode*np.pi/(2*(mesh.n-1)))**2/(mesh.x[1]-mesh.x[0])**2
            amplitude=np.array([.0003,-.0003,.0002]);epsilon=1e-5
            predicted=np.linalg.solve(np.eye(2)-2*ls*ls*lambda_h*U[:,4:6],U[:,:3]@amplitude)
            variation=epsilon*wave[:,None]*amplitude
            plus=mesh.trial(incoming,np.tile(de,(mesh.n,1))+variation,.0125,base['K'])
            minus=mesh.trial(incoming,np.tile(de,(mesh.n,1))-variation,.0125,base['K'])
            actual=(plus['K']-minus['K'])/(2*epsilon);target=wave[:,None]*predicted
            error=np.sqrt(np.sum(mesh.mass[:,None]*(actual-target)**2)/np.sum(mesh.mass[:,None]*target**2))
            rows.append({'ell_s_m':ls,'mode':mode,'continuum_wavelength_m':.5/mode,'discrete_eigenvalue_per_m2':lambda_h,
                         'relative_full_tensor_increment_error':float(error),'forward_force_error':max(plus['norm'],minus['norm'])})
    assert homogeneous<=1e-12,homogeneous
    assert max(r['relative_full_tensor_increment_error'] for r in rows)<=1e-6,rows
    write('joint-dispersion.csv',rows);return homogeneous,max(r['relative_full_tensor_increment_error'] for r in rows)

def tangent(mesh,old,de,dt,new):
    x=mesh.x;v=np.column_stack([1e-4*np.sin(2*np.pi*x/.25),3e-5*np.cos(np.pi*x/.25)])
    d=np.column_stack([.0003*np.cos(np.pi*x/.25),-.0003*np.cos(np.pi*x/.25),.0002*np.sin(np.pi*x/.25)])
    dp=300*np.cos(2*np.pi*x/.25);beta=1/(1+dt/100);b=2e6*beta*(1+1/mesh.h)
    predicted=(new['J']@v.ravel()).reshape(mesh.n,2)
    for i in range(mesh.n):
        Z=new['F'][i]+b*(new['K'][i]-old['K'][i]);_,P,_=project(Z,new['tau'][i])
        dtau=2e6*beta*d[i];dtau[2]*=.5;dtau[:2]-=2e6*beta*sum(d[i,:2])/3
        dF=np.array([.5*(dtau[0]-dtau[1]),dtau[2]])
        predicted[i]+=(np.eye(2)-P[:,:2])@dF-P[:,2:5]@dtau-P[:,5]*dp[i]
    eps=1e-5
    plus=mesh.evaluate(old,de+eps*d,dt,new['K']+eps*v,False,10000+eps*dp)
    minus=mesh.evaluate(old,de-eps*d,dt,new['K']-eps*v,False,10000-eps*dp)
    assert np.array_equal(plus['masks'],new['masks']) and np.array_equal(minus['masks'],new['masks'])
    actual=(plus['residual']-minus['residual'])/(2*eps)
    error=float(np.max(abs(actual-predicted))/max(1000.,float(np.max(abs(predicted)))))
    assert error<=1e-6,error
    return error

def trajectory(n,ls,phi,dt,tag):
    mesh=Strip(n,ls,phi);old=mesh.initial();history=[];profiles=[];start=time.monotonic();fd=None;max_energy=0.;memory=[]
    for step in range(1,round(.25/dt)+1):
        time_s=step*dt;de=np.zeros((n,3))
        if time_s<=.1+1e-12:de[:,2]=.02*dt*(1+.3*np.cos(2*np.pi*mesh.x/.25))
        elif time_s<=.125+1e-12:de[:,0]=.02*dt;de[:,1]=-.02*dt
        elif time_s>.15+1e-12:de[:,2]=-.02*dt*(1+.3*np.cos(2*np.pi*mesh.x/.25))
        before={k:v.copy() for k,v in old.items()};new=mesh.trial(old,de,dt);energy=account(mesh,old,new,de,dt);max_energy=max(max_energy,abs(energy['ledger_error']))
        assert all(np.array_equal(old[k],v) for k,v in before.items())
        if abs(time_s-.1125)<1e-12:fd=tangent(mesh,old,de,dt,new)
        speed=2*np.linalg.norm(new['W'],axis=1);mean=mesh.mass@speed/.25
        width=float((mesh.mass@speed)**2/(mesh.mass@(speed*speed))) if any(speed) else .25
        history.append({'case':tag,'N':n,'ell_s_m':ls,'phi':phi,'dt_s':dt,'time_s':time_s,'force_norm':new['norm'],'iterations':new['iterations'],
            'mean_slip_rate_s-1':float(mean),'peak_slip_rate_s-1':float(max(speed)),'participation_width_m':width,'gradient_storage_J_m-2':mesh.gradient_energy(new['K']),**energy})
        if any(abs(time_s-t)<1e-12 for t in [.1,.1125,.125,.15,.2,.25]):
            for i in range(n):profiles.append({'case':tag,'time_s':time_s,'x_m':mesh.x[i],'weight_m':mesh.mass[i],
                'Kxx':new['K'][i,0],'Kxy':new['K'][i,1],'Wxx_s-1':new['W'][i,0],'Wxy_s-1':new['W'][i,1],
                'tauxx_Pa':new['tau'][i,0],'tauyy_Pa':new['tau'][i,1],'tauxy_Pa':new['tau'][i,2],
                'Xxx_Pa':new['X'][i,0],'Xxy_Pa':new['X'][i,1],'active':int(new['masks'][i])})
        # One exact discard/retry and a serialization fork, preserving every
        # prototype history. This is not a native publication/restart claim.
        if step==16 and n==65 and ls==.006 and phi==.8 and dt==.00625:
            retry=mesh.trial(old,de,dt)
            assert all(np.array_equal(retry[k],new[k]) for k in ['T','A','K','Q','L'])
            np.savez(WORK/'joint-prototype-parent.npz',**old)
            loaded=dict(np.load(WORK/'joint-prototype-parent.npz'));restarted=mesh.trial(loaded,de,dt)
            assert all(np.array_equal(restarted[k],new[k]) for k in ['T','A','K','Q','L']);memory.append('Discard/retry and same-layout NPZ parent fork bitwise on all five tensor/scalar arrays')
        old={k:new[k].copy() for k in old}
        (WORK/'joint-case-progress.json').write_text(json.dumps({'case':tag,'step':step,'time_s':time_s,'completed_cases_pending':True})+'\n')
    return history,profiles,{'case':tag,'seconds':time.monotonic()-start,'max_energy_defect':max_energy,'coupled_pressure_K_tangent_FD':fd,'prototype_lifecycle':memory}

if __name__=='__main__':
    identity,wave=dispersion();history=[];profiles=[];costs=[]
    cases=[(33,0.,.8,.00625,'local-n33'),(33,.006,.8,.00625,'grad-n33'),(65,.006,.8,.00625,'grad-n65'),(129,.006,.8,.00625,'grad-n129'),
           (65,.003,.8,.00625,'grad3-n65'),(65,.006,.8,.003125,'grad-n65-half'),(65,.006,.99,.00625,'deep-grad-n65')]
    for args in cases:
        h,p,c=trajectory(*args);history+=h;profiles+=p;costs.append(c)
        save('joint-checks-progress.json',{'dispersion':wave,'homogeneous':identity,'completed':costs})
    write('joint-history.csv',history);write('joint-profiles.csv',profiles)
    save('joint-checks.json',{'homogeneous_K_identity':identity,'discrete_wavelength_response_relative_error':wave,'cases':costs,
        'scope':'Joint microforce normal-map strip; fixed prescribed strain, nodal mass-lumped P1, microfree. This is not native mixed mechanical resolution.'})
    print(json.dumps(costs,indent=2))
