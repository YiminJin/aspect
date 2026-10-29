#!/usr/bin/env python3
"""Joint fixed-phase mass-lumped P1 strip prototype; prescribed strain.

This solves the microforce normal map without a single-valued prescribed-X
point inverse. It is a spatial reference prototype, not the native mixed block.
"""
import json,time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from gradient_point import OUT,WORK,project

class Strip:
    def __init__(self,n,ls,phi=.8):
        self.n=n;self.x=np.linspace(0,.25,n);dx=self.x[1]-self.x[0];self.mass=np.full(n,dx);self.mass[[0,-1]]*=.5
        diagonal=np.full(n,2/dx);diagonal[[0,-1]]=1/dx
        self.L=sp.diags([-np.ones(n-1)/dx,diagonal,-np.ones(n-1)/dx],[-1,0,1]).tocsr()
        self.ls=ls;self.phi=phi;self.h=4*phi*(1+phi)/(1-phi)**2
        self.Xop=2e6*ls*ls*sp.diags(1/self.mass)@self.L
        self.pattern=[self.Xop.getrow(i) for i in range(n)]
    def initial(self):return {key:np.zeros((self.n,k)) for key,k in [('T',3),('A',2),('K',2),('Q',2),('L',1)]}
    def gradient_energy(self,K):return float(2e6*self.ls*self.ls*np.sum(K*(self.L@K)))
    def evaluate(self,old,de,dt,K,tangent=True,pressure=None):
        G=1e6;beta=1/(1+dt/100);b=2*G*beta*(1+1/self.h)
        D=de.copy();D[:,:2]-=np.sum(de[:,:2],axis=1)[:,None]/3;D[:,2]*=.5
        T=beta*(old['T']+2*G*D);delta=K-old['K'];A=beta*(old['A']+delta);Q=2*G*A/self.h
        tau=T.copy();tau[:,0]-=2*G*A[:,0];tau[:,1]+=2*G*A[:,0];tau[:,2]-=2*G*A[:,1]
        X=self.Xop@K;F=np.column_stack([.5*(tau[:,0]-tau[:,1]),tau[:,2]])-Q-X;Z=F+b*delta
        residual=np.empty_like(K);masks=[];rows=[];cols=[];values=[]
        for i in range(self.n):
            f,J,mask=project(Z[i],tau[i],10000. if pressure is None else pressure[i],jac=tangent);masks.append(mask)
            residual[i]=-b*delta[i] if mask==0 else F[i]-f
            if tangent:
                P=J[:,:2];Pt=np.column_stack([J[:,2]-J[:,3],J[:,4]])
                blocks={i:-b*np.eye(2)+2*G*beta*Pt}
                for j,v in zip(self.pattern[i].indices,self.pattern[i].data):blocks[j]=blocks.get(j,np.zeros((2,2)))-(np.eye(2)-P)*v
                for j,block in blocks.items():
                    for u in range(2):
                        for v in range(2):rows.append(2*i+u);cols.append(2*j+v);values.append(block[u,v])
        scale=max(1.,1000.,float(np.max(np.linalg.norm(T,axis=1))),float(np.max(np.linalg.norm(X,axis=1))))
        result={'T':T,'A':A,'Q':Q,'K':K.copy(),'W':delta/dt,'tau':tau,'X':X,'F':F,'masks':np.array(masks),
                'residual':residual,'norm':float(np.max(np.abs(residual)))/scale,'force_scale':scale,
                'L':old['L']+2*np.linalg.norm(delta,axis=1)[:,None]}
        if tangent:result['J']=sp.coo_matrix((values,(rows,cols)),shape=(2*self.n,2*self.n)).tocsr()
        return result
    def trial(self,old,de,dt,guess=None,pressure=None):
        K=old['K'].copy() if guess is None else guess.copy();evaluations=0
        for iteration in range(80):
            current=self.evaluate(old,de,dt,K,True,pressure);evaluations+=1
            if current['norm']<=1e-12:
                # Exact distribution of elastic K=K_n equations, followed by
                # reevaluation of ALL forces. No stress/component bounds.
                K[current['masks']==0]=old['K'][current['masks']==0]
                check=self.evaluate(old,de,dt,K,True,pressure);evaluations+=1
                if check['norm']<=1e-12:
                    check.update(iterations=iteration,evaluations=evaluations);return check
                current=check
            direction=spsolve(current['J'],-current['residual'].ravel()).reshape(self.n,2)
            direction[current['masks']==0]=old['K'][current['masks']==0]-K[current['masks']==0]
            found=False
            for line in range(30):
                candidate=K+2.**(-line)*direction
                try:
                    other=self.evaluate(old,de,dt,candidate,False,pressure);evaluations+=1
                except RuntimeError:continue
                if other['norm']<current['norm']*(1-1e-4*2.**(-line)):
                    K=candidate;found=True;break
            if not found:
                np.savez(WORK/'joint-failed-state.npz',**{f'old_{k}':v for k,v in old.items()},de=de,K=K,direction=direction)
                raise RuntimeError(f'Joint strip line search fails; phi={self.phi},ls={self.ls},N={self.n},iteration={iteration},force_norm={current["norm"]}')
        raise RuntimeError('Joint strip iteration limit; accepted state unchanged')

def norm_tau(a):return a[:,0]**2+a[:,1]**2+(a[:,0]+a[:,1])**2+2*a[:,2]**2
def old_tau(old):
    t=old['T'].copy();t[:,0]-=2e6*old['A'][:,0];t[:,1]+=2e6*old['A'][:,0];t[:,2]-=2e6*old['A'][:,1];return t
def account(mesh,old,new,de,dt):
    m=mesh.mass;h=mesh.h;tau=new['tau'];tau0=old_tau(old);q2=2*np.sum(new['Q']**2,axis=1);q02=2*np.sum(old['Q']**2,axis=1)
    storage=m@((norm_tau(tau)+h*q2-norm_tau(tau0)-h*q02)/4e6)+mesh.gradient_energy(new['K'])-mesh.gradient_energy(old['K'])
    Db=m@norm_tau(tau)/2e8;Dc=m@(h*q2)/2e8;Df=m@(2*np.sum(new['F']*new['W'],axis=1))
    work=m@np.sum(tau*de,axis=1)
    loss=m@(norm_tau(tau-tau0)+h*2*np.sum((new['Q']-old['Q'])**2,axis=1))/4e6+mesh.gradient_energy(new['K']-old['K'])
    error=work-storage-dt*(Db+Dc+Df)-loss
    assert abs(error)<=1e-10,(error,storage,work)
    assert Db>=0 and Dc>=0 and Df>=0,(Db,Dc,Df)
    pmin=min(10000.-.5*(tau[:,0]+tau[:,1])-np.hypot(.5*(tau[:,0]-tau[:,1]),tau[:,2]));assert pmin>0
    return {'storage_change':float(storage),'Db':float(Db),'Dc':float(Dc),'Df':float(Df),'work':float(work),'BE_loss':float(loss),'ledger_error':float(error),'pmin':float(pmin)}

if __name__=='__main__':
    mesh=Strip(33,.006,.8);old=mesh.initial();dt=.00625;records=[];start=time.monotonic()
    for step in range(1,41):
        time_s=step*dt;de=np.zeros((mesh.n,3))
        if time_s<=.1+1e-12:de[:,2]=.02*dt*(1+.3*np.cos(2*np.pi*mesh.x/.25))
        elif time_s<=.125+1e-12:de[:,0]=.02*dt;de[:,1]=-.02*dt
        elif time_s>.15+1e-12:de[:,2]=-.02*dt*(1+.3*np.cos(2*np.pi*mesh.x/.25))
        new=mesh.trial(old,de,dt);energy=account(mesh,old,new,de,dt)
        records.append({'step':step,'time_s':time_s,'force_norm':new['norm'],'iterations':new['iterations'],'peak_rate':float(2*max(np.linalg.norm(new['W'],axis=1))),**energy})
        old={k:new[k].copy() for k in old}
        (WORK/'joint-progress.json').write_text(json.dumps(records,indent=2)+'\n')
    (OUT/'joint-smoke.json').write_text(json.dumps({'case':'N33 ls.006 phi.8 dt.00625','steps':records,'seconds':time.monotonic()-start},indent=2)+'\n')
    print('Joint microforce prototype finished40 intervals',time.monotonic()-start)
