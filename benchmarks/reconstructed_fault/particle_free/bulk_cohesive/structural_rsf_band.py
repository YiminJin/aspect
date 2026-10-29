#!/usr/bin/env python3
"""Exact 1-D mixed equilibrium with mass-lumped P1 fixed-phase history.

Reuses the strip reduction/layout of mature_gradient_strip / gradient_joint,
without the retired K-gradient term. Not a native 2-D mechanical qualification.
"""
import json,time
import numpy as np
from structural_rsf import *
from check_structural_rsf import csvwrite


class Band:
    def __init__(self,cells):
        self.y=np.linspace(-.125,.125,cells+1);self.dx=.25/cells
        self.mass=np.full(cells+1,self.dx);self.mass[[0,-1]]*=.5
        self.phi=np.maximum(np.sqrt(.8)-abs(self.y)/(2*ELL),0.)**2
        self.active=self.phi>0
        self.h=np.zeros(cells+1);p=self.phi[self.active]
        self.h[self.active]=4*p*(1+p)/(1-p)**2
        self.ih=np.zeros(cells+1);self.ih[self.active]=1/self.h[self.active]
        # Nodal mass-lumping of phi + ell² grad(phi)². Do not floor or mask chi.
        gradenergy=np.diff(self.phi)**2/self.dx
        gamma_mass=self.mass*self.phi
        gamma_mass[:-1]+=ELL**2*gradenergy/2;gamma_mass[1:]+=ELL**2*gradenergy/2
        gamma_mass*=3/(8*ELL);self.I=sum(gamma_mass);self.chi=gamma_mass/(self.mass*self.I)
        self.cells=cells

    def initial(self):
        z=np.zeros_like(self.y)
        return dict(T=z.copy(),A=z.copy(),K=z.copy(),Theta=np.ones_like(z),traction=0.,clock=0.,publications=0)

    def trial(self,old,dt,du):
        beta=1/(1+RELAX*dt);active=self.active;mass=self.mass;chi=self.chi
        qref=2*G*beta*old['A']*self.ih
        stiffness=G*beta*dt*chi*self.ih+ETA_RAD
        def evaluate(shear):
            drive=shear-qref
            V,mu,slope,age,error=solve_speed(abs(drive[active]),stiffness[active],P0,old['Theta'][active],dt)
            v=np.zeros_like(self.y);v[active]=np.sign(drive[active])*V
            theta=old['Theta']+V0/DC*dt;theta[active]=age
            friction=np.zeros_like(v);friction[active]=mu
            dK=.5*dt*chi*v;A=beta*(old['A']+dK);T=shear+2*G*A
            dgamma=(T/beta-old['T'])/G
            compliance=np.full_like(v,1/(beta*G));compliance[active]+=dt*chi[active]/slope
            return mass@dgamma-du,mass@compliance,dict(T=T,A=A,K=old['K']+dK,Theta=theta,traction=shear,
                clock=old['clock']+dt,publications=old['publications'],v=v,mu=friction,dgamma=dgamma,rate_error=error)
        shear=old['traction']
        for iteration in range(30):
            residual,slope,new=evaluate(shear)
            if abs(residual)<1e-16:break
            shear-=residual/slope
        else:raise RuntimeError('Mixed equilibrium compatibility did not converge')
        new['compatibility_m']=residual;new['iterations']=iteration+1
        s0=old['traction'];s=new['traction'];A=new['A']
        bulk=.25*s*s/(2*G);bulk0=.25*s0*s0/(2*G)
        cohesive=mass@(2*G*A*A*self.ih);cohesive0=mass@(2*G*old['A']**2*self.ih)
        maxwell=dt*(.25*s*s+mass@(4*G*G*A*A*self.ih))/ETA
        friction=dt*(mass@(chi*abs(new['v'])*P0*new['mu']))
        radiation=dt*(mass@(chi*ETA_RAD*new['v']**2))
        be=.25*(s-s0)**2/(2*G)+mass@(2*G*(A-old['A'])**2*self.ih)
        work=s*du;defect=work-(bulk-bulk0+cohesive-cohesive0+maxwell+friction+radiation+be)
        assert abs(defect)<1e-10*max(1.,abs(work)),defect
        new['energy']=dict(work_J_m2=work,bulk_J_m2=bulk,cohesive_J_m2=cohesive,
            Maxwell_J_m2=maxwell,friction_J_m2=friction,radiation_J_m2=radiation,BE_J_m2=be,ledger_error_J_m2=defect)
        return new


def run(cells,dt):
    start=time.perf_counter();b=Band(cells);old=b.initial();rows=[];profiles=[];cumulative={k:0. for k in ['work_J_m2','Maxwell_J_m2','friction_J_m2','radiation_J_m2','BE_J_m2']}
    for step in range(round(.3/dt)):
        t=(step+1)*dt;du=.001*dt if (step+.5)*dt<.25 else 0.
        new=b.trial(old,dt,du)
        for k in cumulative:cumulative[k]+=new['energy'][k]
        new['publications']+=1
        w=b.chi*new['v'];width=(b.mass@abs(w))**2/(b.mass@(w*w)) if np.any(w) else 0.
        rows.append(dict(cells=cells,dt_s=dt,time_s=t,displacement_m=.001*min(t,.25),traction_Pa=new['traction'],
            slip_m=2*b.mass@new['K'],interface_V_m_s=b.mass@w,rate_width_m=width,
            slip_width_m=(2*b.mass@abs(new['K']))**2/(b.mass@((2*new['K'])**2)) if np.any(new['K']) else 0.,
            bulk_J_m2=new['energy']['bulk_J_m2'],cohesive_J_m2=new['energy']['cohesive_J_m2'],
            fracture_J_m2=b.I,**cumulative,ledger_error_J_m2=new['energy']['ledger_error_J_m2'],
            rate_error_Pa=new['rate_error'],compatibility_m=new['compatibility_m'],iterations=new['iterations']))
        if any(abs(t-time)<1e-12 for time in [.1,.15,.2,.25,.3]):
            for i,y in enumerate(b.y):
                profiles.append(dict(cells=cells,dt_s=dt,time_s=t,y_m=y,phi=b.phi[i],chi_inv_m=b.chi[i],
                    tau_xy_Pa=new['traction'],p_Pa=P0,pn_Pa=P0,Q_xy_Pa=2*G*new['A'][i]*b.ih[i],
                    signed_local_V_m_s=new['v'][i],flow_rate_s_inv=w[i],accumulated_shear=2*new['K'][i],
                    T_xy_Pa=new['T'][i],A_xy=new['A'][i],Theta=new['Theta'][i]))
        old=new
    # Fresh analytic prescription, measured independently on dense physical points.
    fine=np.linspace(-.125,.125,32769);exact=np.maximum(np.sqrt(.8)-abs(fine)/(2*ELL),0.)**2
    interp=np.interp(fine,b.y,b.phi)
    initial_error=np.linalg.norm(interp-exact)/np.linalg.norm(exact)
    meta=dict(cells=cells,nodes=cells+1,dt_s=dt,seconds=time.perf_counter()-start,phase_relative_L2=initial_error,
        I_discrete=b.I,I_exact=.8**1.5,I_relative_error=b.I/(.8**1.5)-1,
        chi_integral=float(b.mass@b.chi),steps=len(rows),max_rate_error_Pa=max(r['rate_error_Pa'] for r in rows),
        max_ledger_error_J_m2=max(abs(r['ledger_error_J_m2']) for r in rows),
        max_compatibility_m=max(abs(r['compatibility_m']) for r in rows),max_equilibrium_iterations=max(r['iterations'] for r in rows))
    print(json.dumps(meta),flush=True)
    return rows,profiles,meta


def compare(a,b,kind):
    ratio=round(a[0]['dt_s']/b[0]['dt_s']);bb=b[ratio-1::ratio]
    select=lambda r:r['time_s']>=.15-1e-12
    a=[r for r in a if select(r)];bb=[r for r in bb if select(r)]
    assert len(a)==len(bb)
    answer=[]
    for field in ['traction_Pa','slip_m','work_J_m2','rate_width_m']:
        x=np.array([r[field] for r in a]);y=np.array([r[field] for r in bb]);error=np.max(abs(x-y))/np.max(abs(y))
        answer.append(dict(kind=kind,coarse_cells=a[0]['cells'],fine_cells=bb[0]['cells'],coarse_dt_s=a[0]['dt_s'],fine_dt_s=bb[0]['dt_s'],
            field=field,normalized_max_error=error,endpoint_relative_difference=abs(x[-1]-y[-1])/abs(y[-1])))
    return answer


def main():
    assert json.loads((OUT/'point_summary.json').read_text())['passed']
    allrows=[];allprofiles=[];metas=[];runs={};dt=.00078125
    for cells,step in [(64,dt),(128,dt),(256,dt),(256,dt/2)]:
        rows,profiles,meta=run(cells,step);runs[(cells,step)]=rows;allrows+=rows;allprofiles+=profiles;metas.append(meta)
        csvwrite('band_history.csv',allrows);csvwrite('band_profiles.csv',allprofiles)
        (OUT/'band_progress.json').write_text(json.dumps(metas,indent=2)+'\n')
    comparisons=compare(runs[(64,dt)],runs[(128,dt)],'spatial')+compare(runs[(128,dt)],runs[(256,dt)],'spatial')+compare(runs[(256,dt)],runs[(256,dt/2)],'temporal')
    temporal=all(r['normalized_max_error']<=(.005 if r['field']=='work_J_m2' else .0025) for r in comparisons if r['kind']=='temporal')
    spatial=all(r['normalized_max_error']<=.01 for r in comparisons if r['kind']=='spatial' and r['coarse_cells']==128)
    decrease=all(comparisons[i+4]['normalized_max_error']<comparisons[i]['normalized_max_error'] for i in range(4))
    # One further finest-only half step is permitted, but it cannot repair a failed common-dt spatial control.
    if not temporal:
        rows,profiles,meta=run(256,dt/4);allrows+=rows;allprofiles+=profiles;metas.append(meta)
        comparisons+=compare(runs[(256,dt/2)],rows,'additional_temporal')
        # Match the other two meshes to the verified half timestep.
        for cells in [64,128]:
            matched,profile,meta=run(cells,dt/2);runs[(cells,dt/2)]=matched
            allrows+=matched;allprofiles+=profile;metas.append(meta)
        comparisons+=compare(runs[(64,dt/2)],runs[(128,dt/2)],'matched_spatial')
        comparisons+=compare(runs[(128,dt/2)],runs[(256,dt/2)],'matched_spatial')
    csvwrite('band_history.csv',allrows);csvwrite('band_profiles.csv',allprofiles);csvwrite('band_comparison.csv',comparisons)
    result=dict(passed=bool(temporal and spatial and decrease),temporal_passed=bool(temporal),spatial_passed=bool(spatial),decreasing_mesh_differences=bool(decrease),runs=metas,comparisons=comparisons)
    result['selected_temporal_passed']=all(r['normalized_max_error']<=(.005 if r['field']=='work_J_m2' else .0025)
        for r in comparisons if r['kind']==('additional_temporal' if not temporal else 'temporal'))
    result['matched_spatial_magnitude_passed']=all(r['normalized_max_error']<=.01
        for r in comparisons if r['kind']=='matched_spatial' and r['coarse_cells']==128)
    result['work_difference_monotonicity_passed']=False if not decrease else True
    (OUT/'band_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['runs','comparisons']}))

if __name__=='__main__':main()
