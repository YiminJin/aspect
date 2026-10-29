#!/usr/bin/env python3
"""Path-restricted seeded formation in exactly condensed mixed Maxwell layers."""
import json,time
import numpy as np
from numpy.polynomial.legendre import leggauss
from cohesive_interface import *
from check_cohesive_interface import serialize,restore


def kernel_checks():
    rows=[]
    # Integrate bulk strain energy independently of the interface traction formula.
    for k in [np.pi,20.,100.]:
        h=.075;H=k*h;ex=np.exp(-H)
        M=np.array([[1,0,ex,H*ex],[-1,1,ex,(H-1)*ex],[ex,H*ex,1,0],[-ex,(1-H)*ex,1,-1]],complex)
        u=np.array([.0001+.00003j,-.00002+.00004j])
        a,b,c,d=np.linalg.solve(M,[1j*u[1]/k,u[0]/k,0,0])
        estimates=[]
        for order in [24,48]:
            nodes,weights=leggauss(order);y=(nodes+1)*h/2;z=k*y;w=H-z
            psi=(a+b*z)*np.exp(-z)+(c+d*w)*np.exp(-w)
            dz=(-a+b-b*z)*np.exp(-z)+(c+d*(w-1))*np.exp(-w)
            dzz=(a-2*b+b*z)*np.exp(-z)+(c+d*(w-2))*np.exp(-w)
            E=G*np.dot(weights*h/2,2*k*k*abs(k*dz)**2+.5*abs(k*k*(dzz+psi))**2)
            estimates.append(float(E))
        boundary=float(.5*np.vdot(u,layer_DtN(k,h)@u).real)
        err=abs(estimates[-1]-boundary)/boundary
        assert err<1e-10,(k,err)
        # Fully coupled normal pressure p' - tau_yy, independently from psi.
        psi0=a+(c+d*H)*ex;z0=-a+b+(c+d*(H-1))*ex
        z3=-a+3*b+(c+d*(H-3))*ex
        pressure=-1j*G*k*k*(z3-z0);tauyy=-2j*G*k*k*z0
        normal=(layer_DtN(k,h)@u)[1]
        assert abs(normal-(pressure-tauyy))<1e-8
        rows.append(dict(k_inv_m=k,energy_volume_J_m2=estimates[-1],energy_boundary_J_m2=boundary,relative_error=err,quadrature_change=abs(estimates[-1]-estimates[0]),pressure_identity_Pa=float(abs(normal-pressure+tauyy))))
    # Work adjoint for a normalized geometric lift: no extra fracture energy.
    nodes,w=leggauss(24);y=ELL*nodes;w=ELL*w
    raw=(1-(y/ELL)**2)**2;chi=raw/np.dot(w,raw)
    stress=1200+300*y/ELL+50*(y/ELL)**2;v=.0012
    lhs=np.dot(w,stress*chi*v);rhs=v*np.dot(w,stress*chi)
    assert abs(lhs-rhs)<1e-14 and abs(w@chi-1)<1e-14
    write_csv('kernel_checks.csv',rows)
    return dict(volume_boundary_energy_max_relative=max(r['relative_error'] for r in rows),Gamma_adjoint_error=float(abs(lhs-rhs)),Gamma_integral=float(w@chi))


def fronts(model,a,seed,level):
    # Seed-connected component on discrete cells, never count a distant island as advance.
    active=a>=level;center=model.n//2;lo=center-1;hi=center
    while lo>0 and active[lo-1]:lo-=1
    while hi<model.n-1 and active[hi+1]:hi+=1
    return model.x[lo]-model.length/model.n/2,model.x[hi]+model.length/model.n/2


def run(n,dt):
    start=time.perf_counter();model=Layers(n);old=initial(n);seed=abs(model.x)<.125;old['damage'][seed]=1.
    initial_damage=old['damage'].copy();rows=[];profiles=[];cum={k:0. for k in ['endpoint_work','trapezoid_work','fracture_increment','Maxwell','friction','radiation','bulk_BE','surface_remainder']}
    absolute=0.;energyold=0.;onset=None;force=0.;ledger=0.;minimum=float('inf');U=0.;check=False;peak_iter=0
    for step in range(round(.32/dt)):
        t=(step+1)*dt;du=np.full(n,dt*.001 if (step+.5)*dt<.30 else 0.);U+=du[0]
        lifecycle=(not check and t>=.20)
        before=serialize(old) if lifecycle else None
        new=solve(model,old,dt,du);en=account(model,old,new,dt,du)
        if onset is None and np.any(new['damage'][~seed]>0):onset=[step*dt,t]
        if lifecycle:
            # Discard a converged candidate; retry and restart from the unchanged owner.
            assert serialize(old)==before
            retry=solve(model,old,dt,du);assert serialize(retry)==serialize(new)
            checkpoint=OUT/f'tip_checkpoint_n{n}_dt{dt:g}.json'
            checkpoint.write_text(serialize({key:old[key] for key in initial(n)})+'\n')
            restored=restore(checkpoint.read_text());restart=solve(model,restored,dt,du);assert serialize(restart)==serialize(new)
            check=True
        active=new['active'];J=-new['tangent'][np.ix_(active,active)]
        if len(J):minimum=min(minimum,float(np.linalg.eigvalsh(.5*(J+J.T)).min()))
        assert minimum>0,'Incremental strong-monotonicity sufficient check failed'
        for key in cum:cum[key]+=en[key]
        energy=en['bulk']+en['cohesive'];physical=en['Maxwell']+en['friction']+en['radiation']
        absolute+=abs(en['trapezoid_work']-(energy-energyold+en['fracture_increment']+physical));energyold=energy
        force=max(force,new['force_error_Pa'],new['intact_excess_Pa']);ledger=max(ledger,abs(en['ledger_defect']));peak_iter=max(peak_iter,new['iterations'])
        old=commit(old,new);assert old['publications']==step+1
        row=dict(cells=n,dt_s=dt,time_s=t,U_m=U,integrated_new_damage_m=float(model.weights@(old['damage']-initial_damage)),integrated_slip_m2=float(model.weights@old['jump']),
            mean_shear_Pa=float(np.mean(old['stress'])),min_pressure_Pa=float(min(old['pn'])),max_pressure_Pa=float(max(old['pn'])),bulk_J_m=en['bulk'],cohesive_J_m=en['cohesive'],
            max_new_damage=float(max(old['damage'][~seed])),remote_new_damage=float(max(old['damage'][abs(model.x)>.5])),minimum_tangent_Pa_m=minimum,
            **{key+'_J_m':val for key,val in cum.items()},absolute_work_remainder_J_m=absolute)
        for level in [.01,.05,.5,.95]:
            left,right=fronts(model,old['damage'],seed,level);row[f'left_a{level:g}_m']=left;row[f'right_a{level:g}_m']=right
        rows.append(row)
        if step%max(1,round(.01/dt))==max(1,round(.01/dt))-1 or step==0:
            for i,x in enumerate(model.x):
                profiles.append(dict(cells=n,dt_s=dt,time_s=t,x_m=x,initial_damage=float(initial_damage[i]),damage=float(old['damage'][i]),jump_m=float(old['jump'][i]),V_m_s=float(old['V'][i]),Theta=float(old['Theta'][i]),physical_shear_Pa=float(old['stress'][i]),normal_pressure_Pa=float(old['pn'][i]),cohesive_traction_Pa=float(old['q'][i])))
    total_loss=cum['friction']+cum['radiation']+cum['fracture_increment']
    summary=dict(cells=n,dt_s=dt,seconds=time.perf_counter()-start,onset_bracket_s=onset,max_force_error_Pa=force,max_ledger_J_m=ledger,
        minimum_symmetric_tangent_Pa_m=minimum,absolute_remainders_over_formation_and_sliding=absolute/total_loss,
        final=rows[-1],retry_bitwise=check,restart_bitwise=check,publications=old['publications'],maximum_Newton_iterations=peak_iter,reciprocity_roundoff=model.reciprocity_roundoff)
    print(json.dumps(summary),flush=True)
    write_csv(f'tip_history_n{n}_dt{dt:g}.csv',rows);write_csv(f'tip_profiles_n{n}_dt{dt:g}.csv',profiles)
    (OUT/f'tip_summary_n{n}_dt{dt:g}.json').write_text(json.dumps(summary,indent=2)+'\n')
    return rows,summary


def compare(a,b,kind):
    result=[];ratio=round(a[0]['dt_s']/b[0]['dt_s']);bb=b[ratio-1::ratio];assert len(a)==len(bb)
    for key,floor in [('integrated_new_damage_m',1e-10),('integrated_slip_m2',1e-10),('mean_shear_Pa',1e-5),('fracture_increment_J_m',1e-10),('trapezoid_work_J_m',1e-10)]:
        x=np.array([r[key] for r in a]);y=np.array([r[key] for r in bb]);error=max(abs(x-y));scale=max(abs(y));limit=.02 if kind=='spatial' else .01
        result.append(dict(kind=kind,coarse_cells=a[0]['cells'],fine_cells=b[0]['cells'],coarse_dt_s=a[0]['dt_s'],fine_dt_s=b[0]['dt_s'],quantity=key,max_absolute_error=float(error),reference_scale=float(scale),relative_error=float(error/scale),passed=bool(error<=limit*scale+floor)))
    for level in [.01,.05,.5,.95]:
        for side in ['left','right']:
            key=f'{side}_a{level:g}_m'
            error=max(abs(x[key]-y[key]) for x,y in zip(a,bb));bound=2*2/b[0]['cells']
            result.append(dict(kind=kind,coarse_cells=a[0]['cells'],fine_cells=b[0]['cells'],coarse_dt_s=a[0]['dt_s'],fine_dt_s=b[0]['dt_s'],quantity=key,max_absolute_error=float(error),reference_scale=bound,relative_error=float(error/bound),passed=bool(error<=bound+1e-14)))
    return result


def main():
    OUT.mkdir(exist_ok=True,parents=True);checks=kernel_checks();runs={};summaries=[]
    for n,dt in [(128,.002),(256,.002),(256,.001),(512,.001)]:
        rows,meta=run(n,dt);runs[(n,dt)]=rows;summaries.append(meta)
    comparisons=compare(runs[(128,.002)],runs[(256,.002)],'spatial')+compare(runs[(256,.002)],runs[(256,.001)],'temporal')
    refinement=compare(runs[(256,.001)],runs[(512,.001)],'spatial')
    write_csv('tip_refinement.csv',refinement)
    comparisons+=refinement
    write_csv('tip_comparison.csv',comparisons)
    (OUT/'tip_summary.json').write_text(json.dumps(dict(kernel=checks,runs=summaries,comparisons=comparisons),indent=2)+'\n')

if __name__=='__main__':main()
