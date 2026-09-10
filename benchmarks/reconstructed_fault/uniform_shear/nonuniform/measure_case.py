#!/usr/bin/env python3
"""K2 fixed-profile allowances and the actual particle/Q1 surface balance."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import boundary_error, read
from analyze_convergence import profile_primitive
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'linear-correction'))
from summarize import summarize


def friction(v, theta):
    return .025*np.arcsinh(v/(2e-5)*np.exp((.6+.013*np.log(theta*.01))/.025))


def project(segments, xi, volume, samples, n):
    """The same particle-volume Q1 weak measure, not a smoothing operation."""
    shape = np.column_stack((1-xi,xi))
    mass, rhs = np.zeros((n,n)), np.zeros((n,samples.shape[1]))
    for i in range(2):
        for j in range(2):
            np.add.at(mass,(segments+i,segments+j),volume*shape[:,i]*shape[:,j])
        np.add.at(rhs,segments+i,(volume*shape[:,i])[:,None]*samples)
    return mass, rhs, np.linalg.solve(mass,rhs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('log',type=Path)
    p.add_argument('--initial-traction',type=Path)
    p.add_argument('--initial-only',action='store_true')
    args=p.parse_args()
    dest=args.directory.parent/(args.directory.name+'-measurements')
    dest.mkdir(exist_ok=True)
    get=lambda name,k: read(args.directory,name,k,())
    phase=get('phase',0)
    xs,ys=np.unique(phase['x']),np.unique(phase['y'])
    profile=np.sort(np.unique(phase[phase['x']==xs[len(xs)//2]][['y','phi']]),order='y')
    segments=get('segments',0)
    if max(abs(segments['nx']))>1e-12:
        raise ValueError('K2 diagnostic requires the measured horizontal fixture')
    width=float(segments['half_width_plus'][0])
    prim,integral=profile_primitive(profile['y'],profile['phi'],np.array([-width,width]))
    omitted=float(1-(prim[1]-prim[0])/integral)
    profile_integrals, omissions = [], []
    for coordinate in xs:
        column=np.sort(np.unique(phase[phase['x']==coordinate][['y','phi']]),order='y')
        panel,total=profile_primitive(column['y'],column['phi'],np.array([-width,width]))
        profile_integrals.append(total)
        omissions.append(1-(panel[1]-panel[0])/total)
    omitted=float(max(omissions))
    report=dict(mesh=[len(xs)-1,len(ys)-1],support_half_width_m=width,
                independent_Ih_m=float(integral),omitted_fraction=omitted,
                original_containment_target=1e-6,approved_family_allowance=1e-4,
                actual_normalization_requirement=1e-4,
                phi_center=float(np.interp(0,profile['y'],profile['phi'])),steps=[])
    report['independent_Ih_range_m']=[float(min(profile_integrals)),float(max(profile_integrals))]
    report['measured_profile_locations']=len(xs)
    initial=get('surface',0)
    x=initial['x']; weights=np.r_[np.diff(x)/2,0]+np.r_[0,np.diff(x)/2]
    initial_C=initial['C'].copy()
    old_C,old_theta=initial_C.copy(),initial['Theta'].copy()
    slip=np.zeros(len(x))
    logged=summarize(args.log)
    if args.initial_only:
        logged=logged[:1]
    for k,iteration in enumerate(logged):
        time=get('time',k)[0]; dt=float(time['dt'])
        surface=get('surface',k); bulk=get('bulk',k); particles=get('particles',k)
        if not np.array_equal(get('phase',k),phase):
            raise ValueError('The initial Q1 phase field changed')
        associated=particles[particles['active']==1]
        s=associated['segment'].astype(int); xi=associated['xi']; volume=associated['volume']
        interp=lambda nodal:(1-xi)*nodal[s]+xi*nodal[s+1]
        v,theta,ih=interp(surface['V']),interp(old_theta),interp(surface['Ih'])
        C=-1e8*math.expm1(-dt/100)/ih*v+math.exp(-dt/100)*interp(old_C)
        mu=friction(v,theta); radiation=1e5*v
        if k==0:
            initial_path=args.initial_traction or args.directory/'particle_traction_initial_0.csv'
            evaluated=np.atleast_1d(np.genfromtxt(initial_path,names=True,delimiter=','))
            evaluated=np.sort(evaluated,order='id'); order=np.argsort(associated['id'])
            if not np.array_equal(evaluated['id'],associated['id'][order]):
                raise ValueError('Initial evaluated traction has different particles')
            for name in ('x','y','segment','xi','volume'):
                if max(abs(evaluated[name]-associated[name][order]))>1e-13:
                    raise ValueError('Initial diagnostic associations differ: '+name)
            q=np.empty(len(v)); q[order]=evaluated['q']
            production_F=np.empty(len(v)); production_F[order]=evaluated['F']
        else:
            # These are production-published accepted stresses, not FE old stress
            # or a normal-column average. Geometry is measured, not assumed.
            tangent=np.column_stack((np.diff(surface['x']),np.diff(surface['y'])))
            tangent/=np.linalg.norm(tangent,axis=1)[:,None]
            tx,ty=tangent[s,0],tangent[s,1]
            q=(-associated['tau_xx']*tx*ty+associated['tau_yy']*ty*tx
               +associated['tau_xy']*(tx*tx-ty*ty))
        F=q-C-1000*mu-radiation
        samples=np.column_stack((q,C,1000*mu,radiation,F))
        integrated_weak=(args.directory/f'surface_weak_{k}.csv').exists()
        if integrated_weak:
            # Frozen pre-publication weak terms are the actual surface equation.
            # Newly published parent stress is a separate raw-history diagnostic,
            # not a substitute for the domain-varying accepted traction.
            weak=get('surface_weak',k)
            if len(weak)!=len(x) or not np.array_equal(weak['node'],np.arange(len(x))):
                raise ValueError('Unexpected integrated weak-vector layout')
            mass=np.diag(weak['Mdiag'])+np.diag(weak['Moff'][:-1],1)+np.diag(weak['Moff'][:-1],-1)
            rhs=np.column_stack([weak[name] for name in ('q','C','friction','damping','F')])
            if max(abs(rhs[:,0]-rhs[:,1]-rhs[:,2]-rhs[:,3]-rhs[:,4]))>1e-10:
                raise ValueError('Integrated weak traction terms do not close')
            projection=np.linalg.solve(mass,rhs)
            if abs(mass.sum()-sum(volume))>1e-10*sum(volume):
                raise ValueError('Integrated mass differs from full admitted domain volume')
        else:
            mass,rhs,projection=project(s,xi,volume,samples,len(x))
        strong=float(np.sqrt(max(0,rhs[:,-1]@projection[:,-1])/mass.sum()))
        production_rms=iteration['nonlinear'][-1]['surface']
        if abs(strong-production_rms)>1e-9:
            raise ValueError(f'Particle balance differs from production: step {k}: {strong} vs {production_rms}')
        if k==0 and not integrated_weak and max(abs(F-production_F))>1e-9:
            raise ValueError('Initial independent balance differs from production point response')
        bx,index=np.unique(bulk['x'],return_inverse=True)
        normal=np.bincount(index,weights=bulk['weight']*(bulk['chi']*bulk['V']+bulk['history']))
        normal/=np.bincount(index,weights=bulk['weight'])
        ratios=normal/np.interp(bx,x,surface['V'])
        global_ratio=float(np.dot(bulk['weight'],bulk['chi']*bulk['V']+bulk['history'])/np.dot(weights,surface['V']))
        if k:
            slip+=dt*surface['V']
            expected_theta=old_theta*np.exp(-surface['V']*dt/.001)+.001/surface['V']*(-np.expm1(-surface['V']*dt/.001))
        else:
            expected_theta=old_theta
        row=dict(time_s=float(time['time']),dt_s=dt,normalization_max_error=float(max(abs(ratios-1))),
                 surface_rule='domain integrated' if integrated_weak else 'point volume',
                 raw_sample_semantics='published parent history (evaluated parent at zero)',
                 global_normalization=global_ratio,Ih_relative_error=float(max(abs(surface['Ih']/integral-1))),
                 raw_particle_q_min=float(min(q)),raw_particle_q_max=float(max(q)),
                 raw_F_rms_Pa=float(np.sqrt(np.average(F**2,weights=volume))),
                 strong_F_rms_Pa=strong,production_F_rms_Pa=production_rms,
                 balance_rms_difference_Pa=abs(strong-production_rms),
                 theta_update_error_s=float(max(abs(surface['Theta']-expected_theta))),
                 boundary_error_m_s=boundary_error(bulk,float(time['U'])),
                 volume_sum_m2=float(sum(particles['volume'])),min_volume_m2=float(min(particles['volume'])))
        np.savetxt(dest/f'weak_balance_{k}.csv',np.column_stack((x,np.diag(mass),np.r_[np.diag(mass,1),0],rhs)),
                   delimiter=',',header='s,Mdiag,Moff,q,C,friction,damping,F',comments='')
        report['steps'].append(row)
        # Weak projections describe the surface equation; raw samples remain
        # separate and are never altered or used to modify the solve.
        np.savetxt(dest/f'surface_balance_{k}.csv',np.column_stack((x,projection,surface['V'],surface['Theta'],surface['C'],slip)),
                   delimiter=',',header='s,particle_q_Q1,C_evaluated_Q1,friction_Q1,radiation_Q1,F_Q1,V,Theta,C_retained,slip',comments='')
        np.savetxt(dest/f'particle_balance_{k}.csv',np.column_stack((associated['id'],associated['x'],associated['y'],s,xi,volume,samples)),
                   delimiter=',',header='id,x,y,segment,xi,volume,q,C_evaluated,friction,radiation,F',comments='')
        old_C,old_theta=surface['C'].copy(),surface['Theta'].copy()
    report['initial']=dict(C_mean_Pa=float(np.dot(weights,initial_C)/sum(weights)),
                           Theta_min_s=float(min(initial['Theta'])),Theta_max_s=float(max(initial['Theta'])))
    report['completed_to_2_seconds']=report['steps'][-1]['time_s']==2.0
    report['allowances_pass']=bool(omitted<=1e-4 and all(r['normalization_max_error']<=1e-4
        and abs(r['global_normalization']-1)<=1e-4 for r in report['steps']))
    (dest/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['allowances_pass']:
        raise SystemExit('Stop: support/normalization allowance failed')


if __name__=='__main__':
    main()
