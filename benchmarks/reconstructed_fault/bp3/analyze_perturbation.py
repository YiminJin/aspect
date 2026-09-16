"""Offline BP3 stress-change diagnostics from accepted output, never re-updating history.

Surface fields are consistent-mass projections of the actual pre-commit weak
loads. Bulk tau_* arrays are retained Maxwell-history inputs, not total stress.
The normal-column check is an independent FE-profile integral, not surface traction.
"""
import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import solve_banded

sys.path.insert(0, str(Path(__file__).resolve().parent/'airy_dt0'))
import analyze as saved


def profile_width():
    # Same specified stationary AT1 profile, independently integrated in theta.
    # m cancels from the coordinate map; the production table has 5000 points.
    theta=np.linspace(0,np.pi/2,5000)
    phi=.6*np.cos(theta)**2
    h=lambda p:p*(1+p)/(1-p)**2
    hh=h(.6)
    integrand=np.zeros_like(theta)
    integrand[1:-1]=800*.6*np.sin(theta[1:-1])*np.cos(theta[1:-1])*np.sqrt(
        hh/(hh*phi[1:-1]-.6*h(phi[1:-1])))
    hp=(1+3*.6)/(1-.6)**3
    integrand[0]=800*np.sqrt(.6*hh/(.6*hp-hh))
    integrand[-1]=800*np.sqrt(.6*hh/(hh-.6))
    return float(np.trapezoid(integrand,theta))


def columns(points, mesh, data, fault):
    """Cell-split Gauss integration of saved Q1 phi, with unchanged full I_h.

    Report in-box columns at the official stations and dense along-fault points.
    At tips these are one-sided physical columns; no absent half-space is invented.
    """
    corners=data['phase_field'].reshape(-1,9)[:,[0,2,6,8]]
    maps={}
    for i,row in enumerate(mesh):
        h=row['h']
        maps.setdefault(h,{})[(int(round(row['x']/h-.5)),int(round(row['y']/h-.5)))]=i
    def values(p):
        result=np.full(len(p),np.nan)
        for h,lookup in maps.items():
            for j in np.flatnonzero(np.isnan(result)):
                key=tuple(np.floor(p[j]/h).astype(int))
                i=lookup.get(key)
                if i is not None:
                    x,y=(p[j]-points[i,0])/h
                    result[j]=corners[i]@np.array([(1-x)*(1-y),x*(1-y),(1-x)*y,x*y])
        assert np.all(np.isfinite(result))
        return np.maximum(result,0)
    width=profile_width()
    normal=np.array([-np.sqrt(3)/2,.5])
    m=1e5/(8/3*400*(1e12/(2*saved.G)))
    stations=np.loadtxt(Path(__file__).with_name('stations.txt'))
    locations=np.unique(np.r_[stations,np.linspace(0,40000,81),fault['xd'].min(),fault['xd'].max()])
    locations=np.clip(locations,0,1e5/(np.sqrt(3)/2))
    results=[]
    q,w=leggauss(12)
    for s in locations:
        center=np.array([saved.TRACE-.5*s,1e5-np.sqrt(3)/2*s])
        bounds=[-1200.,1200.]
        for d in range(2):
            ends=np.sort((np.array([0.,1e5])-center[d])/normal[d])
            bounds=[max(bounds[0],ends[0]),min(bounds[1],ends[1])]
        # The entire nonzero strip has the same finest h; split on all x/y faces.
        cuts=[*bounds,-width,width,0.]
        h=min(maps)
        for d in range(2):
            lo,hi=np.sort(center[d]+np.array(bounds)*normal[d])
            faces=np.arange(np.floor(lo/h),np.ceil(hi/h)+1)*h
            cuts.extend((faces-center[d])/normal[d])
        cuts=np.unique(np.clip(cuts,*bounds))
        mid=(cuts[:-1]+cuts[1:])/2
        half=np.diff(cuts)/2
        z=(mid[:,None]+half[:,None]*q).ravel()
        phi=values(center+z[:,None]*normal)
        localization=m*phi*(1+phi)/(1-phi)**2
        weights=(half[:,None]*w).ravel()
        full=float(weights@localization)
        supported=float(weights@(localization*(np.abs(z)<=width)))
        ih=np.interp(s,fault['xd'][::-1],fault['Ih'][::-1])
        results.append([s,full,ih,1-supported/full,supported/ih-1,full/ih-1,width])
    return np.array(results)


def bulk_slip_integral(points,mesh,data,fault,previous,order):
    """Saved-field replay at the production QGauss(3) points (8 is diagnostic).

    For this straight, fixed profile the production strip association is
    exactly 0<=s<=L and |zeta|<=width. Retain full I_h and the signed history
    correction; no tail renormalization or update from committed history.
    """
    q,w=leggauss(order);q=(q+1)/2;w=w/2
    coords=points[:,0,None,None,:]+mesh['h'][:,None,None,None]*np.stack(np.meshgrid(q,q,indexing='xy'),axis=-1)[None,...]
    phi_nodes=data['phase_field'].reshape(-1,9)[:,[0,2,6,8]]
    x,y=np.meshgrid(q,q,indexing='xy')
    shape=np.stack([(1-x)*(1-y),x*(1-y),(1-x)*y,x*y],axis=-1)
    phi=np.einsum('ni,abi->nab',phi_nodes,shape)
    phi=np.maximum(phi,0)
    s=(saved.TRACE-coords[...,0])*.5+(1e5-coords[...,1])*np.sqrt(3)/2
    distance=np.abs((saved.TRACE-coords[...,0])*np.sqrt(3)/2-(1e5-coords[...,1])*.5)
    admitted=(s>=0)&(s<=fault['xd'].max())&(distance<=profile_width())
    interp=lambda name,f:np.interp(s,f['xd'][::-1],f[name][::-1])
    ih=interp('Ih',fault);old_ih=interp('Ih',previous)
    old_C=interp('C',previous);velocity=interp('V',fault)
    m=1e5/(8/3*400*(1e12/(2*saved.G)))
    h=m*phi*(1+phi)/(1-phi)**2
    dt=fault['dt'][0] if fault['dt'][0]>0 else 4e6
    beta=np.exp(-dt*saved.G/1e26);kappa=-1e26*np.expm1(-dt*saved.G/1e26)
    instantaneous=h/ih*velocity
    history=beta*old_C/kappa*(h*old_ih/ih-h)
    weight=mesh['h'][:,None,None]**2*np.outer(w,w)[None,...]*admitted
    target=np.trapezoid(fault['V'][::-1],fault['xd'][::-1])
    a=float(np.sum(weight*instantaneous));b=float(np.sum(weight*history))
    return {'order':order,'instantaneous_m2_s':a,'history_m2_s':b,
            'total_m2_s':a+b,'integrated_surface_V_m2_s':float(target),
            'relative_total_normalization_error':float((a+b)/target-1)}


def main(directory):
    directory=Path(directory)
    steps=sorted(int(p.stem.split('_')[-1]) for p in directory.glob('fault_*.csv')
                 if re.fullmatch(r'fault_\d+',p.stem))
    resolved=(directory/'parameters.prm').read_text()
    minimum_rate=float(re.search(r'set Minimum (?:fault )?slip rate\s*=\s*(\S+)',resolved,re.I)[1])
    report={'interpretation':'bulk tau_* = retained perturbation history; surface = accepted pre-commit response',
            'steps':{}}
    log=directory.with_suffix('.log').read_text()
    chunks=re.split(r'\*\*\* Timestep (\d+):',log)
    step_logs={int(chunks[i]):chunks[i+1] for i in range(1,len(chunks),2)}
    target=saved.read(directory/'initial_traction_target.csv')
    reference=None
    points,u,mesh,data=saved.load(directory)
    previous=None
    for step in steps:
        f=saved.read(directory/f'fault_{step}.csv')
        n=saved.read(directory/f'constitutive_normal_{step}_rank0.csv')
        mass=np.zeros((3,len(f)));mass[1]=f['mass_diagonal']
        mass[0,1:]=mass[2,:-1]=f['mass_upper'][:-1]
        sigma=solve_banded((1,1),mass,n['sigma_load'])
        pressure=solve_banded((1,1),mass,n['p_load'])
        tauN=solve_banded((1,1),mass,n['tauN_load'])
        assert np.allclose(sigma,f['sigma_n_bg']+pressure-tauN,rtol=2e-14,atol=1e-6)
        if reference is None: reference=f
        if previous is None: previous=f
        assert np.array_equal(f[['x','y']],reference[['x','y']])
        assert np.array_equal(f['tau_bg'],reference['tau_bg'])
        assert np.array_equal(f['sigma_n_bg'],reference['sigma_n_bg'])
        free=f['prescribed']==0
        def span(x): return {'min':float(np.min(x)),'max':float(np.max(x))}
        entry={'time_s':float(f['time'][0]),'dt_s':float(f['dt'][0]),
               'free_V_over_Vinit':span(f['V'][free]/1e-9),
               'deep_V_absolute_error_m_s':float(np.max(np.abs(f['V'][~free]-1e-9))),
               'sigma_total_actual_samples_Pa':{'min':float(n['sigma_min'].min()),'max':float(n['sigma_max'].max()),
                  'mean':float(n['sigma_load'].sum()/n['weight'].sum())},
               'delta_pressure_actual_samples_Pa':{'min':float(n['p_min'].min()),'max':float(n['p_max'].max())},
               'delta_tau_Q1_Pa':span(f['delta_tau']),
               'background_frozen':True,'geometry_unchanged':True,
               'unprescribed_nodes':int(np.sum(free)),
               'free_nodes':int(np.sum(free&(f['V']>minimum_rate))),
               'lower_bound_active_nodes':int(np.sum(f['V'][free]<=minimum_rate)),
               'deep_prescribed_nodes':int(np.sum(~free))}
        entry['bulk_Q3_supported_slip']=bulk_slip_integral(points,mesh,data,f,previous,3)
        entry['bulk_Q8_diagnostic']=bulk_slip_integral(points,mesh,data,f,previous,8)
        residuals=re.findall(r'after nonlinear iteration (\d+): ([^,\n]+), ([^\n]+)',step_logs[step])
        assert residuals and max(map(float,residuals[-1][1:]))<1e-8, 'Unconverged exported state'
        linear=re.findall(r'Fault linear solve: iterations=(\d+), estimated=([^,]+), fresh=([^,]+), target=([^,]+)',step_logs[step])
        # Fresh failures trigger residual replacement, not acceptance. Retain
        # those attempts and require the same tolerance/cumulative budget until
        # the successful return. Nonlinear convergence is checked separately.
        assert linear
        pending=None
        for row in linear:
            if pending is not None:
                assert int(row[0])>int(pending[0]) and float(row[3])==float(pending[3])
            pending=row if float(row[2])>float(row[3]) else None
        assert pending is None, 'Unconverged final linear direction'
        absolute=re.findall(r'Fault nonlinear residual: bulk=([^,]+), bulk scale=([^,]+), surface=([^,]+), surface scale=([^,]+), velocity=([^,]+), scaled continuity=([^,]+)',step_logs[step])
        entry['convergence']={'iteration':int(residuals[-1][0]),
                              'bulk_relative':float(residuals[-1][1]),'surface_relative':float(residuals[-1][2]),
                              'bulk_absolute':float(absolute[-1][0]),'surface_absolute_Pa':float(absolute[-1][2]),
                              'linear':[dict(iterations=int(n),estimated=float(e),fresh=float(f),target=float(t)) for n,e,f,t in linear]}
        history_path=directory/f'history_{step}.csv'
        if history_path.exists():
            history=saved.read(history_path)[0]
            entry['Theta_update_relative_error']=float(history['Theta_reference_relative_error'])
            entry['committed_particle_stress_max_Pa']=float(history['committed_particle_stress_max_Pa'])
        if step==0:
            entry['Theta_relative_error']=float(np.max(np.abs(f['Theta']/target['Theta'][::-1]-1)))
            entry['background_weak_relative_error']=float(target['weak_relative_error'].max())
            entry['background_friction_representation_correction_Pa']=span(target['Q1_friction_correction'])
        fields=np.column_stack([f[name] for name in ['xd','time','V','Theta','C','Ih','slip','tau_bg','delta_tau','tau_total','sigma_n_bg']]
                               +[pressure-tauN,sigma,pressure,tauN,previous['Theta']])
        header='xd,time,V,Theta,C,Ih,slip,tau_bg,delta_tau,tau_total,sigma_n_bg,delta_sigma_n,sigma_n_total,delta_p,delta_tau_N,Theta_used_by_mechanics'
        np.savetxt(directory/f'perturbations_{step}.csv',fields,delimiter=',',header=header,comments='')
        station=np.loadtxt(Path(__file__).with_name('stations.txt'))
        sampled=np.column_stack([station]+[np.interp(station,fields[::-1,0],fields[::-1,j]) for j in range(1,fields.shape[1])])
        np.savetxt(directory/f'stations_{step}.csv',sampled,delimiter=',',header=header,comments='')
        report['steps'][str(step)]=entry
        previous=f
    report['initial_bulk']=saved.bulk_statistics(points,u,mesh,4e6)[0]
    report['initial_bulk']['retained_stress_max_Pa']=float(max(np.max(np.abs(data[name])) for name in ['tau_xx','tau_yy','tau_xy']))
    report['initial_bulk']['pressure_Pa']={'min':float(data['p'].min()),'max':float(data['p'].max())}
    vel,correction,weight=saved.samples(points,u,mesh,8)
    q,_=leggauss(8);q=(q+1)/2
    coords=points[:,0,None,None,:]+mesh['h'][:,None,None,None]*np.stack(np.meshgrid(q,q,indexing='xy'),axis=-1)[None,...]
    distance=np.abs((saved.TRACE-coords[...,0])*np.sqrt(3)/2-(1e5-coords[...,1])*.5)
    outside=distance>1200
    report['initial_bulk']['outside_3ell_correction_RMS_over_rigid_speed']=saved.rms(correction,weight*outside)/5e-10
    report['initial_bulk']['outside_3ell_correction_max_over_rigid_speed']=float(np.linalg.norm(correction,axis=-1)[outside].max()/5e-10)
    column=columns(points,mesh,data,reference)
    np.savetxt(directory/'initial_columns.csv',column,delimiter=',',
               header='xd,J_full,Ih,omitted_h_fraction,supported_normalization_error,full_representation_error,half_width',comments='')
    report['initial_columns']={'sample_count':len(column),'max_omitted_fraction':float(column[:,3].max()),
                               'max_supported_normalization_error':float(np.max(np.abs(column[:,4]))),
                               'half_width_m':float(column[0,-1]),
                               'history_term':'zero at initialization because current/previous phi and Ih coincide'}
    report['resources']=json.loads(directory.with_suffix('.resources.json').read_text())
    report['last_attempted_step']=max(step_logs)
    report['last_accepted_step']=max(steps)
    report['complete_requested_trajectory']=report['resources']['status']==0 and max(steps)==max(step_logs)
    all_linear=re.findall(r'Fault linear solve: iterations=(\d+), estimated=([^,]+), fresh=([^,]+), target=([^,]+)',log)
    returned=[row for row in all_linear if float(row[2])<=float(row[3])]
    report['linear_check_attempts']={'count':len(all_linear),
        'rejected_fresh_attempts':[dict(iterations=int(n),estimated=float(e),fresh=float(f),target=float(t))
                                   for n,e,f,t in all_linear if float(f)>float(t)]}
    report['all_returned_linear_checks']={'count':len(returned),
        'total_iterations':sum(int(n) for n,_,_,_ in returned),
        'all_fresh_checks_pass':bool(returned) and bool(all_linear)
            and float(all_linear[-1][2])<=float(all_linear[-1][3])}
    if max(step_logs)>max(steps):
        pending=step_logs[max(step_logs)]
        residuals=re.findall(r'after nonlinear iteration (\d+): ([^,\n]+), ([^\n]+)',pending)
        report['unaccepted_attempt']={'step':max(step_logs),
            'last_iteration':int(residuals[-1][0]),'bulk_relative':float(residuals[-1][1]),
            'surface_relative':float(residuals[-1][2]),'not_an_accepted_state':True}
    (directory/'perturbation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='resources'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    main(parser.parse_args().directory)
