#!/usr/bin/env python3
"""Measure the actual native response; distinguish tip growth from nucleation.

Numerical identities/lifecycle are strict checks. A physically unqualified
outcome remains in the result instead of being tuned into a propagation claim.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from assess_p3_extension import csv,gauss,seed_table

G,ETA,ELL,GC,R=1e6,1e8,.15625,26.6666666667,1000.
KF=GC/((8/3)*ELL);M=KF/.5
SEED_R,SEED_P=seed_table()


def norm3(t):
    return t[...,0]**2+t[...,1]**2+(t[...,0]+t[...,1])**2+2*t[...,2]**2


def initial_error(nodes):
    px,py=np.unique(nodes['x']),np.unique(nodes['y'])
    a=np.sort(nodes,order=['y','x'])['phi'].reshape(len(py),len(px))
    z,w=gauss(24);u,v=np.meshgrid(z,z,indexing='ij');ww=w[:,None]*w
    numerator=denominator=0.
    for i,dx in enumerate(np.diff(px)):
        for j,dy in enumerate(np.diff(py)):
            p=(1-u)*(1-v)*a[j,i]+u*(1-v)*a[j,i+1]+(1-u)*v*a[j+1,i]+u*v*a[j+1,i+1]
            x=px[i]+dx*u;y=py[j]+dy*v
            exact=np.interp(np.hypot(np.maximum(x-.5,0),y-.5),SEED_R,SEED_P,right=0.)
            numerator+=dx*dy*np.sum(ww*(p-exact)**2);denominator+=dx*dy*np.sum(ww*exact**2)
    return float(np.sqrt(numerator/denominator))


def connected_front(nodes,level):
    line=np.sort(nodes[nodes['y']==.5],order='x')
    assert len(line)>1 and line['phi'][0]>=level
    # First exit from the component containing the seed. A disconnected
    # boundary island must not become the reported propagating fault tip.
    below=np.flatnonzero(line['phi']<level)
    if not len(below):return float(line['x'][-1])
    j=below[0];i=j-1
    return float(line['x'][i]+(line['x'][j]-line['x'][i])*(line['phi'][i]-level)/(line['phi'][i]-line['phi'][j]))


def measure(path):
    metrics=csv(path/'fixture.csv');ledger=csv(path/'bulk-accepted.csv')
    initial=csv(path/'phase-0000.csv');initial.sort(order='dof')
    px,py=np.unique(initial['x']),np.unique(initial['y'])
    assert np.array_equal(metrics['publications'],metrics['step'])
    assert np.array_equal(ledger['publications'],ledger['step'])
    assert np.array_equal(ledger['step'],np.arange(1,len(ledger)+1))
    gates=csv(path/'gates.csv');assert np.all(gates['passed']==1)
    assert np.max(ledger['mechanical_residual'])<=1e-9
    assert np.max(ledger['phase_KKT'])<=1e-7
    assert np.max(ledger['projection_error'])<=1e-12
    assert np.max(abs(ledger['ledger_error']))<=1e-10*max(1,np.max(ledger['bulk']+ledger['cohesive']+ledger['fracture']))
    assert np.min(ledger['Db'])>=0 and np.min(ledger['Dc'])>=0 and np.min(ledger['Df'])>=0
    previous=initial['phi'];peak_ahead=peak_tip=peak_intact=peak_remote=0.
    initial_hold=0.;peak_intact_tip=0.;history=[]
    tip=(initial['x']>=.45)&(initial['x']<=.85)&(abs(initial['y']-.5)<=.2)
    ahead=initial['x']>.5;remote=(initial['x']<.125)|(abs(initial['y']-.5)>.35)
    for row in metrics:
        step=int(row['step']);nodes=csv(path/f'phase-{step:04d}.csv');nodes.sort(order='dof')
        phase=nodes['phi'];assert len(phase)==len(initial)
        assert np.all(phase>=previous) and np.all(phase<1.)
        delta=phase-initial['phi'];previous=phase
        maxahead=float(np.max(delta[ahead]));maxtip=float(np.max(delta[tip]));maxintact=float(np.max(delta[initial['phi']==0]))
        maxremote=float(np.max(delta[remote]));peak_ahead=max(peak_ahead,maxahead);peak_tip=max(peak_tip,maxtip)
        peak_intact_tip=max(peak_intact_tip,float(np.max(delta[tip&(initial['phi']==0)])))
        peak_intact=max(peak_intact,maxintact);peak_remote=max(peak_remote,maxremote)
        if row['time']<=.05+1e-12:initial_hold=max(initial_hold,float(np.max(abs(delta))))
        history.append(dict(step=step,time=float(row['time']),ahead=maxahead,tip=maxtip,intact=maxintact,remote=maxremote,
                            fronts={str(level):connected_front(nodes,level) for level in [.001,.01,.1,.3,.5]}))
    assert initial_hold<=1e-12
    phase=previous;change=phase-initial['phi'];location=int(np.argmax(change))
    W=np.cumsum(ledger['Wext']);D=np.cumsum(ledger['Db']+ledger['Dc']+ledger['Df']);E=np.cumsum(ledger['energy_change'])
    remainder=W-D-E
    # Additional temporal-work diagnostic, not a replacement for the declared
    # endpoint-work gate. Under the prescribed affine boundary shear, W/dgamma
    # is its generalized reaction. Trapezoidal time integration distinguishes
    # work quadrature from the exact backward-Euler algebraic ledger.
    dgamma=np.where((ledger['time']>.05+1e-12)&(ledger['time']<=.15+1e-12),.018*ledger['dt'],0.)
    reaction=np.divide(ledger['Wext'],dgamma,out=np.zeros(len(dgamma)),where=dgamma>0.)
    trapezoidal_work=float(np.sum(.5*(reaction+np.r_[0.,reaction[:-1]])*dgamma))
    L=np.cumsum(ledger['stress_loss']+ledger['traction_loss']+ledger['phase_remainder']+ledger['fracture_loss'])
    residual=np.cumsum(ledger['phase_work']+ledger['mechanical_work'])
    assert np.max(abs(remainder-L+residual))<1e-9
    field_errors={k:0. for k in ['coverage','stress','cohesive_relation','driving','bulk_energy','cohesive_energy','Coulomb_bound','profile_variance_identity','profile_observer','cumulative_friction']}
    points_range=[]
    final_profile={}
    for file in sorted(path.glob('bulk-[0-9][0-9][0-9][0-9].csv')):
        f=csv(file);h=M*f['phi']*(1+f['phi'])/(1-f['phi'])**2;hp=M*(1+3*f['phi'])/(1-f['phi'])**3
        field_errors['coverage']=max(field_errors['coverage'],abs(np.sum(f['JxW'])-1))
        field_errors['stress']=max(field_errors['stress'],float(np.max(abs(f['tauxy']-(f['Txy']-G*f['a']))))/1000)
        field_errors['cohesive_relation']=max(field_errors['cohesive_relation'],float(np.max(abs(G*f['a']-h*f['q'])))/1000)
        field_errors['driving']=max(field_errors['driving'],float(np.max(abs(f['driving']-hp*f['q']**2/(2*G))))/KF)
        stress=np.stack([f['tauxx'],f['tauyy'],f['tauxy']],axis=-1)
        field_errors['bulk_energy']=max(field_errors['bulk_energy'],float(np.max(abs(f['bulk']-norm3(stress)/(4*G)))))
        field_errors['cohesive_energy']=max(field_errors['cohesive_energy'],float(np.max(abs(f['cohesive']-h*f['q']**2/(2*G)))))
        field_errors['Coulomb_bound']=max(field_errors['Coulomb_bound'],float(np.max(np.maximum(0,abs(f['tauxy']-f['q'])-R)))/1000)
        assert np.all(f['a'][h==0]==0.)
        # Actual tensor-rule columns, with normal weights recovered from JxW
        # and the surface-coordinate Gauss weights. No nodal projected I is
        # substituted into this raw-weight identity.
        nx=len(px)-1;nq=round(np.sqrt(len(f)/(nx*(len(py)-1))))
        z,wx=gauss(nq);cell_x=np.minimum((f['x']*nx).astype(int),nx-1)
        qx=np.argmin(abs((f['x']*nx-cell_x)[:,None]-z),axis=1)
        column=cell_x*nq+qx;outer=wx[qx]/nx;normal=f['JxW']/outer
        nc=nx*nq;I=np.bincount(column,weights=normal*h,minlength=nc)
        se=np.bincount(column,weights=normal*f['a'],minlength=nc)
        ec=np.bincount(column,weights=normal*f['cohesive'],minlength=nc)
        positive=I>0;surface=np.divide(G*se*se,2*I,out=np.zeros(nc),where=positive)
        rho=f['a']-np.divide(h*se[column],I[column],out=np.zeros(len(f)),where=I[column]>0)
        variance=np.bincount(column,weights=np.divide(normal*G*rho*rho,2*h,out=np.zeros(len(f)),where=h>0),minlength=nc)
        field_errors['profile_variance_identity']=max(field_errors['profile_variance_identity'],float(np.max(abs(ec-surface-variance))))
        observer=csv(path/file.name.replace('bulk-','observer-'))
        ocell=np.minimum((observer['x']*nx).astype(int),nx-1)
        oq=np.argmin(abs((observer['x']*nx-ocell)[:,None]-z),axis=1);oi=ocell*nq+oq
        for key,value in [('I',I),('se',se),('Z',G*se),('residual_energy',variance)]:
            field_errors['profile_observer']=max(field_errors['profile_observer'],float(np.max(abs(observer[key]-value[oi])))/max(1.,float(np.max(abs(value)))))
        step=int(file.stem.split('-')[1]);dissipated=float(np.sum(ledger['Df'][ledger['step']<=step]))
        field_errors['cumulative_friction']=max(field_errors['cumulative_friction'],abs(R*np.sum(f['JxW']*f['L'])-dissipated))
        final_profile=dict(raw_weight_surface_cohesive=float(np.sum(np.tile(wx/nx,nx)*surface)),
                           retained_residual_cohesive=float(np.sum(np.tile(wx/nx,nx)*variance)),
                           integrated_bulk_cohesive=float(np.sum(f['JxW']*f['cohesive'])),
                           newly_observed_positive_Z_records=int(np.sum((observer['original_surface']==0)&(observer['rebuilt_surface']==1)&(abs(observer['Z'])>0))))
        points_range.append([float(min(f['JxW'])),float(max(f['JxW']))])
    assert max(field_errors.values())<1e-10
    fronts0=history[0]['fronts'];fronts1=history[-1]['fronts']
    result=dict(cells=int(metrics['cells'][0]),nx=len(px)-1,ny=len(py)-1,
                DG_tensor_DoFs=int(metrics['DG_DoFs'][0]),local_history_points=int(metrics['points'][0]),
                accepted_steps=len(ledger),last_accepted_time=float(metrics['time'][-1]),
                initial_phase_relative_L2=initial_error(initial),unloaded_hold_phase_drift=initial_hold,
                maximum_phase_increase_ahead=peak_ahead,maximum_phase_increase_in_tip_window=peak_tip,
                maximum_phase_increase_in_initially_intact_tip_window=peak_intact_tip,
                maximum_new_phase_in_initially_intact_material=peak_intact,maximum_remote_phase_increase=peak_remote,
                maximum_change_position=[float(initial['x'][location]),float(initial['y'][location])],
                maximum_boundary_phase_increase=float(np.max(change[(initial['x']==0)|(initial['x']==1)|(initial['y']==0)|(initial['y']==1)])),
                seed_connected_contour_advance={k:fronts1[k]-v for k,v in fronts0.items()},
                mechanical_residual=float(max(ledger['mechanical_residual'])),phase_KKT=float(max(ledger['phase_KKT'])),
                projection_error=float(max(ledger['projection_error'])),discrete_ledger_error=float(max(abs(ledger['ledger_error']))),
                independent_field_checks=field_errors,quadrature_weight_range=[min(v[0] for v in points_range),max(v[1] for v in points_range)],
                external_work=float(W[-1]),stored_energy_change=float(E[-1]),
                bulk_viscous_dissipation=float(sum(ledger['Db'])),cohesive_viscous_dissipation=float(sum(ledger['Dc'])),frictional_dissipation=float(sum(ledger['Df'])),
                external_work_remainder=float(remainder[-1]),relative_external_work_remainder=float(abs(remainder[-1])/max(1,abs(W[-1]))),
                trapezoidal_work_diagnostic=trapezoidal_work,
                trapezoidal_work_remainder_diagnostic=trapezoidal_work-float(E[-1]+D[-1]),
                work_quadrature_difference=float(W[-1]-trapezoidal_work),
                total_discretization_remainder=float(L[-1]),phase_residual_work=float(sum(ledger['phase_work'])),mechanical_residual_work=float(sum(ledger['mechanical_work'])),
                mechanism_proxy_passed=peak_ahead>.001,tip_window_increment_proxy_passed=peak_tip>.001,
                centerline_tip_advance_demonstrated=any(fronts1[k]>v for k,v in fronts0.items()),
                physical_energy_screen_passed=bool(abs(remainder[-1])<=.01*max(1,abs(W[-1]))),
                histories_published_once=True,strict_reference_match_passed=False,
                final_native_profile_decomposition=final_profile,
                physical_interpretation='Measured bulk activation; report spatial location before claiming finite-tip propagation.',
                history=history)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('path',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=measure(a.path);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='history'},indent=2))
