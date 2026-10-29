#!/usr/bin/env python3
"""Independent angle/work checks and actual evolved-state geometry restart."""
import copy
import ctypes as ct
import json
from pathlib import Path
import numpy as np

from cohesive_interface_path import ROOT, WORK, OUT, R, axes, new_arc, graph, kernel, PathKernel, publish_geometry, save, table, encode, GC, C0, PEAK
from cohesive_interface_admission import Admission


def reject(action):
    try:action()
    except (AssertionError,RuntimeError):return True
    raise AssertionError('Invalid operation accepted')


def angles_and_work():
    axial=.5*np.tan(np.deg2rad(40));gamma=.001
    tensor=np.array([-2*axial*1000,2*axial*1000,0.,1000.,0.])
    target,info=axes(tensor);assert abs(np.rad2deg(target)-20)<1e-12
    assert abs(info['compression_deg']+25)<1e-12
    assert sorted([round(info['max_shear_1_deg'],10),round(info['max_shear_2_deg'],10)])==[-70.,20.]
    angles=np.deg2rad(np.linspace(-90,90,36001));t=np.c_[np.cos(angles),np.sin(angles)];n=np.c_[-t[:,1],t[:,0]]
    tau=np.array([[tensor[0],tensor[3]],[tensor[3],tensor[1]]]);pn=10000-np.einsum('ij,jk,ik->i',n,tau,n);shear=np.einsum('ij,jk,ik->i',t,tau,n)
    mu=.1;score=abs(shear)-mu*pn
    coulomb=[-25+45-.5*np.rad2deg(np.arctan(mu)),-25-45+.5*np.rad2deg(np.arctan(mu))]
    for angle in coulomb:
        i=np.argmin(abs(np.rad2deg(angles)-angle));assert abs(score[i]-max(score))<1e-5
    data=[]
    for degrees in np.arange(-90,90.0001,1.):
        a=np.deg2rad(degrees);tt=np.array([np.cos(a),np.sin(a)]);nn=np.array([-tt[1],tt[0]]);s=tt@tau@nn;p=10000-nn@tau@nn
        data.append(dict(tangent_deg=degrees,shear_Pa=float(s),normal_Pa=float(p),intact_work_margin_Pa=float(abs(s)-PEAK),diagnostic_Coulomb_margin_Pa=float(abs(s)-mu*p),reference_mu=mu))
    table(OUT/'homogeneous-angles.csv',data)
    errors=[]
    for phi in [-37.,-13.,0.,29.,61.]:
        a=np.deg2rad(phi);Q=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]);rot=Q@tau@Q.T
        value,_=axes([rot[0,0],rot[1,1],0,rot[0,1],0],np.deg2rad(phi))
        errors.append(abs(float((value-(target+a)+np.pi/2)%np.pi-np.pi/2)))
    reversed_tensor=tensor.copy();reversed_tensor[3]*=-1
    reversed_angle,_=axes(reversed_tensor);assert abs(np.rad2deg(reversed_angle)+20)<1e-12
    symmetric=reject(lambda:axes(tensor,np.deg2rad(-25)))
    zero=reject(lambda:axes(np.zeros(5)))
    # The cohesive first variation, without pressure-dependent virgin strength.
    e=np.geomspace(1e-12,1e-5,30);df=2*GC/PEAK;a=e/df
    energy=(1-a)*e*e/(2*C0*a)+GC*a
    exact=PEAK*e-PEAK*e*e/(2*df)
    assert max(abs(energy-exact))<1e-17
    # Fixed-stress orientation update: endpoints are partition invariant;
    # continuous geometry is approached as the interpolation is refined.
    semigroup=[]
    for ds in [.75,.375,.1875]:
        arcs=[dict(x=3.,y=0.,end=6.,a=0.,b=0.,cells=32)]
        for _ in range(round(2.25/ds)):arcs.append(new_arc(arcs,target,ds,.09375))
        angle=np.rad2deg(np.arctan(arcs[-1]['b']));expected=20*(1-np.exp(-2.25/R))
        assert abs(angle-expected)<1e-12
        semigroup.append(dict(extension_m=ds,terminal_deg=float(angle),exact_terminal_deg=float(expected),end_x_m=arcs[-1]['end'],end_y_m=float(graph(arcs[-1],1)[0])))
    angular=[]
    for resolution in [15.,7.,3.5,1.,.25]:
        angle,record=axes(tensor,sampling=resolution);angular.append(dict(step_deg=resolution,selected_deg=float(np.rad2deg(angle)),error_deg=float(np.rad2deg(angle)-20),conjugates_retained=True))
    save(OUT/'analytic-checks.json',dict(homogeneous=info,forward_deg=float(np.rad2deg(target)),
        diagnostic_Coulomb_deg=coulomb,diagnostic_mu=.1,reversal_deg=float(np.rad2deg(reversed_angle)),
        rotated_basis_error_rad=max(errors),symmetric_preference_unresolved=symmetric,zero_direction_unresolved=zero,
        cohesive_energy_max_error_J_m2=float(max(abs(energy-exact))),semigroup=semigroup,angular_sampling=angular,
        physical_R_m=R,scope='Formation-work/persistence hypothesis, not Coulomb intact admission or a global incremental potential.'))
    table(OUT/'homogeneous-step-sensitivity.csv',semigroup);table(OUT/'angular-sampling.csv',angular)


def interpolation_ode():
    # Independently integrate the computed graph's arclength, then compare
    # angle AND curvature with the closed-form physical ODE solution. Endpoint
    # angles alone cannot expose the retained v1 curvature oscillation.
    target=np.deg2rad(20);z,w=np.polynomial.legendre.leggauss(48);z=(z+1)/2;w=w/2
    data=[]
    for ds in [.75,.375,.1875]:
        arcs=[dict(x=3.,y=0.,end=6.,a=0.,b=0.,cells=32)];distance=0.;numerator=0.;denominator=0.;error=0.
        for _ in range(round(2.25/ds)):
            arc=new_arc(arcs,target,ds,.09375);arcs.append(arc);dx=arc['end']-arc['x']
            p=arc['a']+arc['c1']*z+arc['c2']*z*z+arc['c3']*z**3
            dp=(arc['c1']+2*arc['c2']*z+3*arc['c3']*z*z)/dx
            arclength=[]
            for upper in z:
                u=upper*z;pu=arc['a']+arc['c1']*u+arc['c2']*u*u+arc['c3']*u**3
                arclength.append(distance+dx*upper*(w@np.sqrt(1+pu*pu)))
            arclength=np.array(arclength);exact_angle=target*(1-np.exp(-arclength/R));exact_curvature=target/R*np.exp(-arclength/R)
            curvature=dp/(1+p*p)**1.5;measure=dx*w*np.sqrt(1+p*p)
            error=max(error,float(max(abs(np.arctan(p)-exact_angle))))
            numerator+=measure@((curvature-exact_curvature)**2);denominator+=measure@(exact_curvature**2)
            distance+=float(sum(measure))
        data.append(dict(extension_m=ds,theta_error_deg=float(np.rad2deg(error)),curvature_L2_relative=float(np.sqrt(numerator/denominator))))
    assert data[0]['curvature_L2_relative']<.01
    assert all(b['curvature_L2_relative']<a['curvature_L2_relative']/3 for a,b in zip(data,data[1:]))
    table(OUT/'independent-interpolation-ODE.csv',data)
    save(OUT/'independent-interpolation-ODE.json',dict(physical_equation='dtheta/ds=(theta_star-theta)/R',
        exact='theta=theta_star*(1-exp(-s/R)); curvature=theta_star/R*exp(-s/R)',
        relative_norm='sqrt(integral((curvature-exact)^2 ds)/integral(exact^2 ds))',
        quadrature='48 Gauss points per arc; independent 48-point partial arclength integrals',
        angle_error='Maximum over Gauss points; endpoint semigroup checked separately',
        checks=data,earlier_sampled_estimates_retained='interpolation-consistency.csv'))


def lifecycle():
    folder=WORK/'work96';s=json.loads((folder/'before-extension-3.json').read_text());after=json.loads((folder/'after-extension-3.json').read_text())
    old=PathKernel(WORK/s['kernel_folder'],s['geometry'],'work');new=PathKernel(WORK/after['kernel_folder'],after['geometry'],'work')
    before=encode(s);t,metrics=new.birth(old,s);assert encode(t)==encode(after)
    assert metrics['old_stress_pressure_columns_bitwise']
    base=(s['kernel'],s['generation'],s['physical_publications'])
    wrong=copy.deepcopy(t);wrong['histories']['theta'][0]+=1
    rejected=reject(lambda:publish_geometry(s,wrong,metrics,base));assert encode(s)==before
    accepted=publish_geometry(s,t,metrics,base)
    duplicate=reject(lambda:publish_geometry(accepted,t,metrics,base))
    stale_birth=reject(lambda:new.birth(old,accepted))
    target=s['clock_s']+.00025;p=new.trial(accepted,target)
    # A genuinely failed native trial, not just a host-side invalid publication.
    # Deliberately oversized boundary proposal loses compression. It is never
    # published or interpreted as part of the physical loading trajectory.
    before_failed=encode(accepted);n=len(new.nodes);nq=len(new.s);D=ct.POINTER(ct.c_double)
    c0=np.ascontiguousarray(np.r_[accepted['wall_memory'],accepted['memory']])
    h=np.ascontiguousarray(np.array([accepted['histories'][key] for key in ['jump','damage','theta','reference','relaxed_jump','absolute_slip']]).T)
    c=np.empty(n+1);inc=np.empty(n);r=np.empty((nq,14));J=np.empty((n,n));ties=np.empty(nq,dtype=np.int32);meta=np.empty(6);error=ct.create_string_buffer(12000)
    status=new.lib.cohesive_turn_solve(new.context,c0.ctypes.data_as(D),h.ctypes.data_as(D),.00025,.05,None,None,
        *[v.ctypes.data_as(D) for v in [c,inc,r,J]],ties.ctypes.data_as(ct.POINTER(ct.c_int)),meta.ctypes.data_as(D),error,len(error))
    assert status!=0 and 'contact' in error.value.decode().lower()
    assert encode(accepted)==before_failed
    bad=copy.deepcopy(p);bad['state']['histories']['damage'][0]=-1
    physical_rejected=reject(lambda:Admission(new).publish(accepted,bad))
    checkpoint=dict(schema='experimental-path-1',physical_R_m=R,controller=dict(extension_m=.75,load_end_s=.44),state=accepted)
    save(WORK/'geometry-checkpoint.json',checkpoint)
    old.close();new.close()
    restored=json.loads((WORK/'geometry-checkpoint.json').read_text());assert restored['physical_R_m']==R
    new=PathKernel(WORK/restored['state']['kernel_folder'],restored['state']['geometry'],'work');q=new.trial(restored['state'],target)
    assert encode(p['state'])==encode(q['state'])
    updated=Admission(new).publish(restored['state'],q)
    assert updated['physical_publications']==accepted['physical_publications']+1
    physical_duplicate=reject(lambda:Admission(new).publish(updated,q))
    save(OUT/'lifecycle.json',dict(geometry_rejection_unchanged=rejected,geometry_duplicate_rejected=duplicate,
        stale_birth_rejected=stale_birth,physical_rejection_unchanged=physical_rejected,
        native_context_JSON_restart_bitwise=True,physical_duplicate_rejected=physical_duplicate,
        geometry_increment=accepted['publications']-s['publications'],physical_increment=updated['physical_publications']-accepted['physical_publications'],
        actual_native_contact_rejection_unchanged=True,native_rejection_message=error.value.decode(),
        correct_retry_after_failed_trial_bitwise=True,pressure_mode='variational normal contact reaction',
        transfer=metrics,scope='Isolated benchmark owners and native C++ solves; production Simulator/Manager/checkpoints remain unchanged.'))
    new.close()


if __name__=='__main__':angles_and_work();interpolation_ode();lifecycle()
