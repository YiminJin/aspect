#!/usr/bin/env python3
"""Accepted-state path formation in a nested, fixed-background native space."""
import argparse
import copy
import ctypes as ct
import hashlib
import gzip
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import numpy as np

from cohesive_interface_turn import Kernel as TurnKernel, derivative
from cohesive_interface_admission import Admission, bounded_multiplier
from check_cohesive_interface_kink import point, encode, G, ETA, GC, PEAK, C0, V0, DC
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[3]
WORK=ROOT/'build-particle-free/cohesive-interface-path'
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_path'
R=.75;RATE=.004;PREP=.15;END=.44;HOLD=.005;MAX_EXTENSION=3.
HISTORY=['jump','damage','theta','reference','relaxed_jump','absolute_slip']


def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def table(path,data):
    import csv
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def graph(arc,z):
    h=arc['end']-arc['x'];a,b=arc['a'],arc['b']
    if 'c1' in arc:
        c1,c2,c3=arc['c1'],arc['c2'],arc['c3']
        return arc['y']+h*(a*z+.5*c1*z*z+c2*z**3/3+c3*z**4/4),a+c1*z+c2*z*z+c3*z**3
    return arc['y']+a*h*z+(b-a)*h*(z**3-.5*z**4),a+(b-a)*(3*z*z-2*z**3)


def new_arc(arcs,target,ds,h):
    old=arcs[-1];y,slope=graph(old,1.);theta0=np.arctan(slope)
    # Signed shortest-plane difference, chosen by an explicit history preference.
    theta1=theta0+(target-theta0)*(-np.expm1(-ds/R))
    assert abs(theta1)<np.deg2rad(80),'Selected continuation is outside graph capability'
    a,b=slope,np.tan(theta1);z,w=np.polynomial.legendre.leggauss(20);z=(z+1)/2;w=w/2
    # Hermite slope matches the physical dtheta/ds=(theta_star-theta)/R
    # at both ends. Curvature may have bounded jumps when accepted driving
    # changes; the tangent is continuous. Never force curvature to zero.
    d0=(target-theta0)/(R*np.cos(theta0)**3);d1=(target-theta1)/(R*np.cos(theta1)**3)
    def coefficients(dx):return dx*d0,3*(b-a)-2*dx*d0-dx*d1,-2*(b-a)+dx*d0+dx*d1
    def length(dx):
        c1,c2,c3=coefficients(dx);p=a+c1*z+c2*z*z+c3*z**3
        return dx*(w@np.sqrt(1+p*p))
    dx=brentq(lambda dx:length(dx)-ds,ds*.05,ds,xtol=1e-14)
    c1,c2,c3=coefficients(dx)
    return dict(x=old['end'],y=float(y),end=float(old['end']+dx),a=float(a),b=float(b),cells=int(round(ds/h)),c1=float(c1),c2=float(c2),c3=float(c3))


def axes(tensor,previous=0.,sampling=0.):
    xx,yy,zz,xy,p=tensor
    assert np.hypot(.5*(xx-yy),xy)>1e-8,'Zero shear-driving tensor has no defined direction'
    compression=.5*np.arctan2(2*xy,xx-yy)+np.pi/2
    compression=(compression+np.pi/2)%np.pi-np.pi/2
    candidates=[compression+np.pi/4,compression-np.pi/4]
    candidates=[(v+np.pi/2)%np.pi-np.pi/2 for v in candidates]
    distance=[abs((v-previous+np.pi/2)%np.pi-np.pi/2) for v in candidates]
    assert abs(distance[0]-distance[1])>1e-8,'Symmetric directional preference is unresolved'
    target=candidates[int(np.argmin(distance))]
    if sampling:
        angles=np.deg2rad(np.arange(-90,90,sampling));shear=xy*np.cos(2*angles)+.5*(yy-xx)*np.sin(2*angles)
        # Full-circle angular cells around BOTH exact conjugates. Do not
        # manufacture uniqueness with a forward-sector search bound.
        chosen=[]
        for plane in candidates:
            d=abs((angles-plane+np.pi/2)%np.pi-np.pi/2);near=np.flatnonzero(d<=np.deg2rad(sampling)+1e-10)
            chosen.append(int(near[np.argmax(abs(shear[near]))]))
        target=angles[chosen[int(np.argmin(distance))]]
    target=previous+(target-previous+np.pi/2)%np.pi-np.pi/2
    return float(target),dict(compression_deg=float(np.rad2deg(compression)),
        max_shear_1_deg=float(np.rad2deg(candidates[0])),max_shear_2_deg=float(np.rad2deg(candidates[1])),
        selected_preference_deg=float(np.rad2deg(target)),old_tangent_deg=float(np.rad2deg(previous)),
        shear_amplitude_Pa=float(np.hypot(.5*(xx-yy),xy)),both_planes_work_equivalent=True,sampling_deg=sampling)


def kernel(arcs,n=64,degree=2,order=6,pressure_mode='point'):
    assert pressure_mode in ['point','work']
    version=5 if pressure_mode=='work' else 2
    signature=dict(arcs=arcs,n=n,degree=degree,order=order,native_interpolation_version=version)
    key=hashlib.sha256(encode(signature).encode()).hexdigest()[:16];folder=WORK/('kernel-'+key)
    if not (folder/'native.json').exists():
        folder.mkdir(exist_ok=False);save(folder/'input.json',signature)
        table(folder/'geometry.csv',arcs)
        executable=WORK/('build-work' if pressure_mode=='work' else 'build-c1')/'cohesive_interface_path'
        command=[str(executable),str(n),str(degree),str(folder/'geometry.csv'),str(folder),str(order)]
        if pressure_mode=='work':command.append('compress')
        with folder.with_suffix('.log').open('w') as f:
            subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300,
                           env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
    return PathKernel(folder,arcs,pressure_mode)


class PathKernel:
    predictor=TurnKernel.predictor
    admission_predictor=TurnKernel.admission_predictor
    close=TurnKernel.close

    def __init__(self,path,arcs,pressure_mode='point'):
        self.path=path;self.arcs=copy.deepcopy(arcs);self.meta=json.loads((path/'native.json').read_text())
        if (path/'input.json').exists():assert json.loads((path/'input.json').read_text())['arcs']==arcs,'Geometry/cache mismatch'
        self.pressure_mode=pressure_mode
        self.key=sha(path/'stiffness.csv')+('-normal-work-'+sha(path/'normal-work.csv') if pressure_mode=='work' else '');self.name=path.name
        for name,file in [('K','stiffness'),('Q','Q'),('Nm','normal-minus'),('Np','normal-plus'),('probe','probe'),('Jx','jump-x'),('Jy','jump-y')]:
            setattr(self,name,np.loadtxt(path/(file+'.csv'),delimiter=','))
        self.raw_Nm=self.Nm.copy();self.raw_Np=self.Np.copy()
        if pressure_mode=='work':self.Nm=self.Np=np.loadtxt(path/'normal-work.csv',delimiter=',')
        self.N=.5*(self.Nm+self.Np);self.s=np.genfromtxt(path/'surface.csv',delimiter=',',names=True)
        self.nodes=np.genfromtxt(path/'nodes.csv',delimiter=',',names=True);self.w=self.s['weight_m'];self.mass=self.Q.T@self.w
        self.centers=np.genfromtxt(path/'cells.csv',delimiter=',',names=True)
        shape=(len(self.centers),len(self.nodes)+1,5)
        def load_field(name):
            if (path/name).exists():return np.memmap(path/name,mode='r',dtype=np.float64,shape=shape)
            return np.frombuffer(gzip.decompress((path/(name+'.gz')).read_bytes()),dtype=np.float64).reshape(shape)
        self._load_field=load_field;self._field=None;self._mean=None
        assert self.meta['energy_relative']<1e-4 and self.meta['gradient_FD_relative']<1e-6
        if pressure_mode=='work':
            check=np.loadtxt(path/'normal-work-check.csv',delimiter=',')
            assert np.linalg.norm(check-self.N)/np.linalg.norm(self.N)<1e-4
        assert .001*self.meta['normal_operator_error']<1e-12
        self.lib=ct.CDLL(str(WORK/'build/libcohesive_interface_turn_solver.so'));D=ct.POINTER(ct.c_double)
        self.lib.cohesive_turn_create.argtypes=[ct.c_uint,ct.c_uint]+[D]*6;self.lib.cohesive_turn_create.restype=ct.c_void_p
        self.lib.cohesive_turn_solve.argtypes=[ct.c_void_p,D,D,ct.c_double,ct.c_double,D,ct.POINTER(ct.c_int),D,D,D,D,ct.POINTER(ct.c_int),D,ct.POINTER(ct.c_char),ct.c_uint]
        self.lib.cohesive_turn_solve.restype=ct.c_int;self.lib.cohesive_turn_destroy.argtypes=[ct.c_void_p]
        self.arrays=[np.ascontiguousarray(a,dtype=np.float64) for a in [self.K,self.Q,self.Nm,self.Np,self.w,self.mass]]
        self.context=self.lib.cohesive_turn_create(len(self.nodes),len(self.s),*[a.ctypes.data_as(D) for a in self.arrays])

    @property
    def field(self):
        if self._field is None:self._field=self._load_field('field-center.bin')
        return self._field

    @property
    def mean(self):
        if self._mean is None:self._mean=self._load_field('field-mean.bin')
        return self._mean

    def initial(self):
        x=self.s['x_m'];shape=4*((x-3)/3)*(1-(x-3)/3);a=.75+.05*shape
        jump=.0004*shape;reference=jump-C0*a/(1-a)*300.
        h=dict(jump=jump.tolist(),damage=a.tolist(),theta=(1.2+.1*shape).tolist(),reference=reference.tolist(),
               relaxed_jump=np.zeros(len(x)).tolist(),absolute_slip=abs(jump).tolist())
        return dict(clock_s=0.,gamma=.0003,wall_memory=.0003,memory=np.zeros(len(self.nodes)).tolist(),
            delta=np.zeros(len(self.nodes)).tolist(),nodal_rate=np.zeros(len(self.nodes)).tolist(),
            jump_offset=jump.tolist(),histories=h,last_resistance=np.full(len(x),300.).tolist(),
            geometry=copy.deepcopy(self.arcs),kernel=self.key,kernel_folder=self.name,
            physical_publications=0,generation=0,publications=0,virgin_theta=1.,pressure_mode=self.pressure_mode)

    def birth(self,old,s):
        before=encode(s);n=len(old.nodes);nq=len(old.s);nc=n+1
        assert s['kernel']==old.key and s['geometry']==old.arcs,'Stale geometry/history owner'
        assert self.arcs[:-1]==old.arcs and len(self.nodes)>n
        metrics={}
        metrics['old_K_bitwise']=bool(np.array_equal(self.K[:nc,:nc],old.K))
        metrics['old_Q_bitwise']=bool(np.array_equal(self.Q[:nq,:n],old.Q))
        old_columns=np.genfromtxt(old.path/'columns.csv',delimiter=',',names=True,dtype=None,encoding=None)
        new_columns=np.genfromtxt(self.path/'columns.csv',delimiter=',',names=True,dtype=None,encoding=None)
        metrics['old_native_response_hashes_equal']=bool(np.array_equal(old_columns,new_columns[:nc]))
        metrics['old_stress_pressure_columns_bitwise']=bool(np.array_equal(self.field[:,:nc],old.field))
        assert metrics['old_K_bitwise'] and metrics['old_Q_bitwise'] and metrics['old_native_response_hashes_equal'],metrics
        assert metrics['old_stress_pressure_columns_bitwise']
        c=np.r_[s['wall_memory'],s['memory']];cp=np.pad(c,(0,len(self.nodes)-n))
        physical=np.einsum('ijk,j->ik',self.field[:,:nc],c)-np.einsum('ijk,j->ik',old.field,c)
        metrics['physical_tensor_pressure_error_Pa']=float(np.max(abs(physical)))
        metrics['energy_change_J_m']=float(.5*cp@self.K@cp-.5*c@old.K@c)
        metrics['old_sided_pressure_error_Pa']=float(max(np.max(abs(self.Nm[:nq]@cp-old.Nm@c)),np.max(abs(self.Np[:nq]@cp-old.Np@c))))
        metrics['nonuniform_incoming_Maxwell_m']=float(np.max(abs(c[1:])))
        assert metrics['physical_tensor_pressure_error_Pa']<1e-7 and metrics['old_sided_pressure_error_Pa']<1e-7
        assert abs(metrics['energy_change_J_m'])<1e-10*max(1,abs(.5*c@old.K@c))
        t=copy.deepcopy(s);extra=len(self.s)-nq
        for key in HISTORY:t['histories'][key]+=([s['virgin_theta']]*extra if key=='theta' else [0.]*extra)
        for key in ['delta','memory','nodal_rate']:t[key]+=[0.]*(len(self.nodes)-n)
        t['jump_offset']+=[0.]*extra;t['last_resistance']+=[0.]*extra
        tied=np.array(t['histories']['damage'])==0;resistance=np.array(t['last_resistance']);resistance[tied]=0
        force=(self.K@cp)[1:]+self.Q.T@(self.w*resistance)
        lam,error,peak=bounded_multiplier(self,force,tied)
        metrics.update(weak_Pa=error,new_tied_peak_Pa=peak,old_clock_unchanged=True,old_histories_bitwise=True,
                       new_fracture_J_m=0.,new_bond_J_m=0.)
        assert error<=1e-8 and peak<=PEAK,(metrics,'Tied birth is not admissible')
        t.update(kernel=self.key,kernel_folder=self.name,geometry=copy.deepcopy(self.arcs),generation=s['generation']+1,publications=s['publications']+1)
        assert encode(s)==before
        return t,metrics

    def trial(self,s,target,load_end=END):
        assert s['kernel']==self.key and s['geometry']==self.arcs
        dt=target-s['clock_s'];assert dt>0;du=RATE*(min(target,load_end)-min(s['clock_s'],load_end))
        old=np.ascontiguousarray(np.array([s['histories'][key] for key in HISTORY]).T)
        c0=np.ascontiguousarray(np.r_[s['wall_memory'],s['memory']]);n=len(self.nodes);nq=len(self.s);D=ct.POINTER(ct.c_double)
        c=np.empty(n+1);inc=np.empty(n);r=np.empty((nq,14));J=np.empty((n,n));tied=np.empty(nq,dtype=np.int32);meta=np.empty(6);error=ct.create_string_buffer(12000)
        guess,hint=self.admission_predictor(s,dt,du)
        status=self.lib.cohesive_turn_solve(self.context,c0.ctypes.data_as(D),old.ctypes.data_as(D),dt,du,guess.ctypes.data_as(D),hint.ctypes.data_as(ct.POINTER(ct.c_int)),
                *[v.ctypes.data_as(D) for v in [c,inc,r,J]],tied.ctypes.data_as(ct.POINTER(ct.c_int)),meta.ctypes.data_as(D),error,len(error))
        if status:raise RuntimeError(error.value.decode())
        raw_contact=float(min(np.min(10000+self.raw_Nm@c),np.min(10000+self.raw_Np@c)))
        assert raw_contact>0,'Lost compression in raw physical stress diagnostic'
        t=copy.deepcopy(s);t.update(clock_s=target,gamma=s['gamma']+du,wall_memory=c[0],memory=c[1:].tolist(),
             delta=(np.array(s['delta'])+inc).tolist(),nodal_rate=(inc/dt).tolist(),virgin_theta=s['virgin_theta']+V0*dt/DC,
             histories={key:r[:,i].tolist() for i,key in enumerate(HISTORY)},last_resistance=r[:,10].tolist())
        assert np.max(abs(np.array(t['jump_offset'])+self.Q@t['delta']-r[:,0]))<1e-13
        assert np.max(abs(self.Q@c[1:]-r[:,4]))<1e-13
        E=.5*c@self.K@c;E0=.5*c0@self.K@c0;a0=old[:,1];e0=old[:,0]-old[:,3]
        B0=self.w@(.5*(1-a0)/(C0*np.where(a0==0,1,a0))*e0**2);B=self.w@r[:,13];F=GC*self.w@(r[:,1]-a0)
        friction=self.w@r[:,7];radiation=self.w@r[:,8];maxwell=2*G/ETA*dt*E
        be=.5*(c-c0)@self.K@(c-c0);surface=self.w@(r[:,6]*(self.Q@inc))-(B-B0)-F
        work=(self.K@c)[0]*du;trap=.5*((self.K@c)[0]+(self.K@c0)[0])*du
        ledger=work-(E-E0+B-B0+F+friction+radiation+maxwell+be+surface)
        assert abs(ledger)<=1e-10*max(1,abs(work)),ledger
        gap=-self.s['ty']*(self.Jx@np.r_[t['gamma'],t['delta']])+self.s['tx']*(self.Jy@np.r_[t['gamma'],t['delta']])
        assert np.max(abs(gap))<1e-12
        d=dict(time_s=target,dt_s=dt,gamma=t['gamma'],geometry_generation=s['generation'],bulk_J_m=float(E),bond_J_m=float(B),
            fracture_increment_J_m=float(F),total_fracture_J_m=float(GC*self.w@r[:,1]),friction_J_m=float(friction),radiation_J_m=float(radiation),
            sliding_loss_J_m=float(friction+radiation),maxwell_loss_J_m=float(maxwell),endpoint_work_J_m=float(work),trapezoid_work_J_m=float(trap),
            physical_remainder_J_m=float(trap-(E-E0+B-B0+F+friction+radiation+maxwell)),ledger_error_J_m=float(ledger),
            weak_error_Pa=float(meta[0]),minimum_symmetric_tangent_Pa_m=float(meta[1]),minimum_normal_Pa=float(meta[2]),
            iterations=int(meta[3]),raw_minimum_normal_Pa=raw_contact,normal_gap_m=float(max(abs(gap))),integrated_slip_m2=float(self.w@r[:,0]),
            peak_velocity_m_s=float(max(abs(r[:,9]))),max_damage_increment=float(max(r[:,1]-a0)),**functionals(self,t))
        return dict(base_publications=s['physical_publications'],base_generation=s['generation'],state=t,diagnostics=d,
                    tied=tied.tolist(),native=dict(c=c,increment=inc,r=r,J=J))


def functionals(k,s):
    a=np.array(s['histories']['damage']);w=k.w.reshape(-1,3);am=(w*a.reshape(-1,3)).sum(axis=1)/w.sum(axis=1)
    x=k.s['x_m'].reshape(-1,3);y=k.s['y_m'].reshape(-1,3);tx=k.s['tx'].reshape(-1,3);ty=k.s['ty'].reshape(-1,3)
    result={};start=np.searchsorted(x[:,0],6)
    for threshold,label in [(.05,'process'),(.5,'half'),(.999,'full')]:
        count=0
        for value in am:
            if value<threshold:break
            count+=1
        i=max(0,count-1);result.update({label+'_front_x_m':float(x[i,-1]),label+'_front_y_m':float(y[i,-1]),
            label+'_front_arc_m':float(w.sum(axis=1)[:count].sum()),label+'_front_tangent_deg':float(np.rad2deg(np.arctan2(ty[i,-1],tx[i,-1]))),
            label+'_detached_cells':int(np.count_nonzero(am[count:]>=threshold))})
    new=k.s['x_m']>6
    result.update(new_damage_integral_m=float(k.w[new]@a[new]),new_fully_decohered_cells=int(np.count_nonzero(np.min(a.reshape(-1,3)[start:],axis=1)==1)),
        geometry_length_m=float(sum(k.w)),new_geometry_length_m=float(sum(k.w[new])))
    result['new_full_front_available']=False
    if not np.any(new):
        result.update(full_front_x_m=6.,full_front_y_m=0.,full_front_arc_m=3.,full_front_tangent_deg=0.)
    # A diagnostic reconstruction is allowed only after checking its equality
    # with EVERY actual new Gauss history. No histories are changed/interpolated.
    nodal=np.r_[0.,s['delta'],0.];xn=np.r_[3.,k.nodes['x_m'],k.arcs[-1]['end']]
    represented=np.minimum(abs(k.Q@np.array(s['delta']))/(2*GC/PEAK),1.)
    discrepancy=float(np.max(abs(represented[new]-a[new]),initial=0.))
    result['new_damage_trace_identity']=discrepancy
    if discrepancy<1e-11 and np.any(new):
        first=int(np.argmin(abs(xn-6)));assert abs(xn[first]-6)<1e-10
        z,wq=np.polynomial.legendre.leggauss(12);z=(z+1)/2;wq=wq/2
        for threshold,label in [(.05,'process'),(.5,'half'),(1.,'full')]:
            limit=threshold*2*GC/PEAK;front=6.;j=first
            while j<len(xn)-1 and abs(nodal[j])>=limit:
                if abs(nodal[j+1])<limit:
                    front=float(xn[j]+(xn[j+1]-xn[j])*(abs(nodal[j])-limit)/(abs(nodal[j])-abs(nodal[j+1])));break
                front=float(xn[j+1]);j+=1
            arc_length=3.
            for arc in k.arcs[1:]:
                right=min(front,arc['end'])
                if right<=arc['x']:break
                xx=arc['x']+(right-arc['x'])*z;zz=(xx-arc['x'])/(arc['end']-arc['x'])
                _,sl=graph(arc,zz);arc_length+=(right-arc['x'])*(wq@np.sqrt(1+sl*sl))
            arc=next((arc for arc in k.arcs if front<=arc['end']+1e-11),k.arcs[-1]);zz=(front-arc['x'])/(arc['end']-arc['x']);yy,sl=graph(arc,zz)
            result.update({label+'_front_x_m':front,label+'_front_y_m':float(yy),label+'_front_arc_m':float(arc_length),label+'_front_tangent_deg':float(np.rad2deg(np.arctan(sl)))})
            if label=='full':result['new_full_front_available']=front>6+1e-10
            mask=(np.min(a.reshape(-1,3),axis=1)==1) if label=='full' else (am>=threshold)
            result[label+'_detached_cells']=int(np.count_nonzero(mask[start:]&(x[start:,0]>front+1e-10)))
    return result


def publish_geometry(s,t,metrics,base):
    assert (s['kernel'],s['generation'],s['physical_publications'])==base,'Stale geometry publication'
    assert t['clock_s']==s['clock_s'] and t['physical_publications']==s['physical_publications']
    assert t['generation']==s['generation']+1 and t['publications']==s['publications']+1
    assert t['geometry'][:-1]==s['geometry']
    for key in HISTORY:assert t['histories'][key][:len(s['histories'][key])]==s['histories'][key]
    assert metrics['weak_Pa']<=1e-8 and metrics['new_tied_peak_Pa']<=PEAK
    return copy.deepcopy(t)


def run(label,n=64,degree=2,extension=.75,sampling=0.,dt=.00025,end=END,hold=HOLD,surface_h=.09375,checkpoint=None,pressure_mode='point'):
    folder=WORK/label;folder.mkdir(exist_ok=False)
    save(folder/'run-input.json',dict(label=label,n=n,degree=degree,extension_m=extension,sampling_deg=sampling,
        dt_s=dt,end_s=end,hold_s=hold,surface_h_m=surface_h,physical_R_m=R,
        executable_sha256=sha(WORK/('build-work' if pressure_mode=='work' else 'build-c1')/'cohesive_interface_path'),python_sha256=sha(Path(__file__)),pressure_mode=pressure_mode,
        point_source_sha256=sha(ROOT/'source/reconstructed_fault/cohesive_interface.cc'),
        incoming_checkpoint=None if checkpoint is None else str(checkpoint)))
    incoming=None if checkpoint is None else json.loads(Path(checkpoint).read_text())
    arcs=[dict(x=3.,y=0.,end=6.,a=0.,b=0.,cells=round(3/surface_h),c1=0.,c2=0.,c3=0.)] if incoming is None else incoming['geometry']
    k=kernel(arcs,n,degree,pressure_mode=pressure_mode);s=k.initial() if incoming is None else incoming
    assert s['kernel']==k.key,'Checkpoint/operator mismatch'
    save(folder/'initial.json',s);rows=[];events=[];checks=[];snapshots=[];tic=time.perf_counter();last=s
    total_extension=sum(a['cells']*surface_h for a in arcs[1:]);checked=False
    grid=sorted(set(np.round(np.arange(dt,end+hold+dt/2,dt),12).tolist()+[PREP,end,end+hold]))
    try:
        for i,target in enumerate(grid):
            if target<=s['clock_s']+1e-14:continue
            while s['clock_s']>=PREP-1e-12 and total_extension<MAX_EXTENSION-1e-10:
                f=functionals(k,s);reserve=f['geometry_length_m']-f['process_front_arc_m']
                if reserve>R+1e-8:break
                c=np.r_[s['wall_memory'],s['memory']];tensor=k.probe@c;old_theta=np.arctan(arcs[-1]['b'])
                preference,decision=axes(tensor,old_theta,sampling)
                arc=new_arc(arcs,preference,min(extension,MAX_EXTENSION-total_extension),surface_h)
                before=encode(s);save(folder/'last-accepted.json',s)
                candidate=kernel(arcs+[arc],n,degree,pressure_mode=pressure_mode)
                try:t,metrics=candidate.birth(k,s)
                except Exception:
                    candidate.close();save(folder/'birth-failed-incoming.json',s);raise
                retry,repeated=candidate.birth(k,json.loads(before));assert encode(t)==encode(retry) and metrics==repeated
                assert encode(s)==before
                events.append(dict(time_s=s['clock_s'],stage=t['generation'],arc=arc,decision=decision,
                    physical_tensor_Pa=tensor.tolist(),transfer=metrics,endpoint_x_m=arc['end'],endpoint_y_m=float(graph(arc,1)[0]),
                    terminal_angle_deg=float(np.rad2deg(np.arctan(arc['b']))),reserve_before_m=reserve))
                save(folder/f'before-extension-{t["generation"]}.json',s);save(folder/f'after-extension-{t["generation"]}.json',t)
                t=publish_geometry(s,t,metrics,(s['kernel'],s['generation'],s['physical_publications']))
                k.close();k=candidate;s=t;last=s;arcs=copy.deepcopy(s['geometry']);total_extension+=extension
                snapshots.append(copy.deepcopy(s));save(folder/'events.json',events)
                print(label,'extension',s['generation'],s['clock_s'],decision['selected_preference_deg'],events[-1]['terminal_angle_deg'],flush=True)
            if s['clock_s']>=PREP and total_extension>=MAX_EXTENSION-1e-10:
                f=functionals(k,s)
                assert f['geometry_length_m']-f['process_front_arc_m']>=R,'Reached bounded geometric reserve capacity'
            before=encode(s);p=k.trial(s,target,end);assert encode(s)==before
            if not checked and s['generation']>0 and p['diagnostics']['new_damage_integral_m']>1e-5:
                retry=k.trial(json.loads(before),target,end);assert encode(p['state'])==encode(retry['state'])
                bad=copy.deepcopy(p);bad['state']['histories']['damage'][0]=-1
                try:Admission(k).publish(s,bad)
                except AssertionError:pass
                else:raise AssertionError('Bad history accepted')
                assert encode(s)==before
                checks.append(dict(time_s=target,FD_relative=derivative(k,s,p),retry_restart_bitwise=True,rejection_unchanged=True));checked=True
            incoming=s;s=Admission(k).publish(s,p);assert s['physical_publications']==incoming['physical_publications']+1
            if i==0:
                try:Admission(k).publish(s,p)
                except AssertionError:pass
                else:raise AssertionError('Duplicate physical publication')
            rows.append(p['diagnostics']);last=s
            if i%40==0 or target in [PREP,end,end+hold]:snapshots.append(copy.deepcopy(s))
            if i%200==0:
                save(folder/'last-accepted.json',s);print(label,target,round(time.perf_counter()-tic,2),flush=True)
                assert time.perf_counter()-tic<900 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024<3072
    except Exception as e:
        save(folder/'last-accepted.json',last);save(folder/'failure.json',dict(target_s=target,error=str(e),events=events))
        if rows:table(OUT/(label+'-partial.csv'),rows)
        k.close();raise
    table(OUT/(label+'-history.csv'),rows);save(folder/'final.json',s);save(folder/'snapshots.json',snapshots)
    save(folder/'events.json',events);absolute=sum(abs(r['physical_remainder_J_m']) for r in rows);sliding=sum(r['sliding_loss_J_m'] for r in rows);fracture=sum(r['fracture_increment_J_m'] for r in rows)
    result=dict(label=label,n=n,degree=degree,extension_m=extension,sampling_deg=sampling,dt_s=dt,surface_h_m=surface_h,
        seconds=time.perf_counter()-tic,steps=len(rows),geometry_publications=s['publications'],physical_publications=s['physical_publications'],
        peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,events=events,checks=checks,final=rows[-1],
        energy=dict(absolute_J_m=absolute,sliding_J_m=sliding,fracture_J_m=fracture,relative_sliding=absolute/sliding,relative_fracture=absolute/fracture if fracture else None),
        kernels=sorted({ss['kernel_folder'] for ss in snapshots}))
    save(OUT/(label+'-summary.json'),result);k.close();print(json.dumps(result),flush=True);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('label');parser.add_argument('--n',type=int,default=64);parser.add_argument('--degree',type=int,default=2)
    parser.add_argument('--extension',type=float,default=.75);parser.add_argument('--sampling',type=float,default=0.);parser.add_argument('--dt',type=float,default=.00025)
    parser.add_argument('--end',type=float,default=END);parser.add_argument('--surface-h',type=float,default=.09375);parser.add_argument('--checkpoint',type=Path);parser.add_argument('--pressure-mode',choices=['point','work'],default='point');args=parser.parse_args()
    run(args.label,args.n,args.degree,args.extension,args.sampling,args.dt,args.end,surface_h=args.surface_h,checkpoint=args.checkpoint,pressure_mode=args.pressure_mode)


if __name__=='__main__':main()
