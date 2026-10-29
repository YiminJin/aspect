#!/usr/bin/env python3
"""Bounded curved native mixed mechanics and signed cohesive–RSF evolution."""
import argparse
import copy
import csv
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import numpy as np
from scipy.linalg import null_space

from check_cohesive_interface_kink import point, encode, G, ETA, GC, PEAK, DF, C0, V0, DC, P0
from cohesive_interface_admission import Admission, bounded_multiplier

ROOT=Path(__file__).resolve().parents[3]
WORK=ROOT/'build-particle-free/cohesive-interface-turn'
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_turn'
AXIAL=np.sqrt(3)/2
RATE=.004
GAMMA_END=.00145
HOLD=.025
HISTORY=['jump','damage','theta','reference','relaxed_jump','absolute_slip']


def save(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def table(path,data):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def tag(angle,n=32,width=.25,mirror=False,order=6,axial=AXIAL):
    a=('m' if angle<0 else 'p')+format(abs(angle),'g').replace('.','d')
    return f'kernel-{a}-n{n}-w{round(width*1000)}'+('-mirror' if mirror else '')+('' if order==6 else f'-q{order}')+('' if axial==AXIAL else f'-a{round(axial*1000)}')


def kernel(angle,n=32,width=.25,mirror=False,order=6,linear_jump=False,scale=1.,axial=AXIAL,gamma_end=GAMMA_END):
    folder=WORK/tag(angle,n,width,mirror,order,axial)
    if not (folder/'native.json').exists():
        folder.mkdir(exist_ok=True)
        command=[str(WORK/'build/cohesive_interface_kink'),str(n),str(n//2),str(folder),str(order),
                 'smooth',str(angle),str(width),format(axial,'.17g'),'-1' if mirror else '1']
        with folder.with_suffix('.log').open('w') as f:
            subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300,
                           env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
    return Kernel(folder,linear_jump,scale,gamma_end)


class Kernel:
    def __init__(self,path,linear_jump=False,scale=1.,gamma_end=GAMMA_END):
        self.path=path;self.name=path.name
        self.meta=json.loads((path/'native.json').read_text())
        for key,file in [('K','stiffness'),('Q','Q'),('Nm','normal-minus'),('Np','normal-plus'),('E','energy-gauss6')]:
            setattr(self,key,np.loadtxt(path/(file+'.csv'),delimiter=','))
        self.N=.5*(self.Nm+self.Np)
        self.s=np.genfromtxt(path/'surface.csv',delimiter=',',names=True)
        self.nodes=np.genfromtxt(path/'nodes.csv',delimiter=',',names=True)
        self.scale=scale;self.gamma_end=gamma_end;self.axial=self.meta['axial_ratio']
        # Exact isotropic similarity of the homogeneous native mixed operator.
        # Physical strain is unchanged; jump is in metres and Gc is NOT scaled.
        DK=np.diag(np.r_[scale,np.ones(len(self.nodes))]);DFIELD=DK/scale
        self.K=DK@self.K@DK;self.E=DK@self.E@DK
        self.Nm=self.Nm@DFIELD;self.Np=self.Np@DFIELD;self.N=.5*(self.Nm+self.Np)
        for key in ['x_m','y_m']:self.nodes[key]*=scale;self.s[key]*=scale
        self.s['weight_m']*=scale
        self.full_nodes=self.nodes.copy();self.full_Q=self.Q.copy();self.linear_jump=linear_jump
        self.prolongation=np.eye(len(self.nodes)+1)
        if linear_jump:
            vertices=np.flatnonzero(np.abs(2*self.meta['nx']*self.nodes['x_m']/scale-2*np.round(self.meta['nx']*self.nodes['x_m']/scale))<1e-10)
            P=np.zeros((len(self.nodes),len(vertices)));lookup={round(float(self.nodes['x_m'][j]),12):i for i,j in enumerate(vertices)}
            for j,x in enumerate(self.nodes['x_m']):
                if round(float(x),12) in lookup:P[j,lookup[round(float(x),12)]]=1.
                else:
                    for value in [x-.5*scale/self.meta['nx'],x+.5*scale/self.meta['nx']]:
                        if round(float(value),12) in lookup:P[j,lookup[round(float(value),12)]]=.5
            S=np.zeros((len(self.nodes)+1,len(vertices)+1));S[0,0]=1;S[1:,1:]=P
            self.prolongation=S;self.K=S.T@self.K@S;self.E=S.T@self.E@S
            self.Q=self.Q@P;self.Nm=self.Nm@S;self.Np=self.Np@S;self.nodes=self.nodes[vertices]
            self.N=.5*(self.Nm+self.Np);self.name+='-q1'
        self.w=self.s['weight_m'];self.mass=self.Q.T@self.w
        self.old=self.nodes['x_m']<.5*scale;self.oq=self.s['x_m']<.5*scale;self.nq=~self.oq
        self.sign=self.meta['shear_sign'];self.key=sha(path/'stiffness.csv')+('-q1' if linear_jump else '')+f'-scale{scale}'
        assert min(self.mass)>0
        assert self.meta['gradient_FD_relative']<1e-6
        # Operator columns mix unit strain and unit-metre jump. Test a declared
        # 0.1 mm virtual jump here; actual physical gaps are checked each step.
        assert 1e-4*self.meta['normal_jump_operator_error']<1e-12
        assert 1e-4*self.meta['tangent_jump_operator_error']<1e-12
        assert .001*self.meta['affine_stress_error_Pa']<1e-7
        assert np.linalg.norm(self.E-self.K)/np.linalg.norm(self.K)<1e-4
        self.Jx=np.loadtxt(path/'jump-x.csv',delimiter=',');self.Jy=np.loadtxt(path/'jump-y.csv',delimiter=',')
        self.full_Jx=self.Jx.copy();self.full_Jy=self.Jy.copy()
        self.Jx=self.Jx@DK@self.prolongation;self.Jy=self.Jy@DK@self.prolongation
        self.field_transform=DFIELD@self.prolongation
        self.lib=ct.CDLL(str(WORK/'build/libcohesive_interface_turn_solver.so'))
        D=ct.POINTER(ct.c_double)
        self.lib.cohesive_turn_create.argtypes=[ct.c_uint,ct.c_uint]+[D]*6
        self.lib.cohesive_turn_create.restype=ct.c_void_p
        self.lib.cohesive_turn_solve.argtypes=[ct.c_void_p,D,D,ct.c_double,ct.c_double,D,ct.POINTER(ct.c_int),D,D,D,D,ct.POINTER(ct.c_int),D,ct.POINTER(ct.c_char),ct.c_uint]
        self.lib.cohesive_turn_solve.restype=ct.c_int
        self.lib.cohesive_turn_destroy.argtypes=[ct.c_void_p]
        self.arrays=[np.ascontiguousarray(a,dtype=np.float64) for a in [self.K,self.Q,self.Nm,self.Np,self.w,self.mass]]
        self.context=self.lib.cohesive_turn_create(len(self.nodes),len(self.s),*[a.ctypes.data_as(D) for a in self.arrays])

    def close(self):
        if self.context:self.lib.cohesive_turn_destroy(self.context);self.context=None

    def initial(self,mature=False):
        x=self.s['x_m']/self.scale;xn=self.full_nodes['x_m']/self.scale;gamma0=0. if mature else .0003
        if mature:
            z=(xn-.25)/.5;delta=self.sign*.0004*4*z*(1-z)
            a=np.ones(len(x));theta=1.2+.1*4*((x-.25)/.5)*(1-(x-.25)/.5)
            reference=np.zeros(len(x))
        else:
            z=(xn-.25)/.25;delta=np.where((xn>.25)&(xn<.5),self.sign*.0004*4*z*(1-z),0.)
            z=(x-.25)/.25;shape=4*z*(1-z)
            a=np.where(self.oq,.75+.05*shape,0.);theta=np.where(self.oq,1.2+.1*shape,1.)
            e=C0*a/np.where(a==0,1,1-a)*(self.sign*G*gamma0)
            reference=np.where(self.oq,self.full_Q@delta-e,0.)
        jump=self.full_Q@delta
        offset=jump.copy() if self.linear_jump else np.zeros(len(jump))
        permanent=delta.copy() if self.linear_jump else np.zeros(len(delta))
        if self.linear_jump:delta=np.zeros(len(self.nodes))
        h=dict(jump=jump.tolist(),damage=a.tolist(),theta=theta.tolist(),reference=reference.tolist(),
               relaxed_jump=np.zeros(len(x)).tolist(),absolute_slip=abs(jump).tolist())
        s=dict(clock_s=0.,gamma=gamma0,wall_memory=gamma0,delta=delta.tolist(),memory=np.zeros(len(self.nodes)).tolist(),
               histories=h,physical_publications=0,generation=0,publications=0,installed=[0],
               geometry=dict(old_segment=[[.25*self.scale,0.],[.5*self.scale,0.]]),virgin_theta=1.,kernel='old-straight',
               mature_control=mature,load_end_s=(self.gamma_end-gamma0)/RATE,jump_offset=offset.tolist(),
               permanent_jump_coefficients=permanent.tolist())
        return s

    def birth(self,s):
        before=encode(s);assert s['generation']==0
        t=copy.deepcopy(s);c=np.r_[s['wall_memory'],s['memory']]
        old={k:np.array(v) for k,v in s['histories'].items()}
        a=old['damage'];e=old['jump']-old['reference']
        q=np.where(a==0,0.,(1-a)*e/(C0*np.where(a==0,1,a)))
        tied=a==0;force=(self.K@c)[1:]+self.Q[~tied].T@(self.w[~tied]*q[~tied])
        lam,error,peak=bounded_multiplier(self,force,tied)
        assert error<=1e-8 and peak<=PEAK,(error,peak)
        assert min((P0+self.Nm@c).min(),(P0+self.Np@c).min())>0
        t.update(generation=1,publications=1,installed=[0,1],kernel=self.key,
                 geometry=dict(old_segment=[[.25*self.scale,0.],[.5*self.scale,0.]],append=dict(angle_deg=self.meta['angle_deg'],
                     width_m=self.scale*self.meta['turn_width_m'],end_x_m=.75*self.scale),small_strain_closed_contact=True))
        # Reconstructed physical initial stress is uniform on every actual cell.
        cells=np.genfromtxt(self.path/'field-cells.csv',delimiter=',',names=True)
        data=np.fromfile(self.path/'field-cell-response.bin',dtype=np.float64).reshape(len(cells),len(self.full_nodes)+1,5)
        actual=np.einsum('ijk,j->ik',data,self.field_transform@c)
        expected=G*s['wall_memory']*np.array([-2*self.axial,2*self.axial,0.,self.sign,0.])
        physical_error=float(np.max(abs(actual-expected)))
        assert physical_error<1e-7
        expected_bulk=G*(.5+2*self.axial**2)*self.scale**2*s['wall_memory']**2
        assert abs(.5*c@self.K@c-expected_bulk)<1e-11
        assert encode(s)==before and s['histories']==t['histories']
        return t,dict(weak_Pa=error,peak_tied_Pa=peak,stress_error_Pa=physical_error,
            bulk_transfer_error_J_m=float(.5*c@self.K@c-expected_bulk),old_history_bitwise=True,
            physical_clock_change_s=0.,geometry_publications=1,new_fracture_J_m=0.,new_bond_J_m=0.)

    def predictor(self,s,dt,du,release=()):
        """Fixed accepted active-set Newton; the native signed solver certifies
        or releases it. This is a search strategy, not permanent plane/tie locking.
        """
        old={key:np.array(s['histories'][key]) for key in HISTORY}
        ties=old['damage']==0;ties[list(release)]=False;n=len(self.nodes)
        C=null_space(self.Q[ties]) if np.any(ties) else np.eye(n)
        inc=np.array(s.get('nodal_rate',np.zeros(n)))*dt
        inc=C@(C.T@inc);beta=1/(1+G*dt/ETA);wall=beta*(s['wall_memory']+du)
        def residual(v):
            c=np.r_[wall,beta*(np.array(s['memory'])+v)];dq=self.Q@v;dq[ties]=0.
            p=point(old,dq,P0+self.N@c,dt);R=p['resistance'].copy();R[ties]=0.
            J=beta*self.K[1:,1:]+self.Q.T@(self.w[:,None]*(p['derivative'][:,None]*self.Q+p['pressure_derivative'][:,None]*beta*self.N[:,1:]))
            return C.T@((self.K@c)[1:]+self.Q.T@(self.w*R)),C.T@J@C
        tolerance=1e-9*min(self.mass)
        for _ in range(30):
            f,J=residual(inc)
            if np.max(abs(f),initial=0)<=tolerance:break
            step=C@np.linalg.solve(J,-f);scale=1.
            for _ in range(25):
                ff,_=residual(inc+scale*step)
                if np.linalg.norm(ff)<np.linalg.norm(f) or np.max(abs(ff),initial=0)<=tolerance:
                    inc+=scale*step;break
                scale*=.5
            else:break
        return np.ascontiguousarray(inc),np.ascontiguousarray(ties,dtype=np.int32)

    def admission_predictor(self,s,dt,du):
        """Bounded local active-set proposals; every accepted proposal still
        satisfies the native full signed certificate and tangent check.
        """
        inc,ties=self.predictor(s,dt,du)
        old={key:np.array(s['histories'][key]) for key in HISTORY};beta=1/(1+G*dt/ETA)
        def check(v,tied):
            tied=tied.astype(bool);dq=self.Q@v;dq[tied]=0.
            c=np.r_[beta*(s['wall_memory']+du),beta*(np.array(s['memory'])+v)]
            r=point(old,dq,P0+self.N@c,dt)
            F=(self.K@c)[1:]+self.Q[~tied].T@(self.w[~tied]*r['resistance'][~tied])
            lam,error,peak=bounded_multiplier(self,F,tied)
            return lam,error,peak
        lam,error,peak=check(inc,ties)
        if error<=1e-8 and peak<=PEAK:return inc,ties
        ids=np.flatnonzero(ties);critical=ids[np.argsort(abs(lam))[-3:]]
        cells=list(dict.fromkeys([int(ids[0]//3)]+[int(q//3) for q in critical]))[:2]
        candidates=[]
        for cell in cells:
            for mask in range(1,8):
                release=[3*cell+j for j in range(3) if mask&(1<<j) and ties[3*cell+j]]
                if not release:continue
                v,t=self.predictor(s,dt,du,release)
                try:_,err,p=check(v,t)
                except RuntimeError:continue
                if err<=1e-8 and p<=PEAK+1e-9 and np.min(abs((self.Q@v)[release]),initial=np.inf)>0:
                    candidates.append((float(np.sum(self.mass*(v-np.array(s.get('nodal_rate',np.zeros(len(v))))*dt)**2)),v,t))
        if candidates:
            _,inc,ties=min(candidates,key=lambda c:c[0])
        return inc,ties

    def trial(self,s,target):
        assert s['kernel']==self.key and s['generation']==1,'Geometry/history mismatch'
        geometry=s['geometry']['append']
        assert geometry['angle_deg']==self.meta['angle_deg'] and geometry['width_m']==self.scale*self.meta['turn_width_m']
        dt=target-s['clock_s'];assert dt>0
        du=RATE*(min(target,s['load_end_s'])-min(s['clock_s'],s['load_end_s']))
        old=np.ascontiguousarray(np.array([s['histories'][key] for key in HISTORY]).T)
        c0=np.ascontiguousarray(np.r_[s['wall_memory'],s['memory']]);n=len(self.nodes);nq=len(self.s)
        c=np.empty(n+1);inc=np.empty(n);r=np.empty((nq,14));J=np.empty((n,n));tied=np.empty(nq,dtype=np.int32);meta=np.empty(6);error=ct.create_string_buffer(12000);D=ct.POINTER(ct.c_double)
        guess,hint_ties=self.admission_predictor(s,dt,du)
        status=self.lib.cohesive_turn_solve(self.context,c0.ctypes.data_as(D),old.ctypes.data_as(D),dt,du,
            guess.ctypes.data_as(D),hint_ties.ctypes.data_as(ct.POINTER(ct.c_int)),
            *[v.ctypes.data_as(D) for v in [c,inc,r,J]],tied.ctypes.data_as(ct.POINTER(ct.c_int)),meta.ctypes.data_as(D),error,len(error))
        if status:raise RuntimeError(error.value.decode())
        t=copy.deepcopy(s);t.update(clock_s=target,gamma=s['gamma']+du,wall_memory=c[0],memory=c[1:].tolist(),nodal_rate=(inc/dt).tolist(),
                                   delta=(np.array(s['delta'])+inc).tolist(),virgin_theta=s['virgin_theta']+V0*dt/DC)
        t['histories']={key:r[:,i].tolist() for i,key in enumerate(HISTORY)}
        assert max(abs(np.array(s.get('jump_offset',np.zeros(nq)))+self.Q@np.array(t['delta'])-r[:,0]))<1e-13
        assert max(abs(self.Q@c[1:]-r[:,4]))<1e-13
        E=.5*c@self.K@c;E0=.5*c0@self.K@c0
        a0=old[:,1];e0=old[:,0]-old[:,3];B0=self.w@(.5*(1-a0)/(C0*np.where(a0==0,1,a0))*e0**2)
        B=self.w@r[:,13];F=GC*(self.w@(r[:,1]-a0));friction=self.w@r[:,7];radiation=self.w@r[:,8]
        maxwell=2*G/ETA*dt*E;be=.5*(c-c0)@self.K@(c-c0);surface=self.w@(r[:,6]*(self.Q@inc))-(B-B0)-F
        reaction=(self.K@c)[0];work=reaction*du;trap=.5*(reaction+(self.K@c0)[0])*du
        ledger=work-(E-E0+B-B0+F+friction+radiation+maxwell+be+surface)
        assert abs(ledger)<=1e-10*max(1,abs(work)),ledger
        pn=P0+self.N@c;dc=np.r_[du,inc];nx=-self.s['ty'];ny=self.s['tx']
        gap=nx*(self.Jx@dc)+ny*(self.Jy@dc)
        permanent=np.r_[0.,s.get('permanent_jump_coefficients',np.zeros(len(self.full_nodes)))]
        total=np.r_[t['gamma'],t['delta']]
        total_gap=nx*(self.Jx@total+self.full_Jx@permanent)+ny*(self.Jy@total+self.full_Jy@permanent)
        assert max(abs(gap))<1e-12 and max(abs(total_gap))<1e-12
        normal_work=self.w@(pn*gap)
        old_damage_inventory=GC*(self.w@a0);initial=self.initial(s['mature_control'])
        append=self.nq;score=float((self.w[append]@r[append,1])/sum(self.w[append]))
        cell_damage=(self.w*r[:,1]).reshape(-1,3).sum(axis=1)/self.w.reshape(-1,3).sum(axis=1)
        turn=(self.s['x_m']>.52*self.scale)&(self.s['x_m']<.73*self.scale)
        dx=float(self.w[append]@(r[append,1]*self.s['tx'][append]));dy=float(self.w[append]@(r[append,1]*self.s['ty'][append]))
        z=np.clip((self.s['x_m']/self.scale-.5)/self.meta['turn_width_m'],0.,1.)
        slope=np.tan(np.deg2rad(self.meta['angle_deg']))
        curvature=slope*(6*z-6*z*z)/(self.scale*self.meta['turn_width_m'])/(1+(slope*(3*z*z-2*z*z*z))**2)**1.5
        diagnostic=dict(time_s=target,dt_s=dt,gamma=t['gamma'],bulk_J_m=float(E),bond_J_m=float(B),
            fracture_increment_J_m=float(F),old_fracture_inventory_J_m=float(old_damage_inventory),
            total_fracture_J_m=float(GC*self.w@r[:,1]),friction_J_m=float(friction),radiation_J_m=float(radiation),
            maxwell_loss_J_m=float(maxwell),sliding_loss_J_m=float(friction+radiation),
            endpoint_work_J_m=float(work),trapezoid_work_J_m=float(trap),ledger_error_J_m=float(ledger),
            physical_remainder_J_m=float(trap-(E-E0+B-B0+F+friction+radiation+maxwell)),
            weak_error_Pa=float(meta[0]),minimum_symmetric_tangent_Pa_m=float(meta[1]),
            minimum_normal_Pa=float(meta[2]),iterations=int(meta[3]),free_dimensions=int(meta[5]),
            append_mean_damage=score,append_damage_integral_m=float(self.w[append]@r[append,1]),
            append_cells_a005=int(np.sum(cell_damage[len(r)//6:]>=.05)),
            append_damage_direction_deg=float(np.degrees(np.arctan2(dy,dx))) if dx else 0.,
            integrated_abs_jump_m2=float(self.w@abs(r[:,0])),
            integrated_increment_from_initial_m2=float(self.w@abs(r[:,0]-np.array(initial['histories']['jump']))),
            maximum_curvature_times_slip=float(max(abs(curvature*r[:,0]))),
            minimum_abs_turn_velocity_m_s=float(min(abs(r[turn,9]))),
            mean_turn_velocity_m_s=float(self.w[turn]@r[turn,9]/sum(self.w[turn])),
            new_fracture_to_date_J_m=float(GC*self.w@(r[:,1]-np.array(initial['histories']['damage']))),
            normal_work_J_m=float(normal_work),normal_increment_max_m=float(max(abs(gap))),
            normal_total_max_m=float(max(abs(total_gap))),
            peak_velocity_m_s=float(max(abs(r[:,9]))),max_damage_increment=float(max(r[:,1]-a0)))
        payload=dict(base_publications=s['physical_publications'],base_generation=s['generation'],state=t,
                     diagnostics=diagnostic,tied=tied.tolist(),native=dict(c=c,increment=inc,r=r,J=J))
        return payload


def derivative(k,s,p):
    C=null_space(k.Q[np.array(p['tied'],bool)]) if any(p['tied']) else np.eye(len(k.nodes))
    inc=p['native']['increment'];direction=C@(C.T@(np.random.default_rng(19).normal(size=len(inc))*inc))
    direction/=np.linalg.norm(direction);eps=np.linalg.norm(inc)*1e-4
    old={key:np.array(s['histories'][key]) for key in HISTORY};dt=p['diagnostics']['dt_s'];beta=1/(1+G*dt/ETA)
    def force(v):
        c=np.r_[p['native']['c'][0],beta*(np.array(s['memory'])+v)]
        r=point(old,k.Q@v,P0+k.N@c,dt)
        return (k.K@c)[1:]+k.Q.T@(k.w*r['resistance'])
    fd=C.T@(force(inc+eps*direction)-force(inc-eps*direction))/(2*eps)
    exact=C.T@p['native']['J']@direction
    error=float(np.linalg.norm(fd-exact)/max(np.linalg.norm(exact),1.))
    assert error<1e-6,error
    return error


def run(k,label,mature=False,dt=.0005,resume=False,hold_s=HOLD):
    folder=WORK/'runs'/label;folder.mkdir(parents=True,exist_ok=resume)
    s0=k.initial(mature);s,birth=k.birth(s0);start=time.perf_counter()
    if resume:s=json.loads((folder/'last-accepted.json').read_text())
    else:save(folder/'initial.json',s)
    end=s['load_end_s']+hold_s
    grid=sorted(set([min(end,round((i+1)*dt,12)) for i in range(int(np.ceil(end/dt)))]+[s['load_end_s'],.8*s['load_end_s'],end]))
    rows=[]
    if resume:
        with (OUT/(label+'-partial.csv')).open() as f:rows=[{key:float(v) for key,v in r.items()} for r in csv.DictReader(f)]
        assert abs(rows[-1]['time_s']-s['clock_s'])<1e-12
        save(folder/'retry-from.json',dict(clock_s=s['clock_s'],physical_publications=s['physical_publications'],
                                        accepted_sha256=hashlib.sha256(encode(s).encode()).hexdigest(),histories_changed=False))
    snapshots={};checked=False;checks=[];last=s
    publisher=Admission(k)
    try:
        for i,target in enumerate(grid):
            if target<=s['clock_s']+1e-14:continue
            before=encode(s);tic=time.perf_counter();p=k.trial(s,target)
            assert encode(s)==before
            growing=np.any(np.array(p['state']['histories']['damage'])[k.nq]>0)
            if (growing or mature and target>.5*s['load_end_s']) and not checked:
                retry=k.trial(s,target);assert encode(retry['state'])==encode(p['state'])
                save(folder/'restart-check.json',s);restart=json.loads((folder/'restart-check.json').read_text())
                restored=k.trial(restart,target);assert encode(restored['state'])==encode(p['state'])
                bad=copy.deepcopy(p);bad['state']['histories']['damage'][0]=-1
                try:publisher.publish(s,bad)
                except AssertionError:pass
                else:raise AssertionError('Bad history accepted')
                assert encode(s)==before
                checks.append(dict(time_s=target,derivative_relative=derivative(k,s,p),retry_restart_bitwise=True,
                                   rejection_accepted_unchanged=True));checked=True
            old=s;s=publisher.publish(s,p);assert s['physical_publications']==old['physical_publications']+1
            if i==0:
                try:publisher.publish(s,p)
                except AssertionError:pass
                else:raise AssertionError('Duplicate publication')
            d=p['diagnostics'];d['step_seconds']=time.perf_counter()-tic;rows.append(d)
            if i%20==0 or any(abs(target-t)<1e-12 for t in [.8*s['load_end_s'],s['load_end_s'],end]):
                snapshots[str(target)]=copy.deepcopy(s)
            if i%100==0:
                save(folder/'last-accepted.json',s)
                print(label,i,target,round(time.perf_counter()-start,2),flush=True)
                assert time.perf_counter()-start<900 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024<3072
            last=s
    except Exception as e:
        save(folder/'last-accepted.json',last);save(folder/'failure.json',dict(target_s=target,error=str(e)))
        if rows:table(OUT/(label+'-partial.csv'),rows)
        raise
    table(OUT/(label+'-history.csv'),rows);save(folder/'snapshots.json',snapshots);save(folder/'final.json',s)
    load=min(rows,key=lambda r:abs(r['time_s']-s['load_end_s']))
    preview=min(rows,key=lambda r:abs(r['time_s']-.8*s['load_end_s']))
    sliding=sum(r['sliding_loss_J_m'] for r in rows);fracture=sum(r['fracture_increment_J_m'] for r in rows)
    absolute=sum(abs(r['physical_remainder_J_m']) for r in rows)
    result=dict(label=label,kernel=k.name,angle_deg=k.meta['angle_deg'],width_m=k.meta['turn_width_m'],
        n=k.meta['nx'],scale_m=k.scale,physical_width_m=k.scale*k.meta['turn_width_m'],axial_ratio=k.axial,
        linear_jump=k.linear_jump,dt_s=dt,mature=mature,seconds=time.perf_counter()-start,steps=len(rows),birth=birth,
        score=load['append_mean_damage'],score_80percent=preview['append_mean_damage'],resolved_cells=load['append_cells_a005'],
        checks=checks,energy=dict(absolute_remainder_J_m=absolute,sliding_loss_J_m=sliding,new_fracture_J_m=fracture,
            relative_sliding=absolute/sliding if sliding else None,relative_fracture=absolute/fracture if fracture>1e-14 else None,
            max_ledger_J_m=max(abs(r['ledger_error_J_m']) for r in rows)),
        minimum_normal_Pa=min(r['minimum_normal_Pa'] for r in rows),max_weak_Pa=max(r['weak_error_Pa'] for r in rows),
        maximum_normal_increment_m=max(r['normal_increment_max_m'] for r in rows),final=rows[-1])
    save(OUT/(label+'-summary.json'),result);print(json.dumps({key:result[key] for key in ['label','seconds','score','resolved_cells','energy']}),flush=True)
    return result


def main_kernel(angle,n=64,width=.125,mirror=False,gamma_end=.0022):
    return kernel(angle,n,width,mirror,linear_jump=True,scale=12.,
                  axial=.5*np.tan(np.deg2rad(40)),gamma_end=gamma_end)


def main_turn():
    for label,n,width,dt in [('mature64',64,.125,.00025),('mature128',128,.125,.00025),
                              ('maturewide128',128,.25,.00025),('maturehalf128',128,.125,.000125)]:
        k=main_kernel(40,n,width);r=run(k,label,True,dt);k.close()
        assert r['energy']['relative_sliding']<.01
        assert r['maximum_normal_increment_m']<1e-12 and r['minimum_normal_Pa']>0


def main_scan():
    assert (OUT/'maturehalf128-summary.json').exists()
    for angle in [-60,-40,-20,0,20,40,60]:
        label='select-'+('m' if angle<0 else 'p')+str(abs(angle))
        k=main_kernel(angle);run(k,label,False,.00025);k.close()


def score_at(label,time_s=.45):
    with (OUT/(label+'-history.csv')).open() as f:rows=list(csv.DictReader(f))
    r=min(rows,key=lambda r:abs(float(r['time_s'])-time_s));assert abs(float(r['time_s'])-time_s)<1e-10
    return float(r['append_mean_damage'])


def followup():
    # Select the short common window before the coarse +20 trial's 0.45475 s
    # candidate-end proximity event. Its future loading is not relabelled.
    base=[(a,'select-'+('m' if a<0 else 'p')+str(abs(a))) for a in [-60,-40,-20,0,20,40,60]]
    winner=max(base,key=lambda z:score_at(z[1]))[0]
    for a in [winner-10,winner+10]:
        if abs(a)>60:continue
        label='angular-'+('m' if a<0 else 'p')+str(abs(a))
        k=main_kernel(a,gamma_end=.0021);run(k,label,dt=.00025,hold_s=0.);k.close();base.append((a,label))
    leading=sorted(base,key=lambda z:score_at(z[1]),reverse=True)[:2]
    save(OUT/'followup-selection.json',dict(horizon_s=.45,initial_candidates=base,leading=leading,
         candidate_end_first_event_s=.45475,material_parameters_changed=False))
    for a,_ in leading:
        for prefix,dt in [('fine',.00025),('half',.000125)]:
            label=prefix+'-'+('m' if a<0 else 'p')+str(abs(a))
            k=main_kernel(a,128,gamma_end=.0021);run(k,label,dt=dt,hold_s=0.);k.close()
    for a in [-60,-40,-20,0,20,40,60]:
        label='wide-'+('m' if a<0 else 'p')+str(abs(a))
        k=main_kernel(a,width=.25,gamma_end=.0021);run(k,label,dt=.00025,hold_s=0.);k.close()
    wide=max([-60,-40,-20,0,20,40,60],key=lambda a:score_at('wide-'+('m' if a<0 else 'p')+str(abs(a))))
    k=main_kernel(wide,128,.25,gamma_end=.0021);run(k,'wide-fine-'+('m' if wide<0 else 'p')+str(abs(wide)),dt=.00025,hold_s=0.);k.close()
    a=leading[0][0]
    k=main_kernel(-a,128,mirror=True,gamma_end=.0021);run(k,'reflection',dt=.00025,hold_s=0.);k.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['turn','scan','followup','main-turn','main-scan']);args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True);(WORK/'runs').mkdir(exist_ok=True)
    if args.stage=='main-turn':main_turn()
    elif args.stage=='main-scan':main_scan()
    elif args.stage=='turn':
        k=kernel(40);run(k,'turn-n32',True);k.close()
    elif args.stage=='scan':
        for angle in [-60,-40,-20,0,20,40,60]:
            label='preview-'+('m' if angle<0 else 'p')+str(abs(angle))
            k=kernel(angle);run(k,label);k.close()
    else:followup()


if __name__=='__main__':main()
