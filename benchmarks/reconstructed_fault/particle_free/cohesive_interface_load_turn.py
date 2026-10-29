#!/usr/bin/env python3
"""Two physical affine boundary histories and causal append-only geometry."""
import argparse
import copy
import ctypes as ct
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time
from types import SimpleNamespace, MethodType
import numpy as np
from scipy.linalg import null_space

from cohesive_interface_path import ROOT, R, HISTORY, graph, new_arc, axes, functionals as original_functionals, save, table, encode, sha
from cohesive_interface_turn import Kernel as OldKernel
from cohesive_interface_admission import Admission, bounded_multiplier
from check_cohesive_interface_kink import point, G, ETA, GC, C0, V0, DC

WORK=ROOT/'build-particle-free/cohesive-interface-load-turn'
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_load_turn'
TURN_START=.410;TURN_END=.440;LOAD_END=.455;END=.460;RATE=.004
MAX_NEW=4.125;WIDTH=.75


def functionals(k,s):
    result=original_functionals(k,s)
    # Direction changes may unload slip while damage is irreversible. Use
    # actual owned damage records; current-slip threshold reconstruction is
    # not valid in that situation. Give physical brackets, never interpolate
    # those records into another constitutive state.
    new=np.flatnonzero(k.s['x_m']>6);a=np.array(s['histories']['damage'])
    for threshold,name in [(.05,'process'),(.5,'half'),(1.,'full')]:
        j=0
        while j<len(new) and a[new[j]]>=threshold:j+=1
        lower=6. if j==0 else float(k.s['x_m'][new[j-1]])
        upper=float(k.s['x_m'][new[j]]) if j<len(new) else k.arcs[-1]['end']
        middle=.5*(lower+upper) if j else 6.
        arc=next(v for v in k.arcs if middle<=v['end']+1e-12);z=(middle-arc['x'])/(arc['end']-arc['x']);y,p=graph(arc,z)
        def distance(x):
            total=0.;q,w=np.polynomial.legendre.leggauss(12);q=(q+1)/2;w=w/2
            for v in k.arcs:
                end=min(x,v['end'])
                if end<=v['x']:break
                t=(end-v['x'])/(v['end']-v['x']);_,p=graph(v,t*q)
                total+=(end-v['x'])*(w@np.sqrt(1+p*p))
            return float(total)
        lo,hi=distance(lower),distance(upper)
        result.update({name+'_front_x_m':middle,name+'_front_y_m':float(y),name+'_front_arc_m':distance(middle),
            name+'_front_tangent_deg':float(np.rad2deg(np.arctan(p))),name+'_front_lower_x_m':lower,name+'_front_upper_x_m':upper,
            name+'_front_lower_arc_m':lo,name+'_front_upper_arc_m':hi,
            name+'_detached_Gauss':int(np.count_nonzero(a[new[j:]]>=threshold))})
        if name=='full':result['new_full_front_available']=j>0
    return result


def loading(t,turned=True):
    z=np.clip((t-TURN_START)/(TURN_END-TURN_START),0,1)
    alpha=np.deg2rad(-20.)*(z**3*(10-15*z+6*z*z)) if turned else 0.
    gamma=.0003+RATE*min(t,LOAD_END)
    return gamma*np.array([np.cos(2*alpha),np.sin(2*alpha)]),float(alpha),float(gamma)


def kernel(arcs,n=64,degree=2,mesh='cartesian',order=6):
    signature=dict(arcs=arcs,n=n,degree=degree,mesh=mesh,order=order,lifting_half_width_m=WIDTH,
        normal_half_width_m=WIDTH,affine_modes=2,version=1)
    name='kernel-'+hashlib.sha256(encode(signature).encode()).hexdigest()[:16];folder=WORK/name
    if not (folder/'native.json').exists():
        folder.mkdir(exist_ok=False);save(folder/'input.json',signature);table(folder/'geometry.csv',arcs)
        np.savetxt(folder/'probe-coefficients.txt',np.zeros(sum(a['cells'] for a in arcs)+1))
        command=[str(WORK/'build/cohesive_interface_path'),str(n),str(degree),str(folder/'geometry.csv'),str(folder),str(order),
            'compress',str(WIDTH),str(WIDTH),mesh,str(folder/'probe-coefficients.txt'),'two-affine']
        with (WORK/(name+'.log')).open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300,
                env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
    return Kernel(folder,arcs)


class Kernel:
    close=OldKernel.close

    def __init__(self,folder,arcs,native_library_path=None):
        self.path=folder;self.arcs=copy.deepcopy(arcs);self.name=folder.name;self.meta=json.loads((folder/'native.json').read_text())
        assert self.meta['affine_modes']==2 and json.loads((folder/'input.json').read_text())['arcs']==arcs
        for key,name in [('K','stiffness'),('Q','Q'),('N','normal-work'),('raw_Nm','normal-minus'),('raw_Np','normal-plus'),('probe','probe'),('Jx','jump-x'),('Jy','jump-y')]:
            setattr(self,key,np.loadtxt(folder/(name+'.csv'),delimiter=','))
        self.s=np.genfromtxt(folder/'surface.csv',delimiter=',',names=True);self.nodes=np.genfromtxt(folder/'nodes.csv',delimiter=',',names=True)
        self.centers=np.genfromtxt(folder/'cells.csv',delimiter=',',names=True);self.w=self.s['weight_m'];self.mass=self.Q.T@self.w
        self.key=sha(folder/'stiffness.csv')+'-'+sha(folder/'normal-work.csv');self._field=None;self._mean=None
        assert self.meta['energy_relative']<1e-4 and self.meta['gradient_FD_relative']<1e-6
        check=np.loadtxt(folder/'normal-work-check.csv',delimiter=',');assert np.linalg.norm(check-self.N)/np.linalg.norm(self.N)<1e-4
        self.lib=ct.CDLL(str(native_library_path or WORK/'build/libcohesive_interface_turn_solver.so'));D=ct.POINTER(ct.c_double)
        self.lib.cohesive_turn_create_multi.argtypes=[ct.c_uint]*3+[D]*6;self.lib.cohesive_turn_create_multi.restype=ct.c_void_p
        self.lib.cohesive_turn_solve_multi.argtypes=[ct.c_void_p,D,D,ct.c_double,D,D,ct.POINTER(ct.c_int),D,D,D,D,ct.POINTER(ct.c_int),D,ct.POINTER(ct.c_char),ct.c_uint]
        self.lib.cohesive_turn_solve_multi.restype=ct.c_int;self.lib.cohesive_turn_destroy.argtypes=[ct.c_void_p]
        self.arrays=[np.ascontiguousarray(a) for a in [self.K,self.Q,self.N,self.N,self.w,self.mass]]
        self.context=self.lib.cohesive_turn_create_multi(len(self.nodes),len(self.s),2,*[a.ctypes.data_as(D) for a in self.arrays])

    def coefficients(self,s):return np.r_[s['affine_memory'],s['memory']]

    def field_table(self,name):
        p=self.path/name
        data=np.fromfile(p,dtype=np.float64) if p.exists() else np.frombuffer(gzip.decompress((self.path/(name+'.gz')).read_bytes()),dtype=np.float64)
        return data.reshape(len(self.centers),len(self.nodes)+2,5)

    @property
    def field(self):
        if self._field is None:self._field=self.field_table('field-center.bin')
        return self._field

    @property
    def mean(self):
        if self._mean is None:self._mean=self.field_table('field-mean.bin')
        return self._mean

    def initial(self):
        x=self.s['x_m'];shape=4*((x-3)/3)*(1-(x-3)/3);a=.75+.05*shape;jump=.0004*shape
        h=dict(jump=jump.tolist(),damage=a.tolist(),theta=(1.2+.1*shape).tolist(),reference=(jump-C0*a/(1-a)*300).tolist(),
            relaxed_jump=np.zeros(len(x)).tolist(),absolute_slip=abs(jump).tolist())
        return dict(clock_s=0.,gamma=.0003,alpha_deg=0.,imposed=[.0003,0.],affine_memory=[.0003,0.],memory=np.zeros(len(self.nodes)).tolist(),
            delta=np.zeros(len(self.nodes)).tolist(),nodal_rate=np.zeros(len(self.nodes)).tolist(),jump_offset=jump.tolist(),histories=h,
            last_resistance=np.full(len(x),300.).tolist(),geometry=copy.deepcopy(self.arcs),kernel=self.key,kernel_folder=self.name,
            physical_publications=0,generation=0,publications=0,virgin_theta=1.,affine_modes=2)

    def predictor(self,s,dt,da):
        # Algebraically eliminate known affine values in the SEARCH operator.
        # The native solve below still receives both physical histories and
        # verifies the complete original mixed/constitutive residual.
        beta=1/(1+G*dt/ETA);bg=beta*(np.array(s['affine_memory'])+da);n=len(self.nodes)
        K=np.zeros((n+1,n+1));K[1:,1:]=self.K[2:,2:];K[1:,0]=self.K[2:,:2]@bg;K[0,1:]=K[1:,0]
        N=np.c_[self.N[:,:2]@bg,self.N[:,2:]]
        search=SimpleNamespace(K=K,N=N,Nm=N,Np=N,Q=self.Q,w=self.w,mass=self.mass,nodes=self.nodes)
        search.predictor=MethodType(OldKernel.predictor,search)
        fake=copy.deepcopy(s);fake['wall_memory']=1/beta
        return OldKernel.admission_predictor(search,fake,dt,0.)

    def trial(self,s,target,turned=True):
        assert s['kernel']==self.key and s['geometry']==self.arcs
        dt=target-s['clock_s'];assert dt>0
        imposed,alpha,gamma=loading(target,turned);da=np.ascontiguousarray(imposed-np.array(s['imposed']))
        c0=np.ascontiguousarray(self.coefficients(s));old=np.ascontiguousarray(np.array([s['histories'][key] for key in HISTORY]).T)
        n=len(self.nodes);nq=len(self.s);D=ct.POINTER(ct.c_double)
        c=np.empty(n+2);inc=np.empty(n);r=np.empty((nq,14));J=np.empty((n,n));tied=np.empty(nq,dtype=np.int32);meta=np.empty(6);error=ct.create_string_buffer(16000)
        guess,hint=self.predictor(s,dt,da)
        status=self.lib.cohesive_turn_solve_multi(self.context,c0.ctypes.data_as(D),old.ctypes.data_as(D),dt,da.ctypes.data_as(D),
            guess.ctypes.data_as(D),hint.ctypes.data_as(ct.POINTER(ct.c_int)),*[v.ctypes.data_as(D) for v in [c,inc,r,J]],
            tied.ctypes.data_as(ct.POINTER(ct.c_int)),meta.ctypes.data_as(D),error,len(error))
        if status:raise RuntimeError(error.value.decode())
        raw_min=float(min(np.min(10000+self.raw_Nm@c),np.min(10000+self.raw_Np@c)));assert raw_min>0
        t=copy.deepcopy(s);t.update(clock_s=target,gamma=gamma,alpha_deg=float(np.rad2deg(alpha)),imposed=imposed.tolist(),
            affine_memory=c[:2].tolist(),memory=c[2:].tolist(),delta=(np.array(s['delta'])+inc).tolist(),nodal_rate=(inc/dt).tolist(),
            virgin_theta=s['virgin_theta']+V0*dt/DC,histories={key:r[:,i].tolist() for i,key in enumerate(HISTORY)},last_resistance=r[:,10].tolist())
        assert max(abs(np.array(t['jump_offset'])+self.Q@t['delta']-r[:,0]))<1e-13
        assert max(abs(self.Q@c[2:]-r[:,4]))<1e-13
        E=.5*c@self.K@c;E0=.5*c0@self.K@c0;a=old[:,1];e=old[:,0]-old[:,3]
        B0=self.w@(.5*(1-a)/(C0*np.where(a==0,1,a))*e**2);B=self.w@r[:,13];F=GC*self.w@(r[:,1]-a)
        friction=self.w@r[:,7];radiation=self.w@r[:,8];maxwell=2*G/ETA*dt*E
        be=.5*(c-c0)@self.K@(c-c0);surface=self.w@(r[:,6]*(self.Q@inc))-(B-B0)-F
        reaction=(self.K@c)[:2];reaction0=(self.K@c0)[:2];work=reaction@da;trap=.5*(reaction+reaction0)@da
        ledger=work-(E-E0+B-B0+F+friction+radiation+maxwell+be+surface)
        assert abs(ledger)<1e-10*max(1,abs(work)),ledger
        physical=np.r_[imposed,t['delta']];gap=-self.s['ty']*(self.Jx@physical)+self.s['tx']*(self.Jy@physical)
        assert max(abs(gap))<1e-12
        tensor=self.probe@c;preferred,info=axes(tensor,np.arctan(self.arcs[-1]['b']))
        d=dict(time_s=target,dt_s=dt,gamma=gamma,alpha_deg=t['alpha_deg'],imposed_0=imposed[0],imposed_1=imposed[1],
            affine_memory_0=c[0],affine_memory_1=c[1],preferred_deg=float(np.rad2deg(preferred)),compression_deg=info['compression_deg'],
            sample_xx_Pa=tensor[0],sample_yy_Pa=tensor[1],sample_zz_Pa=tensor[2],sample_xy_Pa=tensor[3],sample_pprime_Pa=tensor[4],
            bulk_J_m=float(E),bond_J_m=float(B),fracture_increment_J_m=float(F),total_fracture_J_m=float(GC*self.w@r[:,1]),
            friction_J_m=float(friction),radiation_J_m=float(radiation),sliding_loss_J_m=float(friction+radiation),maxwell_loss_J_m=float(maxwell),
            endpoint_work_J_m=float(work),trapezoid_work_J_m=float(trap),physical_remainder_J_m=float(trap-(E-E0+B-B0+F+friction+radiation+maxwell)),
            ledger_error_J_m=float(ledger),weak_error_Pa=float(meta[0]),minimum_symmetric_tangent_Pa_m=float(meta[1]),minimum_normal_Pa=float(meta[2]),
            raw_minimum_normal_Pa=raw_min,normal_gap_m=float(max(abs(gap))),iterations=int(meta[3]),max_damage_increment=float(max(r[:,1]-a)),
            peak_velocity_m_s=float(max(abs(r[:,9]))),integrated_slip_m2=float(self.w@r[:,0]),**functionals(self,t))
        return dict(base_publications=s['physical_publications'],base_generation=s['generation'],state=t,diagnostics=d,
            native=dict(c=c,increment=inc,r=r,J=J),tied=tied.tolist())

    def birth(self,old,s):
        n=len(old.nodes);nq=len(old.s);nc=n+2;before=encode(s)
        assert s['kernel']==old.key and self.arcs[:-1]==old.arcs
        c=old.coefficients(s);cp=np.pad(c,(0,len(self.nodes)-n))
        assert np.array_equal(self.K[:nc,:nc],old.K) and np.array_equal(self.Q[:nq,:n],old.Q)
        oldcols=np.genfromtxt(old.path/'columns.csv',delimiter=',',names=True,dtype=None,encoding=None)
        newcols=np.genfromtxt(self.path/'columns.csv',delimiter=',',names=True,dtype=None,encoding=None)
        assert np.array_equal(oldcols,newcols[:nc])
        assert np.array_equal(self.field[:,:nc],old.field)
        t=copy.deepcopy(s);extra=len(self.s)-nq
        for key in HISTORY:t['histories'][key]+=[s['virgin_theta'] if key=='theta' else 0.]*extra
        for key in ['delta','memory','nodal_rate']:t[key]+=[0.]*(len(self.nodes)-n)
        for key in ['jump_offset','last_resistance']:t[key]+=[0.]*extra
        tied=np.array(t['histories']['damage'])==0;r=np.array(t['last_resistance']);r[tied]=0
        force=(self.K@cp)[2:]+self.Q.T@(self.w*r);lam,error,peak=bounded_multiplier(self,force,tied)
        if error>=1e-8 or peak>3000:
            save(self.path/'rejected-birth.json',dict(incoming=s,required_multiplier_Pa=peak,weak_error_Pa=error,
                proposed_geometry=self.arcs,coefficient_vector=cp.tolist(),force_N_m=force.tolist(),tied=tied.tolist()))
            raise RuntimeError(f'Tied birth inadmissible: weak error {error:.17g} Pa, required peak {peak:.17g} Pa > 3000 Pa')
        tensor_error=float(max(abs(np.einsum('ijk,j->ik',self.field[:,:nc],c)-np.einsum('ijk,j->ik',old.field,c)).ravel()))
        energy=float(.5*cp@self.K@cp-.5*c@old.K@c);assert tensor_error<1e-7 and abs(energy)<1e-10*max(1,.5*c@old.K@c)
        t.update(kernel=self.key,kernel_folder=self.name,geometry=copy.deepcopy(self.arcs),generation=s['generation']+1,publications=s['publications']+1)
        for key in HISTORY:assert t['histories'][key][:nq]==s['histories'][key]
        assert encode(s)==before
        return t,dict(old_K_bitwise=True,old_Q_bitwise=True,old_response_columns_bitwise=True,old_physical_stress_pressure_columns_bitwise=True,
            tensor_error_Pa=tensor_error,energy_change_J_m=energy,old_histories_bitwise=True,new_fracture_J_m=0.,new_bond_J_m=0.,
            weak_Pa=error,new_tied_peak_Pa=peak,maximum_incoming_memory_m=float(max(abs(c[2:]),default=0)))


def publish_geometry(s,t,base):
    assert encode(s)==base,'stale geometry publication'
    assert t['generation']==s['generation']+1 and t['publications']==s['publications']+1
    assert t['physical_publications']==s['physical_publications'] and t['clock_s']==s['clock_s']
    return copy.deepcopy(t)


def run(label,n=64,reserve=.1875,extension=.375,dt=.00025,turned=True,checkpoint=None,end=END):
    folder=WORK/label;folder.mkdir(exist_ok=False);tic=time.perf_counter()
    s=None if checkpoint is None else json.loads(Path(checkpoint).read_text())
    arcs=[dict(x=3.,y=0.,end=6.,a=0.,b=0.,cells=32,c1=0.,c2=0.,c3=0.)] if s is None else s['geometry']
    k=kernel(arcs,n);s=k.initial() if s is None else s;assert s['kernel']==k.key
    save(folder/'initial.json',s);save(folder/'run-input.json',dict(label=label,n=n,reserve_m=reserve,extension_m=extension,dt_s=dt,turned=turned,end_s=end,
        physical_R_m=R,numerical_support_half_width_m=WIDTH,turn_start_s=TURN_START,turn_end_s=TURN_END,load_end_s=LOAD_END,
        permanent_virgin_directions=True,source_checkpoint=None if checkpoint is None else str(checkpoint),source_sha256=sha(Path(__file__))))
    rows=[];events=[];snapshots=[copy.deepcopy(s)];last=s;start_pub=s['physical_publications']
    grid=sorted(set(np.round(np.arange(dt,end+dt/2,dt),12).tolist()+[.15,TURN_START,TURN_END,LOAD_END,end]))
    try:
        for target in grid:
            if target<=s['clock_s']+1e-13:continue
            while s['clock_s']>=.15-1e-12:
                f=functionals(k,s);remaining=f['geometry_length_m']-f['process_front_arc_m']
                if remaining>=reserve-1e-10:break
                total=sum(a['cells']*.09375 for a in arcs[1:]);assert total+extension<=MAX_NEW+1e-10,'Bounded geometry capacity reached'
                tensor=k.probe@k.coefficients(s);preferred,decision=axes(tensor,np.arctan(arcs[-1]['b']))
                arc=new_arc(arcs,preferred,extension,.09375)
                assert arc['end']<12-R and abs(float(graph(arc,1.)[0]))<6-R,'Boundary margin exhausted'
                save(folder/'last-accepted.json',s);candidate=kernel(arcs+[arc],n)
                base=encode(s)
                try:
                    t,metrics=candidate.birth(k,s);retry,m2=candidate.birth(k,json.loads(base));assert encode(t)==encode(retry) and metrics==m2
                except Exception:
                    assert encode(s)==base
                    candidate.close()
                    raise
                events.append(dict(time_s=s['clock_s'],generation=t['generation'],reserve_before_m=remaining,reserve_target_m=reserve,arc=arc,
                    tensor_Pa=tensor.tolist(),decision=decision,transfer=metrics,terminal_deg=float(np.rad2deg(np.arctan(arc['b'])))))
                save(folder/f'before-extension-{t["generation"]}.json',s);save(folder/f'after-extension-{t["generation"]}.json',t)
                accepted=publish_geometry(s,t,base)
                try:publish_geometry(accepted,t,base)
                except AssertionError:pass
                else:raise RuntimeError('Duplicate geometry publication accepted')
                k.close();k=candidate;s=accepted;arcs=s['geometry'];last=s;snapshots.append(copy.deepcopy(s));save(folder/'events.json',events)
                print(label,'append',s['generation'],s['clock_s'],decision['selected_preference_deg'],events[-1]['terminal_deg'],flush=True)
            before=encode(s);p=k.trial(s,target,turned);assert encode(s)==before
            s=Admission(k).publish(s,p);assert s['physical_publications']==last['physical_publications']+1;last=s;rows.append(p['diagnostics'])
            if len(rows)%40==0 or target in [.15,TURN_START,TURN_END,LOAD_END,end]:snapshots.append(copy.deepcopy(s));save(folder/'last-accepted.json',s)
            if len(rows)%200==0:print(label,target,len(k.nodes),round(time.perf_counter()-tic,2),flush=True)
            assert time.perf_counter()-tic<1200,'Trajectory budget reached'
    except Exception as e:
        save(folder/'last-accepted.json',last);save(folder/'failure.json',dict(target_s=target,error=str(e),events=events,seconds=time.perf_counter()-tic))
        save(folder/'snapshots.json',snapshots);save(folder/'events.json',events)
        if rows:table(OUT/(label+'-partial.csv'),rows)
        k.close();raise
    save(folder/'final.json',s);save(folder/'snapshots.json',snapshots);save(folder/'events.json',events);table(OUT/(label+'-history.csv'),rows)
    physical=sum(abs(r['physical_remainder_J_m']) for r in rows);sliding=sum(r['sliding_loss_J_m'] for r in rows);fracture=sum(r['fracture_increment_J_m'] for r in rows)
    result=dict(label=label,n=n,reserve_m=reserve,extension_m=extension,dt_s=dt,turned=turned,seconds=time.perf_counter()-tic,
        steps=len(rows),physical_publication_increment=s['physical_publications']-start_pub,geometry_publications=s['publications'],events=events,
        final=rows[-1],absolute_physical_error_J_m=physical,sliding_J_m=sliding,fracture_J_m=fracture,
        relative_sliding=physical/sliding,relative_fracture=physical/fracture if fracture else None,owner_peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    save(OUT/(label+'-summary.json'),result);k.close();print('completed',label,result['seconds'],rows[-1]['new_damage_integral_m'],flush=True);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--n',type=int,default=64);p.add_argument('--reserve',type=float,default=.1875)
    p.add_argument('--extension',type=float,default=.375);p.add_argument('--dt',type=float,default=.00025);p.add_argument('--control',action='store_true');p.add_argument('--checkpoint',type=Path);p.add_argument('--end',type=float,default=END)
    a=p.parse_args();run(a.label,a.n,a.reserve,a.extension,a.dt,not a.control,a.checkpoint,a.end)


if __name__=='__main__':main()
