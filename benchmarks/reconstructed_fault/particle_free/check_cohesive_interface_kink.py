#!/usr/bin/env python3
"""Bounded native mixed-FE kink transfer; no production/native time driver edits.
Native C++ supplies actual vector/pressure response; this reference owns trials.
"""
from pathlib import Path
import copy,csv,hashlib,json,time
import numpy as np
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parents[3]
WORK=ROOT/'build-particle-free/cohesive-interface-kink'
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_kink'
G=1e6;ETA=1e8;GC=1.;PEAK=3000.;P0=10000.;V0=5e-6;DC=.001;DIRECT=.005;EVOLUTION=.01;MU0=.1
RAD=G/(2*np.sqrt(G/2670.));C0=2*GC/PEAK**2;DF=2*GC/PEAK

def write(name,value): (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def csvwrite(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def read(path):return np.genfromtxt(path,delimiter=',',names=True)
def encode(state):return json.dumps(state,sort_keys=True,separators=(',',':'),allow_nan=False)

def point(old,increment,pressure,dt):
    a0=old['damage'];e=old['jump']+increment-old['reference'];a=np.maximum(a0,np.minimum(abs(e)/DF,1.));grow=(abs(e)>DF*a0)&(a<1)
    safe=np.where(a==0,1,a);stiff=(1-a)/(C0*safe)
    q=np.where(a==0,PEAK,stiff*e);dq=np.where(grow,-PEAK/DF,stiff);da=np.where(grow,np.sign(e)/DF,0.)
    dq=np.where(a==0,-PEAK/DF,dq);da=np.where(a==0,1/DF,da)
    v=increment/dt;speed=abs(v);sign=np.sign(v);theta=(old['theta']+V0*dt/DC)/(1+abs(increment)/DC)
    exponential=np.exp((MU0+EVOLUTION*np.log(theta))/DIRECT);z=speed*exponential/(2*V0);frac=z/np.hypot(1,z);mu=DIRECT*np.arcsinh(z)
    slope=np.divide(DIRECT*frac,speed,out=DIRECT*exponential/(2*V0),where=speed>0)-EVOLUTION*frac*(dt/DC)/(1+abs(increment)/DC)
    r=sign*(pressure*mu+RAD*speed)
    return dict(resistance=q+a*r,derivative=dq+da*r+a*(pressure*slope+RAD)/dt,pressure_derivative=a*sign*mu,new_damage=a,new_theta=theta,storage=.5*q*e,friction=dt*a*pressure*mu*speed,radiation=dt*a*RAD*speed**2)

class Kernel:
    def __init__(self,name):
        self.name=name;self.path=WORK/name
        for attr,file in [('K','stiffness'),('Q','Q'),('Nm','normal-minus'),('Np','normal-plus'),('E','energy-gauss6')]:setattr(self,attr,np.loadtxt(self.path/(file+'.csv'),delimiter=','))
        self.N=.5*(self.Nm+self.Np);self.nodes=read(self.path/'nodes.csv');self.s=read(self.path/'surface.csv');self.w=self.s['weight_m'];self.old=self.nodes['branch']==0;self.new=~self.old;self.oq=self.s['branch']==0;self.nq=~self.oq
        self.mass=self.Q.T@self.w;self.meta=json.loads((self.path/'native.json').read_text())
    def initial(self,gamma=.001):
        # A prepared, equilibrated loaded snapshot, not a claim to reproduce a
        # particular loading trajectory. Damage is fixed and unloading admissible.
        xx=self.s['x_m'];damage=np.where(self.oq,.4+.1*np.sin(4*np.pi*(xx-.25)),0.)
        reference=np.where(self.oq,2e-5*np.sin(4*np.pi*(xx-.25))**2,0.)
        stiff=np.where(self.oq,(1-damage)/(C0*np.where(damage==0,1,damage)),0.)
        Q=self.Q[:,self.old];M=self.K[1:,1:][np.ix_(self.old,self.old)]+Q.T@(self.w[:,None]*stiff[:,None]*Q)
        rhs=-self.K[1:,0][self.old]*gamma+Q.T@(self.w*stiff*reference)
        delta=np.zeros(len(self.nodes));delta[self.old]=np.linalg.solve(M,rhs)
        jump=self.Q@delta
        assert np.all(abs(jump-reference)[self.oq]<DF*damage[self.oq])
        hist=dict(jump=jump.tolist(),damage=damage.tolist(),theta=np.where(self.oq,1.2+.1*np.sin(8*np.pi*xx),1.001).tolist(),reference=reference.tolist(),relaxed_jump=jump.tolist(),absolute_slip=abs(jump).tolist())
        return dict(generation=0,publications=0,physical_publications=0,clock_s=.2,installed=[0],gamma=gamma,wall_memory=gamma,delta=delta.tolist(),memory=delta.tolist(),histories=hist,virgin_theta=1.001,geometry=[[.25,0],[.5,0]],layout=list(np.flatnonzero(self.old).astype(int).tolist()))
    def state_arrays(self,s):return {key:np.array(value) for key,value in s['histories'].items()}
    def coefficients(self,s):return np.r_[s['wall_memory'],s['memory']]
    def pressure(self,s):
        c=self.coefficients(s);return P0+self.Nm@c,P0+self.Np@c
    def certificate(self,s):
        # Minimize infinity norm of the actual Gauss multiplier, with exact
        # oblique arclength and full pressure-coupled bulk elimination.
        force=-(self.K@self.coefficients(s))[1:][self.new]
        L=self.Q[np.ix_(self.nq,self.new)].T*self.w[self.nq]
        nq=L.shape[1];A=np.c_[np.r_[np.eye(nq),-np.eye(nq)],-np.ones(2*nq)]
        result=linprog(np.r_[np.zeros(nq),1.],A_ub=A,b_ub=np.zeros(2*nq),A_eq=np.c_[L,np.zeros(len(force))],b_eq=force,bounds=[(None,None)]*nq+[(0,None)],method='highs')
        assert result.success,result.message
        error=max(abs(L@result.x[:-1]-force)/self.mass[self.new]);pm,pp=self.pressure(s)
        return dict(peak_required_Pa=float(result.x[-1]),residual_Pa=float(error),min_normal_Pa=float(min(pm.min(),pp.min())),feasible=bool(result.x[-1]<=PEAK and error<=1e-8 and min(pm.min(),pp.min())>0)),result.x[:-1]
    def energies(self,s):
        c=self.coefficients(s);h=self.state_arrays(s);a=h['damage'];e=h['jump']-h['reference'];bond=np.where(a>0,.5*(1-a)/(C0*np.where(a==0,1,a))*e*e,0.)
        return dict(bulk_J_m=float(.5*c@self.E@c),bond_J_m=float(self.w@bond),fracture_J_m=float(GC*self.w@a))
    def trial_step(self,s,dgamma=6e-4,dt=.002):
        beta=1/(1+G*dt/ETA);old=self.state_arrays(s);inc=np.zeros(len(self.nodes));oldm=np.array(s['memory']);gamma=beta*(s['wall_memory']+dgamma)
        ids=self.old
        def evaluate(z):
            inc[ids]=z;c=np.r_[gamma,beta*(oldm+inc)];pn=P0+self.N@c
            r=point(old,self.Q@inc,pn,dt);force=(self.K@c)[1:]+self.Q.T@(self.w*np.where(self.oq,r['resistance'],0.))
            J=beta*self.K[1:,1:]+self.Q.T@(self.w[:,None]*self.oq[:,None]*(r['derivative'][:,None]*self.Q+r['pressure_derivative'][:,None]*beta*self.N[:,1:]))
            return force[ids],J[np.ix_(ids,ids)],r,c
        z=inc[ids].copy();iterations=0
        for iterations in range(100):
            f,J,r,c=evaluate(z);norm=max(abs(f)/self.mass[ids])
            if norm<1e-8:break
            dz=np.linalg.solve(J,-f);alpha=1.
            for _ in range(40):
                fn,_,_,_=evaluate(z+alpha*dz)
                if max(abs(fn)/self.mass[ids])<norm:break
                alpha*=.5
            z+=alpha*dz
        else:raise RuntimeError(('Newton failed',norm))
        f,J,r,c=evaluate(z);inc[ids]=z
        # Coupled finite differences with current pressure response.
        rng=np.random.default_rng(41);direction=rng.normal(size=len(z));direction/=np.linalg.norm(direction)
        # Use 1e-6 of the actual increment scale, safely off stick/activation corners.
        eps=max(np.linalg.norm(z),1e-25)*1e-5
        plus=evaluate(z+eps*direction)[0];minus=evaluate(z-eps*direction)[0];fd=(plus-minus)/(2*eps);tangent_error=np.linalg.norm(fd-J@direction)/max(np.linalg.norm(fd),1.)
        f,J,r,c=evaluate(z);inc[ids]=z
        t=copy.deepcopy(s);t['clock_s']+=dt;t['gamma']+=dgamma;t['wall_memory']=gamma;t['delta']=(np.array(s['delta'])+inc).tolist();t['memory']=(beta*(oldm+inc)).tolist();t['physical_publications']+=1
        for key in old:
            val=old[key].copy()
            if key=='jump':val[self.oq]+=self.Q[self.oq]@inc
            if key=='damage':val[self.oq]=r['new_damage'][self.oq]
            if key=='theta':val[self.oq]=r['new_theta'][self.oq];val[self.nq]+=V0*dt/DC
            if key=='relaxed_jump':val[self.oq]=beta*(val[self.oq]+self.Q[self.oq]@inc)
            if key=='absolute_slip':val[self.oq]+=abs(self.Q[self.oq]@inc)
            t['histories'][key]=val.tolist()
        t['virgin_theta']+=V0*dt/DC
        before_energy=self.energies(s);after_energy=self.energies(t)
        oldc=self.coefficients(s);newc=self.coefficients(t);dc=newc-oldc
        work=float((self.K@newc)[0]*dgamma)
        maxwell=float(2*G*dt/ETA*.5*newc@self.K@newc)
        backward_euler=float(.5*dc@self.K@dc)
        a=old['damage'];elastic=old['jump']-old['reference']
        oldbond=.5*(1-a)/(C0*np.where(a==0,1,a))*elastic**2
        newe=np.array(t['histories']['jump'])-old['reference']
        newa=np.array(t['histories']['damage'])
        q=(1-newa)/(C0*np.where(newa==0,1,newa))*newe
        bondwork=float(self.w[self.oq]@(q*(self.Q@inc))[self.oq])
        bondchange=after_energy['bond_J_m']-before_energy['bond_J_m']
        fracture=after_energy['fracture_J_m']-before_energy['fracture_J_m']
        sliding=float(self.w[self.oq]@(r['friction']+r['radiation'])[self.oq])
        surface_remainder=bondwork-bondchange-fracture
        bulkchange=float(.5*newc@self.K@newc-.5*oldc@self.K@oldc)
        ledger=work-(bulkchange+bondchange+fracture+sliding+maxwell+backward_euler+surface_remainder)
        assert abs(ledger)<1e-10*max(1,abs(work))
        assert min(self.pressure(t)[0].min(),self.pressure(t)[1].min())>0
        pressure_part=self.Q.T@(self.w[:,None]*self.oq[:,None]*r['pressure_derivative'][:,None]*beta*self.N[:,1:])
        pressure_tangent_ratio=float(np.linalg.norm(pressure_part[np.ix_(ids,ids)])/np.linalg.norm(J))
        fixed_pressure_fd_error=float(np.linalg.norm(fd-(J-pressure_part[np.ix_(ids,ids)])@direction)/np.linalg.norm(fd))
        scaled=J/np.sqrt(np.outer(self.mass[ids],self.mass[ids]))
        minimum_symmetric_tangent=float(np.linalg.eigvalsh(.5*(scaled+scaled.T)).min())
        assert minimum_symmetric_tangent>0
        post_certificate,_=self.certificate(t);assert post_certificate['feasible']
        return t,dict(post_tied_certificate=post_certificate,minimum_symmetric_tangent_Pa_m=minimum_symmetric_tangent,fixed_pressure_fd_relative=fixed_pressure_fd_error,work_J_m=work,bulk_change_J_m=bulkchange,bond_change_J_m=bondchange,fracture_increment_J_m=fracture,maxwell_loss_J_m=maxwell,bulk_BE_J_m=backward_euler,surface_remainder_J_m=surface_remainder,ledger_error_J_m=ledger,pressure_tangent_relative=pressure_tangent_ratio,residual_Pa=float(max(abs(f)/self.mass[ids])),tangent_relative=float(tangent_error),iterations=iterations+1,peak_increment_m=float(max(abs(inc))),sliding_loss_J_m=float(self.w[self.oq]@(r['friction']+r['radiation'])[self.oq]))

def topology(vertices):
    p=np.array(vertices,float);d=np.diff(p,axis=0)
    if np.any(np.linalg.norm(d,axis=1)<1e-12):raise ValueError('zero-length edge')
    # This task permits one forward graph append; crossings/backtracking are rejected.
    if np.any(d[:,0]<=0):raise ValueError('non-forward/crossing topology outside control')
    if len(vertices)>3:raise ValueError('only one append qualified')

def insert(kernel,accepted,fail=False):
    before=encode(accepted);candidate=copy.deepcopy(accepted)
    if candidate['geometry']!=[[.25,0],[.5,0]]:raise ValueError('existing material geometry may not rotate or relabel')
    if candidate['installed']!=[0]:raise ValueError('duplicate or stale insertion')
    candidate['geometry'].append([.75,.125]);topology(candidate['geometry'])
    candidate['installed']=[0,1];candidate['layout']=list(range(len(kernel.nodes)))
    cert,mult=kernel.certificate(candidate)
    if fail or not cert['feasible']:
        assert encode(accepted)==before
        return None,cert
    candidate['generation']+=1;candidate['publications']+=1
    return candidate,cert

def run(name):
    k=Kernel(name);s=k.initial();before=encode(s)
    rejected,cert=insert(k,s,True);assert rejected is None and encode(s)==before
    after,cert=insert(k,s);assert after is not None,cert
    retry,retrycert=insert(k,s);assert encode(retry)==encode(after)
    checkpoint=OUT/(name+'-accepted.json');checkpoint.write_text(encode(after)+'\n');restored=json.loads(checkpoint.read_text());assert encode(restored)==encode(after)
    try:insert(k,after)
    except ValueError:duplicate=True
    else:duplicate=False
    assert duplicate and after['publications']==1
    assert s['histories']==after['histories'] and s['clock_s']==after['clock_s']
    assert k.energies(s)==k.energies(after)
    # Actual coefficient expansion in the old/new FE trace spaces.
    Pold=np.eye(len(k.nodes))[:,k.old];oldcoeff=Pold@np.array(s['delta'])[k.old];newcoeff=np.eye(len(k.nodes))@np.array(after['delta'])
    assert np.array_equal(oldcoeff,newcoeff)
    c=k.coefficients(s);natives=np.loadtxt(k.path/'response.csv',delimiter=',');bulk=natives@c
    assert np.array_equal(bulk,natives@k.coefficients(after))
    # A subsequent physical step uses the unchanged point return, twice from
    # the same accepted state (restart/retry), with new tied modes constrained.
    step,stepcheck=k.trial_step(after);repeat,_=k.trial_step(restored);assert encode(step)==encode(repeat)
    preinstalled=copy.deepcopy(s);preinstalled.update(installed=[0,1],geometry=after['geometry'],layout=after['layout'],generation=1,publications=1)
    control,_=k.trial_step(preinstalled);assert encode(control)==encode(step)
    assert stepcheck['tangent_relative']<1e-6
    zero=k.initial(0.);# Remove old history as well for the genuinely unloaded control.
    for key in ['jump','damage','reference','relaxed_jump','absolute_slip']:zero['histories'][key]=np.zeros(len(k.s)).tolist()
    zero['delta']=np.zeros(len(k.nodes)).tolist();zero['memory']=zero['delta'].copy()
    unloaded,uc=insert(k,zero);assert unloaded is not None and k.energies(unloaded)==dict(bulk_J_m=0.,bond_J_m=0.,fracture_J_m=0.)
    # Isolate new-plane admission: axial deviatoric stress gives zero shear on
    # the existing horizontal tied plane, but 4000 Pa on the oblique append.
    ak=Kernel('native-16-axial');overloaded=ak.initial(0.)
    for key in ['jump','damage','reference','relaxed_jump','absolute_slip']:overloaded['histories'][key]=np.zeros(len(ak.s)).tolist()
    overloaded['delta']=np.zeros(len(ak.nodes)).tolist();overloaded['memory']=overloaded['delta'].copy()
    overloaded['gamma']=overloaded['wall_memory']=.0025
    old_force=(ak.K@ak.coefficients(overloaded))[1:][ak.old]/ak.mass[ak.old]
    assert max(abs(old_force))<1e-7
    bad,bc=insert(ak,overloaded);assert bad is None and not bc['feasible']
    bc['old_tied_weak_shear_Pa']=float(max(abs(old_force)));bc['affine_loading']='u=(gamma*x,-gamma*y), gamma=.0025; tau_xx=5000 Pa, tau_yy=-5000 Pa'
    # Native evaluator cross-check of every point partial; covers intact/damaged,
    # reverse and zero increments without changed calibration.
    ref=read(k.path/'point-reference.csv');old={key:ref[key] for key in ['damage','jump','reference','theta']};r=point(old,ref['increment'],ref['pressure'],ref['dt'])
    errors={key:float(np.max(abs(r[key]-ref[key])/np.maximum(abs(ref[key]),1.))) for key in r};assert max(errors.values())<1e-12,errors
    result=dict(native=k.meta,certificate=cert,unloaded_certificate=uc,overloaded_certificate=bc,geometry_publications=after['publications'],physical_publications_after_trial=step['physical_publications'],rollback=True,retry_identity=True,restart_identity=True,preinstalled_identity=True,duplicate_rejected=duplicate,physical_transfer_error=0.,storage_before=k.energies(s),storage_after=k.energies(after),old_history_bitwise=True,point_errors=errors,physical_step=stepcheck,peak_old_jump_m=float(max(abs(np.array(s['delta'])))),quadrature_energy_relative=float(abs(c@(k.E-k.K)@c)/abs(c@k.E@c)),mixed_symmetry_relative=float(np.max(abs(k.K-k.K.T))/np.max(abs(k.K))))
    rows=[];pm,pp=k.pressure(s);hist=k.state_arrays(s);Qdelta=k.Q@np.array(s['delta']);_,mult=k.certificate(s)
    for i in range(len(k.s)):
        rows.append(dict(mesh=name,q=i,x_m=float(k.s['x_m'][i]),y_m=float(k.s['y_m'][i]),weight_m=float(k.w[i]),branch=int(k.s['branch'][i]),damage=hist['damage'][i],jump_m=Qdelta[i],theta=hist['theta'][i],normal_minus_Pa=pm[i],normal_plus_Pa=pp[i],tied_multiplier_Pa=float(mult[np.count_nonzero(k.nq[:i])]) if k.nq[i] else 0.))
    csvwrite(name+'-surface.csv',rows)
    write(name+'-checks.json',result)
    write(name+'-physical-step.json',step)
    return k,s,result

def fields(k,s):
    raw=read(k.path/'field-response.csv');nm=len(k.nodes)+1;arr=raw.reshape(-1,nm);c=k.coefficients(s)
    values={key:arr[key]@c for key in ['tau_xx_Pa','tau_yy_Pa','tau_zz_Pa','tau_xy_Pa','pprime_Pa']}
    # The truly homogeneous affine patch must be exact despite the kinked mesh.
    affine=np.r_[.001,np.zeros(nm-1)]
    affine_error=max(float(np.max(abs(arr[key]@affine-(1000. if key=='tau_xy_Pa' else 0.)))) for key in ['tau_xx_Pa','tau_yy_Pa','tau_zz_Pa','tau_xy_Pa','pprime_Pa'])
    assert affine_error<1e-7
    virtual=np.zeros(nm);xx=k.nodes['x_m'];virtual[1:][k.new]=1e-5*np.sin(4*np.pi*(xx[k.new]-.5))
    energy=float(.5*virtual@k.E@virtual)
    work=float(.5*virtual@k.K@virtual)
    tangents=np.c_[k.s['tx'],k.s['ty']];normals=np.c_[-k.s['ty'],k.s['tx']]
    jumps=(k.Q@virtual[1:])[:,None]*tangents
    gap=float(np.max(abs(np.sum(jumps*normals,axis=1))))
    pnminus=k.Nm@virtual;pnplus=k.Np@virtual
    write(k.name+'-oblique-mode.json',dict(affine_patch_error_Pa=affine_error,normal_gap_m=gap,energy_Gauss6_J_m=energy,half_weak_work_J_m=work,quadrature_relative=abs(energy-work)/energy,normal_pressure_response_Pa=float(max(abs(pnminus).max(),abs(pnplus).max())),minimum_nodal_jump_stiffness_Pa=float(np.linalg.eigvalsh(k.K[1:,1:]).min()),note='virtual new-branch variation tests the mechanical/pressure representation, not accepted slip in tied material'))
    assert energy>0 and gap<1e-18 and max(abs(pnminus).max(),abs(pnplus).max())>1

    x=arr['x_m'][:,0];y=arr['y_m'][:,0];w=arr['weight_m2'][:,0]
    radius=.1;r2=((x-.5)**2+y*y)/radius**2
    # Compact continuous kernel, same physical radius/center on each mesh.
    kw=w*np.maximum(0,1-r2)**2;kw/=sum(kw)
    tensor=np.array([[kw@values['tau_xx_Pa'],kw@values['tau_xy_Pa']],[kw@values['tau_xy_Pa'],kw@values['tau_yy_Pa']]])
    val,vec=np.linalg.eigh(tensor);axis=np.degrees(np.arctan2(vec[1,-1],vec[0,-1]))%180
    rows=[]
    for i in range(len(x)):
        rows.append(dict(mesh=k.name,x_m=x[i],y_m=y[i],weight_m2=w[i],**{key:v[i] for key,v in values.items()},kernel_weight=kw[i]))
    csvwrite(k.name+'-fields.csv',rows)
    return dict(tensor_Pa=tensor.tolist(),tau_zz_Pa=float(kw@values['tau_zz_Pa']),pprime_Pa=float(kw@values['pprime_Pa']),principal_axis_deg=float(axis),maximum_shear_planes_deg=[float((axis+45)%180),float((axis-45)%180)],kernel_radius_m=radius,trace_error_Pa=float(max(abs(values['tau_xx_Pa']+values['tau_yy_Pa']+values['tau_zz_Pa']))))

def main():
    OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter();results={};local={}
    for vertices in [[[.25,0],[.5,0],[.5,0]],[[.25,0],[.5,0],[.4,.1]],[[.25,0],[.5,0],[.75,.125],[.8,.1]]]:
        try:topology(vertices)
        except ValueError:pass
        else:raise AssertionError('Unsupported topology accepted')

    for name in ['native-16','native-32','native-16-q6']:
        k,s,r=run(name);results[name]=r;local[name]=fields(k,s)
    a,b=local['native-16'],local['native-32'];ta=np.array(a['tensor_Pa']);tb=np.array(b['tensor_Pa']);tensor_error=np.sqrt(np.linalg.norm(ta-tb)**2+(a['tau_zz_Pa']-b['tau_zz_Pa'])**2)/np.sqrt(np.linalg.norm(tb)**2+b['tau_zz_Pa']**2);angle=abs((a['principal_axis_deg']-b['principal_axis_deg']+90)%180-90)
    comparison=dict(tensor_relative=float(tensor_error),axis_difference_deg=float(angle),averaged_field_gates_passed=bool(tensor_error<=.05 and angle<=5),interpretation='prescribed kink; no automatic direction or first-activation timing qualified')
    write('local-resolution.json',dict(samples=local,comparison=comparison))
    write('summary.json',dict(cases=results,local_resolution=comparison,seconds=time.perf_counter()-start,passed=True))
    print(json.dumps(dict(transfer='passed',local_resolution=comparison,seconds=time.perf_counter()-start),indent=2))
if __name__=='__main__':main()
