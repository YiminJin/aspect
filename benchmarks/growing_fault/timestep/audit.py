"""Independent mapped Q2/SIPG, section transfer and physical-work checks for the 5B fixture."""
import argparse
from filecmp import cmp as identical
import json
import re
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix

from frames import Comparison, aligned, compare, field_names, read, shards

G, ETA, ELL = 1.e6, 1.e8, .25
KF = 26.6666666667/((8/3)*ELL)
M = KF/(1000**2/(2*G))
CONTRACT = np.array([1.,1.,1.,2.])
ORDER = [0,2,6,8,3,5,1,7,4]  # continuous Q2 system ordering (x fastest in tensor product)


def shapes(x,y,width,nodal=False):
    def axis(t):
        if nodal:
            return np.array([2*t*t-3*t+1,4*t*(1-t),2*t*t-t]), np.array([4*t-3,4-8*t,4*t-1])
        return np.array([(1-t)**2,2*t*(1-t),t*t]), np.array([2*(t-1),2-4*t,2*t])
    bx,dx = axis(x)
    by,dy = axis(y)
    def product(a,b):
        out = (a[:,None,:]*b[None,:,:]).reshape(9,-1).T
        return out[:,ORDER] if nodal else out
    return product(by,bx),product(by,dx)/width[0],product(dy,bx)/width[1]


class Geometry:
    def __init__(self,mesh,points):
        self.cells, ix, inv = np.unique(points['cell'],return_index=True,return_inverse=True)
        if len(self.cells) != 128 or len(np.unique(points['section'])) != 18:
            raise ValueError('Expected the 16 x 8 specimen with all 18 oblique strips')
        self.groups = np.split(np.arange(len(inv)),ix[1:])
        self.weights = points['weight']
        self.strip = points['section'].astype(int)
        self.native_shapes = []
        self.xy = []
        self.mass = []
        self.bounds = []
        for cell,group in zip(self.cells,self.groups):
            sel = mesh['cell']==cell
            lo,hi = [np.array([np.unique(mesh[k][sel]).item() for k in axes])
                     for axes in [('x0','y0'),('x1','y1')]]
            width = hi-lo
            np.testing.assert_allclose(sum(self.weights[group]),np.prod(width),rtol=1e-13,atol=1e-17)
            xy = (np.stack([points[k][group] for k in ('x','y')],axis=1)-lo)/width
            if np.min(xy)<0 or np.max(xy)>1: raise ValueError(f'Point outside {cell}')
            self.bounds.append((lo,hi))
            self.xy.append(xy.T)
            self.native_shapes.append(shapes(*xy.T,width)[0])
            self.mass.extend([np.prod(width)/9]*9)
        self.mass=np.array(self.mass)
        self.H=self.fracture_matrix()

    def fracture_matrix(self):
        # Differentiate the mapped AT1 volume energy and the SIPG face
        # energy. Count right/top interior faces once, with outward normal.
        nodes,weights = np.polynomial.legendre.leggauss(4)
        nodes,weights=(nodes+1)/2,weights/2
        x,y=np.meshgrid(nodes,nodes)
        ids=np.arange(9*len(self.cells)).reshape(-1,9)
        starts={tuple(lo):i for i,(lo,hi) in enumerate(self.bounds)}
        rows,cols,entries=[],[],[]
        def add(idx,A):
            rows.extend(np.repeat(idx,len(idx)))
            cols.extend(np.tile(idx,len(idx)))
            entries.extend(A.ravel())
        k=2*KF*ELL**2
        for i,(lo,hi) in enumerate(self.bounds):
            d=hi-lo
            _,gx,gy=shapes(x.ravel(),y.ravel(),d)
            w=np.prod(d)*np.outer(weights,weights).ravel()
            add(ids[i],k*(gx.T@(w[:,None]*gx)+gy.T@(w[:,None]*gy)))
            for axis in (0,1):
                neighbor=lo.copy()
                neighbor[axis]=hi[axis]
                j=starts.get(tuple(neighbor))
                if j is None: continue
                dn=self.bounds[j][1]-self.bounds[j][0]
                def face(z,width):
                    side=np.full_like(nodes,z)
                    return shapes(side,nodes,width) if axis==0 else shapes(nodes,side,width)
                high,low=face(1.,d),face(0.,dn)
                jump=np.column_stack([high[0],-low[0]])
                avg=.5*k*np.column_stack([high[axis+1],low[axis+1]])
                measure=d[1-axis]
                penalty=90*k*measure/min(np.prod(d),np.prod(dn))
                add(np.r_[ids[i],ids[j]],measure*(jump.T@(weights[:,None]*(penalty*jump-avg))-avg.T@(weights[:,None]*jump)))
        return coo_matrix((entries,(rows,cols)),shape=(len(self.mass),len(self.mass))).tocsr()

    def phase(self,coeff):
        return np.concatenate([B@p for B,p in zip(self.native_shapes,coeff.reshape(-1,9))])

    def load(self,values):
        return np.concatenate([B.T@(self.weights[g]*values[g]) for B,g in zip(self.native_shapes,self.groups)])

    def fracture(self,b):
        return .5*b@(self.H@b)+KF*(self.mass@b)

    def velocity(self,mesh):
        # The boundary-only nodal lift tests the prescribed rows of the
        # momentum residual. Interior basis functions have zero coefficients.
        outer_lo=np.min([v[0] for v in self.bounds],axis=0)
        outer_hi=np.max([v[1] for v in self.bounds],axis=0)
        grads,lifts,pressure=[],[],[]
        nodes=np.array([(i%3,i//3) for i in ORDER])*.5
        for cell,xy,(lo,hi) in zip(self.cells,self.xy,self.bounds):
            group=mesh['cell']==cell
            def component(c):
                return mesh['value'][group & (mesh['component']==c)]
            v=np.stack([component(0),component(1)],axis=1)
            _,dx,dy=shapes(*xy,hi-lo,True)
            on_wall=np.any(((lo+nodes*(hi-lo))==outer_lo)|((lo+nodes*(hi-lo))==outer_hi),axis=1)
            for result,coeff in [(grads,v),(lifts,v*on_wall[:,None])]:
                result.append(np.stack([dx@coeff,dy@coeff],axis=1))
            x,y=xy
            pressure.append(np.column_stack([(1-x)*(1-y),x*(1-y),(1-x)*y,x*y])@component(2))
        return np.concatenate(grads),np.concatenate(lifts),np.concatenate(pressure)


def tensor(pts,stem):
    return np.stack([pts[stem+k] for k in ('xx','yy','zz','xy')],axis=1)


def dot(a,b):
    return np.sum(a*b*CONTRACT,axis=1)


def energy(p,A):
    phi=np.maximum(p,0.)
    val=np.zeros_like(phi)
    mobile=phi>0
    val[mobile]=G*((1-phi[mobile])**2/(M*phi[mobile]*(1+phi[mobile])))*dot(A,A)[mobile]
    return val


def aged(old,v,t):
    x=t*np.abs(v)/.01
    return old*np.exp(-x)+t*np.divide(-np.expm1(-x),x,out=np.ones_like(x),where=x!=0)


def resistance(v,theta):
    return 3*np.arcsinh(v/.002*np.exp((.6+.02*np.log(.1*theta))/.015))+1000*v


class Frame:
    def __init__(self,path,step):
        tag=f'{step:04}'
        self.mesh=shards(path,'mesh-'+tag,['cell','component','local'])
        self.point=shards(path,'points-'+tag,['cell','q'])
        self.strip=read([path/f'sections-{tag}.csv'],['section'])
        self.phase_component=field_names(path).index('phase_field')
        self.coeff=self.mesh['value'][self.mesh['component']==self.phase_component]
        self.book={k:v.item() for k,v in read([path/f'balance-{tag}.csv']).items()} if step else {}
        self.time=self.book.get('time',0.)


def donor_endpoint_drive(points,planes,rates):
    """Conditional endpoint drive in Pa, on the archived native point keys."""
    np.testing.assert_array_equal(planes['xi'],rates['xi'])
    intact=points['phi']<=0
    drive=points['Y_endpoint'].copy()
    strip=points['section'][intact].astype(int)
    a=np.pi/6+planes['beta'][strip]
    traction=(.5*(points['tauyy'][intact]-points['tauxx'][intact])*np.sin(2*a)
              + points['tauxy'][intact]*np.cos(2*a))
    reaction=resistance(rates['resolved_V'],planes['age'])
    drive[intact]=M/(2*G)*(traction-reaction[strip])**2
    return drive


def endpoint_rows(geom,coeff,gradient,drive):
    load=geom.load(drive)
    residual=gradient-load
    downhill=(coeff<1)&(residual<0)
    projected=np.clip(coeff-residual/(KF*geom.mass),coeff,1)-coeff
    return load,residual,downhill,projected>1e-7


def interval(old,new,geom,donor_geom,donor,step,clock,audit_rows,reference_mesh,previous_mesh,cmp,returned_rows=None):
    p0,p1=old.point,new.point
    aligned(p0,p1,['cell','q','section','x','y','weight'])
    aligned(old.mesh,new.mesh,['cell','component','local','x0','x1','y0','y1'])
    J,k=geom.weights,geom.strip
    dt=new.time-old.time
    np.testing.assert_allclose([old.time,(new.time+old.time)/2,new.time,dt],clock,atol=1e-15,rtol=0)
    h=dt/2
    sec=new.strip
    v,f,ds=sec['V'],sec['force'],sec['ds']
    phi0,phi1=[geom.phase(frame.coeff) for frame in (old,new)]
    for phi,p in [(phi0,p0),(phi1,p1)]:
        np.testing.assert_allclose(phi,p['phi'],atol=2e-15,rtol=0)
    phi=(phi1+phi0)/2
    inc=phi1-phi0
    tau0,tau1=tensor(p0,'tau'),tensor(p1,'tau')
    A0,A1=tensor(p0,'A'),tensor(p1,'A')
    tau,A=(tau1+tau0)/2,(A1+A0)/2
    a=np.pi/6+sec['beta'][k]
    S=np.column_stack([-.5*np.sin(2*a),.5*np.sin(2*a),np.zeros_like(a),.5*np.cos(2*a)])
    q=dot(tau,S)-f[k]
    rate=2*dot(A1-A0,S)/dt + (G/ETA)*2*dot(A,S)
    driving=M*(1+3*phi)/(1-phi)**3*q*q/(2*G)
    grad,lift,pressure=geom.velocity(new.mesh)
    def strain(g):
        tr=g[:,0,0]+g[:,1,1]
        return np.stack([g[:,0,0]-tr/3,g[:,1,1]-tr/3,-tr/3,(g[:,0,1]+g[:,1,0])/2],axis=1),tr
    D,div=strain(grad)
    lifted,div_lift=strain(lift)
    check=lambda x,y,tol: np.testing.assert_allclose(x,y,atol=tol,rtol=0,equal_nan=False)
    # Verify h in the actual returned Maxwell/engineering-shear update.
    B=1/(1+h*G/ETA)
    check(tau,B*(tau0+2*G*h*(D-S*rate[:,None])),1e-9)
    check(p1['signed_shear']-p0['signed_shear'],dt*rate,1e-14)
    check(p1['absolute_shear']-p0['absolute_shear'],dt*abs(rate),1e-14)
    check(np.bincount(k,J*rate,minlength=len(ds))/ds,v,1e-13)
    check(resistance(v,aged(old.strip['age'],v,h)),f,1e-10)
    check(sec['age'],aged(old.strip['age'],v,dt),1e-13)
    check(sec['slip']-old.strip['slip'],dt*v,1e-15)
    wall=dt*(J@(dot(tau,lifted)-pressure*div_lift))
    mech=dt*(J@(dot(tau,D)-pressure*div))-wall
    elastic=J@dot(tau1-tau0,tau1+tau0)/(4*G)
    dcoh=J@(energy(phi1,A1)-energy(phi0,A0))
    Db=dt*(J@dot(tau,tau))/(2*ETA)
    Dc=dt*(2*G/ETA)*(J@energy(phi,A))
    Df=dt*(J@(rate*f[k]))
    check(Df,dt*np.dot(ds*v,f),1e-12)
    deltafract=geom.fracture(new.coeff)-geom.fracture(old.coeff)
    physical=wall-elastic-dcoh-deltafract-Db-Dc-Df
    rem=J@(2*q*dot(A1-A0,S)-driving*inc)-dcoh
    stage=.5*(new.coeff+old.coeff)
    lower=old.coeff
    upper=(1+lower)/2
    fg=geom.H@stage+KF*geom.mass
    load=geom.load(driving)
    residual=fg-load
    pred=stage-residual/(KF*geom.mass)
    active=np.where(pred<=lower,-1,np.where(pred>=upper,1,0))
    KKT=max(abs(stage-np.clip(pred,lower,upper)))
    phase_work=(new.coeff-old.coeff)@residual
    endfg=geom.H@new.coeff+KF*geom.mass
    # Conditional driving at fixed endpoint stress and recoverable strain.
    q1=dot(tau1,S)-resistance(v,sec['age'])[k]
    drive1=M*q1*q1/(2*G)
    positive=phi1>0
    p=phi1[positive]
    drive1[positive]=G*dot(A1,A1)[positive]*(1+3*p)*(1-p)/(M*(p*(1+p))**2)
    endload,endres,downhill,resolved=endpoint_rows(geom,new.coeff,endfg,drive1)
    check(new.book['phase_KKT'],KKT,2e-12)
    for name,value in [('Wext',wall),('Db',Db),('Dc',Dc),('Df',Df),('fracture',geom.fracture(new.coeff)),
                       ('phase_remainder',rem),('phase_work',phase_work),('bulk',J@dot(tau1,tau1)/(4*G)),('cohesive',J@energy(phi1,A1))]:
        check(new.book[name],value,1e-11)
    check(physical,rem-phase_work-mech,1e-11)
    native=shards(donor,f'tip-points-{step:04}',['cell','q'])
    ids=aligned(native,p1,['cell','q'])
    for key,value,tol in [('stage_phi',phi,1e-9),('stage_Axy',A[:,3],1e-10),('stage_tauxy',tau[:,3],1e-6),
                          ('stage_pressure',pressure,1e-6),('stage_normal_x',tau[:,0]-pressure,1e-6),
                          ('stage_normal_y',tau[:,1]-pressure,1e-6)]:
        cmp.check('stage/'+key,native[key],value,tol,1e-7,ids)
    planes=read([donor/f'tip-planes-{step:04}.csv'],['section'])
    for key,tol in [('formed',0),('slip',1e-11),('age',1e-11)]:
        cmp.check('parent_strip/'+key,planes['parent_'+key],old.strip[key],tol,1e-7 if tol else 0.,[sec['section']])
    # The archive directly observed two stage row vectors. At the other
    # clocks use the donor's own FE coefficients and native driving force.
    donor_phase=[]
    for path,n in [(previous_mesh,step-1),(reference_mesh,step)]:
        m=shards(path,f'mesh-{n:04}',['cell','component','local'])
        aligned(m,new.mesh,['cell','component','local','x0','x1','y0','y1'])
        donor_phase.append(m['value'][m['component']==field_names(path).index('phase_field')])
    dob,deb=donor_phase
    dsb=(dob+deb)/2
    donor_res=geom.H@dsb+KF*geom.mass-geom.load(native['Y'])
    d_pred=dsb-donor_res/(KF*geom.mass)
    d_status=np.where(d_pred<=dob,-1,np.where(d_pred>=(1+dob)/2,1,0))
    row_ids=[new.mesh[k][new.mesh['component']==new.phase_component] for k in ('cell','local')]
    rates=read([donor/f'tip-sections-{step:04}.csv'],['xi'])
    donor_load=donor_geom.load(donor_endpoint_drive(native,planes,rates))
    donor_gradient=donor_geom.H@deb+KF*donor_geom.mass
    cmp.check('reconstructed_row/residual',donor_res,residual,1e-8,1e-7,row_ids)
    cmp.check('reconstructed_row/classification',d_status,active,0,ids=row_ids)
    if returned_rows:
        live=shards(returned_rows,f'rows-{step:04}',['cell','i'])
        for values,name in zip(row_ids,['cell','i']):
            np.testing.assert_array_equal(values,live[name])
        # The instrumented run records these values after the two coupled
        # convergence gates, directly before the returned FE fields are written.
        for col,value,tol in [('lower',lower,2e-15),('upper',upper,2e-15),('stage',stage,2e-15),
                              ('mass',geom.mass,1e-15),('h',h,1e-16),('B',B,1e-15),
                              ('residual',residual,1e-11)]:
            check(live[col],value,tol)
        actual_pred=live['stage']-live['residual']/(KF*live['mass'])
        actual_status=np.where(actual_pred<=live['lower'],-1,np.where(actual_pred>=live['upper'],1,0))
        np.testing.assert_array_equal(actual_status,active)
        cmp.check('assembled_row/residual',donor_res,live['residual'],1e-8,1e-7,row_ids)
        cmp.check('assembled_row/classification',d_status,actual_status,0,ids=row_ids)
    if audit_rows:
        donor_rows=shards(audit_rows,f'tip-balance-{step:04}',['cell','i'])
        selected=new.mesh['component']==new.phase_component
        ids=[new.mesh['cell'][selected],new.mesh['local'][selected]]
        for x,y in zip(ids,[donor_rows['cell'],donor_rows['i']]): np.testing.assert_array_equal(x,y)
        da=donor_rows
        donor_gradient=da['fracture_gradient']
        d_pred=da['stage_B']-da['residual']/(KF*da['mass'])
        classification=np.where(d_pred<=da['lower'],-1,np.where(d_pred>=(1+da['lower'])/2,1,0))
        for key,out,tol in [('lower',lower,1e-9),('stage_B',stage,1e-9),('B',new.coeff,1e-9),
                            ('mass',geom.mass,1e-15),('residual',residual,1e-8),('driving_load',load,1e-8),
                            ('stage_fracture_gradient',fg,1e-8),('fracture_gradient',endfg,1e-8)]:
            cmp.check('row/'+key,da[key],out,tol,1e-7,ids)
        cmp.check('row/classification',classification,active,0,ids=ids)
    donor_endres=donor_gradient-donor_load
    donor_downhill=(deb<1)&(donor_endres<0)
    # Feasible additional rise at the *accepted* donor endpoint; its
    # obstacle has already advanced. The stage has its own box and residual.
    donor_resolved=np.minimum(1-deb,-donor_endres/(KF*donor_geom.mass))>1e-7
    for name,reference,measured in [('load',donor_load,endload),('residual',donor_endres,endres)]:
        cmp.check('endpoint/'+name,reference,measured,1e-8,1e-7,row_ids)
    for name,reference,measured in [('strict_downhill',donor_downhill,downhill),
                                    ('projected_above_tolerance',donor_resolved,resolved)]:
        cmp.check('endpoint/'+name,reference.astype(int),measured.astype(int),0,ids=row_ids)
    return dict(time_s=new.time,duration_s=dt,stage_s=new.time-h,
                lower=int(sum(active==-1)),free=int(sum(active==0)),upper=int(sum(active==1)),
                downhill=int(sum(downhill)),resolved_downhill=int(sum(resolved)),
                min_abs_endpoint_residual_J_m=float(min(abs(endres))),max_endpoint_load_J_m=float(max(endload)),
                kkt=KKT,mechanics=new.book['mechanical_residual'],
                registrations=int(sum(sec['formed']-old.strip['formed'])),
                signed_slip_m2=ds@(sec['slip']-old.strip['slip']),phase_increment_L2_m=np.sqrt(J@inc**2),
                phase_increment_sample_max=np.max(inc),fracture_change_J_m=deltafract,
                work_residual_J_m=physical,midpoint_remainder_J_m=rem,phase_work_J_m=phase_work,mechanical_work_J_m=mech,
                external=wall,elastic_change=elastic,cohesive_change=dcoh,Db=Db,Dc=Dc,Df=Df)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',type=Path)
    p.add_argument('reference',type=Path)
    p.add_argument('destination',type=Path)
    p.add_argument('--returned-rows',type=Path,help='Separate matching trajectories from the scratch returned-row recorder')
    args=p.parse_args(argv)
    dest=args.destination
    dest.mkdir(exist_ok=False)
    r=args.reference/'runs'
    d=args.runs
    expected=json.loads((args.reference/'expected/timestep.json').read_text())
    diagnosis=json.loads((args.reference/'expected/diagnosis.json').read_text())
    endpoint_reference=[diagnosis['endpoint_balance'][key] for key in ('parent','coarse','first_half')]
    endpoint_reference.append(diagnosis['frozen']['refined_final_balance'])
    cases=[('parent',1,'localized-oneo1',None),('coarse',2,'localized-oneo2','strip18-audit-full'),
           ('half',2,'strip18-midpoint-half','strip18-audit-half1'),('half',3,'strip18-midpoint-half',None)]
    initial=Frame(d/'parent/output',0)
    geom=Geometry(initial.mesh,initial.point)
    replay=d/'donor-parent/output'
    donor_geom=Geometry(shards(replay,'mesh-0000',['cell','component','local']),
                        shards(replay,'tip-points-0000',['cell','q']))
    np.testing.assert_array_equal(geom.cells,donor_geom.cells)
    parent=Frame(d/'parent/output',1)
    first=Frame(d/'half/output',2)
    # Both schedules must branch from exactly the same published state.
    for n,count in [(0,7),(1,8)]:
        files=sorted((d/'parent/output').glob(f'*-{n:04}*.csv'))
        if len(files)!=count or not all(identical(f,d/'half/output'/f.name,shallow=False) for f in files):
            raise ValueError(f'Fine trajectory has a different parent frame at step {n}')
    for case,numbers in [('parent',[0,1]),('coarse',[2]),('half',[0,1,2,3])]:
        completion=np.loadtxt(d/case/'output/completions',ndmin=2)
        np.testing.assert_array_equal(completion[:,0],numbers)
        np.testing.assert_array_equal(completion[:,1],np.maximum(0,np.array(numbers)-1))
        np.testing.assert_array_equal(completion[:,3],np.ones(len(numbers)))
        log=(d/case/'log').read_text()
        if 'Repeating the current time step' in log or log.count('native section and point history')!=len(numbers):
            raise ValueError(f'{case}: unexpected repeat or publication count')
    result={}
    failed=0
    seed=Comparison()
    compare(0,d/'parent/output',r/'localized-oneo1/output',d/'donor-parent/output',seed)
    failed+=seed.write(dest/'seed.csv')
    for (case,step,ref,obs),conditional in zip(cases,endpoint_reference,strict=True):
        ids=Comparison()
        donor=r/ref/'output'
        compare(step,d/case/'output',donor,d/('donor-'+case)/'output',ids)
        old=initial if step==1 else parent if step==2 else first
        new=Frame(d/case/'output',step)
        captured=args.returned_rows/case/'output' if args.returned_rows else None
        if captured:
            for f in (d/case/'output').glob(f'*-{step:04}*.csv'):
                if not identical(f,captured/f.name,shallow=False):
                    raise ValueError(f'Row recorder changed physical output {f}')
        clock=read([donor/'tip-time-location.csv'])
        idx=clock['step']==step
        clocks=[clock[k][idx].item() for k in ['parent_time','stage_time','endpoint_time']]
        # The attempted duration is printed by ASPECT before nonlinear assembly.
        log=(d/case/'log').read_text()
        match=re.search(rf'\*\*\* Timestep {step}:\s+t=([\d.eE+-]+) seconds, dt=([\d.eE+-]+) seconds',log)
        if not match: raise ValueError('Missing attempt clock')
        np.testing.assert_allclose(float(match[1]),clocks[2],atol=1e-15,rtol=0)
        actual=interval(old,new,geom,donor_geom,donor,step,clocks+[float(match[2])],r/obs/'output' if obs else None,
                        d/('donor-'+case)/'output',d/('donor-half' if step==3 else 'donor-parent')/'output',ids,captured)
        for key,observed,tol in [('max_load','max_endpoint_load_J_m',1e-8),('downhill','downhill',0),
                                 ('above_stage_tolerance','resolved_downhill',0)]:
            ids.check('archived_endpoint/'+key,conditional[key],actual[observed],tol,1e-7 if tol else 0.)
        completion=np.loadtxt(d/case/'output/completions',ndmin=2)
        completed=completion[completion[:,0]==step,2].item()
        solver=read([d/('donor-'+case)/'output/tip-solver.csv'],['step'])
        actual['donor_iterations']=solver['nonlinear_iterations'][solver['step']==step].item()
        actual['target_iterations']=completed
        ids.check('solver/target_balance_vs_completion',completed,new.book['iterations'],0)
        if step==1:
            donor_points=shards(donor,'tip-points-0001',['cell','q'])
            seed_points=shards(d/'donor-parent/output','tip-points-0000',['cell','q'])
            aligned(seed_points,donor_points,['cell','q'])
            planes=read([donor/'tip-planes-0001.csv'],['section'])
            initial_fracture=read([d/'donor-parent/output/tip-initial.csv'])['fracture'].item()
            ledger={k:v.item() for k,v in read([donor/'bulk-accepted.csv']).items()}
            inc=donor_points['phi']-seed_points['phi']
            e=dict(signed_slip_m2=planes['ds']@(planes['slip']-planes['parent_slip']),
                   phase_increment_L2_m=np.sqrt(donor_points['weight']@inc**2),
                   fracture_change_J_m=ledger['fracture']-initial_fracture,
                   work_residual_J_m=ledger['Wext']-ledger['energy_change']-ledger['Db']-ledger['Dc']-ledger['Df'],
                   midpoint_remainder_J_m=ledger['phase_remainder'],phase_work_J_m=ledger['phase_work'],
                   mechanical_work_J_m=ledger['mechanical_work'],
                   registrations=sum(planes['formed']-planes['parent_formed']))
        else:
            e=expected['half' if case=='half' else 'reference']['steps'][step-2]
        for k in ('signed_slip_m2','phase_increment_L2_m','fracture_change_J_m','work_residual_J_m','midpoint_remainder_J_m','phase_work_J_m','mechanical_work_J_m','registrations'):
            tol={'signed_slip_m2':1e-12,'phase_increment_L2_m':1e-10,'registrations':0}.get(k,1e-11)
            ids.check('interval/'+k,e[k],actual[k],tol,1e-5 if tol else 0.)
        result[f'{case}-{step}']=actual
        failed+=ids.write(dest/f'{case}-{step}.csv')
    endpoint={}
    for case,steps in [('coarse',[2]),('half',[2,3])]:
        target=Frame(d/case/'output',steps[-1])
        diff=target.point['phi']-parent.point['phi']
        endpoint[case]=diff
        entries=[result[f'{case}-{i}'] for i in steps]
        summary=dict(signed_slip_m2=sum(x['signed_slip_m2'] for x in entries),
                     added_fracture_J_m=sum(x['fracture_change_J_m'] for x in entries),
                     phase_increment_L2_m=np.sqrt(geom.weights@diff**2),phase_increment_sample_max=float(diff.max()),
                     signed_residual_J_m=sum(x['work_residual_J_m'] for x in entries),
                     absolute_residual_J_m=sum(abs(x['work_residual_J_m']) for x in entries),
                     signed_midpoint_remainder_J_m=sum(x['midpoint_remainder_J_m'] for x in entries),
                     absolute_midpoint_remainder_J_m=sum(abs(x['midpoint_remainder_J_m']) for x in entries))
        result[case]=summary
        ids=Comparison()
        for k,value in summary.items():
            tol={'signed_slip_m2':1e-12,'phase_increment_L2_m':1e-10,'phase_increment_sample_max':1e-9}.get(k,1e-11)
            ids.check(k,expected['reference' if case=='coarse' else 'half'][k],value,tol,1e-5)
        failed+=ids.write(dest/f'{case}-window.csv')
    diff=endpoint['coarse']-endpoint['half']
    ids=Comparison()
    for k,v,tol in [('phase_increment_field_difference_L2_m',np.sqrt(geom.weights@diff**2),1e-10),
                    ('phase_increment_field_difference_sample_max',max(abs(diff)),1e-9)]:
        ids.check(k,expected[k],v,tol,1e-5)
        result[k]=v
    failed+=ids.write(dest/'separation.csv')
    (dest/'intervals.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f"{failed} out-of-budget comparisons; all scheduled intervals audited.")
    return bool(failed)


if __name__=='__main__':
    raise SystemExit(main())
