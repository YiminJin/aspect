#!/usr/bin/env python3
"""Mesh-independent norms of the actual native mixed matrices and slip fields."""
from pathlib import Path
import argparse,collections,json,struct,time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from numpy.polynomial.legendre import leggauss
from coulomb_localization_analysis import ROOT,WORK,OUT,read,write,save,columns,rate_columns,symbol,minimum

def basis(run):
    data=collections.defaultdict(list)
    for r in read(run/'localization-cell-dofs.csv'):
        data[r['cell']].append((int(r['component']),float(r['u']),float(r['v']),int(r['dof'])))
    cells={r['cell']:r for r in read(run/'means-0000.csv')}
    return data,cells
def L(x,location,degree):
    if degree==1:return ((1-x,-np.ones_like(x)) if location==0 else (x,np.ones_like(x)))
    if location==0:return (1-3*x+2*x*x,-3+4*x)
    if location==.5:return (4*x*(1-x),4-8*x)
    return (2*x*x-x,4*x-1)
def shapes(entries,x,y,h):
    B=np.zeros((len(x),len(entries),3));P=np.zeros((len(x),len(entries)))
    for i,(c,u,v,_) in enumerate(entries):
        lx,dx=L(x,u,1 if c==2 else 2);ly,dy=L(y,v,1 if c==2 else 2)
        if c==2:P[:,i]=lx*ly
        elif c==0:B[:,i,0]=dx*ly/h;B[:,i,2]=lx*dy/h
        else:B[:,i,1]=lx*dy/h;B[:,i,2]=dx*ly/h
    B4=np.stack([B[:,:,0]-(B[:,:,0]+B[:,:,1])/3,B[:,:,1]-(B[:,:,0]+B[:,:,1])/3,
                 -(B[:,:,0]+B[:,:,1])/3,B[:,:,2]],axis=2)
    return B,B4,P
def matrix_file(path):
    raw=path.read_bytes();n,count=struct.unpack_from('<QQ',raw)
    dtype=np.dtype([('i','<u4'),('j','<u4'),('v','<f8')]);a=np.frombuffer(raw,offset=16,dtype=dtype)
    assert len(a)==count
    return sp.coo_matrix((a['v'],(a['i'],a['j'])),shape=(n,n)).tocsr()
def assemble(run,stem,check=False):
    entries,cells=basis(run);coords=read(run/(stem+'-coordinates.csv'));free=[int(r['dof']) for r in coords]
    size=max(v[3] for rows in entries.values() for v in rows)+1
    beta=json.loads((run/(stem+'.json')).read_text())['beta'];rows=[];cols=[];values=[];mass=np.zeros(size)
    actual_rows=[];actual_cols=[];actual_values=[]
    point_data=collections.defaultdict(list)
    if check:
        for path in run.glob(stem+'-rank-*.csv'):
            for r in read(path):point_data[r['cell']].append(r)
    q,w=leggauss(3);q=(q+1)/2;w=w/2;qx,qy=np.meshgrid(q,q,indexing='xy');wx,wy=np.meshgrid(w,w,indexing='xy')
    for cell,local in entries.items():
        h=np.sqrt(float(cells[cell]['area']));B,B4,P=shapes(local,qx.ravel(),qy.ravel(),h)
        D4=B4.copy();D4[:,:,3]/=2;weights=(wx*wy).ravel()*h*h
        K=2*beta*np.einsum('qia,qja,q->ij',B4,D4,weights)+np.einsum('qi,qj,q->ij',P,P,weights)/beta
        ids=np.array([v[3] for v in local]);ii,jj=np.meshgrid(ids,ids,indexing='ij')
        rows.extend(ii.ravel());cols.extend(jj.ravel());values.extend(K.ravel());mass[ids]+=weights@P
        if check:
            points=sorted(point_data[cell],key=lambda r:int(r['q']));x=np.array([float(r['x']) for r in points]);y=np.array([float(r['y']) for r in points])
            cx=float(cells[cell]['x']);cy=float(cells[cell]['y']);B,B4,P=shapes(local,(x-cx+h/2)/h,(y-cy+h/2)/h,h)
            weights=np.array([float(r['JxW']) for r in points]);C=np.array([columns(r) for r in points]);div=B[:,:,0]+B[:,:,1]
            J=np.einsum('qia,qab,qjb,q->ij',B4,C[:,:4,:3],B,weights)
            J+=np.einsum('qia,qa,qj,q->ij',B4,C[:,:4,3],P,weights)
            J-=np.einsum('qi,qj,q->ij',div,P,weights)+np.einsum('qi,qj,q->ij',P,div,weights)
            actual_rows.extend(ii.ravel());actual_cols.extend(jj.ravel());actual_values.extend(J.ravel())
    reference=sp.coo_matrix((values,(rows,cols)),shape=(size,size)).tocsr()
    pressure=sorted({v[3] for rows in entries.values() for v in rows if v[0]==2});p0=pressure[0]
    cr=list(free);cc=list(range(len(free)));cv=[1.]*len(free)
    for j,i in enumerate(free):
        if i in pressure:cr.append(p0);cc.append(j);cv.append(-mass[i]/mass[p0])
    C=sp.coo_matrix((cv,(cr,cc)),shape=(size,len(free))).tocsr();R=(C.T@reference@C).toarray()
    if check:
        J=sp.coo_matrix((actual_values,(actual_rows,actual_cols)),shape=(size,size)).tocsr();J=C.T@J@C;native=matrix_file(run/(stem+'.bin'))
        defect=la.norm((J-native).toarray())/max(1.,la.norm(native.toarray()));assert defect<=1e-10,defect
        return R,defect
    return R,None
def conditioning(case,stem):
    run=WORK/'runs'/case;start=time.monotonic();R,_=assemble(run,stem);J=matrix_file(run/(stem+'.bin')).toarray()
    L=la.cholesky(R,lower=True);A=la.solve_triangular(L,J,lower=True);A=la.solve_triangular(L,A.T,lower=True).T
    s=la.svdvals(A,check_finite=False);record={'case':case,'stem':stem,'unknowns':len(s),'Riesz_sigma_min':float(s[-1]),'Riesz_sigma_max':float(s[0]),'Riesz_condition_2':float(s[0]/s[-1]),'seconds':time.monotonic()-start}
    print(json.dumps(record),flush=True);return record
def batch_min(C):
    theta=np.linspace(0,np.pi,721);n=np.stack([np.cos(theta),np.sin(theta)],axis=-1);a=np.stack([-np.sin(theta),np.cos(theta)],axis=-1)
    E=np.stack([a[:,0]*n[:,0],a[:,1]*n[:,1],a[:,0]*n[:,1]+a[:,1]*n[:,0]],axis=-1)
    s=np.einsum('baj,tj->bta',C[:,:4,:3],E)
    v=np.stack([s[:,:,0]*n[:,0]+s[:,:,3]*n[:,1],s[:,:,3]*n[:,0]+s[:,:,1]*n[:,1]],axis=-1)
    b=np.stack([(C[:,0,3,None]-1)*n[:,0]+C[:,3,3,None]*n[:,1],C[:,3,3,None]*n[:,0]+(C[:,1,3,None]-1)*n[:,1]],axis=-1)
    aa=np.einsum('ti,bti->bt',a,v);na=np.einsum('ti,bti->bt',n,v);ab=np.einsum('ti,bti->bt',a,b);nb=np.einsum('ti,bti->bt',n,b)
    return np.min(aa-ab*na/nb,axis=1),np.min(aa,axis=1)
def profiles(name,time,data):
    w=np.array([float(r['JxW_m2']) for r in data]);rate=2*np.hypot([float(r['Wxx_s-1']) for r in data],[float(r['Wxy_s-1']) for r in data])
    theta=2.913739130895119;position=np.array([[float(r['x_m']),float(r['y_m'])] for r in data]);s=position@np.array([np.cos(theta),np.sin(theta)])
    radius=.046875/2;result=[]
    for center in np.linspace(-.25,.25,151):
        kernel=np.exp(-.5*((s-center)/radius)**2);weight=w@kernel
        result.append({'case':name,'time_s':time,'normal_coordinate_m':center,'kernel_sigma_m':radius,
            'wave_normal_rad':theta,'kernel_mass_m2':float(weight),'slip_rate_s-1':float((w*kernel)@rate/weight) if weight>1e-12 else None})
    return result

def only_profiles():
    result=[]
    for run in sorted((WORK/'runs').iterdir()):
        if not (run/'bulk-accepted.csv').exists():continue
        for row in read(run/'bulk-accepted.csv'):
            time=float(row['time']);step=int(row['step'])
            if abs(time-.1125)<1e-12 or abs(time-.125)<1e-12:
                result+=profiles(run.name,time,read(run/f'coulomb-{step:04}.csv'))
    write('normal-profiles.csv',result)

def summarize():
    summary=[];profile_rows=[];case_results={}
    for run in sorted((WORK/'runs').iterdir()):
        if not run.is_dir():continue
        name=run.name;ledger=read(run/'bulk-accepted.csv') if (run/'bulk-accepted.csv').exists() else []
        initial=read(run/'phase-0000.csv');phi0=float(initial[0]['phi']);n=int(name.split('-n')[1].split('-')[0]);dt=float(ledger[0]['dt']) if ledger else (.00625 if name.endswith('half') else .0125)
        log=ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence'/('coulomb-localization-'+name+'.json')
        case_results[name]={'exit_code':json.loads(log.read_text())['exit_code'],'accepted_steps':len(ledger),'last_time_s':float(ledger[-1]['time']) if ledger else 0.,'h_m':.25/n,'ell_m':.046875}
        for file in sorted(run.glob('coulomb-[0-9][0-9][0-9][0-9].csv')):
            step=int(file.stem[-4:]);physical_time=float(next(r['time'] for r in ledger if int(r['step'])==step)) if step else 0.;data=read(file);w=np.array([float(r['JxW_m2']) for r in data]);area=w.sum()
            W=np.array([[float(r['Wxx_s-1']),float(r['Wxy_s-1'])] for r in data]);rate=2*la.norm(W,axis=1);mean=w@W/area;fluct=2*la.norm(W-mean,axis=1)
            width=lambda z:float((w@z)**2/(w@(z*z))/.25) if w@(z*z)>0 else None
            phase=read(run/f'phase-{step:04}.csv');growth=max(abs(float(r['phi'])-phi0) for r in phase);assert growth==0.
            force=read(run/f'phase-forces-{step:04}.csv') if step else []
            row={'case':name,'step':step,'time_s':physical_time,'dt_s':dt,'h_m':.25/n,'ell_m':.046875,'phi':phi0,'phase_increment_max':growth,
                 'mean_slip_rate_s-1':float(w@rate/area),'peak_slip_rate_s-1':float(max(rate)),'slip_participation_width_m':width(rate),
                 'fluctuation_participation_width_m':width(fluct),'phase_obstacle_force_min_Pa':min(float(r['residual'])/float(r['mass']) for r in force) if force else 8.}
            if step:
                entry=next(r for r in ledger if int(r['step'])==step)
                row['mechanical_residual']=float(entry['mechanical_residual']);row['source_error']=float(entry['projection_error']);row['ledger_error_J_m']=float(entry['ledger_error'])
                assert row['mechanical_residual']<=1e-9 and row['source_error']<=1e-12
            else:row.update(mechanical_residual=0.,source_error=0.,ledger_error_J_m=0.)
            stem=f'localization-{step:04}-converged';points=[]
            if (run/(stem+'.bin')).exists():
                for p in sorted(run.glob(stem+'-rank-*.csv')):points+=read(p)
                minima=[];fixed=[];rate_min=[]
                for i in range(0,len(points),64):
                    C=np.array([columns(r) for r in points[i:i+64]]);a,b=batch_min(C);minima.extend(a);fixed.extend(b)
                    Cr=np.array([rate_columns(r) for r in points[i:i+64]]);a,_=batch_min(Cr);rate_min.extend(a)
                weights=np.array([float(r['JxW']) for r in points]);minima=np.array(minima);rate_min=np.array(rate_min)
                row.update(mixed_acoustic_min_over_G=float(min(minima)),fixed_pressure_min_over_G=float(min(fixed)),
                           negative_algorithmic_area_fraction=float(weights[minima<-1e-8].sum()/area),negative_rate_area_fraction=float(weights[rate_min<-1e-8].sum()/area))
            else:row.update(mixed_acoustic_min_over_G=None,fixed_pressure_min_over_G=None,negative_algorithmic_area_fraction=None,negative_rate_area_fraction=None)
            summary.append(row)
            if abs(physical_time-.1125)<1e-12 or abs(physical_time-.125)<1e-12:
                profile_rows+=profiles(name,physical_time,data)
        # Existing native gate coverage remains active; no certificate is relaxed.
        if (run/'gates.csv').exists():assert all(r['passed']=='1' for r in read(run/'gates.csv'))
    write('native-histories.csv',summary);write('normal-profiles.csv',profile_rows);save('native-cases.json',case_results)
    print(json.dumps(case_results,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--conditioning',nargs=2);p.add_argument('--assembly-check',nargs=2);p.add_argument('--profiles-only',action='store_true');args=p.parse_args()
    if args.profiles_only:only_profiles()
    elif args.conditioning:
        result=conditioning(*args.conditioning);save('-'.join(args.conditioning)+'-conditioning.json',result)
    elif args.assembly_check:
        _,error=assemble(WORK/'runs'/args.assembly_check[0],args.assembly_check[1],True);save('independent-assembly.json',{'scaled_matrix_defect':error,'bound':1e-10});print(error)
    else:summarize()
