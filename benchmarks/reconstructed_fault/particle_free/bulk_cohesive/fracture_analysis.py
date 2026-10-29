#!/usr/bin/env python3
"""Compact physical-coordinate data from existing native bulk exports only."""
import argparse,json,csv as csv_module
from pathlib import Path
import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from fracture_strip import seed_table,compliance

ROOT=Path(__file__).resolve().parents[4]
BUILD=ROOT/'build-particle-free/bulk-fracture'
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/bulk_fracture'
OLD=ROOT/'build-particle-free/bulk-parallel/runs/mixed'
NEW=BUILD/'runs/candidate'
LEVELS=[.001,.01,.05,.1,.3,.5]
G,ETA,R=1e6,1e8,1000.

def read(path):
    return np.atleast_1d(np.genfromtxt(path,delimiter=',',names=True,dtype=None,encoding='utf8'))

def save_csv(path,header,rows):
    with path.open('w') as stream:
        writer=csv_module.writer(stream);writer.writerow(header);writer.writerows(rows)

def grid_value(x,y,p,xx,yy):
    i=np.clip(np.searchsorted(x,xx,side='right')-1,0,len(x)-2)
    j=np.clip(np.searchsorted(y,yy,side='right')-1,0,len(y)-2)
    u=(xx-x[i])/(x[1]-x[0]);v=(yy-y[j])/(y[1]-y[0])
    return (1-u)*(1-v)*p[j,i]+u*(1-v)*p[j,i+1]+(1-u)*v*p[j+1,i]+u*v*p[j+1,i+1]

def fracture(p,dx,dy,Kf,ell):
    a,b,c,d=p[:-1,:-1],p[:-1,1:],p[1:,:-1],p[1:,1:]
    mean=(a+b+c+d)/4
    grad=((b-a)**2+(b-a)*(d-c)+(d-c)**2)/(3*dx*dx)+((c-a)**2+(c-a)*(d-b)+(d-b)**2)/(3*dy*dy)
    return Kf*dx*dy*np.sum(mean+ell*ell*grad)

def crossing(x,p,level):
    below=np.flatnonzero(p<level)
    if not len(below):return float(x[-1])
    i=below[0]
    if i==0:return float('nan')
    return float(x[i-1]+(x[i]-x[i-1])*(p[i-1]-level)/(p[i-1]-p[i]))

def initial_error(x,y,p,ell,tip):
    # Fresh continuum, evaluated independently; no accepted-history remapping.
    rr,pp=seed_table(ell,n=262145);z,w=np.polynomial.legendre.leggauss(20)
    num=den=capnum=capden=0.;dx=x[1]-x[0];dy=y[1]-y[0]
    u=(z+1)/2;weight=w[:,None]*w[None,:]*dx*dy/4
    for j in range(len(y)-1):
        X=x[:-1,None,None]+dx*u[None,:,None];Y=y[j]+dy*u[None,None,:]
        exact=np.interp(np.hypot(np.maximum(X-tip,0),Y-.5),rr,pp,right=0.)
        represented=(1-u[None,:,None])*(1-u[None,None,:])*p[j,:-1,None,None]+u[None,:,None]*(1-u[None,None,:])*p[j,1:,None,None]+(1-u[None,:,None])*u[None,None,:]*p[j+1,:-1,None,None]+u[None,:,None]*u[None,None,:]*p[j+1,1:,None,None]
        cap=(X>=tip)&(np.hypot(X-tip,Y-.5)<=2.25*ell)
        error=weight*(represented-exact)**2;mass=weight*exact**2
        num+=error.sum();den+=mass.sum();capnum+=error[cap].sum();capden+=mass[cap].sum()
    return dict(relative_L2=float(np.sqrt(num/den)),cap_relative_L2=float(np.sqrt(capnum/capden)))

def load_case(name):
    if name=='released':
        r=OLD;ell=.15625;Gc=26.6666666667;tip=.5;m=128.00000000016
        d=np.load(ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/bulk_parallel/movie-data.npz')
        x,y=d['x'],d['y'];gridids=d['grid_ids'];phase=d['phase'][:,gridids]
        times=d['time'];stress=d['stress'];driving=d['Y'];rate=d['rate']
    else:
        r=NEW;ell=.046875;Gc=1.;tip=1.;m=4.
        resources=read(r/'resources.csv');steps=resources['step'].astype(int)
        # The resources row is written after every accepted export completes.
        assert np.array_equal(steps,np.arange(len(steps)))
        nodes=np.loadtxt(r/'phase-0000.csv',delimiter=',',skiprows=1);nodes=nodes[np.argsort(nodes[:,0])]
        x,y=np.unique(nodes[:,1]),np.unique(nodes[:,2]);gridids=np.lexsort((nodes[:,1],nodes[:,2])).reshape(len(y),len(x))
        phase=[];stress=[];driving=[];rate=[]
        for n in steps:
            p=np.loadtxt(r/f'phase-{n:04d}.csv',delimiter=',',skiprows=1);p=p[np.argsort(p[:,0])]
            phase.append(p[gridids,3])
            a=np.loadtxt(r/f'means-{n:04d}.csv',delimiter=',',skiprows=1,usecols=(1,2,4,5,6))
            ids=np.lexsort((a[:,0],a[:,1]));a=a[ids]
            for target,k in [(stress,2),(driving,3),(rate,4)]:target.append(a[:,k].reshape(len(y)-1,len(x)-1))
        phase=np.array(phase);stress=np.array(stress);driving=np.array(driving);rate=np.array(rate)
        times=resources['time']
    ledger=read(r/'bulk-accepted.csv');ledger=ledger[ledger['step']<len(times)]
    assert np.array_equal(ledger['step'],np.arange(1,len(times)))
    assert np.array_equal(ledger['publications'],ledger['step'])
    assert max(ledger['mechanical_residual'])<=1e-9 and max(ledger['phase_KKT'])<=1e-7 and max(ledger['projection_error'])<=1e-12
    assert np.min(np.diff(phase,axis=0))>=0 and np.min(phase)>=0 and np.max(phase)<1
    assert np.all(read(r/'gates.csv')['passed']==1)
    return dict(name=name,run=r,x=x,y=y,phase=phase,times=times,stress=stress,Y=driving,rate=rate,
                ell=ell,Gc=Gc,tip=tip,m=m,Kf=Gc/((8/3)*ell),ledger=ledger,gridids=gridids)

def kernels(points,weights,centers,radius):
    tree=cKDTree(points);rows=[];cols=[];values=[]
    for j,ids in enumerate(tree.query_ball_point(centers,radius)):
        ids=np.asarray(ids);q=points[ids]-centers[j]
        w=(1-np.sum(q*q,axis=-1)/radius**2)**3*weights[ids]
        assert w.sum()>0;w/=w.sum()
        rows.extend([j]*len(ids));cols.extend(ids);values.extend(w)
    result=csr_matrix((values,(rows,cols)),shape=(len(centers),len(points)))
    assert np.max(abs(result@np.ones(len(points))-1))<1e-14
    return result

def prepare(name):
    d=load_case(name);r=d['run'];x,y=d['x'],d['y'];dx=x[1]-x[0];dy=y[1]-y[0]
    phase=d['phase'];times=d['times'];p0=phase[0];tip=d['tip'];ell=d['ell'];Kf=d['Kf'];ledger=d['ledger']
    count=len(times);xx,yy=np.meshgrid(x,y);X,Y=np.meshgrid((x[:-1]+x[1:])/2,(y[:-1]+y[1:])/2)
    radius=seed_table(ell)[0][-1];cap=(xx>=tip)&(np.hypot(xx-tip,yy-.5)<2.25*ell)
    virgin=p0==0;outside=np.hypot(np.maximum(xx-tip,0),yy-.5)>=radius
    # Reporting window only; the evolution law has no spatial activation gate.
    remote=(xx>tip+.4)&virgin
    growth=phase-p0;zero=np.zeros(count)
    first=np.full(p0.shape,np.nan)
    for n in range(1,count):
        chosen=np.isnan(first)&(growth[n]>1e-12);first[chosen]=times[n]
    onset={}
    for threshold in [1e-12,1e-5]:
        for label_,mask in [('whole',np.ones_like(p0,bool)),('cap',cap),('virgin',virgin),('remote',remote)]:
            act=np.flatnonzero(np.max(growth[:,mask],axis=1)>threshold)
            onset[f'{label_}_{threshold:g}']=None if not len(act) else [float(times[max(0,act[0]-1)]),float(times[act[0]])]
    save_csv(OUT/f'{name}-activation-nodes.csv',['x_m','y_m','initial_phi','activation_lower_s','activation_upper_s','initially_represented_zero'],
             [[float(xx[j,i]),float(yy[j,i]),float(p0[j,i]),float(times[np.searchsorted(times,first[j,i])-1]),float(first[j,i]),int(virgin[j,i])] for j,i in zip(*np.where(np.isfinite(first)))])
    # Centerline phase is exact Q1 at cell centers; rates retain native cell
    # averages. Cross-fault integral of |w| is activity, not signed surface V.
    mid=np.argmin(abs(y-.5));center=.5*(phase[:,mid,:-1]+phase[:,mid,1:]);delta=center-center[0]
    integrated=np.sum(d['rate'],axis=1)*dy;band=.5*(d['rate'][:,mid-1,:]+d['rate'][:,mid,:])
    save_csv(OUT/f'{name}-centerline.csv',['time_s','x_m','y_m','phi','phase_increment','band_mean_abs_w_s_inv','cross_fault_integral_abs_w_m_s'],
             [[float(times[n]),float(a),.5,float(center[n,i]),float(delta[n,i]),float(band[n,i]),float(integrated[n,i])] for n in range(count) for i,a in enumerate(X[0])])
    # Cell-center classification/integration is separated from nodal onset.
    pcell=(phase[:,:-1,:-1]+phase[:,:-1,1:]+phase[:,1:,:-1]+phase[:,1:,1:])/4
    dg=pcell-pcell[0];intact=np.hypot(np.maximum(X-tip,0),Y-.5)>=radius
    capcell=(X>=tip)&(np.hypot(X-tip,Y-.5)<2.25*ell)
    # Region integrals use actual Gauss8 positions and Q1 basis, rather than
    # classifying a partially intersected cell by its center.
    z,w=np.polynomial.legendre.leggauss(8);u,v=np.meshgrid((z+1)/2,(z+1)/2)
    u=u.ravel();v=v.ravel();weights=(w[:,None]*w[None,:]).ravel()*dx*dy/4
    Xq=x[:-1][None,:,None]+dx*u;Yq=y[:-1][:,None,None]+dy*v
    nq=np.stack([(1-u)*(1-v),u*(1-v),(1-u)*v,u*v],axis=-1)
    masks={'cap':(Xq>=tip)&(np.hypot(Xq-tip,Yq-.5)<2.25*ell),
           'outside':np.hypot(np.maximum(Xq-tip,0),Yq-.5)>=radius}
    region={}
    pieces=[growth[:,:-1,:-1],growth[:,:-1,1:],growth[:,1:,:-1],growth[:,1:,1:]]
    for key,mask in masks.items():
        coefficients=np.einsum('ijq,qk,q->ijk',mask,nq,weights)
        region[key]=sum(np.einsum('tij,ij->t',a,coefficients[:,:,k]) for k,a in enumerate(pieces))
    E0=fracture(p0,dx,dy,Kf,ell);energy=[];hist=[];previousQ=0.;previousE=E0;Wtr=0.
    dense_x=np.linspace(x[0],x[-1],(len(x)-1)*4+1);dense_y=np.linspace(y[0],y[-1],(len(y)-1)*4+1)
    XX,YY=np.meshgrid(dense_x,dense_y);topology=[]
    history_header=['step','time_s','max_phase_increment','cap_peak_increment','initial_zero_peak_increment','remote_peak_increment','whole_growth_m2','cap_growth_m2','outside_seed_growth_m2','fracture_J_m','added_fracture_J_m','bulk_J_m','cohesive_J_m','reaction_N_m','prescribed_strain','endpoint_work_J_m','trapezoidal_work_J_m','Db_J_m','Dc_J_m','Df_J_m','endpoint_remainder_J_m','trapezoidal_remainder_J_m','stress_loss_J_m','traction_loss_J_m','phase_remainder_J_m','fracture_loss_J_m','phase_residual_work_J_m','mechanical_residual_work_J_m','pressure_work_J_m']
    strain=0.
    for n,t in enumerate(times):
        a=np.loadtxt(r/f'boundary-reactions-{n:04d}.csv',delimiter=',',skiprows=1)
        selected=(a[:,1]==0)&(a[:,6]==1);Q=float(np.sum(a[selected,4]*(a[selected,3]-.5)))
        increments=a[selected,5];b=a[selected,3]-.5;chosen=b!=0
        dgamm=float(np.median(increments[chosen]/b[chosen])) if chosen.any() else 0.
        strain+=dgamm;work=float(np.sum(a[:,4]*a[:,5]));trap=.5*(Q+previousQ)*dgamm;previousQ=Q;Wtr+=trap
        Ef=fracture(phase[n],dx,dy,Kf,ell)
        if n:
            row=ledger[n-1];Eb=float(row['bulk']);Ec=float(row['cohesive']);D=float(row['Db']+row['Dc']+row['Df'])
            assert abs(work-row['Wext'])<1e-9 and abs(Ef-row['fracture'])<1e-9
            deltaE=Eb+Ec+Ef-previousE;be=work-deltaE-D;tr=trap-deltaE-D
            assert abs(be-(row['stress_loss']+row['traction_loss']+row['phase_remainder']+row['fracture_loss']-row['phase_work']-row['mechanical_work']))<1e-9
            terms=[float(row[k]) for k in ['Db','Dc','Df']]+[be,tr]+[float(row[k]) for k in ['stress_loss','traction_loss','phase_remainder','fracture_loss','phase_work','mechanical_work','pressure_work']]
            previousE=Eb+Ec+Ef
        else:Eb=Ec=0.;terms=[0.]*12
        hist.append([n,float(t),float(growth[n].max()),float(growth[n][cap].max(initial=0)),float(growth[n][virgin].max(initial=0)),float(growth[n][remote].max(initial=0)),float(dg[n].sum()*dx*dy),float(region['cap'][n]),float(region['outside'][n]),float(Ef),float(Ef-E0),Eb,Ec,Q,strain,work,trap,*terms])
        raster=grid_value(x,y,phase[n],XX,YY)
        rr=np.linspace(0,min(x[-1]-tip,.5)*np.sqrt(2),1025)
        rays=[grid_value(x,y,phase[n],tip+rr/np.sqrt(2),.5+sgn*rr/np.sqrt(2)) for sgn in [1,-1]]
        for level_ in LEVELS:
            comps,_=label(raster>=level_);seed=comps[np.argmin(abs(dense_y-.5)),0]
            connected=comps==seed;assert seed!=0
            present=np.unique(comps);disconnected=present[(present!=0)&(present!=seed)]
            right=float(XX[connected].max());cl=crossing(x,phase[n,mid],level_)
            topology.append([n,float(t),level_,cl,right,crossing(rr,rays[0],level_),crossing(rr,rays[1],level_),len(disconnected),int(np.any(connected[:,-1]))])
    save_csv(OUT/f'{name}-history.csv',history_header,hist)
    save_csv(OUT/f'{name}-contours.csv',['step','time_s','phase_threshold','centerline_first_exit_x_m','seed_component_rightmost_raster_x_m','upper_45_radius_m','lower_45_radius_m','detached_components','seed_component_touches_right'],topology)
    # Early times are saved native accepted states, including the bracket
    # before first initially-zero activation; no histories are resampled.
    selected_onset=onset['remote_1e-12'] or onset['virgin_1e-12'] or onset['whole_1e-12']
    base=int(np.searchsorted(times,selected_onset[1])) if selected_onset else count-1
    selected=sorted(set(max(1,min(count-1,k)) for k in [base-16,base-5,base-1,base,base+3,count-1]))
    centers=np.stack([np.linspace(0,x[-1],193),np.full(193,.5)],axis=-1)
    Kc=kernels(np.stack([X.ravel(),Y.ravel()],axis=-1),np.full(X.size,dx*dy),centers,.03125)
    force=read(r/f'phase-forces-{selected[0]:04d}.csv');force.sort(order='dof')
    node_points=np.empty((len(force),2));node_points[d['gridids'].ravel()]=np.stack([xx.ravel(),yy.ravel()],axis=-1)
    Kn=kernels(node_points,force['mass'],centers,.03125)
    profiles=[]
    for n in selected:
        f=read(r/f'phase-forces-{n:04d}.csv');f.sort(order='dof')
        mech=-f['mechanical_gradient']/f['mass'];fract=f['fracture_gradient']/f['mass']
        fields=[Kc@d['stress'][n].ravel(),Kc@d['Y'][n].ravel(),Kn@mech,Kn@fract,Kn@(mech-fract)]
        for i,a in enumerate(centers):profiles.append([n,float(times[n]),float(a[0]),.5,.03125,*[float(v[i]) for v in fields]])
    save_csv(OUT/f'{name}-early-profiles.csv',['step','time_s','x_m','y_m','kernel_radius_m','physical_tau_xy_Pa','cell_mean_Y_Pa','weak_mechanical_drive_Pa','weak_fracture_resistance_Pa','weak_drive_minus_resistance_Pa'],profiles)
    # Compact plotting cache belongs with full solver artifacts, not deliverables.
    np.savez_compressed(BUILD/f'{name}-plot-data.npz',x=x,y=y,phase=phase,time=times,stress=d['stress'],Y=d['Y'],rate=d['rate'],first=first)
    history=np.asarray(hist);header={s:i for i,s in enumerate(history_header)}
    totaltr=history[:,header['trapezoidal_work_J_m']].sum();variation=np.abs(history[:,header['trapezoidal_remainder_J_m']]).sum()
    s=dict(name=name,steps=count-1,dt_s=float(times[1]-times[0]),end_time_s=float(times[-1]),ell_m=ell,Gc_J_m2=d['Gc'],m=d['m'],tip_x_m=tip,
           initial_representation=initial_error(x,y,p0,ell,tip),initial_fracture_J_m=float(E0),
           initial_phase_min=float(p0.min()),initial_phase_max=float(p0.max()),
           unloaded_phase_drift=float(np.max(abs(growth[times<=.05+1e-12]))),onset=onset,
           final=dict(zip(history_header,map(float,hist[-1]))),
           mechanical_residual=float(max(ledger['mechanical_residual'])),phase_KKT=float(max(ledger['phase_KKT'])),source_error=float(max(ledger['projection_error'])),
           ledger_error_J_m=float(max(abs(ledger['ledger_error']))),
           cumulative={key:float(history[:,header[key]].sum()) for key in history_header[15:]},
           energy_total_variation_relative=float(variation/abs(totaltr)),
           selected_profile_steps=selected,remote_reporting_window_x_gt_m=tip+.4,
           contour_raster_subdivisions=4,activation_increment_thresholds=[1e-12,1e-5])
    f=read(r/'phase-forces-0001.csv')
    s['unloaded_min_weak_fracture_over_mass_Kf']=float(np.min(f['fracture_gradient']/f['mass']/Kf))
    resources=list(csv_module.DictReader((r/'resources.csv').open()))
    assert len({a['particle_fingerprint'] for a in resources})==1 and all(a['freeze']=='1' for a in resources)
    s['stationary_particles']=dict(count=int(resources[0]['particles']),fingerprint=resources[0]['particle_fingerprint'],
        maximum_rank_peak_MiB=max(float(a['max_rank_KiB']) for a in resources)/1024,
        sum_rank_peaks_MiB=max(float(a['sum_rank_peaks_KiB']) for a in resources)/1024)
    s['detached_contour_components_max']={str(level_):int(max(a[7] for a in topology if a[2]==level_)) for level_ in LEVELS}
    snap=min(int(np.searchsorted(times,.175)),count-1)
    save_csv(OUT/f'{name}-map-cells.csv',['time_s','x_m','y_m','mean_phi','mean_phase_increment','physical_tau_xy_Pa','Y_Pa','mean_abs_w_s_inv'],
       [[float(times[snap]),float(X[j,i]),float(Y[j,i]),float(pcell[snap,j,i]),float(dg[snap,j,i]),float(d['stress'][snap,j,i]),float(d['Y'][snap,j,i]),float(d['rate'][snap,j,i])] for j in range(len(y)-1) for i in range(len(x)-1)])
    (OUT/f'{name}-summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case',choices=['released','candidate']);a=p.parse_args();prepare(a.case)
