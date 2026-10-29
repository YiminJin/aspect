#!/usr/bin/env python3
"""One mixed-boundary control against the retained affine loading trajectory.

Only Q1 phase is evaluated on common physical points. Local stress/driving are
read at their original QPs or averaged with their original positive weights.
"""
from pathlib import Path
import argparse,json,csv as standard_csv
import numpy as np
from scipy.ndimage import label
from assess_native import csv,G,KF,ELL,SEED_R,norm3
from assess_spatial import layout,q1,masks
from assess_continued import contour_geometry,LEVELS
from recover_remote import kernels

OLD=Path('build-particle-free/bulk-continued/runs/continued')
NEW=Path('build-particle-free/bulk-parallel/runs/mixed')
EVIDENCE=Path('benchmarks/reconstructed_fault/particle_free/evidence/bulk_continued')


def phase(path,n):
    a=csv(path/f'phase-{n:04d}.csv');a.sort(order='dof');return a['phi']


def reaction(path,n):
    a=csv(path/f'boundary-reactions-{n:04d}.csv')
    horizontal=a['component']==0
    prescribed=a['prescribed']==1 if 'prescribed' in a.dtype.names else np.ones(len(a),bool)
    chosen=horizontal & prescribed
    # On both specimens the prescribed horizontal components have exactly the
    # same velocity basis y-.5; lateral ux in NEW is excluded from work.
    b=a['y']-.5
    Q=float(np.sum(a['force'][chosen]*b[chosen]))
    grip=chosen & ((a['y']==0)|(a['y']==1))
    side=chosen & ~grip
    Qgrip=float(np.sum(a['force'][grip]*b[grip]));Qside=float(np.sum(a['force'][side]*b[side]))
    du=a['increment'][chosen];ratio=du[b[chosen]!=0]/b[chosen][b[chosen]!=0]
    dg=float(np.median(ratio)) if len(ratio) else 0.
    assert np.max(abs(du-dg*b[chosen]),initial=0.)<1e-15
    assert np.max(abs(a['increment'][~prescribed]),initial=0.)==0.
    W=float(np.sum(a['force']*a['increment']))
    return Q,Qgrip,Qside,dg,W


def assess(path,L,output,name):
    rows=csv(path/'bulk-accepted.csv');count=len(rows);assert int(rows['step'][-1])==148
    assert np.array_equal(rows['step'],np.arange(1,count+1)) and np.array_equal(rows['publications'],rows['step'])
    assert max(rows['mechanical_residual'])<=1e-9 and max(rows['phase_KKT'])<=1e-7 and max(rows['projection_error'])<=1e-12
    assert np.all(csv(path/'gates.csv')['passed']==1)
    p0=phase(path,0);phases=np.stack([phase(path,n) for n in range(count+1)])
    values=q1(L,phases[0],L['points'])[0];grad=q1(L,phases[0],L['points'])[1]
    E0=KF*np.sum(L['weights']*(values+ELL**2*np.sum(grad*grad,axis=-1)))
    regions=dict(whole=np.ones_like(L['weights'],bool),**masks(L['points']))
    support=np.hypot(np.maximum(L['points'][...,0]-.5,0),L['points'][...,1]-.5)
    regions['initially_intact']=support>=SEED_R[-1]
    regions['represented_zero']=values==0
    regions['remote']=L['points'][...,0]>.9
    regions['right_strip']=L['points'][...,0]>=1.5-2*L['dx']
    history=[];contours={};previousQ=0.;previousE=E0;energy=[]
    locations=L['node_xy'];r=np.hypot(np.maximum(locations[:,0]-.5,0),locations[:,1]-.5)
    nodal_regions=dict(whole=np.ones(len(p0),bool),cap=(locations[:,0]>.5)&(np.linalg.norm(locations-.5,axis=1)<.35),remote=(locations[:,0]>.9)&(p0==0),right=locations[:,0]==1.5)
    nodal_regions['upper']=nodal_regions['cap']&(locations[:,1]>.5)
    nodal_regions['lower']=nodal_regions['cap']&(locations[:,1]<.5)
    onset={k:None for k in nodal_regions}
    for n,p in enumerate(phases):
        t=0. if n==0 else float(rows[n-1]['time']);growth=p-p0;dp=q1(L,growth,L['points'])[0]
        Q,Qgrip,Qside,dg,W=reaction(path,n)
        result=dict(step=n,time=t,Q=Q,Qgrip=Qgrip,Qside=Qside,strain_increment=dg,peak_growth=float(max(growth)),integrals={k:float(np.sum(L['weights'][m]*dp[m])) for k,m in regions.items()},growth_L2={k:float(np.sqrt(np.sum(L['weights'][m]*dp[m]**2))) for k,m in regions.items()},peaks={k:float(max(growth[m],default=0.)) for k,m in nodal_regions.items()})
        for key,mask in nodal_regions.items():
            if onset[key] is None and np.max(growth[mask],initial=0)>1e-12:
                onset[key]=[float(rows[n-2]['time']) if n>1 else 0.,t]
        if n:
            row=rows[n-1];Wtr=.5*(Q+previousQ)*dg
            endpoint_error=abs(W-row['Wext']);assert endpoint_error<1e-9
            E=float(row['bulk']+row['cohesive']+row['fracture']);D=float(row['Db']+row['Dc']+row['Df'])
            tr=Wtr-(E-previousE)-D;BE=W-(E-previousE)-D
            energy.append([n,t,W,Wtr,E-E0,D,BE,tr,row['stress_loss'],row['traction_loss'],row['phase_remainder'],row['fracture_loss'],row['phase_work'],row['mechanical_work'],row['pressure_work'],endpoint_error])
            previousE=E
        previousQ=Q;history.append(result)
        contours[str(n)]=contour_geometry(L,p,8)
        if n==148:
            check=contour_geometry(L,p,16)
            assert all(check[k]['disconnected_components']==contours[str(n)][k]['disconnected_components'] for k in check)
            assert max(abs(check[k]['rightmost']-contours[str(n)][k]['rightmost']) for k in check)<L['dx']/8
    energy=np.array(energy)
    np.savetxt(output/f'{name}-energy.csv',energy,delimiter=',',header='step,time,endpoint_work,trapezoidal_work,storage_change,physical_dissipation,endpoint_remainder,trapezoidal_remainder,stress_loss,traction_loss,phase_remainder,fracture_loss,phase_work,mechanical_work,pressure_work,reaction_work_discrepancy',comments='')
    summary=dict(steps=count,initial_fracture=E0,onset=onset,final=history[-1],endpoint_work=float(sum(energy[:,2])),trapezoidal_work=float(sum(energy[:,3])),storage_change=float(energy[-1,4]),physical_dissipation=float(sum(energy[:,5])),endpoint_remainder=float(sum(energy[:,6])),trapezoidal_remainder=float(sum(energy[:,7])),trapezoidal_total_variation=float(sum(abs(energy[:,7]))),CE=float(sum(abs(energy[:,7]))/sum(abs(energy[:,3]))),CT='not measured for this extended-loading interval',max_reaction_work_error=float(max(energy[:,-1])),mechanics=float(max(rows['mechanical_residual'])),KKT=float(max(rows['phase_KKT'])),source=float(max(rows['projection_error'])),ledger=float(max(abs(rows['ledger_error']))),final_contours=contours[str(count)],initial_contours=contours['0'])
    (output/f'{name}-history.json').write_text(json.dumps(history,indent=2)+'\n');(output/f'{name}-contours.json').write_text(json.dumps(contours,indent=2)+'\n')
    return summary,phases,np.array([0.,*rows['time']])


def main(out):
    out.mkdir(exist_ok=True,parents=True);L=layout(NEW);oldL=layout(OLD)
    a=csv(OLD/'phase-0000.csv');b=csv(NEW/'phase-0000.csv');a.sort(order=['x','y']);b.sort(order=['x','y'])
    assert np.array_equal(a['x'],b['x']) and np.array_equal(a['y'],b['y']) and np.array_equal(a['phi'],b['phi'])
    summary,phases,times=assess(NEW,L,out,'mixed')
    old=json.load(open(EVIDENCE/'assessment.json'));oldh=json.load(open(EVIDENCE/'history.json'))
    # Reuse the already independently checked affine ledger/topology. Repeat the
    # reaction functional here to expose the lateral prescribed-work part.
    oldreaction=np.array([reaction(OLD,n) for n in range(149)])
    reaction_check=float(max(abs(oldreaction[:,0]-np.array([r['Q'] for r in oldh]))));assert reaction_check<1e-9
    summary['baseline_reaction_recheck']=reaction_check
    pressure=csv(NEW/'pressure.csv');assert np.all(pressure['free_side_scaled_residual'][1:]<=1e-9)
    summary['pressure']=dict(initial_guess_mean=float(pressure['mean'][0]),unloaded_mean_at_first_step=float(pressure['mean'][1]),final_mean=float(pressure['mean'][-1]),minimum=float(min(pressure['min'][1:])),maximum=float(max(pressure['max'][1:])),max_natural_residual=float(max(pressure['free_side_scaled_residual'][1:])))
    profile=csv(NEW/'bulk-profile.csv');assert np.all(profile['open_normal']==1)
    resource=list(standard_csv.DictReader((NEW/'resources.csv').open()))
    assert len({r['particle_fingerprint'] for r in resource})==1 and all(r['freeze']=='1' for r in resource)
    summary['particles']=dict(count=int(resource[0]['particles']),fixed_fingerprint=resource[0]['particle_fingerprint'],max_rank_MiB=max(float(r['max_rank_KiB']) for r in resource)/1024,sum_rank_peaks_MiB=max(float(r['sum_rank_peaks_KiB']) for r in resource)/1024)
    # All 40 unloaded intervals must preserve the same physical obstacle exactly.
    summary['unloaded_phase_drift']=float(np.max(abs(phases[:41]-phases[0])));assert summary['unloaded_phase_drift']==0.
    summary['initial_representation']=old['initialization_reference']
    assert np.array_equal(times,np.array([r['time'] for r in oldh]))
    force=csv(NEW/'phase-forces-0120.csv');force.sort(order='dof')
    Kn=kernels(L['node_xy'],force['mass'],np.concatenate([np.stack([np.linspace(0,1.5,193),np.full(193,.5)],axis=1),np.stack([.5+np.linspace(0,.69,121)/np.sqrt(2),.5+np.linspace(0,.69,121)/np.sqrt(2)],axis=1),np.stack([.5+np.linspace(0,.69,121)/np.sqrt(2),.5-np.linspace(0,.69,121)/np.sqrt(2)],axis=1)]))
    common=[120,121,124,128,139,140,148];profiles={};checks={}
    xx=np.linspace(0,1.5,193);rr=np.linspace(0,.69,121)
    centers=np.concatenate([np.stack([xx,xx*0+.5],axis=1),np.stack([.5+rr/np.sqrt(2),.5+rr/np.sqrt(2)],axis=1),np.stack([.5+rr/np.sqrt(2),.5-rr/np.sqrt(2)],axis=1)])
    K=kernels(L['points'].reshape(-1,2),L['weights'].ravel(),centers)
    for n in common:
        file=NEW/f'bulk-{n:04d}.csv'
        with file.open() as stream:header=stream.readline().strip().split(',')
        raw=np.loadtxt(file,delimiter=',',skiprows=1,usecols=range(1,len(header)))
        d={key:raw[:,j] for j,key in enumerate(header[1:])};tau=np.stack([d[k] for k in ['tauxx','tauyy','tauxy']],axis=-1)
        expected=np.stack([d[k] for k in ['Txx','Tyy','Txy']],axis=-1);expected[:,2]-=G*d['a'];assert np.max(abs(tau-expected))<1e-10
        accepted=csv(NEW/'bulk-accepted.csv');row=accepted[n-1]
        eb=np.sum(d['JxW']*norm3(tau)/(4*G));ec=np.sum(d['JxW']*128.00000000016*d['phi']*(1+d['phi'])/(1-d['phi'])**2*d['q']**2/(2*G))
        assert max(abs(eb-row['bulk']),abs(ec-row['cohesive']))<1e-9
        force=csv(NEW/f'phase-forces-{n:04d}.csv');force.sort(order='dof')
        D=-force['mechanical_gradient']/force['mass'];F=force['fracture_gradient']/force['mass']
        source=np.zeros(len(force));fracture=np.zeros(len(force));grad=q1(L,phases[n],L['points'])[1]
        Y=d['driving'].reshape(L['weights'].shape)
        for j in range(4):
            np.add.at(source,L['ids'][:,j],np.sum(L['weights']*L['N'][:,j]*Y,axis=1))
            np.add.at(fracture,L['ids'][:,j],KF*np.sum(L['weights']*(L['N'][:,j]+2*ELL**2*np.sum(grad*L['gradN'][:,:,j],axis=-1)),axis=1))
        source_error=float(max(abs(source+force['mechanical_gradient'])));fracture_error=float(max(abs(fracture-force['fracture_gradient'])))
        assert max(source_error,fracture_error)<1e-10
        profiles[str(n)]=np.stack([K@tau[:,2],K@d['driving'],Kn@D,Kn@F,Kn@(D-F)],axis=-1)
        h=128.00000000016*d['phi']*(1+d['phi'])/(1-d['phi'])**2
        scale=np.maximum(1000.,np.sqrt(norm3(np.stack([d[k] for k in ['Txx','Tyy','Txy']],axis=-1))))
        elastic=float(max(abs(G*d['a']-h*d['q'])/scale))
        friction=float(max(np.maximum(abs(tau[:,2]-d['q'])-1000.,0.)/scale))
        virgin=d['phi']==0
        virgin_q=float(np.max(abs(d['q'][virgin]-np.sign(tau[virgin,2])*np.maximum(abs(tau[virgin,2])-1000.,0.))/scale[virgin],initial=0.))
        friction_history=float(abs(1000.*np.sum(d['JxW']*d['L'])-np.sum(accepted['Df'][:n])))
        assert max(elastic,friction,virgin_q)<=1e-12 and friction_history<1e-9
        checks[str(n)]=dict(elastic_relation=elastic,friction_interval=friction,virgin_multiplier=virgin_q,cumulative_friction_history_error_J_m=friction_history,max_tau=float(max(tau[:,2])),min_tau=float(min(tau[:,2])),max_Y=float(max(d['driving'])),raw_mechanical_phase_force_error=float(source_error),raw_fracture_force_error=float(fracture_error),bulk_storage_error=float(abs(eb-row['bulk'])),cohesive_storage_error=float(abs(ec-row['cohesive'])))
    summary['dense_independent_checks']=checks
    (out/'boundary-assessment.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savez_compressed(out/'mixed-fields.npz',x=L['x'],y=L['y'],grid_ids=L['grid_ids'],phase=phases,time=times,
        old_reactions=oldreaction,line_points=centers,xx=xx,rr=rr,profile_steps=common,profiles=np.stack([profiles[str(n)] for n in common]))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
