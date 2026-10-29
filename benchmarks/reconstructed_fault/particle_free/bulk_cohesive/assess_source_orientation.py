#!/usr/bin/env python3
"""One qualified orientation pair: original QP histories and physical kernels."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from assess_native import csv,KF,norm3
from assess_orientation import state,initial_error,phase_only_kernel

BASE=Path('build-particle-free/bulk-source-history')
RUNS={'aligned':BASE/'runs/orientation-aligned','rotated':BASE/'runs/orientation-rotated-centered'}
MESH={'aligned':Path('build-particle-free/bulk-orientation/runs/identity/mesh-qualification.csv'),
      'rotated':RUNS['rotated']/'mesh-qualification.csv'}


def average(s,centers,radius=.0625):
    tree=cKDTree(s['points']);rows=[]
    for center,ids in zip(centers,tree.query_ball_point(centers,radius)):
        ids=np.asarray(ids);d=s['points'][ids]-center;r2=np.sum(d*d,axis=-1)/radius**2
        K=(1-r2)**3;gradK=-6*d/radius**2*(1-r2[:,None])**2;w=s['weight'][ids];mass=np.sum(w*K)
        F=KF*np.sum(w*(K+2*.15625**2*np.sum(s['gradient'][ids]*gradK,axis=-1)))/mass
        p,q,Y=[np.sum(w*K*s[k][ids])/mass for k in ['phi','q','Y']]
        rows.append([*center,p,q,Y,F,F-Y,mass])
    return np.array(rows)


def phases(path):
    result={}
    for f in sorted(path.glob('phase-[0-9]*.csv')):
        n=int(f.stem.split('-')[1]);s=csv(f);s.sort(order='dof');result[n]=s
    return result


def first_cap(p):
    xy=np.stack([p[0]['x'],p[0]['y']],axis=-1);r=np.linalg.norm(xy-.5,axis=-1)
    cap=(xy[:,0]>.5+1e-12)&(r<.35)
    for step in sorted(p):
        ids=np.flatnonzero(cap&(p[step]['phi']-p[0]['phi']>1e-12))
        if len(ids):return step,ids
    raise AssertionError('No accepted cap activation')


def main(output):
    output.mkdir(parents=True,exist_ok=True)
    p={k:phases(v) for k,v in RUNS.items()};first={k:first_cap(v) for k,v in p.items()}
    before=min(v[0] for v in first.values())-1;final=min(max(v) for v in p.values())
    radii=[.225,.2881107642904027,.29851520585223124,.325];angles=np.linspace(-90,90,181)
    centers=np.concatenate([np.stack([.5+r*np.cos(np.deg2rad(angles)),.5+r*np.sin(np.deg2rad(angles))],axis=-1) for r in radii])
    report=dict(common_pre_step=before,common_final_step=final,cases={},strict_reference_match_passed=False,
                kernel_radius_m=.0625,history_interpolation=False,orientation_run_through_onset_completed=True,mesh_independent_onset_qualified=False)
    fields={};arcs={}
    for name,path in RUNS.items():
        step,ids=first[name];cache={s:state(path,MESH[name],s) for s in sorted({0,before,step-1,step,final})}
        initial=cache[0];last=cache[final];ledger=csv(path/'bulk-accepted.csv')
        assert np.array_equal(ledger['publications'],np.arange(1,final+1))
        assert max(ledger['projection_error'])<=1e-12 and max(ledger['mechanical_residual'])<=1e-9 and max(ledger['phase_KKT'])<=1e-7
        xy=initial['node_xy'];radius=np.linalg.norm(xy-.5,axis=-1);cap=(xy[:,0]>.5+1e-12)&(radius<.35)
        delta=last['phase_nodes']-initial['phase_nodes'];forces={n:csv(path/f'phase-forces-{n:04d}.csv') for n in [step-1,step]}
        for f in forces.values():f.sort(order='dof')
        growing=[]
        for i in ids:
            row=dict(dof=int(i),x=float(xy[i,0]),y=float(xy[i,1]),radius=float(radius[i]),
                     angle=float(np.rad2deg(np.arctan2(xy[i,1]-.5,xy[i,0]-.5))),
                     initial_phi=float(initial['phase_nodes'][i]),first_increment=float(p[name][step]['phi'][i]-initial['phase_nodes'][i]))
            for label,n in [('pre',step-1),('onset',step)]:
                f=forces[n]
                for key in ['fracture_gradient','mechanical_gradient','residual']:
                    row[label+'_'+key+'_Pa']=float(f[key][i]/f['mass'][i])
            growing.append(row)
        W=sum(ledger['Wext']);D=sum(ledger['Db']+ledger['Dc']+ledger['Df']);E=sum(ledger['energy_change']);rem=W-D-E
        dense=csv(path/f'bulk-{final:04d}.csv');R=np.stack([dense[k] for k in ['Rxx','Ryy','Rxy']],axis=-1).astype(float)
        T=np.stack([dense[k] for k in ['Txx','Tyy','Txy']],axis=-1).astype(float);tau=T.copy();tau[:,2]-=1e6*dense['a']
        exported=np.stack([dense[k] for k in ['tauxx','tauyy','tauxy']],axis=-1).astype(float)
        stress_error=float(max(np.sqrt(norm3(tau-exported))/np.maximum(1000.,np.sqrt(norm3(tau)))))
        energy_error=abs(float(np.sum(dense['JxW']*norm3(tau)/(4e6)))-ledger['bulk'][-1])
        assert stress_error<1e-10 and energy_error<1e-10
        tangent=csv(path/'bulk-tangent.csv')
        record=dict(cells=len(initial['vertices']),DG_tensor_DoFs=int(csv(path/'fixture.csv')['DG_DoFs'][0]),
            native_velocity_memory_DoFs=(path/'native-history-0000.bin').stat().st_size//8,
            first_cap_step=step,first_cap_time=cache[step]['time'],first_growing_nodes=growing,
            initial_error=initial_error(initial),maximum_cap_increment=float(max(delta[cap])),
            maximum_boundary_increment=float(max(delta[xy[:,0]>1.])),minimum_phase_increment=float(min(delta)),
            maximum_native_source_error=float(max(ledger['projection_error'])),maximum_mechanical_residual=float(max(ledger['mechanical_residual'])),
            maximum_phase_KKT=float(max(ledger['phase_KKT'])),maximum_tangent_FD_error=float(max(tangent['maximum_FD_error'])),
            final_remainder_max_Pa=float(max(np.sqrt(norm3(R)))),external_work_J_per_m=float(W),physical_dissipation_J_per_m=float(D),
            final_independent_physical_stress_error=stress_error,final_independent_bulk_energy_error_J_per_m=energy_error,
            stored_energy_change_J_per_m=float(E),external_remainder_J_per_m=float(rem),relative_external_remainder=float(abs(rem)/max(1,abs(W))),
            physical_energy_screen_passed=bool(abs(rem)<=.01*max(1,abs(W))),
            maximum_independent_weak_error=float(max(s['weak_gradient_normalized_error'] for s in cache.values())))
        record['energy_terms_J_per_m']={k:float(sum(ledger[k])) for k in ['stress_loss','traction_loss','phase_remainder','fracture_loss','phase_work','mechanical_work']}
        record['first_cap_by_side']={}
        for side,mask in [('upper',xy[:,1]>.5),('lower',xy[:,1]<.5)]:
            for n in sorted(p[name]):
                active=np.flatnonzero(cap&mask&(p[name][n]['phi']-initial['phase_nodes']>1e-12))
                if len(active):
                    record['first_cap_by_side'][side]=dict(step=n,time=float(ledger['time'][n-1]),
                        angles=list(map(float,np.rad2deg(np.arctan2(xy[active,1]-.5,xy[active,0]-.5)))))
                    break
        arrays={};peaks={}
        for label,s in [('initial',initial),('pre',cache[before]),('common',last)]:
            a=average(s,centers);arcs[(name,label)]=a
            np.savetxt(output/f'{name}-{label}-arcs.csv',np.column_stack([np.repeat(radii,len(angles)),np.tile(angles,len(radii)),a]),
                       delimiter=',',header='radius,angle,x,y,phi,q,Y,fracture,total,kernel_mass',comments='')
            arrays.update({label+'_'+k:v for k,v in s.items() if isinstance(v,np.ndarray)})
            if label=='common':
                for ir,r in enumerate(radii):
                    v=a[ir*len(angles):(ir+1)*len(angles)]
                    peaks[str(r)]={side:float(angles[np.flatnonzero(mask)[np.argmax(v[mask,4])]]) for side,mask in [('upper',angles>=0),('lower',angles<=0)]}
        # Common-kernel forces at the actual first-growing physical nodes,
        # evaluated on each side's own unmodified QP histories.
        actual=average(cache[step-1],xy[ids]);np.savetxt(output/f'{name}-first-node-kernels.csv',actual,delimiter=',',
            header='x,y,phi,q,Y,fracture,total,kernel_mass',comments='')
        record['final_Y_peak_angles']=peaks
        np.savez_compressed(output/f'{name}-fields.npz',**arrays);fields[name]=arrays
        report['cases'][name]=record
    report['common_pre_time']=state(RUNS['aligned'],MESH['aligned'],before)['time']
    report['common_final_time']=last['time'];report['common_kernel_differences']=[]
    for i,r in enumerate(radii):
        sl=slice(i*len(angles),(i+1)*len(angles));a=arcs['aligned','common'][sl];b=arcs['rotated','common'][sl]
        report['common_kernel_differences'].append(dict(radius=r,Y_relative_L2=float(np.linalg.norm(a[:,4]-b[:,4])/np.linalg.norm(a[:,4])),
            Y_maximum_difference_Pa=float(max(abs(a[:,4]-b[:,4]))),fracture_initial_maximum_difference_Pa=float(max(abs(arcs['aligned','initial'][sl,5]-arcs['rotated','initial'][sl,5])))))
    # Repeat the already-established phase-only integration check, without
    # evaluating a new constitutive state on another quadrature layout.
    centers24=centers[::16]
    report['phase_kernel_quadrature_8_24_Pa']={}
    for name in RUNS:
        s=state(RUNS[name],MESH[name],final);a=average(s,centers24);b=phase_only_kernel(s,centers24,24)
        report['phase_kernel_quadrature_8_24_Pa'][name]=float(max(abs(a[:,5]-b[:,5])))
    # Affine reference agreement at exactly the same accepted time sequence.
    old=Path('build-particle-free/bulk-sensitivity/runs/right-distance');new=RUNS['aligned']
    a=csv(old/'bulk-accepted.csv')[:final];b=csv(new/'bulk-accepted.csv');assert np.array_equal(a['time'],b['time']) and np.array_equal(a['dt'],b['dt'])
    oldphase=csv(old/f'phase-{final:04d}.csv');oldphase.sort(order='dof')
    olddense=csv(old/f'bulk-{final:04d}.csv') if (old/f'bulk-{final:04d}.csv').exists() else None
    report['retained_affine_comparison']=dict(phase_Linf=float(max(abs(oldphase['phi']-p['aligned'][final]['phi']))),
        bulk_storage_difference_J_per_m=float(abs(a['bulk'][-1]-b['bulk'][-1])),cohesive_storage_difference_J_per_m=float(abs(a['cohesive'][-1]-b['cohesive'][-1])))
    dense_step=4*(final//4);original=csv(old/f'bulk-{dense_step:04d}.csv');current=csv(new/f'bulk-{dense_step:04d}.csv')
    difference=np.stack([current[k]-original[k] for k in ['tauxx','tauyy','tauxy']],axis=-1)
    original_tau=np.stack([original[k] for k in ['tauxx','tauyy','tauxy']],axis=-1).astype(float)
    report['retained_affine_comparison'].update(stress_step=dense_step,
        full_stress_relative_L2=float(np.sqrt(np.sum(current['JxW']*norm3(difference))/np.sum(current['JxW']*norm3(original_tau)))))
    (output/'assessment.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);main(parser.parse_args().output)
