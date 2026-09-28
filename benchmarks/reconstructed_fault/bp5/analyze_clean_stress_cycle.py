"""Analyze the small production cycle, preserving particle/FE time levels."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from analyze_stress_cycle_cells import q2

def rows(path):
    with path.open() as stream:return list(csv.DictReader(stream))

def rms(v):return float(np.sqrt(np.mean(np.asarray(v)**2)))

def tensor(r,prefix=''):
    return np.array([float(r[prefix+c]) for c in ('xx','yy','xy')])

def weak_load(qps,stress,include_local_norm=False):
    """Independent Q2 velocity weak moments on this fixed uniform periodic box.

    Sum shared nodes, identify x periodic nodes, and remove the prescribed
    top/bottom velocity rows. No hanging nodes exist in this fixture.
    """
    h=1/64
    gauss=np.polynomial.legendre.leggauss(3)[0]/2+.5
    def basis(x):return np.array([2*(x-.5)*(x-1),4*x*(1-x),2*x*(x-.5)])
    def derivative(x):return np.array([4*x-3,4-8*x,4*x-1])
    loads=defaultdict(float);local=defaultdict(float)
    for key,r in qps.items():
        x=float(r['x']);y=float(r['y'])+.5;w=float(r['JxW']);xx,yy,xy=stress[key]
        cx=int(x/h);cy=int(y/h);rx=x/h-cx;ry=y/h-cy
        assert min(abs(rx-gauss))<1e-10 and min(abs(ry-gauss))<1e-10
        bx,by=basis(rx),basis(ry);dx,dy=derivative(rx)/h,derivative(ry)/h
        for i in range(3):
            for j in range(3):
                ix=(2*cx+i)%32;iy=2*cy+j
                terms=(w*(xx*dx[i]*by[j]+xy*bx[i]*dy[j]),w*(xy*dx[i]*by[j]+yy*bx[i]*dy[j]))
                for c,value in enumerate(terms):
                    local[(r['cell'],i,j,c)]+=value
                    if iy not in (0,128):loads[(ix,iy,c)]+=value
    return (loads,float(np.linalg.norm(list(local.values())))) if include_local_norm else loads

def main(path):
    result={};qp={};parents={};old_at_parent={};new_at_parent={}
    support=np.array([(x,y) for x in (0.,.5,1.) for y in (0.,.5,1.)])
    gauss=np.polynomial.legendre.leggauss(3)[0]/2+.5
    previous=None
    for step in range(5):
        before={r['id']:r for r in rows(path/f'clean_before_{step}_rank0.csv')}
        after={r['id']:r for r in rows(path/f'clean_after_{step}_rank0.csv')}
        assert before.keys()==after.keys()
        for key in before:
            assert [before[key][x] for x in ('x','y')]==[after[key][x] for x in ('x','y')]
            if previous is not None:np.testing.assert_array_equal(tensor(before[key]),tensor(previous[key]))
        if step==0:assert max(np.max(abs(tensor(r))) for r in after.values())==0.
        previous=after;parents[step]=after
        samples=rows(path/f'clean_qp_{step}_rank0.csv');qp[step]={(r['cell'],r['q']):r for r in samples}
        if step:
            for key,r in qp[step].items():
                for field in ('x','y','phi','chi','V'):
                    assert float(r[field])==float(qp[0][key][field]),(step,key,field)
        cells=defaultdict(list)
        for r in samples:cells[r['cell']].append(r)
        old_at_parent[step]={};new_at_parent[step]={}
        coefficients={};published_difference=[]
        for cell,values in cells.items():
            xs=sorted({float(r['x']) for r in values});ys=sorted({float(r['y']) for r in values})
            assert len(xs)==len(ys)==3
            unit=np.array([(gauss[xs.index(float(r['x']))],gauss[ys.index(float(r['y']))]) for r in values])
            coefficients[cell]=np.linalg.solve(q2(unit,support),[tensor(r,'old_') for r in values])
            published_difference.extend([tensor(r,'old_')-tensor(r,'published_') for r in values])
        for key,r in before.items():
            unit=[[float(r['ref_x']),float(r['ref_y'])]]
            old_at_parent[step][key]=(q2(unit,support)@coefficients[r['cell']])[0]
        info=dict(particles=len(before),cells=len(cells),
                  published_working_difference_max_Pa=float(np.max(abs(np.array(published_difference)))),
                  incoming_FE_RMS_by_component=np.sqrt(np.mean(np.array([tensor(r,'old_') for r in samples])**2,axis=0)).tolist())
        if step:
            updates={r['particle_id']:r for r in rows(path/f'stress_update_{step}_rank0.csv')}
            assert len(updates)==len(before)
            unresolved=[];increment=[];new=[];strain=[];crack=[];closure=[]
            band=[]
            for key,r in before.items():
                u=updates[key];old=tensor(r);candidate=tensor(after[key]);beta=float(u['beta']);kappa=float(u['kappa'])
                np.testing.assert_array_equal(tensor(u,'old_'),old)
                np.testing.assert_array_equal(tensor(u,'new_'),candidate)
                expected=beta*old+2*kappa*(tensor(u,'eps_')-tensor(u,'crack_'))
                closure.extend(candidate-expected)
                unresolved.append(old-old_at_parent[step][key]);increment.append(candidate-old);new.append(candidate)
                strain.append(2*kappa*tensor(u,'eps_'));crack.append(-2*kappa*tensor(u,'crack_'))
                # Parent-quadrature FE history mismatch uses identical physical
                # positions; do not compare raw arrays from different samplers.
                new_at_parent[step][key]=beta*old_at_parent[step][key]+2*kappa*(tensor(u,'eps_')-tensor(u,'crack_'))
                band.append(abs(float(r['y']))<.2)
            unresolved=np.array(unresolved);increment=np.array(increment);new=np.array(new);band=np.array(band)
            info.update(dt=sorted({float(r['dt']) for r in updates.values()}),
                        formula_error_max_Pa=float(max(abs(np.array(closure)))),
                        particle_RMS_by_component=np.sqrt(np.mean(new**2,axis=0)).tolist(),
                        unresolved_old_RMS_by_component=np.sqrt(np.mean(unresolved**2,axis=0)).tolist(),
                        increment_RMS_by_component=np.sqrt(np.mean(increment**2,axis=0)).tolist(),
                        band_unresolved_xy_RMS=rms(unresolved[band,2]),
                        band_increment_xy_RMS=rms(increment[band,2]))
            if np.linalg.norm(unresolved[band,2])>1e-12:
                info['band_increment_projection_on_unresolved_xy']=float(np.dot(unresolved[band,2],increment[band,2])/
                                                                        np.linalg.norm(unresolved[band,2])/np.sqrt(sum(band)))
            # Deviation about each cell's mean, not global spatial variation.
            groups=defaultdict(list)
            for r in after.values():groups[r['cell']].append(tensor(r))
            deviation=np.concatenate([np.array(v)-np.mean(v,axis=0) for v in groups.values()])
            info['within_cell_particle_RMS_by_component']=np.sqrt(np.mean(deviation**2,axis=0)).tolist()
        loads=rows(path/f'clean_weak_history_{step}_rank0.csv')
        info['local_history_load_l2']=float(np.linalg.norm([float(r['load']) for r in loads]))
        _,independent=weak_load(qp[step],{k:tensor(r,'old_') for k,r in qp[step].items()},True)
        info['independent_local_load_norm_error']=abs(independent-info['local_history_load_l2'])
        assert info['independent_local_load_norm_error']<1e-10*max(1.,independent)
        # These signed local loads precede global constraint cancellation, not
        # the nonlinear residual. Keep the label explicit.
        result[str(step)]=info
    for step in range(1,4):
        defect=[];same_point=[];xband=[]
        for key,r in parents[step].items():
            defect.append(tensor(r)-old_at_parent[step+1][key])
            same_point.append(old_at_parent[step+1][key]-new_at_parent[step][key])
            xband.append(abs(float(r['y']))<.2)
        result[str(step)]['after_transfer_unresolved_RMS_by_component']=np.sqrt(np.mean(np.array(defect)**2,axis=0)).tolist()
        result[str(step)]['next_FE_minus_current_mechanical_at_parent_RMS_by_component']=np.sqrt(np.mean(np.array(same_point)**2,axis=0)).tolist()
        # Freeze the accepted bulk velocity/source and compare its constitutive
        # stress with the next step's transferred retained tensor. This is a
        # representation-jump diagnostic, not the next nonlinear residual.
        kappa=max(float(r['kappa']) for r in qp[step].values())
        mechanical={}
        for key,r in qp[step].items():
            strain=np.array([float(r['grad_xx']),float(r['grad_yy']),
                             (float(r['grad_xy'])+float(r['grad_yx']))/2])
            crack=np.array([0.,0.,float(r['chi'])*float(r['V'])/2])
            mechanical[key]=tensor(r,'old_')+2*kappa*(strain-crack)
        next_history={k:tensor(r,'old_') for k,r in qp[step+1].items()}
        defect={k:next_history[k]-mechanical[k] for k in mechanical}
        a=weak_load(qp[step],mechanical);b=weak_load(qp[step],defect)
        result[str(step)]['mechanical_weak_load_free_l2']=float(np.linalg.norm(list(a.values())))
        result[str(step)]['transfer_jump_weak_load_free_l2']=float(np.linalg.norm(list(b.values())))
        result[str(step)]['transfer_jump_QP_RMS_by_component']=np.sqrt(np.mean(np.array(list(defect.values()))**2,axis=0)).tolist()
        if step==1:
            cells=defaultdict(list)
            for key,r in qp[step].items():cells[r['cell']].append((r,mechanical[key][2]))
            means=[];first=[]
            for values in cells.values():
                w=np.array([float(r['JxW']) for r,t in values]);y=np.array([float(r['y']) for r,t in values])
                t=np.array([t for r,t in values]);mean=np.average(t,weights=w)
                means.append(mean);first.append(np.average((t-mean)*(y-np.average(y,weights=w))/(1/64),weights=w))
            chosen=max(cells,key=lambda c:np.ptp([t for r,t in cells[c]]))
            result['1']['first_update_cell_moments']=dict(cell_mean_shear_min=min(means),cell_mean_shear_max=max(means),
                max_abs_centered_first_moment_Pa=float(max(abs(np.array(first)))),example_cell=chosen,
                example_QPs=[dict(q=r['q'],x=float(r['x']),y=float(r['y']),tau_xy=t) for r,t in cells[chosen]])
    target=path/'clean_cycle_analysis.json';target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    main(p.parse_args().output)
