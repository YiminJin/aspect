#!/usr/bin/env python3
"""Fixed aligned-mesh work quadratures, exact BE accounting and growth errors."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from assess_native import csv,G,ETA,KF,ELL,M,norm3,gauss
from assess_orientation import geometry,rule
from assess_source_history import basis

BASE=Path('build-particle-free/bulk-energy')
OLD=Path('build-particle-free/bulk-sensitivity/runs/right-distance')
MESH=Path('build-particle-free/bulk-orientation/runs/identity/mesh-qualification.csv')
CASES={'h':OLD,'half':BASE/'runs/half','quarter':BASE/'runs/quarter'}
TIMES=[.05,.10,.1275,.13,.135,.1375,.14,.145,.15,.1525,.16]
RADII=[.225,.2881107642904027,.29851520585223124]
ANGLES=np.arange(-90.,91.)


def layout():
    mesh,nodes,vertices,ids=geometry(OLD,MESH)
    points,weights,N,gradN=rule(vertices,8)
    coarse=np.array([int(c.split('_')[0]) for c in mesh['cell']]);inverse=np.argsort(np.argsort(coarse))
    # DG indices follow active-cell (Morton) order; the separate std::map of
    # local histories uses numeric CellId order. FE_DGQ2 is lexicographic;
    # this ordering is independently checked
    # against the old actual Gauss8 exports below.
    z,w=gauss(8);u,v=np.meshgrid(z,z);qp=np.stack([u.ravel(),v.ravel()],axis=-1)
    a,b=np.meshgrid([0.,.5,1.],[0.,.5,1.]);dg_nodes=np.stack([a.ravel(),b.ravel()],axis=-1)
    V,_=basis(qp,dg_nodes)
    centers=np.concatenate([np.stack([.5+r*np.cos(np.deg2rad(ANGLES)),.5+r*np.sin(np.deg2rad(ANGLES))],axis=-1) for r in RADII])
    tree=cKDTree(points.reshape(-1,2));rows=[];columns=[];values=[]
    for j,(center,indices) in enumerate(zip(centers,tree.query_ball_point(centers,.0625))):
        indices=np.asarray(indices);distance=points.reshape(-1,2)[indices]-center
        k=(1-np.sum(distance*distance,axis=-1)/.0625**2)**3*weights.ravel()[indices];k/=sum(k)
        rows.extend([j]*len(indices));columns.extend(indices);values.extend(k)
    kernel=csr_matrix((values,(rows,columns)),shape=(len(centers),weights.size))
    cap=(points[:,:,0]>.5+1e-12)&(np.linalg.norm(points-.5,axis=-1)<.35)
    node_xy=np.stack([nodes['x'],nodes['y']],axis=-1);cap_nodes=(node_xy[:,0]>.5+1e-12)&(np.linalg.norm(node_xy-.5,axis=-1)<.35)
    return dict(mesh=mesh,nodes=nodes,vertices=vertices,ids=ids,points=points,weights=weights,N=N,gradN=gradN,V=V,inverse=inverse,kernel=kernel,centers=centers,cap=cap,cap_nodes=cap_nodes,node_xy=node_xy)


def phase(path,step):
    p=csv(path/f'phase-{step:04d}.csv');p.sort(order='dof');return p


def qp_state(path,step,L):
    n=len(L['vertices']);qfile=path/f'quadrature-q-{step:04d}.bin'
    if qfile.exists():return np.fromfile(qfile).reshape(n,64),None,None
    raw=np.fromfile(path/f'state-{step:04d}.bin');ndg=27*n
    assert raw[-3]==step and raw[-1]==step
    local=raw[ndg:ndg+5*n*64].reshape(n,64,5)[L['inverse']]
    coefficients=raw[:ndg].reshape(3,n,9).transpose(1,2,0)
    T=np.einsum('qi,cid->cqd',L['V'],coefficients)
    tau=T.copy();tau[:,:,2]-=G*local[:,:,0]
    return local[:,:,3],tau,local[:,:,0]


def physical_fields(phi,q,L):
    nodal=phi[L['ids']];p=np.einsum('qi,ci->cq',L['N'],nodal)
    gradient=np.einsum('cqib,ci->cqb',L['gradN'],nodal)
    h=M*p*(1+p)/(1-p)**2;hp=M*(1+3*p)/(1-p)**3
    a=h*q/G;Y=hp*q*q/(2*G)
    return p,gradient,h,hp,a,Y


def reactions(path,ledger,L):
    times=np.r_[0.,ledger['time']];dt=np.r_[0.,ledger['dt']]
    loading=(times>.05+1e-12)&(times<=.15+1e-12);dg=np.where(loading,.018*dt,0.)
    Q=np.full(len(times),np.nan);endpoint=np.zeros(len(times));trap=np.zeros(len(times));errors=dict(endpoint_work=0.,generalized_work=0.,affine_increment=0.)
    if (path/'boundary-reactions-0000.csv').exists():
        previous=None;dofs=None
        for step in range(len(times)):
            r=csv(path/f'boundary-reactions-{step:04d}.csv');r.sort(order='dof')
            if dofs is None:dofs=r['dof']
            assert np.array_equal(dofs,r['dof'])
            b=np.where(r['component']==0,r['y']-.5,0.)
            errors['affine_increment']=max(errors['affine_increment'],float(max(abs(r['increment']-dg[step]*b))))
            Q[step]=sum(r['force']*b);endpoint[step]=sum(r['force']*r['increment'])
            if previous is not None:trap[step]=sum(.5*(previous+r['force'])*r['increment'])
            previous=r['force'].copy()
        errors['endpoint_work']=float(max(abs(endpoint[1:]-ledger['Wext'])))
        errors['generalized_work']=float(max(abs(endpoint-Q*dg)))
        assert max(errors.values())<1e-10
        origin='direct accepted unconstrained nodal reactions and increments'
    else:
        Q[0]=0.;Q[loading]=ledger['Wext'][loading[1:]]/dg[loading]
        endpoint[1:]=ledger['Wext']
        # The pre-load generalized force is zero; hold forces are recovered
        # independently from stress below, never by dividing zero work by zero.
        Q[times<=.05+1e-12]=0.
        for step in np.flatnonzero(loading):trap[step]=.5*(Q[step-1]+Q[step])*dg[step]
        origin='loading generalized reactions recovered from saved W_BE/dgamma; independent stress check'
    assert abs(sum(dg)-.0018)<1e-14
    return dict(time=times,dt=dt,dgamma=dg,Q=Q,endpoint=endpoint,trap=trap,errors=errors,origin=origin)


def analyse(name,path,L,output):
    ledger=csv(path/'bulk-accepted.csv');nominal={'h':64,'half':128,'quarter':256}[name];expected=len(ledger)
    assert expected>=nominal and abs(ledger['time'][-1]-.16)<1e-15
    if expected>nominal:
        assert np.all(ledger['dt'][nominal:]<1e-12) and np.all(ledger['Wext'][nominal:]==0.)
    assert np.array_equal(ledger['publications'],np.arange(1,expected+1))
    assert max(ledger['mechanical_residual'])<=1e-9 and max(ledger['phase_KKT'])<=1e-7 and max(ledger['projection_error'])<=1e-12
    assert np.all(csv(path/'gates.csv')['passed']==1)
    R=reactions(path,ledger,L);P=[];arcs={};history=[];previous=None;E0=None;first=None
    errors={k:0. for k in ['fracture','cohesive','bulk','bulk_dissipation','cohesive_dissipation','friction_dissipation','traction_loss','phase_remainder','fracture_loss','stress_loss','stored_change','DG_decode','reaction_moment']}
    checks=0;first_nodes=[];previous_tau=None
    for step,time in enumerate(R['time']):
        nodes=phase(path,step);assert np.array_equal(nodes['x'],L['nodes']['x']) and np.array_equal(nodes['y'],L['nodes']['y'])
        if step==0:assert np.array_equal(nodes['phi'],L['nodes']['phi'])
        phi=nodes['phi'];P.append(phi.copy());q,tau,old_a=qp_state(path,step,L)
        p,gradient,h,hp,a,Y=physical_fields(phi,q,L);weight=L['weights']
        Ef=KF*np.sum(weight*(p+ELL**2*np.sum(gradient*gradient,axis=-1)));Ec=np.sum(weight*h*q*q/(2*G))
        if E0 is None:E0=Ef
        delta=phi-L['nodes']['phi'];peak=float(max(delta[L['cap_nodes']]))
        ids=np.flatnonzero(L['cap_nodes']&(delta>1e-12))
        if len(ids) and first is None:
            first=dict(step=step,time=float(time),previous_time=float(R['time'][step-1]))
            first_nodes=[dict(dof=int(i),x=float(nodes['x'][i]),y=float(nodes['y'][i]),angle=float(np.rad2deg(np.arctan2(nodes['y'][i]-.5,nodes['x'][i]-.5)))) for i in ids]
        if tau is not None:
            Eb=np.sum(weight*norm3(tau)/(4*G));Qvol=np.sum(weight*tau[:,:,2])
            if np.isfinite(R['Q'][step]):errors['reaction_moment']=max(errors['reaction_moment'],abs(float(Qvol-R['Q'][step])))
            else:R['Q'][step]=Qvol
            if step in [0,52,60,64]:
                d=csv(path/f'bulk-{step:04d}.csv');coarse=np.array([int(c.split('_')[0]) for c in d['cell']]);d=d[np.argsort(coarse,kind='stable')]
                observed=np.stack([d[k] for k in ['tauxx','tauyy','tauxy']],axis=-1).astype(float).reshape(len(L['vertices']),64,3)[L['inverse']]
                errors['DG_decode']=max(errors['DG_decode'],float(max(np.sqrt(norm3(tau-observed)).ravel())))
                assert errors['DG_decode']<1e-8
                checks+=1
        elif abs(time-.15)<1e-12 or abs(time-.16)<1e-12 or step==0:
            d=csv(path/f'bulk-{step:04d}.csv');tau0=np.stack([d[k] for k in ['tauxx','tauyy','tauxy']],axis=-1).astype(float)
            Eb=np.sum(d['JxW']*norm3(tau0)/(4*G));checks+=1
        else:Eb=ledger['bulk'][step-1]
        if step:
            row=ledger[step-1];old_phi,old_gradient,old_h,old_q,old_a_rebuilt=previous
            assert np.all(phi>=P[-2]) and np.all(phi<1.)
            independent={'fracture':Ef,'cohesive':Ec,'bulk':Eb,
                'bulk_dissipation':R['dt'][step]*2*G/ETA*Eb,
                'cohesive_dissipation':R['dt'][step]*2*G/ETA*Ec,
                'friction_dissipation':1000*np.sum(weight*abs((1+R['dt'][step]*G/ETA)*a-old_a_rebuilt)),
                'traction_loss':np.sum(weight*old_h*(q-old_q)**2/(2*G)),
                'phase_remainder':np.sum(weight*q*q/(2*G)*(h-old_h-hp*(p-old_phi))),
                'fracture_loss':KF*ELL**2*np.sum(weight*np.sum((gradient-old_gradient)**2,axis=-1))}
            names={'fracture':'fracture','cohesive':'cohesive','bulk':'bulk','bulk_dissipation':'Db','cohesive_dissipation':'Dc','friction_dissipation':'Df','traction_loss':'traction_loss','phase_remainder':'phase_remainder','fracture_loss':'fracture_loss'}
            for key,value in independent.items():errors[key]=max(errors[key],abs(float(value-row[names[key]])))
            if tau is not None:errors['stress_loss']=max(errors['stress_loss'],abs(float(np.sum(weight*norm3(tau-previous_tau)/(4*G))-row['stress_loss'])))
            errors['stored_change']=max(errors['stored_change'],abs(float(Eb+Ec+Ef-E0-np.sum(ledger['energy_change'][:step]))))
        previous=(p,gradient,h,q,a);previous_tau=tau
        growth_qp=np.einsum('qi,ci->cq',L['N'],delta[L['ids']])
        history.append([step,time,peak,np.sqrt(np.sum(weight[L['cap']]*growth_qp[L['cap']]**2)),float(sum(delta))])
        for common in TIMES:
            if abs(time-common)<1e-12:
                arcs[str(common)]=np.stack([L['kernel']@p.ravel(),L['kernel']@q.ravel(),L['kernel']@Y.ravel()],axis=-1)
    assert max(v for k,v in errors.items() if k not in ['DG_decode','reaction_moment'])<1e-10
    assert first is not None
    D=ledger['Db']+ledger['Dc']+ledger['Df'];E=ledger['energy_change'];loss=ledger['stress_loss']+ledger['traction_loss']+ledger['phase_remainder']+ledger['fracture_loss'];rw=ledger['phase_work']+ledger['mechanical_work']
    endpoint=R['endpoint'][1:];trap=R['trap'][1:];re=endpoint-E-D;rt=trap-E-D
    assert max(abs(re-loss+rw))<1e-9
    Wscale=sum(abs(trap));assert Wscale>0
    stages={}
    for label,mask in [('initial_hold',ledger['time']<=.05+1e-12),('loading',(ledger['time']>.05+1e-12)&(ledger['time']<=.15+1e-12)),('final_hold',ledger['time']>.15+1e-12)]:
        stages[label]={k:float(sum(v[mask])) for k,v in [('endpoint_work',endpoint),('trapezoidal_work',trap),('storage_change',E),('physical_dissipation',D),('trapezoidal_signed_remainder',rt),('trapezoidal_absolute_remainder',abs(rt))]}
    record=dict(dt=float(ledger['dt'][0]),nominal_steps=nominal,accepted_steps=expected,
        terminal_roundoff_steps=[dict(step=int(r['step']),time=float(r['time']),dt=float(r['dt'])) for r in ledger[nominal:]],first_cap=first,first_nodes=first_nodes,
        final_cap_increment=history[-1][2],final_cap_growth_L2=history[-1][3],initial_fields_bitwise_equal=True,
        endpoint_work=float(sum(endpoint)),trapezoidal_work=float(sum(trap)),work_scale=float(Wscale),
        storage_change=float(sum(E)),physical_dissipation=float(sum(D)),
        final_endpoint_remainder=float(sum(re)),historical_endpoint_relative=float(abs(sum(re))/max(1,abs(sum(endpoint)))),
        final_trapezoidal_remainder=float(sum(rt)),trapezoidal_total_variation=float(sum(abs(rt))),
        maximum_cumulative_trapezoidal_remainder=float(max(abs(np.cumsum(rt)))),
        trapezoidal_variation_relative=float(sum(abs(rt))/Wscale),
        cancellation_fraction=float(1-abs(sum(rt))/sum(abs(rt))),stages=stages,
        independent_errors=errors,boundary_checks=R['errors'],reaction_origin=R['origin'],dense_stress_checks=checks,
        mechanical_residual_max=float(max(ledger['mechanical_residual'])),phase_KKT_max=float(max(ledger['phase_KKT'])),source_max=float(max(ledger['projection_error'])))
    record['numerical_terms_J_per_m']={k:float(sum(ledger[k])) for k in ['stress_loss','traction_loss','phase_remainder','fracture_loss','phase_work','mechanical_work','pressure_work']}
    record['velocity_residual_work_J_per_m']=float(sum(ledger['mechanical_work']-ledger['pressure_work']))
    record['hold_reaction_start_end']=[float(R['Q'][np.argmin(abs(R['time']-t))]) for t in [.15,.16]]
    start=np.argmin(abs(R['time']-.05));stop=np.argmin(abs(R['time']-.15))
    predicted=.5*.018*ledger['dt'][0]*(R['Q'][stop]-R['Q'][start])
    record['endpoint_trapezoidal_difference']=float(sum(endpoint-trap))
    record['constant_increment_telescoping_prediction']=float(predicted)
    record['work_telescoping_defect']=float(abs(sum(endpoint-trap)-predicted))
    assert record['work_telescoping_defect']<1e-10
    record['initial_fracture_storage']=float(E0)
    record['final_storage_changes']={'bulk':float(Eb),'cohesive':float(Ec),'fracture':float(Ef-E0)}
    record['physical_dissipation_components']={k:float(sum(ledger[k])) for k in ['Db','Dc','Df']}
    record['historical_endpoint_screen_passed']=record['historical_endpoint_relative']<=.01
    record['bulk_independent_scope']='every saved baseline state' if path==OLD else 'initial, loading stop and final snapshots'
    if path!=OLD:
        for key in ['stress_loss','DG_decode','reaction_moment']:record['independent_errors'][key]=None
    np.savez_compressed(output/f'{name}-fields.npz',time=R['time'],phase=np.array(P),history=np.array(history),**{'arc_'+k:v for k,v in arcs.items()})
    keys=['time','dt','Q','dgamma'];cols=[R[k][1:] for k in keys]
    for k,v in [('endpoint_work',endpoint),('trapezoidal_work',trap),('storage_change',E),('physical_dissipation',D),('endpoint_remainder',re),('trapezoidal_remainder',rt),('cumulative_endpoint_remainder',np.cumsum(re)),('cumulative_trapezoidal_remainder',np.cumsum(rt)),('cumulative_absolute_trapezoidal_remainder',np.cumsum(abs(rt)))]:keys.append(k);cols.append(v)
    for k in ['bulk','cohesive','fracture','Db','Dc','Df','stress_loss','traction_loss','phase_remainder','fracture_loss','phase_work','mechanical_work','pressure_work']:keys.append(k);cols.append(ledger[k])
    keys.append('velocity_residual_work');cols.append(ledger['mechanical_work']-ledger['pressure_work'])
    np.savetxt(output/f'{name}-energy.csv',np.stack(cols,axis=-1),delimiter=',',header=','.join(keys),comments='')
    (output/f'{name}.json').write_text(json.dumps(record,indent=2)+'\n');return record


def comparisons(output,L,records):
    data={n:dict(np.load(output/f'{n}-fields.npz')) for n in CASES};pairs={}
    for coarse,fine in [('h','half'),('half','quarter'),('h','quarter')]:
        a,b=data[coarse],data[fine];rows=[]
        for time in TIMES:
            ia=np.argmin(abs(a['time']-time));ib=np.argmin(abs(b['time']-time));assert abs(a['time'][ia]-b['time'][ib])<1e-12
            pa=a['phase'][ia]-L['nodes']['phi'];pb=b['phase'][ib]-L['nodes']['phi']
            ga=np.einsum('qi,ci->cq',L['N'],pa[L['ids']]);gb=np.einsum('qi,ci->cq',L['N'],pb[L['ids']]);w=L['weights'][L['cap']]
            denominator=np.sqrt(sum(w*gb[L['cap']]**2));peak=max(pb[L['cap_nodes']])
            row=dict(time=time,growth_L2_relative=float(np.sqrt(sum(w*(ga-gb)[L['cap']]**2))/denominator) if denominator>1e-14 else None,
                growth_Linf_relative=float(max(abs(pa-pb)[L['cap_nodes']])/peak) if peak>1e-12 else None,
                peak_relative=float(abs(max(pa[L['cap_nodes']])-peak)/peak) if peak>1e-12 else None,
                phase_difference_Linf=float(max(abs(pa-pb)[L['cap_nodes']])) )
            aa=a['arc_'+str(time)][:,2];bb=b['arc_'+str(time)][:,2]
            row['Y_relative_L2']=float(np.linalg.norm(aa-bb)/np.linalg.norm(bb)) if np.linalg.norm(bb)>1e-12 else None
            rows.append(row)
        pairs[coarse+'_'+fine]=rows
    orders={}
    for key in ['endpoint_work','trapezoidal_work','storage_change','physical_dissipation','final_cap_increment','final_cap_growth_L2']:
        x=[records[n][key] for n in ['h','half','quarter']];d1=x[0]-x[1];d2=x[1]-x[2]
        order=float(np.log2(abs(d1/d2))) if d1*d2>0 else None
        orders[key]=dict(values=x,first_difference=d1,second_difference=d2,observed_order=order,
            extrapolated_finest_uncertainty=abs(d2)/(2**order-1) if order is not None and order>0 else None)
    temporal={}
    for coarse,fine in [('h','half'),('half','quarter')]:
        a=csv(output/f'{coarse}-energy.csv');b=csv(output/f'{fine}-energy.csv')
        indices=np.array([np.flatnonzero(abs(b['time']-t)<1e-12)[-1] for t in a['time']])
        assert max(abs(a['time']-b['time'][indices]))<1e-12
        components={k:abs(np.cumsum(a[k])-np.cumsum(b[k])[indices]) for k in ['trapezoidal_work','storage_change','physical_dissipation']}
        temporal[coarse+'_'+fine]=dict(maximum_component_differences={k:float(max(v)) for k,v in components.items()},
            maximum_sum_component_difference_relative=float(max(sum(components.values()))/records[fine]['work_scale']))
    return dict(growth_normalization='delta_phi relative to identical initial phase; physical cap Q1 L2 / peak',pairs=pairs,orders=orders,temporal_budgets=temporal,
        strict_reference_match_passed=False,physical_energy_criterion_is_postprocessing=True)


def recommended_criterion(result):
    fine=result['cases']['quarter'];comparison=result['comparisons']
    balance=fine['trapezoidal_variation_relative']
    temporal=comparison['temporal_budgets']['half_quarter']['maximum_sum_component_difference_relative']
    rows=[r for r in comparison['pairs']['half_quarter'] if r['growth_L2_relative'] is not None]
    l2=max(r['growth_L2_relative'] for r in rows);linf=max(r['growth_Linf_relative'] for r in rows)
    return dict(work_normalization='sum absolute trapezoidal work over this event; initial storage excluded',
        balance_total_variation=balance,temporal_component_Cauchy_budget=temporal,
        combined_energy_budget=balance+temporal,recommended_energy_tolerance=.01,
        growth_L2_maximum=l2,growth_Linf_maximum=linf,recommended_growth_tolerance=.05,
        energy_budget_passed=bool(balance+temporal<=.01),growth_budget_passed=bool(max(l2,linf)<=.05),
        native_acceptance_rules_changed=False,
        onset_common_bracket=[max(v['first_cap']['previous_time'] for v in result['cases'].values()),
                              min(v['first_cap']['time'] for v in result['cases'].values())],
        qualification='bounded practical work/temporal accuracy on this fixed mesh; no spatial or propagation claim')


def main(output):
    output.mkdir(parents=True,exist_ok=True);L=layout();records={}
    for name,path in CASES.items():
        print('Assessing',name,flush=True);records[name]=analyse(name,path,L,output)
    result=dict(cases=records,comparisons=comparisons(output,L,records));result['recommended_criterion']=recommended_criterion(result)
    (output/'assessment.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(output/'geometry.npz',xy=L['node_xy'],indices=L['ids'],vertices=L['vertices'],cap_nodes=L['cap_nodes'],centers=L['centers'],initial_phase=L['nodes']['phi'])
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);main(p.parse_args().output)
