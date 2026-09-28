"""Offline cell-level audit of saved native stress-cycle tensors (no solves)."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from analyze_stress_cycle import rows, TRANSFER_COLUMNS, COMPONENTS
from stage_normal_stress_diagnostic import parameters


def tensor(r,prefix):
    return np.array([float(r[prefix+c]) for c in COMPONENTS])


def stats(v,w=None):
    v=np.asarray(v,dtype=float)
    assert len(v) and np.isfinite(v).all()
    w=np.ones(len(v)) if w is None else np.asarray(w)
    mean=np.average(v,weights=w)
    return dict(mean=float(mean),rms=float(np.sqrt(np.average(v*v,weights=w))),
                centered_rms=float(np.sqrt(np.average((v-mean)**2,weights=w))),
                minimum=float(v.min()),maximum=float(v.max()))


def q2(points,support):
    # Recover the exact represented tensor-product Q2 polynomial from its nine
    # exported native Gauss values. This is not smoothing or a new profile.
    points=np.asarray(points);support=np.asarray(support)
    def l(x,a):
        return 2*(x-.5)*(x-1) if a==0 else (4*x*(1-x) if a==.5 else 2*x*(x-.5))
    assert set(np.unique(support)).issubset({0.,.5,1.})
    return np.array([[l(p[0],s[0])*l(p[1],s[1]) for s in support] for p in points])


def main(root):
    out=root/'output-full'
    qp=rows(out,'normal_qp_*_rank*.csv')
    parents=rows(out,'stress_particles_before_rank*.csv')
    update={r['particle_id']:r for r in rows(out,'stress_update_*_rank*.csv')}
    before={r['id']:r for r in parents}
    transfer=rows(out,'stress_transfer_*_rank*.csv',TRANSFER_COLUMNS)
    proposals=defaultdict(dict);published=defaultdict(dict);dofs=defaultdict(list)
    for r in transfer:
        field=r['field'].split('(')[-1].rstrip(')')
        if field not in ['tau_'+c for c in COMPONENTS]:continue
        key=(r['cell'],field)
        if r['stage']=='support_proposal':
            proposals[key][int(r['index'])]=r
            dofs[(field,r['dof'])].append(float(r['value']))
        else:published[key][int(r['index'])]=r
    cell_data={};nodal=[];qp_error=[]
    for cell,field in proposals:
        s=[proposals[(cell,field)][i] for i in range(9)]
        p=[published[(cell,field)][i] for i in range(9)]
        sr=[[float(r['ref_x']),float(r['ref_y'])] for r in s]
        pr=[[float(r['ref_x']),float(r['ref_y'])] for r in p]
        xy=np.array([[float(r['x']),float(r['y'])] for r in s])
        nodal_values=np.linalg.solve(q2(pr,sr),[float(r['value']) for r in p])
        qp_error.extend(q2(pr,sr)@nodal_values-np.array([float(r['value']) for r in p]))
        cell_data[(cell,field)]=(sr,nodal_values,xy.min(axis=0),xy.max(axis=0))
        for i,r in enumerate(s):
            # Only compare complete incident-cell sets in uniform patches;
            # patch-edge nodes with missing proposals must not be averaged anew.
            expected=2**sum(x in (0.,1.) for x in sr[i])
            vals=dofs[(field,r['dof'])]
            if len(vals)==expected:
                nodal.append(nodal_values[i]-np.mean(vals))
    # The observer saved full neighboring stencils. Deduplicate ghosts by ID,
    # then independently recompute DWA at the exact exported support points.
    neighbours={}
    for r in rows(out,'normal_incoming_particles_*_rank*.csv'):
        if r['particle_id'] in neighbours:
            old=neighbours[r['particle_id']]
            assert all(float(r[k])==float(old[k]) for k in ('x','y','tau_xx','tau_yy','tau_xy'))
        neighbours[r['particle_id']]=r
    nx=np.array([[float(r['x']),float(r['y'])] for r in neighbours.values()])
    nt=np.array([tensor(r,'tau_') for r in neighbours.values()])
    dwa_error=[];support_trace=[]
    for cell in sorted({k[0] for k in proposals}):
        _,_,lo,hi=cell_data[(cell,'tau_xx')]
        radius=np.linalg.norm(hi-lo)/2
        for i in range(9):
            r=proposals[(cell,'tau_xx')][i]
            pos=np.array([float(r['x']),float(r['y'])]);dist=np.linalg.norm(nx-pos,axis=1)
            w=np.maximum(1-dist/radius,0.);assert sum(w)>0
            recomputed=w@nt/sum(w)
            actual=np.array([float(proposals[(cell,'tau_'+c)][i]['value']) for c in COMPONENTS])
            dwa_error.extend(recomputed-actual)
            sr,_,_,_=cell_data[(cell,'tau_xx')]
            entry=dict(cell=cell,support=i,x=pos[0],y=pos[1],ref_x=sr[i][0],ref_y=sr[i][1],radius=radius)
            for j,c in enumerate(COMPONENTS):
                entry['proposal_'+c]=actual[j]
                entry['recomputed_'+c]=recomputed[j]
                entry['assembled_'+c]=cell_data[(cell,'tau_'+c)][1][i]
                entry['component_'+c]=int(proposals[(cell,'tau_'+c)][i]['component'])
                entry['property_'+c]=int(proposals[(cell,'tau_'+c)][i]['property'])
            support_trace.append(entry)
    with (root/'cell_support_trace.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(support_trace[0]));writer.writeheader();writer.writerows(support_trace)
    geometry=rows(out,'normal_restored_fault.csv')
    origin=np.array([float(geometry[0]['x']),float(geometry[0]['y'])])
    tangent=np.array([float(geometry[1]['x']),float(geometry[1]['y'])])-origin
    tangent/=np.linalg.norm(tangent);normal=np.array([-tangent[1],tangent[0]])
    N=-np.array([normal[0]**2,normal[1]**2,2*normal[0]*normal[1]])
    trace=[]
    for r in parents:
        cell=r['cell'];xy=np.array([float(r['x']),float(r['y'])]);unit=[[float(r['ref_x']),float(r['ref_y'])]]
        f=[]
        for c in COMPONENTS:
            sr,values,lo,hi=cell_data[(cell,'tau_'+c)]
            assert np.max(abs((xy-lo)/(hi-lo)-unit[0]))<1e-9
            f.append((q2(unit,sr)@values)[0])
        old=tensor(r,'tau_');new=tensor(update[r['id']],'new_');f=np.array(f)
        entry=dict(cell=cell,id=r['id'],xd=(100000-xy[1])/.86602540378443864676,
                   x=xy[0],y=xy[1],r=(xy-origin)@normal,ref_x=unit[0][0],ref_y=unit[0][1],
                   d_particle=old@N,d_FE_at_parent=f@N,d_after=new@N,
                   d_transfer_difference=(f-old)@N,d_particle_update=(new-old)@N)
        for j,c in enumerate(COMPONENTS):
            entry['particle_'+c]=old[j];entry['FE_at_parent_'+c]=f[j];entry['after_'+c]=new[j]
        trace.append(entry)
    with (root/'cell_parent_trace.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(trace[0]));writer.writeheader();writer.writerows(trace)

    result=dict(q2_reconstruction_error_max_Pa=float(max(abs(np.array(qp_error)))),
                complete_incident_nodes=len(nodal),incident_mean_error_max_Pa=float(max(abs(np.array(nodal)))),
                support_DWA_recompute_error_max_Pa=float(max(abs(np.array(dwa_error)))),windows={})
    result['component_and_property_indices']={c:sorted({(r['component_'+c],r['property_'+c]) for r in support_trace})
                                               for c in COMPONENTS}
    result['point_stress_sum_closure_max_Pa']=float(max(np.max(abs(tensor(r,'tau_')-
        tensor(r,'history_')-tensor(r,'strain_')-tensor(r,'slip_'))) for r in qp))
    result['inherited_coefficient_closure_max_Pa']=float(max(np.max(abs(tensor(r,'history_')-
        float(r['beta'])*tensor(r,'incoming_FE_'))) for r in qp))
    selected=[r for r in qp if (r['cell'],'tau_xx') in published]
    with (root/'cell_QP_trace.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(selected[0]));writer.writeheader();writer.writerows(selected)
    for name,lo,hi in [('70-71',70000,71000),('79-80',79000,80000)]:
        p=[r for r in trace if lo<r['xd']<hi]
        q=[r for r in qp if lo<float(r['down_dip_s_m'])<hi]
        w=[float(r['work_weight']) for r in q]
        info=dict(parents=len(p),cells=len({r['cell'] for r in p}),
                  parent_equal_weight={key:stats([r[key] for r in p]) for key in
                    ('d_particle','d_FE_at_parent','d_after','d_transfer_difference','d_particle_update')})
        info['QP_work_weighted']={key:stats([float(r[key]) for r in q],w)
                                 for key in ('d_history','d_update','p','minus_n_tau_n')}
        for j,c in enumerate(COMPONENTS):
            info['QP_work_weighted']['FE_minus_direct_particle_'+c]=stats(
                [float(r['incoming_FE_'+c])-float(r['particle_interp_'+c]) for r in q],w)
        info['QP_work_weighted']['FE_minus_direct_particle_normal']=stats(
            [(tensor(r,'incoming_FE_')-tensor(r,'particle_interp_'))@N for r in q],w)
        # Within-cell variation is reported separately from broad along-fault
        # variation. All values here remain at the same parent locations.
        groups=defaultdict(list)
        for r in p:groups[r['cell']].append(r)
        info['within_cell_equal_weight']={}
        for key in ('d_particle','d_FE_at_parent','d_after','d_particle_update'):
            deviations=[r[key]-np.mean([t[key] for t in values]) for values in groups.values() for r in values]
            info['within_cell_equal_weight'][key]=stats(deviations)
        unresolved=np.array([r['d_particle']-r['d_FE_at_parent'] for r in p])
        increment=np.array([r['d_particle_update'] for r in p])
        info['unresolved_history_update_correlation']=float(np.corrcoef(unresolved,increment)[0,1])
        info['update_projection_onto_unresolved_history_Pa']=float(np.dot(unresolved,increment)/
                                                                  np.linalg.norm(unresolved)/np.sqrt(len(p)))
        # A short, identifiable nine-parent cell example in each window.
        example=min(groups,key=lambda k:abs(np.mean([r['r'] for r in groups[k]]))+
                    abs(np.mean([r['xd'] for r in groups[k]])-(lo+hi)/2))
        info['example_cell']=example
        info['example_parents']=sorted(groups[example],key=lambda r:(r['ref_y'],r['ref_x']))
        # Within each actual particle row, remove its physical-coordinate chord.
        # This quantifies cell-scale curvature, not an assumed analytic solution.
        chords={key:[] for key in ('particle_xx','particle_yy','particle_xy','d_particle','d_FE_at_parent')}
        for values in groups.values():
            assert len(values)==9
            order=sorted(values,key=lambda r:r['ref_y'])
            for offset in (0,3,6):
                left,mid,right=sorted(order[offset:offset+3],key=lambda r:r['ref_x'])
                xi=(mid['ref_x']-left['ref_x'])/(right['ref_x']-left['ref_x'])
                for key in chords:chords[key].append(mid[key]-((1-xi)*left[key]+xi*right[key]))
        info['parent_row_chord_defect']={k:stats(v) for k,v in chords.items()}
        info['center_strip_QP_groups']=[]
        for rx in sorted({float(r['ref_x']) for r in q}):
            samples=[r for r in q if float(r['ref_x'])==rx and abs(float(r['r_m']))<=2.]
            if samples:
                weights=[float(r['work_weight']) for r in samples]
                info['center_strip_QP_groups'].append(dict(ref_x=rx,count=len(samples),
                    r=stats([float(r['r_m']) for r in samples],weights),
                    history=stats([float(r['d_history']) for r in samples],weights),
                    update=stats([float(r['d_update']) for r in samples],weights)))
        # Native QP reference-coordinate groups: report r as well, rather than
        # mislabeling transverse-profile variation as a two-band interpolation error.
        bands=defaultdict(list)
        for r in q:bands[(float(r['ref_x']),float(r['ref_y']))].append(r)
        info['native_QP_groups']=[]
        for (rx,ry),samples in sorted(bands.items()):
            weights=[float(r['work_weight']) for r in samples]
            info['native_QP_groups'].append(dict(ref_x=rx,ref_y=ry,count=len(samples),
                r=stats([float(r['r_m']) for r in samples],weights),
                inherited=stats([float(r['d_history']) for r in samples],weights),
                update=stats([float(r['d_update']) for r in samples],weights)))
        result['windows'][name]=info

    base={(r['cell'],r['qp']):r for r in qp}
    previous=base
    result['timestep_changes']={}
    result['normal_strain_RMS']={}
    for name in ('full','half','quarter'):
        result['normal_strain_RMS'][name]={}
        samples=rows(root/('output-'+name),'normal_qp_*_rank*.csv')
        for window,lo,hi in [('70-71',70000,71000),('79-80',79000,80000)]:
            values=[r for r in samples if lo<float(r['down_dip_s_m'])<hi]
            w=[float(r['work_weight']) for r in values]
            eps=np.array([float(r['n_x'])**2*float(r['grad_xx'])+
                          float(r['n_y'])**2*float(r['grad_yy'])+
                          float(r['n_x'])*float(r['n_y'])*(float(r['grad_xy'])+float(r['grad_yx'])) for r in values])
            closure=max(abs(float(r['d_update'])+2*float(r['kappa'])*e) for r,e in zip(values,eps))
            dt=float(values[0]['stress_dt'])
            result['normal_strain_RMS'][name][window]=dict(eps_nn=stats(eps,w),dt_eps_nn=stats(eps*dt,w),
                beta=sorted({float(r['beta']) for r in values}),kappa=sorted({float(r['kappa']) for r in values}),
                update_from_gradient_closure_max_Pa=closure)
    result['parameter_differences']={}
    original=parameters((out/'parameters.prm').read_text())
    for name in ('half','quarter'):
        config=parameters((root/('output-'+name)/'parameters.prm').read_text())
        result['parameter_differences'][name]={'/'.join(k):[original.get(k),config.get(k)]
                                              for k in original.keys()|config.keys() if original.get(k)!=config.get(k)}
        new={(r['cell'],r['qp']):r for r in rows(root/('output-'+name),'normal_qp_*_rank*.csv')}
        assert new.keys()==base.keys()
        result['timestep_changes'][name]={}
        for window,lo,hi in [('70-71',70000,71000),('79-80',79000,80000)]:
            keys=[k for k,r in base.items() if lo<float(r['down_dip_s_m'])<hi]
            w=[float(base[k]['work_weight']) for k in keys]
            changes={}
            for field in ('d_update','p','minus_n_tau_n','perturbation_normal'):
                changes[field+'_from_full']=stats([float(new[k][field])-float(base[k][field]) for k in keys],w)
                changes[field+'_from_previous']=stats([float(new[k][field])-float(previous[k][field]) for k in keys],w)
            for field in ('grad_xx','grad_xy','grad_yx','grad_yy'):
                changes['dt_'+field+'_from_full']=stats([float(new[k]['stress_dt'])*float(new[k][field])-
                    float(base[k]['stress_dt'])*float(base[k][field]) for k in keys],w)
            changes['beta']=sorted({float(new[k]['beta']) for k in keys})
            changes['kappa']=sorted({float(new[k]['kappa']) for k in keys})
            result['timestep_changes'][name][window]=changes
        previous=new
    (root/'cell_cycle_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='timestep_changes'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    main(parser.parse_args().directory)
