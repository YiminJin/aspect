"""Complete production-quadrature aging commutator and its one disposable solve."""
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from analyze_theta_interpolation import column, mu, read, update, velocity

HERE=Path(__file__).resolve().parent
SAVED=HERE/'fault-grid-50-local4'
ROOT=HERE/'theta-exact-local4'
OUT=ROOT/'conditional'


def write(name, records):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]))
        w.writeheader();w.writerows(records)


def samples(prefix):
    return np.concatenate([np.genfromtxt(OUT/f'{prefix}_rank{r}.csv',delimiter=',',names=True,
                                       ndmin=1) for r in range(4)])


def main():
    execution=json.loads((OUT/'execution.json').read_text())
    assert execution['verified'] and execution['original_checkpoint_unchanged']
    assert execution['copied_checkpoint_unchanged'] and not execution['accepted_output_written']
    f11,f12,f13=[read(SAVED/f'fault_{k}.csv') for k in (11,12,13)]
    weak=read(OUT/'exact_weak.csv')
    A,B=samples('frozen_A'),samples('frozen_B')
    assert len(A)==len(B)>0
    for key in ['particle','fault','segment','xi','weight','V','p','sigma_n','q','C','phi','chi']:
        assert np.array_equal(A[key],B[key]),key
    j=A['segment'].astype(int);z=A['xi'];w=A['weight']
    assert set(j)=={795,796} and np.all(w>0) and np.all((z>=0)&(z<=1))
    t11,t12=column(f11,'Theta'),column(f12,'Theta')
    v12,v13=column(f12,'V'),column(f13,'V')
    dt=float(f12[0]['dt'])
    thetaA=(1-z)*t12[j]+z*t12[j+1]
    thetaB=update((1-z)*t11[j]+z*t11[j+1],(1-z)*v12[j]+z*v12[j+1],dt)
    errors={}
    # C++ may fuse the affine multiply/add; compare on the nodal rate scale,
    # not relatively to a nearly cancelled quadrature-point rate.
    errors['Q1_rate_absolute']=float(np.max(np.abs(velocity(v13,j,z)-A['V'])))
    assert errors['Q1_rate_absolute']<4*np.finfo(float).eps*np.max(v13[j])
    for label,data,theta in [('A',A,thetaA),('B',B,thetaB)]:
        err=float(np.max(np.abs(mu(data['V'],theta)-data['mu'])))
        errors[label+'_mu_absolute']=err
        assert err<1e-14
        assert np.max(np.abs(data['friction']-data['mu']*data['sigma_n']))<1e-7
    node_load=[]
    for i in (795,796,797):
        shape=np.where(j==i,1-z,np.where(j+1==i,z,0.))
        row=weak[i];mass=float(row['weight'])
        a=float(np.dot(w*shape,A['friction']));b=float(np.dot(w*shape,B['friction']))
        delta=b-a
        expected_delta=float(row['B_friction'])-float(row['A_friction'])
        assert abs(delta-expected_delta)<1e-12*abs(float(row['A_friction']))
        if i==796:
            assert abs(a/float(row['A_friction'])-1)<1e-13
            assert abs(b/float(row['B_friction'])-1)<1e-13
        node_load.append(dict(node=i,xd_m=float(f13[i]['xd']),full_test_weight_m2=mass,
            patch_A_friction_N=a,patch_B_friction_N=b,delta_friction_N=delta,
            A_friction_Pa=float(row['A_friction'])/mass,B_friction_Pa=float(row['B_friction'])/mass,
            A_R_Pa=float(row['A_R'])/mass,B_R_Pa=float(row['B_R'])/mass,
            delta_friction_Pa=delta/mass))
    write('analysis_exact_load.csv',node_load)
    untouched=[i for i in range(len(weak)) if i not in (795,796,797)]
    assert all(weak[i]['A_friction']==weak[i]['B_friction'] and weak[i]['A_R']==weak[i]['B_R'] for i in untouched)
    point_records=[]
    for n in range(len(A)):
        point_records.append(dict(particle=int(A['particle'][n]),segment=int(j[n]),xi=z[n],
            weight_m2=w[n],xd_m=(1-z[n])*float(f13[j[n]]['xd'])+z[n]*float(f13[j[n]+1]['xd']),
            V13=A['V'][n],Theta12_A=thetaA[n],Theta12_B=thetaB[n],mu_A=A['mu'][n],mu_B=B['mu'][n],
            sigma_n_Pa=A['sigma_n'][n],delta_friction_Pa=B['friction'][n]-A['friction'][n]))
    write('analysis_complete_state_samples.csv',point_records)
    result=dict(execution_seconds=execution['seconds'],production_QP_count=len(A),
        particle_count=len(set(A['particle'])),segments={str(k):dict(qps=int(np.sum(j==k)),weight_m2=float(np.sum(w[j==k]))) for k in (795,796)},
        independent_friction_errors=errors,exact_loads=node_load,
        theta_A_over_B_range=[float(np.min(thetaA/thetaB)),float(np.max(thetaA/thetaB))])
    log=(OUT/'run.log').read_text()
    match=re.search(r'physical-term relative error=([^\s]+)',log)
    result['saved_load_reproduction_relative_error']=float(match[1])
    if 'Noncommitting fault diagnostic converged:' in log:
        final=read(OUT/'noncommitting_surface.csv');S=samples('solve')
        for key in ['particle','fault','segment','xi','weight','phi','chi']:
            assert np.array_equal(A[key],S[key]),key
        assert np.max(np.abs(mu(S['V'],thetaB)-S['mu']))<1e-14
        result['raw_max_absolute_change_Pa']={key:float(np.max(np.abs(S[key]-A[key])))
                                             for key in ['p','tau_N','sigma_n','q']}
        oldparts=[read(SAVED/f'history_surface_step13_rank{r}.csv') for r in range(4)]
        newparts=[read(OUT/f'history_surface_rank{r}.csv') for r in range(4)]
        summed=lambda parts,i,key:sum(float(p[i][key]) for p in parts)
        nodes=[]
        for i in range(792,804):
            mass=float(weak[i]['weight']);new=final[i]
            r=dict(node=i,xd_m=float(f13[i]['xd']),V_A=float(f13[i]['V']),V_solve=float(new['V']),
                prescribed=int(new['prescribed']),lower_active_A=int(float(f13[i]['V'])==1e-20 and not int(new['prescribed'])),
                lower_active_solve=int(new['lower_active']))
            for label,parts in [('A',oldparts),('solve',newparts)]:
                assert abs(summed(parts,i,'weight')/mass-1)<1e-13
                for key in ['p','particle_q','particle_sigma','particle_C','particle_friction','particle_damping','particle_R']:
                    r[label+'_'+key+'_Pa']=summed(parts,i,key)/mass
                r[label+'_minus_tauN_Pa']=r[label+'_particle_sigma_Pa']-50e6-r[label+'_p_Pa']
            nodes.append(r)
        write('analysis_neighbouring_nodes.csv',nodes)
        extrema=[]
        for label,data in [('A',A),('solve',S)]:
            xd=(1-z)*column(f13,'xd')[j]+z*column(f13,'xd')[j+1]
            for field in ['p','tau_N','sigma_n','q']:
                lo,hi=np.argmin(data[field]),np.argmax(data[field])
                extrema.append(dict(state=label,field=field,minimum_Pa=data[field][lo],maximum_Pa=data[field][hi],
                    min_xd_m=xd[lo],min_parent_x=data['parent_x'][lo],min_parent_y=data['parent_y'][lo],
                    min_particle=int(data['particle'][lo]),min_segment=int(j[lo]),min_xi=z[lo],
                    min_delta_p_Pa=data['p'][lo],min_minus_tauN_Pa=-data['tau_N'][lo],
                    weighted_mean_Pa=float(np.dot(w,data[field])/sum(w))))
            result[label+'_tensile_weight_m2']=float(np.sum(w[data['sigma_n']<0]))
        write('analysis_raw_extrema.csv',extrema)
        result['raw_extrema']=extrema
        result['nodes']=nodes
        result['lower_active_count_A']=int(np.sum((column(f13,'V')==1e-20)&(column(f13,'prescribed')==0)))
        result['lower_active_count_solve']=sum(int(r['lower_active']) for r in final)
        result['derivative_checks']=read(OUT/'derivative_checks.csv')
        linear=[(float(a),float(b),float(c)) for a,b,c in re.findall(
            r'Fault linear solve: iterations=\d+, estimated=([^,]+), fresh=([^,]+), target=([^,]+)',log)]
        assert linear and all(fresh<=target for estimated,fresh,target in linear)
        result['fresh_linear_checks']=len(linear)
        result['max_fresh_linear_over_target']=max(fresh/target for estimated,fresh,target in linear)
        result['convergence_line']=next(line.strip() for line in log.splitlines() if 'Noncommitting fault diagnostic converged:' in line)
        assert 'BP3 noncommitting rollback verified:' in log
        retained=read(OUT/'noncommitting_history.csv')
        assert np.array_equal(column(retained,'Theta_retained'),t12)
        assert np.array_equal(column(retained,'V_committed'),v12)
        result['rollback_verified']=True
    result['analysis_source_sha256']={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__),HERE/'analyze_theta_interpolation.py',HERE/'run_theta_exact.py']}
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
