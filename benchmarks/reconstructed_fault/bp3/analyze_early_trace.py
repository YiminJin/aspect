"""First-dip timing and one frozen early independent-trace comparison."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/aspect-trace-mpl')
import numpy as np
from analyze_uniform_sliding import cat, read, records, select, write
from analyze_free_trace import weak

HERE=Path(__file__).resolve().parent
BASE=HERE/'work-replay-50-local4'
ROOT=HERE/'early-free-trace'
VP=1e-9
G=32038120320.
ETA=1e26
S=np.array([-np.sqrt(3)/4, np.sqrt(3)/4, -.25])
N=np.array([.75, .25, -np.sqrt(3)/4])


def tensor_contract(t, basis):
    return t[0]*basis[0]+t[1]*basis[1]+2*t[2]*basis[2]


def friction(v, theta):
    # Independent evaluation; this window is wholly on the a=.025 plateau.
    return .025*np.arcsinh(v/(2e-6)*np.exp((.6+.015*np.log(1e-6*theta/.008))/.025))


def window(path, k):
    raw=cat(path.glob(f'work_qp_{k}_rank*.csv'), ('cell',))
    raw=select(raw,(raw['source_active']==1)&(raw['chi']>0)&(raw['xd']>=37000)&(raw['xd']<=43000))
    raw['weight']=raw['JxW']*raw['chi'];raw['shape1']=raw['xi']
    return raw


def retained(raw, dt):
    kappa=-ETA*np.expm1(-G*dt/ETA)
    tau=np.array([raw['tau_xx'],raw['tau_yy'],raw['tau_xy']])
    eps=np.array([raw['eps_xx'],raw['eps_yy'],raw['eps_xy']])
    return tau-2*kappa*(eps-S[:,None]*raw['chi']*raw['V'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prefix','probe'])
    parser.add_argument('--directory',default='early-free-trace-matched')
    args=parser.parse_args()
    global ROOT
    ROOT=HERE/args.directory
    out=ROOT/'analysis';out.mkdir(exist_ok=True)
    if args.action=='prefix':
        checks=[]
        for k in (0,1):
            a,b=[read(p/f'fault_{k}.csv') for p in (BASE,ROOT/'prefix')]
            for key in ('x','y','time','dt'):
                np.testing.assert_array_equal(a[key],b[key])
            for key in ('V','Theta','Ih','slip'):
                error=float(max(abs(a[key]-b[key]))/max(1e-30,max(abs(a[key]))))
                assert error<1e-11,(k,key,error)
                checks.append(dict(step=k,field=key,relative_error=error))
            for rank in range(4):
                pa,pb=[read(p/f'mature_history_{k}_rank{rank}.csv') for p in (BASE,ROOT/'prefix')]
                np.testing.assert_array_equal(pa['id'],pb['id'])
                for key in ('H_inert','tau_xx','tau_yy','tau_xy'):
                    np.testing.assert_allclose(pa[key],pb[key],rtol=1e-10,atol=1e-6)
                    checks.append(dict(step=k,field=f'rank{rank}_{key}',relative_error=float(max(abs(pa[key]-pb[key]))/max(1.,max(abs(pa[key]))))))
            qa,qb=window(BASE,k),window(ROOT/'prefix',k)
            for key in qa:
                np.testing.assert_allclose(qa[key],qb[key],rtol=1e-10,atol=1e-6)
                if key in ('phi','Ih','chi','p','tau_xx','tau_yy','tau_xy','eps_xx','eps_yy','eps_xy'):
                    checks.append(dict(step=k,field='QP_'+key,
                                       relative_error=float(max(abs(qa[key]-qb[key])))/max(1e-30,max(abs(qa[key])))))
        (ROOT/'prefix/comparison.json').write_text(json.dumps(dict(passed=True,checks=checks),indent=2)+'\n')
        rows=[];old_profiles=[]
        for k in range(5):
            f=read(BASE/f'fault_{k}.csv');old=read(BASE/f'fault_{max(0,k-1)}.csv')
            raw=window(BASE,k);j=raw['segment'].astype(int);xi=raw['xi'];n=len(f['V'])
            dt=f['dt'][0] if k else 4e6
            frozen=retained(raw,dt)
            old_q=tensor_contract(frozen,S);old_minus_n=-tensor_contract(frozen,N)
            theta=(1-xi)*old['Theta'][j]+xi*old['Theta'][j+1]
            mu=friction(raw['V'],theta)
            mass=weak(raw,np.ones(len(j)),n)
            old_load=weak(raw,old_q-mu*old_minus_n,n)
            old_shear=weak(raw,old_q,n)
            old_normal=weak(raw,old_minus_n,n)
            chord=old['Theta'][797]+(raw['xd']-39900.)/100.*(old['Theta'][795]-old['Theta'][797])
            change=np.where((j==795)|(j==796),raw['sigma_n']*(mu-friction(raw['V'],np.maximum(chord,1.))),0.)
            theta_load=weak(raw,change,n)
            q_mean=weak(raw,raw['q'],n)/np.where(mass>0,mass,1.)
            sigma_mean=weak(raw,raw['sigma_n'],n)/np.where(mass>0,mass,1.)
            residual=weak(raw,raw['q']-mu*raw['sigma_n']-4624440.*raw['V'],n)
            assert abs(residual[796]/mass[796])<.01
            v=f['V'][[797,796,795]]/VP
            rows.append(dict(step=k,time=f['time'][0],chord_defect=.5*(v[0]+v[2])-v[1],
                             V39900=v[0],V39950=v[1],V40000=v[2],
                             incoming_chord_defect=(.5*(old['V'][797]+old['V'][795])-old['V'][796])/VP if k else 0.,
                             incoming_theta_peak=old['Theta'][796]-.5*(old['Theta'][797]+old['Theta'][795]),
                             old_shear_mean=old_shear[796]/mass[796],old_minus_normal_mean=old_normal[796]/mass[796],
                             old_residual_mean=old_load[796]/mass[796],theta_peak_friction_mean=theta_load[796]/mass[796],
                             old_residual_chord=(old_load[796]/mass[796]-.5*(old_load[795]/mass[795]+old_load[797]/mass[797])),
                             current_q_mean=q_mean[796],current_q_chord=q_mean[796]-.5*(q_mean[795]+q_mean[797]),
                             current_sigma_mean=sigma_mean[796],current_sigma_chord=sigma_mean[796]-.5*(sigma_mean[795]+sigma_mean[797]),
                             reconstructed_residual_mean=residual[796]/mass[796],
                             sigma_min=float(min(raw['sigma_n'])),sigma_max=float(max(raw['sigma_n'])),
                             min_free_V_over_Vp=float(min(f['V'][f['xd']<40000-1e-6])/VP)))
            for i in range(791,805):
                old_profiles.append(dict(step=k,node=i,xd=f['xd'][i],old_shear=old_shear[i]/mass[i],
                                         old_minus_normal=old_normal[i]/mass[i],old_residual=old_load[i]/mass[i]))
        records(out/'onset.csv',rows);records(out/'old_stress_profile.csv',old_profiles)
        print(json.dumps(rows,indent=2))
        return

    execution=json.loads((ROOT/'probe/execution.json').read_text())
    assert execution['converged'] and execution['rollback'] and execution['fresh_linear_passed']
    f=read(BASE/'fault_2.csv');old=read(ROOT/'prefix/fault_1.csv')
    result=read(ROOT/'probe/noncommitting_surface.csv')
    raw=cat((ROOT/'probe').glob('state_qp_rank*.csv'),('cell',))
    j=raw['segment'].astype(int);xi=raw['xi'];n=len(f['V'])
    np.testing.assert_allclose(raw['Theta'],(1-xi)*old['Theta'][j]+xi*old['Theta'][j+1],rtol=3e-14)
    assert np.all(result['V'][result['prescribed']==1]==VP)
    metrics=[]
    for name,v in [('incoming',old['V']),('A',f['V']),('B',result['V'])]:
        left,mid,right=v[[797,796,795]]/VP
        metrics.append(dict(case=name,V39900=left,V39950=mid,V40000_free=right,
                            chord_defect=.5*(left+right)-mid,local_min_depth=min(left,right)-mid,
                            deep_minus_free_jump=1-right))
    metrics[1]['new_chord_growth']=metrics[1]['chord_defect']-metrics[0]['chord_defect']
    metrics[2]['new_chord_growth']=metrics[2]['chord_defect']-metrics[0]['chord_defect']
    records(out/'early_comparison.csv',[{**r,'new_chord_growth':r.get('new_chord_growth',0.)} for r in metrics])
    a=window(BASE,2)
    hist=cat((ROOT/'probe').glob('early_working_history_rank*.csv'),('cell',))
    hist=select(hist,hist['chi']>0)
    a=select(a,np.lexsort((a['x'],a['y'])));hist=select(hist,np.lexsort((hist['x'],hist['y'])))
    np.testing.assert_array_equal(a['x'],hist['x']);np.testing.assert_array_equal(a['y'],hist['y'])
    frozen=retained(a,f['dt'][0]);exported=np.array([hist['old_xx'],hist['old_yy'],hist['old_xy']])
    error=float(max(abs(frozen-exported).ravel()))
    assert error<1e-5,error
    history_stats=[]
    for lo,hi in ((39800.,40200.),(39000.,39400.)):
        mask=(a['xd']>=lo)&(a['xd']<=hi)
        w=hist['weight'][mask]
        for key in ('old_shear','old_minus_normal'):
            v=hist[key][mask];mean=float(w@v/sum(w))
            history_stats.append(dict(lo=lo,hi=hi,field=key,minimum=float(min(v)),maximum=float(max(v)),
                                      mean=mean,rms_about_mean=float(np.sqrt(w@((v-mean)**2)/sum(w)))))
    records(out/'incoming_raw_history.csv',history_stats)
    xd=(1-xi)*f['xd'][j]+xi*f['xd'][j+1]
    r=select(raw,(xd>=37000)&(xd<=43000));r=select(r,np.lexsort((r['x'],r['y'])))
    np.testing.assert_array_equal(a['x'],r['x']);np.testing.assert_array_equal(a['y'],r['y'])
    pairs=dict(xd=a['xd'],x=a['x'],y=a['y'],weight=r['weight'])
    a['sigma']=a['sigma_n']
    for key in ('p','tau_xx','tau_yy','tau_xy','sigma','q'):
        pairs[key+'_A']=a[key];pairs[key+'_B']=r[key]
    write(out/'early_matched_stress.csv',pairs)
    stress=[]
    for name,data in [('A',a),('B',r)]:
        for key in ('p','tau_xx','tau_yy','tau_xy','sigma','q'):
            stress.append(dict(case=name,field=key,minimum=float(min(data[key])),maximum=float(max(data[key])),peak_to_peak=float(np.ptp(data[key]))))
    records(out/'early_stress_extrema.csv',stress)
    close=[]
    mask=(a['xd']>=39800)&(a['xd']<=40200)
    for key in ('p','tau_xx','tau_yy','tau_xy','sigma','q'):
        for name,data in [('A',a),('B',r)]:
            v=data[key][mask]
            close.append(dict(case=name,field=key,minimum=float(min(v)),maximum=float(max(v)),peak_to_peak=float(np.ptp(v))))
    records(out/'junction_close_stress.csv',close)
    budgets=[]
    mass=weak(raw,np.ones(len(j)),n)
    for field,column in [('q','weak_q'),('sigma','weak_sigma'),('R','weak_R')]:
        assert max(abs(weak(raw,raw[field],n)-result[column])/mass)<2e-4
    aj=a['segment'].astype(int)
    a['friction']=a['sigma']*friction(a['V'],(1-a['xi'])*old['Theta'][aj]+a['xi']*old['Theta'][aj+1])
    a['damping']=4624440.*a['V'];a['R']=a['q']-a['friction']-a['damping'];a['shape0']=1-a['xi']
    for name,data,nodes in [('A',a,(796,)),('B',raw,(795,796,797))]:
        for i in nodes:
            for segment in (i-1,i):
                mask=data['segment']==segment
                shape=data['shape1'][mask] if i==segment+1 else data['shape0'][mask]
                w=data['weight'][mask]*shape
                budgets.append(dict(case=name,node=i,segment=segment,measure=float(sum(w)),
                                    **{key:float(w@data[key][mask]) for key in ('q','friction','damping','R','sigma')}))
    records(out/'early_element_budgets.csv',budgets)
    summary=dict(metrics=metrics,working_history_max_error_Pa=error,
                 chord_growth_reduction=1-metrics[2]['new_chord_growth']/metrics[1]['new_chord_growth'],
                 seconds=execution['seconds'],lower_active=int(sum(result['lower_active'])))
    (out/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mask=(f['xd']>=39600)&(f['xd']<=40150)
    order=np.flatnonzero(mask)[::-1]
    free=f['xd'][order]<=40000+1e-6
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for data,label in ((old['V'],'incoming'),(f['V'],'shared trace A')):
        axes[0].plot(f['xd'][order]/1000,(data[order]/VP-1)*1e6,'.-',label=label)
    axes[0].plot(f['xd'][order][free]/1000,(result['V'][order][free]/VP-1)*1e6,'.-',label='independent free trace B')
    axes[0].plot([40,40.15],[0,0],'--',label='prescribed deep side')
    axes[0].set_ylabel('(V/Vp - 1) × 1e6');axes[0].set_xlabel('Down-dip distance (km)')
    axes[0].legend(fontsize=8);axes[0].grid(True,alpha=.3)
    for data,label in ((a,'A'),(r,'B')):
        x=a['xd']/1000
        near=(x>39.8)&(x<40.2)
        axes[1].scatter(x[near],data['sigma'][near]-50e6,s=2,label=label)
    axes[1].set_xlabel('Down-dip distance (km)');axes[1].set_ylabel('Raw sigma_n - 50 MPa (Pa)')
    axes[1].legend();axes[1].grid(True,alpha=.3)
    fig.tight_layout();fig.savefig(out/'early_comparison.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
