"""Matched, evolving-reference disturbance norms and signed friction budgets.

The exact nonlinear friction difference is telescoped at the incoming state:
 -mu_P (sigma_P-sigma_R)
 -sigma_R [mu_P(V_P,Theta_P)-mu_R(V_P,Theta_P)]  (mixture)
 -sigma_R [mu_R(V_P,Theta_P)-mu_R(V_P,Theta_R)]  (state)
 -sigma_R [mu_R(V_P,Theta_R)-mu_R(V_R,Theta_R)]  (rate).
All terms use the same owned production QPs and JxW*chi weights.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE/'first_long_run/state-disturbance'


def table(path):
    return np.atleast_1d(np.genfromtxt(path,delimiter=',',names=True))


def quadratic(x, y, nodes):
    d,u=nodes['mass_diagonal'],nodes['mass_upper'][:-1]
    return np.dot(d*x,y)+np.dot(u*x[:-1],y[1:])+np.dot(u*x[1:],y[:-1])


def mu(v,theta,a):
    z=np.log(v/(2e-6))+(.6+.015*np.log(theta*1e-6/.008))/a
    return a*np.arcsinh(np.exp(z))


def infer_a(q):
    # Recover the actual mixture coefficient from the exported production mu,
    # not the nominal sharp-fault depth profile. mu is monotone in a here.
    lo=np.full(len(q),.01-1e-13);hi=np.full(len(q),.025+1e-13)
    for _ in range(45):
        mid=(lo+hi)/2
        larger=mu(q[:,5],q[:,6],mid)>q[:,9]
        lo=np.where(larger,mid,lo);hi=np.where(larger,hi,mid)
    a=(lo+hi)/2
    assert np.max(abs(mu(q[:,5],q[:,6],a)-q[:,9]))<5e-13
    return a


def qp(directory,step):
    blocks=[]
    for p in sorted(directory.glob(f'disturbance_qp_{step}_rank*.bin')):
        data=np.fromfile(p,dtype='<f8').reshape(-1,12)
        rank=int(p.stem.split('rank')[-1])
        assert np.all(data[:,0]<1e7)
        # active_cell_index is rank-local for the distributed triangulation.
        # These branches retain the same partition and rank count.
        data[:,0]+=rank*1e7
        blocks.append(data)
    q=np.concatenate(blocks)
    order=np.lexsort((q[:,1],q[:,0]));q=q[order]
    assert len(np.unique(q[:,:2],axis=0))==len(q)
    return q


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference');parser.add_argument('perturbed')
    parser.add_argument('--nodes-only',action='store_true')
    args=parser.parse_args()
    ref,pert=ROOT/args.reference,ROOT/args.perturbed
    launch=json.loads((pert/'launch.json').read_text());amplitude=launch['epsilon']
    assert amplitude!=0
    records=[]
    initial=None
    for path in sorted(pert.glob('disturbance_nodes_*.csv'),key=lambda p:int(p.stem.split('_')[-1])):
        step=int(path.stem.split('_')[-1]);refpath=ref/path.name
        if not refpath.exists(): continue
        # Only fully accepted states, never incomplete diagnostic writes.
        accepted=table(pert/'accepted_steps.csv');accepted_ref=table(ref/'accepted_steps.csv')
        if step not in accepted['step'] or step not in accepted_ref['step']:continue
        p,r=table(path),table(refpath)
        assert np.array_equal(p['node'],r['node']) and np.array_equal(p['xd'],r['xd'])
        assert np.max(abs(p['time']-r['time']))<1e-5 and np.array_equal(p['dt'],r['dt'])
        for key in ('mass_diagonal','mass_upper'):
            assert np.linalg.norm(p[key]-r[key])/np.linalg.norm(r[key])<1e-12
        # An unperturbed reference may have been exported with a different
        # diagnostic wavelength. Normalize/project with this branch's pattern.
        one=np.ones(len(r));w=p['w'];den=quadratic(w,w,r);measure=quadratic(one,one,r)
        def project(v):return float(quadratic(w,v,r)/den)
        def norm(v):return float(np.sqrt(max(0.,quadratic(v,v,r))/measure))
        if step==12:
            assert np.max(abs(np.log(p['Theta_in']/r['Theta_in'])-amplitude*w))<5e-16
            delta=r['Theta_in']*np.expm1(amplitude*w)
            initial=dict(time_s=launch['start_s'],Theta_projection=project(delta),Theta_norm=norm(delta),
                         logTheta_projection=amplitude,logTheta_norm=norm(amplitude*w),V_projection=0.,V_norm=0.)
        row=dict(step=step,time_s=float(r['time'][0]),elapsed_years=float((r['time'][0]-launch['start_s'])/31557600),
                 epsilon=amplitude,mode_mass=den,measure=measure)
        for key,difference in [('V',p['V']-r['V']),('Theta',p['Theta_out']-r['Theta_out']),
                               ('logTheta',np.log(p['Theta_out']/r['Theta_out'])),
                               ('incoming_logTheta',np.log(p['Theta_in']/r['Theta_in']))]:
            row[key+'_projection']=project(difference);row[key+'_norm']=norm(difference)
        row['logTheta_projection_gain']=row['logTheta_projection']/amplitude
        row['logTheta_norm_gain']=row['logTheta_norm']/(abs(amplitude)*np.sqrt(den/measure))
        row['V_projection_over_eps_Vp']=row['V_projection']/(amplitude*1e-9)
        row['V_norm_over_eps_Vp']=row['V_norm']/(abs(amplitude)*1e-9)
        row['V_norm_over_eps_Vp_pattern_norm']=row['V_norm_over_eps_Vp']/np.sqrt(den/measure)
        if not args.nodes_only:
            a,b=qp(pert,step),qp(ref,step)
            assert np.array_equal(a[:,:4],b[:,:4]),'QP ownership/coordinate mismatch'
            assert np.max(abs(a[:,4]/b[:,4]-1))<1e-12
            ar=infer_a(b)
            mu_rate=mu(a[:,5],b[:,6],ar);mu_state=mu(a[:,5],a[:,6],ar)
            used_sigma=b[:,8] if launch.get('control')=='normal' else a[:,8]
            terms=dict(shear=a[:,7]-b[:,7],normal_friction=-a[:,9]*(used_sigma-b[:,8]),
                       state_friction=-b[:,8]*(mu_state-mu_rate),
                       rate_friction=-b[:,8]*(mu_rate-b[:,9]),
                       mixture_friction=-b[:,8]*(a[:,9]-mu_state),
                       damping=-(a[:,10]-b[:,10]))
            total=sum(terms.values());exact=a[:,11]-b[:,11]
            assert np.max(abs(total-exact))<1e-7
            j=b[:,2].astype(int);xi=b[:,3];weight=b[:,4]
            wq=(1-xi)*w[j]+xi*w[j+1]
            assert abs(np.dot(weight,wq*wq)/den-1)<1e-12
            for name,v in terms.items():
                row[name+'_projection_Pa']=float(np.dot(weight*wq,v)/den)
                row[name+'_norm_Pa']=float(np.sqrt(np.dot(weight,v*v)/weight.sum()))
            row['budget_max_closure_Pa']=float(np.max(abs(total-exact)))
            row['actual_sigma_projection_Pa']=float(np.dot(weight*wq,a[:,8]-b[:,8])/den)
            row['actual_sigma_norm_Pa']=float(np.sqrt(np.dot(weight,(a[:,8]-b[:,8])**2)/weight.sum()))
        records.append(row)
    assert records,'No common accepted states'
    stem='comparison_nodes' if args.nodes_only else 'comparison'
    with (pert/(stem+'.csv')).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (pert/(stem+'.json')).write_text(json.dumps(records,indent=2)+'\n')
    if initial is not None:(pert/'initial_disturbance.json').write_text(json.dumps(initial,indent=2)+'\n')
    print(json.dumps(dict(first=records[0],last=records[-1]),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=[r['elapsed_years'] for r in records]
    fig,ax=plt.subplots(2,2,figsize=(10,7),sharex=True)
    for axes,key in zip(ax.flat,('V_projection_over_eps_Vp','V_norm_over_eps_Vp','logTheta_projection_gain','logTheta_norm_gain')):
        axes.plot(t,[r[key] for r in records],'.-');axes.set_ylabel(key);axes.grid(alpha=.3);axes.set_xlabel('Years since accepted step 11')
    fig.suptitle(args.perturbed+' minus evolving '+args.reference);fig.tight_layout();fig.savefig(pert/(stem+'.png'),dpi=160);plt.close(fig)


if __name__=='__main__':main()
