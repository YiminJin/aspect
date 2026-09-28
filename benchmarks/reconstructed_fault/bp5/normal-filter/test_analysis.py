"""Cheap synthetic test of the report pipeline; this is NOT BP5 evidence."""
import tempfile
from pathlib import Path
import numpy as np
import csv
from analyze import offline,evolution,solve,mul


def write(path,rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)


with tempfile.TemporaryDirectory(prefix='normal-filter-analysis-') as temp:
    root=Path(temp);out=root/'R/output';out.mkdir(parents=True)
    s=np.arange(0.,115601.,100.);n=len(s);h=100.
    d=np.zeros(n);e=np.zeros(n);k=np.zeros(n);ke=np.zeros(n)
    md=np.zeros(n);me=np.zeros(n);b=np.zeros(n);bf=np.zeros(n)
    samples=[]
    for j in range(n-1):
        for q,xi in enumerate((.5-.5/np.sqrt(3),.5+.5/np.sqrt(3))):
            x=s[j]+h*xi;N=np.array([1-xi,xi]);w=h/2
            sigma=5e7+1e5*np.sin(2*np.pi*x/300)+2e5*np.cos(2*np.pi*x/10000)
            mu=.4+.05*np.cos(2*np.pi*x/s[-1])
            d[j:j+2]+=w*N*N;e[j]+=w*N[0]*N[1]
            md[j:j+2]+=w*N*N*mu;me[j]+=w*N[0]*N[1]*mu
            k[j:j+2]+=w/h**2;ke[j]-=w/h**2
            b[j:j+2]+=w*N*sigma;bf[j:j+2]+=w*N*mu*sigma
            samples.append(dict(cell=str(j),qp=q,fault=0,segment=j,xi=xi,x=x,y=0,xd=x,
                                weight=w,mu=mu,raw_normal=sigma,friction_normal=sigma))
    fields=dict(vertex=np.arange(n),xd=s,pending_step=np.full(n,5613),actual_dt=np.full(n,.0028),
                M_diag=d,M_right=e,K_diag=k,K_right=ke,Mmu_diag=md,Mmu_right=me,
                raw_normal_load=b,friction_load=bf)
    with (out/'normal_initial_operator.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(fields);writer.writerows(zip(*fields.values()))
    with (out/'initial_normal_friction_qp_rank0.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=samples[0]);writer.writeheader();writer.writerows(samples)
    offline(root)
    assert (root/'offline.png').is_file()
    for branch,L in [('R',0),('F1',100),('F2',200)]:
        dest=root/branch/'output';dest.mkdir(parents=True,exist_ok=True)
        write(dest/'normal_property_schema.csv',[dict(name='cumulative_signed_slip_m',position=0)])
        write(dest/'normal_restored_fault.csv',[dict(property_0=0.) for i in range(n)])
        summary=[];accepted=[]
        for step in (5613,5614):
            z=solve(d+L*L*k,e[:-1]+L*L*ke[:-1],b)
            actual=b if branch=='R' else mul(d,e[:-1],z)
            fric=bf if branch=='R' else mul(md,me[:-1],z)
            write(dest/f'normal_filter_{step}.csv',[dict(M_diag=d[i],M_right=e[i],raw_normal_load=b[i],
                actual_normal_load=actual[i],normal_coefficient=z[i],friction_load=fric[i],residual=0.) for i in range(n)])
            write(dest/f'normal_profile_{step}.csv',[dict(vertex_id=i,down_dip_s_m=s[i],weak_normal_sum_Pa=z[i],
                V_m_per_s=.001,cumulative_slip_m=(step-5612)*.0000028) for i in range(n)])
            write(dest/f'normal_filter_extrema_{step}.csv',[dict(raw_QP_min_Pa=4.97e7,raw_QP_max_Pa=5.03e7,
                friction_QP_min_Pa=z.min(),friction_QP_max_Pa=z.max())])
            qps=[dict(row,friction_normal=row['raw_normal'] if branch=='R' else (1-row['xi'])*z[row['segment']]+row['xi']*z[row['segment']+1]) for row in samples]
            write(dest/f'normal_friction_qp_{step}_rank0.csv',qps)
            write(dest/f'normal_qp_{step}_rank0.csv',[dict(n_x=0.,n_y=1.,incoming_FE_xx=0.,incoming_FE_xy=0.,
                incoming_FE_yy=5e7-row['raw_normal'],down_dip_s_m=row['xd'],work_weight=row['weight']) for row in samples])
            write(dest/f'normal_incoming_particles_{step}_rank0.csv',[dict(is_ghost=0,y=100000.-row['xd']*np.sqrt(.75),
                tau_xx=0.,tau_xy=0.,tau_yy=5e7-row['raw_normal']) for row in samples])
            summary.append(dict(step=step,dt=.0028,projection_residual_over_mass_Pa=0.,V_max=.001))
            accepted.append(dict(step=step,newton_updates=3,krylov_iterations=4,min_alpha=1.,lower_active=0,free=n,
                normalized_nonlinear_residual=1e-12,surface_RMS_Pa=1e-8,fresh_linear_checks_passed=1))
        write(dest/'normal_summary.csv',summary);write(dest/'accepted_steps.csv',accepted)
    evolution(root)
    assert (root/'evolution.png').is_file()
    print('Synthetic offline and evolution report pipelines passed; no physical conclusion.')
