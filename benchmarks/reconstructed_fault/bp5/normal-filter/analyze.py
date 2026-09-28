"""Offline actual-work operator comparison and matched restart summaries.

Input is exported production M/K/M_mu, not mass inferred from row averages.
The independent band solve is checked against the production filtered output.
Never mix the first resumed Newton base with an accepted checkpoint stress.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import csv
from scipy.linalg import solve_banded
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    return np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True, dtype=None,
                                     encoding='utf8')).view(np.recarray)


def record(row):
    return {name:float(row[name]) for name in row.dtype.names}


def mul(d,e,x):
    y=d*x
    y[:-1]+=e*x[1:];y[1:]+=e*x[:-1]
    return y


def solve(d,e,b):
    ab=np.zeros((3,len(d)));ab[1]=d;ab[0,1:]=e;ab[2,:-1]=e
    return solve_banded((1,1),ab,b)


def stats(x,w):
    mean=float(np.dot(w,x)/w.sum())
    return dict(mean_Pa=mean,RMS_mean_removed_Pa=float(np.sqrt(np.dot(w,(x-mean)**2)/w.sum())),
                min_Pa=float(x.min()),max_Pa=float(x.max()))


def q1_stats(x,d,e):
    mass=mul(d,e,np.ones(len(d)));mean=float(np.dot(mass,x)/mass.sum());centered=x-mean
    return dict(mean_Pa=mean,RMS_mean_removed_Pa=float(np.sqrt(np.dot(centered,mul(d,e,centered))/mass.sum())),
                min_coefficient_Pa=float(x.min()),max_coefficient_Pa=float(x.max()))


def chord(x,s):
    return x[1:-1]-(x[:-2]+(x[2:]-x[:-2])*(s[1:-1]-s[:-2])/(s[2:]-s[:-2]))


def offline(root,extra50=False):
    out=root/'R/output';a=read(out/'normal_initial_operator.csv')
    d=a.M_diag;e=a.M_right[:-1]
    k=a.K_diag;ke=a.K_right[:-1]
    b=a.raw_normal_load;mass=mul(d,e,np.ones(len(d)))
    assert np.all(mass>0) and np.max(abs(mul(k,ke,np.ones(len(d)))))<1e-12
    s=a.xd;order=np.argsort(s)
    modes={str(L):solve(d+L*L*k,e+L*L*ke,b) for L in ((0,50,100,200) if extra50 else (0,100,200))}
    raw_qp=np.concatenate([read(p) for p in sorted(out.glob('initial_normal_friction_qp_rank*.csv'))]).view(np.recarray)
    assert len(raw_qp)>0
    report={'stage':'first resumed Newton base: committed checkpoint histories, pending stress_dt; NOT accepted step-5612 current stress',
            'pending_step':int(a.pending_step[0]),'actual_dt':float(a.actual_dt[0]),
            'fault_spacing_m':[float(np.min(abs(np.diff(s)))),float(np.max(abs(np.diff(s))))],
            'raw_work_mean_Pa':float(b.sum()/mass.sum()),'lengths':{},'windows':{}}
    fields={'raw':raw_qp.raw_normal}
    for name,z in modes.items():
        j=raw_qp.segment.astype(int);xi=raw_qp.xi
        fields[name]=(1-xi)*z[j]+xi*z[j+1]
        friction=mul(a.Mmu_diag,a.Mmu_right[:-1],z)
        error=friction-a.friction_load
        cz=chord(z[order],s[order])
        report['lengths'][name]=dict(**q1_stats(z,d,e),whole_mean_error_Pa=float((mul(d,e,z).sum()-b.sum())/mass.sum()),
            chord_RMS_Pa=float(np.sqrt(np.mean(cz*cz))),friction_load_change_l2=float(np.linalg.norm(error)),
            friction_load_relative_l2=float(np.linalg.norm(error)/np.linalg.norm(a.friction_load)))
        assert abs(report['lengths'][name]['whole_mean_error_Pa'])<1e-6
    windows={'shallow':(22000,35000),'deep':(70000,71000),'feature':(79000,80000)}
    for label,(lo,hi) in windows.items():
        mask=(raw_qp.xd>=lo)&(raw_qp.xd<=hi);w=raw_qp.weight[mask]
        assert len(w)>0
        report['windows'][label]={name:stats(value[mask],w) for name,value in fields.items()}
        for name in modes:
            removed=(fields['raw']-fields[name])[mask]
            report['windows'][label][name]['removed_RMS_Pa']=float(np.sqrt(np.dot(w,removed**2)/w.sum()))
            # Actual native friction, with identical mu and weights.
            report['windows'][label][name]['friction_integral_change']=float(np.dot(w,raw_qp.mu[mask]*(-removed)))
    # Endpoints are evaluated on the FULL operator, never window factorizations.
    fig,axs=plt.subplots(2,3,figsize=(14,8))
    ranges=[(s.min(),s.max()),windows['shallow'],windows['deep'],windows['feature'],(s.min(),s.min()+2000),(s.max()-2000,s.max())]
    for ax,(lo,hi) in zip(axs.flat,ranges):
        mask=(s>=lo)&(s<=hi)
        for name,z in modes.items():
            ix=order[mask[order]];ax.plot(s[ix]/1000,(z[ix]-5e7)/1e6,label='Q1 L='+name+' m')
        maskq=(raw_qp.xd>=lo)&(raw_qp.xd<=hi)
        if maskq.any():
            ax.scatter(raw_qp.xd[maskq]/1000,(raw_qp.raw_normal[maskq]-5e7)/1e6,s=1,alpha=.15,color='k',label='raw work QP')
        ax.set(xlabel='Down dip (km)',ylabel='Normal input − 50 MPa (MPa)',xlim=(lo/1000,hi/1000));ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(root/'offline.png',dpi=160);plt.close(fig)
    (root/'offline.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


def evolution(root):
    available={}
    for branch in ('R','F1','F2'):
        path=root/branch/'output/normal_summary.csv'
        if path.is_file():
            table=read(path)
            if len(table): available[branch]=table
    branches=list(available);tables={};profiles={};result={};initial_slip={}
    if 'R' not in branches or len(branches)<2:
        (root/'comparison.json').write_text(json.dumps({'status':'No comparable accepted prefix',
            'available_accepted_steps':{b:len(t) for b,t in available.items()}},indent=2)+'\n')
        return
    common=min(len(t) for t in available.values())
    for branch in branches:
        out=root/branch/'output'
        schema=list(csv.DictReader((out/'normal_property_schema.csv').open()))
        slot=next(int(r['position']) for r in schema if r['name']=='cumulative_signed_slip_m')
        initial_slip[branch]=read(out/'normal_restored_fault.csv')[f'property_{slot}']
        t=available[branch][:common];tables[branch]=t
        profiles[branch]=[read(out/f'normal_profile_{k}.csv') for k in t.step]
        result[branch]={'accepted_steps':len(available[branch]),'compared_prefix_steps':common,'actual_intervals':list(t.dt),'elapsed':list(np.cumsum(t.dt)),
                        'max_projection_residual_Pa':float(t.projection_residual_over_mass_Pa.max()),'states':[]}
        accepted=read(out/'accepted_steps.csv')
        for k,p in zip(t.step,profiles[branch]):
            f=read(out/f'normal_filter_{k}.csv')
            d=f.M_diag;e=f.M_right[:-1];m=mul(d,e,np.ones(len(d)))
            raw=solve(d,e,f.raw_normal_load);used=f.normal_coefficient
            # Even R's 'coefficient' is its consistent diagnostic projection.
            error=mul(d,e,used)-f.actual_normal_load
            assert np.max(abs(error)/m)<1e-5
            item={'step':int(k),'raw_Q1':q1_stats(raw,d,e),'used_Q1':q1_stats(used,d,e),
                'Vmax':float(p.V_m_per_s.max()),'friction_load_l2':float(np.linalg.norm(f.friction_load)),
                'residual_l2':float(np.linalg.norm(f.residual)),
                'extrema':record(read(out/f'normal_filter_extrema_{k}.csv')[0]),
                'solver':{c:float(accepted[accepted.step==k][0][c]) for c in ('newton_updates','krylov_iterations','min_alpha','lower_active','free','normalized_nonlinear_residual','surface_RMS_Pa','fresh_linear_checks_passed')}}
            qp=np.concatenate([read(path) for path in out.glob(f'normal_qp_{k}_rank*.csv')]).view(np.recarray)
            particles=np.concatenate([read(path) for path in out.glob(f'normal_incoming_particles_{k}_rank*.csv')]).view(np.recarray)
            particles=particles[particles.is_ghost==0]
            old=-(qp.n_x**2*qp.incoming_FE_xx+2*qp.n_x*qp.n_y*qp.incoming_FE_xy+qp.n_y**2*qp.incoming_FE_yy)
            # Fixed straight fault. Particle statistics count unique real IDs,
            # with equal particle weights; they are NOT work-QP statistics.
            nx,ny=float(qp.n_x[0]),float(qp.n_y[0])
            pdip=(100000.-particles.y)/np.sqrt(.75)
            particle_normal=-(nx*nx*particles.tau_xx+2*nx*ny*particles.tau_xy+ny*ny*particles.tau_yy)
            item['history_windows']={}
            for label,(lo,hi) in {'shallow':(22000,35000),'deep':(70000,71000),'feature':(79000,80000)}.items():
                mask=(qp.down_dip_s_m>=lo)&(qp.down_dip_s_m<=hi)
                pmask=(pdip>=lo)&(pdip<=hi)
                item['history_windows'][label]={'working_FE_normal_native_work':stats(old[mask],qp.work_weight[mask]),
                    'retained_particle_normal_equal_real_particle_weights':stats(particle_normal[pmask],np.ones(pmask.sum())) if pmask.any() else None}
            result[branch]['states'].append(item)
    # Compare actual dt, not subtraction of absolute timestamps.
    for b in branches[1:]:
        assert np.array_equal(tables[b].dt,tables['R'].dt)
    fig,axs=plt.subplots(2,3,figsize=(14,8))
    for b in branches:
        t=tables[b];elapsed=np.cumsum(t.dt);p=np.sort(profiles[b][-1],order='down_dip_s_m').view(np.recarray)
        for ax,col,label in zip(axs[0],['weak_normal_sum_Pa','V_m_per_s','cumulative_slip_m'],['Raw normal (Pa)','V (m/s)','Slip since checkpoint (m)']):
            value=p[col]-initial_slip[b][p.vertex_id.astype(int)] if col=='cumulative_slip_m' else p[col]
            ax.plot(p.down_dip_s_m/1000,value,label=b);ax.set(xlabel='Down dip (km)',ylabel=label)
        axs[1,0].plot(elapsed,t.V_max,label=b)
        line=axs[1,1].plot(elapsed,[r['raw_Q1']['RMS_mean_removed_Pa'] for r in result[b]['states']],label=b+' raw Q1')[0]
        if b!='R':
            axs[1,1].plot(elapsed,[r['used_Q1']['RMS_mean_removed_Pa'] for r in result[b]['states']],color=line.get_color(),linestyle='--',label=b+' friction Q1')
        axs[1,2].plot(elapsed,[r['friction_load_l2'] for r in result[b]['states']],label=b)
    for ax in axs.flat:ax.legend();ax.grid(alpha=.2)
    for ax,label in zip(axs[1],['Max V (m/s)','Raw full-fault Q1 variation (Pa)','Friction load norm (Pa m)']):
        ax.set(xlabel='Cumulative actual elapsed seconds',ylabel=label)
    fig.tight_layout();fig.savefig(root/'evolution.png',dpi=160);plt.close(fig)
    # Attribution is permitted only after checking maps and weights pointwise.
    for b in branches[1:]:
        result[b]['friction_attribution']=[]
        for k in tables[b].step:
            def qps(branch):
                q=np.concatenate([read(p) for p in (root/branch/'output').glob(f'normal_friction_qp_{k}_rank*.csv')])
                return np.sort(q,order=['cell','qp','fault','segment']).view(np.recarray)
            r=qps('R');f=qps(b)
            same=np.array_equal(r[['cell','qp','fault','segment']],f[['cell','qp','fault','segment']]) and np.array_equal(r[['xi','x','y','weight']],f[['xi','x','y','weight']])
            if not same:
                result[b]['friction_attribution'].append({'step':int(k),'same_map':False});continue
            direct=r.mu*(f.friction_normal-r.friction_normal)
            coefficient=f.friction_normal*(f.mu-r.mu)
            total=f.mu*f.friction_normal-r.mu*r.friction_normal
            err=np.max(abs(direct+coefficient-total))
            result[b]['friction_attribution'].append({'step':int(k),'same_map':True,
                'normal_input_window_integral':float(np.dot(r.weight,direct)),
                'mu_window_integral':float(np.dot(r.weight,coefficient)),
                'identity_max_Pa':float(err)})
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--extra50',action='store_true',help='Only the authorized offline fallback if 200 m plainly oversmooths')
    p.add_argument('--evolution',action='store_true');a=p.parse_args()
    evolution(a.directory) if a.evolution else offline(a.directory,a.extra50)
