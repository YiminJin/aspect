"""Read-only numerical assessment of the saved matched half-clock branches.

Run analyze.py (offline and --evolution) first. This adds clock/lifecycle checks,
windowed nodal diagnostics and comparisons against the evolving raw branch.
It neither launches a solve nor substitutes nodal values for native friction.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from analyze import read, mul, chord


def rms(x, w):
    return float(np.sqrt(np.dot(w, x*x)/w.sum()))


def describe(x, w):
    mean=float(np.dot(w,x)/w.sum())
    return dict(mean=mean, RMS_mean_removed=rms(x-mean,w), minimum=float(x.min()), maximum=float(x.max()))


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def assess(root, plot=False):
    branches=('R','F1','F2')
    comparison=json.loads((root/'comparison.json').read_text()) if (root/'comparison.json').exists() else {}
    report={'units':'SI; nodal window statistics use lumped production work-row masses; native history statistics remain separately labeled',
            'branches':{},'differences':{}}
    tables={};profiles={};operators={}
    base=root/'R/output'
    expected=np.arange(5613,5623)
    windows={'whole':(-1,116000),'shallow':(22000,35000),'deep':(70000,71000),
             'feature':(79000,80000),'top':(0,2000),'bottom':(113470,115471)}
    for b in branches:
        out=root/b/'output';t=read(out/'normal_summary.csv');tables[b]=t
        assert np.array_equal(t.step,expected)
        assert np.array_equal(t.dt,np.loadtxt(out/'normal_actual_intervals.txt'))
        if b!='R':
            assert np.array_equal(t.dt,tables['R'].dt)
            assert np.array_equal(t.time_s,tables['R'].time_s)
        restored=[base/'normal_restored_fault.csv',*sorted(base.glob('normal_restored_particles_rank*.csv'))]
        for p in restored:
            assert digest(p)==digest(out/p.name), (b,p.name,'restored state differs')
        a=read(out/'accepted_steps.csv');a=a[np.isin(a.step,expected)]
        assert np.array_equal(a.step,expected)
        assert np.all(a.fresh_linear_checks_passed==1)
        assert np.all(t.geometry_change_m==0) and np.all(t.I_relative_change_from_restore==0)
        log=(out/'log.txt').read_text()
        linear=re.findall(r'fresh=([0-9.eE+-]+), target=([0-9.eE+-]+)',log)
        assert linear and all(float(f)<=float(g) for f,g in linear)
        assert 'Termination requested by criterion: BP5 normal diagnostic complete' in log
        elapsed=float(t.dt.sum())
        item=dict(steps=len(t),elapsed_s=elapsed,final_time_s=float(t.time_s[-1]),
                  wall_s=float(re.search(r'Total wallclock time elapsed since start\s*\|\s*([0-9.eE+-]+)s',log)[1]),
                  newton_updates=list(map(int,a.newton_updates)),total_krylov=int(a.krylov_iterations.sum()),
                  minimum_alpha=float(a.min_alpha.min()),lower_active=list(map(int,a.lower_active)),
                  max_reported_normalized_nonlinear=float(a.normalized_nonlinear_residual.max()),
                  max_surface_RMS_Pa=float(a.surface_RMS_Pa.max()),max_Theta_audit=float(a.Theta_relative_error.max()),
                  max_weighted_state_change=float(t.realized_weighted_log_change.max()),
                  fresh_linear_records=len(linear),max_fresh_over_target=max(float(f)/float(g) for f,g in linear),
                  restored_files_identical=len(restored),raw_to_filtered_mean_error_Pa=[],states=[])
        profiles[b]=[];operators[b]=[]
        inert=[]
        for k in expected:
            p=read(out/f'normal_profile_{k}.csv');f=read(out/f'normal_filter_{k}.csv')
            profiles[b].append(p);operators[b].append(f)
            m=mul(f.M_diag,f.M_right[:-1],np.ones(len(f)))
            item['raw_to_filtered_mean_error_Pa'].append(float((f.actual_normal_load.sum()-f.raw_normal_load.sum())/m.sum()))
            chk=read(out/f'normal_checks_{k}.csv')[0]
            inert.append([float(chk[c]) for c in ('phase_l2','H_sum','H_squared')])
            state=dict(step=int(k),Vmax=float(p.V_m_per_s.max()),windows={})
            for name,(lo,hi) in windows.items():
                mask=(p.down_dip_s_m>=lo)&(p.down_dip_s_m<=hi)
                ix=np.where(mask)[0];ix=ix[np.argsort(p.down_dip_s_m[ix])]
                w=m[ix];ws={}
                for label,col in [('raw_normal','weak_normal_sum_Pa'),('friction_normal','existing_production_weak_normal_Pa'),
                                  ('pressure','weak_pressure_Pa'),('deviatoric','weak_minus_n_tau_n_Pa'),('V','V_m_per_s')]:
                    x=p[col][ix];ws[label]=describe(x,w)
                    ws[label]['chord_RMS']=rms(chord(x,p.down_dip_s_m[ix]),w[1:-1]) if len(ix)>2 else None
                state['windows'][name]=ws
            item['states'].append(state)
        assert np.array_equal(np.array(inert),np.tile(inert[0],(10,1))), (b,'inert fields changed')
        report['branches'][b]=item
    for b,ref in [('F1','R'),('F2','R'),('F2','F1')]:
        diffs=[]
        for j,k in enumerate(expected):
            p=profiles[b][j];r=profiles[ref][j];f=operators[ref][j]
            assert np.array_equal(p.vertex_id,r.vertex_id) and np.array_equal(p.down_dip_s_m,r.down_dip_s_m)
            w=mul(f.M_diag,f.M_right[:-1],np.ones(len(f)))
            s=dict(step=int(k),windows={})
            for name,(lo,hi) in windows.items():
                mask=(p.down_dip_s_m>=lo)&(p.down_dip_s_m<=hi);wm=w[mask];ws={}
                for label,col in [('V','V_m_per_s'),('Theta_in','incoming_Theta_s'),('Theta_out','committed_Theta_s'),
                                  ('slip','cumulative_slip_m'),('raw_normal','weak_normal_sum_Pa'),
                                  ('friction_normal','existing_production_weak_normal_Pa')]:
                    delta=(p[col]-r[col])[mask]
                    ws[label]=dict(RMS=rms(delta,wm),max_abs=float(abs(delta).max()),mean=float(np.dot(wm,delta)/wm.sum()))
                    if label=='V':
                        ws[label]['relative_L2']=rms(delta,wm)/rms(r[col][mask],wm)
                        meaningful=mask & (r.V_m_per_s>1e-12)
                        ws[label]['point_relative_max_above_1e_minus_12_m_per_s']=float(np.max(abs(p[col][meaningful]/r[col][meaningful]-1))) if meaningful.any() else None
                s['windows'][name]=ws
            diffs.append(s)
        report['differences'][b+'-'+ref]=diffs
    (root/'assessment.json').write_text(json.dumps(report,indent=2)+'\n')
    if comparison:
        (root/'summary.json').write_text(json.dumps(dict(
            qualification=report,offline=json.loads((root/'offline.json').read_text()),
            native_evolution=comparison),indent=2)+'\n')
    for b,item in report['branches'].items():
        print(b, {k:item[k] for k in ('elapsed_s','wall_s','total_krylov','newton_updates','minimum_alpha','lower_active','max_reported_normalized_nonlinear','max_surface_RMS_Pa','max_Theta_audit','max_weighted_state_change','max_fresh_over_target')})
        print('  Vmax',item['states'][0]['Vmax'],item['states'][-1]['Vmax'])
        for name in ('shallow','deep','feature','top','bottom'):
            v=item['states'][-1]['windows'][name]
            print(' ',name,'raw/used chord Pa',v['raw_normal']['chord_RMS'],v['friction_normal']['chord_RMS'])
        if b not in comparison:
            continue
        c=comparison[b]
        print('  extrema first/last',c['states'][0]['extrema'],c['states'][-1]['extrema'])
        for name in ('shallow','deep','feature'):
            for kind in c['states'][0]['history_windows'][name]:
                aa=c['states'][0]['history_windows'][name][kind];bb=c['states'][-1]['history_windows'][name][kind]
                print('  history',name,kind,aa['RMS_mean_removed_Pa'],bb['RMS_mean_removed_Pa'])
    for key,states in report['differences'].items():
        for j in (0,-1):
            print(key,states[j]['step'],{name:{c:states[j]['windows'][name][c] for c in ('V','slip')} for name in ('whole','shallow','deep','feature','top','bottom')})
    if plot:
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(3,3,figsize=(14,10))
        r=profiles['R'][-1]
        for col,(name,(lo,hi)) in enumerate(list(windows.items())[1:4]):
            mask=(r.down_dip_s_m>=lo)&(r.down_dip_s_m<=hi)
            ix=np.where(mask)[0];ix=ix[np.argsort(r.down_dip_s_m[ix])];x=r.down_dip_s_m[ix]/1000
            for b,color in zip(branches,['k','tab:blue','tab:orange']):
                p=profiles[b][-1]
                axs[0,col].plot(x,(p.weak_normal_sum_Pa[ix]-5e7)/1e6,color=color,label=b+' raw Q1')
                if b!='R':
                    axs[0,col].plot(x,(p.existing_production_weak_normal_Pa[ix]-5e7)/1e6,'--',color=color,label=b+' friction Q1')
                    axs[1,col].plot(x,(p.V_m_per_s-r.V_m_per_s)[ix],color=color,label=b+' − R')
                    # Sum accepted rate differences to avoid subtracting metres
                    # of accumulated slip to resolve 1e-13-m deep increments.
                    delta=sum(dt*(pb.V_m_per_s-pr.V_m_per_s)
                              for dt,pb,pr in zip(tables['R'].dt,profiles[b],profiles['R']))
                    axs[2,col].plot(x,delta[ix],color=color,label=b+' − R')
            axs[0,col].set_title(name+' — final accepted state')
            for row,label in enumerate(['Normal input − 50 MPa (MPa)','Velocity difference (m/s)','Slip-increment difference (m)']):
                ax=axs[row,col];ax.set(xlabel='Down dip (km)',ylabel=label);ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle('Matched half-clock comparison: 10 steps, 0.0139428441 s\nSolid normal curves are raw mechanical diagnostics; dashed curves are the filtered friction input',fontsize=11)
        fig.tight_layout();fig.savefig(root/'evolution.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    parser.add_argument('--plot',action='store_true',help='Replace evolution.png with matched-window detail, retaining the two-figure limit')
    args=parser.parse_args();assess(args.directory,args.plot)
