#!/usr/bin/env python3
"""Finish bounded P5b uniform, history-recurrence and negative restart checks."""
from pathlib import Path
import argparse,json
import numpy as np
from p5b_analysis import read,save,sha,constrained_actions,floor
ROOT=Path(__file__).resolve().parents[4]
def main(work,out):
    uniform=work/'runs/uniform';reference=read(out/'radial-history.csv')
    stats=constrained_actions(uniform,out/'uniform')
    assert stats['hanging_constraint_rows']==0
    maximum=0.
    keys=['phi','tauxx_Pa','tauxy_Pa','Axx','Axy','Kxx','Kxy','Qxx','Qxy','Y_Pa']
    for p in sorted(uniform.glob('history-*.csv')):
        # Known exported layout: only the CellId column is non-numeric.
        names=p.open().readline().strip().split(',')
        use=[names.index(k) for k in ['JxW_m2',*keys]]
        a=np.loadtxt(p,delimiter=',',skiprows=1,usecols=use)
        w=a[:,0];r=reference[int(p.stem[-4:])]
        for j,k in enumerate(keys):
            value=float(w@a[:,j+1]/sum(w));maximum=max(maximum,abs(value-r[k])/max(floor(k),abs(r[k])))
    assert maximum<=1e-12,maximum
    stats['physical_mean_defect_against_hanging']=maximum
    save(out/'uniform-control.json',stats)
    recurrence={}
    G,eta,m,R,Kf,area,dt=1e6,1e8,4.,1000.,8.,.25**2,.0125
    beta=1/(1+dt*G/eta)
    for mode in ['scalar','fixed','radial']:
        a=read(out/f'{mode}-history.csv');ledger=read(out/f'{mode}-ledger.csv');errors={}
        def check(label,x,y,scale):
            e=float(np.linalg.norm(np.asarray(x)-np.asarray(y))/max(scale,np.linalg.norm(y)))
            errors[label]=max(errors.get(label,0.),e);assert e<=1e-12,(mode,label,e)
        for i in range(1,len(a)):
            r,n=a[i],a[i-1];t=r['time_s'];w=np.array([r['Wxx'],r['Wxy']])
            A=np.array([r['Axx'],r['Axy']]);oldA=np.array([n['Axx'],n['Axy']])
            K=np.array([r['Kxx'],r['Kxy']]);oldK=np.array([n['Kxx'],n['Kxy']]);Q=np.array([r['Qxx'],r['Qxy']])
            h=m*r['phi']*(1+r['phi'])/(1-r['phi'])**2;hp=m*(1+3*r['phi'])/(1-r['phi'])**3
            T=np.array([r['Txx_Pa'],r['Txy_Pa']]);oldT=np.array([n['Txx_Pa'],n['Txy_Pa']])
            rate_xy=.015 if t<=.3+1e-12 else 0.;rate_normal=.015 if .3+1e-12<t<=.6+1e-12 else 0.
            expected=beta*(oldT+G*dt*np.array([rate_normal,rate_xy]))
            check('once_only_affine_mean_T_source',T,expected,1000.)
            check('A_Maxwell_plus_flow',A,beta*(oldA+dt*w),.001)
            check('accumulated_tensor_K',K,oldK+dt*w,.001)
            check('accumulated_magnitude_L',r['L'],n['L']+dt*2*np.linalg.norm(w),.001)
            check('A_Q_compliance_identity',2*G*A,h*Q,1000.)
            tau=T-2*G*A;check('physical_stress',np.array([r['tauxx_Pa'],r['tauxy_Pa']]),tau,1000.)
            check('physical_driving',r['Y_Pa'],hp*np.dot(Q,Q)/(2*G),8.)
            check('recoverable_storage',r['Eb_J_m3']+r['Ec_J_m3'],(np.dot(tau,tau)+h*np.dot(Q,Q))/(2*G),1.)
            row=ledger[i-1]
            check('bulk_viscosity',row['Db'],dt*area*np.dot(tau,tau)/eta,1.)
            check('cohesive_viscosity',row['Dc'],dt*area*h*np.dot(Q,Q)/eta,1.)
            check('friction_dissipation',row['Df'],dt*area*R*2*np.linalg.norm(w),1.)
            check('uniform_AT1_storage',row['fracture'],area*Kf*r['phi'],1.)
        recurrence[mode]=errors
    save(out/'history-recurrences.json',recurrence)
    negative={}
    for case,label,marker in [('wrong-mode','p5b-restart-mode-rejected','checkpoint mode/version mismatch'),('wrong-layout','p5b-restart-layout-rejected','Bulk restart changed quadrature or coefficients')]:
        command=ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence'/label
        record=json.load(open(command.with_suffix('.json')));log=command.with_suffix('.log').read_text()
        assert record['exit_code']==1 and marker in log
        target=work/'runs'/case;assert not (target/'bulk-accepted.csv').exists()
        checkpoint_files=[]
        for p in (work/'runs/split/restart').rglob('*'):
            if p.is_file():
                relative=p.relative_to(work/'runs/split/restart');assert sha(p)==sha(target/'restart'/relative);checkpoint_files.append(str(relative))
        negative[case]=dict(exit_code=record['exit_code'],expected_marker=marker,no_publication=True,original_checkpoint_files_unchanged=checkpoint_files)
    save(out/'negative-restart-checks.json',negative)
    fixed_reference=work/'runs/fixed-small-step';fixed_resume=work/'runs/fixed-resume-previous';matched=[]
    for kind in ['record','solution','history','weak','phase-forces']:
        suffix='bin' if kind in ('record','solution') else 'csv';name=f'{kind}-0064.{suffix}'
        assert sha(fixed_reference/name)==sha(fixed_resume/name);matched.append(name)
    fixed_step=read(fixed_resume/'bulk-accepted.csv');assert fixed_step['step'][0]==64 and fixed_step['dt'][0]==.0125
    fixed_restart=dict(restored_publication=63,accepted_publication=64,dt_s=float(fixed_step['dt'][0]),bitwise_files=matched)
    save(out/'fixed-restart.json',fixed_restart)
    print(json.dumps(dict(uniform=stats,history_recurrences=recurrence,negative_restarts=negative,fixed_restart=fixed_restart),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('out',type=Path);a=p.parse_args();(a.out/'uniform').mkdir(exist_ok=True,parents=True);main(a.work,a.out)
