"""Frozen-mesh temporal comparison at retained common physical times."""
import csv
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here=Path(__file__).resolve().parent
base=here/'junction-matched-qualified-local4/refined'
trial=here/'junction-contact-half-qualified-local4'


def rows(path):
    with path.open() as f:return list(csv.DictReader(f))


def col(data,key):return np.array([float(r[key]) for r in data])


def profile(root,k):
    folder=root/f'analysis/step{k}'
    if not folder.exists():
        subprocess.run([sys.executable,str(here/'analyze_normal_stress.py'),str(root),
            '--step',str(k),'--output',str(folder)],check=True,stdout=subprocess.DEVNULL)
    return rows(folder/'projected_full.csv')


def weak(root,k):
    parts=[rows(root/f'history_surface_step{k}_rank{r}.csv') for r in range(4)]
    result=[]
    for i,r in enumerate(parts[0]):
        item={key:float(r[key]) for key in list(r)[:5]}
        assert all(all(float(p[i][key])==item[key] for key in list(r)[:5]) for p in parts)
        item.update({key:sum(float(p[i][key]) for p in parts) for key in list(r)[5:]})
        for mode in ['particle','fe']:
            terms=item[mode+'_q']-item[mode+'_C']-item[mode+'_friction']-item[mode+'_damping']
            assert abs(terms-item[mode+'_R'])<1e-12*max(abs(item[mode+'_q']),abs(item[mode+'_friction']))
        result.append(item)
    return result


status=json.loads((trial/'execution.json').read_text())
assert status['status']==0 and status['source_checkpoint_unchanged']
log=(trial/'run.log').read_text()
accepted=rows(trial/'accepted_steps.csv');original=rows(base/'accepted_steps.csv')
assert [int(r['step']) for r in accepted]==[12,13,14,15]
sequence=rows(trial/'sequence.csv')
for r in accepted:
    k=int(r['step'])
    assert re.search(r'BP3 accepted state\s+'+str(k)+r'\b',log)
    for key in ['time','dt']:
        assert abs(float(r[key])-float(sequence[k][key]))<=1e-12*float(r[key])
checks=[]
for line in log.splitlines():
    if 'Fault linear solve:' in line:
        get=lambda key:float(re.search(r'\b'+key+r'=([^,\s]+)',line)[1])
        checks.append(dict(fresh=get('fresh'),target=get('target'),iterations=int(get('iterations'))))
assert checks and all(r['fresh']<=r['target'] for r in checks)
summary=dict(execution=status,linear_returns=len(checks),krylov=sum(r['iterations'] for r in checks),
             worst_fresh_target=max(r['fresh']/r['target'] for r in checks))
summary['nonlinear']=[]
sections=re.split(r'\*\*\* Timestep (\d+):',log)
for j in range(1,len(sections),2):
    text=sections[j+1]
    relative=re.findall(r'Relative nonlinear residuals .*?: ([^,\n]+), ([^\n]+)',text)[-1]
    assert all(float(v)<1e-8 for v in relative)
    line=[s for s in text.splitlines() if 'Fault nonlinear residual:' in s][-1]
    values={key:float(re.search(r'(?:^|[ :,])'+re.escape(key)+r'=([^,\s]+)',line)[1])
            for key in ['bulk','bulk scale','bulk target','surface','surface scale']}
    assert values['bulk']<=values['bulk target'] and values['surface']<=1e-8*values['surface scale']
    summary['nonlinear'].append(dict(step=int(sections[j]),relative=list(map(float,relative)),**values))
baseline_hashes=json.loads((base/'provenance.json').read_text())['sha256']
new_hashes=json.loads((trial/'provenance.json').read_text())['sha256']
binary=str(here.parents[2]/'build-pf-cpdi/aspect-release')
assert baseline_hashes[binary]==new_hashes[binary]
summary['production_executable_identical']=True
reference=rows(base/'fault_11.csv');history=[];saved={}
for label,root,indices,states in [('baseline',base,[11,12,13],original),('half',trial,[12,13,14,15],accepted)]:
    previous=reference
    for k in indices:
        fault=rows(root/f'fault_{k}.csv');w=weak(root,k);p=profile(root,k)
        state=next(r for r in states if int(r['step'])==k)
        np.testing.assert_array_equal(col(fault,'x'),col(reference,'x'))
        np.testing.assert_array_equal(col(fault,'y'),col(reference,'y'))
        np.testing.assert_allclose(col(fault,'Ih'),col(reference,'Ih'),rtol=1e-12,atol=0)
        np.testing.assert_array_equal(col(fault,'tau_bg'),col(reference,'tau_bg'))
        np.testing.assert_array_equal(col(fault,'sigma_n_bg'),col(reference,'sigma_n_bg'))
        np.testing.assert_allclose(col(fault,'weak_residual'),[r['particle_R'] for r in w],rtol=0,atol=1e-5)
        if k!=11:
            dt=float(state['dt']);V=col(fault,'V');x=V*dt/.008
            theta=col(previous,'Theta')*np.exp(-x)-(.008/V)*np.expm1(-x)
            error=max(abs(col(fault,'Theta')-theta)/np.maximum(abs(theta),abs(col(fault,'Theta'))))
            assert error<1e-12
            np.testing.assert_allclose(col(fault,'slip'),col(previous,'slip')+dt*V,rtol=1e-13,atol=1e-15)
            assert float(rows(root/f'history_{k}.csv')[0]['Theta_reference_relative_error'])<1e-12
        else:error=0
        saved[label,k]=(fault,w,p)
        for i in [756,757,758,905]:
            a=w[i];weight=a['weight']
            h=dict(case=label,step=k,time=float(state['time']),dt=float(state['dt']),node=i,
                xd=float(fault[i]['xd']),V=a['V'],active=int(p[i]['lower_active']),
                Theta_committed=float(fault[i]['Theta']),Theta_used=float(previous[i]['Theta']) if k!=11 else None,
                C_committed=float(fault[i]['C']),slip=float(fault[i]['slip']),
                sigma_projected=float(p[i]['sigma_n']),p_projected=float(p[i]['delta_p']),
                minus_tau_N_projected=float(p[i]['minus_delta_tau_N']),theta_check=error)
            h.update({key:a[key]/weight for key in ['particle_q','particle_C','particle_friction',
                'particle_damping','particle_R','particle_sigma','fe_R','fe_sigma','particle_Kdiag','fe_Kdiag']})
            h['reaction']=-h['particle_R'] if h['active'] else 0.
            h['delta_R']=h['fe_R']-h['particle_R']
            history.append(h)
        previous=fault

summary['nodes']=history;summary['common_times']=[]
for bk,hk in [(12,13),(13,15)]:
    f0,w0,p0=saved['baseline',bk];f1,w1,p1=saved['half',hk]
    assert abs(float(f0[0]['time'])-float(f1[0]['time']))<1e-5
    x=col(p0,'xd');entry=dict(time=float(f0[0]['time']),baseline_step=bk,half_step=hk,regions={})
    for name,(lo,hi) in {'junction':(37000,43000),'control':(24000,26000)}.items():
        mask=(x>=lo-1e-7)&(x<=hi+1e-7);region={}
        for key in ['delta_p','minus_delta_tau_N','sigma_n','q','V','slip','Theta_committed']:
            a=col(p0,key)[mask];b=col(p1,key)[mask]
            region[key]=dict(baseline_ptp=float(np.ptp(a)),half_ptp=float(np.ptp(b)),
                difference_rms=float(np.sqrt(np.mean((b-a)**2))),difference_max=float(max(abs(b-a))))
        for label,root,k,w in [('baseline',base,bk,w0),('half',trial,hk,w1)]:
            raw=rows(root/f'analysis/step{k}/raw_selected.csv')
            select=[r for r in raw if lo<=float(r['xd'])<=hi]
            region[label+'_raw']=dict(minimum=min(select,key=lambda r:float(r['sigma_n'])) if select else None,
                maximum=max(select,key=lambda r:float(r['sigma_n'])) if select else None,
                particle_tensile_weight=sum(r['particle_tensile_weight'] for i,r in enumerate(w) if mask[i]),
                fe_tensile_weight=sum(r['fe_tensile_weight'] for i,r in enumerate(w) if mask[i]))
        entry['regions'][name]=region
    summary['common_times'].append(entry)

with (trial/'node_history.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
(trial/'subdivision_comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,axes=plt.subplots(3,1,figsize=(9,9),sharex=True)
start=float(original[11]['time'])
for label in ['baseline','half']:
    data=[r for r in history if r['case']==label and r['node']==756]
    if label=='half':data=[r for r in history if r['case']=='baseline' and r['step']==11 and r['node']==756]+data
    times=(col(data,'time')-start)/31557600
    for ax,key,scale in zip(axes,['V','reaction','Theta_committed'],[1,1e6,31557600]):
        ax.plot(times,col(data,key)/scale,'o-',label=label);ax.set_ylabel(key);ax.grid(alpha=.3)
axes[0].set_yscale('log');axes[0].legend();axes[1].set_ylabel('reaction (MPa)')
axes[2].set_ylabel('committed Theta (yr)');axes[-1].set_xlabel('years since common step-11 state')
fig.tight_layout();fig.savefig(trial/'contact_history.png',dpi=150)
fig,axes=plt.subplots(4,1,figsize=(9,10),sharex=True)
for label,k in [('baseline',13),('half',15)]:
    p=saved[label,k][2];x=col(p,'xd');mask=(x>=37000-1e-7)&(x<=43000+1e-7)
    for ax,key in zip(axes,['delta_p','minus_delta_tau_N','sigma_n','V']):
        ax.plot(x[mask]/1000,col(p,key)[mask]/(1 if key=='V' else 1e6),label=label)
        ax.set_ylabel(key+(' (m/s)' if key=='V' else ' (MPa)'));ax.grid(alpha=.3)
        ax.axvline(40,color='gray',ls=':')
axes[0].legend();axes[-1].set_xlabel('down-dip distance (km)')
fig.tight_layout();fig.savefig(trial/'final_junction_profiles.png',dpi=150)
print(json.dumps({k:v for k,v in summary.items() if k!='nodes'},indent=2))
