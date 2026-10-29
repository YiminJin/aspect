#!/usr/bin/env python3
"""Independent native constraint/Schur checks and compact P5b evidence.

No solver orchestration. Read actual native outputs; preserve failed runs.
"""
from pathlib import Path
import csv,json,hashlib,argparse
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
ROOT=Path(__file__).resolve().parents[4]
def read(p):return np.atleast_1d(np.genfromtxt(p,delimiter=',',names=True,dtype=None,encoding=None))
def table(p,rows):
    with Path(p).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def matrix(p):
    with Path(p).open('rb') as f:
        n,nnz=np.fromfile(f,dtype='<u8',count=2)
        a=np.fromfile(f,dtype=[('i','<u4'),('j','<u4'),('v','<f8')],count=int(nnz))
        assert not f.read()
    return coo_matrix((a['v'],(a['i'],a['j'])),shape=(int(n),int(n))).tocsr()
def constrained_actions(run,out):
    raw=matrix(run/'constraint-raw.bin');native=matrix(run/'continuation-tangent.bin')
    coords=read(run/'continuation-coordinates.csv');master=coords['dof'].astype(int)
    rows=read(run/'constraint-residual.csv');rules=read(run/'constraint-rules.csv')
    entries={}
    for r in rules:entries.setdefault(int(r['slave']),[]).extend([] if r['master']<0 else [(int(r['master']),r['weight'])])
    index={int(i):k for k,i in enumerate(master)};memo={}
    def expand(i,active=()):
        assert i not in active
        if i in memo:return memo[i]
        if i not in entries:return {index[i]:1.} if i in index else {}
        result={}
        for j,w in entries[i]:
            for k,v in expand(j,active+(i,)).items():result[k]=result.get(k,0.)+w*v
        memo[i]=result;return result
    ii=[];jj=[];vv=[]
    for i in range(raw.shape[0]):
        for j,v in expand(i).items():ii.append(i);jj.append(j);vv.append(v)
    C=coo_matrix((vv,(ii,jj)),shape=(raw.shape[0],len(master))).tocsr()
    meta=json.load(open(run/'continuation-state.json'));phase=coords['phase_dof']>=0
    phase_offset=int((coords['dof'][phase]-coords['phase_dof'][phase])[0])
    # Infer physical phase size from all raw phase force records, not master count.
    at=min(run.glob('phase-forces-*.csv'),key=lambda p:abs(int(p.stem[-4:])-32))
    np_phase=len(read(at));factors=np.full(raw.shape[0],meta['beta']*1e6)
    factors[phase_offset:phase_offset+np_phase]=meta['Kf']
    calculated=C.T@raw.multiply(factors[:,None])@C
    error=float(abs(calculated-native).max()/max(1.,abs(native).max()))
    gres=np.asarray(C.T@rows['raw']);reference=rows['condensed'][master]
    gerror=float(np.linalg.norm(gres-reference)/max(1.,np.linalg.norm(reference)))
    assert error<=1e-12 and gerror<=1e-12,(error,gerror)
    mechanics=np.flatnonzero(~phase);all_phase=np.flatnonzero(phase)
    cp=coords[all_phase];density=cp['physical_gradient']/(cp['mass']*meta['Kf'])
    critical_local=np.flatnonzero((cp['phi']>cp['obstacle'])|(density<=1e-7))
    critical=all_phase[critical_local];assert len(critical)>0,'Exercise nonempty mechanically relaxed phase space'
    A=native[mechanics][:,mechanics].tocsc();B=native[mechanics][:,critical].toarray()
    cross=native[critical][:,mechanics].toarray();D=native[critical][:,critical].toarray()
    relaxed=D-cross@splu(A).solve(B)
    metric=meta['Kf']*cp['mass'][critical_local]
    normalized=relaxed/np.sqrt(metric[:,None]*metric[None,:])
    values=np.linalg.eigvalsh(.5*(normalized+normalized.T));frozen=np.linalg.eigvalsh(.5*(D+D.T)/np.sqrt(metric[:,None]*metric[None,:]))
    gate=read(run/'bulk-stability.csv');gate=gate[np.argmin(abs(gate['time']-meta['time']))]
    defect=abs(values[0]-gate['minimum_curvature'])/max(1.,abs(values[0]))
    assert defect<=1e-10 and gate['stable']==1 and len(critical)==gate['critical_masters']
    correction=D-relaxed
    direction=np.sin(np.arange(len(critical))+.37);direction/=np.linalg.norm(direction)
    frozen_action=D/np.sqrt(metric[:,None]*metric[None,:])
    reduction=float(direction@(frozen_action-normalized)@direction)
    assert np.linalg.norm(correction)>0. and reduction>0.,'Mechanics must actually respond to nonuniform phase directions'
    result=dict(relaxation_matrix_norm=float(np.linalg.norm(correction)),normalized_relaxation_matrix_norm=float(np.linalg.norm(frozen_action-normalized)),nonuniform_direction_curvature_reduction=reduction,physical_time_s=meta['time'],raw_dofs=raw.shape[0],independent_dofs=len(master),hanging_constraint_rows=sum(bool(v) and C.getrow(i).nnz>0 for i,v in entries.items()),native_constraint_rows_with_entries=sum(bool(v) for v in entries.values()),CTg_defect=gerror,CTHC_defect=error,critical_masters=len(critical),independent_relaxed_minimum=float(values[0]),native_relaxed_minimum=float(gate['minimum_curvature']),relaxed_defect=float(defect),frozen_mechanics_minimum=float(frozen[0]),native_uncertainty=float(gate['uncertainty']),cross_symmetry=float(gate['cross_symmetry_relative']),pressure_eliminated=True,artificial_constrained_rows_excluded=True,second_condensation=False)
    save(out/'constraint-stability.json',result);return result

def floor(name):
    if name.startswith('Q') or ('_Pa' in name and name!='Y_Pa'):return 1000.
    if name=='Y_Pa':return 8.
    if name[0] in 'AKWL' and name!='JxW_m2':return .001
    return 1.
def histories(runs,out):
    result={};summary=[]
    for mode,run in runs.items():
        accepted=read(run/'bulk-accepted.csv');fixture=read(run/'fixture.csv')
        assert np.array_equal(accepted['step'],np.arange(1,65)) and np.array_equal(accepted['step'],accepted['publications'])
        assert (accepted['mechanical_residual']<=1e-9).all() and (accepted['phase_KKT']<=1e-7).all()
        assert (accepted['projection_error']<=1e-12).all() and (abs(accepted['ledger_error'])<=1e-10).all()
        gates=read(run/'gates.csv');assert (gates['passed']==1).all()
        rows=[]
        for p in sorted(run.glob('history-*.csv')):
            a=read(p);w=a['JxW_m2'];average=lambda key:float(w@a[key]/sum(w))
            row=dict(mode=mode,step=int(p.stem[-4:]),time_s=int(p.stem[-4:])*.0125)
            for key in ['phi','Txx_Pa','Txy_Pa','tauxx_Pa','tauxy_Pa','Axx','Axy','Kxx','Kxy','Qxx','Qxy','Wxx','Wxy','L','Y_Pa','Eb_J_m3','Ec_J_m3']:row[key]=average(key)
            row['phase_spread']=float(a['phi'].max()-a['phi'].min());row['active_flow']=bool(np.max(np.hypot(a['Wxx'],a['Wxy']))>0.)
            if row['active_flow']:row['alpha_deg']=float(np.degrees(np.arctan2(row['Wxx'],row['Wxy'])))
            else:row['alpha_deg']=None
            rows.append(row);summary.append(row)
        table(out/f'{mode}-history.csv',rows)
        profile=read(run/'bulk-profile.csv');stability=read(run/'bulk-stability.csv')
        result[mode]=dict(accepted_steps=64,cells=int(fixture['cells'][0]),native_dofs=int(fixture['DoFs'][0]),max_mechanical=float(accepted['mechanical_residual'].max()),max_KKT=float(accepted['phase_KKT'].max()),max_source=float(accepted['projection_error'].max()),max_ledger=float(abs(accepted['ledger_error']).max()),final_phi=rows[-1]['phi'],max_phase_spread=max(r['phase_spread'] for r in rows),publication_count=64,peak_KiB=int(fixture['peak_KiB'].max()),solver_seconds=float(profile['total'].sum()),assembly_seconds=float(profile['assembly'].sum()),linear_seconds=float(profile['linear'].sum()),history_seconds=float(profile['history'].sum()),stability_seconds=float(profile['stability'].sum()),growing_steps=int((stability['max_phase_increment']>1e-9).sum()))
    defects={};a=runs['scalar'];b=runs['fixed']
    for p in sorted(a.glob('history-*.csv')):
        x=read(p);y=read(b/p.name);assert np.array_equal(x['cell'],y['cell']) and np.array_equal(x['q'],y['q'])
        for name in x.dtype.names:
            if name=='cell':continue
            error=float(np.max(abs(x[name]-y[name]))/max(floor(name),np.max(abs(x[name]))))
            defects[name]=max(defects.get(name,0.),error)
    for kind in ['weak','phase-forces']:
        for p in sorted(a.glob(kind+'-*.csv')):
            x=read(p);y=read(b/p.name)
            for name in x.dtype.names:
                scale=1. if kind=='weak' or 'gradient' in name or name=='residual' else .001
                error=float(np.max(abs(x[name]-y[name]))/max(scale,np.max(abs(x[name]))))
                defects[kind+':'+name]=max(defects.get(kind+':'+name,0.),error)
    assert max(defects.values())<=1e-12,max(defects.items(),key=lambda kv:kv[1])
    save(out/'fixed-identity.json',defects)
    # Before the loading turns, radial and fixed laws are the same physical state.
    prefix=0.
    for step in range(25):
        x=read(runs['radial']/f'history-{step:04d}.csv');y=read(b/f'history-{step:04d}.csv')
        for name in ['phi','Axx','Axy','Kxx','Kxy','Qxx','Qxy','tauxx_Pa','tauxy_Pa','Y_Pa']:
            prefix=max(prefix,float(np.max(abs(x[name]-y[name]))/max(floor(name),np.max(abs(y[name])))))
    assert prefix<=1e-12,prefix
    table(out/'active-direction.csv',[r for r in summary if r['active_flow']])
    return dict(modes=result,fixed_identity_max=max(defects.values()),radial_fixed_prefix_defect=prefix)

def restart(reference,resumed,out):
    exact={};steps=[]
    for kind in ['record','solution','history','weak','phase-forces']:
        matched=[]
        for p in sorted(resumed.glob(kind+'-*')):
            if not (reference/p.name).exists():continue
            assert sha(p)==sha(reference/p.name),(kind,p.name)
            matched.append(p.name)
        assert matched,kind
        exact[kind]=matched
    gates=read(resumed/'gates.csv');assert (gates['passed']==1).all()
    for label in ['converged trial leaves all accepted histories unchanged','outer rejection restores solution and physical owners','retry reproduces solution without publication','second rejection leaves accepted state unchanged']:
        assert (gates['gate']==label).sum()==1,label
    a=read(reference/'bulk-accepted.csv');b=read(resumed/'bulk-accepted.csv')
    # The copied checkpoint retains earlier ledgers in the resumed run.
    common=a[np.isin(a['step'],b['step'])]
    assert np.array_equal(common,b)
    result=dict(bitwise_files=exact,bitwise_accepted_ledger=True,rejected_trials=2,retry_deterministic=True,publications=list(map(int,b['publications'])),final_publication=int(b['publications'][-1]))
    save(out/'restart-lifecycle.json',result);return result

def plots(out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for mode,color in [('scalar','.5'),('fixed','#d27817'),('radial','#136f9b')]:
        a=read(out/f'{mode}-history.csv');t=a['time_s']
        axs[0,0].plot(t,a['phi'],color=color,label=mode)
        axs[0,1].plot(t,a['tauxy_Pa']/1000,color=color,label=mode+' τxy')
        axs[0,1].plot(t,a['tauxx_Pa']/1000,color=color,ls='--',label=mode+' τxx')
        axs[1,0].plot(t,a['Axy'],color=color,label=mode+' Axy');axs[1,0].plot(t,a['Axx'],color=color,ls='--',label=mode+' Axx')
        active=a[a['active_flow']];axs[1,1].plot(active['time_s'],active['alpha_deg'],'.-',color=color,label=mode)
    for ax in axs.flat:
        ax.set_xlabel('Time [s]');ax.grid(alpha=.2);ax.legend(fontsize=7)
        for t in [.3,.6]:ax.axvline(t,color='.7',ls=':',lw=.8)
    axs[0,0].set_ylabel('Physical phase φ [1]');axs[0,1].set_ylabel('Physical stress [kPa]')
    axs[1,0].set_ylabel('Cohesive tensor A [1]');axs[1,1].set_ylabel('Active flow angle atan2(Wxx,Wxy) [deg]')
    fig.suptitle('Native material-block qualification: xy load, turning load, hold\nConstant initial damage obstacle; not a propagation specimen')
    fig.savefig(out/'native-response.png',dpi=155);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for mode,color in [('scalar','.5'),('fixed','#d27817'),('radial','#136f9b')]:
        a=read(out/f'{mode}-ledger.csv');t=a['time']
        axs[0].plot(t,a['bulk']+a['cohesive']+a['fracture']-.1,color=color,label=mode+' storage change')
        axs[0].plot(t,np.cumsum(a['Wext']),color=color,ls='--',label=mode+' endpoint work')
        axs[1].plot(t,np.cumsum(a['Db']+a['Dc']+a['Df']),color=color,label=mode+' physical dissipation')
        axs[1].plot(t,np.cumsum(a['stress_loss']+a['traction_loss']+a['phase_remainder']+a['fracture_loss']),color=color,ls=':',label=mode+' BE remainder')
    for ax in axs:
        ax.set_xlabel('Time [s]');ax.set_ylabel('Per out-of-plane thickness [J/m]');ax.grid(alpha=.2);ax.legend(fontsize=7)
        for t in [.3,.6]:ax.axvline(t,color='.7',ls=':',lw=.8)
    fig.suptitle('Native constitutive ledger; not physical-energy/time qualification')
    fig.savefig(out/'native-energy.png',dpi=155);plt.close(fig)

def main(work,out):
    out.mkdir(exist_ok=True,parents=True)
    runs=dict(scalar=work/'runs/scalar-small-step',fixed=work/'runs/fixed-small-step',radial=work/'runs/radial-action-fixed')
    import shutil
    for mode,run in runs.items():shutil.copy2(run/'bulk-accepted.csv',out/f'{mode}-ledger.csv')
    results=histories(runs,out);results['constraints_and_stability']=constrained_actions(runs['radial'],out)
    if (work/'runs/resume/bulk-accepted.csv').exists():results['restart']=restart(runs['radial'],work/'runs/resume',out)
    entry=json.load(open(work/'entry.json'));assert all(sha(ROOT/p)==h for p,h in entry['protected'].items())
    results.update(protected_files_unchanged=len(entry['protected']),strict_reference_match_passed=False,native_tensor_scope='Serial fixed-mesh CellId/Gauss8, homogeneous bulk only',directional_specimen_run=False)
    save(out/'verification.json',results);plots(out);print(json.dumps(results,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('out',type=Path);a=p.parse_args();main(a.work,a.out)
