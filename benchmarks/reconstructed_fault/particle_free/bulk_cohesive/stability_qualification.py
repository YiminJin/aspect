#!/usr/bin/env python3
"""Compact verification of native publication stability and four-rank restart.

Run native cases sequentially with run_logged.py; this script only checks and
plots their actual exports. Full checkpoints and cell records stay in build/.
"""
from pathlib import Path
import csv,hashlib,json,shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'build-particle-free/bulk-stability'
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/bulk_stability'

def read(p):
    return np.atleast_1d(np.genfromtxt(p,delimiter=',',names=True,encoding=None))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    names=['supported-records','predictor-curvature','split-first','restart','loaded-growth']
    result={'cases':{},'comparisons':[]}
    for name in names:
        run=BASE/'runs'/name
        a=read(run/'bulk-accepted.csv');s=read(run/'bulk-stability.csv');p=read(run/'bulk-profile.csv')
        assert np.all(a['step']==a['publications']) and np.all(np.diff(a['step'])==1)
        maxima={k:float(np.max(abs(a[k]))) for k in ['mechanical_residual','phase_KKT','projection_error','ledger_error']}
        for k,bound in [('mechanical_residual',1e-9),('phase_KKT',1e-7),('projection_error',1e-12),('ledger_error',1e-10)]:
            assert maxima[k]<=bound,(name,k,maxima)
        for step in a['step'].astype(int):
            ss=s[s['step']==step]
            assert ss[-1]['stable']==1 and np.sum(ss['stable'])==len(ss[ss['attempt']==0])
            assert (run/f'record-{step:04d}.bin').stat().st_size==12288*392*8
        with (run/'gates.csv').open() as f:g=list(csv.DictReader(f))
        assert all(r['passed']=='1' for r in g)
        with (run/'resources.csv').open() as f:resources=list(csv.DictReader(f))
        assert {r['particle_fingerprint'] for r in resources}=={'11428280267573531170'}
        assert {r['freeze'] for r in resources}=={'1'}
        if name=='split-first':
            for gate in ['trial does not publish','rejection restores all accepted owners','deterministic retry','second rejection unchanged']:
                assert sum(r['gate']==gate for r in g)==1,gate
        timing=json.load(open(ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence'/('bulk-stability-'+name+'.json')))
        assert timing['exit_code']==0
        result['cases'][name]=dict(steps=a['step'].astype(int).tolist(),maxima=maxima,
            solver_seconds=float(sum(p['total'])),stability_seconds=float(sum(p['stability'])),
            linear_solves=int(sum(p['linear_solves'])),wall_seconds=timing['seconds'],
            peak_rank_MiB=max(int(r['max_rank_KiB']) for r in resources)/1024,
            additional_outer_rejected_trials=len(p)-len(a),
            publications=a['publications'].astype(int).tolist())
        for filename in ['bulk-accepted.csv','bulk-stability.csv','bulk-profile.csv','resources.csv','gates.csv','pressure.csv']:
            shutil.copy2(run/filename,OUT/(name+'-'+filename))
        for suffix in ['json']:
            shutil.copy2(ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence'/('bulk-stability-'+name+'.'+suffix),OUT/('run-'+name+'.'+suffix))
    base=BASE/'runs/supported-records'
    for name,steps in [('predictor-curvature',[141]),('split-first',[141]),('restart',[142,143])]:
        other=BASE/'runs'/name
        for step in steps:
            files=[f'{prefix}-{step:04d}.{extension}' for prefix,extension in [('phase','csv'),('record','bin'),('means','csv'),('phase-forces','csv'),('boundary-reactions','csv')]]
            assert all(sha(base/f)==sha(other/f) for f in files),(name,step)
            result['comparisons'].append(dict(cases=['supported-records',name],step=step,
                bitwise_phase_all_persistent_histories_stress_means_weak_forces_reactions=True,
                hashes={f:sha(base/f) for f in files}))
    # The unchanged loaded predictor must recover the prior native state.
    loaded=BASE/'runs/loaded-growth';old=ROOT/'build-particle-free/bulk-continuation/runs/reference-load-final'
    for name in ['phase','means','phase-forces','boundary-reactions']:
        f=f'{name}-0140.csv';assert sha(loaded/f)==sha(old/f),f
    rejection=read(BASE/'runs/predictor-curvature/bulk-stability.csv')
    assert len(rejection)==2 and list(rejection['stable'])==[0,1]
    r=rejection[0]
    assert r['accepted_parent_publications']==140 and rejection[1]['accepted_parent_publications']==140
    assert abs(r['minimum_curvature']-(-.20855848252475137))<1e-8
    assert abs(r['max_phase_increment']-.01583691082410646)<1e-14
    assert max(r['feasible_plus'],r['feasible_minus'])>0 and r['minimum_curvature']<-r['uncertainty']
    result['negative_curvature_rejected_before_publication']={k:float(r[k]) for k in rejection.dtype.names}
    grow=read(loaded/'bulk-stability.csv')[0]
    assert grow['max_phase_increment']>.08 and grow['minimum_curvature']>grow['uncertainty']
    assert abs(grow['minimum_curvature']-.1399073145470665)<1e-8
    result['genuine_loading_growth_retained']={k:float(grow[k]) for k in grow.dtype.names}
    parents={}
    for p in BASE.glob('*-input.json'):
        meta=json.load(open(p));source=ROOT/meta['source']
        assert all(sha(source/f)==v for f,v in meta['files'].items()),p
        parents[p.name]=meta
    result['immutable_checkpoint_inputs']=parents
    entry=json.load(open(BASE/'entry-hashes.json'))
    assert all(sha(ROOT/f)==v for f,v in entry.items())
    protected=json.load(open(ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/bulk_fracture/entry.json'))['protected_bp3']
    assert all(sha(ROOT/f)==v['sha256'] for f,v in protected.items())
    result['original_law_history_P4_package_and_BP3_preserved']=True
    result['independent_mixed_matrix_checks']=json.load(open(BASE/'independent-checks.json'))
    result['hold_instability_resolved_in_tested_bulk_setting']=True
    result['quantitative_propagation_qualified']=False
    result['P5_free_direction_implemented']=False
    result['strict_reference_match_passed']=False
    result['inherited_P4_limits']={
        'maximum_temporal_growth_L2_percent':8.302,
        'maximum_spatial_interval_growth_L2_percent':7.442,
        'declared_growth_error_target_percent':5.,
        'boundary_induced_growth_reduction_percent':57.26,
        'source':'Preserved BULK_CONTINUATION_REPORT.md and original evidence archive; no repeated study.'}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    plot()

def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    parent=read(ROOT/'build-particle-free/bulk-continuation/runs/reference-load-final/phase-0140.csv')
    parent.sort(order='dof')
    m=read(BASE/'runs/predictor-curvature/bulk-rejected-mode-141-0.csv')
    rejected=parent['phi'].copy();mode=np.zeros(len(parent))
    ids=m['phase_dof'].astype(int);rejected[ids]=m['phi'];mode[ids]=m['mode']
    recovered=read(BASE/'runs/predictor-curvature/phase-0141.csv');recovered.sort(order='dof')
    order=np.lexsort((parent['x'],parent['y']));x=np.unique(parent['x']);y=np.unique(parent['y']);shape=(len(y),len(x))
    data=[rejected-parent['phi'],recovered['phi']-parent['phi'],mode]
    fig,axes=plt.subplots(1,3,figsize=(12,4.2))
    fig.subplots_adjust(top=.80,bottom=.27,left=.06,right=.98,wspace=.38)
    for k,(ax,v,title) in enumerate(zip(axes,data,['Rejected predictor: phase growth','Published continuation: phase growth','Rejected relaxed phase mode'])):
        im=ax.pcolormesh(x,y,v[order].reshape(shape),shading='gouraud',cmap='magma' if k<2 else 'RdBu_r',vmin=0 if k<2 else -1,vmax=float(data[0].max()) if k<2 else 1)
        phi=rejected if k!=1 else recovered['phi']
        ax.contour(x,y,phi[order].reshape(shape),levels=[.01,.05,.1,.3],colors='white' if k<2 else 'black',linewidths=.7)
        ax.axvline(1.,c='.5',ls='--',lw=.7);ax.set_xlim(.98,1.48);ax.set_ylim(.38,.62);ax.set_aspect('equal')
        ax.set_xlabel('x [m]');ax.set_ylabel('y [m]');ax.set_title(title,fontsize=10)
        bar=fig.colorbar(im,ax=ax,orientation='horizontal',label='dimensionless phase' if k<2 else 'mode / peak magnitude',pad=.30)
        bar.set_ticks([0.,float(data[0].max())/2,float(data[0].max())] if k<2 else [-1.,0.,1.])
        bar.ax.tick_params(labelsize=9)
    fig.suptitle('Same incoming state, first hold at 0.17625 s; negative relaxed curvature = −0.208558',fontsize=12,y=.98)
    fig.savefig(OUT/'publication-recovery.png',dpi=160);plt.close(fig)
    keep=(parent['x']>=.98)&(abs(parent['y']-.5)<=.12)
    with (OUT/'publication-recovery.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['x_m','y_m','parent_phi','rejected_phi','published_phi','negative_phase_mode_peak_normalized'])
        w.writerows(zip(parent['x'][keep],parent['y'][keep],parent['phi'][keep],rejected[keep],recovered['phi'][keep],mode[keep]))

if __name__=='__main__':main()
