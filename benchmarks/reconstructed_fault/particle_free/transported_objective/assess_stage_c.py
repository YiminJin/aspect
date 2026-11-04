#!/usr/bin/env python3
"""Reduce retained native Stage C evidence; no simulations or checkpoint edits."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'build-particle-free/transported-objective/stage-c'
RUNS=BASE/'runs'
EVIDENCE=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_c'


def rows(case,name):
    with (RUNS/case/'output'/name).open() as stream:
        return list(csv.DictReader(stream))


def record(case,expected=0):
    result=json.loads((RUNS/case/'command.json').read_text())
    assert result['exit_code']==expected,(case,result['exit_code'])
    return result


def tensor_records(case):
    result={}
    for path in (RUNS/case/'output').glob('objective-dg-final-rank*.csv'):
        for r in csv.DictReader(path.open()):
            key=(r['cell'],int(r['c']),int(r['i']))
            assert key not in result
            result[key]=(float(r['T_Pa']),float(r['R_Pa']))
    assert result
    return result


def compare_tensors(a,b):
    x,y=tensor_records(a),tensor_records(b)
    assert x.keys()==y.keys()
    error=[max(abs(x[k][j]-y[k][j]) for k in x) for j in (0,1)]
    scale=[max(1.,max(abs(x[k][j]) for k in x)) for j in (0,1)]
    return dict(coefficients=len(x),T_error_Pa=error[0],R_error_Pa=error[1],
                T_scale_Pa=scale[0],R_scale_Pa=scale[1],
                passed=all(e<=2e-11*s for e,s in zip(error,scale)))


def history_records(case,step):
    path=RUNS/case/'output'/f'cohesive-{step:06d}-physical-history.csv'
    with path.open() as stream:
        return {r[0]:list(map(float,r[1:])) for r in csv.reader(stream)}


def compare_cut(a,b,step):
    x,y=history_records(a,step),history_records(b,step)
    assert x.keys()==y.keys()
    tensor_error=native_error=0.;tensor_scale=native_scale=1.
    for key,r in x.items():
        s=y[key]
        assert len(r)==len(s)
        if key.startswith('@'):
            assert r==s
            continue
        assert r[:8]==s[:8] and r[53]==s[53] # fixed vertices, region count
        tensor_error=max(tensor_error,max(abs(u-v) for u,v in zip(r[8:35],s[8:35])))
        tensor_scale=max(tensor_scale,max(map(abs,r[8:35])))
        native_error=max(native_error,max(abs(u-v) for u,v in zip(r[35:53],s[35:53])))
        native_scale=max(native_scale,max(map(abs,r[35:53])))
        for region in range(int(r[53])):
            at=54+32*region
            assert r[at:at+5]==s[at:at+5] # material region and basis bounds
            tensor_error=max(tensor_error,max(abs(u-v) for u,v in zip(r[at+5:at+32],s[at+5:at+32])))
            tensor_scale=max(tensor_scale,max(map(abs,r[at+5:at+32])))
    return dict(tensor_error_Pa=tensor_error,tensor_scale_Pa=tensor_scale,
                native_displacement_error_m=native_error,native_scale_m=native_scale,
                partitions_and_vertices_exact=True,
                passed=tensor_error<=2e-11*tensor_scale and native_error<=2e-11*native_scale)


def energy(case):
    totals={k:0. for k in ['fracture_increment_J_m','physical_remainder_with_objective_J_m',
                          'objective_projection_J_m','ledger_error_J_m']}
    frames=sorted((RUNS/case/'output').glob('cohesive-*-diagnostics.csv'))
    for path in frames:
        with path.open() as stream:
            d={r['quantity']:float(r['value']) for r in csv.DictReader(stream)}
        for key in totals:totals[key]+=d.get(key,0.)
    totals['fracture_relative_error']=abs(totals['physical_remainder_with_objective_J_m'])/totals['fracture_increment_J_m']
    totals['physical_screen_passed']=totals['fracture_relative_error']<=.01
    totals['accepted_steps']=len(frames)-1
    return totals


def main():
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    inventory=[]
    for path in sorted(RUNS.glob('*/command.json')):
        d=json.loads(path.read_text())
        inventory.append(dict(case=path.parent.name,exit_code=d['exit_code'],elapsed_s=d['elapsed_s'],
                              core_sha256=d['core_sha256'],plugin_sha256=d['plugin_sha256'],
                              input_sha256=d['input_sha256'],command=d['command']))
        assert hashlib.sha256((path.parent/'input.prm').read_bytes()).hexdigest()==d['input_sha256']
    result=dict(stage_c_complete=False,production_ready=False,inventory=inventory)
    for name in ['native-qualified-fixed','native-reject','native-cutback-control-time',
                 'native-uninterrupted-five','native-restart-four','native-transfer-field',
                 'native-amr-uninterrupted5','native-amr-restart4',
                 'cut-phase-restart4','cut-phase-uninterrupted10','cut-rejected-phase','cut-reject-control']:
        record(name)
    result['mature_rejection']=compare_tensors('native-reject','native-cutback-control-time')
    result['mature_restart_2_to_4']=compare_tensors('native-uninterrupted-five','native-restart-four')
    result['mature_amr']=rows('native-transfer-field','objective-dg-transfer.csv')[0]
    record('native-transfer-cache-final')
    record('cut-native-final')
    result['final_cache_validation']=dict(
      transfer=rows('native-transfer-cache-final','objective-dg-transfer.csv')[0],
      cut_history=rows('cut-native-final','cohesive-objective-check.csv')[0])
    for name in ['material-owner-native-final','material-owner-cut-final']:record(name)
    result['final_binary_regression']=compare_tensors('native-transfer-cache-final','material-owner-native-final')
    result['final_cut_history']=rows('material-owner-cut-final','cohesive-objective-check.csv')[0]
    result['mature_amr_restart_2_to_4']=compare_tensors('native-amr-uninterrupted5','native-amr-restart4')
    result['cohesive_restart_2_to_4']=compare_cut('cut-phase-restart4','cut-phase-uninterrupted10',10)
    result['cohesive_rejection']=compare_cut('cut-rejected-phase','cut-reject-control',9)
    a=RUNS/'legacy-pre-c-control/output/accepted_steps.csv'
    b=RUNS/'legacy-stage-c-control/output/accepted_steps.csv'
    result['legacy_accepted_steps_identical']=a.read_bytes()==b.read_bytes()
    moments=rows('native-qualified-fixed','objective-dg-summary.csv')
    result['mature_moments']={k:max(float(r[k]) for r in moments) for k in
                              ['moment_error_Pa','scale_Pa','residual_norm_Pa','max_angle_rad']}
    result['transport']={}
    for label,prefix in [('uniform','transport-uniform-'),('layered','layered-native-')]:
        series=[]
        for level in (3,4,5):
            name=f'{prefix}r{level}';record(name)
            series.append([{k:float(v) for k,v in r.items()} for r in rows(name,'objective-dg-transport.csv')])
        orders=[]
        for direction in (0,1):
            orders.append([math.log(series[i][direction]['relative_L2']/series[i+1][direction]['relative_L2'])/math.log(2.) for i in (0,1)])
        result['transport'][label]=dict(series=series,orders=orders,
          passed=all(p>1.8 for pair in orders for p in pair) and all(r['relative_L2']<.005 for r in series[-1]))
    result['cohesive_spaces']=[]
    for level in (0,1,2):
        name=f'cut-native-history-{level}';record(name)
        r={k:float(v) for k,v in rows(name,'cohesive-objective-check.csv')[0].items()}
        r['uncut_space_passed']=r['rotation_increment_relative_L2']<1e-3
        r['native_cut_space_passed']=r['native_cut_increment_relative_L2']<1e-3
        result['cohesive_spaces'].append(r)
    result['phase_energy']=[dict(case=name,dt_s=dt,**energy(name)) for name,dt in
                            [('cut-native-phase8',.000125),('cut-phase16',.0000625),('cut-phase32',.00003125)]]
    result['tip_admission']=[]
    for name in ['cut-tip-airy','tip-airy-legacy','tip-airy-half-dt']:
        record(name,1)
        assert 'Native signed line search failed' in (RUNS/name/'run.log').read_text(errors='replace')
        result['tip_admission'].append(dict(case=name,passed=False,reason='Native signed line search failed'))
    result['conversion_rejections']=[]
    for name in ['reject-new-dg-to-legacy','reject-legacy-to-new-dg','cut-reject-old-objective-checkpoint']:
        record(name,1)
        text=(RUNS/name/'run.log').read_text(errors='replace')
        reason='BP3 model identity' if 'BP3 restart requires the same' in text else 'cohesive law/representation configuration'
        result['conversion_rejections'].append(dict(case=name,rejected=True,first_guard=reason))
    name='reject-old-cohesive-import'
    record(name,1)
    assert 'Cut objective evolution does not admit old prototype' in (RUNS/name/'run.log').read_text(errors='replace')
    result['conversion_rejections'].append(dict(case=name,rejected=True,
      first_guard='fresh cut-law initialization rejects old prototype trajectory import'))
    unit=(BASE/'unit-all.log').read_text()
    found=re.search(r'All tests passed \((\d+) assertions in (\d+) test cases\)',unit)
    assert found
    result['unit_suite']=dict(assertions=int(found[1]),test_cases=int(found[2]),passed=True)
    mpi=re.sub(r'\x1b\[[0-9;]*m','',(BASE/'unit-material-mpi4.log').read_text())
    assert mpi.count('All tests passed (93 assertions in 3 test cases)')==4
    normal_error=float(re.search(r'Material geometry shear: max inverse-transpose normal error=([^\s]+)',unit)[1])
    assert normal_error<5e-15
    result['material_geometry_container']=dict(ranks=4,cases_per_rank=3,
      assertions_per_rank=93,normal_error=normal_error,passed=True,
      scope='replicated label/geometry container, not complete bulk/profile transport')
    shutil.copy2(BASE/'unit-material-mpi4.log',EVIDENCE/'unit-material-mpi4.log')
    result['remaining_stage_c_requirements']=[
      'native material motion/profile and surface transport transaction',
      'native coupled heterogeneous bulk/surface reaction and its converged tangent',
      'moving-state resampling, computational-boundary inflow and moving AMR',
      'deformed cohesive current-measure shape/pressure work',
      'successful persistent tip extension with the updated history']
    result['numerical_controls_passed']=all(result[k]['passed'] for k in
      ['mature_rejection','mature_restart_2_to_4','mature_amr_restart_2_to_4','cohesive_restart_2_to_4','cohesive_rejection'])
    result['numerical_controls_passed']&=result['legacy_accepted_steps_identical'] and all(v['passed'] for v in result['transport'].values())
    result['numerical_controls_passed']&=result['final_binary_regression']['passed']
    assert result['numerical_controls_passed']
    (EVIDENCE/'assessment.json').write_text(json.dumps(result,indent=2)+'\n')
    shutil.copy2(BASE/'unit-all.log',EVIDENCE/'unit-all.log')
    plot(result)
    print(json.dumps({k:result[k] for k in ['stage_c_complete','production_ready','numerical_controls_passed','unit_suite']}))


def plot(result):
    os.environ.setdefault('MPLCONFIGDIR',str(BASE/'matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14.5,4.1),layout='constrained')
    for label,data in result['transport'].items():
        for d in (0,1):
            h=[math.sqrt(2e6/s[d]['cells']) for s in data['series']]
            error=[s[d]['relative_L2'] for s in data['series']]
            axes[0].loglog(h,error,'o-' if d else 's--',label=f'{label}, {"+" if d else "−"}')
    axes[0].set(xlabel='Uniform bulk cell width (m)',ylabel='Full tensor relative L2 error',title='Native transport, 400 m')
    data=result['cohesive_spaces'];h=[1/math.sqrt(s['cells']) for s in data]
    axes[1].loglog(h,[s['rotation_increment_relative_L2'] for s in data],'o-',label='uncut DGQ2: rejected')
    axes[1].loglog(h,[s['native_cut_increment_relative_L2'] for s in data],'s-',label='native piece/side Q2')
    axes[1].axhline(1e-3,color='0.5',ls=':',label='capacity target')
    axes[1].set(xlabel='Bulk cell width (m)',ylabel='Rotation-increment relative L2 error',title='Cohesive history representation')
    data=result['phase_energy']
    axes[2].loglog([s['dt_s'] for s in data],[s['fracture_relative_error'] for s in data],'o-',label='cumulative physical remainder')
    axes[2].axhline(.01,color='0.5',ls=':',label='retained 1% screen')
    axes[2].set(xlabel='Physical timestep (s)',ylabel='|Σ physical remainder| / Σ fracture work',title='Phase energy: screen still fails')
    for ax in axes:
        ax.grid(True,which='both',alpha=.2);ax.legend(fontsize=8)
    fig.savefig(EVIDENCE/'native-qualification.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':main()
