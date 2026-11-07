#!/usr/bin/env python3
"""Record the selected two-run experiment without rewriting prior evidence."""
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[4]
E=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence'
OUT=E/'transported_objective/positive_refinement'
RUNS=ROOT/'build-particle-free/transported-objective/stage-c/material-runs'
CASES=['bpf-c-p-m-ref','bpr-c-p-m-surface64','bpr-c-p-m-halfdt64']

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    entry=read(OUT/'entry.json');old=read(E/'transported_objective/stage_c/entry.json')
    bounded=E/'transported_objective/bounded_profile'
    sets=dict(native=entry['native_source'],binaries=entry['binaries'],reference=entry['reference'],
              protected=old['protected_files'],pre_C_controls={r['copy']:r['sha256'] for r in old['controls']},
              coupled_controls=read(E/'transported_objective/stage_c_coupled/entry.json')['saved_controls'],
              profile_controls=read(bounded/'entry.json')['saved_controls'],
              server=read(E/'objective_stress/bp3-input-manifest.json'),
              cleanup=read(ROOT/'benchmarks/reconstructed_fault/maintenance/disk_cleanup_2026_10_30/protected-critical-sha256.json'))
    for name in ['pre-support','support-admission']:
        r=read(bounded/(name+'.json'));sets[name]={r['core_copy']:r['core_sha256']}
    sets['pre_support_checkers']=read(bounded/'pre-support-checkers.json')
    preservation={}
    for key,records in sets.items():
        bad=[name for name,h in records.items() if not (ROOT/name).is_file() or sha(ROOT/name)!=(h['sha256'] if isinstance(h,dict) else h)]
        assert not bad,(key,bad)
        preservation[key]=dict(checked=len(records),mismatches=bad)
    assert sha(bounded/'final-provenance.json')==entry['previous_provenance_sha256']
    runs={name:read(RUNS/name/'command.json') for name in CASES}
    core='build-particle-free/native/aspect-release'
    plugin='build-particle-free/transported-objective/stage-c/checks/libtransported_material_energy.release.so'
    for name,run in runs.items():
        assert run['exit_code']==0 and run['core_sha256']==entry['binaries'][core]
        assert run['plugin_sha256']==entry['binaries'][plugin]
        args=run['arguments'];assert args['level']==6 and args['speed']==.4 and args['eta']=='1' and args['energy']
        assert args['profile_method']=='bounded MacCormack' and args['support']=='full normal cross-section' and args['inflow']=='accepted trace'
        assert args['initial_dt']==.1 and args['shear']==0 and args['ranks']==4
    result=read(OUT/'comparison.json');work={r['case']:r for r in read(OUT/'coupled-work.json')}
    flat=[]
    for r in result['cases']:
        flat.append(dict(case=r['case'],label_spacing=r['arguments'].get('spacing',.1),dt=r['arguments']['dt'],
            physical_steps=len(r['steps'])-1,wall_s=r['wall_s'],linear_solves=r['linear_solves'],linear_iterations=r['linear_iterations'],
            endpoint_signed=r['endpoint_signed'],endpoint_absolute=r['endpoint_absolute'],endpoint_relative=r['endpoint_relative'],
            trapezoid_signed=r['trapezoid_signed'],trapezoid_absolute=r['trapezoid_absolute'],trapezoid_relative=r['trapezoid_relative'],
            coupled_endpoint_relative=work[r['case']]['relative_active'],
            face_power=work[r['case']]['steps'][-1]['face_mechanical_power'],weak_power=work[r['case']]['steps'][-1]['weak_mechanical_power']))
    with (OUT/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]),lineterminator='\n');writer.writeheader();writer.writerows(flat)
    (OUT/'preservation.json').write_text(json.dumps(preservation,indent=2)+'\n')
    docs=['POSITIVE_TRANSLATION_REFINEMENT_REPORT.md','STATUS.md','SOURCE_AUDIT.md','RESEARCH_HANDOFF.md','REVIEW_PLAN.md','TRANSPORTED_OBJECTIVE_TASK_STATE.md','BOUNDED_PROFILE_TRANSPORT_REPORT.md','TRANSPORTED_OBJECTIVE_SPEC.md']
    files=[Path(__file__),Path(__file__).with_name('assess_positive_refinement.py'),Path(__file__).with_name('plot_positive_refinement.py'),Path(__file__).with_name('run_material_native.py')]
    files.extend(ROOT/'doc/reconstructed_fault/particle_free'/n for n in docs)
    files.append(ROOT/'doc/reconstructed_fault/CURRENT_STATUS.md')
    for name in CASES:
        files.extend(RUNS/name/n for n in ['command.json','input.prm','runner.py','fault.txt','run.log'])
        files.extend((RUNS/name/'output').glob('material-*.csv'))
    files.extend(p for p in OUT.iterdir() if p.is_file() and p.name!='final-provenance.json')
    records={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}
    record=dict(utc=datetime.now(timezone.utc).isoformat(),entry_head=entry['head'],
                recorded_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                root=str(ROOT),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
                unchanged_native_source=True,no_rebuild=True,maximum_sequential_MPI_ranks=4,
                runs=runs,preservation=preservation,input_code_report_evidence_sha256=records,
                data_losses=['Inherited loss of 39 of 45 historical animation frames.','Earlier temporary build logs lost in inherited cleanup.'],
                no_production_or_server_run=True)
    (OUT/'final-provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Verified unchanged native code, binaries, reference and all protected manifests; recorded two-run provenance.')

if __name__=='__main__':main()
