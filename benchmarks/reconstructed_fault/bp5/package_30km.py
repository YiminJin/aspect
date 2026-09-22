"""Freeze portable first-event inputs and exact source provenance; never launch."""
import hashlib
import ast
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from startup_30km import HERE, ROOT, BP3, BIN, PLUGIN, STUDY, FIXTURE, parameters, render, digest


def package():
    # Keep server-30km immutable: the preserved failure replay refers to its
    # exact binary by path/hash. This is the newly qualified adaptive package.
    destination=HERE/'server-30km-adaptive'
    followup=STUDY/'startup-followup'
    checks={name:json.loads((followup/name).read_text()) for name in
            ['temporal_300s.json','adaptive_checks.json','restart_checks.json']}
    assert all(check['passed'] for check in checks.values())
    tested=json.loads((followup/'adaptive/launch.json').read_text())
    for path in [BIN,PLUGIN]:assert digest(path)==tested['hashes'][str(path)]
    destination.mkdir()
    shutil.copy2(HERE/'server_environment.sh',destination/'environment.sh')
    shutil.copy2(HERE/'first_event.slurm',destination/'first_event.slurm')
    shutil.copy2(HERE/'server_README.md',destination/'README.md')
    shutil.copy2(HERE/'verify_server_package.py',destination/'verify_package.py')
    inputs=destination/'fixture';inputs.mkdir()
    for path in FIXTURE.glob('*.txt'):shutil.copy2(path,inputs/path.name)
    shutil.copy2(FIXTURE/'manifest.json',inputs/'generation.json')
    values=parameters((followup/'adaptive/run.prm').read_text())
    values['Additional shared libraries',]='./libbp5_initialization.release.so'
    values['Output directory',]='output-30km'
    values['Fault reconstruction','Prescribed faults file']='fixture/fault.txt'
    values['Mesh refinement','BP3 saved mesh','Target cells file']='fixture/target_cells.txt'
    values['Postprocess','BP3','Mature prestress file']='fixture/prestress.txt'
    values['Postprocess','BP3','Bottom normalization completion file']='fixture/completion.txt'
    values['End time',]='47336400000'  # 1500 Julian years; not event success.
    values['Postprocess','BP3','Stop after first event']='true'
    values['Postprocess','BP3','Last accepted step']='2147483647'
    values['Postprocess','BP3','Graceful wall seconds']='84600'
    values['Postprocess','BP3','Audit full state every step']='false'
    values['Postprocess','BP3','Profile time interval']='31557600'
    values['Postprocess','BP3','Profile slip interval']='0.01'
    values['Checkpointing','Steps between checkpoint']='0'
    values['Checkpointing','Time between checkpoint']='3600'
    values['Termination criteria','Checkpoint on termination']='true'
    assert values['Maximum time step',]=='4e6'
    assert values['Time stepping','BP5 state startup','Maximum logarithmic state change']=='0.02'
    (destination/'first_event.prm').write_text(render(values))
    values['Resume computation',]='true'
    (destination/'resume.prm').write_text(render(values))

    # Preserve the existing relative include layout, but include only headers
    # used by the normal plugin, not optional investigation/test code.
    (destination/'plugin/bp3').mkdir(parents=True)
    (destination/'plugin/bp5').mkdir()
    for name in ['bp3.cc','bp3_model.h','first_event.h','output_schedule.h',
                 'mature_fault.h','work_replay.h','matched_resolution.h']:
        shutil.copy2(BP3/name,destination/'plugin/bp3'/name)
    for name in ['CMakeLists.txt','weak_initialization.h','steady_initialization.h','startup_time_step.cc']:
        shutil.copy2(HERE/name,destination/'plugin/bp5'/name)

    # HEAD plus the patch reconstructs all tracked source, including the
    # previously accepted dirty changes required by this plugin. Overlay the
    # explicitly selected untracked sources, not historical run directories.
    source_paths=['source','include','unit_tests','CMakeLists.txt','cmake',
                  'doc/reconstructed_fault/current_design.md',
                  'benchmarks/reconstructed_fault/bp3/bp3.cc',
                  'benchmarks/reconstructed_fault/bp3/bp3_model.h',
                  'benchmarks/reconstructed_fault/bp3/work_replay.h']
    patch=subprocess.check_output(['git','diff','--binary','HEAD','--']+source_paths,cwd=ROOT)
    (destination/'source.patch').write_bytes(patch)
    sources=list(BP3.glob('*.h'))+[BP3/'bp3.cc',HERE/'CMakeLists.txt',HERE/'weak_initialization.h',HERE/'steady_initialization.h',
        HERE/'startup_time_step.cc',
        HERE/'startup_30km.py',HERE/'check_startup_30km.py',Path(__file__).resolve(),
        HERE/'server_environment.sh',HERE/'first_event.slurm',
        HERE/'server_README.md',HERE/'verify_server_package.py',HERE/'analyze_startup_30km.py',
        HERE/'startup_30km_report.md',HERE/'startup_followup_report.md',HERE/'small_startup_report.md',
        HERE/'run_startup_followup.py',HERE/'analyze_startup_followup.py',
        BP3/'test_first_event.cc']
    # Include only the local Python dependencies needed to read/reproduce the
    # saved qualification, without copying experiment directories or binaries.
    pending=[p for p in sources if p.suffix=='.py'];seen=set(sources)
    while pending:
        for node in ast.walk(ast.parse(pending.pop().read_text())):
            if isinstance(node,ast.ImportFrom) and node.module:
                for directory in [HERE,BP3]:
                    dependency=directory/(node.module.replace('.','/')+'.py')
                    if dependency.is_file() and dependency not in seen:
                        seen.add(dependency);sources.append(dependency);pending.append(dependency)
    with tarfile.open(destination/'source-overlay.tar.gz','w:gz') as archive:
        for path in sorted(set(sources)):archive.add(path,arcname=str(path.relative_to(ROOT)))
    (destination/'provenance').mkdir()
    shutil.copy2(PLUGIN,destination/'provenance/qualified-local-plugin.release.so')
    evidence=destination/'evidence';evidence.mkdir()
    for path in [STUDY/'startup_analysis.json',STUDY/'startup/execution.json',
                 STUDY/'startup/accepted_prefix_checks.json',STUDY/'startup/accepted_steps.csv']:
        if path.exists():shutil.copy2(path,evidence/path.name)
    shutil.copytree(STUDY/'verification',evidence/'verification')
    for name in checks:shutil.copy2(followup/name,evidence/name)
    for name in ['startup_followup_report.md','small_startup_report.md','startup_30km_report.md']:
        shutil.copy2(HERE/name,evidence/name)
    for case in ['75','adaptive','resume']:
        (evidence/case).mkdir()
        for name in ['execution.json','launch.json','run.prm','accepted_steps.csv','state_startup_predictor.csv']:
            shutil.copy2(followup/case/name,evidence/case/name)
    (destination/'qualification.json').write_text(json.dumps(dict(
        launch_approved=True,checks={name:check['passed'] for name,check in checks.items()},
        tested_mpi_ranks=4,physical_ceiling_s=4e6,predictor_limit=0.02,
        first_physical_dt_s=66.964271709050308,
        explanation='Regenerated at user request after passing bounded adaptive startup and same-rank restart. '
                    'Not a full-cycle or spatial-accuracy qualification.'),indent=2)+'\n')
    record=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_patch_sha256=hashlib.sha256(patch).hexdigest(),
        local_binary_sha256=digest(BIN),local_plugin_sha256=digest(PLUGIN),
        source_files={str(p.relative_to(ROOT)):digest(p) for p in sources},
        files={str(p.relative_to(destination)):digest(p) for p in destination.rglob('*') if p.is_file()})
    (destination/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(destination)


if __name__=='__main__':package()
