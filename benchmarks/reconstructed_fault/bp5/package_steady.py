"""Package only a passing steady-state timestep comparison; never launch ASPECT."""
import argparse
import ast
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from startup_30km import HERE, BP3, ROOT, BIN, STUDY, FIXTURE, parameters, render, digest
from run_steady_startup import PLUGIN
from run_steady_large_step import OUT


def package(one,half):
    comparison=json.loads((OUT/f'comparison-{half}.json').read_text())
    assert comparison['one']==one and comparison['identical_incoming_state']
    for field in ('V','Theta_out','weak_q','weak_sigma','slip'):
        assert comparison['metrics'][field]['error_over_change_RMS']<=.05,field
    for case in ('startup',one,half):
        assert json.loads((OUT/case/'checks.json').read_text())['passed']
    assert json.loads((STUDY/'steady-startup/comparisons.json').read_text())['passed']
    tested=json.loads((OUT/'startup/launch.json').read_text())
    for p in (BIN,PLUGIN):assert digest(p)==tested['hashes'][str(p)]

    dest=HERE/'server-30km-steady'
    dest.mkdir()  # Never replace another package, including an earlier failed attempt.
    for name,target in [('server_environment.sh','environment.sh'),('verify_server_package.py','verify_package.py')]:
        shutil.copy2(HERE/name,dest/target)
    (dest/'first_event.slurm').write_text((HERE/'first_event.slurm').read_text().replace(
        'libbp5_initialization.release.so','libbp5_steady_initialization.release.so'))
    inputs=dest/'fixture';inputs.mkdir()
    for name in ('fault.txt','target_cells.txt','completion.txt'):
        shutil.copy2(FIXTURE/name,inputs/name)
    values=parameters((OUT/'startup/run.prm').read_text())
    ceiling=parameters((OUT/one/'run.prm').read_text())['Maximum time step',]
    values.update({('Additional shared libraries',):'./libbp5_steady_initialization.release.so',
        ('Output directory',):'output-30km-steady',('Resume computation',):'false',
        ('Maximum time step',):ceiling,('End time',):'47336400000',
        ('Fault reconstruction','Prescribed faults file'):'fixture/fault.txt',
        ('Mesh refinement','BP3 saved mesh','Target cells file'):'fixture/target_cells.txt',
        ('Postprocess','BP3','Mature prestress file'):'',
        ('Postprocess','BP3','Bottom normalization completion file'):'fixture/completion.txt',
        ('Postprocess','BP3','Stop after first event'):'true',
        ('Postprocess','BP3','Last accepted step'):'2147483647',
        ('Postprocess','BP3','Graceful wall seconds'):'84600',
        ('Postprocess','BP3','Audit full state every step'):'false',
        ('Postprocess','BP3','Profile time interval'):'31557600',
        ('Postprocess','BP3','Profile slip interval'):'0.01',
        ('Checkpointing','Steps between checkpoint'):'0',
        ('Checkpointing','Time between checkpoint'):'3600',
        ('Termination criteria','Checkpoint on termination'):'true'})
    assert values['Time stepping','BP5 state startup','Maximum logarithmic state change']=='0.02'
    (dest/'first_event.prm').write_text(render(values))
    values['Resume computation',]='true'
    (dest/'resume.prm').write_text(render(values))
    (dest/'README.md').write_text((HERE/'steady_server_README.md').read_text().replace('@CEILING@',ceiling))

    (dest/'plugin/bp3').mkdir(parents=True)
    (dest/'plugin/bp5').mkdir()
    sources=[BP3/name for name in ('bp3.cc','bp3_model.h','first_event.h','output_schedule.h',
                                 'mature_fault.h','work_replay.h','matched_resolution.h')]
    for p in sources:shutil.copy2(p,dest/'plugin/bp3'/p.name)
    for name in ('steady_initialization.h','startup_time_step.cc'):
        p=HERE/name;sources.append(p);shutil.copy2(p,dest/'plugin/bp5'/name)
    (dest/'plugin/bp5/CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.13.4)
find_package(Aspect REQUIRED HINTS ${Aspect_DIR})
DEAL_II_INITIALIZE_CACHED_VARIABLES()
project(bp5_steady_initialization)
add_library(bp5_steady_initialization SHARED ../bp3/bp3.cc startup_time_step.cc)
target_compile_definitions(bp5_steady_initialization PRIVATE ASPECT_BP5_STEADY_INITIALIZATION)
ASPECT_SETUP_PLUGIN(bp5_steady_initialization)
''')
    patch=subprocess.check_output(['git','diff','--binary','HEAD','--','source','include','unit_tests',
        'tests/phase_field_fault_test_access.h','CMakeLists.txt','cmake'],cwd=ROOT)
    (dest/'source.patch').write_bytes(patch)
    sources += [HERE/name for name in ('CMakeLists.txt','weak_initialization.h','test_steady_initialization.cc',
        'run_steady_startup.py','analyze_steady_startup.py','run_steady_large_step.py',
        'analyze_steady_large_step.py','package_steady.py','steady_large_step_report.md',
        'steady_startup_report.md','steady_server_README.md','server_environment.sh','first_event.slurm',
        'verify_server_package.py')]
    pending=[p for p in sources if p.suffix=='.py'];seen=set(sources)
    while pending:
        for node in ast.walk(ast.parse(pending.pop().read_text())):
            if isinstance(node,ast.ImportFrom) and node.module:
                for folder in (HERE,BP3):
                    p=folder/(node.module.replace('.','/')+'.py')
                    if p.is_file() and p not in seen:
                        seen.add(p);sources.append(p);pending.append(p)
    with tarfile.open(dest/'source-overlay.tar.gz','w:gz') as archive:
        for p in sorted(set(sources)):archive.add(p,arcname=str(p.relative_to(ROOT)))
    (dest/'provenance').mkdir()
    shutil.copy2(PLUGIN,dest/'provenance/qualified-local-plugin.release.so')
    evidence=dest/'evidence';evidence.mkdir()
    for name in ('steady_large_step_report.md','steady_startup_report.md'):
        shutil.copy2(HERE/name,evidence/name)
    shutil.copy2(STUDY/'steady-startup/comparisons.json',evidence/'steady_restart_comparisons.json')
    for p in OUT.glob('comparison-*.json'):shutil.copy2(p,evidence/p.name)
    for case in ('startup','half4m','one1m','half1m',one,half):
        target=evidence/case
        if target.exists():continue
        target.mkdir()
        for name in ('run.prm','launch.json','execution.json','checks.json','accepted_steps.csv',
                     'timestep_selection.csv','state_startup_predictor.csv','checkpoint_source.json'):
            p=OUT/case/name
            if p.exists():shutil.copy2(p,target/name)
    (dest/'qualification.json').write_text(json.dumps(dict(launch_approved=True,
        initialization='steady Dc/Vinit with native weak variable prestress',physical_ceiling_s=float(ceiling),
        predictor_limit=.02,tested_mpi_ranks=4,temporal_screen_relative_to_evolving_change=.05,
        comparison=comparison,scope='Bounded startup/comparison and same-rank restart, not full-cycle accuracy.'),indent=2)+'\n')
    manifest=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        local_binary_sha256=digest(BIN),local_plugin_sha256=digest(PLUGIN),source_patch_sha256=digest(dest/'source.patch'),
        source_files={str(p.relative_to(ROOT)):digest(p) for p in sources},
        files={str(p.relative_to(dest)):digest(p) for p in dest.rglob('*') if p.is_file()})
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(dest)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('one');parser.add_argument('half')
    args=parser.parse_args();package(args.one,args.half)
