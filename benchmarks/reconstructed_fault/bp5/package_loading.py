"""Create the loading-driven package only after the two bounded runs pass."""
import json
import argparse
import shutil
import subprocess
import tarfile
from pathlib import Path
from startup_30km import HERE, BP3, ROOT, BIN, FIXTURE, parameters, render, digest
from run_loading_startup import OUT
from run_steady_startup import PLUGIN


def package(destination='server-30km-loading'):
    comparison=json.loads((OUT/'comparison.json').read_text())
    assert comparison['passed'] and all(comparison['screens'].values())
    for case in ('startup','half'):
        assert json.loads((OUT/case/'checks.json').read_text())['passed']
    tested=json.loads((OUT/'startup/launch.json').read_text())
    for p in (BIN,PLUGIN):assert digest(p)==tested['hashes'][str(p)]
    dest=HERE/destination; dest.mkdir()
    (dest/'evidence').mkdir()
    for name,target in [('server_environment.sh','environment.sh'),('verify_server_package.py','verify_package.py'),
                        ('loading_server_README.md','README.md')]:
        shutil.copy2(HERE/name,dest/target)
    (dest/'first_event.slurm').write_text((HERE/'first_event.slurm').read_text().replace(
        'libbp5_initialization.release.so','libbp5_steady_initialization.release.so').replace('BP5-30km','BP5-loading'))
    (dest/'fixture').mkdir()
    values=parameters((OUT/'startup/run.prm').read_text())
    for key,name in [(('Fault reconstruction','Prescribed faults file'),'fault.txt'),
                     (('Mesh refinement','BP3 saved mesh','Target cells file'),'target_cells.txt'),
                     (('Postprocess','BP3','Bottom normalization completion file'),'completion.txt')]:
        shutil.copy2(values[key],dest/'fixture'/name)
    values.update({('Additional shared libraries',):'./libbp5_steady_initialization.release.so',
        ('Output directory',):destination.replace('server-','output-',1),('Resume computation',):'false',
        ('Maximum time step',):'1e7',('End time',):'47336400000',
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
    (dest/'first_event.prm').write_text(render(values))
    values['Resume computation',]='true'; (dest/'resume.prm').write_text(render(values))
    (dest/'plugin/bp3').mkdir(parents=True);(dest/'plugin/bp5').mkdir()
    sources=[BP3/name for name in ('bp3.cc','bp3_model.h','first_event.h','output_schedule.h',
                                  'mature_fault.h','work_replay.h','matched_resolution.h')]
    for p in sources:shutil.copy2(p,dest/'plugin/bp3'/p.name)
    for name in ('steady_initialization.h','startup_time_step.cc'):
        p=HERE/name;sources.append(p);shutil.copy2(p,dest/'plugin/bp5'/name)
    (dest/'plugin/bp5/CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.13.4)
find_package(Aspect REQUIRED HINTS ${Aspect_DIR})
DEAL_II_INITIALIZE_CACHED_VARIABLES()
project(bp5_loading_initialization)
add_library(bp5_steady_initialization SHARED ../bp3/bp3.cc startup_time_step.cc)
target_compile_definitions(bp5_steady_initialization PRIVATE ASPECT_BP5_STEADY_INITIALIZATION)
ASPECT_SETUP_PLUGIN(bp5_steady_initialization)
''')
    patch=subprocess.check_output(['git','diff','--binary','HEAD','--','source','include','unit_tests',
        'tests/phase_field_fault_test_access.h','CMakeLists.txt','cmake'],cwd=ROOT)
    (dest/'source.patch').write_bytes(patch)
    sources += [HERE/name for name in ('run_loading_startup.py','analyze_loading_startup.py','package_loading.py',
        'test_steady_initialization.cc','run_steady_large_step.py','run_steady_startup.py','startup_30km.py',
        'check_startup_30km.py','loading_initialization_report.md')]
    sources += [BP3/name for name in ('run_coupled_substeps.py','check_first_cycle_restart.py','analyze_length_coupled.py')]
    panels=int(values.get(('Material model','Phase field fault','I h surface quadrature subdivisions'),'1'))
    if panels!=1:
        sources += [HERE/name for name in ('run_loading_surface.py','run_surface_quadrature.py',
            'analyze_surface_quadrature.py','reintegrate_initial_profile.py','loading_surface_quadrature_report.md',
            'first_event_source_files.md')]
        shutil.copy2(HERE/'loading_surface_quadrature_report.md',dest/'evidence/surface_quadrature_report.md')
        shutil.copy2(HERE/'first_event_source_files.md',dest/'SOURCE_FILES.md')
        readme=(dest/'README.md').read_text().replace('output-30km-loading/',values['Output directory',]+'/')
        readme+='''
## Eight-panel normalization qualification

This package explicitly sets `I h surface quadrature subdivisions = 8` and
includes completion data for all 24 surface origins per element. Do not use the
old three-point completion file. Physical weakening remains 0–30 km and the
transition 30–33 km. The authoritative new startup/half-step evidence is
`evidence/report.md`; the earlier loading report is historical context only.
The existing remote backend is retained. Budget at least 32 GiB per four-rank
job (more is prudent); the larger cold lookup cache has been exercised locally,
but no claim about multi-event peak memory is made. Leave normal-control,
mechanical-probe and fine-grained profiling flags unset.
'''
        (dest/'README.md').write_text(readme)
    with tarfile.open(dest/'source-overlay.tar.gz','w:gz') as archive:
        for p in sources:archive.add(p,arcname=str(p.relative_to(ROOT)))
    (dest/'provenance').mkdir();shutil.copy2(PLUGIN,dest/'provenance/qualified-local-plugin.release.so')
    for p in [HERE/'loading_initialization_report.md',OUT/'comparison.json',OUT/'loading-whole.png',OUT/'loading-transition.png']:
        shutil.copy2(p,dest/'evidence'/p.name)
    if (OUT/'report.md').exists():
        shutil.copy2(OUT/'report.md',dest/'evidence/report.md')
    for case in ('startup','half'):
        target=dest/'evidence'/case;target.mkdir()
        for name in ('run.prm','launch.json','execution.json','checks.json','accepted_steps.csv',
                     'timestep_selection.csv','state_startup_predictor.csv','checkpoint_source.json',
                     'steady_initialization.csv','first_update_maxwell.csv'):
            p=OUT/case/name
            if p.exists():shutil.copy2(p,target/name)
    (dest/'qualification.json').write_text(json.dumps(dict(launch_approved=True,
        initialization='projected-mixture R_VW=0.8; native weak variable prestress',
        physical_ceiling_s=1e7,first_step_cap_s=1e6,artificial_interval_s=1e6,
        normalization_surface_panels=panels,
        predictor_limit=.02,tested_mpi_ranks=4,comparison=comparison,
        scope='Bounded startup and checkpoint subdivision; NOT full-cycle temporal qualification.'),indent=2)+'\n')
    manifest=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        local_binary_sha256=digest(BIN),local_plugin_sha256=digest(PLUGIN),
        files={str(p.relative_to(dest)):digest(p) for p in dest.rglob('*') if p.is_file()})
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(dest)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path)
    parser.add_argument('--destination',default='server-30km-loading')
    args=parser.parse_args()
    if args.study:OUT=args.study.resolve()
    package(args.destination)
