#!/usr/bin/env python3
"""Prepare a bounded stationary DG BP3 pilot; never launch a simulation.

Run with a fresh output directory. ASPECT/plugin builds must use one compatible
MPI/deal.II stack; the two libraries must come from the same supplied source.
"""
import argparse,hashlib,json,shutil,subprocess,tarfile
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('directory',type=Path)
p.add_argument('--aspect',type=Path,required=True)
p.add_argument('--bp3-library',type=Path,required=True)
p.add_argument('--observer-library',type=Path,required=True)
p.add_argument('--qualification-report',type=Path,required=True)
a=p.parse_args()
qualification=json.loads(a.qualification_report.read_text())
assert qualification['functional_checks_passed'], 'Functional qualification is incomplete.'
root=Path(__file__).resolve().parents[4];out=a.directory.resolve();out.mkdir(parents=True,exist_ok=False)
source=root/'benchmarks/reconstructed_fault/bp3_native/server'
text=(source/'parameters.prm').read_text();text='\n'.join(line for line in text.splitlines() if not line.startswith('#'))+'\n'
shutil.copy2(a.qualification_report,out/'qualification-summary.json')
for name in ('launch.py','diagnose.py','README.md'):
    shutil.copy2(Path(__file__).with_name(name),out/name)
shutil.copy2(Path(__file__).parents[1]/'export_pf6_mature_tractions.py',out/'export_tractions.py')
artifacts={}
for path in (a.aspect,a.bp3_library,a.observer_library):
    path=path.resolve();shutil.copy2(path,out/path.name);artifacts[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
text=text.replace('set Additional shared libraries = ./plugin/build/libbp3_restore_150x50.release.so',
    f'set Additional shared libraries = {out/a.bp3_library.name}, {out/a.observer_library.name}')
text=text.replace('set Output directory = ./output',f'set Output directory = {out/"output"}')
text=text.replace('set Prescribed faults file = ./fault.txt',f'set Prescribed faults file = {out/"fault.txt"}')
# Preserve physical initial interval, full geometry, mature law, Gauss6,
# structural spacing, profile, native state controller, filter and particle layout.
text=text.replace('set Bulk history representation = accepted FE strain','set Bulk history representation = DG trial stress')
text=text.replace('set Generate particle domains = true','set Generate particle domains = false')
text=text.replace('set Generate CPDI data for particle domains = true','set Generate CPDI data for particle domains = false')
text=text.replace('set Compositional field methods = particles, particles, particles, particles, particles','set Compositional field methods = static, static, static, static, static')
text='\n'.join('  set Mapped particle properties =' if line.strip().startswith('set Mapped particle properties') else line for line in text.splitlines())+'\n'
text += '''
set End time = 400000
set Maximum first time step = 50000
set Maximum time step = 50000
subsection Material model
 subsection Phase field fault
  set Use homogeneous mesh data = true
  set Enable prescribed DG accepted history = true
  set Particle stress strain convention = three dimensional deviator
 end
end
subsection Particles
 set Generate face data for particle domains = false
end
subsection Termination criteria
 set Checkpoint on termination = true
end
subsection Checkpointing
 set Steps between checkpoint = 1
end
subsection Time stepping
 set List of model names = convection time step, reconstructed fault time step, repeat on nonlinear solver failure
end
set Nonlinear solver failure strategy = cut timestep size
subsection Postprocess
 set List of postprocessors = memory statistics, reconstructed fault BP3, particles, BP3 output complete, BP3 restored monitor, BP3 serial coupled checks
 subsection BP3
  set Stop after first event = false
  set Last accepted step = 8
  set Graceful wall seconds = 7200
 end
 subsection BP3 coupled checks
  set Mature lifecycle checks = true
  set Prescribed schedule check = false
  set Write authoritative stress = true
  set Write FE coefficients = true
 end
end
'''
shutil.copy2(source/'fault.txt',out/'fault.txt');(out/'parameters.prm').write_text(text)
restart=text.replace('set Resume computation = false','set Resume computation = true')
(out/'resume.prm').write_text(restart)
# A supervisor enforces the hard wall limit; the BP3 wall stop acts only after
# acceptance, and native nonlinear cutback has its own bounded repeat count.
argv=['mpirun','--bind-to','core','-np','RANKS',str(out/a.aspect.name),str(out/'parameters.prm')]
paths=subprocess.check_output(['git','ls-files','-c','-o','--exclude-standard','source','include','cmake','CMakeLists.txt','unit_tests','tests','benchmarks/reconstructed_fault/bp3_native','benchmarks/reconstructed_fault/particle_free'],cwd=root,text=True).splitlines()
record=dict(executable=a.aspect.name,qualification_status='bounded fixed-geometry controls passed',strict_reference_match_passed=False,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 artifacts=artifacts,inputs={q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in (out/'parameters.prm',out/'resume.prm',out/'fault.txt')},
 sources={s:hashlib.sha256((root/s).read_bytes()).hexdigest() for s in sorted(set(paths)) if (root/s).is_file()},
 command_template=argv,hard_supervisor_seconds=7500,scope='PREPARED; not a full event or a continuum accuracy qualification',
 limits=dict(positive_steps=8,end_seconds=400000,graceful_wall_seconds=7200),
 fixed=dict(limiter='unlimited DGQ2',bulk_quadrature='Gauss6',source_quadrature='Gauss3',filter='Helmholtz 20 m',
 initial_constitutive_interval_seconds=4000000,maximum_timestep_seconds=50000,finest_m=3.90625,geometry='150x50 km, 60 degree mature fault'))
record['inputs'].update({name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ('launch.py','diagnose.py','export_tractions.py','README.md','qualification-summary.json')})
(out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
(out/'source.patch').write_bytes(subprocess.check_output(['git','diff','--binary','--','source','include','cmake','CMakeLists.txt','unit_tests','tests','benchmarks/reconstructed_fault/bp3_native','benchmarks/reconstructed_fault/particle_free'],cwd=root))
untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard','source','include','cmake','unit_tests','tests','benchmarks/reconstructed_fault/bp3_native','benchmarks/reconstructed_fault/particle_free'],cwd=root,text=True).splitlines()
# Include only source/build inputs, never large evidence or build outputs.
with tarfile.open(out/'untracked-sources.tar.xz','w:xz',preset=3) as archive:
    for name in untracked:
        path=root/name
        if path.is_file() and (path.suffix in ('.cc','.h','.prm','.cmake','.py','.md') or path.name=='CMakeLists.txt') and '/evidence/' not in name:
            archive.add(path,arcname=name)
record['source_payload']={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ('source.patch','untracked-sources.tar.xz')}
(out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ('command_template','limits','fixed')},indent=2))
