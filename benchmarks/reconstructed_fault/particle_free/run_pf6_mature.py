#!/usr/bin/env python3
"""Bounded fresh stationary 60-degree mature history comparison; no phase evolution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten(path, root):
    lines = []
    for line in path.read_text().replace('$ASPECT_SOURCE_DIR', str(root)).splitlines():
        if line.strip().startswith('include '):
            lines.append(flatten(Path(line.strip()[8:]), root))
        elif not re.match(r'\s*set Additional shared libraries\s*=', line):
            lines.append(line)
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('label')
    p.add_argument('--backend', choices=('dg', 'fe', 'particle'), default='dg')
    p.add_argument('--operator', action='store_true')
    p.add_argument('--quadrature', type=int, choices=(3, 6, 8), default=3)
    p.add_argument('--dt', type=float, default=200000)
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--refinement', type=int, choices=(6, 7, 8), default=6)
    p.add_argument('--seconds', type=float, default=700)
    p.add_argument('--build-directory', type=Path, default=Path('build-particle-free/mature-dg'))
    p.add_argument('--phase-reference', type=Path)
    p.add_argument('--common-points', type=Path)
    p.add_argument('--ranks', type=int, choices=(1,2,4), default=1)
    p.add_argument('--accepted', action='store_true', help='Native accepted mature DG lifecycle/checkpoint route')
    p.add_argument('--checkpoint', action='store_true')
    p.add_argument('--restart', type=Path)
    p.add_argument('--outer-rejection', action='store_true')
    p.add_argument('--candidate-rejection', action='store_true')
    p.add_argument('--nonlinear-rejection', action='store_true')
    p.add_argument('--audit-all', action='store_true')
    p.add_argument('--schedule', help='Prescribed timestep function for a matching cutback control')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[3]
    source = root/'benchmarks/reconstructed_fault/bp3_native'
    build = (root/a.build_directory).resolve()
    out = build/'runs'/a.label
    out.mkdir(parents=True, exist_ok=False)
    binary = root/'build-particle-free/native/aspect-release'
    plugin = build/'bp3-plugin/libbp3_restore_150x50.release.so'
    observer = build/('observers/libbp3_native_checks.release.so' if a.operator
                      else 'observers/libbp3_coupled_checks.release.so')
    artifacts = [binary, plugin, observer]
    for f in artifacts:
        shutil.copy2(f, out/f.name)
    fixture = source/('history-graded-q6.prm' if a.operator else 'section7b-fe-perturbed.prm')
    text = flatten(fixture, root)
    if a.operator:
        # Keep the existing no-Stokes kinematic fixture and its direct function
        # phase/Theta prescriptions, using strengthening material everywhere.
        text = text.replace('set Direct effect parameters = 0.010, 0.025',
                            'set Direct effect parameters = 0.025, 0.025')
    mode = {'dg': 'DG trial stress', 'fe': 'accepted FE strain', 'particle': 'particle stress'}[a.backend]
    text += f'''
set Additional shared libraries = {out/plugin.name}, {out/observer.name}
set Output directory = {out/'run'}
set End time = {0 if a.operator else a.steps*a.dt:.17g}
set Maximum time step = {a.dt:.17g}
set Maximum first time step = {a.dt:.17g}
subsection Material model
  subsection Phase field fault
    set Bulk history representation = {mode}
    set FE bulk normal quadrature points = {a.quadrature}
    set Initial time step = 200000
    set Particle stress strain convention = three dimensional deviator
  end
end
subsection Time stepping
  set List of model names = function
  subsection Function
    set Function expression = {a.dt:.17g}
  end
end
subsection Mesh refinement
  set Initial global refinement = {a.refinement}
end
subsection Postprocess
  subsection BP3
    set Last accepted step = {0 if a.operator else a.steps}
    set Graceful wall seconds = {max(1,a.seconds-10):.17g}
  end
'''
    if not a.operator:
        text += '''  subsection BP3 coupled checks
    set Write authoritative stress = true
    set Write FE coefficients = true
  end
'''
    if a.common_points:
        text += f'''  subsection BP3 coupled checks
    set Common stress points file = {(root/a.common_points).resolve()}
  end
'''
    text += 'end\n'
    if a.phase_reference:
        text += f'''
subsection Initial composition model
  subsection BP3
    set Fixed phase reference file = {(root/a.phase_reference).resolve()}
  end
end
'''
    if a.backend != 'particle':
        text += '''
subsection Material model
  subsection Phase field fault
    set Use homogeneous mesh data = true
  end
end
subsection Compositional fields
  set Compositional field methods = static, static, static, static, static
  set Mapped particle properties =
end
subsection Particles
  set Generate particle domains = false
  set Generate CPDI data for particle domains = false
  set Generate face data for particle domains = false
end
'''
    if a.accepted:
        assert a.backend=='dg' and not a.phase_reference and not a.operator
        text += """
subsection Material model
 subsection Phase field fault
  set Enable prescribed DG accepted history = true
 end
end
subsection Postprocess
 subsection BP3 coupled checks
  set Mature lifecycle checks = true
  set Prescribed schedule check = false
 end
end
"""
    assert a.ranks==1 or a.accepted
    assert not a.restart or a.accepted
    if a.checkpoint:
        text += '\nsubsection Termination criteria\n set Checkpoint on termination = true\nend\n'
    if a.restart:
        text += '\nset Resume computation = true\n'
        shutil.copytree(a.restart.resolve(),out/'run/restart')
    if a.outer_rejection:
        text += '\nsubsection Time stepping\n set List of model names = function, BP3 one outer rejection\nend\n'
    if a.candidate_rejection:
        text += '\nsubsection Postprocess\n subsection BP3 coupled checks\n  set Inject candidate failure = true\n end\nend\n'
    if a.audit_all:
        text += '\nsubsection Postprocess\n subsection BP3\n  set Audit full state every step = true\n end\nend\n'
    if a.nonlinear_rejection:
        assert a.accepted
        text += '\nset Nonlinear solver failure strategy = cut timestep size\nsubsection Time stepping\n set List of model names = function, repeat on nonlinear solver failure\nend\nsubsection Postprocess\n subsection BP3 coupled checks\n  set Inject nonlinear failure = true\n end\nend\n'
    if a.schedule:
        text += f'\nsubsection Time stepping\n subsection Function\n  set Variable names = time\n  set Function expression = {a.schedule}\n end\nend\n'
    prm = out/'input.prm'
    prm.write_text(text)
    tracked = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard',
        'source','include','benchmarks/reconstructed_fault/bp3_native','tests/phase_field_fault_test_access.h'],
        cwd=root,text=True).splitlines()
    record = dict(arguments={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(a).items()}, head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        artifacts={str(f.relative_to(root)):sha(f) for f in artifacts}, input_sha256=sha(prm),
        source_hashes={f:sha(root/f) for f in sorted(set(tracked)) if (root/f).is_file()}, ranks=a.ranks, threads=1,
        command=[str(out/binary.name),str(prm)])
    if a.ranks>1:
        record['command']=['/opt/openmpi/5.0.6/bin/mpirun','--map-by','core','--bind-to','core','-np',str(a.ranks)]+record['command']
    record['external_inputs']={str((root/f).resolve()):sha((root/f).resolve())
                               for f in (a.phase_reference,a.common_points) if f}
    (out/'source.patch').write_text(subprocess.check_output(['git','diff','--','source','include',
        'benchmarks/reconstructed_fault/bp3_native'],cwd=root,text=True))
    env = os.environ.copy()
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','DEAL_II_NUM_THREADS'):
        env[k] = '1'
    # Match the recorded GCC12/OpenMPI5.0.6/deal.II dependency stack; the
    # launcher's PATH alone does not select libraries used by the executable.
    env['LD_LIBRARY_PATH']='/opt/openmpi/5.0.6/lib:'+env.get('LD_LIBRARY_PATH','')
    env['OMP_PROC_BIND']='false'
    env['OMPI_MCA_mpi_yield_when_idle']='1'
    record['runtime_environment']={k:env[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','DEAL_II_NUM_THREADS','LD_LIBRARY_PATH','OMP_PROC_BIND','OMPI_MCA_mpi_yield_when_idle')}
    start=time.monotonic()
    with (out/'run.log').open('w') as log:
        try:
            code=subprocess.run(record['command'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,
                                timeout=a.seconds,check=False).returncode
        except subprocess.TimeoutExpired:
            code=124
    record.update(exit_code=code,seconds=time.monotonic()-start,
                  max_rss_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                  csv_hashes={f.name:sha(f) for f in (out/'run').glob('*.csv')})
    (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ('exit_code','seconds','max_rss_KiB')}),flush=True)
    raise SystemExit(code)


if __name__=='__main__':
    main()
