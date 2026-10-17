#!/usr/bin/env python3
"""Prepare/finalize one bounded native fixed-reference transfer run.

Launch the printed argv with refactoring_boundary/run_logged.py (local MPI
requires host sockets). No physical or limiter parameters are tuned here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from run_fixed_profile import fixture_text


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'finalize'])
    p.add_argument('label')
    p.add_argument('--ranks', type=int, choices=[1, 2, 4], default=1)
    p.add_argument('--end', type=float, choices=[.125, .25], default=.25)
    p.add_argument('--restart', type=Path)
    p.add_argument('--no-transfer', action='store_true')
    p.add_argument('--checkpoint', action='store_true')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[3]
    out = root/'build-particle-free/runs'/a.label
    logroot = root/'benchmarks/reconstructed_fault/refactoring_boundary/evidence'
    if a.action == 'finalize':
        r = json.loads((out/'record.json').read_text())
        r.update(json.loads((logroot/(a.label+'.json')).read_text()))
        shutil.copy2(logroot/(a.label+'.log'), out/'run.log')
        r['outputs'] = {str(f.relative_to(out/'run')): sha(f)
                        for f in (out/'run').glob('*') if f.is_file()}
        (out/'record.json').write_text(json.dumps(r, indent=2)+'\n')
        print(json.dumps({k: r[k] for k in ['exit_code', 'seconds', 'ranks']}))
        return
    out.mkdir(parents=True, exist_ok=False)
    binary = root/'build-particle-free/native/aspect-release'
    dso = root/'build-particle-free/plugin/libpf6_transfer_checks.release.so'
    for f in [binary, dso]:
        shutil.copy2(f, out/f.name)
    observer = Path(__file__).with_name('pf6_transfer_checks.cc')
    shutil.copy2(observer, out/'observer-source.cc')
    # Physical fixture matches the passed serial gate, with explicitly fixed RK2.
    prm = '''include $ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/particle_free/pf6a-common.prm
set End time = END_TIME
set Maximum time step = 0.125
set Maximum first time step = 0.125
subsection Geometry model
 subsection Box
  set Y repetitions = 4
 end
end
subsection Mesh refinement
 set Strategy = PF6 transfer mesh
 set Initial global refinement = 1
 set Minimum refinement level = 0
 set Refinement fraction = 0.5
 set Coarsening fraction = 0.5
end
subsection Time stepping
 subsection Function
  set Function expression = 0.125
 end
end
subsection Boundary velocity model
 subsection Function
  set Function expression = 0.256*max(0,t-0.0625)*(y-0.5); 0
 end
end
subsection Particles
 set Integration scheme = rk2
end
'''.replace('END_TIME', str(a.end))
    (out/'fixture.prm').write_text(prm)
    text = fixture_text(out/'fixture.prm', root, 'PF6 fixed reference transfer')
    text += f'''\nset Additional shared libraries = {out/dso.name}
set Output directory = {out/'run'}
set Resume computation = {'true' if a.restart else 'false'}
subsection Postprocess
 subsection PF6 transfer checks
  set Transfer checks = {'false' if a.no_transfer else 'true'}
 end
end
subsection Termination criteria
 set Checkpoint on termination = {'true' if a.checkpoint else 'false'}
end
'''
    (out/'input.prm').write_text(text)
    if a.restart:
        shutil.copytree(a.restart.resolve(), out/'run/restart')
    paths = subprocess.check_output(['git', 'ls-files', '-c', '-o', '--exclude-standard',
                                    'source', 'include'], cwd=root, text=True).splitlines()
    r = dict(head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
             ranks=a.ranks, threads=1, budget_seconds=180,
             executable_sha256=sha(out/binary.name), plugin_sha256=sha(out/dso.name),
             input_sha256=sha(out/'input.prm'), observer_sha256=sha(observer),
             source_hashes={s: sha(root/s) for s in sorted(set(paths)) if (root/s).is_file()},
             case=vars(a) | {'restart': str(a.restart) if a.restart else None},
             scope='fixed reference surface; no transport; prescribed uniform mesh transfers; native endpoint lifecycle')
    (out/'record.json').write_text(json.dumps(r, indent=2)+'\n')
    (out/'source.patch').write_bytes(subprocess.check_output(['git', 'diff', '--', 'source', 'include'], cwd=root))
    command = ['env', 'ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1', 'OMP_PROC_BIND=false',
               '/opt/openmpi/5.0.6/bin/mpirun', '--map-by', 'core', '--bind-to', 'core',
               '--rank-by', 'slot', '--report-bindings', '-n', str(a.ranks),
               str(out/binary.name), str(out/'input.prm')]
    (out/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    print(json.dumps(command))


if __name__ == '__main__':
    main()
