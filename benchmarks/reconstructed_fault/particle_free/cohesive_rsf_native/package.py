#!/usr/bin/env python3
"""Package the qualified local source/evidence, without builds or solver archives."""
from pathlib import Path
import argparse
import csv
import difflib
import hashlib
import json
import platform
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
WORK = ROOT/'build-particle-free/cohesive-rsf-native'
OUT = HERE.parent/'evidence/cohesive_rsf_native'
DOC = ROOT/'doc/reconstructed_fault/particle_free'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_files():
    files = set()
    for directory in ['source', 'include', 'cmake', 'contrib', 'unit_tests']:
        files.update(p for p in (ROOT/directory).rglob('*') if p.is_file()
                     and '.git' not in p.parts and '__pycache__' not in p.parts)
    files.update((ROOT/'tests/cmake').rglob('*'))
    files.update((ROOT/'tests').glob('*.h'))
    for name in ['CMakeLists.txt', 'VERSION', 'LICENSE', 'AUTHORS.md', 'README.md',
                 'doc/CMakeLists.txt', 'doc/COPYING', 'doc/manual/plugin_graph.svg',
                 'doc/reconstructed_fault/current_design.md', 'doc/reconstructed_fault/specification.tex']:
        files.add(ROOT/name)
    files.update(p for p in HERE.iterdir() if p.is_file())
    files.add(HERE.parent/'cohesive_interface_evolution.h')
    files.update(DOC.glob('COHESIVE_RSF_NATIVE*.md'))
    files.update([DOC/'STATUS.md', DOC/'SOURCE_AUDIT.md'])
    return sorted(p for p in files if p.is_file())


def write_zip(path, files):
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for p in sorted(set(files)):
            archive.write(p, p.relative_to(ROOT))
    path.with_suffix(path.suffix+'.sha256').write_text(digest(path)+'  '+path.name+'\n')


def metadata():
    cache = WORK/'native/CMakeCache.txt'
    selected = [line for line in cache.read_text().splitlines() if
                line.startswith(('ASPECT_', 'CMAKE_BUILD_TYPE:', 'CMAKE_CXX_COMPILER:', 'DEAL_II_DIR:'))]
    binary = WORK/'native/aspect-release'
    plugin = WORK/'checks-native/libcohesive_rsf_native_checks.release.so'
    env = dict(baseline='f1f9f29cf40c32d1ddcea000b724375b64853532',
               branch='pf-rsf-particle-free', system=platform.platform(),
               build_cache=selected, binary_sha256=digest(binary), plugin_sha256=digest(plugin),
               deal_II='9.6.2', Trilinos='14.2',
               compiler=subprocess.check_output(['/opt/gcc/12.4.0/bin/g++', '--version'], text=True),
               MPI=subprocess.check_output(['/opt/openmpi/5.0.6/bin/mpiexec', '--version'], text=True),
               linked_libraries=subprocess.check_output(['ldd', str(binary)], text=True),
               run_environment=dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', DEAL_II_NUM_THREADS='1'),
               dependencies='deal.II MPI, Trilinos, p4est, zlib, SUNDIALS; native build also enables NetCDF/WorldBuilder',
               source_archive_scope='Current source; unrelated example databases/regression outputs excluded')
    (OUT/'environment.json').write_text(json.dumps(env, indent=2)+'\n')
    validation = WORK/'runs/final-parameter-validation/execution.json'
    (OUT/'parameter-validation.json').write_bytes(validation.read_bytes())
    units = {'Shear modulus':'Pa', 'Maxwell viscosity':'Pa s', 'Fracture energy':'J/m^2',
             'Intact strength':'Pa', 'Background confinement':'Pa', 'Reference velocity':'m/s',
             'State slip distance':'m', 'Radiation impedance':'Pa s/m',
             'Physical persistence length':'m', 'Numerical lifting half width':'m',
             'Future surface spacing':'m', 'Candidate extent':'m',
             'Candidate refresh interval':'s', 'Maximum formed length':'m'}
    rows, stack = [], []
    for line in (WORK/'runs/pilot-final-current/output/parameters.prm').read_text().splitlines():
        line = line.strip()
        if line.startswith('subsection '):stack.append(line[len('subsection '):])
        elif line == 'end':stack.pop()
        elif line.startswith('set ') and '=' in line:
            key, value = map(str.strip, line[4:].split('=',1))
            value = value.split('#',1)[0].strip()
            if stack == ['Cohesive RSF pilot']:
                rows.append(dict(parameter=key, value=value, unit=units.get(key,'dimensionless/text'),
                                 source='native resolved parameters.prm'))
    with (OUT/'native-parameters.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['parameter','value','unit','source'])
        writer.writeheader();writer.writerows(rows)
    with (OUT/'native-timers.csv').open('w', newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['scope','calls','wall_s','overlap'])
        writer.writerows([
            ['Cohesive RSF: native mixed solve',180,681,'includes Stokes/history/surface'],
            ['Solve Stokes system',392,241,'inside coupled route'],
            ['Create snapshot',23,5.68,'native timer'],
            ['Build Stokes preconditioner',1,.0916,'inside coupled route']])


def source():
    metadata()
    files = source_files()
    manifest = {str(p.relative_to(ROOT)):digest(p) for p in files}
    (OUT/'source-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    write_zip(OUT/'native-source.zip', files+[OUT/'source-manifest.json'])
    with zipfile.ZipFile(WORK/'entry-source.zip') as before:
        patch = []
        for name in before.namelist():
            old = before.read(name).decode().splitlines(keepends=True)
            new = (ROOT/name).read_text().splitlines(keepends=True)
            patch.extend(difflib.unified_diff(old, new, fromfile='entry/'+name, tofile='current/'+name))
    (OUT/'entry-to-current.patch').write_text(''.join(patch))
    print(json.dumps(dict(source_files=len(manifest), source_zip_bytes=(OUT/'native-source.zip').stat().st_size)))


def evidence():
    # Include compact data/figures/current source, not raw output/checkpoints or
    # superseded preliminary summaries. Those remain untouched in the worktree.
    files = [p for p in OUT.iterdir() if p.is_file() and 'preliminary' not in p.name
             and p.name != 'checksums.sha256']
    files += [p for p in HERE.iterdir() if p.is_file()]
    files += list(DOC.glob('COHESIVE_RSF_NATIVE*.md'))
    files += [WORK/'legacy-unit.log', WORK/'runs/pilot-final-current/output/parameters.prm',
              WORK/'runs/pilot-final-current/execution.json',
              WORK/'runs/final-current-restarted/execution.json',
              WORK/'runs/final-current-completed/execution.json',
              WORK/'runs/native-halfstep-mature/execution.json']
    hashes = '\n'.join(digest(p)+'  '+str(p.relative_to(ROOT)) for p in sorted(set(files)))+'\n'
    (OUT/'checksums.sha256').write_text(hashes)
    target = HERE.parent/'cohesive_rsf_native_evidence.zip'
    write_zip(target, files+[OUT/'checksums.sha256'])
    print(json.dumps(dict(evidence_files=len(set(files))+1, archive=str(target.relative_to(ROOT)),
                          archive_bytes=target.stat().st_size, sha256=digest(target))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['source', 'evidence'])
    args = parser.parse_args()
    if args.stage == 'source':
        source()
    else:
        evidence()
