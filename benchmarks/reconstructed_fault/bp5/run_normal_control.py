"""One fresh initialization, with fixed frictional normal traction; no real steps."""
import argparse
import json
import shutil
from pathlib import Path

import run_steady_large_step as runner
from startup_30km import ROOT, HERE, BP3, BIN, STUDY, parameters, render, digest

BASE = STUDY/'loading-startup/startup'
OUT = STUDY/'loading-normal-control'
runner.OUT = OUT


def prepare():
    path = OUT/'initial'
    path.mkdir()
    baseline = json.loads((BASE/'launch.json').read_text())
    assert digest(OUT/'provenance/aspect-release.before') == baseline['hashes'][str(BIN)]
    for name, sha in baseline['hashes'].items():
        if name.endswith(('/run.prm', '/fault.txt', '/target_cells.txt', '/completion.txt', '.so')):
            assert digest(Path(name)) == sha, name
    values = parameters((BASE/'run.prm').read_text())
    original = dict(values)
    plugin = HERE/'build/libbp5_normal_control.release.so'
    libraries = values['Additional shared libraries',].split(', ')
    libraries[0] = str(plugin)
    values['Additional shared libraries',] = ', '.join(libraries)
    values['Output directory',] = str(path)
    values['Postprocess','BP3','Last accepted step'] = '0'
    values['Material model','Phase field fault','Use adiabatic pressure in fault friction'] = 'true'
    assert values['Surface pressure',] == '0'
    assert values['Gravity model','Vertical','Magnitude'] == '0'
    assert values['Resume computation',] == 'false'
    assert values['Material model','Phase field fault','Initial time step'] == '1e6'
    (path/'run.prm').write_text(render(values))
    # This is an input copy, not a recalculated background or a restart.
    shutil.copy2(BASE/'steady_initialization.csv', path/'steady_initialization.csv')
    env = dict(baseline['environment'])
    env['ASPECT_BP5_INITIAL_REFERENCE'] = str(path/'steady_initialization.csv')
    sources = [BP3/'bp3.cc', BP3/'work_replay.h', HERE/'steady_initialization.h',
               HERE/'normal_control_checks.h', HERE/'CMakeLists.txt', Path(__file__).resolve(),
               ROOT/'source/reconstructed_fault/surface_system.cc']
    for source in sources:
        destination = path/'source-tested'/source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    inputs = [BIN, *map(Path,libraries), path/'run.prm', path/'steady_initialization.csv', *sources]
    record = dict(command=[*baseline['command'][:-1], str(path/'run.prm')], environment=env,
                  cap_seconds=1200, hashes={str(p):digest(p) for p in inputs},
                  source_revision=baseline['source_revision'], reference=str(BASE),
                  parameter_changes={' / '.join(k):[original[k],v] for k,v in values.items() if original[k]!=v})
    (path/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record['parameter_changes'],indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run'])
    args=p.parse_args()
    prepare() if args.action=='prepare' else runner.run('initial')
