"""Fresh initialization only; old and clean-background cases remain untouched."""
import json
import sys
from pathlib import Path
from run_short import OUT, HERE, BP3, parameters, render, digest, run

LABEL = 'weak-state-init'


def prepare():
    base = OUT/'clean-background-init'
    dest = OUT/LABEL
    dest.mkdir()
    values = parameters((base/'run.prm').read_text())
    plugin = HERE/'build/libbp5_initialization.release.so'
    values['Output directory',] = str(dest)
    values['Additional shared libraries',] = values['Additional shared libraries',].replace(
        str(BP3/'build/libbp3.release.so'), str(plugin))
    assert str(plugin) in values['Additional shared libraries',]
    (dest/'run.prm').write_text(render(values))
    record = json.loads((base/'launch.json').read_text())
    # This old observer explicitly checks the superseded physical nodal inverse.
    # The new initializer supplies its own projected/weak-state and derivative
    # audits; retain the ordinary mechanical, history and profile observers.
    record['environment'].pop('ASPECT_BP3_LENGTH_STUDY')
    record['command'][-1] = str(dest/'run.prm')
    record['hashes'].pop(str(base/'run.prm'))
    record['hashes'].pop(str(BP3/'build/libbp3.release.so'))
    for path in [dest/'run.prm', plugin, HERE/'weak_initialization.h', BP3/'bp3.cc', Path(__file__)]:
        record['hashes'][str(path)] = digest(path)
    record['parameter_changes'] = {'Output directory': str(dest),
                                  'Additional shared libraries': values['Additional shared libraries',]}
    record['initialization'] = 'projected material nodal inverse, then positive Q1 weak friction balance'
    (dest/'launch.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    if sys.argv[1] == 'prepare': prepare()
    elif sys.argv[1] == 'run': run(LABEL)
    else: raise ValueError('Use prepare or run')
