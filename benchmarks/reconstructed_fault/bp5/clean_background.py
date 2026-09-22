"""One fresh initialization; replace the effective prestress, not just its correction."""
import json
import sys
import numpy as np
from run_short import OUT, prepare, run, parameters, render, digest

LABEL = 'clean-background-init'


def setup():
    prepare(LABEL, 'candidate', False)
    dest = OUT/LABEL
    baseline = OUT/'candidate-six'
    values = parameters((baseline/'run.prm').read_text())
    old = np.loadtxt(OUT/'fixtures/candidate/prestress.txt', skiprows=1)
    config = np.genfromtxt(baseline/'friction_configuration.csv', delimiter=',', names=True)
    nominal = float(config['nominal_q_Pa'][0])
    assert np.all(config['nominal_q_Pa'] == nominal)
    # The production evaluator subtracts a + b/d from stored shear at each QP.
    # Replace all four coefficients together: effective shear is exactly nominal.
    new = old.copy()
    new[:, 2:] = [nominal, 50e6, 0., 0., 1.]
    path = dest/'uniform_effective_prestress.txt'
    np.savetxt(path, new, fmt='%.17g', header=str(len(new)), comments='')
    changes = {
        ('Output directory',): str(dest),
        ('Postprocess', 'BP3', 'Mature prestress file'): str(path),
        ('Postprocess', 'BP3', 'Last accepted step'): '0',
        ('Checkpointing', 'Steps between checkpoint'): '0',
        ('Termination criteria', 'Checkpoint on termination'): 'false',
    }
    values.update(changes)
    (dest/'run.prm').write_text(render(values))
    record = json.loads((dest/'launch.json').read_text())
    record['cap_seconds'] = 900
    record['hashes'][str(dest/'run.prm')] = digest(dest/'run.prm')
    record['hashes'][str(path)] = digest(path)
    record['hashes'][str(baseline/'run.prm')] = digest(baseline/'run.prm')
    record['effective_background_Pa'] = [nominal, 50e6]
    record['parameter_changes'] = {'/'.join(k): v for k, v in changes.items()}
    (dest/'launch.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        setup()
    elif sys.argv[1] == 'run':
        run(LABEL)
        data = np.atleast_1d(np.genfromtxt(OUT/LABEL/'accepted_steps.csv', delimiter=',', names=True))
        assert list(data['step']) == [0], 'This comparison authorizes initialization only'
    else:
        raise ValueError('Use prepare or run')
