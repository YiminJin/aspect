"""Independent nested-time K3 reference; no ASPECT launch or history reset."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from reference import Reference, read_parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dt', type=float, required=True)
    parser.add_argument('--cells', type=int, default=2048)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    assert not args.output.exists(), 'Preserve prior reference evidence.'
    start = time.monotonic()
    parameters = here/'spatial0375_n128_f32_periodic/parameters.prm'
    model = Reference(read_parameters(parameters), args.cells, 6, .00225)
    count = round(3/args.dt)
    assert count*args.dt == 3
    sequence = [dict(time=k*args.dt, dt=args.dt,
                     U=1e-4+(.00225-1e-4)*min(k*args.dt/2, 1))
                for k in range(1, count+1)]
    report, snapshots = model.run(sequence)
    report['time_sequence_source'] = 'predeclared nested timestep through 3 s; verify against accepted ASPECT time/dt/U'
    report['resolved_parameters'] = str(parameters)
    report['elapsed_seconds'] = time.monotonic()-start
    # Audit the specified maximum, without replacing it by an additive update.
    candidates = []
    old_C = report['initial']['retained_histories']['C']
    for k, row in enumerate(report['steps'], 1):
        g, _, _ = model.localization(snapshots[k][0])
        _, hp, _ = model.localization(snapshots[k-1][0])
        beta = np.exp(-args.dt*model.G/model.eta)
        kappa = -model.eta*np.expm1(-args.dt*model.G/model.eta)
        a = row['C']/g
        b = np.divide(beta*hp*old_C, 1-g, out=np.zeros_like(g), where=g!=1)
        candidate = args.dt*(a-b)*(a+b)/(2*kappa)
        H_old = snapshots[k-1][1]
        candidates.append(dict(step=k, time=k*args.dt,
            max_candidate_over_old_H=float(np.max((candidate/H_old)[model.admitted])),
            max_candidate_minus_old_H=float(np.max((candidate-H_old)[model.admitted]))))
        np.testing.assert_allclose(snapshots[k][1],
            np.where(model.admitted, np.maximum(H_old,candidate), H_old), rtol=0, atol=1e-8)
        old_C = row['C']
    report['maximum_rule_audit'] = candidates
    report['cumulative_feedback'] = dict(
        H=float(np.max(snapshots[-1][1]-snapshots[0][1])),
        phi=float(np.max(abs(snapshots[-1][0]-snapshots[0][0]))),
        Ih=report['steps'][-1]['Ih']-report['initial']['Ih'])
    args.output.mkdir(parents=True)
    for k, (phi,H) in enumerate(snapshots):
        np.savetxt(args.output/f'phase-{k}.csv',np.c_[model.y,phi],delimiter=',',header='y,phi',comments='')
        np.savetxt(args.output/f'history-{k}.csv',np.c_[model.qy.ravel(),H.ravel()],delimiter=',',header='y,H',comments='')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(dt=args.dt,cells=args.cells,feedback=report['cumulative_feedback'],
        max_candidate_ratio=max(r['max_candidate_over_old_H'] for r in candidates),
        max_normalization=max(r['supported_slip_normalization_error'] for r in report['steps']),
        seconds=report['elapsed_seconds'])))


if __name__ == '__main__':
    main()
