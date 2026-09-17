"""Frozen definitions for the short BP3 filesystem restart comparison.

Check every component separately: pressure/history magnitude cannot hide a
velocity error. 1e-8 is the unchanged nonlinear relative tolerance; exact
identity is required for stable IDs, geometry and frozen background fields.
This is not an analytic stress-reproduction or first-event accuracy test.
"""
import argparse
import json
import io
from pathlib import Path
import re
import numpy as np


def difference(a, b, scale_floor=0.):
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    absolute = float(np.max(np.abs(a-b), initial=0.))
    scale = max(float(np.max(np.abs(a), initial=0.)), scale_floor)
    assert absolute <= 1e-8*scale, (absolute, scale)
    return dict(absolute=absolute, scale=scale, relative=absolute/scale if scale else 0.)


def load_parts(path, pattern):
    parts = sorted(path.glob(pattern))
    assert parts, (path, pattern)
    rows=[]
    for p in parts:
        with p.open() as stream:
            next(stream)  # Ranks without boundary-source parents write a header only.
            body=stream.read()
        if body.strip(): rows.append(np.loadtxt(io.StringIO(body),delimiter=',',ndmin=2))
    assert rows,(path,pattern)
    data = np.concatenate(rows)
    data = data[np.argsort(data[:, 0])]
    assert len(np.unique(data[:, 0])) == len(data)
    return data


def convergence(path):
    text = path.read_text()
    result = {}
    chunks = re.split(r'\*\*\* Timestep (\d+):', text)
    for i in range(1, len(chunks), 2):
        step, body = int(chunks[i]), chunks[i+1]
        nonlinear = re.findall(r'after nonlinear iteration\s+(\d+): ([^,\n]+), ([^\n]+)', body)
        assert nonlinear and max(map(float, nonlinear[-1][1:])) < 1e-8, step
        linear = re.findall(r'Fault linear solve: iterations=(\d+),(?: estimated=([^,\n]+),)? fresh=([^,\n]+), target=([^,\n]+)', body)
        pending = None
        for row in linear:
            if pending:
                assert int(row[0]) > int(pending[0]) and float(row[3]) == float(pending[3])
            pending = row if float(row[2]) > float(row[3]) else None
        assert pending is None
        result[step] = dict(final_bulk=float(nonlinear[-1][1]), final_surface=float(nonlinear[-1][2]),
                            fresh_attempts=len(linear))
    assert result
    return result


def compare(a, b):
    result = {}
    for step in (2, 3):
        entry = {}
        x = load_parts(a, f'audit_bulk_{step}_rank*.csv')
        y = load_parts(b, f'audit_bulk_{step}_rank*.csv')
        assert np.array_equal(x[:, :2], y[:, :2])
        entry['bulk_components'] = {str(int(c)): difference(x[x[:,1]==c,2], y[y[:,1]==c,2])
                                    for c in np.unique(x[:,1])}
        x = load_parts(a, f'audit_particles_{step}_rank*.csv')
        y = load_parts(b, f'audit_particles_{step}_rank*.csv')
        assert np.array_equal(x[:,0], y[:,0])
        entry['particle_columns'] = {str(c): difference(x[:,c],y[:,c]) for c in range(1,x.shape[1])}
        x = np.atleast_1d(np.genfromtxt(a/f'fault_{step}.csv', names=True, delimiter=','))
        y = np.atleast_1d(np.genfromtxt(b/f'fault_{step}.csv', names=True, delimiter=','))
        entry['fault'] = {}
        for field in ('x','y','tau_bg','sigma_n_bg','prescribed'):
            assert np.array_equal(x[field],y[field]), field
        for field in ('V','Theta','C','Ih','slip','q','time','dt'):
            entry['fault'][field] = difference(x[field], y[field])
        result[step] = entry
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('continuous', type=Path)
    parser.add_argument('resumed', type=Path)
    parser.add_argument('--continuous-log', type=Path, required=True)
    parser.add_argument('--resume-log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'continuous_convergence': convergence(args.continuous_log),
              'resumed_convergence': convergence(args.resume_log),
              'comparison': compare(args.continuous, args.resumed), 'passed': True}
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print('BP3 filesystem restart: per-component bulk, stable-ID particles, surface/history/slip/background and timestep agreement pass.')
