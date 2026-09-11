#!/usr/bin/env python3
"""Check these fixtures' explicit overrides and Box geometry without running ASPECT.

This reads the simple include/subsection/set syntax used by this fixture family;
it is not an ASPECT parameter validator or a substitute for realized mesh checks.
"""
import json
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent


def parameters(path):
    values = {}
    sections = []
    for raw in path.read_text().splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        if line.startswith('include '):
            assert not sections
            included = line[8:].replace('$ASPECT_SOURCE_DIR', str(ROOT))
            values.update(parameters(Path(included)))
        elif line.startswith('subsection '):
            sections.append(line[11:])
        elif line == 'end':
            sections.pop()
        elif line.startswith('set '):
            key, value = line[4:].split('=', 1)
            values['/'.join(sections + [key.strip()])] = value.strip()
        else:
            raise ValueError(f'Unsupported fixture syntax in {path}: {line}')
    assert not sections
    return values


def main():
    allowed = {'Output directory', 'Geometry model/Box/X repetitions',
               'Geometry model/Box/Y repetitions',
               'Mesh refinement/Initial global refinement'}
    report = {'scope': 'static fixture/source preflight; no simulation', 'cases': {}}
    for case, baseline in [('pilot', 'pilot64'), ('homogeneous', 'homogeneous64')]:
        old = parameters(HERE.parent / 'true-pressure' / (baseline + '.prm'))
        new = parameters(HERE / (case + '.prm'))
        changes = {key: [old.get(key), new.get(key)] for key in old.keys() | new.keys()
                   if old.get(key) != new.get(key)}
        assert set(changes) == allowed, changes
        assert new['Mesh refinement/Initial adaptive refinement'] == '0'
        assert new['Mesh refinement/Time steps between mesh refinement'] == '0'
        assert new['Solver parameters/Stokes solver parameters/Stokes solver type'] == 'block AMG'
        report['cases'][case] = dict(sorted(changes.items()))
    h = Fraction(1, 255)
    lower = -Fraction(1, 2) + 127*h
    upper = lower + h
    assert lower == -Fraction(1, 510) and upper == Fraction(1, 510)
    assert -lower/h == Fraction(1, 2)
    report['source_implied_geometry'] = {
        'cells': [64, 255], 'active_cells': 64*255,
        'hx_m': .25/64, 'hy_m': float(h),
        'normal_cell_size_increase_percent': float((h/Fraction(1, 256)-1)*100),
        'fault_y_m': 0, 'central_row_y_bounds_m': [float(lower), float(upper)],
        'fault_reference_y': .5, 'particles_with_3_by_3_generator': 64*255*9,
        'baseline_active_cells': 64*256, 'baseline_particles': 64*256*9}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
