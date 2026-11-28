"""Advance the public examples and compare their accepted fields across parameter layouts."""
import argparse
from pathlib import Path
import sys
from xml.etree import ElementTree

import numpy as np

results = Path(__file__).resolve().parents[1]
benchmarks = results.parents[2] / 'benchmarks'
sys.path[:0] = [str(benchmarks / 'fixed_fault/curved_mature'),
                str(benchmarks / 'growing_fault'), str(results / 'material-unification')]
from check_interface import compare_growing_frames
from check_restart import SAVE, run
from cross_build import check_growing_tables, growing_clocks


def mature_frames(old, new, steps):
    for output in (old, new):
        clocks = np.loadtxt(output / 'statistics', usecols=(0, 1, 2), ndmin=2)
        np.testing.assert_allclose(clocks, [[0, 0, 0], [1, .0125, .0125], [2, .025, .0125]],
                                   atol=1e-15, rtol=0, equal_nan=False)
    for step in steps:
        labels = f'fault-labels-{step:05}.txt'
        if (old / labels).read_bytes() != (new / labels).read_bytes():
            raise ValueError(f'Physical labels differ at step {step}')
        rows = np.loadtxt(new / labels)
        np.testing.assert_array_equal(rows[:, 0], np.arange(5))
        if not np.isfinite(rows).all():
            raise ValueError(f'Nonfinite label at step {step}')
        filename = f'solution/solution-{step:05}.0000.vtu'
        trees = [ElementTree.parse(d / filename) for d in (old, new)]
        for name in ('fault_tau_xx', 'fault_tau_yy', 'fault_tau_xy'):
            a, b = [t.find(f'.//PointData/DataArray[@Name="{name}"]') for t in trees]
            if a is None or b is None or not a.text or (a.attrib, a.text) != (b.attrib, b.text):
                raise ValueError(f'Physical stress {name} differs at step {step}')
    print(f'Mature steps {list(steps)}: five material IDs and all three accepted stress arrays identical.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix', type=lambda x: Path(x).resolve())
    parser.add_argument('destination', type=lambda x: Path(x).resolve())
    parser.add_argument('--reference', type=lambda x: Path(x).resolve(),
                        help='Outputs and checkpoint generation 02 from the older executable.')
    args = parser.parse_args()
    args.destination.mkdir(parents=True)
    for name, example in [('growing', 'growing_fault/growing_seed.prm'),
                          ('mature', 'fixed_fault/curved_mature/curved_mature.prm')]:
        source = args.prefix / 'benchmarks' / example
        settings = SAVE + ('set End time = 0.015\n' if name == 'growing' else '')
        old = args.reference / name / 'output' if args.reference else None
        for suffix in (('', '-old-input', '-restart') if old else ('',)):
            resume = suffix == '-restart'
            # original.prm is the writer's expanded input, including its legacy spellings.
            deck = old / 'original.prm' if suffix == '-old-input' else source
            output = run(name + suffix, deck, [], root=args.destination, prefix=args.prefix,
                         settings=settings, checkpoint=old if resume else None)
            if not old:
                print(f'Captured {name}: {output}', flush=True)
                continue
            if name == 'mature':
                mature_frames(old, output, range(2 if resume else 0, 3))
            else:
                growing_clocks(output, 3)
                growing_clocks(old, 3)
                for step in range(2 if resume else 0, 4):
                    errors = compare_growing_frames(old, output, steps=[step])
                    check_growing_tables(old, output, step)
                    print(f'Growing step {step}: max |Δτxy| {errors["tauxy"]:g} Pa, '
                          f'|ΔV| {errors["V"]:g} m/s, |Δslip| {errors["slip"]:g} m.')


if __name__ == '__main__':
    main()
