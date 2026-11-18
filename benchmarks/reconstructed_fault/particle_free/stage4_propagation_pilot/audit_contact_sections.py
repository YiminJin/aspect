"""Check the six strip derivatives sampled by the near-contact Bernstein row."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path


def resistance(v, age):
    # Input values of the synthetic pilot. The midpoint law ages over 20 μs.
    dt, dc, v0 = 2e-5, 3.2211126126299625e-7, .001
    travel = abs(v) * dt / dc
    theta = age * math.exp(-travel) + dt * (-math.expm1(-travel) / travel if travel else 1.)
    scale = math.exp((.6 + .02 * math.log(theta * v0 / dc)) / .015) / (2*v0)
    return 200 * .015 * math.asinh(v * scale) + 1000*v


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('restore', type=Path)
    p.add_argument('trace', type=Path)
    p.add_argument('destination', type=Path)
    args = p.parse_args()
    stages = {}
    for file in args.trace.glob('tip-search-points-r*.csv'):
        with file.open() as stream:
            for r in csv.DictReader(stream):
                if r['row'] == '258249' and r['alpha'] == '1' and r['shift'] == '0':
                    stages[int(r['iteration']), int(r['section'])] = r
    assert stages
    keys = {k for _, k in stages}
    with gzip.open(args.restore/'restore-sections.csv.gz', 'rt') as stream:
        sections = list(csv.DictReader(stream))
    # Six Gauss weights mapped to [0,1], indexed as in the verified native export.
    w = [.08566224618958517, .1803807865240693, .23395696728634552]
    w += w[::-1]
    mobility = {k: 0. for k in keys}
    count = {k: 0 for k in keys}
    beta = 1/(1 + 1e6/1e8*2e-5)
    for file in args.restore.glob('restore-points-r*.csv.gz'):
        with gzip.open(file, 'rt') as stream:
            for r in csv.DictReader(stream):
                q = int(r['q'])
                x = float(r['x'])
                k = 6 * math.floor(x*32) + q % 6
                if k not in keys:
                    continue
                assert abs(x-float(sections[k]['s'])) < 1e-12
                phi = float(r['phi'])
                c = 4 * phi * (1 + phi)
                fraction = c / (c + (1-phi)**2)
                jxw = w[q%6] * w[q//6] / 32**2
                mobility[k] += fraction*jxw / (float(sections[k]['ds'])*1e6*beta*2e-5)
                count[k] += 1
    assert all(n == 128*6 for n in count.values())
    results = []
    for (iteration, k), r in sorted(stages.items()):
        v, age = float(r['trial_rate']), float(sections[k]['age'])
        force, slope = float(r['trial_force']), float(r['trial_df_dfree'])
        eps = abs(v)*1e-5
        # Parameterize the implicit law by v. Differentiate F against v_free=v+M F
        # at the *independently integrated parent-profile* mobility.
        delta = resistance(v+eps, age) - resistance(v-eps, age)
        secant = delta / (2*eps + mobility[k]*delta)
        results.append(dict(iteration=iteration, section=k, parent_mobility=mobility[k],
                            rate=v, resistance=force, force_error=resistance(v, age)-force,
                            native_slope=slope, finite_difference=secant,
                            relative_difference=secant/slope-1))
    args.destination.write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
