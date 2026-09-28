"""Export the legacy slip schema from saved profiles only, never unsaved steps."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from plot_cumulative_slip import HEADER, read_scheduled_profiles


def export(index, output, fault=0):
    index, output = Path(index), Path(output)
    count = 0
    # Validate the entire input before creating a legacy-looking result.
    for _ in read_scheduled_profiles(index, fault):
        count += 1
    if not count:
        raise ValueError('No saved profiles to export')
    with index.open() as stream:
        first = next(csv.DictReader(stream))
    with (index.parent/first['file']).open() as stream:
        nodes = [r for r in csv.DictReader(stream) if int(r['fault']) == fault]
    xy = np.array([[float(r['x_m']), float(r['y_m'])] for r in nodes])
    arclength = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    metadata = output.with_suffix(output.suffix+'.metadata.json')
    if output.exists() or metadata.exists():
        raise ValueError('Refuse to overwrite an existing export or its provenance')
    metadata.write_text(json.dumps(dict(source=str(index.resolve()), saved_profiles_only=True,
        profiles=count, unsaved_timesteps_reconstructed=False,
        note='Use the original profiles and saved V for event classification; this export is sparse.'), indent=2)+'\n')
    with output.open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADER)
        for profile in read_scheduled_profiles(index, fault):
            writer.writerows([profile.step, format(profile.time, '.17g'), fault, int(node),
                             format(s, '.17g'), format(xd, '.17g'), format(slip, '.17g')]
                            for node, s, xd, slip in zip(profile.nodes, arclength, profile.xd, profile.slip))
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--fault', type=int, default=0)
    args = parser.parse_args()
    print(f'Exported {export(args.index, args.output, args.fault)} saved profiles only; no unsaved timesteps.')
