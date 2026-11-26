"""Plot accepted material-vertex motion across the measured viscosity transition."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

from check import read


def accepted_positions(output):
    """Read replicated rank-zero surface coordinates at the three poster times (s)."""
    clock = read(output/'layered-background.csv')
    frames = []
    ids = None
    positions = []
    for time in (0., .0125, .025):
        matches = clock[np.isclose(clock['time'], time, rtol=0, atol=1e-14)]
        if len(matches) != 1:
            raise ValueError(f'Expected one accepted frame at {time:g} s, found {len(matches)}')
        frame = matches[0]
        surface = np.sort(read(output/f'material-{frame["step"]}-surface-0.csv'), order='i')
        current_ids = surface['i']
        if current_ids.dtype.kind not in 'iu' or len(current_ids) < 2 or np.any(np.diff(current_ids) == 0):
            raise ValueError(f'Expected distinct material vertex IDs at {time:g} s')
        if ids is None:
            ids = current_ids
        elif not np.array_equal(ids, current_ids):
            raise ValueError(f'Material vertex IDs changed at {time:g} s')
        frames.append(frame)
        positions.append(np.column_stack((surface['x'], surface['y'])))
    return np.array(frames), ids, np.array(positions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='layered moving fault output directory')
    parser.add_argument('destination', type=Path, help='directory for layered-motion.png and layered-motion.pdf')
    args = parser.parse_args()

    clocks, ids, positions = accepted_positions(args.output)
    times = clocks['time']
    # Mark the two ends and the middle material vertex (lower middle for even counts).
    marked = np.unique([0, (len(ids)-1)//2, len(ids)-1])
    profile = read(args.output/f'background-{clocks["step"][0]}.csv')
    params = json.loads((args.output/'parameters.json').read_text())
    translation = float(params['Postprocess']['Layered background']['Translation']['value'])
    mean_wall_motion = times * translation
    dx = positions[:, :, 0] - positions[0, :, 0]

    plt.rcParams.update({'font.size': 13, 'axes.titlesize': 16, 'axes.titleweight': 'bold',
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, (section, motion) = plt.subplots(1, 2, figsize=(12.5, 5.4),
                                          gridspec_kw={'width_ratios': [1.3, 1]})
    fig.subplots_adjust(left=.07, right=.97, top=.87, bottom=.30, wspace=.27)
    x_min, y_min = positions.min(axis=(0, 1)) - [.18, .18]
    x_max, y_max = positions.max(axis=(0, 1)) + [.18, .18]
    y = np.linspace(y_min, y_max, 601)
    # The background observer exports the physical Maxwell viscosity at its
    # column quadrature points. Use its measured vertical profile for shading.
    eta = np.interp(y, profile['y'], profile['eta'])
    palette = plt.get_cmap('YlGnBu')(np.linspace(0, 1, 256))
    # Opaque light colors keep the vector PDF free of seams between tiles.
    palette[:, :3] = .25 + .75*palette[:, :3]
    surface = section.pcolormesh([x_min, x_max], y, (.5*(eta[1:]+eta[:-1]))[:, None],
                                 cmap=colors.ListedColormap(palette), norm=colors.Normalize(1, 10),
                                 shading='flat', edgecolors='face', linewidth=.05)
    section.set(xlim=(x_min, x_max), ylim=(y_min, y_max), xlabel='x (m)', ylabel='y (m)',
                aspect='equal')
    section.set_title('Accepted material trace', loc='left', pad=13)
    for i, (color, style) in enumerate([('#1b2935', '--'), ('#60717b', ':'), ('#ad371a', '-')]):
        section.plot(*positions[i].T, color=color, linestyle=style, linewidth=2,
                     label=f'{1e3*times[i]:g} ms')
    # Replot the same material IDs at successive endpoints. Label the
    # IDs once; only the final clock has filled symbols.
    marker_shapes = ['s', 'o', '^']
    for index, marker in zip(marked, marker_shapes):
        section.plot(*positions[0, index], marker=marker, markerfacecolor='white',
                     markeredgecolor='#202c32', ms=6, lw=0)
        section.plot(*positions[-1, index], marker=marker, markerfacecolor='#ad371a',
                     markeredgecolor='white', ms=7, lw=0)
        section.annotate(f'ID {ids[index]}', positions[-1, index], xytext=(0, 12), ha='center',
                         textcoords='offset points', fontsize=11,
                         bbox={'boxstyle': 'round,pad=.18', 'facecolor': 'white',
                               'edgecolor': 'none', 'alpha': .92})
    section.legend(loc='lower right', framealpha=.96, fontsize=10.5)
    bar = fig.colorbar(surface, cax=fig.add_axes([.07, .105, .441, .028]), orientation='horizontal',
                       ticks=[1, 3, 5, 7, 9, 10])
    bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor('face')
    bar.set_label('Maxwell viscosity, η (Pa s)')

    motion.axhline(0, color='#78848d', lw=1.1)
    offsets = 1e3 * (dx - mean_wall_motion[:, None])
    for index, marker, color in zip(marked, marker_shapes, ['#785378', '#147b72', '#a73319']):
        motion.plot(times*1e3, offsets[:, index], marker=marker, lw=2.2, markersize=6, color=color)
        motion.annotate(f'ID {ids[index]}', (times[-1]*1e3, offsets[-1, index]), xytext=(7, 0),
                        textcoords='offset points', va='center', fontsize=12, color=color)
    motion.set(xlim=(-1, 31), xticks=times*1e3, xlabel='Accepted time (ms)',
               ylabel='Horizontal displacement − mean wall motion (mm)')
    count = 'three' if len(marked) == 3 else 'two'
    motion.set_title(f'Differential motion of {count} vertices', loc='left', pad=13)
    motion.grid(axis='y', color='#dce3e3', linewidth=.7)
    motion.set_axisbelow(True)
    args.destination.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        metadata = {'CreationDate': None, 'ModDate': None} if extension == 'pdf' else None
        fig.savefig(args.destination/f'layered-motion.{extension}', dpi=350,
                    bbox_inches='tight', pad_inches=.18, metadata=metadata)
    plt.close(fig)
    print(f'Mean wall motion at {1e3*times[-1]:g} ms: {1e3*mean_wall_motion[-1]:g} mm')
    for index in marked:
        print(f'ID {ids[index]}: x = {positions[0,index,0]:.10g} to {positions[-1,index,0]:.10g} m, '
              f'Δx = {1e3*dx[-1,index]:.9f} mm, relative displacement = {offsets[-1,index]:.9f} mm.')


if __name__ == '__main__':
    main()
