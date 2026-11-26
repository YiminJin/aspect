"""Plot the accepted growing-seed phase and the first turning material section."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
import numpy as np

from compare import rows


def bernstein(x):
    return np.stack(((1-x)**2, 2*x*(1-x), x**2), axis=-1)


def phase_patches(output, step, origin, extent, repetitions, component):
    """Evaluate the exported DGQ2 field separately on each Cartesian cell."""
    points = rows(output.glob(f'points-{step:04}-r*.csv'), ('cell', 'q'))
    fields = rows(output.glob(f'fields-{step:04}-r*.csv'), ('cell', 'component', 'local'))
    cells = defaultdict(list)
    for (cell, _), point in points.items():
        cells[cell].append([float(point[k]) for k in ('x', 'y', 'phi')])

    widths = extent / repetitions
    unit = np.linspace(0, 1, 65)
    basis = bernstein(unit)
    patches = {}
    for cell, samples in cells.items():
        x, y, phi = np.array(samples).T
        indices = np.floor((np.column_stack((x, y)) - origin) / widths).astype(int)
        np.testing.assert_array_equal(indices, np.broadcast_to(indices[0], indices.shape))
        lower = origin + indices[0] * widths
        coefficients = np.array([float(fields[cell, component, j]['value']) for j in range(9)]).reshape(3, 3)
        # In tensor Bernstein order the x index varies fastest. Verify the
        # mapping, basis and component against the native point observations.
        native_x = bernstein((x - lower[0]) / widths[0])
        native_y = bernstein((y - lower[1]) / widths[1])
        np.testing.assert_allclose(np.einsum('qi,ij,qj->q', native_y, coefficients, native_x),
                                   phi, rtol=0, atol=1e-13)
        key = tuple(indices[0])
        if key in patches:
            raise ValueError(f'Multiple DG patches at Cartesian index {key}')
        patches[key] = (lower[0] + widths[0]*unit, lower[1] + widths[1]*unit,
                        basis @ coefficients @ basis.T)
    if len(patches) != np.prod(repetitions):
        raise ValueError('Expected the complete unrefined Cartesian seed mesh')
    return patches


def first_turn(output, tip, tangent, normal, origin):
    seed = rows([output/'sections-0000.csv'], ('section',))
    accepted = rows([output/'sections-0001.csv'], ('section',))
    if seed.keys() != accepted.keys():
        raise ValueError('Material section identities changed')
    # Reconstruct the station-to-station graph. The arriving station owns
    # the signed angle; xi is distance projected on the original tangent.
    joint = tip
    previous_xi = (tip - origin) @ tangent
    for (section,), strip in sorted(accepted.items()):
        xi = float(strip['xi'])
        if xi <= previous_xi:
            continue
        if not int(strip['formed']):
            break
        beta = float(strip['beta'])
        arrival = joint + (xi - previous_xi) * (tangent + np.tan(beta)*normal)
        if beta != 0 and not int(seed[section,]['formed']):
            width = float(seed[section,]['ds'])
            np.testing.assert_allclose(float(strip['ds']), width / np.cos(beta), rtol=0, atol=1e-14)
            return section, joint, arrival, beta
        joint, previous_xi = arrival, xi
    raise ValueError('No newly formed section turned ahead of the seed tip')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='ordinary growing fault history output directory')
    parser.add_argument('destination', type=Path, help='directory for growing-seed.png and growing-seed.pdf')
    args = parser.parse_args()
    prm = json.loads((args.output/'parameters.json').read_text())
    box = prm['Geometry model']['Box']
    origin, extent, repetitions = (
        np.array([float(box[name.format(d)]['value']) for d in ('X', 'Y')])
        for name in ('Box origin {} coordinate', '{} extent', '{} repetitions'))
    names = prm['Compositional fields']['Names of fields']['value'].replace(' ', '').split(',')
    phase = prm['Phase field']['Phase composition']['value'] or 'phase_field'
    component = 4 + names.index(phase)  # 2 velocities, pressure, temperature
    patches = [phase_patches(args.output, step, origin, extent, repetitions, component)
               for step in (0, 1)]
    time_ms = 1e3 * float(rows([args.output/'balance-0001.csv'], ('step',))[1,]['time'])
    np.testing.assert_allclose(time_ms, 5., rtol=0, atol=1e-12)
    material = prm['Material model']['Growing fault']
    alpha = np.deg2rad(float(material['Seed tangent angle']['value']))
    tangent = np.array([np.cos(alpha), np.sin(alpha)])
    normal = np.array([-tangent[1], tangent[0]])
    tip = np.array([float(v) for v in material['Forward seed tip']['value'].split(',')])
    section, joint, arrival, beta = first_turn(args.output, tip, tangent, normal, origin)
    along = (arrival-joint) @ tangent
    offset = (arrival-joint) @ normal

    plt.rcParams.update({'font.size': 13, 'axes.titlesize': 16, 'axes.titleweight': 'bold',
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, (field, detail) = plt.subplots(1, 2, figsize=(12.5, 4.8),
                                        gridspec_kw={'width_ratios': [1.65, 1]})
    fig.subplots_adjust(left=.07, right=.975, top=.87, bottom=.29, wspace=.24)
    scale = colors.Normalize(0, np.ceil(max(phi.max() for _, _, phi in patches[1].values())*20)/20)
    # Keep DG jumps at their cell boundaries instead of interpolating across cells.
    for key, (x, y, phi) in patches[1].items():
        mesh = field.pcolormesh(x, y, (phi[:-1, :-1]+phi[:-1, 1:]+phi[1:, :-1]+phi[1:, 1:])/4,
                                cmap='Blues', norm=scale, shading='flat', edgecolors='face',
                                linewidth=.03)
        initial = patches[0][key][2]
        if initial.min() < .05 < initial.max():
            field.contour(x, y, initial, levels=[.05], colors='#303943', linewidths=1.4,
                          linestyles='dashed')
    turn_color = '#ad371a'
    field.scatter(*tip, s=51, color='white', edgecolor='#182d41', linewidth=1.6, zorder=4)
    field.plot(*np.array([tip, joint]).T, color='#677585', lw=1.7)
    field.plot(*np.array([joint, arrival]).T, 'o-', color=turn_color, lw=3, ms=5)
    field.set(xlim=(origin[0], origin[0]+extent[0]), ylim=(origin[1], origin[1]+extent[1]),
              xlabel='x (m)', ylabel='y (m)', aspect='equal')
    field.set_title(f'Accepted phase at {time_ms:g} ms', loc='left', pad=13)
    key = [Line2D([], [], ls='--', color='#303943', label='Seed at 0 ms (φ = 0.05)'),
           Line2D([], [], marker='o', ls='none', markerfacecolor='white', markeredgecolor='#182d41',
                  label='Prescribed seed tip')]
    field.legend(handles=key, loc='lower right', framealpha=.96, fontsize=10.5)
    colorbar = fig.colorbar(mesh, cax=fig.add_axes([.07, .105, .496, .030]), orientation='horizontal')
    colorbar.solids.set_rasterized(False)
    colorbar.solids.set_edgecolor('face')
    colorbar.set_label('Phase field, φ (dimensionless)')

    # Show physical distances with equal scales so the angle is undistorted.
    detail.axhline(0, color='#677585', linestyle='--', lw=1.4)
    detail.plot([0, along], [0, offset], 'o-', color=turn_color, ms=6, lw=3)
    detail.annotate(f'Station {section-1}', (0, 0), xytext=(-4, 10), textcoords='offset points')
    detail.annotate(f'Station {section}', (along, offset), xytext=(-1, -21), ha='right',
                    textcoords='offset points', color=turn_color)
    angles = np.linspace(beta, 0)
    radius = .49*along
    detail.plot(radius*np.cos(angles), radius*np.sin(angles), color='#283743', lw=1.3)
    detail.annotate(f'{abs(np.rad2deg(beta)):.2f}° clockwise',
                    (.5*along, .25*offset), xytext=(.62*along, .15*along),
                    arrowprops={'arrowstyle': '-', 'color': '#5e6870'}, color='#242e38')
    detail.set(xlim=(-.11*along, 1.18*along), ylim=(-.38*along, .35*along),
               xlabel='Along seed from bend, Δs (m)', ylabel='Normal offset, Δn (m)', aspect='equal')
    detail.set_title(f'Selected direction in strip {section}', loc='left', pad=13)
    detail.yaxis.set_major_locator(plt.MultipleLocator(.05))
    args.destination.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        metadata = {'CreationDate': None, 'ModDate': None} if extension == 'pdf' else None
        fig.savefig(args.destination/f'growing-seed.{extension}', dpi=350, bbox_inches='tight',
                    pad_inches=.18, metadata=metadata)
    plt.close(fig)
    print(f'Strip {section} at {time_ms:g} ms: beta {beta:.16g} rad, {np.degrees(beta):.9f} degrees; '
          f'laboratory inclination {np.degrees(alpha+beta):.9f} degrees. '
          f'Station separation in seed coordinates ({along:.9g}, {offset:.9g}) m.')


if __name__ == '__main__':
    main()
