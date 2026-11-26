"""Pair work at fixed stress, then measure up-dip relative motion of a solved 60-degree fault."""
import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from run import KAPPA, run_pair


def frame(vertices):
    edges = np.diff(vertices, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    tangents = edges / lengths[:, None]
    return lengths, tangents, np.column_stack((-tangents[:, 1], tangents[:, 0]))


def q8_work(vertices, integrals):
    """Independently resolve the physical virtual shear on each eligible local normal."""
    lengths, tangents, normals = frame(vertices)
    grad = np.outer(tangents[0], normals[0])
    engineering_strain = grad + grad.T
    resolved = np.einsum('ik,kl,il->i', tangents, engineering_strain, normals)
    xi, w = np.polynomial.legendre.leggauss(8)
    xi, w = (1+xi)/2, w/2
    weight = .35*(1+.35)/(1-.35)**2
    contributions = [[], []]
    for y, wy in zip(xi, w):
        for x, wx in zip(xi, w):
            offsets = np.array([x, y]) - vertices[:-1]
            station = np.sum(offsets*tangents, axis=1)/lengths
            eligible, = np.where((station >= 0) & (station <= 1))
            if len(eligible) == 0:
                continue
            distances = abs(np.sum(offsets[eligible]*normals[eligible], axis=1))
            j = eligible[np.argmin(distances)]
            s = station[j]
            mid_hat = s if j == 0 else 1-s
            normal_integral = (1-s)*integrals[j] + s*integrals[j+1]
            contributions[j].append(wx*wy*mid_hat*weight/normal_integral)
    measures = np.array([math.fsum(c) for c in contributions])
    if min(measures) <= 0:
        raise ValueError('Work probe needs active samples on both segments')
    return resolved @ measures, measures, resolved


def vertex_entries(vertices):
    return '; '.join(f'{x:.17g},{y:.17g}' for x, y in vertices)


def compare_work(aspect, plugin, root):
    down = np.array([.5, -math.sqrt(3)/2])
    center = np.array([.5, .5])
    straight = center + np.array([-.5, 0., .5])[:, None]*down/abs(down[1])
    curved = np.array([[.12, 1.], [.33, .41], [.85, 0.]])
    parent = np.array([[.13, .04], [.04, -.08]])
    for name, vertices in [('down-dip', straight), ('up-dip', straight[::-1]), ('bend', curved)]:
        lengths, tangents, normals = frame(vertices)
        ages = [9., 12., 16.]
        # The physical age and background loading attached to each station
        # follow that station through reversal; both cancel from paired work.
        shears = [.15, -.07, .09]
        if name == 'up-dip':
            ages.reverse()
            shears.reverse()
        records = []
        for sign, sense in [(1, 'along tangent'), (-1, 'opposite tangent')]:
            dest = root / f'{name}-{sign:+d}'
            prm = f"""
subsection Initial composition model
  subsection Function
    set Function expression = {parent[0,0]};{parent[1,1]};{parent[0,1]};0.35
  end
end
subsection Material model
  subsection Phase field fault
    set Fault vertices = {vertex_entries(vertices)}
    set Mature slip sense = {sense}
    set Initial label ages = {', '.join(map(str, ages))}
    set Initial label shear tractions = {', '.join(map(str, shears))}
  end
end
"""
            result = run_pair(aspect, plugin, dest, prm)
            expected, measures, resolved = q8_work(vertices, result['normal_integrals'])
            for key in ('stokes', 'section'):
                np.testing.assert_allclose(result[key], sign*expected, rtol=0., atol=2e-13)
            if name != 'bend':
                np.testing.assert_allclose(expected, .58901895576274, rtol=0., atol=2e-13)
            else:
                chord = vertices[2]-vertices[0]
                chord /= np.linalg.norm(chord)
                normal = np.array([-chord[1], chord[0]])
                grad = np.outer(tangents[0], normals[0])
                chord_work = measures.sum()*(chord @ (grad+grad.T) @ normal)
                if abs(chord_work-expected) <= .01:
                    raise ValueError('The bent-trace experiment failed to distinguish the chord frame')
                print(f'bend: local engineering shear per segment {resolved}; chord-only prediction {sign*chord_work:.15g} m')
            # Observe actual normal traction at every eligible production point;
            # the anisotropic incoming stress prevents a pressure-only pass.
            values = np.genfromtxt(dest/'minus/output/compression.csv', delimiter=',', names=True)
            j = values['segment'].astype(int)
            n = normals[j]
            compression = 5 - math.exp(-.1)*np.einsum('qi,ij,qj->q', n, parent, n)
            grad = np.outer(tangents[0], normals[0])
            with_shear = compression - KAPPA*np.einsum('qi,ij,qj->q', n, grad+grad.T, n)
            np.testing.assert_allclose(values['rest_Pa'], compression, atol=2e-13, rtol=0.)
            np.testing.assert_allclose(values['shear_Pa'], with_shear, atol=2e-13, rtol=0.)
            records.append(values)
            print(f"{name}, {sense}: Stokes {result['stokes']:.15g}, reverse {result['section']:.15g},"
                  f" independent {sign*expected:.15g} m ({len(j)} active Q8 points)", flush=True)
        np.testing.assert_array_equal(records[0], records[1])
        print(f"{name}: compression identical by point, {min(records[0]['shear_Pa']):.12g} to"
              f" {max(records[0]['shear_Pa']):.12g} Pa", flush=True)
    return straight


def solved_motion(aspect, root, vertices, checkpoint, geometry="boundary spanning"):
    up_dip = np.array([-.5, math.sqrt(3)/2])
    normal = np.array([up_dip[1], -up_dip[0]])
    stations = np.array([.5 + .075*normal, .5 - .075*normal])
    outcomes = []
    for name, sense in [('thrust', 'opposite tangent'), ('companion', 'along tangent')]:
        dest = root/name
        dest.mkdir()
        prm = dest/'case.prm'
        prm.write_text(f"""include $ASPECT_SOURCE_DIR/benchmarks/fixed_fault/stationary.prm
set Output directory = {dest/'output'}
subsection Mesh refinement
  set Initial global refinement = 3
end
subsection Material model
  subsection Phase field fault
    set Fault geometry = {geometry}
    set Fault vertices = {vertex_entries(vertices)}
    set Fit trace to phase = false
    set Mature slip sense = {sense}
    set Background shear traction = 3.71
  end
end
subsection Boundary traction model
  subsection Function
    set Function expression = 0;0
  end
end
subsection Initial composition model
  subsection Function
    set Function expression = 0;0;0;0.08+0.42*exp(-((sqrt(3)*(x-0.5)+(y-0.5))/0.3)^2)
  end
end
subsection Postprocess
  set List of postprocessors = point values, mature fault statistics
  subsection Point values
    set Evaluation points = {vertex_entries(stations)}
  end
end
""")
        if name == 'thrust' and checkpoint:
            with prm.open('a') as f:
                f.write('subsection Checkpointing\n set Steps between checkpoint = 1\n'
                        ' set Number of checkpoints to keep = 4\nend\n')
        with (dest/'log').open('w') as log:
            subprocess.run([aspect, prm], cwd=dest, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = np.loadtxt(dest/'output/point_values.txt')
        rows = rows[rows[:, 0] == .025]
        # ASPECT point values prints coordinates and Cartesian velocities at six
        # significant digits. Check the station, then pair measured velocities.
        np.testing.assert_allclose(rows[:, 1:3], stations, rtol=0., atol=5e-7)
        labels = np.loadtxt(dest/'output/fault-labels-00002.txt')
        if not (np.isfinite(labels[:, -2:]).all() and (labels[:, -2:] > 0.).all()):
            raise ValueError('Expected positive accepted speed and slip distance')
        delta = (rows[0, 3:5]-rows[1, 3:5]) @ up_dip
        outcomes.append(delta)
        print(f'{name}: (u_hanging-u_foot) dot up-dip = {delta:.9g} m/s at 25 ms;'
              f' midpoint slip speed {labels[1,-1]:.12g} m/s', flush=True)
    print(f'Hanging wall {stations[0]} m, footwall {stations[1]} m, up dip {up_dip}')
    if not outcomes[0] > 0. > outcomes[1]:
        raise ValueError('Reversing slip sense did not reverse the relative tangential motion')
    if checkpoint:
        clean = root/'thrust'
        dest = root/'thrust-resume'
        saved = clean/'output/restart/02'
        shutil.copytree(saved, dest/'output/restart/02')
        prm = dest/'case.prm'
        prm.write_text(f'include {clean/"case.prm"}\nset Output directory = {dest/"output"}\n'
                       'set Resume computation = true\nsubsection Checkpointing\n set Resume checkpoint = 2\nend\n')
        with (dest/'log').open('w') as log:
            subprocess.run([aspect, prm], cwd=dest, stdout=log, stderr=subprocess.STDOUT, check=True)
        for filename in ('fault-labels-00002.txt', 'point_values.txt'):
            if (dest/'output'/filename).read_bytes() != (clean/'output'/filename).read_bytes():
                raise ValueError(f'Restored right-dipping thrust differs at {filename}')
        print('Boundary-spanning thrust restored generation 02; endpoint labels and both point histories byte-identical.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('aspect', 'plugin', 'directory'):
        parser.add_argument(name, type=lambda path: Path(path).resolve())
    parser.add_argument('--checkpoint', action='store_true', help='also resume the thrust from its fresh checkpoint')
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('ASPECT_SOURCE_DIR', str(Path(__file__).resolve().parents[3]))
    os.environ.update(DEAL_II_NUM_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    vertices = compare_work(args.aspect, args.plugin, args.directory)
    solved_motion(args.aspect, args.directory, vertices, args.checkpoint)
    # Fixed interior uses a separate trial section and a projected stress pair.
    # Shorten the same seed around the middle vertex so both endpoints are interior.
    interior = args.directory/'fixed-interior'
    interior.mkdir()
    solved_motion(args.aspect, interior, .5+.8*(vertices-.5), False, 'fixed interior')


if __name__ == '__main__':
    main()
