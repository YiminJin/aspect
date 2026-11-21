"""Write the small fixed-frame covariance or nonaligned specimen as a native input."""
import argparse
import math
from pathlib import Path


def vector(degrees):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return tuple(0. if abs(v) < 1e-15 else v for v in (c, s))


def fixture(mesh_angle, frame_angle, rotated_mesh, normal_rate=0., one_ended=False):
    base = Path(__file__).resolve().parents[1] / 'bulk_tip/finite_tip.prm'
    text = base.read_text()
    # Follow the original box center through the rigid rotation. For the cut
    # case the seed and loading turn about the center of the unchanged box.
    m, n = vector(mesh_angle)
    c, s = vector(frame_angle)
    cx, cy = m-.5*n, n+.5*m
    xi = f'(({c:.17g})*(x-({cx:.17g}))+({s:.17g})*(y-({cy:.17g})))'
    zeta = f'((-({s:.17g}))*(x-({cx:.17g}))+({c:.17g})*(y-({cy:.17g})))'
    seed_xi = f'max(0,{xi})' if one_ended else xi
    text = text.replace('((x-1)/.5)', f'({seed_xi}/.5)').replace('((y-.5)/.25)', f'({zeta}/.25)')
    ut, un = f'(.02*{zeta}+({normal_rate:.17g})*{xi})', f'(-({normal_rate:.17g})*{zeta})'
    text = text.replace('.02*(y-.5);0', f'({c:.17g})*{ut}-({s:.17g})*{un};({s:.17g})*{ut}+({c:.17g})*{un}')
    text = text.replace('set Surface pressure = 1000', 'set Surface pressure = 200')
    text = text.replace('  set Bulk tip initial age', f'  set Bulk tip reference angle = {frame_angle:.17g}\n'
                        '  set Bulk tip section closure = material strips\n  set Bulk tip initial age')
    if rotated_mesh:
        text = text.replace('set Model name = box', 'set Model name = section test box\n'
                            f' set Section test box rotation = {mesh_angle:.17g}\n'
                            ' subsection Initial topography model\n  set Model name = function\n'
                            '  subsection Function\n   set Function expression = 0\n'
                            '   set Maximum topography value = 0\n  end\n end')
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--mesh-angle', type=float, default=0.)
    parser.add_argument('--frame-angle', type=float, default=0.)
    parser.add_argument('--rotated-mesh', action='store_true')
    parser.add_argument('--normal-rate', type=float, default=0., help='Deviatoric normal strain rate in the specimen frame')
    args = parser.parse_args()
    with args.destination.open('x') as stream:
        stream.write(fixture(args.mesh_angle, args.frame_angle, args.rotated_mesh, args.normal_rate))
