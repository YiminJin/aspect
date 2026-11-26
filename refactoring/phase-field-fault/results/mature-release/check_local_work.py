"""Measure the projected Stokes work in the contact and bend cells of a resolved band."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prefix', 'observed_curve', 'destination'):
        parser.add_argument(name, type=lambda s: Path(s).resolve())
    args = parser.parse_args()
    bench = args.prefix/'benchmarks/fixed_fault'
    sys.path[:0] = [str(bench/'transfer_work'), str(bench/'curved_mature')]
    from run import run_pair
    from check_curve import Fields, association, cell_rule, compliance, normal_integrals, table

    os.environ['ASPECT_SOURCE_DIR'] = str(args.prefix)
    # The real middle-row probe has three labels. Keep both contacts and the
    # central bend, and use the very same resolved DG phase as the five-ID solve.
    vertices = np.array([[0., .82], [.5, .69], [1., .14]])
    settings = f'''include {bench}/curved_mature/curved_mature.prm
set Output directory = output
set End time = 0
subsection Material model
  set Model name = fault transfer work probe
  subsection Phase field fault
    set Fault vertex file =
    set Fault vertices = 0,0.82; 0.5,0.69; 1,0.14
  end
end
subsection Postprocess
  set List of postprocessors =
end
'''
    measured = run_pair(args.prefix/'bin/aspect-release', bench/'transfer_work/libtransfer_work.release.so',
                        args.destination, settings)
    field = Fields(args.observed_curve, 0)
    q, w = cell_rule(8, field.n)
    edges = np.diff(vertices, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    t = edges/lengths[:, None]
    n = np.column_stack([-t[:, 1], t[:, 0]])
    I = normal_integrals(vertices, t, lengths, field, 24)
    np.testing.assert_allclose(I, measured['normal_integrals'], rtol=0., atol=2e-13)
    k, xi, count = association(q, vertices)
    middle = np.where(k == 0, xi, 1-xi)
    weight = np.where(count > 0, compliance(field.at('phi', q))/((1-xi)*I[k]+xi*I[k+1]), 0.)
    # MeasuredLoad pairs the unconstrained local vector with the affine
    # field t0 (n0·x) in the first segment's geometric frame. Slip sense is -1.
    grad = np.outer(t[0], n[0])
    resolved = -np.einsum('ki,ij,kj->k', t, grad+grad.T, n)
    expected = np.einsum('ijab,ab->ij', (weight*middle*resolved[k]).reshape(field.n,field.n,8,8),
                         w/field.n**2)
    ids = {}
    for shard in args.observed_curve.glob('material-0-rank*.csv'):
        for row in table(shard):
            ij = tuple(np.floor(field.n*np.array([float(row['x']), float(row['y'])])).astype(int))
            ids[ij] = row['cell']
    actual = np.array([[measured['stokes_cells'][ids[i,j]] for j in range(field.n)] for i in range(field.n)])
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=0.)
    np.testing.assert_allclose(measured['section'], expected.sum(), atol=2e-13, rtol=0.)
    for location, ij in [('contact', (0,int(vertices[0,1]*field.n))),
                         ('bend', tuple(np.floor(field.n*vertices[1]).astype(int)))]:
        if abs(actual[ij]) < 1e-6:
            raise ValueError(f'Middle-label perturbation did not reach {location} cell {ij}')
        print(f'{location} cell {ids[ij]}: Stokes {actual[ij]:.12g}, independent {expected[ij]:.12g} m')
    print(f"all cells: Stokes {actual.sum():.15g}, reverse {measured['section']:.15g}, "
          f"independent {expected.sum():.15g} m; max cell defect {np.max(abs(actual-expected)):.3g} m")


if __name__ == '__main__':
    main()
