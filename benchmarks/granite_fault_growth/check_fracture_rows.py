"""Check the production AT1 mass and SIP matrix on a small adaptive Cartesian mesh.

Independently integrate each physical interface piece once, finding neighbors by
rectangle intersection. Pair physical polynomial traces on both sides, with
the normal from left to right or bottom to top. All results below are divided
by Kf=Gc/(c0 ell), as for FractureRows (units m² in 2-D).
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix


@dataclass
class Cell:
    name: str
    lo: np.ndarray
    hi: np.ndarray
    dofs: np.ndarray
    owner: int

    @property
    def size(self):
        return self.hi - self.lo

    def shape(self, points):
        """Physical tensor Bernstein Q2 values and x/y derivatives, x fastest."""
        x, y = ((points-self.lo)/self.size).T
        def basis(t):
            return (np.array([(1-t)**2, 2*t*(1-t), t**2]).T,
                    np.array([2*(t-1), 2-4*t, 2*t]).T)
        bx, dx = basis(x)
        by, dy = basis(y)
        def tensor(v, u):
            return (v[:, :, None]*u[:, None, :]).reshape(-1, 9)
        val = tensor(by, bx)
        grad = [tensor(by, dx)/self.size[0], tensor(dy, bx)/self.size[1]]
        return val, grad


def read_rows(path, step, ranks):
    cells, mass, entries = [], {}, []
    scale = None
    for rank in range(ranks):
        shard = path/f"fracture-rows-{step:04d}-r{rank}.txt"
        for line in shard.read_text().splitlines():
            kind, *words = line.split()
            if kind == "scale":
                settings = tuple(map(float, words))
                assert scale in (None, settings)
                scale = settings
            elif kind == "cell":
                coords = np.array(words[1:5], float)
                cells.append(Cell(words[0], coords[:2], coords[2:],
                                  np.array(words[5:], int), rank))
            elif kind == "mass":
                i, val = words
                assert int(i) not in mass, (rank, i)
                mass[int(i)] = float(val)
            elif kind == "stiffness":
                i, j, val = words
                entries.append((int(i), int(j), float(val)))
            else:
                raise ValueError(line)
    n = 9*len(cells)
    assert set(c.owner for c in cells) == set(range(ranks))
    assert sorted(i for c in cells for i in c.dofs) == list(range(n))
    mass = np.array([mass[i] for i in range(n)])
    i, j, v = zip(*entries)
    assert np.all(np.isfinite(v)) and np.all(np.isfinite(mass))
    return scale, cells, mass, coo_matrix((v, (i, j)), shape=(n, n)).tocsr()


def check(path, step, ranks):
    (ell, penalty), cells, mass, actual = read_rows(path, step, ranks)
    k = 2*ell**2
    t, w = np.polynomial.legendre.leggauss(7)
    t, w = (t+1)/2, w/2
    grid = np.array(np.meshgrid(t, t)).reshape(2, -1).T
    weights = np.outer(w, w).ravel()
    n = 9*len(cells)
    stage, direction, continuous = np.zeros((3, n))
    local_mass = np.zeros(n)
    boundary_flux = np.zeros(n)
    box_lo = np.min([c.lo for c in cells], axis=0)
    box_hi = np.max([c.hi for c in cells], axis=0)
    I, J, V = [], [], []
    def block(rows, cols, a):
        I.extend(np.repeat(rows, len(cols)))
        J.extend(np.tile(cols, len(rows)))
        V.extend(a.ravel())

    for c in cells:
        area = np.prod(c.size)
        v, grad = c.shape(c.lo+grid*c.size)
        local_mass[c.dofs] = area*weights@v
        block(c.dofs, c.dofs, k*area*sum(g.T@(weights[:, None]*g) for g in grad))
        # Deliberately discontinuous cell-local shapes, tied to physical cells
        # rather than partition-dependent DoF indices.
        j = np.arange(9)
        stage[c.dofs] = .33 + .17*np.cos(2.1*c.lo[0]+3.4*c.hi[1]+.83*j)
        direction[c.dofs] = .4*np.sin(c.hi[0]-2.7*c.lo[1]+.64*j)
        x, y = c.lo[:, None]+c.size[:, None]*np.array([j%3, j//3])/2
        continuous[c.dofs] = .1+.2*x+.3*y+.13*x*y
        # For the globally bilinear field the volume Laplacian and all
        # interior jumps vanish. Only the outer Neumann flux remains in K b.
        for d in range(2):
            e = 1-d
            for side, sign, wall in ((c.lo, -1, box_lo), (c.hi, 1, box_hi)):
                if not np.isclose(wall[d], side[d], rtol=0., atol=1e-13):
                    continue
                points = np.empty((len(w), 2))
                points[:, d], points[:, e] = side[d], c.lo[e] + t*c.size[e]
                val, _ = c.shape(points)
                derivative = (.2 if d == 0 else .3) + .13*points[:, e]
                boundary_flux[c.dofs] += k*sign*c.size[e] * ((w*derivative)@val)

    hanging = remote = 0
    omitted = []
    for left in cells:
        for right in cells:
            for d in range(2):
                e = 1-d
                if not np.isclose(left.hi[d], right.lo[d], rtol=0., atol=1e-13):
                    continue
                lo, hi = max(left.lo[e], right.lo[e]), min(left.hi[e], right.hi[e])
                if hi-lo <= 1.e-13:
                    continue
                pts = np.empty((len(w), 2))
                pts[:, d], pts[:, e] = left.hi[d], lo+(hi-lo)*t
                lv, lg = left.shape(pts)
                rv, rg = right.shape(pts)
                jump = np.column_stack((lv, -rv))
                flux = .5*k*np.column_stack((lg[d], rg[d]))
                measure = (hi-lo)*w[:, None]
                sigma = 9*penalty*k/min(left.size[d], right.size[d])
                a = jump.T@(measure*(sigma*jump-flux)) - flux.T@(measure*jump)
                indices = np.r_[left.dofs, right.dofs]
                block(indices, indices, a)
                if not np.allclose(left.size, right.size, rtol=0., atol=1e-13):
                    hanging += 1
                    remote += int(left.owner != right.owner)
                    # The second half of a coarse face cannot be integrated
                    # by reusing the neighbor's first half. Dropping one piece
                    # is a concrete discrimination control.
                    omitted.append(max(abs(a@stage[indices])))
    assert hanging and (ranks == 1 or remote), (hanging, remote, ranks)
    ref = coo_matrix((V, (I, J)), shape=(n, n)).tocsr()
    units = max(1., abs(ref).max())
    h = 1e-4
    residual = lambda x: local_mass+ref@x
    fd_action = (residual(stage+h*direction)-residual(stage-h*direction))/(2*h)
    def norm(x):
        return float(np.max(np.abs(x)))
    results = dict(cells=len(cells), hanging_pieces=hanging, hanging_partition_pieces=remote,
                   mass_error=norm(local_mass-mass), matrix_error=float(abs(ref-actual).max()),
                   discontinuous_residual_error=norm(residual(stage)-(mass+actual@stage)),
                   directional_action_error=norm(fd_action-actual@direction),
                   continuous_flux_error=norm(actual@continuous-boundary_flux),
                   omitted_subface_error=max(omitted))
    # Geometrical Gauss rules are exact through degree 4 for these polynomial
    # terms. Accumulation, element evaluation and the centered difference give
    # the following absolute roundoff budgets relative to the largest row entry.
    for name in ("mass_error", "matrix_error", "discontinuous_residual_error", "continuous_flux_error"):
        assert results[name] <= units*1.e-12, (name, results)
    assert results["directional_action_error"] <= units*2.e-10, results
    assert results["omitted_subface_error"] > units*1.e-5, results
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--ranks", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.output, args.step, args.ranks), indent=2))
