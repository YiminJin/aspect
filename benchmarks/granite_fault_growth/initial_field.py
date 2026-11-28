"""Check the published initial granite field and extract its connected midline tips."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


def basis(t):
    return np.stack(((1-t)**2, 2*t*(1-t), t*t), axis=-1)


def reference_phase(x, y):
    return 0.95 * np.maximum(0, 1-abs(y-0.375)/0.125)**2 * np.clip((1.125-x)/0.125, 0, 1)**2


def measure(directory, nx, snapshot, shear_prestress=0.):
    h = 1.5/nx
    native_error = polynomial_error = ahead_phase = stress_error = area = 0.
    minimum_shear, maximum_shear = np.inf, -np.inf
    count = 0
    patches = np.full((nx//2, nx, 3, 3), np.nan)
    for shard in sorted(directory.glob("points-0000-r*.csv")):
        coefficients = {}
        with shard.with_name(shard.name.replace("points-", "fields-")).open() as stream:
            for row in csv.DictReader(stream):
                if int(row["component"]) == 4:
                    patch = coefficients.setdefault(row["cell"], np.full(9, np.nan))
                    patch[int(row["local"])] = float(row["value"])
        data = np.genfromtxt(shard, names=True, delimiter=",", dtype=None, encoding="utf8")
        assert all(np.isfinite(data[k]).all() for k in data.dtype.names if k != "cell")
        # Native output groups points by CellId. Use mapped coordinates to place
        # each cell, independently of the partition and deal.II's CellId order.
        ids, first, counts = np.unique(data["cell"], return_index=True, return_counts=True)
        for cell, start, n in zip(ids, first, counts):
            rows = data[start:start+n]
            assert np.all(rows["cell"] == cell)
            x, y, phi = rows["x"], rows["y"], rows["phi"]
            ix, iy = int(x[0]/h), int(y[0]/h)
            patch = coefficients[cell].reshape(3, 3)
            assert np.isnan(patches[iy, ix]).all() and np.isfinite(patch).all()
            patches[iy, ix] = patch
            evaluated = np.einsum("qi,ij,qj->q", basis(y/h-iy), patch, basis(x/h-ix))
            polynomial_error = max(polynomial_error, np.max(abs(evaluated-phi)))
            native_error = max(native_error, np.max(abs(reference_phase(x, y)-phi)))
            if ix*h >= 1.125:
                ahead_phase = max(ahead_phase, np.max(abs(phi)))
            minimum_shear = min(minimum_shear, float(np.min(rows["tauxy"])))
            maximum_shear = max(maximum_shear, float(np.max(rows["tauxy"])))
            stress_error = max(stress_error, np.max(abs(rows["tauxy"] - shear_prestress)),
                               *(np.max(abs(rows[c])) for c in ("tauxx", "tauyy", "tauzz")))
            for column in ("Axx", "Ayy", "Azz", "Axy", "signed_shear", "absolute_shear"):
                if np.any(rows[column] != 0):
                    raise ValueError(f"Nonzero initial {column} in cell {cell}")
            area += sum(rows["weight"])
            count += n
    assert np.isfinite(patches).all() and max(native_error, polynomial_error) < 1e-12
    assert count == nx*(nx//2)*36 and abs(area-1.125) < 1e-12
    assert ahead_phase == 0 and stress_error == 0, (minimum_shear, maximum_shear, stress_error)
    stations = np.genfromtxt(directory/"sections-0000.csv", delimiter=",", names=True)
    assert len(stations) == nx and np.all(stations["age"] == 1)
    for column in ("slip", "V", "formed", "force"):
        assert np.all(stations[column] == 0)
    # The two traces at mid-height bound the seed above and below. Verify the
    # complete cell polynomials against the seed (nine unisolvent points/cell).
    for i, j in np.ndindex(3, 3):
        xi, eta = j/2, i/2
        x = (np.arange(nx)+xi)*h
        y = (np.arange(nx//2)+eta)*h
        np.testing.assert_allclose(basis(eta) @ patches @ basis(xi),
                                   reference_phase(x[None, :], y[:, None]), atol=1e-13, rtol=0)
    ends = {}
    for level in (0.01, 0.05):
        tips = []
        for edge in (patches[nx//4-1, :, -1, :], patches[nx//4, :, 0, :]):
            # On this initial field the connected contour is furthest forward
            # on the midline. Walk out from the left boundary to the first exit.
            ix = next(i for i, row in enumerate(edge) if row[-1] < level)
            tips.append(h*(ix+brentq(lambda t: edge[ix] @ basis(t)-level, 0., 1.)))
        expected = 1.125-0.125*np.sqrt(level/0.95)
        np.testing.assert_allclose(tips, expected, atol=1e-13, rtol=0)
        ends[str(level)] = float(max(tips))
    np.savez_compressed(snapshot, h=h, phase=patches)
    return dict(cells=nx*nx//2, points=int(count), sections=nx, max_native_error=float(native_error),
                max_reconstruction_error=float(polynomial_error), area_m2=area,
                peak_abs_stress_Pa=max(abs(minimum_shear), abs(maximum_shear)),
                shear_range_Pa=[minimum_shear, maximum_shear], max_phase_ahead=ahead_phase, connected_tip_m=ends)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("nx", type=int, choices=(48, 96))
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--shear-prestress", type=float, default=0.,
                        help="Expected time-zero Cartesian xy deviatoric stress in Pa")
    args = parser.parse_args()
    print(json.dumps(measure(args.output, args.nx, args.snapshot, args.shear_prestress), indent=2))
