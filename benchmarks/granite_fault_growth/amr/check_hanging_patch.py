"""Verify accepted Q2 displacement, Q1 pressure, Maxwell stress and wall work.

The prescribed polynomial has div(v)=0, normal strain rates ±a*y and constant
engineering shear gamma. The x-normal strips have unit transverse measure.
With homogeneous phase/friction the shear response is uniform, so the exact
midpoint pressure slope is -2*G*B*h*a, independent of the pressure gauge.
"""

import argparse
import csv
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


def read_table(file):
    data = np.atleast_1d(np.genfromtxt(file, names=True, delimiter=",", dtype=None, encoding="utf8"))
    for column in data.dtype.names:
        if column != "cell" and not np.isfinite(data[column]).all():
            raise ValueError(f"{file}: nonfinite {column}")
    return data


def check(output, ranks):
    ledger, = read_table(output / "balance-0001.csv")
    dt, G, eta = ledger["time"], 1e6, 1e8
    np.testing.assert_allclose(dt, .005, rtol=0, atol=1e-15)
    h = dt / 2
    B = 1 / (1 + h*G/eta)
    a, gamma = .003*(1+40*h), .02*(1+40*h)
    Kf = 26.6666666667/(8/3*.25)
    m = 2*G*Kf/1000**2
    F = (m*.2*1.2)/(m*.2*1.2 + .8**2)

    def resistance(speed):
        x = speed*h/.01
        theta = 10*np.exp(-x) + h*(-np.expm1(-x)/x if x else 1.)
        return 200*.015*np.arcsinh(speed/.002*np.exp((.2+.02*np.log(theta*.001/.01))/.015)) + 1000*speed

    predictor = G*B*h*gamma
    speed = brentq(lambda v: v-F*(predictor-resistance(v))/(G*B*h), 0., gamma, xtol=1e-17)
    shear = (1-F)*predictor + F*resistance(speed)
    expected_work = dt*(8/3 * G*B*h*a*a + 2*shear*gamma)
    work_error = abs(ledger["Wext"] - expected_work)
    adjoint_error = abs(ledger["adjoint_error"])

    # Integral comparison uses the published native weights. Per-DoF support
    # points use the release FE_Q vertex, face, then cell numbering in 2-D.
    support = np.array([[0,0], [1,0], [0,1], [1,1], [0,.5], [1,.5], [.5,0], [.5,1], [.5,.5]])
    gauss = np.polynomial.legendre.leggauss(6)[0]
    boxes = []
    areas = np.zeros(8)
    pressure = []
    velocity_error = stress_error = phase_error = 0.
    for rank in range(ranks):
        fields = read_table(output / f"fields-0001-r{rank}.csv")
        points = read_table(output / f"points-0001-r{rank}.csv")
        if not len(fields) or not len(points):
            raise ValueError(f"Empty owner {rank}")
        areas += np.bincount(points["section"], weights=points["weight"], minlength=8)
        for cell in np.unique(points["cell"]):
            q = points[points["cell"] == cell]
            y = q["y"]
            side = (max(y)-min(y))*2/(gauss[-1]-gauss[0])
            np.testing.assert_allclose(side, .25 if side < .375 else .5, atol=2e-14, rtol=0)
            # The horizontal native chords cover the square, with symmetric
            # clipping in x even when a coarse cell spans two strips.
            center = np.array([(min(q[k])+max(q[k]))/2 for k in ("x","y")])
            lo = np.round((center-side/2)*4).astype(int)
            width = round(side*4)
            boxes.append((rank, *lo, width))
            nodes = (lo + width*support)/4
            x, y = nodes.T
            u = (a*x*y + gamma*y, -a*(x*x+y*y)/2)
            for comp in range(3):
                on_cell = fields[(fields["cell"] == cell) & (fields["component"] == comp)]
                values = np.sort(on_cell, order="local")["value"]
                if comp < 2:
                    velocity_error = max(velocity_error, max(abs(u[comp]-values)))
                else:
                    pressure.extend(zip(y[:4], values))
            # Published native Maxwell stress is the endpoint tensor 2*sigma_m.
            normal = 4*G*B*h*a*q["y"]
            stress_error = max(stress_error, max(abs(q["tauxx"]-normal)),
                               max(abs(q["tauyy"]+normal)), max(abs(q["tauzz"])),
                               max(abs(q["tauxy"]-2*shear)))
            phase_error = max(phase_error, max(abs(q["phi"]-.2)))
    np.testing.assert_allclose(areas, .25, atol=5e-14, rtol=0)
    # Every real coarse/fine segment has width 1 in these quarter-metre units.
    hanging = partition = 0
    for i, (r, x, y, w) in enumerate(boxes):
        for r2, x2, y2, w2 in boxes[:i]:
            ox = min(x+w, x2+w2) - max(x,x2)
            oy = min(y+w, y2+w2) - max(y,y2)
            if w != w2 and min(ox,oy) == 0 and max(ox,oy) > 0:
                hanging += 1
                partition += r != r2
    if not hanging or (ranks == 2 and not partition):
        raise ValueError(f"Missing hanging-interface coverage: pieces={hanging}, partition={partition}")
    y, p = np.asarray(pressure).T
    if min(abs(p)) != 0.:
        raise ValueError("The midpoint gauge did not pin a physical Q1 coefficient")
    pressure_error = np.ptp(p + 2*G*B*h*a*y)
    sections = read_table(output / "sections-0001.csv")
    np.testing.assert_array_equal(sections["formed"], np.ones(8))
    np.testing.assert_allclose(sections["ds"], .25, rtol=0, atol=1e-14)
    rate_error = max(abs(sections["V"]-speed))
    # Roundoff in sparse factorization and fixed 1e-12 residual tolerances;
    # the manufactured trial and test polynomials are integrated exactly.
    for name, error, tolerance in (("velocity [m/s]", velocity_error, 2e-10),
                                   ("pressure [Pa]", pressure_error, 1e-6),
                                   ("stress [Pa]", stress_error, 1e-6),
                                   ("rate [m/s]", rate_error, 2e-10),
                                   ("phase", phase_error, 3e-14),
                                   ("wall work [J/m]", work_error, 1e-11),
                                   ("work adjoint [J/m]", adjoint_error, 1e-13)):
        print(f"{name}: error {error:.9g} (allowance {tolerance:g})")
        if error > tolerance:
            raise ValueError(name)
    print(f"cells {len(boxes)}, hanging subfaces {hanging}, partition subfaces {partition}, "
          f"analytic Wext {expected_work:.13g} J/m, signed V {speed:.13g} m/s")


def check_corruption(output, ranks):
    # Replace the fine Q2 node at (1/8,1/2) by linear interpolation between
    # the two coarse endpoints at x=0 and x=1/2. At this interface the
    # quadratic y-velocity is curved along x, including at the midpoint time.
    with TemporaryDirectory() as temporary:
        scratch = Path(temporary)
        for entry in output.glob("*.csv"):
            (scratch / entry.name).symlink_to(entry.resolve())
        for rank in range(ranks):
            points = read_table(output / f"points-0001-r{rank}.csv")
            match = ((points["x"] < .25) & (points["y"] > .25) & (points["y"] < .5))
            if not match.any():
                continue
            cell, = np.unique(points["cell"][match])
            path = scratch / f"fields-0001-r{rank}.csv"
            with path.open() as stream:
                reader = csv.DictReader(stream)
                columns = reader.fieldnames
                records = list(reader)
            row, = [r for r in records if r["cell"] == cell and r["component"] == "1" and r["local"] == "7"]
            a = .003*(1+40*.0025)
            wrong = -a*(.5**2 + .25*.5**2)/2
            change = wrong-float(row["value"])
            row["value"] = str(wrong)
            path.unlink()
            with path.open("w") as stream:
                writer = csv.DictWriter(stream, columns, lineterminator="\n")
                writer.writeheader()
                writer.writerows(records)
            try:
                check(scratch, ranks)
            except ValueError as error:
                if str(error) != "velocity [m/s]":
                    raise
                print(f"Rejected linear interpolation at ({cell},uy,7): change {change:.9g} m/s")
                return
            raise ValueError("Linear hanging interpolation passed the quadratic patch")
        raise ValueError("Missing fine test node")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ranks", choices=(1,2), required=True, type=int)
    parser.add_argument("--check-corruption", action="store_true", help="Replace one fine velocity by a linear hanging trace")
    args = parser.parse_args()
    check(args.output, args.ranks)
    if args.check_corruption:
        check_corruption(args.output, args.ranks)
