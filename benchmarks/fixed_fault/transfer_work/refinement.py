"""Measure mature shear-transfer work on a resolved dipping phase ridge."""

import argparse
import csv
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss


# In the unit box the ridge centre is x=1/4+y/2. Its signed distance is
# r=(-2(x-1/4)+y)/sqrt(5). The phase is an exact DGQ2 polynomial, so
# interpolation cannot account for the variation in the measured actions.
KNOT = 0.47
LENGTH = np.sqrt(1.25)
VERTICES = np.array([0.0, KNOT, 1.0])
GRIDS = (8, 16, 32)


def inclined_input(cells):
    return f"""
subsection Mesh refinement
  set Initial global refinement = {cells.bit_length() - 1}
end
subsection Material model
  subsection Phase field fault
    set Fault vertices = 0.25,0; 0.485,0.47; 0.75,1
  end
end
subsection Initial composition model
  set List of model names = function
  subsection Function
    set Function expression = 0;0;0;0.8-0.34*(-2*(x-0.25)+y)^2
  end
end
"""


@lru_cache(maxsize=None)
def rule(order):
    x, w = leggauss(order)
    return (1 + x) / 2, w / 2


def normal_integral(y, order):
    """Integrate actual degradation along the in-box physical normal in m."""
    x = 0.25 + 0.5 * y
    normal_x, normal_y = -2 / np.sqrt(5), 1 / np.sqrt(5)
    lo = np.maximum((1 - x) / normal_x, -y / normal_y)
    hi = np.minimum(-x / normal_x, (1 - y) / normal_y)
    q, w = rule(order)
    distance = lo[:, None] + (hi - lo)[:, None] * q
    phi = 0.8 - 1.7 * distance**2
    excess = phi * (1 + phi) / (1 - phi)**2
    return (hi - lo) * (excess @ w)


def reference(order):
    # Independently reconstruct the normalization specified by the experiment:
    # Q3 along each label interval, full-ray integrals and a consistent (exact
    # here) trace mass. A uniform degradation magnitude cancels from the work.
    loads = np.zeros(3)
    mass = np.zeros((3, 3))
    q, w = rule(3)
    basis = np.array([1 - q, q])
    for j, h in enumerate(np.diff(VERTICES)):
        ray = normal_integral(VERTICES[j] + h * q, order)
        loads[j:j+2] += LENGTH * h * (basis @ (w * ray))
        mass[j:j+2, j:j+2] += (LENGTH * h / 6) * np.array([[2, 1], [1, 2]])
    denominator = np.linalg.solve(mass, loads)

    # Integrate N_middle * excess/I_h over the physical box in the
    # orthonormal fault frame. Only feet between the two contacts participate.
    # Split at both changes of clipped ray face and at the interior label.
    # The tangent projection of each box corner gives every face-change event.
    projected = [(x - 0.25) * 0.4 + y * 0.8 for x in (0., 1.) for y in (0., 1.)]
    edges = sorted({*VERTICES, *(u for u in projected if 0 < u < 1)})
    q, w = rule(order)
    work = 0.
    for a, b in zip(edges, edges[1:]):
        y = a + (b - a) * q
        hat = np.interp(y, VERTICES, [0., 1., 0.])
        normalizer = np.interp(y, VERTICES, denominator)
        work += LENGTH * (b - a) * np.dot(w, hat * normal_integral(y, order) / normalizer)
    return work, denominator


def measure(aspect, plugin, root, mode):
    from run import run_pair

    work, denominator = reference(48)
    independent, _ = reference(32)
    if abs(independent - work) > 1.e-13:
        raise ValueError("Independent split integration did not converge")
    root.mkdir(parents=True, exist_ok=False)
    records = []
    finest_serial = {}
    for cells, ranks in [(n, 1) for n in GRIDS] + [(GRIDS[-1], 2)]:
        # The middle vertex is strictly inside one bulk cell at every resolution.
        if any(abs(cells * x - round(cells * x)) < 1.e-10 for x in (0.485, KNOT)):
            raise ValueError("Interior label coincides with a cell edge")
        result = run_pair(aspect, plugin, root / f"n{cells}-p{ranks}", inclined_input(cells), ranks)
        if len(result['by_rank']) != ranks or np.min(result['by_rank']) <= 0:
            raise ValueError("Every requested MPI owner must contribute to both actions")
        if cells == GRIDS[-1]:
            cells_work = result['stokes_cells']
            if len(cells_work) != cells**2:
                raise ValueError("Incomplete set of owned Stokes cells")
            if ranks == 1:
                finest_serial = cells_work
            else:
                if cells_work.keys() != finest_serial.keys():
                    raise ValueError("Serial and distributed CellIds differ")
                largest_change = max(abs(cells_work[k] - finest_serial[k]) for k in cells_work)
                if largest_change > 5.e-13:
                    raise ValueError(f"Serial/parallel Stokes cell actions differ by {largest_change} m")
                print(f"{len(cells_work)} matched CellIds, max local action change {largest_change:.3g} m")
        # The fixture fixes the degradation magnitude at one. The prescribed
        # phase and geometry also fix the three normal integrals on every grid.
        np.testing.assert_allclose(result['normal_integrals'], denominator, atol=5.e-12, rtol=5.e-13)
        stokes, section = result['stokes'], result['section']
        if mode == 'repaired':
            np.testing.assert_allclose(stokes, section, atol=5.e-13, rtol=0.)
        print(f"{cells:2} x {cells:<2}, {ranks} rank(s)  RHS {stokes:.15g}  row {section:.15g}  "
              f"reference {work:.15g}  active work by owner {result['by_rank']}")
        records.append(dict(cells=cells, ranks=ranks, stokes_m=stokes, section_m=section,
                            split_integral_m=work, reference_difference_m=abs(independent - work)))
    for key in ('stokes_m', 'section_m'):
        np.testing.assert_allclose(records[-2][key], records[-1][key], atol=5.e-13, rtol=0.)
        errors = [abs(r[key] - work) for r in records[:-1]]
        if errors[-1] > 1.e-6 or not all(a > b for a, b in zip(errors, errors[1:])):
            raise ValueError(f"{key} defect did not decrease under refinement: {errors}")
    with (root / 'work.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, records[0], lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


def plot(old_csv, new_csv, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.2), layout="constrained")
    for path, label, key, style in ((old_csv, "Original Stokes RHS", 'stokes_m', 'o--'),
                                     (new_csv, "Repaired Stokes RHS", 'stokes_m', 's-'),
                                     (new_csv, "Section row", 'section_m', '+:')):
        with path.open() as stream:
            rows = [row for row in csv.DictReader(stream) if row['ranks'] == '1']
        n = [int(row['cells']) for row in rows]
        errors = [abs(float(row[key]) - float(row['split_integral_m'])) for row in rows]
        ax.loglog(n, errors, style, label=label, linewidth=1.5, markersize=8)
    ax.set_xticks(GRIDS, labels=[str(n) for n in GRIDS], minor=False)
    ax.tick_params(axis='x', which='minor', bottom=False, labelbottom=False)
    ax.set(xlabel="Bulk cells per side of the 1 m box", ylabel="Absolute normalized work error (m)")
    ax.legend()
    ax.grid(axis='y', which='major', alpha=.25)
    fig.savefig(output, dpi=180)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help="Run all three grids and repeat the finest on two ranks")
    for name in ('aspect', 'plugin', 'root'):
        run.add_argument(name, type=Path)
    run.add_argument('mode', choices=('original', 'repaired'))
    figure = commands.add_parser('plot', help="Plot the two measured CSVs without running ASPECT")
    for name in ('old_csv', 'new_csv', 'output'):
        figure.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.command == 'run':
        measure(args.aspect, args.plugin, args.root, args.mode)
    else:
        plot(args.old_csv, args.new_csv, args.output)
