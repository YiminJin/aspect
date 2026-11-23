"""Run the installed phase-only diagnostic against Cartesian integrals in SI units."""
import argparse
from math import isclose, pi, sqrt
import os
from pathlib import Path
import re
import subprocess
import sys


FIXTURES = Path(__file__).resolve().parent
AFFINE = "include affine.prm\n"
TERMS = ("local", "gradient", "total")
# Analytic pairs (integral alpha dx, c0). The gradient integral is 5/8 m^0.
AFFINE_LAWS = {"AT1": (1, 8/3), "AT2": (7/12, 2), "CZM": (17/12, pi)}


def parameters(*entries):
    return ("subsection Postprocess\n  subsection Phase field surface energy\n"
            + "".join(f"    set {entry}\n" for entry in entries) + "  end\nend\n")


def statistics(path):
    lines = path.read_text().splitlines()
    columns = {}
    rows = []
    for line in lines:
        match = re.fullmatch(r"# (\d+): (.*)", line)
        if match:
            columns[match[2]] = int(match[1]) - 1
        elif line.strip() and not line.startswith("#"):
            rows.append(line.split())
    if len(rows) != 1:
        raise ValueError(f"{path}: expected only the initial state")
    return {name: float(rows[0][index]) for name, index in columns.items()}


def run(command, destination, name, settings, diagnostic=None):
    case = destination / name
    case.mkdir()
    prm = case / "input.prm"
    prm.write_text(settings + f"\nset Output directory = {case / 'output'}\n")
    environment = dict(os.environ, OMP_NUM_THREADS="1", DEAL_II_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1")
    log = case / "run.log"
    with log.open("w") as stream:
        result = subprocess.run([*command, prm], cwd=FIXTURES, env=environment,
                                stdout=stream, stderr=subprocess.STDOUT, timeout=120)
    if diagnostic:
        message = " ".join(log.read_text().split())
        if not result.returncode or diagnostic not in message:
            raise ValueError(f"{name}: expected a failure containing {diagnostic!r}, see {log}")
        print(f"{name}: diagnosed {diagnostic}", flush=True)
        return message
    result.check_returncode()
    return statistics(case / "output" / "statistics")


def check_energy(name, stats, expected, ranks=1, dim=2):
    cells = stats["Number of mesh cells"]
    smallest, largest = (stats[f"{word} cells per process"] for word in ("Minimal", "Maximal"))
    if smallest < 1 or cells != 4**dim or stats["Average cells per process"] * ranks != cells:
        raise ValueError(f"{name}: unexpected mesh partition {smallest}, {largest}, {cells}")
    # Q3 has 3^dim points per cell. Allow one epsilon per accumulated term,
    # 32 for FE evaluation/scaling/reduction, and rounding of the printed column.
    tolerance = (cells * 3**dim + 32) * sys.float_info.epsilon + 0.5e-16
    unit = "J/m" if dim == 2 else "J"
    measured = [stats[f"Crack surface energy {term} ({unit})"] for term in TERMS]
    for term, value, target in zip(TERMS, measured, expected):
        if not isclose(value, target, rel_tol=tolerance):
            raise ValueError(f"{name} {term}: {value:.17g}, expected {target:.17g}")
    print(f"{name}: " + ", ".join(f"{value:.16g}" for value in measured)
          + f" {unit}; owned cells min/max {smallest:g}/{largest:g}", flush=True)


def phase_input(expression):
    return AFFINE + f"""subsection Initial composition model
  subsection Function
    set Function expression = {expression}
  end
end
"""


def check_invalid_phase(command, destination, name, expression, bounds, ranks):
    message = run(command, destination, name, phase_input(expression),
                  diagnostic=f"Invalid sampled phase on 1 of {ranks} MPI ranks")
    match = re.search(r"Finite sampled minimum = ([^,]+), maximum = (\S+?)\. ", message)
    if not match or not all(isclose(float(x), y, rel_tol=1e-13) for x, y in zip(match.groups(), bounds)):
        raise ValueError(f"{name}: expected global quadrature extrema {bounds}")
    table = destination / name / "output" / "statistics"
    if table.exists() and "Crack surface energy" in table.read_text():
        raise ValueError(f"{name}: invalid energy was published")
    print(f"{name}: sampled extrema {match[1]}, {match[2]}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aspect", type=Path)
    parser.add_argument("destination", type=Path, help="fresh output directory")
    parser.add_argument("--mpi", metavar="MPIEXEC",
                        help="run the affine selections on two ranks with this launcher")
    args = parser.parse_args()
    destination = args.destination.resolve()
    destination.mkdir(parents=True)
    command = [args.aspect.resolve()]
    ranks = 2 if args.mpi else 1
    if args.mpi:
        command = [args.mpi, "-np", "2", *command]

    for law, (area_integral, c0) in AFFINE_LAWS.items():
        settings = AFFINE + parameters(f"Crack geometry = {law}")
        stats = run(command, destination, law, settings)
        local = 2 * area_integral / (c0 * 0.25)
        gradient = 2 * 0.25 * (5/8) / c0
        check_energy(law, stats, (local, gradient, local + gradient), ranks)
    if args.mpi:
        # Only the upper-right cell exceeds 1. The global minimum lies
        # in the opposite corner. The nearest Q3 point is this fraction
        # of the entire side from the boundary on the 4x4 mesh.
        inset = (1 - sqrt(3/5)) / 8
        check_invalid_phase(command, destination, "one-rank-violation", "1.1*x*y/2",
                            (1.1 * inset**2, 1.1 * (1-inset)**2), ranks)
        return

    check_invalid_phase(command, destination, "negative-phase", "-0.01", (-0.01, -0.01), ranks)
    for name, phi in (("unity", 1.), ("upper-roundoff", 1. + 4*sys.float_info.epsilon),
                      ("lower-roundoff", -4*sys.float_info.epsilon)):
        stats = run(command, destination, name, phase_input(repr(phi)))
        # The AT1 local term on this rectangle is 6 phi. A constant field's
        # mapped gradient contains only interpolation roundoff.
        for term, target in zip(TERMS, (6*phi, 0., 6*phi)):
            if not isclose(stats[f"Crack surface energy {term} (J/m)"], target,
                           rel_tol=4e-14, abs_tol=1e-26):
                raise ValueError(f"{name}: wrong {term} energy for constant phase {phi}")
        print(f"{name}: accepted phase {phi:.17g}", flush=True)

    # The straight crack also exercises the default law and the finite peak at 1.
    stats = run(command, destination, "straight", "include straight_crack.prm\n")
    check_energy("straight", stats, (1, 1, 2))

    # Extrude the affine field by 3 m to check the volume measure and J output.
    settings = AFFINE + """set Dimension = 3
subsection Geometry model
  subsection Box
    set Z extent = 3
  end
end
"""
    stats = run(command, destination, "volume", settings)
    check_energy("volume", stats, (9, 45/128, 1197/128), dim=3)

    # Select the second, continuous field with a decoy in the first component.
    settings = AFFINE + """subsection Compositional fields
  set Number of fields = 2
  set Names of fields = decoy, damage
end
subsection Discretization
  set Use discontinuous composition discretization = false
end
subsection Initial composition model
  subsection Function
    set Function expression = 0.9;x/4+y/2
  end
end
""" + parameters("Phase composition = damage")
    stats = run(command, destination, "named-field", settings)
    check_energy("named-field", stats, (3, 15/128, 399/128))

    # A zero-composition model must initialize without looking up phase_field.
    settings = """set Dimension = 2
set End time = 0
set Nonlinear solver scheme = no Advection, no Stokes
subsection Geometry model
  set Model name = box
end
subsection Material model
  set Model name = simple
end
subsection Gravity model
  set Model name = vertical
  subsection Vertical
    set Magnitude = 0
  end
end
subsection Initial temperature model
  set List of model names = function
end
subsection Mesh refinement
  set Initial global refinement = 1
  set Initial adaptive refinement = 0
end
subsection Postprocess
  set List of postprocessors = load balance statistics
end
"""
    stats = run(command, destination, "disabled", settings)
    if any("Crack surface energy" in name for name in stats) or "Minimal cells per process" not in stats:
        raise ValueError("Unexpected statistics with the diagnostic disabled")
    print("disabled: ordinary zero-composition model passed", flush=True)

    for name, setting, message in [
        ("unknown-law", "Crack geometry = imaginary",
         "imaginary for the entry named Crack geometry does not match"),
        ("missing-field", "Phase composition = absent", "compositional field absent"),
        ("invalid-toughness", "Fracture energy = 0", "positive fracture energy"),
        ("invalid-length", "Length scale = -1", "positive length scale"),
    ]:
        run(command, destination, name, AFFINE + parameters(setting), diagnostic=message)


if __name__ == "__main__":
    main()
