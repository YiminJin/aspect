"""Measure the signed, normalized work of both production transfers at initialization."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess

BASE_RATE = 0.005  # m/s
EPSILON = 0.001  # centered middle-rate perturbation, m/s
KAPPA = -10 * math.expm1(-10 * 0.1 / 10)  # Pa s


def read_frame(directory, ranks):
    owners = []
    rows = []
    normal_integrals = None
    for rank in range(ranks):
        cells = {}
        with (directory / f"stokes-r{rank}.csv").open() as file:
            for cell, work, stiffness in csv.reader(file):
                if cell in cells or not math.isfinite(float(work)):
                    raise ValueError(f"Duplicate cell or invalid work in {directory}: {cell}")
                if not math.isclose(float(stiffness), KAPPA, rel_tol=2e-14):
                    raise ValueError(f"Unexpected Stokes coefficient {stiffness}")
                cells[cell] = float(work)
        owners.append(cells)
        with (directory / f"section-r{rank}.txt").open() as file:
            row, count, *integrals = map(float, file.read().split())
        if not math.isfinite(row) or count <= 0:
            raise ValueError(f"Missing active section action on rank {rank}")
        if len(integrals) != 3 or not all(math.isfinite(v) and v > 0 for v in integrals):
            raise ValueError("Invalid Q1 normal integral")
        if normal_integrals is not None and normal_integrals != integrals:
            raise ValueError("Replicated normal integrals disagree")
        normal_integrals = integrals
        rows.append(row)
    return owners, rows, normal_integrals


def run_pair(aspect, plugin, directory, extra_prm="", ranks=1):
    """Run opposite middle-rate perturbations with the same physical parent."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True)
    frames = []
    for sign, name in ((-1, "minus"), (1, "plus")):
        dest = directory / name
        dest.mkdir()
        prm = dest / "case.prm"
        prm.write_text(f"""include $ASPECT_SOURCE_DIR/benchmarks/fixed_fault/stationary.prm
set Additional shared libraries = {Path(plugin).resolve()}
set Output directory = {dest / "output"}
set End time = 0
subsection Mesh refinement
  set Initial global refinement = 0
end
subsection Material model
  set Model name = fault transfer work probe
  subsection Phase field fault
    set Initial slip rate = {BASE_RATE}
    set Background shear traction = 0
    set Fit trace to phase = false
    set Profile half width = 0
  end
  subsection Transfer work probe
    set Middle rate increment = {sign * EPSILON}
  end
end
subsection Boundary traction model
  subsection Function
    set Function expression = 0;0
  end
end
subsection Postprocess
  set List of postprocessors =
end
{extra_prm}
""")
        cmd = [str(Path(aspect).resolve()), str(prm)]
        if ranks > 1:
            cmd = [os.environ.get("MPIEXEC", "mpirun"), "-np", str(ranks)] + cmd
        env = dict(os.environ, DEAL_II_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        with (dest / "log").open("w") as log:
            subprocess.run(cmd, cwd=dest, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        frames.append(read_frame(dest / "output", ranks))

    minus, plus = frames
    if minus[2] != plus[2]:
        raise ValueError("The rate perturbation changed the normal integrals")
    by_rank, stokes_cells = [], {}
    for rank in range(ranks):
        lo, hi = minus[0][rank], plus[0][rank]
        if lo.keys() != hi.keys() or stokes_cells.keys() & lo.keys():
            raise ValueError("Inconsistent cell ownership")
        # W(v) is in W/m at unit engineering shear rate (1/s); after dividing
        # by kappa, shear rate, and the rate perturbation, both actions are in m.
        delta = {key: (hi[key]-lo[key])/(2*EPSILON*KAPPA) for key in lo}
        stokes_cells.update(delta)
        # The middle velocity row in N/m is already paired with a unit
        # middle perturbation. Divide by the same kappa and shear rate.
        section = minus[1][rank]/KAPPA
        if not math.isclose(section, plus[1][rank]/KAPPA, abs_tol=2e-13):
            raise ValueError("Uncancelled rate or friction dependence in reverse action")
        by_rank.append((math.fsum(delta.values()), section))
    result = dict(stokes=math.fsum(s for s, _ in by_rank),
                  section=math.fsum(f for _, f in by_rank), by_rank=by_rank,
                  normal_integrals=plus[2], stokes_cells=stokes_cells)
    (directory / "work.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("aspect", "plugin", "directory"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--baseline", action="store_true",
                        help="Check the two distinct actions of the uncorrected operator")
    args = parser.parse_args()
    result = run_pair(args.aspect, args.plugin, args.directory)
    print(f"Stokes {result['stokes']:.15g} m, section {result['section']:.15g} m, split reference 0.5 m")
    if args.baseline:
        expected = (0.569668517088, 0.494235968293)
        if any(abs(result[k]-v) > 1e-12 for k, v in zip(("stokes", "section"), expected)):
            raise ValueError("Failed to reproduce the predicted pre-repair counterexample")
    elif abs(result["stokes"]-result["section"]) > 2e-13:
        raise ValueError("Stokes and section work differ on the same mesh")


if __name__ == "__main__":
    main()
