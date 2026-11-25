"""Read the accepted mature shear in the installed homogeneous example at 12.5 ms."""
import argparse
import math
import os
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aspect", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    binary = args.aspect.resolve()
    fixture = Path(__file__).with_name("fixed_fault.prm").resolve()
    dest = args.destination.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, DEAL_II_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for solver in ("block AMG", "block GMG", "direct solver"):
        case = dest / solver.replace(" ", "-")
        case.mkdir()
        prm = case / "input.prm"
        prm.write_text(f"""include {fixture}
set End time = 0.0125
set Output directory = {case / 'output'}
subsection Solver parameters
  subsection Stokes solver parameters
    set Stokes solver type = {solver}
  end
end
subsection Postprocess
  set List of postprocessors = visualization
  subsection Visualization
    set Output format = gnuplot
    set List of output variables = named additional outputs, stress
  end
end
""")
        with (case / "log").open("w") as log:
            result = subprocess.run([binary, prm], cwd=case, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=120)
        message = " ".join((case / "log").read_text().split())
        if solver != "block AMG":
            if not result.returncode or "ASPECT elasticity disabled, block AMG" not in message:
                raise ValueError(f"{solver}: expected mature solver diagnostic, see {case}/log")
            print(f"{solver}: diagnosed the mature block AMG requirement")
            continue
        result.check_returncode()
        if "Mature fault Stokes solver: ASPECT block AMG." not in message:
            raise ValueError("Missing mature solver startup diagnostic")
        lines = (case / "output/solution/solution-00001.0000.gnuplot").read_text().splitlines()
        names = next(re.findall(r"<([^>]+)>", line) for line in lines if "<fault_tau_xy>" in line)
        samples = [list(map(float, line.split())) for line in lines if line.strip() and not line.startswith("#")]
        if not samples:
            raise ValueError("Missing graphical stress samples")
        for row in samples:
            for component in ("xx", "yy", "xy"):
                reported = row[names.index("fault_tau_" + component)]
                if not math.isfinite(reported) or reported != row[names.index("fault_stress_" + component)]:
                    raise ValueError(f"{component} differs from its accepted DG history")
            physical = row[names.index("fault_tau_xy")]
            viscous = row[names.index("stress") + 1]  # xx, xy, yx, yy
            if abs(physical - .0325) > 1e-10 or not math.isclose(viscous, -.0121783, abs_tol=1e-7):
                raise ValueError(f"Unexpected shear on the loaded wall: {physical}, {viscous}")
        print(f"{len(samples)} samples at 12.5 ms: accepted xy {physical:g} Pa, viscous visualization {viscous:g} Pa")


if __name__ == "__main__":
    main()
