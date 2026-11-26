"""Check the Gaussian initial composition with Simple material and an additive field.

From the worktree, with I the installation prefix and E a fresh run directory:

    python3 -BO refactoring/phase-field-fault/results/mature-release/check_phase_field_roles.py "$I" "$E/field-roles"
"""
import argparse
import math
import os
from pathlib import Path
import runpy
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prefix", "destination"):
        parser.add_argument(name, type=lambda s: Path(s).resolve())
    args = parser.parse_args()
    example = args.prefix / "benchmarks/phase_field_surface_energy"
    read_statistics = runpy.run_path(str(example / "check.py"))["statistics"]
    args.destination.mkdir(parents=True, exist_ok=False)
    table = args.destination / "vertices.txt"
    table.write_text("id x_m y_m\n0 1 0\n1 1 1\n")
    env = dict(os.environ, ASPECT_SOURCE_DIR=str(args.prefix), OMP_NUM_THREADS="1",
               DEAL_II_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for phase_name in ("damage", "missing_phase"):
        case = args.destination / phase_name
        case.mkdir()
        prm = case / "case.prm"
        prm.write_text(f"""include {example / 'affine.prm'}
set Output directory = output
subsection Phase field
  set Phase composition = {phase_name}
end
subsection Compositional fields
  set Number of fields = 2
  set Names of fields = damage, tracer
end
subsection Material model
  subsection Phase field fault
    set Fault vertex file = {table}
  end
end
subsection Initial composition model
  set List of model names = function, fault polyline Gaussian
  set List of model operators = add
  subsection Function
    set Function expression = 0; x/8+y/4
  end
end
subsection Postprocess
  set List of postprocessors = composition statistics
end
""")
        with (case / "run.log").open("w") as log:
            result = subprocess.run([args.prefix / "bin/aspect-release", prm], cwd=case,
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
        if phase_name == "missing_phase":
            message = "The compositional field missing_phase you asked for is not used in the simulation"
            if result.returncode == 0 or message not in " ".join((case / "run.log").read_text().split()):
                raise ValueError(f"Expected the missing phase diagnostic in {case}")
            print("Absent configured phase rejected at startup.")
            continue

        result.check_returncode()
        columns = read_statistics(case / "output/statistics")
        # A 2 m by 1 m box split into 4 x 4 cells. The Gaussian depends only on
        # x. Simpson weights integrate its actual nodal DGQ2 interpolant exactly.
        phase = [0.03 + 0.32 * math.exp(-((i / 4 - 1) / 0.15)**2) for i in range(9)]
        phase_mass = sum(phase[i] + 4*phase[i+1] + phase[i+2] for i in range(0, 8, 2)) / 12
        for field, expected in (("damage", (min(phase), max(phase), phase_mass)),
                                ("tracer", (0., 0.5, 0.5))):
            measured = [columns[f"{stat} for composition {field}"]
                        for stat in ("Minimal value", "Maximal value", "Global mass")]
            # ASPECT writes these statistics with eight digits after the decimal in scientific notation.
            for value, reference in zip(measured, expected):
                if not math.isclose(value, reference, rel_tol=5e-9, abs_tol=1e-14):
                    raise ValueError(f"{field}: {measured} differs from {expected}")
            print(f"{field}: min {measured[0]:.9g}, max {measured[1]:.9g}, integral {measured[2]:.9g} m^2")


if __name__ == "__main__":
    main()
