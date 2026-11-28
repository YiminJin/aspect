"""Exercise crack-law admission and finite seed coverage through an installed ASPECT."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

from check_interior_seed import check as check_seed


def check_support(executable, cases, benchmarks):
    # Reuse the installed affine fixture, its integral oracle and launch helper.
    sys.path.insert(0, str(benchmarks / "phase_field_surface_energy"))
    from check import check_energy, run

    phase_model = "subsection Phase field\n  set Crack geometry = {}\nend\n"
    affine = "include affine.prm\n" + phase_model.format("AT2")
    stats = run([executable], cases, "energy-AT2", affine)
    check_energy("AT2 energy", stats, (7/3, 5/32, 239/96))
    marker = "subsection Mesh refinement\n  set Strategy = growing fault\nend\n"
    run([executable], cases, "marker-AT2", affine + marker,
        diagnostic="A finite reference profile is required, but AT2 has infinite support.")

    for law in ("AT2", "CZM"):
        name = f"midpoint-{law}"
        settings = f"include {benchmarks / 'growing_fault/growing_seed.prm'}\n" + phase_model.format(law)
        run([executable], cases, name, settings,
            diagnostic=f"requires the configured AT1 surface energy. Selected crack geometry: {law}.")
    for name in ("marker-AT2", "midpoint-AT2", "midpoint-CZM"):
        output = cases / name / "output"
        if any(list(output.glob(pattern)) for pattern in ("sections-*.csv", "points-*.csv", "balance-*.csv")):
            raise ValueError(f"{name} published native history after an inadmissible selection")


def write_seed_input(case, benchmarks, overrides):
    geometry = benchmarks / "granite_fault_growth/amr"
    case.mkdir()
    shutil.copy(benchmarks / "growing_fault/growing_seed.prm", case)
    shutil.copy(geometry / "interior_seed.prm", case)
    prm = case / "startup.prm"
    prm.write_text("include interior_seed.prm\nset Output directory = output\n" + overrides)
    return prm


def check_missing_anchor(executable, cases, benchmarks):
    case = cases / "no-anchor"
    # Keep all three damaged tents and the tangent. Translate just the
    # mechanical seed line and initialize the actual FE phase on the roots.
    prm = write_seed_input(case, benchmarks, """subsection Material model
  subsection Phase field fault
    set Forward seed tip = 4,4.9
  end
end
subsection Mesh refinement
  set Skip setup initial conditions on initial refinement = false
end
""")
    driver = benchmarks / "granite_fault_growth/run.py"
    result = subprocess.run([sys.executable, driver, executable, prm, "timing", "--ranks", "1"])
    log = case / "timing/solver.log"
    diagnostic = "the incoming seed has no resolved anchor"
    if not result.returncode or not log.is_file() or diagnostic not in " ".join(log.read_text().split()):
        raise ValueError(f"Expected the missing-anchor diagnostic in {log}")
    if any((case / "output").glob("balance-*.csv")):
        raise ValueError("The misplaced seed reached physical publication")
    print(f"no-anchor: diagnosed {diagnostic}", flush=True)


def check_resolution(executable, cases, benchmarks):
    driver = benchmarks / "granite_fault_growth/run.py"
    for name, target in (("inherited", .25), ("finer", .125)):
        case = cases / name
        settings = ""
        if name == "finer":
            settings += ("subsection Mesh refinement\n  subsection Growing fault\n"
                         f"    set Maximum cell edge = {target}\n  end\nend\n")
        prm = write_seed_input(case, benchmarks, settings)
        subprocess.run([sys.executable, driver, executable, prm, "timing",
                        "--ranks", "2"], check=True)
        check_seed(case / "output", target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("destination", type=Path, help="Fresh run directory")
    args = parser.parse_args()
    executable = args.executable.resolve(strict=True)
    cases = args.destination.resolve()
    cases.mkdir(parents=True)
    benchmarks = executable.parents[1] / "benchmarks"
    check_support(executable, cases, benchmarks)
    check_missing_anchor(executable, cases, benchmarks)
    check_resolution(executable, cases, benchmarks)


if __name__ == "__main__":
    main()
