"""Run through installed ASPECT with a separately built resistance-law plugin."""
import argparse
import csv
from math import isclose
import os
from pathlib import Path
import re
import subprocess


def material(subsection, entries):
    return f"\nsubsection Material model\n  subsection {subsection}\n{entries}\n  end\nend\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("aspect", "law_plugin", "destination"):
        parser.add_argument(name, type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    growing = Path(__file__).resolve().parents[1]
    public, old = "Phase field fault", "Growing fault"
    base, count = re.subn(r"(?m)^\s*set Viscosity\s*=.*$", "", (growing/"onecut.prm").read_text())
    if count != 1:
        raise ValueError("Expected one historical growing viscosity")
    pressure = (growing/"adiabatic_pressure.prm").read_text()
    base = base + "\n" + pressure.replace("include onecut.prm", "")
    public_name = "\nsubsection Material model\n set Model name = phase field fault\nend\n"

    def run(name, source, plugin=False, error=()):
        case = args.destination/name
        case.mkdir()
        extension = f"\nset Additional shared libraries = {args.law_plugin}\n" if plugin else ""
        (case/"input.prm").write_text(source + extension + "\nset Output directory = output\n")
        with (case/"log").open("w") as stream:
            p = subprocess.run([args.aspect, "input.prm"], cwd=case, stdout=stream,
                               stderr=subprocess.STDOUT, timeout=120,
                               env=os.environ | dict(OMP_NUM_THREADS="1", DEAL_II_NUM_THREADS="1",
                                                     OPENBLAS_NUM_THREADS="1"))
        if error:
            message = " ".join((case/"log").read_text().split())
            if not p.returncode or any(key not in message for key in error):
                raise ValueError(f"Expected {error}, see {case}/log")
            print(name, "rejected:", *error)
        else:
            p.check_returncode()
        return case/"output"

    def check_stress(name, settings, eta, reference, modulus=1e6):
        output = run(name, base+settings)
        with (output/"points-0001-r0.csv").open() as stream:
            stress, = [float(r["tauxy"]) for r in csv.DictReader(stream) if (r["cell"], r["q"]) == ("0_0:", "0")]
        with (output/"balance-0001.csv").open() as stream:
            clock = float(next(csv.DictReader(stream))["time"])
        reported = re.findall(r"Fault Maxwell material \(growing strips\): shear modulus = (\S+) Pa, viscosity = (\S+) Pa s",
                              (output.parent/"log").read_text())
        if len(reported) != 1 or tuple(map(float, reported[0])) != (modulus, eta):
            raise ValueError(f"{name}: incorrect startup report {reported}")
        if abs(clock-.005) > 1e-15 or not isclose(stress, reference, abs_tol=1e-6, rel_tol=1e-7):
            raise ValueError(f"{name}: shear stress {stress} Pa at {clock} s")
        print(f"{name}: G={modulus:g} Pa, eta={eta:g} Pa s, 5 ms tau_xy={stress:.15g} Pa")
        return (output/"points-0001-r0.csv").read_bytes()

    # Before the common material name, the growing parser ignored Simple viscosity.
    irrelevant = material("Simple model", "set Viscosity = 1e6")
    for eta, reference, explicit in ((1e8, 75.978862635792, ""),
                                     (1e6, 75.791283897615, material(old, "set Viscosity = 1e6"))):
        unchanged = check_stress(f"growth-{eta:g}", explicit, eta, reference)
        other = irrelevant if not explicit else material("Simple model", "set Viscosity = 5e24")
        if check_stress(f"growth-{eta:g}-simple", explicit+other, eta, reference) != unchanged:
            raise ValueError("Simple viscosity changed the historical growth solution")
    # Drop the remaining shared entries so the reported pair tests actual
    # route defaults. The legacy geometry and boundary loading stay the same.
    for key in ("Shear modulus", "Initial age", "Reference friction coefficient", "Direct effect",
                "Evolution effect", "Reference speed", "Slip distance", "Radiation damping"):
        base, count = re.subn(r"(?m)^[ \t]*set[ \t]+" + re.escape(key) + r"[ \t]*=.*$", "", base)
        if count != 1:
            raise ValueError(f"Expected one historical {key}")
    check_stress("public-omitted", public_name, 1e8, 75.978862635792)
    check_stress("public-simple", public_name+irrelevant, 1e6, 75.791283897615)

    check_stress("public-modulus", public_name+material(public, "set Shear modulus = 2e6"),
                 1e8, 151.954052677598, modulus=2e6)

    # Loading the law alone exercises a declaration that sizes its list from Dimension.
    affine = (growing.parent/"phase_field_surface_energy/affine.prm").read_text()
    output = run("fault-free-affine", affine, plugin=True)
    statistics = (output/"statistics").read_text().splitlines()
    row = next(line for line in statistics if line.strip() and not line.startswith("#")).split()
    col = next(int(m[1])-1 for line in statistics
               if (m := re.match(r"# (\d+): Crack surface energy total \(J/m\)", line)))
    if not isclose(float(row[col]), 3. + 15./128., rel_tol=1e-12):
        raise ValueError(f"Fault-free affine AT1 energy is {row[col]} J/m")
    print("Fault-free affine AT1 with dimension-dependent law declaration:", row[col], "J/m")

    base += public_name + material(public, "set Resistance law = weighted dashpot")
    key = "Dashpot impedance (Pa.s/m)"
    def weights(values):
        return f"subsection Nested law\n set Weights = {values}\nend"

    def law_case(name, settings, impedance):
        output = run(name, base+settings, plugin=True)
        with (output/"sections-0001.csv").open() as stream:
            table = list(csv.DictReader(stream))
        if len(table) != 9 or "age" in table[0]:
            raise ValueError("Expected nine stateless strips")
        for r in table:
            v = float(r["V"])
            if v == 0. or not isclose(float(r["force"])/v, impedance, rel_tol=1e-12):
                raise ValueError(f"Expected {impedance} Pa s/m, strip {r}")
        sample = next(r for r in table if int(r["section"]) == 8)
        print(f"{name}: strip 8 V={sample['V']} m/s, f={sample['force']} Pa, f/V={impedance:g} Pa s/m")
        return (output/"sections-0001.csv").read_bytes()

    a = law_case("literal-public", material(public, f"set {key} = 750"), 750)
    if law_case("literal-legacy", material(old, f"set {key} = 750"), 750) != a:
        raise ValueError("The spelling changed the accepted strip response")
    weighted = material(public, weights("1, 2"))
    a = law_case("typed-single", weighted, 1500)
    if law_case("typed-equal", weighted+material(old, weights("1.0, 2.0")), 1500) != a:
        raise ValueError("Equal numeric weights changed the accepted strip response")
    run("typed-conflict", base+weighted+material(old, weights("2, 2")), plugin=True,
        error=("Phase field fault / Nested law / Weights", "Growing fault / Nested law / Weights"))


if __name__ == "__main__":
    main()
