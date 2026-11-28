"""Verify the named mature phase, initial vertex speed and resistance checkpoint signature."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "growing_fault"))
from parameter_file import legacy_mature_input


def section(name, entries):
    return f"\nsubsection {name}\n{entries}\nend\n"


def material(entries):
    return section("Material model", section("Phase field fault", entries))


def compare_accepted(left, right, steps):
    for step in steps:
        label = f"fault-labels-{step:05d}.txt"
        if (left / label).read_bytes() != (right / label).read_bytes():
            raise ValueError(f"Changed accepted material labels in {right / label}")
        vtu = f"solution/solution-{step:05d}.0000.vtu"
        trees = [ET.parse(run / vtu) for run in (left, right)]
        for name in ("fault_tau_xx", "fault_tau_xy", "fault_tau_yy"):
            a, b = (t.find(f".//PointData/DataArray[@Name='{name}']") for t in trees)
            if a is None or b is None or not a.text or (a.attrib, a.text) != (b.attrib, b.text):
                raise ValueError(f"Changed accepted {name} in {right / vtu}")


def energies(output):
    header = (output / "statistics").read_text()
    indices = [int(re.search(rf"^# (\d+): Crack surface energy {term} \(J/m\)$", header, re.M)[1])-1
               for term in ("local", "gradient", "total")]
    return np.loadtxt(output / "statistics", usecols=indices, ndmin=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("aspect", "fixed_probe", "dashpot_plugin", "destination"):
        parser.add_argument(arg, type=lambda p: Path(p).resolve())
    parser.add_argument("--legacy-writer", type=lambda p: Path(p).resolve(),
                        help="Pre-repair ASPECT executable used to write an RSF label archive without a signature")
    args = parser.parse_args()
    root = args.destination
    root.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    base = (here / "fixed_fault.prm").read_text()
    output_list = "mature fault statistics, visualization"
    environment = os.environ | dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", DEAL_II_NUM_THREADS="1")

    def run(name, body, *, error=None, seed=None, aspect=args.aspect, plugins=()):
        case = root / name
        case.mkdir()
        output = case / "output"
        if plugins:
            body += "\nset Additional shared libraries = " + ", ".join(map(str, plugins)) + "\n"
        if seed:
            shutil.copytree(seed / "restart/02", output / "restart/02")
            body += "\nset Resume computation = true\n" + section("Checkpointing", "set Resume checkpoint = 2")
        (case / "input.prm").write_text(body + "\nset Output directory = output\n")
        with (case / "log").open("w") as stream:
            proc = subprocess.run([aspect, "input.prm"], cwd=case, env=environment, stdout=stream,
                                  stderr=subprocess.STDOUT, timeout=120)
        if error:
            if not proc.returncode or error not in " ".join((case / "log").read_text().split()):
                raise ValueError(f"{name}: expected {error!r} (see {case / 'log'})")
            print(f"{name}: {error}", flush=True)
        else:
            proc.check_returncode()
        return output

    def phase_input(field, reporter, role):
        return (base + section("Compositional fields",
                               f"set Names of fields = fault_stress_xx, fault_stress_yy, fault_stress_xy, {field}\n"
                               f"set Types of fields = stress, stress, stress, {role}")
                + section("Phase field", f"set Phase composition = {field}")
                + section("Initial composition model", section("Function", "set Function expression = 0;0;0;.35+.1*x"))
                + material("set Fault geometry = boundary spanning\nset Fit trace to phase = false")
                + section("Postprocess", "set List of postprocessors = " + output_list +
                          (", phase field surface energy" if reporter else "")))

    original = run("phase_field", phase_input("phase_field", True, "generic"))
    for name, reporter, role in (("damage", True, "generic"), ("no-reporter", False, "generic"),
                                 ("static-phase-role", True, "phase field")):
        result = run(name, phase_input("damage", reporter, role))
        compare_accepted(original, result, range(3))
        if reporter:
            # Unit square, AT1, Gc=1 J/m², ell=1 m, phi=.35+.1*x.
            observed = energies(result)
            np.testing.assert_array_equal(observed, energies(original))
            np.testing.assert_allclose(observed, np.tile([.15, .00375, .15375], (3, 1)), rtol=0, atol=3e-15)
            print(name, "local, gradient, total at 25 ms (J/m)", observed[-1], flush=True)
        print(name, "matched all three accepted label and physical stress frames", flush=True)
    run("stress-slot", phase_input("damage", False, "generic") +
        section("Phase field", "set Phase composition = fault_stress_yy"),
        error="Phase composition cannot be a mature fault stress component: fault_stress_yy")

    affine = (here.parent / "phase_field_surface_energy/affine.prm").read_text()
    independent = run("fault-off", affine +
                      section("Compositional fields", "set Names of fields = damage") +
                      section("Phase field", "set Phase composition = damage\nset Crack geometry = AT2"))
    np.testing.assert_allclose(energies(independent)[0], [7/3, 5/32, 239/96], rtol=0, atol=3e-15)
    print("fault-free AT2 energies (J/m)", energies(independent)[0], flush=True)

    # The ordinary initial Stokes solve already changes V at t=0. The probe reads the
    # material vertices just before that solve so the test measures the true input.
    probe = section("Postprocess", "set List of postprocessors = mature fault statistics, fixed fault probe\n" +
                    section("Fixed fault probe", "set Compare homogeneous reference = false"))

    def initial(name, input_text, expected, dashpot=False):
        plugins = (args.fixed_probe, args.dashpot_plugin) if dashpot else (args.fixed_probe,)
        out = run(name, input_text + probe, plugins=plugins)
        np.testing.assert_array_equal(np.loadtxt(out / "initial-vertex-rates"), [expected, expected])
        rate = np.loadtxt(out / "fault-labels-00002.txt")[0, -1]
        print(f"{name}: incoming {expected:.12g} m/s, accepted rate at 25 ms {rate:.12g} m/s", flush=True)
        return rate

    initial("rsf-compatibility", base, .001)
    rsf = initial("rsf-explicit", base + material("set Initial slip rate = .01"), .01)
    changed = initial("rsf-other-reference", base + material("set Initial slip rate = .01\nset Reference speed = .002"), .01)
    if abs(rsf - changed) < 1e-3:
        raise ValueError("Changing the constitutive RSF reference speed did not change the solved rate.")
    # Remove the explicitly specified RSF values from the scientist's input.
    rsf_names = ("Reference speed", "Initial age", "Direct effect", "Evolution effect", "Slip distance",
                 "Reference friction coefficient", "Radiation damping")
    stateless = "\n".join(line for line in base.splitlines()
                          if not any(line.strip().startswith(f"set {n} =") for n in rsf_names)) + "\n"
    stateless += material("set Resistance law = linear dashpot")
    for name, expected, settings in (("dashpot-compatibility", .001, "set Reference speed = .001"),
                                     ("dashpot-default", 1e-6, ""),
                                     ("dashpot-explicit", .01, "set Initial slip rate = .01"),
                                     ("inactive-zero", .01, "set Initial slip rate = .01\nset Reference speed = 0")):
        value = initial(name, stateless + material(settings), expected, dashpot=True)
        np.testing.assert_allclose(value, (3.85-.045)/20, rtol=0, atol=2e-12)

    # Generation 02 carries the first physical endpoint at 12.5 ms.
    def interior(input_text):
        return (input_text + section("Compositional fields", "set Types of fields = stress, stress, stress, phase field")
                + material("set Fault geometry = fixed interior\nset Fault vertices = .5,.05; .5,.95\nset Fit trace to phase = false")
                + section("Checkpointing", "set Steps between checkpoint = 1\nset Number of checkpoints to keep = 3")
                + section("Postprocess", "set List of postprocessors = " + output_list))
    for name, model, edit, plugins in (("rsf", base, "set Slip distance = .009", ()),
                                       ("dashpot", stateless, "set Dashpot impedance = 21", (args.dashpot_plugin,))):
        deck = interior(model)
        clean = run(f"{name}-clean", deck, plugins=plugins)
        continued = run(f"{name}-resume", deck, plugins=plugins, seed=clean)
        compare_accepted(clean, continued, [2])
        print(f"{name}: generation 02 reproduced 25 ms labels and all physical stress arrays", flush=True)
        run(f"{name}-changed", deck + material(edit), plugins=plugins, seed=clean,
            error="Mature fault resistance parameters differ from the checkpoint")
        if name == "rsf":
            generic = run("fixed-interior-generic", deck + section("Compositional fields",
                          "set Types of fields = stress, stress, stress, generic"))
            compare_accepted(clean, generic, range(3))
            run("changed-law", deck + material("set Resistance law = linear dashpot"),
                plugins=(args.dashpot_plugin,), seed=clean, error="Mature fault resistance law differs from the checkpoint")
        if name == "rsf" and args.legacy_writer:
            if args.aspect.samefile(args.legacy_writer):
                raise ValueError("Supply a separate pre-repair checkpoint writer")
            old = run("legacy-writer", interior(legacy_mature_input(base)), aspect=args.legacy_writer)
            compare_accepted(old, clean, range(3))
            resumed = run("legacy-resume", deck, seed=old)
            compare_accepted(clean, resumed, [2])
            # Distinguish the actual legacy archive from a second copy of the
            # current writer. That format has no saved values to compare.
            legacy_changed = run("legacy-unverifiable-parameters", deck + material(edit), seed=old)
            rate = np.loadtxt(legacy_changed / "fault-labels-00002.txt")[0,-1]
            clean_rate = np.loadtxt(clean / "fault-labels-00002.txt")[0,-1]
            if abs(rate-clean_rate) < 1e-3:
                raise ValueError("The legacy reader control needs a measurable parameter change")
            print("Legacy RSF archive: original settings continued identically; changed unrecorded "
                  f"slip distance gave {rate:.12g} versus {clean_rate:.12g} m/s", flush=True)


if __name__ == "__main__":
    main()
