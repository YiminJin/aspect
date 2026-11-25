"""Change one physical input of the nine-strip fault and read its accepted response."""
import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from check_history import check as check_history, fracture_energy
from check_repairs import subsection
from compare import continuation, rows


def growing(settings):
    return subsection("Material model", subsection("Growing fault", settings))


def penalty(temperature, phase=None):
    settings = f"set Discontinuous penalty = {temperature}"
    if phase is not None:
        settings += f"\nset Phase discontinuous penalty = {phase}"
    return subsection("Discretization", subsection("Stabilization parameters", settings))


def compression_controls(case, legacy, canonical, pressure_plugin):
    base = case("200", canonical)
    continuation(base, case("legacy", legacy), [1])
    continuation(base, case("both", canonical + "\nset Surface pressure = 2e2\n"), [1])
    high_setting = growing("set Prescribed fault compression = 250")
    high = case("250", canonical + high_setting)
    continuation(high, case("legacy-250", legacy + "\nset Surface pressure = 250\n"), [1])
    shifted = case("pressure-datum", canonical, extra_library=pressure_plugin)

    fields = [rows(out.glob("fields-0001-r*.csv"), ("cell", "component", "local")) for out in (base, shifted)]
    if fields[0].keys() != fields[1].keys():
        raise ValueError("Pressure datum changed the FE identities")
    gauge = [float(fields[1][k]["value"]) - float(row["value"]) for k, row in fields[0].items() if k[1] == 2]
    if len(gauge) != 32 * 4:
        raise ValueError("Expected all four Q1 pressure coefficients on each cell")
    np.testing.assert_allclose(gauge, 100., atol=1e-6, rtol=0.)
    print(f"Pressure shift across {len(gauge)} cell-local coefficients: {min(gauge):.12g} .. {max(gauge):.12g} Pa")

    speeds = []
    for out, compression in ((base, 200.), (high, 250.), (shifted, 200.)):
        strips = rows([out / "sections-0001.csv"], ("section",))
        for row in strips.values():
            v = float(row["V"])
            # Exact half-step aging at h=2.5 ms from the prescribed 10 s parent.
            x = .0025 * abs(v) / .01
            age = 10 * math.exp(-x) + .0025 * (-math.expm1(-x)/x if x else 1.)
            strength = .6 + .02 * math.log(.1 * age)
            resistance = compression * .015 * math.asinh(v/.002 * math.exp(strength/.015)) + 1000 * v
            np.testing.assert_allclose(resistance, float(row["force"]), rtol=0., atol=1e-10)
        first = strips[0,]
        speeds.append(float(first["V"]))
        print(f"{out.parent.name}: strip 0 V = {first['V']} m/s, resistance = {first['force']} Pa")
    np.testing.assert_allclose(speeds[2], speeds[0], rtol=1e-10, atol=0.)
    if not 0 < speeds[1] < speeds[0] / 20:
        raise ValueError("The higher compression did not reduce cold creep")

    # With damping 1.1 the monotonicity bound is 1.0 at 200 Pa and 1.25
    # at 250 Pa. Only the second 5 ms attempt must cut back.
    cut = (growing("set Radiation damping = 1.1") + "\nset Nonlinear solver failure strategy = cut timestep size\n"
           + subsection("Termination criteria", "set Termination criteria = end step\nset End step = 1"))
    for pressure, duration, cuts in ((200, .005, 0), (250, .0025, 1)):
        out = case(f"duration-{pressure}", canonical + growing(f"set Prescribed fault compression = {pressure}") + cut)
        clock = float(rows([out / "balance-0001.csv"], ("step",))[1,]["time"])
        if clock != duration or (out.parent / "log").read_text().count("Trying to cut the current time step") != cuts:
            raise ValueError(f"Wrong admissible duration at {pressure} Pa")
        print(f"{pressure} Pa: {cuts} cutbacks, first accepted interval {clock} s")

    for value in ("0", "250"):
        case("conflict-" + value, canonical + f"\nset Surface pressure = {value}\n",
             diagnostic="Conflicting Prescribed fault compression and Surface pressure")
    case("zero", canonical + growing("set Prescribed fault compression = 0"),
         diagnostic="positive prescribed fault compression")


def functional_controls(case, legacy, canonical):
    old = case("old", legacy)
    new = case("new", canonical)
    reported = case("reported", canonical, reporter=True)
    continuation(old, new, [1])
    continuation(new, reported, [1])
    if "Crack surface energy total (J/m)" not in (reported / "statistics").read_text():
        raise ValueError("Missing active surface-energy reporter")
    renamed = case("named-phase", canonical
                   + subsection("Compositional fields", "set Names of fields = tau_xx, tau_yy, tau_xy, damage")
                   + subsection("Phase field", "set Phase composition = damage"), reporter=True)
    continuation(reported, renamed, [1])

    tough = case("toughness", canonical + subsection("Phase field", "set Fracture energy = 40"))
    energies = [float(rows([out / "balance-0001.csv"], ("step",))[1,]["fracture"]) for out in (new, tough)]
    print(f"Fracture ledger at 5 ms, Gc=26.6666666667 and 40 J/m²: {energies[0]:.13g}, {energies[1]:.13g} J/m")
    if abs(energies[1] - energies[0]) < 1.:
        raise ValueError("Fracture energy ignored toughness")
    points = [(out / "points-0001-r0.csv").read_bytes() for out in (old, new, reported, tough)]
    if not points[0] == points[1] == points[2] or points[1] == points[3]:
        raise ValueError("Unexpected response to reporter, alias or toughness change")

    # Each explicit legacy default must conflict with a different canonical value.
    for name, value in [("Fracture energy", "1"), ("Length scale", "1"), ("Phase composition", "phase_field"),
                        ("Crack geometry", "AT1")]:
        other = {"Phase composition": "other", "Crack geometry": "AT2"}.get(name)
        changed = subsection("Phase field", f"set {name} = {other}") if other else ""
        legacy_value = subsection("Postprocess", subsection("Phase field surface energy", f"set {name} = {value}"))
        case("conflict-" + name.replace(" ", "-"), canonical + changed + legacy_value, diagnostic=f"Conflicting {name}")
    for law in ("AT2", "CZM"):
        case("coupled-" + law, canonical + subsection("Phase field", "set Crack geometry = " + law),
             diagnostic="requires the configured AT1 surface energy")


def face_controls(case, canonical):
    base = case("inherited-10", canonical)
    high = case("phase-20", canonical + penalty(10, 20))
    continuation(high, case("inherited-20", canonical + penalty(20)), [1])
    continuation(base, case("phase-10", canonical + penalty(20, 10)), [1])
    for out, gamma in ((base, 10), (high, 20)):
        fields = rows(out.glob("fields-0001-r*.csv"), ("cell", "component", "local"))
        points = rows(out.glob("points-0001-r*.csv"), ("cell", "q"))
        patches = {}
        for (cell, q), row in points.items():
            if q == 0:
                index = tuple(int(float(row[d])/.25) for d in ("x", "y"))
                patches[index] = np.array([float(fields[cell, 7, i]["value"]) for i in range(9)]).reshape(3, 3)
        if set(patches) != {(i, j) for i in range(8) for j in range(4)}:
            raise ValueError("Unexpected nine-strip mesh")
        total, face = fracture_energy(patches, .25, gamma)
        ledger = float(rows([out / "balance-0001.csv"], ("step",))[1,]["fracture"])
        np.testing.assert_allclose(total, ledger, rtol=0., atol=1e-12)
        wrong, _ = fracture_energy(patches, .25, 30 - gamma)
        if abs(wrong - ledger) < .1:
            raise ValueError("Substituting the other face penalty passed the energy check")
        print(f"Penalty {gamma}: Gauss(4) faces {face:.13g}, total {total:.13g}, defect {total-ledger:.3g} J/m")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("aspect", "plugin", "destination"):
        parser.add_argument(key, type=lambda p: Path(p).resolve())
    parser.add_argument("group", choices=["compression", "functional", "faces", "history", "backend", "restart"])
    parser.add_argument("--pressure-plugin", type=lambda p: Path(p).resolve())
    parser.add_argument("--old-output", type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)

    legacy = Path(__file__).with_name("onecut.prm").read_text()
    # Move the four material settings together and retain the frozen seed and loading.
    header = "subsection Postprocess\n set List of postprocessors =\n subsection Phase field surface energy\n"
    if legacy.count(header) != 1 or legacy.count("set Surface pressure = 200\n") != 1:
        raise ValueError("Missing legacy functional or compression setting in the control")
    common, energy = legacy.split(header)
    energy, tail = energy.split(" end\nend\n", 1)
    canonical = (common.replace("set Surface pressure = 200\n", "") + subsection("Phase field", energy)
                 + tail + growing("set Prescribed fault compression = 200"))

    def case(name, settings, *, reporter=False, diagnostic=None, extra_library=None, restart=False, end=.005):
        run = args.destination / name
        run.mkdir()
        output = run / "output"
        if restart:
            shutil.copytree(args.old_output / "restart/02", output / "restart/02")
            settings += ("\nset Resume computation = true\n"
                         + subsection("Checkpointing", "set Resume checkpoint = 2\nset Number of checkpoints to keep = 4"))
        libraries = str(args.plugin) + (f", {extra_library}" if extra_library else "")
        observers = "growing fault history" + (", phase field surface energy" if reporter else "")
        prm = run / "input.prm"
        prm.write_text(settings + f"\nset End time = {end}\nset Output directory = {output}\n"
                       + f"set Additional shared libraries = {libraries}\n"
                       + subsection("Postprocess", "set List of postprocessors = " + observers))
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", DEAL_II_NUM_THREADS="1")
        with (run / "log").open("w") as log:
            proc = subprocess.run([args.aspect, prm], cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        if diagnostic:
            message = " ".join((run / "log").read_text().split())
            if not proc.returncode or diagnostic not in message or list(output.glob("balance*.csv")):
                raise ValueError(f"{name}: expected {diagnostic}, see {run}/log")
            print(f"{name}: diagnosed {diagnostic}")
        else:
            proc.check_returncode()
        return output

    if args.group == "compression":
        if not args.pressure_plugin:
            parser.error("compression needs --pressure-plugin")
        compression_controls(case, legacy, canonical, args.pressure_plugin)
    elif args.group == "functional":
        functional_controls(case, legacy, canonical)
    elif args.group == "faces":
        face_controls(case, canonical)
    elif args.group == "history":
        out = case("export", canonical, end=.015)
        check_history(out)
        if list(out.glob("operator-*")):
            raise ValueError("History export invoked the specimen operator test")
    elif args.group == "backend":
        def solver(name):
            return subsection("Solver parameters", subsection("Stokes solver parameters", f"set Stokes solver type = {name}"))
        amg = case("amg", canonical)
        gmg = case("gmg", canonical + solver("block GMG"))
        for out in (amg, gmg):
            if "UMFPACK direct factorization (separate velocity and pressure blocks)" not in (out.parent / "log").read_text():
                raise ValueError("Missing effective midpoint backend")
        continuation(amg, gmg, [1])
        case("direct", canonical + solver("direct solver"), diagnostic="requires separate velocity and pressure blocks")
    elif args.group == "restart":
        if not args.old_output:
            parser.error("restart needs --old-output from the Stage A writer")
        resumed = case("same", canonical + penalty(20, 10), restart=True, end=.015)
        continuation(args.old_output, resumed, [2, 3])
        for name, extra in [("compression", growing("set Prescribed fault compression = 250")),
                            ("toughness", subsection("Phase field", "set Fracture energy = 40")),
                            ("penalty", penalty(10, 20))]:
            case(name, canonical + extra, restart=True, end=.015, diagnostic="physical parameters differ from the checkpoint")


if __name__ == "__main__":
    main()
