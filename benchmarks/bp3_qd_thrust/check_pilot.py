"""Check solved BP3 pilot fields, FE wall velocities and accepted RSF history."""
import argparse
import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET

from history import read_statistics, slip_budget


def require(condition, message):
    if not condition:
        raise ValueError(message)


def labels(path):
    with path.open() as stream:
        require(stream.readline().strip() == "# label x(m) y(m) age(s) slip(m) rate(m/s)", "Unknown label columns")
        rows = [tuple(map(float, line.split())) for line in stream]
    require(rows and all(len(row) == 6 and row[0] == i and all(map(math.isfinite, row))
                         for i, row in enumerate(rows)), f"Bad material IDs or data in {path}")
    return rows


def check_initial(rows):
    residual = 0.
    rate_error = 0.
    # Use the measured station depth and age in the forward law, independent
    # of the inverse-age construction in make_vertices.py.
    height = 40000 * math.sin(math.pi / 3)
    for _, x, y, age, slip, rate in rows:
        s = (height-y) / math.sin(math.pi/3)
        require(abs(x - 0.5*s) < 1e-8 and age > 0 and slip == 0, f"Initial geometry/history at {s} m")
        a = 0.01 + 0.015 * min(1., max(0., (s-15000.) / 3000.))
        f = 0.6 + 0.015 * math.log(1e-6 * age / 0.008)
        shear = 50e6 * a * math.asinh(0.0005 * math.exp(f/a)) + 4624440 * 1e-9
        residual = max(residual, abs(shear-26546122.365133367))
        rate_error = max(rate_error, abs(rate/1e-9-1))
    require(residual < 1e-6, f"SEAS initial age implies incorrect shear by {residual} Pa")
    # Bound the solved startup bias across the full log-age transition.
    require(rate_error < 1e-5, f"Initial solved rate differs from Vinit by {100*rate_error:.5g}%")
    print(f"Initial traction error {residual:.4g} Pa at Vinit, maximum solved speed error {100*rate_error:.6g}%")


def check_pilot(directory):
    frames = list(read_statistics(directory / "statistics"))
    previous = None
    history = []
    parameters = json.loads((directory / "parameters.json").read_text())
    budgets = parameters["Time stepping"]["Fault RSF"]
    age_cap = float(budgets["Maximum logarithmic state change"]["value"])
    clocks = [float(frame["BP3 clock (s)"]) for frame in frames]
    require(len(clocks) >= 2 and clocks[0] == 0 and clocks[-1] == float(parameters["End time"]["value"]),
            "Incomplete pilot interval")
    max_slip = max_log_age = 0.
    for step, (time, frame) in enumerate(zip(clocks, frames)):
        require(int(frame["Time step number"]) == step and math.isfinite(time), "Pilot clock")
        rows = labels(directory / f"fault-labels-{step:05d}.txt")
        require(len(rows) == 179 and rows[0][1] == 0 and rows[-1][1] == 20000, "Incomplete down-dip profile")
        history.append(rows)
        if previous is None:
            check_initial(rows)
        else:
            dt = time-previous_time
            require(dt > 0 and math.isclose(float(frame["BP3 interval (s)"]), dt, rel_tol=3e-13), "Pilot duration")
            for old, new in zip(previous, rows):
                require(old[:3] == new[:3] and new[5] > 0., "Moving geometry or invalid rate")
                rate = new[5]
                age = old[3] + (0.008/rate-old[3])*(-math.expm1(-dt*rate/0.008))
                require(math.isclose(new[3], age, rel_tol=5e-15), f"Accepted age at ID {new[0]}")
                require(math.isclose(new[4], old[4]+dt*rate, rel_tol=5e-15), f"Accepted slip at ID {new[0]}")
                increment = dt*abs(rate)/0.008
                require(increment <= slip_budget(parameters, 2*new[1])*(1+1e-12),
                        f"Accepted slip exceeds the local budget at ID {new[0]}")
                max_slip = max(max_slip, increment)
                max_log_age = max(max_log_age, abs(math.log(new[3]/old[3])))
        fastest = max(rows, key=lambda row: row[5])
        require(float(frame["BP3 peak speed (m/s)"]) == fastest[5] and
                abs(float(frame["BP3 peak distance (m)"])-2*fastest[1]) < 1e-7, "Incorrect peak record")
        for row in rows:
            s = round(2*row[1])
            if s % 2500 == 0:
                for value, column in zip(row[3:], ["age (s)", "slip (m)", "rate (m/s)"]):
                    require(value == float(frame[f"BP3 {s} m {column}"]), f"Station output at {s} m")
        previous, previous_time = rows, time
    require(age_cap == 0 or max_log_age <= age_cap*(1+1e-12), "Accepted logarithmic age exceeds its budget")
    print(f"Maximum accepted slip/Dc {max_slip:.10g}, absolute log-age change {max_log_age:.10g}")
    material = parameters["Material model"]["Phase field fault"]
    for key, expected in (("Mature slip sense", "opposite tangent"), ("Fault geometry", "boundary spanning"),
                          ("Resistance law", "BP3 depth dependent RSF"), ("Fit trace to phase", "false")):
        require(material[key]["value"] == expected, f"Incorrect configured {key}")
    expected = {(t, x, y) for t in clocks for x,y in
                ((-30000,16000), (50000,16000), (19750,0), (20000,0), (20250,0))}
    wall_error = 0.
    with (directory / "point_values.txt").open() as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            t,x,y,vx,vy,_,_,_,_,_,phase = map(float, line.split())
            # Point values uses a shorter text precision than BP3 statistics.
            matches = [key for key in expected if key[1:] == (x,y) and math.isclose(key[0], t, rel_tol=5e-6)]
            require(len(matches) == 1, f"Missing or ambiguous point sample {(t,x,y)}")
            expected.remove(matches[0])
            r = math.sqrt(0.75)*(x-20000) + 0.5*y
            v = -0.5e-9*math.tanh(r/250.)
            require(all(map(math.isfinite, (vx,vy,phase))), "Nonfinite FE sample")
            wall_error = max(wall_error, abs(vx-0.5*v), abs(vy+math.sqrt(0.75)*v))
            h = 6/math.cosh(r/250.)**2
            exact_phase = 2*h/(2*h+1+math.sqrt(1+8*h))
            require(abs(phase-exact_phase) < 1e-6, "Phase profile at a plate sample")
    require(not expected and wall_error < 6e-16, f"Wall velocities differ by {wall_error} m/s")
    print(f"179 labels, {len(clocks)} accepted clocks, 17 reported stations, 5 FE wall probes; maximum velocity error {wall_error:.4g} m/s")
    return clocks, history


def check_native(directory, clocks, history, read_fields):
    parameters = json.loads((directory/"parameters.json").read_text())["Postprocess"]
    models = {s.strip() for s in parameters["List of postprocessors"]["value"].split(",")}
    if "visualization" not in models:
        require(not (directory/"solution.pvd").exists(), "Unexpected native frames")
        return
    settings = parameters["Visualization"]
    cadence = float(settings["Time between graphical output"]["value"])
    step_limit = int(settings["Time steps between graphical output"]["value"])
    threshold = float(settings["Fault slip interval"]["value"])
    expected = []
    previous = 0
    next_time = 0.
    for step, (time,rows) in enumerate(zip(clocks,history)):
        displacement = max(abs(row[4]-old[4]) for row,old in zip(rows,history[previous]))
        if step==0 or time >= next_time or step-previous >= step_limit or ("fault slip output" in models and displacement>=threshold):
            expected.append(time)
            previous = step
            next_time = (math.floor(time/cadence*(1+2*math.ulp(1.)))+1)*cadence if cadence>0 else time
    frames = ET.parse(directory/"solution.pvd").findall(".//DataSet")
    measured = [float(frame.attrib["timestep"]) for frame in frames]
    require(len(frames)==len(expected) and all(math.isclose(t,u,rel_tol=5e-6) for t,u in zip(expected,measured)),
            f"Expected graphical clocks {expected}, got {measured}")
    for frame in frames:
        path = directory/frame.attrib["file"]
        descriptor = ET.parse(path)
        names = {entry.attrib["Name"] for entry in descriptor.findall(".//PPointData/PDataArray")}
        require({"velocity","p","phase_field","fault_tau_xx","fault_tau_yy","fault_tau_xy"}<=names,
                f"Missing physical field in {path}")
        pieces = descriptor.findall(".//Piece")
        require(pieces and all((path.parent/p.attrib["Source"]).stat().st_size>0 for p in pieces), "Missing VTU pieces")
        if read_fields:
            from native_frames import read
            _, _, size = read(path)
            print(f'{frame.attrib["timestep"]} s native frame: {size} bytes')
    print("Verified native frame clocks",measured)


def compare_labels(reference, output, steps):
    errors = [0., 0., 0.]
    for step in steps:
        actual = labels(output / f"fault-labels-{step:05d}.txt")
        expected = reference[step]
        require(len(actual) == len(expected), "Different label counts")
        for a, b in zip(actual, expected):
            require(a[:3] == b[:3], "Different label position or identity")
            for j in range(3):
                require(math.isclose(a[3+j], b[3+j], rel_tol=1e-8), f"Different state at label {a[0]}")
                errors[j] = max(errors[j], abs(a[3+j]-b[3+j]))
    print(f"{output}: maximum age, slip and rate differences {errors[0]:.4g} s, {errors[1]:.4g} m, {errors[2]:.4g} m/s")


def compare_production(clean, resumed, *, exact=True):
    def physical_rows(path):
        return [{k: float(v) for k,v in row.items() if k.startswith("BP3 ")}
                for row in read_statistics(path / "statistics")]
    a, b = physical_rows(clean), physical_rows(resumed)
    times = [row["BP3 clock (s)"] for row in a]
    require(len(times) >= 2 and times[0] == 0 and all(t > s for s,t in zip(times,times[1:])), "Production clocks")
    for run in (clean, resumed):
        params = json.loads((run/"parameters.json").read_text())
        criteria = params["Termination criteria"]
        if criteria["Termination criteria"]["value"] == "end step":
            require(len(times)-1 == int(criteria["End step"]["value"]), "Incomplete step-limited trajectory")
    require(len(a) == len(b) and all(len(row) == 55 for row in a), "Incomplete accepted station history")
    for left, right in zip(a,b):
        require(left.keys() == right.keys(), "Different station identities")
        for key in left:
            same = left[key] == right[key] if exact else math.isclose(left[key],right[key],rel_tol=1e-8)
            require(same, f"Different {key} at {left['BP3 clock (s)']} s")
    print(f"Matched 55 accepted statistics at {len(times)} clocks through {times[-1]} s")


def plot(clocks, history, path):
    import matplotlib.pyplot as plt
    fig, (age, speed) = plt.subplots(2, 1, sharex=True, figsize=(7.2, 5.4), layout="constrained")
    x = [row[1]*0.002 for row in history[0]]
    age.semilogy(x, [row[3] for row in history[0]], color="#225577")
    age.set_ylabel("Initial state age (s)")
    age.set_title("60° thrust pilot")
    for frame, color, time in zip((history[0], history[-1]), ("#225577", "#b34b20"), (clocks[0], clocks[-1])):
        speed.plot(x, [row[5]*1e9 for row in frame], color=color, label=f"{time} s")
    for axis in (age, speed):
        axis.axvspan(15, 18, color="gray", alpha=.14)
        axis.grid(axis="y", alpha=.2)
    speed.set(xlabel="Distance down dip (km)", ylabel="Solved slip speed (nm/s)", xlim=(0,40))
    speed.legend()
    fig.savefig(path, dpi=170)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--mpi-output", type=Path)
    parser.add_argument("--restart-output", type=Path, help="Pilot resumed from an accepted checkpoint")
    parser.add_argument("--production-pair", type=Path, nargs=2, metavar=("CLEAN", "RESUMED"))
    parser.add_argument("--read-vtu", action="store_true", help="Read each piece and verify finite physical arrays (Python VTK and NumPy)")
    args = parser.parse_args()
    clocks, history = check_pilot(args.output)
    check_native(args.output, clocks, history, args.read_vtu)
    if args.mpi_output:
        require(check_pilot(args.mpi_output)[0] == clocks, "MPI clocks")
        compare_labels(history, args.mpi_output, range(len(clocks)))
    if args.restart_output:
        compare_production(args.output, args.restart_output, exact=False)
        emitted = [step for step in range(len(clocks)) if (args.restart_output/f"fault-labels-{step:05d}.txt").exists()]
        require(emitted and emitted == list(range(emitted[0],len(clocks))), "Missing resumed label endpoints")
        compare_labels(history, args.restart_output, emitted)
    if args.production_pair:
        compare_production(*args.production_pair)
    if args.figure:
        plot(clocks, history, args.figure)
