"""Check solved BP3 pilot fields, FE wall velocities and accepted RSF history."""
import argparse
import json
import math
from pathlib import Path

from history import read_statistics


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
    # Solved Q1 rates and linearly interpolated ages need not give a uniform
    # discrete equilibrium in the a-gradient. Bound the largest startup bias.
    require(rate_error < 0.01, f"Initial solved rate differs from Vinit by {100*rate_error:.5g}%")
    print(f"Initial traction error {residual:.4g} Pa at Vinit, maximum solved speed error {100*rate_error:.6g}%")


def check_pilot(directory):
    frames = list(read_statistics(directory / "statistics"))
    previous = None
    history = []
    for step, time in enumerate((0., 400., 900.)):
        frame = frames[step]
        require(int(frame["Time step number"]) == step and float(frame["BP3 clock (s)"]) == time, "Pilot clock")
        rows = labels(directory / f"fault-labels-{step:05d}.txt")
        require(len(rows) == 179 and rows[0][1] == 0 and rows[-1][1] == 20000, "Incomplete down-dip profile")
        history.append(rows)
        if previous is None:
            check_initial(rows)
        else:
            dt = time-previous_time
            require(float(frame["BP3 interval (s)"]) == dt, "Pilot duration")
            for old, new in zip(previous, rows):
                require(old[:3] == new[:3] and new[5] > 0., "Moving geometry or invalid rate")
                rate = new[5]
                age = old[3] + (0.008/rate-old[3])*(-math.expm1(-dt*rate/0.008))
                require(math.isclose(new[3], age, rel_tol=5e-15), f"Accepted age at ID {new[0]}")
                require(math.isclose(new[4], old[4]+dt*rate, rel_tol=5e-15), f"Accepted slip at ID {new[0]}")
                require(dt <= min(0.0008/old[5], 0.1*old[3])*(1+1e-12), "Parent timestep bound exceeded")
        fastest = max(rows, key=lambda row: row[5])
        require(float(frame["BP3 peak speed (m/s)"]) == fastest[5] and
                abs(float(frame["BP3 peak distance (m)"])-2*fastest[1]) < 1e-7, "Incorrect peak record")
        for row in rows:
            s = round(2*row[1])
            if s % 2500 == 0:
                for value, column in zip(row[3:], ["age (s)", "slip (m)", "rate (m/s)"]):
                    require(value == float(frame[f"BP3 {s} m {column}"]), f"Station output at {s} m")
        previous, previous_time = rows, time
    require(len(frames) == 3, "Unexpected pilot frame count")
    parameters = json.loads((directory / "parameters.json").read_text())
    material = parameters["Material model"]["Phase field fault"]
    for key, expected in (("Mature slip sense", "opposite tangent"), ("Fault geometry", "boundary spanning"),
                          ("Resistance law", "BP3 depth dependent RSF"), ("Fit trace to phase", "false")):
        require(material[key]["value"] == expected, f"Incorrect configured {key}")
    expected = {(t, x, y) for t in (0., 400., 900.) for x,y in
                ((-30000,16000), (50000,16000), (19750,0), (20000,0), (20250,0))}
    wall_error = 0.
    with (directory / "point_values.txt").open() as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            t,x,y,vx,vy,_,_,_,_,_,phase = map(float, line.split())
            expected.remove((t,x,y))
            r = math.sqrt(0.75)*(x-20000) + 0.5*y
            v = -0.5e-9*math.tanh(r/250.)
            require(all(map(math.isfinite, (vx,vy,phase))), "Nonfinite FE sample")
            wall_error = max(wall_error, abs(vx-0.5*v), abs(vy+math.sqrt(0.75)*v))
            h = 6/math.cosh(r/250.)**2
            exact_phase = 2*h/(2*h+1+math.sqrt(1+8*h))
            require(abs(phase-exact_phase) < 1e-6, "Phase profile at a plate sample")
    require(not expected and wall_error < 6e-16, f"Wall velocities differ by {wall_error} m/s")
    print(f"179 labels, 3 accepted clocks, 17 reported stations, 5 FE wall probes; maximum velocity error {wall_error:.4g} m/s")
    return history


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


def compare_production(clean, resumed):
    def physical_rows(path):
        return [{k: float(v) for k,v in row.items() if k.startswith("BP3 ")}
                for row in read_statistics(path / "statistics")]
    a, b = physical_rows(clean), physical_rows(resumed)
    require([row["BP3 clock (s)"] for row in a] == [0., 400., 900., 1525.], "Expected four production clocks")
    require(a == b and all(len(row) == 55 for row in a), "Changed production clock, peak or station history")
    print("Production continuation: 55 accepted values identical at all four clocks through 1525 s")


def plot(history, path):
    import matplotlib.pyplot as plt
    fig, (age, speed) = plt.subplots(2, 1, sharex=True, figsize=(7.2, 5.4), layout="constrained")
    x = [row[1]*0.002 for row in history[0]]
    age.semilogy(x, [row[3] for row in history[0]], color="#225577")
    age.set_ylabel("Initial state age (s)")
    age.set_title("60° thrust pilot")
    for frame, color, time in zip((history[0], history[-1]), ("#225577", "#b34b20"), (0, 900)):
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
    parser.add_argument("--restart-output", type=Path, help="Pilot resumed from its 400 s accepted parent")
    parser.add_argument("--production-pair", type=Path, nargs=2, metavar=("CLEAN", "RESUMED"))
    args = parser.parse_args()
    history = check_pilot(args.output)
    if args.mpi_output:
        check_pilot(args.mpi_output)
        compare_labels(history, args.mpi_output, range(3))
    if args.restart_output:
        compare_labels(history, args.restart_output, [2])
    if args.production_pair:
        compare_production(*args.production_pair)
    if args.figure:
        plot(history, args.figure)
