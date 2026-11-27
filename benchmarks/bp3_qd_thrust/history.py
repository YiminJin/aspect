"""Export ASPECT's accepted BP3 statistics for the existing three-event catalogue."""
import argparse
import csv
from math import isfinite
from pathlib import Path


def slip_budget(settings, station):
    """Independent dimensionless BP3 slip allowance at down-dip distance in metres."""
    control = settings["Time stepping"]["Fault RSF"]
    coefficient = float(control["Maximum slip fraction"]["value"])
    if coefficient == 0:
        return float("inf")
    policy = control.get("Slip resolution policy", {"value": "uniform"})["value"]
    if policy == "uniform":
        return coefficient
    if policy != "weakening dependent":
        raise ValueError(f"Unknown slip policy {policy}")
    law = settings["Material model"]["Phase field fault"]
    shallow = float(law["Direct effect"]["value"])
    b = float(law["Evolution effect"]["value"])
    a = shallow + (0.025 - shallow) * min(1., max(0., (station - 15000.) / 3000.))
    return coefficient * (a / b) if 0 < b and a <= b else float("inf")


def read_statistics(path):
    columns = []
    with path.open() as stream:
        for line in stream:
            if line.startswith("#"):
                number, name = line[1:].strip().split(": ", 1)
                if int(number) != len(columns) + 1:
                    raise ValueError(f"Unexpected statistics header: {line}")
                columns.append(name)
            elif line.strip():
                fields = line.split()
                if len(fields) != len(columns):
                    raise ValueError(f"Incomplete statistics frame in {path}")
                yield dict(zip(columns, fields))


def export(statistics, destination):
    destination.mkdir()
    with (destination / "accepted_steps.csv").open("w") as peaks, (destination / "stations.csv").open("w") as stations:
        peak = csv.writer(peaks)
        station = csv.writer(stations)
        peak.writerow(["step", "time", "dt", "max_V", "xd_at_max_V"])
        station.writerow(["step", "time", "dt", "xd", "V", "slip"])
        count = 0
        for frame in read_statistics(statistics):
            step = int(frame["Time step number"])
            if step != count:
                raise ValueError(f"Gap in accepted history at step {count}")
            clock = [step, frame["BP3 clock (s)"], frame["BP3 interval (s)"]]
            peak.writerow([*clock, frame["BP3 peak speed (m/s)"], frame["BP3 peak distance (m)"]])
            station_names = (name.split()[1] for name in frame if name.startswith("BP3 ") and name.endswith(" m rate (m/s)"))
            for s in sorted(station_names, key=float):
                prefix = f"BP3 {s} m "
                values = [frame[prefix + "rate (m/s)"], frame[prefix + "slip (m)"]]
                if not all(isfinite(float(v)) for v in values):
                    raise ValueError(f"Nonfinite history at {s} m, step {step}")
                station.writerow([*clock, s, *values])
            count += 1
        if count == 0:
            raise ValueError("Empty BP3 history")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("statistics", type=Path)
    parser.add_argument("new_directory", type=Path)
    args = parser.parse_args()
    export(args.statistics, args.new_directory)
