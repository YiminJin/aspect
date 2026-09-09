#!/usr/bin/env python3
"""Audit the pressure correction against saved, unmodified accepted states.

Mean subtraction below is a diagnostic only: it distinguishes a gauge shift
from a spatial pressure change. Neither input nor simulation data is modified.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def read(directory, name, step):
    return np.atleast_1d(np.genfromtxt(directory / f"{name}_{step}.csv",
                                     names=True, delimiter=","))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    report = []
    for path in sorted(args.after.glob("time_*.csv")):
        step = path.stem.removeprefix("time_")
        if not (args.before / path.name).exists():
            continue
        time_before = read(args.before, "time", step)["time"][0]
        time_after = read(args.after, "time", step)["time"][0]
        if time_before != time_after:
            raise ValueError("Only matching accepted times may be compared")
        a, b = (read(directory, "bulk", step) for directory in (args.before, args.after))
        if a.shape != b.shape or max(abs(a["x"]-b["x"])) > 1e-14 or max(abs(a["y"]-b["y"])) > 1e-14:
            raise ValueError("Native quadrature coordinates differ")
        mean = lambda values: float(np.average(values, weights=a["weight"]))
        dp = b["p"]-a["p"]
        row = dict(time_s=float(time_after), before_mean_pressure_Pa=mean(a["p"]),
                   after_mean_pressure_Pa=mean(b["p"]),
                   pressure_difference_nonconstant_max_Pa=float(max(abs(dp-mean(dp)))),
                   velocity_difference_max_m_per_s=float(max(max(abs(b[f]-a[f])) for f in ("ux", "uy"))),
                   phase_difference_max=float(max(abs(b["phi"]-a["phi"]))))
        for table, fields in [("surface", ("V", "Theta", "C", "Ih")),
                              ("particles", ("H", "tau_xx", "tau_yy", "tau_xy"))]:
            a, b = (read(directory, table, step) for directory in (args.before, args.after))
            if a.shape != b.shape:
                raise ValueError(f"Mismatched {table} layout")
            row[table+"_maximum_absolute_changes"] = {
                field: float(max(abs(b[field]-a[field]))) for field in fields}
        report.append(row)
    if not report:
        raise ValueError("No matching accepted states")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
