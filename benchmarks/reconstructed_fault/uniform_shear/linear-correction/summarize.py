#!/usr/bin/env python3
"""Retain every estimated/fresh residual and the timestep it belongs to."""
import argparse
import json
import re
from pathlib import Path


def summarize(path):
    steps = []
    current = None
    for line in path.read_text().splitlines():
        match = re.search(r"\*\*\* Timestep (\d+):\s+t=([^ ]+) seconds", line)
        if match:
            current = dict(step=int(match[1]), time=float(match[2]),
                           linear=[], nonlinear=[], accepted_updates=0)
            steps.append(current)
        if current is None:
            continue
        for marker, key in (("Fault linear solve: ", "linear"),
                            ("Fault nonlinear residual: ", "nonlinear")):
            if marker in line:
                current[key].append({name: float(value) for name, value in
                                     (item.split("=") for item in
                                      line.split(marker)[1].split(", "))})
        if "Reconstructed-fault line search accepted" in line:
            current["accepted_updates"] += 1
        if "line search exhausted" in line:
            current["armijo_exhausted"] = True
    return steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.log), indent=2))
