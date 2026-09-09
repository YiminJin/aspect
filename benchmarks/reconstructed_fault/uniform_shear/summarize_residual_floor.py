#!/usr/bin/env python3
"""Summarize fresh checks and dimensional bulk precision, without fitting it."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent/"linear-correction"))
from summarize import summarize


def smooth_velocity_residual_gain(nx, ny, kappa):
    """Independent Q2 bulk weak residual of ux=sin(pi*(y+.5)), uy=0.

    The x direction is periodic, the y endpoints are prescribed. This is a
    dimensional smooth-mode calibration, NOT a condition bound for arbitrary
    coupled perturbations. Actual remaining Newton corrections are also saved.
    """
    values = np.sin(np.pi*np.linspace(0.,1.,2*ny+1))
    action = np.zeros_like(values)
    element = ny*np.array([[7.,-8.,1.],[-8.,16.,-8.],[1.,-8.,7.]])/3.
    for cell in range(ny):
        action[2*cell:2*cell+3] += element@values[2*cell:2*cell+3]
    periodic_mass_norm = (.25/nx)*np.sqrt(5.*nx)/3.
    return kappa*periodic_mass_norm*np.linalg.norm(action[1:-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log",type=Path)
    parser.add_argument("--nx",type=int,default=16)
    args = parser.parse_args()
    steps = summarize(args.log)
    checks = [call for step in steps for call in step["linear"]]
    if not checks or any(call["fresh"] > call["target"]
                         or abs(call["residual null"]) > call["compatibility bound"] for call in checks):
        raise ValueError("A returned direction fails fresh/compatibility checks")
    records = []
    previous_time = 0.
    for step in steps:
        final = step["nonlinear"][-1]
        dt = step["time"]-previous_time if step["step"] else 2.
        kappa = -1e8*np.expm1(-dt/100.)
        gain = smooth_velocity_residual_gain(args.nx,4*args.nx,kappa)
        records.append(dict(step=step["step"],time_s=step["time"], **final,
                            bulk_normalized=final["bulk"]/(final["bulk target"]/1e-8),
                            surface_normalized=final["surface"]/final["surface scale"],
                            smooth_bulk_precision_equivalent_velocity_m_s=final["bulk precision"]/gain,
                            accepted_updates=step["accepted_updates"],
                            last_linear=step["linear"][-1]))
        previous_time = step["time"]
    print(json.dumps(dict(fresh_checks_pass=True,linear_calls=len(checks),steps=records),indent=2))


if __name__ == "__main__":
    main()
