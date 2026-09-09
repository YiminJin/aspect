#!/usr/bin/env python3
"""Extract temporary linearization probes without changing simulation output."""
import argparse
import json
from pathlib import Path


FIELDS = {
    "EVAL": "evaluation velocity pressure",
    "LINEAR": "rhs_norm target iterations reported_residual true_residual velocity_error pressure_error",
    "BASE": "iteration bulk_scale initial_bulk reference_bulk floor_factor velocity_residual pressure_residual bulk_residual surface_residual surface_scale merit pressure_scaling",
    "DIRECTION": "solver_velocity solver_pressure physical_velocity physical_pressure",
    "FULL_LINEAR": "jacobian_velocity jacobian_pressure velocity_error pressure_error",
    "ACTION": "Auu Aup Apu B_delta_V",
    "REASSEMBLY": "bulk_norm velocity_difference pressure_difference",
    "WEAK_COMPONENT": "mode velocity_norm pressure_norm",
    "WEAK_BV": "norm",
    "CENTERED": "multiplier fd_norm jacobian_norm velocity_error pressure_error surface_error",
    "CENTERED_INADMISSIBLE": "multiplier",
    "TRIAL": "alpha bulk_residual surface_residual merit armijo_bound pressure_adjustment",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    rows = []
    iteration = None
    last_evaluation = None
    for line in args.log.read_text().splitlines():
        if not line.startswith("K1_AUDIT "):
            continue
        tokens = line.split()[1:]
        kind = tokens[0]
        numbers = []
        for token in tokens[1:]:
            try:
                numbers.append(float(token))
            except ValueError:
                if numbers:
                    raise ValueError(f"Label interrupted data: {line}")
        fields = FIELDS[kind].split()
        if len(fields) != len(numbers):
            raise ValueError(f"Wrong field count: {line}")
        row = dict(zip(fields, numbers))
        if kind == "BASE":
            iteration = int(row["iteration"])
            row["absolute_bulk_target"] = 1e-8*row["bulk_scale"]
        if kind == "EVAL":
            last_evaluation = row
        row["kind"] = kind
        if kind not in ("EVAL", "LINEAR"):
            row["newton_iteration"] = iteration
        if kind == "TRIAL":
            row["velocity_residual"] = last_evaluation["velocity"]
            row["pressure_residual"] = last_evaluation["pressure"]
            row["passes_armijo"] = row["merit"] <= row["armijo_bound"]
        rows.append(row)
    if not rows:
        raise ValueError("No failing-linearization diagnostics found")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
