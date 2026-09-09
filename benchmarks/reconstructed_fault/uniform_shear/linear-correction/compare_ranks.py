#!/usr/bin/env python3
"""Compare accepted serial/MPI exports, never use one run as a physics reference."""
import argparse
import json
from pathlib import Path
import numpy as np


def read(path):
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("one", type=Path)
    parser.add_argument("two", type=Path)
    args = parser.parse_args()
    result = []
    for path in sorted(args.one.glob("time_*.csv"), key=lambda p: int(p.stem.split("_")[-1])):
        step = int(path.stem.split("_")[-1])
        if not (args.two/f"time_rank0_{step}.csv").exists():
            continue
        entry = dict(step=step, time=float(read(path)["time"][0]))
        for kind, fields in (("bulk", ("ux", "uy", "p", "phi", "old_tau_xy")),
                             ("surface", ("V", "Theta", "C", "Ih"))):
            a = read(args.one/f"{kind}_{step}.csv")
            ranks = [read(p) for p in sorted(args.two.glob(f"{kind}_rank*_{step}.csv"))]
            # Surface geometry/properties are replicated, unlike owned QP rows.
            b = np.concatenate(ranks) if kind == "bulk" else ranks[0]
            a = a[np.lexsort((a["y"], a["x"]))]
            b = b[np.lexsort((b["y"], b["x"]))]
            if a.shape != b.shape or max(np.max(abs(a[c]-b[c])) for c in ("x", "y"))>1e-13:
                raise ValueError("Not the same sampled geometry")
            entry[kind+"_max_abs_difference"] = {
                field: float(np.max(abs(a[field]-b[field]))) for field in fields}
            if kind == "bulk":
                entry["pressure_mean_Pa"] = [float(np.average(v["p"], weights=v["weight"]))
                                             for v in (a,b)]
        result.append(entry)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
