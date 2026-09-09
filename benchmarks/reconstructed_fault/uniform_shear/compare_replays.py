#!/usr/bin/env python3
"""Compare replay/MPI/restart exports without using them as a physics reference."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze import read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--first-step", type=int, default=0)
    parser.add_argument("--last-step", type=int, default=12)
    args = parser.parse_args()
    records = []
    # Replay agreement is much stricter than the physical benchmark budget:
    # 1e-9 of its characteristic scales. Preserve raw dimensional differences.
    fields = {
        "bulk": {"ux": 1e-4, "uy": 1e-4, "p": 1500., "phi": 1., "old_tau_xy": 1500.},
        "surface": {"V": 1e-4, "Theta": 200., "C": 1500., "Ih": 100.},
        "particles": {"H": 1500., "tau_xy": 1500.},
    }
    for step in range(args.first_step,args.last_step+1):
        time = [read(path,"time",step,("time",))[0] for path in (args.reference,args.replay)]
        if any(time[0][key] != time[1][key] for key in ("time","dt","U")):
            raise ValueError("Different accepted loading/timestep sequences")
        record = {"step": step, "time_s": float(time[0]["time"])}
        for name, scales in fields.items():
            a, b = [read(path,name,step,scales) for path in (args.reference,args.replay)]
            keys = ("id",) if name == "particles" else ("x","y")
            a, b = [v[np.lexsort(tuple(v[key] for key in reversed(keys)))] for v in (a,b)]
            if a.shape != b.shape or any(np.max(abs(a[key]-b[key]))>1e-13 for key in keys):
                raise ValueError(f"Different {name} sampling at step {step}")
            differences = {key: float(np.max(abs(a[key]-b[key]))) for key in scales}
            record[name] = differences
            if any(differences[key] > 1e-9*scale for key,scale in scales.items()):
                raise ValueError(f"Replay difference above comparison tolerance: {record}")
        records.append(record)
    print(json.dumps({"passed": True, "steps": records},indent=2))


if __name__ == "__main__":
    main()
