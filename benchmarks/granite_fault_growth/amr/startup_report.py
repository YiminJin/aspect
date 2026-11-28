"""Collect completed or wall-limited granite startups without inferring a physical step."""
import argparse
import csv
import json
from pathlib import Path


def measured(directory):
    out, timing = directory / "output", directory / "timed"
    elapsed = json.loads((timing / "exit.json").read_text())
    ranks = [json.loads(p.read_text()) for p in sorted(timing.glob("rank-*.json"))]
    clocks = []
    for path in sorted(out.glob("balance-*.csv")):
        with path.open() as stream:
            row, = csv.DictReader(stream)
            clocks.append(float(row["time"]))
    lines = (timing / "wall.log").read_text(errors="replace").splitlines()
    diagnostics = [s for s in lines if "Midpoint iteration " in s or
                   "roundoff tolerance" in s or "native strip rate" in s or
                   "Exception" in s or "did not converge" in s]
    memory = [r.get("peak_KiB", r.get("sampled_peak_KiB")) for r in ranks]
    return dict(**elapsed, ranks_observed=len(ranks), accepted_physical_clocks_s=clocks,
                rank_high_water_KiB=memory, rank_cpu_s=[
                    r["user_s"] + r["system_s"] if "user_s" in r else None for r in ranks],
                initial_sections_present=(out / "sections-0000.csv").exists(), solver_messages=diagnostics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps({d.name: measured(d) for d in args.runs}, indent=2))
