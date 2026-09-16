"""Extract available nodal evidence only; never reinterpret FE history as stress.

This does not evaluate a constitutive response or reconstruct missing step-12
data. V_left is the last free node, not the mathematical Q1 limit at Wf.
"""
import argparse
import csv
from pathlib import Path


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for path in sorted(args.input.glob("fault_*.csv"),
                       key=lambda p: int(p.stem.split("_")[1])):
        with path.open() as stream:
            rows = [{k: float(v) for k, v in row.items()}
                    for row in csv.DictReader(stream)]
        rows.sort(key=lambda r: r["xd"])
        left = max((r for r in rows if not r["prescribed"]), key=lambda r: r["xd"])
        right = min((r for r in rows if r["prescribed"]), key=lambda r: r["xd"])
        step = int(path.stem.split("_")[1])
        entry = dict(step=step, time=left["time"], left_free_xd=left["xd"],
                     left_free_V=left["V"], first_prescribed_xd=right["xd"],
                     prescribed_V=right["V"], V_difference=left["V"]-right["V"],
                     relative_V_difference=left["V"]/right["V"]-1.)
        for lo, hi in [(13000., 20000.), (37000., 43000.)]:
            region = [r for r in rows if lo-1e-7 <= r["xd"] <= hi+1e-7]
            minimum = min(region, key=lambda r: r["q"])
            maximum = max(region, key=lambda r: r["q"])
            prefix = f"q_{int(lo/1000)}_{int(hi/1000)}km"
            entry.update({prefix+"_min": minimum["q"], prefix+"_max": maximum["q"],
                          prefix+"_ptp": maximum["q"]-minimum["q"],
                          prefix+"_min_xd": minimum["xd"], prefix+"_max_xd": maximum["xd"]})
        summary.append(entry)
        if step in (11, 12):
            profile = []
            for r in rows:
                if 10000.-1e-7 <= r["xd"] <= 45000.+1e-7:
                    profile.append(dict(xd=r["xd"], V=r["V"], prescribed=int(r["prescribed"]),
                                        q_consistent_Q1=r["q"], Theta_committed=r["Theta"],
                                        marker_km=next((km for km in (15, 18, 40)
                                                       if abs(r["xd"]-km*1000.) < 1e-7), "")))
            write_csv(args.output/f"saved_nodal_{step}.csv", profile)
    write_csv(args.output/"saved_junction_history.csv", summary)
    print(f"Extracted {len(summary)} saved states. No raw traction or missing step was reconstructed.")


if __name__ == "__main__":
    main()
