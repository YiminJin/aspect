#!/usr/bin/env python3
"""Convert read-only K1 coupled residual records into dimensional/scaled tables."""
import argparse
import csv
import json
from pathlib import Path
import re


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    rows=[]
    for block in args.trace.read_text().split("COUPLED_RESIDUAL_RECORD\n")[1:]:
        values=[float(v) for v in re.findall(r"\$\d+ = ([\d.eE+-]+)\n",block)[:9]]
        if len(values)!=9:
            raise ValueError("Incomplete coupled residual record")
        step,time,iteration,bulk,surface,bulk_scale,surface_scale,rb,rs=values
        if abs(bulk/bulk_scale-rb)>1e-12 or abs(surface/surface_scale-rs)>1e-12:
            raise ValueError("Norm/scale mismatch")
        rows.append(dict(step=int(step),time_s=time,iteration=int(iteration),
                         bulk_norm=bulk,surface_RMS_Pa=surface,bulk_scale=bulk_scale,
                         surface_scale_Pa=surface_scale,relative_bulk=rb,relative_surface=rs,
                         converged=rb<1e-8 and rs<1e-8))
    if not rows:
        raise ValueError("Missing production residual records")
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/"nonlinear_residuals.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    accepted=[row for row in rows if row["converged"]]
    (args.output/"converged_residuals.json").write_text(json.dumps(accepted,indent=2)+"\n")
    print(json.dumps(accepted,indent=2))


if __name__ == "__main__":
    main()
