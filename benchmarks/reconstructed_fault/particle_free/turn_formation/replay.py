"""Compare independently scanned Coulomb maxima with registered planes at birth."""
import argparse
import csv
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path, help="Native run output with accepted point and plane exports")
parser.add_argument("--mu", type=float, required=True, help="Reference friction coefficient from input")
parser.add_argument("--through-step", type=int, help="Last accepted step to include")
parser.add_argument("--parent-step", type=int, help="Sample this earlier stress at all eventual birth stations")
parser.add_argument("--confinement", type=float, required=True, help="Scalar confinement (Pa), for orientation diagnostics")
parser.add_argument("--csv", type=Path, required=True)
args = parser.parse_args()
degrees = np.linspace(-90, 90, 18001)
s, c = np.sin(np.deg2rad(2*degrees)), np.cos(np.deg2rad(2*degrees))
with args.csv.open("x") as out:
    writer = csv.writer(out)
    writer.writerow(["birth_step","stress_step", "section", "x", "incoming_deg", "selected_deg", "scan_deg", "mean_u_Pa", "mean_v_Pa","resolved_shear_Pa","normal_minus_confinement_Pa"])
    for planes in sorted(args.output.glob("tip-planes-*.csv")):
        step = int(planes.stem.split("-")[-1])
        if args.through_step is not None and step>args.through_step:
            break
        rows = np.genfromtxt(planes, delimiter=",", names=True)
        births = rows[(rows["formed"] == 1) & (rows["parent_formed"] == 0)]
        if not len(births):
            continue
        stress_step = step-1 if args.parent_step is None else args.parent_step
        if stress_step>=step:
            raise ValueError("Stress must be sampled before formation")
        parent = np.genfromtxt(args.output/f"tip-planes-{stress_step:04}.csv", delimiter=",", names=True)
        files = sorted(args.output.glob(f"tip-points-{stress_step:04}-r*.csv*"))
        if not files:
            raise ValueError(f"Missing accepted stress at step {stress_step}")
        weights, us, vs, means = np.zeros((4,len(rows)))
        for path in files:
            a = np.genfromtxt(path, delimiter=",", names=True, usecols=("section","weight","tauxx","tauyy","tauxy"))
            ids = a["section"].astype(int)
            for acc, values in ((weights,a["weight"]), (us,a["weight"]*(a["tauyy"]-a["tauxx"])/2),
                                (vs,a["weight"]*a["tauxy"]), (means,a["weight"]*(a["tauxx"]+a["tauyy"])/2)):
                acc += np.bincount(ids, weights=values, minlength=len(rows))
        # This fixture has a 4 m full-height column for every x-Gauss station.
        np.testing.assert_allclose(weights,4*rows["reference_dx"],rtol=1e-12,atol=1e-15)
        for row in births:
            k = int(row["section"])
            u,v = us[k]/weights[k],vs[k]/weights[k]
            incoming = np.rad2deg(parent["angle"][k])
            shear = u*s+v*c
            score = abs(shear)+args.mu*(u*c-v*s)
            peaks = np.where((score[1:-1]>score[:-2]) & (score[1:-1]>=score[2:]))[0]+1
            feasible = [j for j in peaks if abs(degrees[j])<45]
            if u==0 and v==0:
                angle = incoming
            elif feasible:
                selected = min(feasible, key=lambda j:(abs(degrees[j]-incoming),-shear[j]))
                angle = degrees[selected]
            else:
                raise ValueError(f"No interior conjugate at section {k}, parent {stress_step}")
            sn,cs = np.sin(np.deg2rad(2*angle)),np.cos(np.deg2rad(2*angle))
            normal = means[k]/weights[k] + u*cs-v*sn-args.confinement
            writer.writerow([step,stress_step,k,row["x"],incoming,np.rad2deg(row["angle"]),angle,u,v,u*sn+v*cs,normal])
