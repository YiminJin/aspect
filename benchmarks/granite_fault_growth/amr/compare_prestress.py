"""Compare the two prestressed meshes only at shared accepted physical clocks."""
import argparse
import json
from pathlib import Path

import numpy as np

from granite_history import measure, write_csv
from sector_check import audit


def reported_cost(run):
    timing = run/"timing"
    end = timing/"exit.json"
    ranks = [json.loads(p.read_text()) for p in sorted(timing.glob("rank-*.json"))]
    return dict(elapsed_s=json.loads(end.read_text())["wall_s"] if end.exists() else None,
                per_rank_peak_KiB=[r.get("peak_KiB",r.get("sampled_peak_KiB")) for r in ranks])


def check(left,right,dest,last):
    dest.mkdir(parents=True,exist_ok=True)
    frames = []
    for mode,run in (("disk",left),("uniform",right)):
        folder = run/"output"
        geometry,_ = audit(folder,mode)
        initial = np.genfromtxt(folder/"sections-0000.csv",delimiter=",",names=True)
        np.testing.assert_array_equal(initial["beta"],0)
        edges = np.r_[initial["xi"]-initial["ds"]/2, initial["xi"][-1]+initial["ds"][-1]/2]
        np.testing.assert_allclose(edges,np.arange(97)*.015625,rtol=0,atol=1e-14)
        np.testing.assert_allclose(initial["xi"]+initial["ds"]/2,edges[1:],rtol=0,atol=1e-14)
        rows = measure([folder],16,2,None,last)
        frames.append({row["time_s"]:row for row in rows})
        write_csv(rows,dest/f"{mode}.csv")
        print(mode, json.dumps(dict(**geometry,edges=len(edges),**reported_cost(run),
                                    accepted_s=rows[-1]["time_s"])),flush=True)
    rows = []
    for time in sorted(frames[0].keys() & frames[1].keys()):
        a,b = [frame[time] for frame in frames]
        row = dict(time_s=time)
        for col in ("mean_delta_tauxy_Pa","wall_work_J_m","advance001_x_m","advance005_x_m",
                    "fixed_V_m_s","fixed_slip_m","peak_m_s","max_slip_Dc","formed"):
            row[f"disk_{col}"]=a[col]
            row[f"uniform_{col}"]=b[col]
            row[f"difference_{col}"]=a[col]-b[col]
        stations = [np.genfromtxt(run/"output"/f"sections-{entry['step']:04}.csv",names=True,delimiter=",")
                    for run,entry in ((left,a),(right,b))]
        x,y=stations
        for key in ("section","xi"):
            np.testing.assert_array_equal(x[key],y[key])
        for key in ("V","slip"):
            error=np.abs(x[key]-y[key])
            i=np.argmax(error)
            row[f"max_{key}_difference"]=error[i]
            row[f"max_{key}_difference_x_m"]=x["xi"][i]
            row[f"{key}_outside_station_tolerance"]=np.count_nonzero(error>1e-11+1e-7*np.abs(y[key]))
        rows.append(row)
    write_csv(rows,dest/"matched.csv")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("disk",type=Path)
    p.add_argument("uniform",type=Path)
    p.add_argument("destination",type=Path)
    p.add_argument("--last-step",type=int)
    args=p.parse_args()
    check(args.disk,args.uniform,args.destination,args.last_step)
