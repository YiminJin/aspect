"""Measure the accepted compact-seed trajectories and plot their physical FE traces.

Each run uses the same initially intact forward cells, fixed material stations
and physical clocks. Cell shapes and phase values come from accepted native
writes. For replay without fresh solver outputs use --draw on the saved data.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from trajectory import Case, bernstein, check_interval, compare, field_names


def measure(destination, root, modes):
    records, mesh_data = [], {}
    fine = Case(root/"physical-uniform"/"output")
    for mode in modes:
        prefix = root/("physical-"+mode)
        case = Case(prefix/"output")
        timing = prefix/"timing"
        log = (timing/"solver.log").read_text()
        elapsed = json.loads((timing/"exit.json").read_text())
        assert elapsed["status"] == 0
        costs = [json.loads(f.read_text()) for f in timing.glob("rank-*.json")]
        assert costs
        prev = previous_clock = None
        for step, clock in enumerate((.0,.0025,.005)):
            points, fields, sec, book = case.load(step)
            cells = case.phase_patches(points,fields)
            # These two policies cover the entire small domain. Enforce
            # agreement on every finite-element, native, section and ledger
            # column; equality of the outlined contours is insufficient.
            if mode in ("disk", "sector"):
                compare(fine.load(step), (points, fields, sec, book), field_names(case.folder), True)
            if step:
                np.testing.assert_allclose(book["time"],clock,rtol=0,atol=1e-15)
                check_interval(case,prev,sec,clock-previous_clock)
            else:
                # Entire parent polynomials in the forward half-plane must
                # vanish, not just the values sampled at the material station.
                ahead = [abs(patch).max() for lo,size,patch in cells if lo[0]>=case.seed[0]-1e-12]
                assert ahead and max(ahead)<1e-14
            old = case.metrics(cells,sec,prev,32)
            m = case.metrics(cells,sec,prev,64)
            assert abs(m["contour_005_s_m"]-old["contour_005_s_m"])<=old["pixel_diagonal_m"]
            phi9 = case.trace(cells,[sec["xi"][9],.5])
            assert (phi9>.05 if step else phi9==0) and sec["formed"][9]==int(bool(step))
            np.testing.assert_allclose(m["tracked_x_m"],sec["xi"][9] if step else 1.25,atol=1e-12,rtol=0)
            assert m["contour_005_s_m"]>case.seed[0] if step else m["contour_005_s_m"]<case.seed[0]
            work = (book["Wext"]-book["energy_change"]-book["Db"]-book["Dc"]-book["Df"]).item() if step else None
            records.append(dict(policy=mode,step=step,time_s=clock,**m,phi_at_station_9=phi9,
                                V9_m_s=sec["V"][9],adjoint_J_m=book["adjoint_error"].item() if step else None,
                                physical_work_defect_J_m=work,compliance_remainder_J_m=book["phase_remainder"].item() if step else None,
                                passed_intervals=len(list(case.folder.glob("balance-*.csv"))),
                                preventive=log.count("Growing fault preventive refinement:"),
                                missed=log.count("Growing fault missed extension:"),
                                solves=sum(1 for ln in log.splitlines() if "Midpoint iteration 0," in ln),
                                wall_s=elapsed["wall_s"],cpu_s=sum(c["user_s"]+c["system_s"] for c in costs),
                                peak_rank_KiB=max(c["peak_KiB"] for c in costs)))
            if step in (0,2):
                mesh_data[f"{mode}_{step}"] = np.asarray([np.r_[lo,size,p.ravel()] for lo,size,p in cells])
            prev, previous_clock = sec, clock
    with (destination/"compact-trajectories.csv").open("w") as stream:
        writer = csv.DictWriter(stream, records[0],lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(destination/"compact-fields.npz",**mesh_data)


def draw(destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    data = np.load(destination/"compact-fields.npz")
    rows = list(csv.DictReader((destination/"compact-trajectories.csv").open()))
    modes = list(dict.fromkeys(r["policy"] for r in rows))
    fig, axes = plt.subplots(2,len(modes),figsize=(3.4*len(modes),5.7),layout="constrained",
                             gridspec_kw={"height_ratios":[1,1]})
    xlim=(.95,1.8)
    # Evaluate cell-interior Bernstein polynomials at common physical centers.
    x=np.linspace(0.,2.,513)[:-1]+1/512
    y=np.linspace(0.,1.,257)[:-1]+1/512
    for c,mode in enumerate(modes):
        space, series = axes[:,c]
        space.set_title(mode.capitalize())
        for step,line in [(0,"--"),(2,"-")]:
            field=np.full((len(y),len(x)),np.nan)
            segments=[]
            for box in data[f"{mode}_{step}"]:
                lo,dim=box[:2],box[2:4]
                dx,dy=dim
                xx=(x-lo[0])/dx
                yy=(y-lo[1])/dy
                i,j=(np.flatnonzero((a>0)&(a<1)) for a in (xx,yy))
                field[np.ix_(j,i)]=bernstein(yy[j]) @ box[4:].reshape(3,3) @ bernstein(xx[i]).T
                corners=lo+[[0,0],[dx,0],[dx,dy],[0,dy],[0,0]]
                segments.append(corners)
            assert np.isfinite(field).all()
            if step==2:
                space.add_collection(LineCollection(segments,linewidths=.35,colors="0.7"))
            space.contour(x,y,field,levels=[.01,.05],colors=["#327388","#b74d18"],linestyles=line,linewidths=1.45)
        points=[r for r in rows if r["policy"]==mode]
        time=[float(r["time_s"])*1000 for r in points]
        for key,color,marker,label in [("contour_005_s_m","#b74d18","o","Connected 0.05"),
                                        ("tracked_x_m","#37568a","s","Supported station"),
                                        ("formed_tip_x_m","#447346","^","Last formed")]:
            keep=1 if key=="formed_tip_x_m" else 0
            series.plot(time[keep:],[float(r[key]) for r in points[keep:]],marker=marker,c=color,ms=4,label=label)
        space.plot([1.25,float(points[-1]["tracked_x_m"])],[.5,.5],"s-",color="#37568a",ms=3)
        space.set(xlim=xlim,ylim=(.08,.92),aspect="equal",xlabel="x (m)")
        series.set(xlabel="Accepted time (ms)",ylim=(1.2,1.57))
        if c==0:
            space.set_ylabel("y (m)")
            series.set_ylabel("Forward x (m)")
            series.legend(frameon=False,fontsize=8)
    fig.savefig(destination/"compact-comparison.png",dpi=170)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination",type=Path)
    parser.add_argument("--run-root",type=Path)
    parser.add_argument("--modes",nargs="+",default=["uniform","sector","disk","reactive"])
    args=parser.parse_args()
    if args.run_root:
        measure(args.destination,args.run_root,args.modes)
    draw(args.destination)
