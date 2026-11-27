"""Recompute limits from every material station in each archived donor snapshot through onset."""
import argparse
import csv
from pathlib import Path
import re
import numpy as np

YEAR=31557600.
DC=.008


def durations(s, speed, theta, budget=.2, coefficient=.5, weakening=True):
    """At fixed signed rate, age moves monotonically toward Dc/|V|."""
    speed=np.abs(speed)
    a=.01+.015*np.clip((s-15000)/3000,0.,1.)
    with np.errstate(divide="ignore",invalid="ignore"):
        steady=DC/speed
        fraction=np.expm1(np.copysign(budget,steady-theta))*theta/(steady-theta)
        age=np.where((fraction>0)&(fraction<1),-steady*np.log1p(-fraction),np.inf)
        age=np.where(speed==0,theta*np.expm1(budget),age)
        allowance=np.where(a<=.015,coefficient*a/.015,np.inf) if weakening else coefficient
        slip=allowance*DC/speed
    return slip,age


def replay(donor,output):
    log=(donor/"output_start/log.txt").read_text()
    pattern=r"\*\*\* Timestep (\d+):\s+t=([\d.e+-]+) seconds, dt=([\d.e+-]+) seconds"
    steps={int(i):(float(t),float(h)) for i,t,h in re.findall(pattern,log)}
    longest=current=0
    for t,dt in steps.values():
        current=current+1 if dt==4e7 else 0
        longest=max(longest,current)
    print("Longest archived 4e7-s streak:",longest)
    paths=sorted((donor/"output_start/profiles").glob("fault_*.csv"),key=lambda p:int(p.stem.split("_")[1]))
    rows=[]
    for p in paths:
        step=int(p.stem.split("_")[1])
        if step>774: break
        data=np.genfromtxt(p,delimiter=",",names=True)
        s,v,age=(data[k] for k in ("xd_m","V_m_per_s","Theta_s"))
        assert len(v)==5776 and np.isfinite(np.stack([s,v,age])).all() and min(age)>0
        slip,state=durations(s,v,age)
        limits=np.minimum(slip,state)
        j=limits.argmin()
        t=data["time_s"][0]
        assert t==steps[step][0]
        rows.append(dict(step=step,time_s=t,old_dt_s=steps[step][1],peak_m_s=max(abs(v)),peak_s_m=s[np.argmax(abs(v))],
                         criterion="slip" if slip[j]<state[j] else "age",id=len(v)-1-j,station_m=s[j],
                         min_slip_s=min(slip),min_age_s=min(state),raw_s=limits[j],proposal_s=.99*limits[j],
                         old_uniform_slip_s=.1*DC/max(abs(v))))
        if step in (0,62,226,740,774): print(rows[-1])
    with output.open("w") as stream:
        w=csv.DictWriter(stream,fieldnames=rows[0],lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    data={r["step"]:r for r in rows}
    for step,unit,expected in [(62,YEAR,2.21),(226,86400,82.29),(774,1,3.16)]:
        assert abs(data[step]["min_age_s"]/unit-expected)<.01
    assert data[62]["proposal_s"]>4e7 and longest>=13
    assert data[774]["criterion"]=="slip" and abs(data[774]["min_slip_s"]-2.57)<.01
    assert abs(data[774]["station_m"]-10850)<1e-6


def plot(csv_path,png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with csv_path.open() as f: rows=list(csv.DictReader(f))
    x=[int(r["step"]) for r in rows]
    fig,ax=plt.subplots(figsize=(8,4.4),layout="constrained")
    for key,label,color,style in [("min_age_s","All-station age bound","#3573a0","-"),
                                  ("min_slip_s","Weakening-zone slip bound","#a04012","-"),
                                  ("old_uniform_slip_s","Uniform 0.1 Dc slip bound","#757575","--")]:
        ax.semilogy(x,[float(r[key]) for r in rows],color=color,ls=style,label=label)
    ax.axhline(4e7,color="#333",ls=":",label="Global ceiling")
    ax.set(xlabel="Archived accepted step",ylabel="Raw duration bound (s)")
    ax.legend(frameon=False,ncol=2,fontsize=9)
    ax.grid(axis="y",alpha=.2)
    fig.savefig(png,dpi=170)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("donor",type=Path)
    p.add_argument("csv",type=Path)
    p.add_argument("png",type=Path)
    a=p.parse_args()
    replay(a.donor,a.csv)
    plot(a.csv,a.png)
