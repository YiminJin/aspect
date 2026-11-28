"""Summarize the two 1 ms loading intervals and plot both measured midpoint rates.

The direct effect uses the measured initial mean shear-loading rate and fixed
RSF age. The second comparison extrapolates a duration-weighted straight-line
fit to the measured near-tip rates over the first millisecond.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def summarize(trajectory, initial_record, output):
    rows=load(trajectory)
    def col(k): return np.array([float(r[k]) if r[k] else np.nan for r in rows])
    t=col("time_s")
    dt=np.diff(t)
    if not (t[0]==0 and np.all(dt>0) and abs(t[-1]-.002)<1e-15):
        raise ValueError("Expected a complete accepted trajectory from 0 to 2 ms")
    ends=[np.flatnonzero(abs(t-s)<1e-15).item() for s in (0.,.001,.002)]
    mid=col("midpoint_s")[1:]
    np.testing.assert_allclose(mid,(t[1:]+t[:-1])/2,rtol=0,atol=1e-15)
    peak=col("peak_m_s")[1:]*1e6
    tip=col("fixed_V_m_s")[1:]*1e6
    dc=1e-6
    # The fixed material station is ID 35 on this grid. It remains the
    # previously formed station immediately behind the tip in this experiment.
    np.testing.assert_array_equal(col("fixed_id"),35)
    old=load(initial_record)
    early,=[row for row in old if row["case"]=="drive" and abs(float(row["time_s"])-60e-6)<1e-16]
    initial_loading=float(early["mean_delta_xy_Pa"])/float(early["time_s"])
    rate_constant=initial_loading/(.0137*7e7)
    report=[]
    for i,j in zip(ends,ends[1:]):
        clock=mid[i:j]
        interval=dt[i:j]
        v=tip[i:j]
        slope,offset=np.polyfit(clock,v,1,w=np.sqrt(interval))
        if i==0:
            initial_fit=slope,offset
        k=i+np.argmax(peak[i:j])
        report.append(dict(start_ms=t[i]*1000,end_ms=t[j]*1000,steps=j-i,
            peak_um_s=peak[k],peak_midpoint_ms=mid[k]*1000,
            peak_id=int(col("peak_id")[k+1]),peak_x_m=col("peak_xi_m")[k+1],
            end_midpoint_ms=mid[j-1]*1000,near_end_um_s=tip[j-1],
            fixed_mean_um_s=(col("fixed_slip_m")[j]-col("fixed_slip_m")[i])/(t[j]-t[i])*1e6,
            fixed_slip_Dc=col("fixed_slip_m")[j]/dc,
            fixed_interval_slip_Dc=(col("fixed_slip_m")[j]-col("fixed_slip_m")[i])/dc,
            max_slip_Dc=col("max_slip_Dc")[j],
            near_slope_um_s2=slope,near_secant_um_s2=(v[-1]-v[0])/(clock[-1]-clock[0]),
            near_first_um_s=v[0],early_trend_um_s=np.exp(rate_constant*mid[j-1]),
            max_ahead_phi_upper=max(col("ahead_phi_upper")[i:j+1]),
            advance001_m=col("x001_m")[j]-col("x001_m")[i],
            advance005_m=col("x005_m")[j]-col("x005_m")[i],
            max_excursion001_m=max(abs(col("x001_m")[i:j+1]-col("x001_m")[i])),
            max_excursion005_m=max(abs(col("x005_m")[i:j+1]-col("x005_m")[i]))))
    fig,(a,b,c)=plt.subplots(3,1,sharex=True,figsize=(8.5,7.8),layout="constrained",
                             gridspec_kw={"height_ratios":[1.5,1.,.8]})
    x=mid*1e3
    a.plot(x,peak,color="#156386",lw=2,label="Fastest strip")
    a.plot(x,tip,color="#aa4f2b",lw=2,label="Signed rate just behind seed tip")
    a.plot(x,np.exp(rate_constant*mid),"--",color="#222222",lw=1.3,label="Early mean loading, fixed age")
    a.set_ylabel("Midpoint slip rate (µm/s)")
    a.legend(frameon=False)
    b.plot(x,(tip-1)*1000,color="#aa4f2b",lw=2,label="Near-tip excess over 1 µm/s")
    slope,offset=initial_fit
    b.plot(x,(slope*mid+offset-1)*1000,"--",color="#444444",lw=1.3,label="First-interval linear fit")
    b.set_ylabel("Near-tip excess (nm/s)")
    b.legend(frameon=False)
    c.plot(1000*t,col("advance001_x_m")*1e3,color="#254e72",lw=2.5,label="φ = 0.01")
    c.plot(1000*t,col("advance005_x_m")*1e3,"--",color="#dd8159",lw=1.5,label="φ = 0.05")
    for level,style in (("001",":"),("005","-.")):
        along=col(f"advance{level}_arc_m")
        if max(abs(along-col(f"advance{level}_x_m")))>1e-12:
            c.plot(t*1000,along*1000,style,label=f"φ = {int(level)/100:g}, along graph")
    if max(abs(col("tracked_tip_arc_m")))>1e-12:
        c.plot(t*1000,1000*col("tracked_tip_arc_m"),color=".25",label="Mechanical tip along graph")
    c.set(xlabel="Physical time (ms)",ylabel="Front advance (mm)")
    if max(abs(col("advance001_x_m")))<1e-12 and max(abs(col("advance005_x_m")))<1e-12:
        c.set_ylim(-.015,.015)
    c.legend(frameon=False,ncol=2)
    for ax in (a,b,c):
        ax.axvline(1,ls=":",color=".6",lw=.8)
        ax.grid(axis="y",alpha=.2)
        ax.set_xlim(0,2)
    output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/"two-ms-rates-and-fronts.png",dpi=180)
    plt.close(fig)
    with (output/"intervals.json").open("w") as stream:
        json.dump(dict(early_loading_Pa_s=initial_loading,fixed_age_direct_effect_per_s=rate_constant,
                       intervals=report),stream,indent=2)
        stream.write("\n")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("trajectory",type=Path)
    p.add_argument("initial_record",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    print(json.dumps(summarize(a.trajectory,a.initial_record,a.output),indent=2))
