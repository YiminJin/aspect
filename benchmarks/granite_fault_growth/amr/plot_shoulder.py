"""Replay a compact shoulder control from its accepted physical DG traces."""
import argparse
import csv
from pathlib import Path

import numpy as np
from trajectory import Case


def measure(folder, dest):
    case = Case(folder)
    x = np.linspace(.5, 1.125, 501)
    y = .5
    slices, records = [], []
    for step, _ in enumerate(sorted(folder.glob("sections-*.csv"))):
        pts, fields, sections, book = case.load(step)
        cells = case.phase_patches(pts,fields)
        time_s = book["time"].item() if book else 0.
        m = case.metrics(cells, sections, None, 64)
        # The convex hull of the Bernstein coefficients bounds the entire
        # polynomial. The initially intact region begins on a cell face here.
        intact = [np.max(abs(patch)) for lo, size, patch in cells if lo[0] >= 1.-1.e-12]
        assert intact and max(intact) < 1.e-14
        records.append([time_s, m["contour_005_s_m"], m["tracked_x_m"], m["formed_tip_x_m"],
                        case.trace(cells,[.875,y]), sections["V"][np.isclose(sections["xi"],.875)].item()])
        if step < 2:
            slices.append([case.trace(cells, [p,y]) for p in x])
    assert len(records) >= 2
    result = np.array(records)
    np.testing.assert_allclose(result[:2,2], [.75,.875], rtol=0, atol=1e-12)
    assert result[0,4] < .05 < result[1,4]
    assert result[0,1] < .75 < result[1,1] < 1.
    np.savez_compressed(dest/"shoulder-profile.npz", x=x, slices=slices, history=result)
    with (dest/"shoulder-history.csv").open("w") as stream:
        rows = csv.writer(stream, lineterminator="\n")
        rows.writerow(("time_s", "connected_005_x_m", "supported_station_x_m", "formed_station_x_m",
                       "phase_at_x0875", "V_at_x0875_m_s"))
        for i, record in enumerate(records):
            rows.writerow(["" if i == 0 and k == 3 else f"{value:.16g}"
                           for k, value in enumerate(record)])


def draw(dest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = np.load(dest/"shoulder-profile.npz")
    x, slices, history = (data[k] for k in ("x","slices","history"))
    fig, axes = plt.subplots(1,2,figsize=(9.2,3.1),layout="constrained")
    ax, time_axis = axes
    ax.axvspan(.75,1.,color="#e2e9df",label="Initial diffuse shoulder")
    ax.axhline(.05,linestyle=":",color="0.28",lw=1)
    for i,(phase,color) in enumerate(zip(slices,["#324c81","#ac4d0d"])):
        ax.plot(x,phase,color=color,label=f"{1000*history[i,0]:g} ms")
        ax.plot(.875,history[i,4],"o",color=color,ms=5)
    ax.set(xlim=(.69,1.025),ylim=(-.006,.26),xlabel="x on the initial centerline (m)",ylabel="Phase at y = 0.5 m")
    ax.legend(frameon=False)
    for c, label, color, marker in [(1,"Connected 0.05 contour","#ac4d0d","o"),
                                    (2,"Phase-supported station","#324c81","s"),
                                    (3,"Last formed station","#44694a","^")]:
        # Initially no material station is formed; the seed is a prescribed position.
        series = history[1:] if c == 3 else history
        time_axis.plot(1000*series[:,0],series[:,c],color=color,marker=marker,ms=3,label=label)
    time_axis.set(ylim=(.72,1.04),xlabel="Accepted time (ms)",ylabel="Forward x coordinate (m)")
    time_axis.legend(frameon=False)
    fig.savefig(dest/"shoulder-front.png",dpi=180)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("destination",type=Path)
    p.add_argument("--output",type=Path,help="Fresh physical output to extract instead of the retained arrays")
    args = p.parse_args()
    if args.output:
        measure(args.output,args.destination)
    draw(args.destination)
