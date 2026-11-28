"""Check the physical mesh of a finite, inclined seed independently of AMR marking."""
import argparse
import csv
from pathlib import Path

import numpy as np


def phase(x, y):
    return sum(.4*np.maximum(0., 1.-2.*abs(x-xc))*np.maximum(0., 1.-abs(y-yc)/.75)
               for xc, yc in [(2.5,3.), (3.,3.25), (3.5,3.5)])


def read_native_mesh(output):
    # Store physical area and first moments about the translated origin to
    # recover each axis-aligned 2:1 rectangular cell's bounds from native
    # quadrature. Its CellId and the ordering of Gauss points are opaque.
    cells = {}
    phase_error = 0.
    shards = sorted(output.glob("points-0000-r*.csv"))
    if len(shards) != 2:
        raise ValueError("Expected two native quadrature shards")
    for shard in shards:
        owned_cells = set()
        with shard.open() as f:
            for row in csv.DictReader(f):
                x,y,w,p = (float(row[col]) for col in ("x","y","weight","phi"))
                if not np.isfinite([x,y,w,p]).all() or w <= 0.:
                    raise ValueError(f"Invalid native sample in {shard}")
                owned_cells.add(row["cell"])
                m = cells.setdefault(row["cell"], np.zeros(4))
                m += (w,w*(x+1.),w*(y-2.),w*p)
                phase_error = max(phase_error, abs(p-phase(x,y)))
        if not owned_cells:
            raise ValueError(f"{shard} contains no owned cells")
    log = (output/"log.txt").read_text()
    if "2 MPI processes" not in log or not cells:
        raise ValueError("Expected a two-rank native history")
    moments = np.array(list(cells.values()))
    area = moments[:,0]
    np.testing.assert_allclose(sum(area), 24., atol=2e-12, rtol=0)
    if phase_error > 1e-12:
        raise ValueError(f"Maximum phase error {phase_error}")
    hx = np.sqrt(2*area)
    hy = hx/2
    x = moments[:,1]/area-1.
    y = moments[:,2]/area+2.
    return x, y, hx, hy, moments[:,3]/area, phase_error


def check(output, target):
    x, y, hx, hy, _, phase_error = read_native_mesh(output)

    # Each tent exceeds 0.01 over a contiguous area reaching its center,
    # and the neighboring tents overlap above 0.01. A physically damaged
    # cell therefore belongs to the seed whenever its supremum exceeds
    # 0.01. All breakpoints are root faces, so each cell's phase is bilinear
    # and the supremum is attained at a corner. Allow only roundoff when a
    # true corner value equals 0.01. Off-axis positive centers demonstrate
    # that the independently prescribed phase exceeds the AT1 profile width.
    corners = [phase(x + i*hx/2, y + j*hy/2) for i in (-1,1) for j in (-1,1)]
    connected = np.maximum.reduce(corners) > .01+1e-13
    wide = (phase(x,y) > .01) & (abs(y-.5*x-1.75)/np.sqrt(1.25) > .2)
    if not np.any(wide):
        raise ValueError("Missing off-axis witnesses")
    # A point on the backward ray well beyond all three tents is separated
    # from every positive sample. Refining it to the target invents a wake.
    px,py = 1.0625,2.28125
    wake = (abs(x-px) < hx/2+1e-13) & (abs(y-py) < hy/2+1e-13)
    assert wake.any() and phase(px,py) == 0.
    # The outer predictive disk has radius 4 ell. Its full interior should
    # be fine, including this point in intact granite forward of the tip.
    fx,fy = 4.3125,3.90625
    forward = (abs(x-fx) < hx/2+1e-13) & (abs(y-fy) < hy/2+1e-13)
    assert forward.any() and phase(fx,fy) == 0. and np.hypot(fx-4.,fy-3.75) < .4

    strips = np.genfromtxt(output/"sections-0000.csv",names=True,delimiter=",")
    # With tangent (2,1)/sqrt(5), projecting the box diagonal gives 19/sqrt(5)
    # metres. Uniform bins no wider than .25 m determine IDs and *both* edges.
    span = 19./np.sqrt(5.)
    edges = np.linspace(0., span, int(np.ceil(span/.25))+1)
    np.testing.assert_array_equal(strips["section"],np.arange(len(edges)-1))
    np.testing.assert_allclose(strips["xi"]-strips["ds"]/2,edges[:-1],atol=3e-14,rtol=0)
    np.testing.assert_allclose(strips["xi"]+strips["ds"]/2,edges[1:],atol=3e-14,rtol=0)
    np.testing.assert_array_equal(strips["age"],10.)
    np.testing.assert_array_equal(strips["slip"],0.)
    print(f"{len(x)} cells, largest seed edge {hx[connected].max():.8g} m, "
          f"{sum(wide)} off-axis witnesses ({sum(wide & (hx > target+1e-12))} underresolved), largest edge {hx[wide].max():.8g} m; "
          f"backward-ray edge {hx[wake].min():.8g} m, forward disk {hx[forward].max():.8g} m; "
          f"{len(strips)} strips over {span:.12g} m, seed error {phase_error:.3g}")
    if not (np.all(hx[connected] <= target+1e-12) and np.all(hx[forward] <= target+1e-12)
            and np.all(hx[wake] > target+1e-12)):
        raise ValueError("The connected seed, predictive disk or intact backward ray has the wrong resolution")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("maximum_edge",type=float)
    args = p.parse_args()
    check(args.output,args.maximum_edge)
