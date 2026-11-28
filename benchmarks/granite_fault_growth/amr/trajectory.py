"""Measure connected phase and formed history on an adaptive growing-fault mesh.

The native quadrature's physical area and first moments locate each Cartesian
cell independently of CellId ordering or MPI partition. The 0.01 and 0.05
contours are approximated on a common uniform lattice of sample centers;
--samples gives the number across the shortest fine edge. Face-connected
samples touching the seed one length behind its physical tip define the front.
Positive projected distance is forward along the prescribed seed tangent.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import label

growing = Path(__file__).resolve().parents[2] / "growing_fault"
sys.path[:0] = [str(growing), str(growing / "timestep")]
from frames import read, field_names
from parameter_file import strip_material


def values(par, *path):
    for key in path:
        par = par[key]
    return par["value"]


def bernstein(z):
    return np.column_stack(((1-z)**2, 2*z*(1-z), z*z))


class Case:
    def __init__(self, folder):
        self.folder = folder
        par = json.loads((folder / "parameters.json").read_text())
        box = par["Geometry model"]["Box"]
        self.size = np.array([float(values(box, a+" extent")) for a in "XY"])
        self.origin = np.array([float(values(box, "Box origin "+a+" coordinate")) for a in "XY"])
        root = self.size / [int(values(box, a+" repetitions")) for a in "XY"]
        self.aspect = root[0]/root[1]
        growing = strip_material(par)
        angle = np.deg2rad(float(values(growing, "Seed tangent angle")))
        self.t = np.array([np.cos(angle), np.sin(angle)])
        self.n = np.array([-self.t[1], self.t[0]])
        self.seed = np.fromstring(values(growing, "Forward seed tip"), sep=",")
        self.seed_s = (self.seed-self.origin) @ self.t
        self.ell = float(values(par["Phase field"] if "Phase field" in par and
                                values(par["Phase field"], "Length scale") else
                                par["Postprocess"]["Phase field surface energy"], "Length scale"))
        self.dc = float(values(growing, "Slip distance"))
        self.phase = field_names(folder).index("phase_field")

    def load(self, step):
        folder = self.folder
        tag = f"{step:04}"
        def shards(name, keys):
            return read(folder.glob(f"{name}-{tag}-r*.csv"), keys)
        pts = shards("points", ("cell","q"))
        fields = shards("fields", ("cell","component","local"))
        sections = read([folder / f"sections-{tag}.csv"], ("section",))
        book = read([folder / f"balance-{tag}.csv"], ("step",)) if step else None
        for table in (pts, fields, sections, book or {}):
            for key, data in table.items():
                if key != "cell" and not np.isfinite(data).all():
                    raise ValueError(f"{folder}, step {step}: nonfinite {key}")
        return pts, fields, sections, book

    def phase_patches(self, points, fields):
        """Recover cell bounds from mapped integration, check against native phase."""
        ids, index = np.unique(points["cell"], return_index=True)
        phase = fields["component"] == self.phase
        np.testing.assert_array_equal(fields["cell"][phase], np.repeat(ids,9))
        patches = fields["value"][phase].reshape(-1,3,3)
        cells = []
        for patch, group in zip(patches, np.split(np.arange(len(points["weight"])), index[1:])):
            w = points["weight"][group]
            area = sum(w)
            pos = np.column_stack([points[a][group] for a in ("x","y")])
            size = np.sqrt(area * np.array([self.aspect,1/self.aspect]))
            lower = w @ pos / area - size/2
            u, v = ((pos-lower)/size).T
            actual = np.einsum("qi,ij,qj->q", bernstein(v), patch, bernstein(u))
            np.testing.assert_allclose(actual, points["phi"][group], rtol=0, atol=1e-11)
            cells.append((lower, size, patch))
        return cells

    @staticmethod
    def trace(cells, point):
        traces = []
        for base, width, patch in cells:
            u, v = (point-base)/width
            if min(u,v) >= -1e-11 and max(u,v) <= 1+1e-11:
                traces.append((bernstein(np.array([v])) @ patch @ bernstein(np.array([u])).T).item())
        return max(traces, default=0.)

    def metrics(self, cells, sections, previous, samples):
        finest = np.min([size for _, size, _ in cells],axis=0)
        count = np.rint(self.size/finest).astype(int)*samples
        spacing = self.size/count
        image = np.full(count[::-1], np.nan)
        for base, width, patch in cells:
            i,j = np.rint((base-self.origin)/spacing).astype(int)
            ni,nj = np.rint(width/spacing).astype(int)
            u, v = ((np.arange(n)+.5)/n for n in (ni,nj))
            image[j:j+nj,i:i+ni] = bernstein(v) @ patch @ bernstein(u).T
        assert np.isfinite(image).all()
        x, y = (self.origin[d]+(np.arange(count[d])+.5)*spacing[d] for d in range(2))
        anchor = self.seed-self.t*self.ell
        # Anchor at the nearest admissible sample to a tracked seed point.
        distance = (x[None,:]-anchor[0])**2 + (y[:,None]-anchor[1])**2
        contours = []
        for level in (.01,.05):
            connected, _ = label(image >= level)
            j,i = np.unravel_index(np.argmin(np.where(connected,distance,np.inf)),image.shape)
            if connected[j,i] == 0 or distance[j,i] > (self.ell/2 + np.linalg.norm(spacing))**2:
                raise ValueError(f"Phase {level} has no connected seed anchor")
            jj,ii = np.nonzero(connected==connected[j,i])
            ss = (x[ii]-self.origin[0])*self.t[0]+(y[jj]-self.origin[1])*self.t[1]
            contours.append(max(ss))
        # The mesh disk follows formed stations; the physical direction tracker
        # additionally requires a contiguous section trace above 0.05.
        tip = tracked = self.seed.copy()
        contiguous = True
        station = self.seed_s
        for s,beta,formed in zip(sections["xi"],sections["beta"],sections["formed"]):
            if s <= self.seed_s:
                if s >= self.seed_s-self.ell:
                    contiguous &= self.trace(cells, self.seed + (s-self.seed_s)*self.t) >= .05
                continue
            if not formed:
                break
            tip = tip+(s-station)*(self.t + np.tan(beta)*self.n)
            if contiguous:
                contiguous = self.trace(cells, tip) >= .05
                if contiguous:
                    tracked = tip
            station = s
        formed = sections["formed"].astype(bool)
        # Signed rate on the last *previously* formed station immediately
        # behind the present connected contour, and independently at the
        # graph station meeting the physical tip's 0.05 condition.
        old = np.where(previous["formed"].astype(bool) & (previous["xi"] <= contours[1]))[0] if previous else []
        tip_rate = sections["V"][old[-1]] if len(old) else ""
        tracked_station = np.flatnonzero(formed & np.isclose(sections["xi"], (tracked-self.origin)@self.t,
                                                            rtol=0, atol=1e-12))
        tracked_rate = sections["V"][tracked_station[0]] if len(tracked_station) else ""
        return dict(cells=len(cells), h_m=min(finest), pixel_diagonal_m=np.linalg.norm(spacing),
                    contour_001_s_m=contours[0], contour_005_s_m=contours[1], formed_tip_x_m=tip[0], formed_tip_y_m=tip[1],
                    tracked_x_m=tracked[0], tracked_y_m=tracked[1],
                    formed_strips=sum(formed), formed_length_m=sum(sections["ds"][formed]),
                    previous_tip_id=old[-1] if len(old) else "", previous_tip_V_m_s=tip_rate,
                    tracked_V_m_s=tracked_rate)


def check_interval(case, prev, sections, dt):
    np.testing.assert_array_equal(prev["section"], sections["section"])
    np.testing.assert_array_equal(prev["xi"], sections["xi"])
    assert np.all(sections["formed"] >= prev["formed"]) and dt > 0
    np.testing.assert_allclose(sections["slip"]-prev["slip"], dt*sections["V"], rtol=1e-12, atol=1e-14)
    if "age" in sections:
        x = abs(sections["V"])*dt/case.dc
        theta = prev["age"]*np.exp(-x)+dt*np.divide(-np.expm1(-x),x,out=np.ones_like(x),where=x!=0)
        np.testing.assert_allclose(sections["age"], theta, rtol=1e-13, atol=1e-14)


def compare(left, right, fields, strict):
    # Use the existing continuation tolerance on identical runs; for MPI
    # reorderings use the established physical native/FE tolerances.
    for prefix, a, b, keys in zip(("native","fields","strip","balance"), left, right,
                                  (("cell","q"),("cell","component","local"),("section",),("step",))):
        if a is None and b is None:
            continue
        if not all(np.isfinite(v).all() for data in (a,b) for k,v in data.items() if k != "cell"):
            raise ValueError(f"{prefix}: nonfinite data")
        if a.keys() != b.keys():
            raise ValueError(f"{prefix}: column names differ")
        for name in a:
            if name in keys or name in ("formed","section","iterations") and (name!="iterations" or strict):
                np.testing.assert_array_equal(a[name], b[name], err_msg=prefix+"/"+name)
                continue
            if name=="iterations":
                continue
            if name=="value" and prefix=="fields" and not strict:
                for c in np.unique(a["component"]):
                    sel=a["component"]==c
                    atol={"velocity_x":1e-10, "velocity_y":1e-10, "pressure":1e-6,
                          "phase_field":1e-9}.get(fields[int(c)], 1e-12)
                    np.testing.assert_allclose(b[name][sel],a[name][sel],rtol=1e-7,atol=atol,err_msg=fields[int(c)])
                continue
            if strict:
                atol, rtol = 1e-12, 1e-12
            elif prefix=="native":
                atol = 1e-6 if name.startswith("tau") else 1e-9 if name=="phi" else 1e-14 if name in ("x","y","weight") else 1e-10
                rtol = 0 if atol==1e-14 else 1e-7
            elif prefix=="strip":
                atol,rtol = {"beta":1e-7, "force":1e-6}.get(name,1e-11),1e-7
            else:
                atol,rtol = {"phase_KKT":1e-7, "mechanical_residual":1e-9,
                             "adjoint_error":1e-12,"time":1e-15}.get(name, 1e-11),1e-6
                if name=="time":
                    rtol=0.
            np.testing.assert_allclose(b[name],a[name],rtol=rtol,atol=atol,err_msg=prefix+"/"+name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--compare",type=Path,help="Other trajectory with the same physical mesh at the selected steps")
    parser.add_argument("--strict",action="store_true",help="Compare with the project continuation tolerance (1e-12)")
    parser.add_argument("--steps",type=int,nargs="+", help="Compare outputs only at these steps")
    parser.add_argument("--samples",type=int,default=16,help="Samples per finest edge for connected contours")
    args = parser.parse_args()
    case = Case(args.output)
    expected = Case(args.compare) if args.compare else None
    tags = sorted(int(f.stem.split("-")[1]) for f in args.output.glob("sections-*.csv"))
    if not tags or any(b != a+1 for a,b in zip(tags,tags[1:])):
        raise ValueError("Missing station frame")
    if args.steps and not set(args.steps)<=set(tags):
        raise ValueError("Requested comparison frame is missing")
    prev = old_clock = None
    writer = None
    for step in tags:
        frame = case.load(step)
        pts, fe, sec, book = frame
        time = book["time"][0] if step else 0.
        midpoint = (old_clock+time)/2 if prev else ""
        if prev:
            check_interval(case, prev, sec, time-old_clock)
        if expected and (not args.steps or step in args.steps):
            compare(expected.load(step), frame, field_names(case.folder), args.strict)
        row = dict(step=step,time_s=time,midpoint_s=midpoint,
                   **case.metrics(case.phase_patches(pts,fe),sec,prev,args.samples))
        if writer is None:
            writer=csv.DictWriter(sys.stdout, row.keys(),lineterminator="\n")
            writer.writeheader()
        writer.writerow(row)
        prev, old_clock = sec,time


if __name__ == "__main__":
    main()
