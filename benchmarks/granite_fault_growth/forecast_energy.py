"""Retrospective energy warnings on uniform Cartesian RSF growing-fault trajectories.

At decision time use only the accepted phase, stress, recoverable strain,
signed section rate, age and plane angles. Compare predictions with new
seed-connected 0.05 contour crossings after opening the next endpoint.
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import label

growing = Path(__file__).resolve().parents[1] / "growing_fault"
sys.path[:0] = [str(growing), str(growing / "timestep")]
from audit import ELL, KF, Geometry, dot, shapes, tensor
from amr.trajectory import Case, values
from parameter_file import strip_material


@dataclass
class Frame:
    point: dict
    strip: dict
    coeff: np.ndarray
    time: float

    @classmethod
    def read(cls, case, step):
        point, fields, strip, ledger = case.load(step)
        component = fields["component"] == case.phase
        ids = np.unique(point["cell"])
        np.testing.assert_array_equal(fields["cell"][component], np.repeat(ids, 9))
        np.testing.assert_array_equal(fields["local"][component], np.tile(np.arange(9), len(ids)))
        return cls(point, strip, fields["value"][component], ledger["time"].item() if ledger else 0.)


class UniformFaultGeometry(Geometry):
    """Supply uniform physical rectangles to the independently integrated 5B SIP matrix."""

    def __init__(self, case, frame, par):
        p = frame.point
        base = par["Geometry model"]["Box"]
        self.cells, first = np.unique(p["cell"], return_index=True)
        self.weights = p["weight"]
        self.groups = np.split(np.arange(len(self.weights)),first[1:])
        self.strip = p["section"].astype(int)
        self.native_shapes, self.bounds, self.mass = [], [], []
        root = np.array([int(values(base, d+" repetitions")) for d in "XY"])
        ratio = np.sqrt(len(self.cells)/np.prod(root))
        grid = np.rint(ratio*root).astype(int)
        assert np.prod(grid) == len(self.cells)
        width = case.size/grid
        for group in self.groups:
            w = self.weights[group]
            np.testing.assert_allclose(sum(w), np.prod(width), rtol=1e-13, atol=1e-17)
            points = np.column_stack((p["x"][group],p["y"][group]))
            center = w@points/sum(w)
            i = np.floor((center-case.origin)/width)
            lo, hi = case.origin+i*width, case.origin+(i+1)*width
            unit = (points-lo)/(hi-lo)
            assert np.min(unit) >= -1.e-12 and np.max(unit) <= 1+1.e-12
            self.bounds.append((lo, hi))
            self.native_shapes.append(shapes(*unit.T, hi-lo)[0])
            self.mass.extend([np.prod(hi-lo)/9]*9)
        self.mass = np.array(self.mass)
        np.testing.assert_allclose(self.phase(frame.coeff),p["phi"],atol=1.e-12,rtol=0.)
        growing = strip_material(par)
        phase = par["Phase field"] if values(par["Phase field"],"Fracture energy") else par["Postprocess"]["Phase field surface energy"]
        assert values(phase,"Crack geometry") == "AT1"
        self.kf = float(values(phase,"Fracture energy"))/((8/3)*case.ell)
        self.G = float(values(growing, "Shear modulus"))
        self.M = 2*self.G*self.kf/float(values(growing, "Cohesion"))**2
        # Every term in the independent uniform SIP matrix is proportional to
        # 2 Kf ell^2 at the same degree and penalty. Reuse the 5B quadrature.
        params = par["Discretization"]["Stabilization parameters"]
        assert float(values(params, "Phase discontinuous penalty") or values(params, "Discontinuous penalty")) == 10.
        self.H = self.fracture_matrix() * ((self.kf*case.ell**2)/(KF*ELL**2))
        # These controls use the signed law independently qualified in 5B.
        assert values(growing, "Resistance law") == "rate and state"
        self.mu0 = float(values(growing,"Reference friction coefficient"))
        params = ("Direct effect","Evolution effect","Reference speed","Slip distance","Radiation damping")
        assert [float(values(growing,k)) for k in params] == [.015,.02,.001,.01,1000.]
        if values(growing,"Friction normal stress") == "adiabatic pressure":
            model = par["Adiabatic conditions model"]["Function"]
            assert values(model,"Function constants").replace(" ", "") == "p0=200,dp_dz=0"
            assert values(model,"Function expression").replace(" ","").split(";")[1] == "p0+dp_dz*depth"
        else:
            assert float(values(growing,"Prescribed fault compression") or values(par,"Surface pressure")) == 200.


def tip(frame, geom, seed, tangent, ell):
    """Replay the accepted piecewise plane graph and maximum incident DG trace."""
    stations = frame.strip
    origin = np.min([lo for lo, hi in geom.bounds],axis=0)
    start = (seed-origin)@tangent
    normal = np.array([-tangent[1], tangent[0]])
    lo = np.array([b[0] for b in geom.bounds])
    hi = np.array([b[1] for b in geom.bounds])
    result = seed.copy()
    graph_point = seed.copy()
    last = start
    for station, angle, formed in zip(stations["xi"], stations["beta"], stations["formed"]):
        if station < start-ell:
            continue
        if station > start and not formed:
            break
        if station <= start:
            point = seed+(station-start)*tangent
        else:
            graph_point += (station-last)*(tangent+np.tan(angle)*normal)
            point = graph_point
            last = station
        values = []
        for i in np.flatnonzero(np.all((lo-1e-12 <= point) & (point <= hi+1e-12), axis=1)):
            xy = (point-lo[i])/(hi[i]-lo[i])
            phi = shapes(xy[:1], xy[1:], hi[i]-lo[i])[0]@frame.coeff[9*i:9*i+9]
            values.append(phi.item())
        if not values or max(values) < .05:
            break
        if station > start:
            result = point.copy()
    return result


def scores(frame, geom, base_angle):
    """Conditional endpoint shear drive and energy derivative in a positive Bernstein direction."""
    p = frame.point
    strip = geom.strip
    a = base_angle+frame.strip["beta"][strip]
    traction = .5*(p["tauyy"]-p["tauxx"])*np.sin(2*a)+p["tauxy"]*np.cos(2*a)
    rate, age = frame.strip["V"], frame.strip["age"]
    # Signed regularized friction at the accepted age and 200 Pa compression.
    friction = 3*np.arcsinh(rate/.002*np.exp((geom.mu0+.02*np.log(.1*age))/.015))+1000*rate
    drive = geom.M*(traction-friction[strip])**2/(2*geom.G)
    positive = p["phi"] > 0.
    phi = p["phi"][positive]
    A = tensor(p, "A")[positive]
    # At fixed endpoint recoverable strain, differentiate G(A:A)(1-phi)^2/
    # (M phi(1+phi)). At zero phase the intact limiting drive uses traction.
    drive[positive] = geom.G*dot(A, A)*(1+3*phi)*(1-phi)/(geom.M*(phi*(1+phi))**2)
    assert np.isfinite(drive).all()
    gradient_and_faces = geom.H@frame.coeff
    load = geom.load(drive)
    row_scale = geom.kf*geom.mass
    # R_i = Kf*int N_i + (H*b)_i - int N_i Y. At R_i=0 the
    # normalized warning reaches 1. Only admissible one-sided rows count.
    one_sided = (load-gradient_and_faces)/row_scale
    one_sided[frame.coeff >= 1] = -np.inf
    flat = ((np.max(abs(frame.coeff.reshape(-1, 9)), axis=1) < 1.e-14) &
            (np.max(abs((gradient_and_faces/row_scale).reshape(-1, 9)), axis=1) < 1.e-12))
    pointwise = np.array([np.max(drive[g])/geom.kf for g in geom.groups])
    cellwise = np.max(one_sided.reshape(-1, 9), axis=1)
    return np.where(flat, pointwise, cellwise), flat


def connected_samples(frame, geom, samples, seed, tangent, ell):
    """Use 4-connected physical sample bins with a positive seed row as anchor."""
    lo, hi = np.moveaxis(np.asarray(geom.bounds), 1, 0)
    h = hi-lo
    np.testing.assert_allclose(h, np.tile(h[0], (len(h), 1)), atol=1e-13)
    boxes = np.rint((lo-lo.min(axis=0))/h).astype(int)
    nx, ny = boxes.max(axis=0)+1
    index = np.full((ny*samples, nx*samples), -1, dtype=int)
    values = np.full_like(index, np.nan, dtype=float)
    s = (np.arange(samples)+.5)/samples
    x, y = np.meshgrid(s, s)
    b = shapes(x.ravel(), y.ravel(), h[0])[0]
    for i,(ix,iy) in enumerate(boxes):
        sl = np.s_[iy*samples:(iy+1)*samples, ix*samples:(ix+1)*samples]
        index[sl] = i
        values[sl] = (b@frame.coeff[9*i:9*i+9]).reshape(samples,samples)
    assert (index >= 0).all() and np.isfinite(values).all()
    components, _ = label(values >= .05)
    origin = lo.min(axis=0)
    anchor_x, anchor_y = np.floor((seed-ell*tangent-origin)*samples/h[0]).astype(int)
    anchor = components[anchor_y,anchor_x]
    if anchor == 0:
        raise ValueError("Seed anchor has no connected 0.05 phase.")
    x = origin[0]+(np.arange(nx*samples)+.5)*h[0,0]/samples
    y = origin[1]+(np.arange(ny*samples)+.5)*h[0,1]/samples
    distance_ahead = (x-seed[0])[None,:]*tangent[0]+(y-seed[1])[:,None]*tangent[1]
    return values, components == anchor, index, distance_ahead


def replay(path, samples):
    frames = sorted(path.glob("balance-*.csv"))
    case = Case(path)
    prm = json.loads((path/'parameters.json').read_text())
    seed, tangent, ell = case.seed, case.t, case.ell
    parent = Frame.read(case, 0)
    geom = UniformFaultGeometry(case, parent, prm)
    lo, hi = np.moveaxis(np.asarray(geom.bounds), 1, 0)
    result = []
    prepared = np.zeros(len(geom.cells), dtype=bool)
    for step in range(1, len(frames)+1):
        center = tip(parent, geom, seed, tangent, ell)
        distance = np.linalg.norm(np.maximum(0., np.maximum(lo-center, center-hi)), axis=1)
        if np.any((distance < 3*ell) & ~prepared):
            prepared |= distance <= 4*ell
        # Restrict this comparison to ahead-of-tip cells. The other cells in
        # the preexisting seed band do not change a forward warning decision.
        forward = np.sum(np.where(tangent>=0, hi, lo)*tangent, axis=1) > center@tangent
        score, flat = scores(parent, geom, np.arctan2(tangent[1],tangent[0]))
        old, old_component, index, ahead = connected_samples(parent, geom, samples, seed, tangent, ell)
        predicted = {f: forward & (score >= f) for f in (.7, .9)}
        child = Frame.read(case, step)  # The forecast above has not seen this endpoint.
        for key in ("cell", "q", "section", "x", "y", "weight"):
            np.testing.assert_array_equal(child.point[key],parent.point[key],err_msg="Fixed physical quadrature: "+key)
        np.testing.assert_allclose(geom.phase(child.coeff), child.point["phi"], atol=1.e-12, rtol=0.)
        new, new_component, _, _ = connected_samples(child, geom, samples, seed, tangent, ell)
        crossing = new_component & ~old_component & (new-old > 1e-5)
        advancing = forward & np.isin(np.arange(len(lo)), np.unique(index[crossing]))
        row = dict(parent_s=parent.time, endpoint_s=child.time, tip_xy_m=list(center),
                   next_tip_xy_m=list(tip(child,geom,seed,tangent,ell)),
                   contour_ahead_m=float(np.max(ahead[new_component])),
                   crossings_ahead=int(np.sum(crossing & (ahead > (center-seed)@tangent))),
                   forward_cells=int(sum(forward)), flat=int(sum(flat & forward)),
                   flat_max=float(max(score[flat & forward],default=0.)),
                   diffuse_max=float(max(score[~flat & forward],default=0.)),
                   crossing_cells=int(sum(advancing)), disk_missed=int(sum(advancing & ~prepared)))
        for f, flagged in predicted.items():
            row[str(f)] = dict(flagged=int(sum(flagged)), flat=int(sum(flat & flagged)),
                               predicted_crossings=int(sum(advancing & flagged)),
                               additional_fine=int(sum(flagged & ~prepared)),
                               missed_with_disk=int(sum(advancing & ~(prepared | flagged))))
        result.append(row)
        parent = child
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("uniform_output",type=Path)
    p.add_argument("--samples",type=int,default=16)
    args = p.parse_args()
    print(json.dumps(replay(args.uniform_output, args.samples), indent=2))
