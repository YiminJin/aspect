"""Compare accepted ridge controls from run.py, including independent FE normal integration."""
import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def near(actual, expected, atol, rtol=0):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def labels(output, step, rank=0):
    values = np.loadtxt(output / f"labels-{step}-{rank}.txt", skiprows=1)
    near(values[:, 0], np.arange(len(values)), 0)
    return values


def bulk(output, step, rank=0):
    # Retain cell identity independently of traversal/partition order.
    return np.genfromtxt(output / f"bulk-{step}-{rank}.txt", names=True, dtype=None, encoding="utf8")


@lru_cache(None)
def gauss(order):
    x, w = np.polynomial.legendre.leggauss(order)
    return (x+1)/2, w/2


def lagrange(x):
    nodes = gauss(3)[0]
    return np.stack([np.prod([(x-nodes[k])/(z-nodes[k]) for k in range(3) if k != j], axis=0)
                     for j, z in enumerate(nodes)], axis=-1)


class Phase:
    """Owner-cell DGQ2 polynomial from the nine exported interior values."""
    def __init__(self, points, grid, ell):
        self.grid = grid
        self.m = 20 / ell
        self.values = np.full((grid, grid, 3, 3), np.nan)
        x = np.floor((points["x"]+.5)*grid).astype(int)
        y = np.floor((points["y"]+.5)*grid).astype(int)
        self.values[y, x, points["q"]//3, points["q"]%3] = points["phi"]
        require(np.isfinite(self.values).all(), "Missing DG sample")

    def evaluate(self, xy):
        coords = (xy+.5)*self.grid
        ix, iy = np.clip(np.floor(coords).astype(int), 0, self.grid-1).T
        vx = lagrange(coords[:, 0]-ix)
        vy = lagrange(coords[:, 1]-iy)
        return np.einsum("ij,ijk,ik->i", vy, self.values[iy, ix], vx)

    def h(self, value):
        require(np.all((value >= -1.e-4) & (value < 1.)), "Inadmissible independent FE evaluation")
        d = np.maximum(value, 0.)
        return self.m*d*(1+d)/(1-d)**2

    def ray(self, origin, normal, order=32, width=np.inf):
        # Split at all Cartesian faces, including the box boundary. On y=0
        # floor picks the upper DG trace. The data come from interior samples.
        axes = np.abs(normal) > 1.e-14
        lower = (-.5-origin[axes])/normal[axes]
        upper = (.5-origin[axes])/normal[axes]
        a = max(-width, np.min([lower, upper], axis=0).max())
        b = min(width, np.max([lower, upper], axis=0).min())
        faces = np.linspace(-.5, .5, self.grid+1)
        cuts = ((faces[:, None]-origin[axes]) / normal[axes]).ravel()
        cuts = np.unique(np.r_[a, cuts[(cuts > a) & (cuts < b)], b])
        q, w = gauss(order)
        times = cuts[:-1, None]+np.diff(cuts)[:, None]*q
        xy = origin+times.reshape(-1, 1)*normal
        h = self.h(self.evaluate(xy)).reshape(times.shape)
        return np.sum(np.diff(cuts)*(h @ w))


def normal_projection(phase, vertices, integrals, nominal_width):
    # Full dense independent mass/RHS. Retain endpoint and middle station
    # ray integrals separately from their Q1 representation.
    n = len(vertices)
    mass = np.zeros((n, n))
    rhs = np.zeros(n)
    rays = []
    for j, (left, right) in enumerate(zip(vertices[:-1], vertices[1:])):
        length = np.linalg.norm(right-left)
        t = (right-left)/length
        normal = np.array([-t[1], t[0]])
        for xi, w in zip(*gauss(3)):
            shape = np.array([1-xi, xi])
            center = shape @ vertices[j:j+2]
            exact = phase.ray(center, normal)
            admitted = phase.ray(center, normal, width=nominal_width)
            rows = slice(j, j+2)
            mass[rows, rows] += w*length*np.outer(shape, shape)
            rhs[rows] += w*length*shape*exact
            rays.append((exact, shape @ integrals[rows], (exact-admitted)/exact))
    return np.linalg.solve(mass, rhs), np.array(rays)


def metrics(output, step, rank=0):
    with (output / f"checks-{step}-{rank}.txt").open() as data:
        return {key: np.array([float(x) for x in values]) for key, *values in map(str.split, data)}


def geometry_error(case, vertices, grid):
    center = .07 if case == "translated" else 0.
    inclined = case.startswith("inclined") or case == "free"
    normal = np.array([-np.sqrt(3)/2, .5]) if inclined else np.array([-1., 0.])
    errors = abs((vertices-[center, 0]) @ normal)
    length = np.sum(np.linalg.norm(np.diff(vertices, axis=0), axis=1))
    near(vertices[[0, -1], 1], [-.5, .5], 2.e-15)
    require(np.all(np.diff(vertices[:, 1]) > 0), "Label order changed")
    core = errors[np.abs(vertices[:, 1]) <= .25].max()
    require(len(vertices) == (38 if inclined else 8), "Unexpected number of material labels")
    if case == "translated":
        require(errors.max() < .015625, "Translated seed failed at a contact or interior label")
    if case in ("inclined", "inclined-q1"):
        near(length, 2/np.sqrt(3), 2.e-15)
        near(errors, 0., 2.e-15)
    if case in ("translated", "inclined-fit"):
        require(core < .015625*abs(normal[0]), "Displaced seed failed to follow the interior FE ridge")
    expected_x = center + (vertices[[0, -1], 1]/np.sqrt(3) if inclined else np.zeros(2))
    contacts = abs(vertices[[0, -1], 0]-expected_x)
    near_ends = errors[[1, 2, -3, -2]]
    require(max(contacts) <= .25/grid, f"Contact errors {contacts} exceed quarter-cell budget")
    require(max(near_ends) <= .25/grid, f"Near-end errors {near_ends} exceed quarter-cell budget")
    expected_length = 2/np.sqrt(3) if inclined else 1.
    require(abs(length/expected_length-1) <= .01, "Relative length error exceeds 1%")
    require(errors[1:-1].max()<=.25/grid, "Interior ridge misses the quarter-cell budget")
    if case in ("translated", "inclined-fit"):
        require(max(near_ends) < .015625*abs(normal[0]) and max(contacts) < .015625,
                "Displaced seed failed to improve at its ends")
    return length, core, errors[1:-1].max(), contacts, near_ends


def check_case(root, case, grid):
    output = root / f"{case}-{grid}" / "output"
    init = labels(output, 0)
    points = bulk(output, 0)
    vertices = init[:, 1:3]
    geometry = geometry_error(case, vertices, grid)
    ell = .078125 if case == "narrow" else .15625
    phase = Phase(points, grid, ell)
    nominal_width = np.loadtxt(output.parent/"profile.txt")[-1,0]
    recovered, rays = normal_projection(phase, vertices, init[:, 3], nominal_width)
    print(f"  nominal profile half width {nominal_width:.9g} m: maximum omitted fraction {max(rays[:,2]):.8g}")
    integral_error = max(abs(recovered-init[:, 3]))
    # Compare the actual bulk Q3 points with the Q8 section rule on the same
    # nonconstant manufactured rate and virtual traction used by the probe.
    lengths = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    station = np.r_[0, np.cumsum(lengths)]
    j, xi = points["seg"], points["xi"]
    s = np.linspace(0., 1., len(vertices))
    v = .005*(1+.3*s+.2*s*s)
    virtual = .7+np.sin(3*(station[j]+xi*lengths[j]))
    w = gauss(3)[1]
    volumes = w[points["q"]//3]*w[points["q"]%3]/grid**2
    q3 = volumes @ (virtual*points["chi"]*((1-xi)*v[j]+xi*v[j+1]))
    print(f"{case}-{grid} Q3 minus Q8 bulk work {q3-metrics(output,0)['work'][0]:.6g}")
    if case == "free":
        # Reassemble the seed's first normal moment directly from FE points.
        # The seed is centered and straight, so an unbiased fit would have zero RHS.
        t = np.array([.5, np.sqrt(3)/2])
        normal = np.array([-t[1], t[0]])
        seed = np.array([-.5/np.sqrt(3), -.5])
        points2 = np.column_stack((points["x"], points["y"]))-seed
        eta = points2 @ normal
        seed_spacing = (2/np.sqrt(3))/(len(vertices)-1)
        node = np.clip(points2 @ t/seed_spacing, 0, len(vertices)-1)
        radius = np.loadtxt(output.parent/"profile.txt")[-1,0] + np.sqrt(2)/grid
        weight = volumes*np.maximum(0, points["phi"]-.1)*(np.linalg.norm(points2-node[:,None]*seed_spacing*t,axis=1) <= radius)
        for label in (0, len(vertices)//2):
            row = weight*np.maximum(0., 1.-abs(node-label))
            print(f"  seed row {label} unregularized weighted mean normal displacement {row @ eta / row.sum():.9g} m")
    # Both integrate the same bounded degree-two FE representation, split at
    # cell faces. 32-point rays and dense solve are independent of production.
    near(recovered, init[:, 3], 2.e-12, 3.e-13)
    middle, end = rays[len(rays)//2,:2], rays[0,:2]
    print(f"{case}-{grid} length, core/interior/contact errors {geometry}; "
          f"I defect {integral_error:.3g}; ray/Q1 middle {middle}, clipped {end}")
    if case in ("narrow", "vertical", "translated"):
        line = np.loadtxt(output / "transect-0-0.txt", skiprows=1)
        field = phase.evaluate(line[:, :2])
        near(field, line[:, 3], 2.e-14)
        near(phase.h(field), line[:, 4], 1.e-10, 2.e-13)
        width = float(next(line.split("=")[1] for line in (output.parent/"input.prm").read_text().splitlines()
                           if "set Profile half width" in line))
        center = vertices[len(vertices)//2, 0]
        normal_i = phase.ray(np.array([center, 0.]), np.array([1., 0.]), width=width)
        actual = line[:, 2] @ line[:, 5]
        i0 = np.interp(0., vertices[:, 1], init[:, 3])
        independent = .005*normal_i/i0
        # Only the compact admission boundary can cut the 24-point cut rule.
        # Bound that edge panel by its measured, nonnegative FE tail.
        edge = np.abs(np.abs(line[:, 0]-center)-width) <= 1/grid
        allowance = .005/i0*np.max(line[edge, 4],initial=0.)*2/grid
        near(actual, independent, allowance+1.e-13)
        # The translated face maximum can differ from the interior centroid.
        # Measure the resulting full-ray Q1 projection error independently.
        full = phase.ray(np.array([center,0.]),np.array([1.,0.]))
        projected = np.interp(0., vertices[:,1], recovered)
        projection_error = .005*abs(full/projected-1)
        near(actual,.005,projection_error + 2.1e-6*.005 + 1.e-13)
        print(f"  integrated slip {actual:.12g} independent {independent:.12g}, "
              f"peak engineering rate {max(line[:, 5]):.8g}; tail bound {allowance:.2g}")
    if case in ("free","inclined-fit"):
        for j,x in ((0,-.48),(-1,.48)):
            mutation=vertices.copy()
            mutation[j,0]=x
            try:
                geometry_error(case,mutation,grid)
            except ValueError as error:
                require("Contact errors" in str(error), "Contact mutation failed an unrelated check")
            else:
                raise ValueError("A displaced contact passed the fitted geometry check")
    free = case == "free"
    expected_dt = .0125 if free or case in ("narrow", "vertical", "translated") else .1
    near(init[:, 4], 11.682941969615793, 0.)
    near(init[:, 5], 0., 0.)
    previous = init
    outputs = sorted(output.glob("labels-*-0.txt"))
    require(len(outputs) == (3 if free else 2), f"Unexpected clock count for {case}")
    if free and grid == 32:
        record = Path(__file__).resolve().parents[3] / "refactoring/phase-field-fault/results/3a-repair-free-labels.csv"
        reference = np.loadtxt(record, delimiter=",", skiprows=1)
        history = np.vstack([labels(output, step) for step in range(3)])
        np.testing.assert_array_equal(history[:, 0], np.tile(np.arange(38), 3))
        np.testing.assert_array_equal(history[:, 7], np.repeat([0., .0125, .025], 38))
        np.testing.assert_array_equal(history[:, [0, 7]], reference[:, [0, 7]])
        near(history[:, 1:7], reference[:, 1:7], 1.e-10, 1.e-8)
        print(f"  corrected 3A reference: 38 labels at all three clocks; max absolute differences "
              f"(x, y, I, age, slip, rate) {np.max(np.abs(history[:, 1:7]-reference[:, 1:7]), axis=0)}")
    for step in range(len(outputs)):
        current = labels(output, step)
        near(current[:, 7], expected_dt*step, 1.e-16)
        near(current[:, :4], init[:, :4], 0)
        dt = expected_dt if step else 0.
        speed = current[:, 6]
        if not free:
            near(speed, .005, 0.)
        x = speed*dt/.008
        # Direct integration of theta'=1-theta V/Dc on each accepted interval.
        near(current[:, 4], previous[:, 4]*np.exp(-x)+.008/speed*(-np.expm1(-x)), 2.e-13)
        near(current[:, 5], previous[:, 5]+dt*speed, 2.e-16)
        if step and free:
            require(np.ptp(speed) > .001 and max(abs(speed-previous[:, 6])) > .001, "No resolved free rate response")
        previous = current
        check = metrics(output, step)
        near(check["imposed_balance"], 0 if free else 1, 0)
        work, dual = check["work"]
        near(work, dual, 256*np.finfo(float).eps*abs(work))
        error, scale = check["tangent"]
        # Difference of two assembled residuals divided by 2e-5. Allow for
        # serial accumulation of quadrature terms, plus the O(epsilon^2) term.
        require(error/scale < 2.e-8, f"Fixed-bulk tangent differs: {error} of {scale}")
        is_native = case.startswith("inclined")
        stress_scale = 50000. if is_native else 5.
        require(max(check["traction"]) < 32*np.finfo(float).eps*stress_scale, "Bulk traction mismatch")
        if free:
            require(max(check["residual"][0], check["independent_balance"][0]) < 128*np.finfo(float).eps,
                    "Free section failed its fixed-parent balance")
        if step:
            require(check["moments"][0] > 0, "No cell crossed a Q1 knot")
        print(f"  {step} work={abs(work-dual):.3g}/{work:.4g}; tangent={error:.3g}/{scale:.4g}; "
              f"traction={check['traction']}; balance={check['independent_balance'][0]:.3g}; "
              f"Q16 cut moments (cells, max, scale, sum)={check['moments']}")
        if step:
            # Fixed phase at every sampled cell identity and point.
            sampled = bulk(output, step)
            for key in ("cell", "q", "phi"):
                np.testing.assert_array_equal(sampled[key], points[key])
    return metrics(output, 1)["moments"]


def rank_check(root):
    serial = root / "free-32/output"
    parallel = root / "free-32-mpi2/output"
    for step in range(3):
        reference = labels(serial, step)
        for rank in (0, 1):
            near(labels(parallel, step, rank), reference, 1.e-10, 1.e-8)
        counts = [int(metrics(parallel,step,r)["samples"][0]) for r in (0,1)]
        require(min(counts)>0 and sum(counts)==metrics(serial,step)["samples"][0], "Distributed active Q8 count mismatch")
        a = np.sort(bulk(serial, step), order=["cell", "q"])
        shards = [bulk(parallel, step, rank) for rank in (0, 1)]
        require(all(len(shard) > 0 for shard in shards), "Empty work partition")
        b = np.sort(np.concatenate(shards), order=["cell", "q"])
        for key in ("cell", "q"):
            np.testing.assert_array_equal(a[key], b[key])
        for key in a.dtype.names[2:]:
            near(a[key], b[key], 1.e-10, 1.e-8)
        active = a[a["chi"] > 0]
        sample = active[np.argmin(active["x"]**2+active["y"]**2)]
        label_defect = np.abs(labels(parallel, step)-reference).max()
        stress_defect = max(np.abs(a[k]-b[k]).max() for k in ("xx", "yy", "xy"))
        print(f"free MPI step {step}: {list(map(len,shards))} bulk samples, "
              f"section samples {[int(metrics(parallel,step,r)['samples'][0]) for r in (0,1)]}, "
              f"max label {label_defect:.3g}, DG {stress_defect:.3g}; "
              f"sample {tuple(sample[k] for k in ('cell','q','x','y','xx','yy','xy','p','compression','tau'))}")
        sample8 = metrics(serial, step)["central_section"]
        distributed = [m["central_section"] for m in (metrics(parallel,step,r) for r in (0,1)) if "central_section" in m]
        require(len(distributed)==1, "The central cell must have one owner")
        near(sample8, distributed[0], 1.e-10, 1.e-8)
        print("  actual Q8 x,y,DG xx,yy,xy,p,compression,shear",sample8)
        print("  ranges (age,slip,rate)", reference[:, 4:7].min(axis=0), reference[:, 4:7].max(axis=0))


def donor_bulk_ids(coordinates):
    """Recover permanent CellId/Q3 identities for the archived 4x4 box at level four."""
    # The archived full-field binary contains coordinates and no CellId column.
    # Its coarse cells use Z ordering; each refinement appends x_bit + 2*y_bit.
    scaled = (coordinates+.5)*64
    indices = np.floor(scaled).astype(int)
    require(np.all((indices>=0)&(indices<64)), "Donor point is outside the reference grid")
    unit = scaled-indices
    nodes = gauss(3)[0]
    nearest = abs(unit[:,:,None]-nodes).argmin(axis=2)
    near(unit, nodes[nearest], 2.e-14)
    keys = []
    for (x,y),(qx,qy) in zip(indices,nearest):
        digits = [((x>>bit)&1) + 2*((y>>bit)&1) for bit in reversed(range(6))]
        root = 4*digits[0]+digits[1]
        path = "".join(map(str,digits[2:]))
        keys.append((f"{root}_4:{path}", qx+3*qy))
    return keys


def donor_check(root, donor, cell_reference=None):
    geom = np.loadtxt(donor/"geometry.csv", skiprows=1, delimiter=",")
    for case in ("inclined", "inclined-q1"):
        output = root/f"{case}-64"/"output"
        near(labels(output, 0)[:, :3], geom, 2.e-15)
        for step in (0, 1):
            a = np.sort(bulk(output, step), order=["cell", "q"])
            t = np.genfromtxt(donor/f"traction_{step}_rank0.csv", delimiter=",", names=True, dtype=None, encoding="utf8")
            keys = {(cell, q): i for i, (cell, q) in enumerate(zip(a["cell"], a["q"]))}
            b = a[[keys[cell, q] for cell, q in zip(t["cell"], t["q"])]]
            near(b["x"], t["x"], 2.e-15)
            near(b["y"], t["y"], 2.e-15)
            near(b["rate"], .005, 0, 2*np.finfo(float).eps)
            df = np.fromfile(donor/f"fields_{step}_rank0.bin").reshape(-1, 15)
            donor_ids = donor_bulk_ids(df[:, :2])
            donor_rows = {key: i for i,key in enumerate(donor_ids)}
            require(len(keys)==len(a)==len(donor_rows)==len(df) and keys.keys()==donor_rows.keys(),
                    "Donor and target full (cell,q) universes differ")
            ordered = [donor_rows[key] for key in zip(a["cell"],a["q"])]
            near(np.column_stack((a["x"],a["y"])),df[ordered,:2],2.e-15)
            if case=="inclined-q1":
                selected=np.zeros(len(a), dtype=bool)
                selected[[keys[cell,q] for cell,q in zip(t["cell"],t["q"])]]=True
                require(selected.sum()==len(t), "Duplicate donor traction identity")
                # The archive emits positive-slip traction points. Classify
                # zero-phase associations geometrically on the complement.
                xy=np.column_stack((a["x"],a["y"]))-geom[0,1:3]
                direction=(geom[-1,1:3]-geom[0,1:3])
                length=np.linalg.norm(direction)
                tangent=direction/length
                along=xy@tangent
                eta=xy@np.array([-tangent[1],tangent[0]])
                width=np.loadtxt(output.parent/"profile.txt")[-1,0]
                admitted=(along>=-1.e-14)&(along<=length+1.e-14)&(abs(eta)<=width)
                np.testing.assert_array_equal(a["associated"],admitted)
                require(b["associated"].all(), "Donor stations outside target support")
                np.testing.assert_array_equal(b["seg"],t["segment"])
                near(b["xi"],t["xi"],2.e-13)
                for key in ("slip_xx","slip_yy","slip_xy"):
                    near(a[key][~selected],0.,0.)
                kappa=-1.e20*np.expm1(-1.e6*.1/1.e20)
                nx,ny=t["nx"],t["ny"]
                factor=-2*kappa*b["chi"]*b["rate"]
                for key,frame in zip(("slip_xx","slip_yy","slip_xy"),(ny*nx,-nx*ny,.5*(ny*ny-nx*nx))):
                    near(b[key],factor*frame,1.e-8)
                print(f"  Q1 universe {len(a)}, common {len(t)}, zero-load complement {(~selected).sum()},"
                      f" associated zero-phase {(admitted & ~selected).sum()}")
            df = df[[donor_rows[cell,q] for cell,q in zip(t["cell"],t["q"])]]
            nx, ny = t["nx"], t["ny"]
            tx, ty = ny, -nx
            # Hold the donor's actual gradient, incoming stress and mechanical
            # pressure fixed. Exchange only the production transfer chi*V.
            kappa = -1.e20*np.expm1(-1.e6*.1/1.e20)
            gxx, gxy, gyx, gyy = df[:, 5:9].T
            div3 = (gxx+gyy)/3
            source = b["chi"]*.005
            xx = df[:, 12]+2*kappa*(gxx-div3-source*tx*nx)
            yy = df[:, 13]+2*kappa*(gyy-div3-source*ty*ny)
            xy = df[:, 14]+kappa*(gxy+gyx-source*(tx*ny+ty*nx))
            stress = np.column_stack((xx, yy, xy))
            normal = t["p"]-(xx*nx*nx+yy*ny*ny+2*xy*nx*ny)
            tau = tx*(xx*nx+xy*ny)+ty*(xy*nx+yy*ny)
            ref_tau = tx*(df[:,9]*nx+df[:,11]*ny)+ty*(df[:,11]*nx+df[:,10]*ny)
            err = b["chi"]-t["chi"]
            print(f"donor {case} step {step} {len(b)} eligible: chi max/RMS {max(abs(err)):.6g}/{np.sqrt(np.mean(err**2)):.6g}, "
                  f"scale {max(t['chi']):.6g}; common-input stress {abs(stress-df[:,9:12]).max():.6g} Pa; "
                  f"normal {abs(normal-t['sigma']).max():.6g}, shear {abs(tau-ref_tau).max():.6g} Pa")
            if case == "inclined-q1":
                if cell_reference:
                    # The remote donor uses embedded quadrature across cell faces.
                    # Isolate that integration error by changing only its backend.
                    interval = np.genfromtxt(cell_reference/f"traction_{step}_rank0.csv", delimiter=",",
                                             names=True, dtype=None, encoding="utf8")
                    np.testing.assert_array_equal(interval["cell"], t["cell"])
                    np.testing.assert_array_equal(interval["q"], t["q"])
                    bound = np.max(abs(interval["chi"]-t["chi"]))
                    print(f"  donor backend spread {bound:.6g}, matched interval difference "
                          f"{np.max(abs(b['chi']-interval['chi'])):.6g} 1/m")
                    # The donor interval panels use 1e-10 integration tolerance.
                    near(b["chi"], interval["chi"], 1.e-9, 1.e-9)
                    require(max(abs(err)) <= bound+2.e-8, "Q1 discrepancy exceeds independent quadrature spread")
            # In-plane slip contributes zero normal stress and -kappa*chi*V
            # to shear. Test both signs with the actual archived gradients.
            # The historical postprocessor applies 2*kappa*sym(grad) directly.
            # The target carries the three-dimensional plane-strain deviator.
            # Its trace term shifts normal traction but cancels in shear.
            normal_shift = 2*kappa*div3
            print(f"  donor full-strain / target plane-strain normal shift {max(abs(normal_shift)):.6g} Pa; "
                  f"common-convention normal defect {max(abs(normal-normal_shift-t['sigma'])):.4g} Pa")
            near(normal-normal_shift, t["sigma"], 1.e-8)
            near(tau-ref_tau, -kappa*.005*err, 1.e-8)


def plot(root, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(6.8, 6.6), sharex=True, constrained_layout=True)
    for case, legend in (("vertical", "0.15625 m"), ("narrow", "0.078125 m"), ("translated", "0.15625 m, translated")):
        data = np.loadtxt(root/f"{case}-64/output/transect-0-0.txt", skiprows=1)
        data = data[data[:, 0].argsort()]
        for ax, col in zip(axes, [3, 4, 5]):
            ax.plot(data[:, 0], data[:, col], label=legend, lw=1.6)
    for ax, name in zip(axes, ["Phase", "Degradation weight h", "Slip shear rate (1/s)"]):
        ax.set_ylabel(name)
        ax.grid(alpha=.22)
    axes[0].legend(title="Length scale", fontsize=9)
    axes[-1].set_xlabel("Horizontal position at y = 0 (m)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--donor", type=Path, help="Historical native-history-reference output tree")
    parser.add_argument("--donor-cell", type=Path, help="Q1 donor with cell-interval integration")
    parser.add_argument("--retry", action="store_true", help="Check the localized free-32-retry control")
    parser.add_argument("--history", type=Path, help="Export the free run's accepted label history")
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    if bool(args.donor) != bool(args.donor_cell):
        parser.error("Supply the archived donor and the cell-interval replay together to check the quadrature difference.")
    from moments import check_moments
    for case in ("vertical", "narrow", "translated", "inclined", "inclined-fit", "inclined-q1", "free"):
        grids = [64] if case == "inclined-q1" else [32, 64]
        for grid in grids:
            check_case(args.root,case,grid)
        if case in ("inclined","inclined-fit","inclined-q1"):
            for grid in grids:
                actual, reference, scale = np.loadtxt(args.root/f"{case}-{grid}/output/rhs-action.txt")
                near(actual,reference,512*np.finfo(float).eps*scale)
            defects = [check_moments(args.root/f"{case}-{grid}",grid,1,.1) for grid in grids]
            require(len(defects)==1 or defects[1]<defects[0] or max(defects)<=.01, "Cut moments failed refinement")
            print(f"  signed Q3 load on {case}: {actual:.12g} vs {reference:.12g}, scale {scale:.6g}")
        if case=="free":
            for step in (1,2):
                defects=[check_moments(args.root/f"free-{grid}",grid,step,.0125) for grid in grids]
                require(defects[1]<defects[0] or max(defects)<=.01,"Free cut moments failed refinement")
    for grid in (32, 64):
        wide, narrow = [np.loadtxt(args.root/f"{c}-{grid}/output/transect-0-0.txt",skiprows=1)
                        for c in ("vertical","narrow")]
        require(max(narrow[:,5]) > max(wide[:,5]), "Narrower phase failed to concentrate the returned shear")
    rank_check(args.root)
    if args.history:
        history = np.vstack([labels(args.root/"free-32/output", step) for step in range(3)])
        np.savetxt(args.history, history, fmt=["%d"]+["%.17g"]*7, delimiter=",", comments="",
                   header="label_id,x_m,y_m,I_h_m,age_s,slip_m,rate_m_per_s,time_s")
    if args.retry:
        clean, retry = args.root/"free-32/output", args.root/"free-32-retry/output"
        require((retry.parent/"run.log").read_text().count("Repeating the current time step") == 1,
                "Expected one rejected long step")
        require(len(list(retry.glob("labels-*-0.txt"))) == 3, "Rejected state was published")
        for step in range(3):
            near(labels(clean, step), labels(retry, step), 1.e-10, 1.e-8)
            a, b = bulk(clean, step), bulk(retry, step)
            for key in ("q", "cell", "phi"):
                np.testing.assert_array_equal(a[key], b[key])
            for key in a.dtype.names[2:]:
                near(a[key], b[key], 1.e-10, 1.e-8)
            print(f"free retry {step}: label {np.max(abs(labels(clean,step)-labels(retry,step))):.3g}, "
                  f"bulk {max(np.max(abs(a[key]-b[key])) for key in a.dtype.names[2:]):.3g}")
    if args.plot:
        plot(args.root, args.plot)
    if args.donor:
        donor_check(args.root, args.donor, args.donor_cell)


if __name__ == "__main__":
    main()
