"""Compare accepted granite strip rates, elastic loading and connected phase contours."""
import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path
import re
import sys

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"growing_fault"/"timestep"))
from frames import field_names, read, shards
from initial_field import basis

H, NX, NY = .03125, 48, 24
SEED = 1.125
PRESTRESS = 52500004.2031734


def roots(row, level):
    """The level partitions one quadratic Bernstein row into constant-sign intervals."""
    a, b, c = row-level
    # Split at the derivative zero before bracketing either root.
    boundaries = [0., 1.]
    if a+c != 2*b:
        turn = (a-b)/(a-2*b+c)
        if 0 < turn < 1:
            boundaries.insert(1, turn)
    found = set(boundaries)
    def value(x):
        return float(row @ basis(x) - level)
    for lo, hi in zip(boundaries, boundaries[1:]):
        if value(lo)*value(hi) < 0:
            found.add(brentq(value, lo, hi, xtol=5e-16))
    return sorted(found)


@lru_cache(maxsize=512)
def phase_metrics(key):
    patch = np.frombuffer(key).reshape(NY, NX, 3, 3)
    result = {}
    for level, name in ((.01, "001"), (.05, "005")):
        # Walk the two one-sided DG traces from the seed at the left wall.
        # A level crossing in either trace supplies a connected lower bound.
        tips = []
        for edge in (patch[NY//2-1, :, -1], patch[NY//2, :, 0]):
            for col, row in enumerate(edge):
                if min(row) >= level:
                    continue
                cuts = roots(row, level)
                exits = [a for a,b in zip(cuts,cuts[1:]) if row @ basis((a+b)/2) < level]
                if exits:
                    tips.append(H*(col+exits[0]))
                    break
        if len(tips) != 2:
            raise ValueError("Midline contours must both leave the seed inside this box")
        tip = max(tips)
        # At any y the patch is a convex combination of three x-polynomials.
        # Bound the furthest possible superlevel point over ALL y and cells.
        # Equality certifies this midline root is also the forward 2-D contour.
        candidates = np.nonzero(np.max(patch, axis=(0,2,3)) >= level)[0]
        upper = 0.
        for col in candidates[::-1]:
            if H*(col+1) < max(tip,upper):
                break
            for row in patch[:,col].reshape(-1,3):
                if max(row) < level:
                    continue
                cuts = roots(row, level)
                for a,b in zip(cuts,cuts[1:]):
                    if row @ basis((a+b)/2) >= level:
                        upper = max(upper, H*(col+b))
                for x in cuts:
                    if row @ basis(x) >= level:
                        upper = max(upper, H*(col+x))
        if abs(upper-tip) > 1e-11:
            raise ValueError(f"Connected tip {tip} and global upper bound {upper} disagree at {level}")
        result[f"tip_{name}_m"] = tip
    result["max_abs_phase_coefficient_ahead"] = np.max(abs(patch[:,round(SEED/H):]))
    return result


def full_frame(folder, step, expected_wall):
    tag = f"{step:04}"
    points = shards(folder, f"points-{tag}", ("cell","q"))
    fields = shards(folder, f"fields-{tag}", ("cell","component","local"))
    if not all(np.isfinite(v).all() for t in (points,fields) for k,v in t.items() if k != "cell"):
        raise ValueError(f"Nonfinite field in {folder}, step {step}")
    # The native quadrature is ordered by physical cell. Sort the FE coefficients
    # by the same CellId and the element's local index before reconstruction.
    cells, first, counts = np.unique(points["cell"], return_index=True, return_counts=True)
    assert len(cells) == NX*NY and np.all(counts == 36)
    ix, iy = (np.floor(points[k][first]/H).astype(int) for k in ("x","y"))
    assert len(set(zip(ix,iy))) == NX*NY
    def component(name):
        subset = fields["component"] == field_names(folder).index(name)
        np.testing.assert_array_equal(fields["cell"][subset], np.repeat(cells,9))
        np.testing.assert_array_equal(fields["local"][subset], np.tile(np.arange(9),len(cells)))
        return fields["value"][subset].reshape(-1,9)
    patch = np.empty((NY,NX,3,3))
    local = component("phase_field").reshape(-1,3,3)
    patch[iy,ix] = local
    u = points["x"].reshape(-1,36)/H-ix[:,None]
    v = points["y"].reshape(-1,36)/H-iy[:,None]
    sampled = np.einsum("cqi,cij,cqj->cq", basis(v), local, basis(u))
    np.testing.assert_allclose(sampled.ravel(),points["phi"],rtol=0,atol=1e-12)
    # Q2 nodes at the two wall vertices and the middle of that face.
    if step:
        for k, speed in (("velocity_x",expected_wall), ("velocity_y",0.)):
            velocity = component(k)
            for j, nodes, sign in ((0,[0,1,6],-1),(NY-1,[2,3,7],1)):
                np.testing.assert_allclose(velocity[iy==j][:,nodes], sign*speed, rtol=0, atol=1e-14)
    stress = points["tauxy"]-PRESTRESS
    if step == 0:
        np.testing.assert_array_equal(stress, 0.)
    w = points["weight"]
    np.testing.assert_allclose(w.sum(), 1.125, atol=1e-12, rtol=0)
    intact = points["x"] >= SEED
    stress_data = dict(mean_delta_xy_Pa=np.average(stress,weights=w),
                       min_delta_xy_Pa=min(stress),max_delta_xy_Pa=max(stress),
                       mean_ahead_delta_xy_Pa=np.average(stress[intact],weights=w[intact]),
                       max_abs_native_phase_ahead=max(abs(points["phi"][intact])))
    strips = read([folder/f"sections-{tag}.csv"],("section",))
    if step:
        balance = read([folder/f"balance-{tag}.csv"])
        t, work = balance["time"].item(), balance["Wext"].item()
    else:
        t, work = 0., 0.
    return t, strips, patch, dict(interval_wall_work_J_m=work, **stress_data)


def compact_frame(folder, step):
    tag = f"{step:04}"
    strips = read([folder/f"granite-strips-{tag}.csv"],("section",))
    t = strips["time_s"][0]
    assert np.all(strips["time_s"] == t)
    patch = np.zeros((NY,NX,3,3))
    seen = set()
    for rank in (0,1):
        path = folder/f"granite-phase-{tag}-r{rank}.csv"
        with path.open() as stream:
            for row in csv.DictReader(stream):
                x, y, hx, hy = (float(row[k]) for k in ("x0","y0","hx","hy"))
                i, j = round(x/H), round(y/H)
                np.testing.assert_allclose([x,y,hx,hy],[i*H,j*H,H,H],atol=1e-13,rtol=0)
                assert (i,j) not in seen and 0 <= i < NX and 0 <= j < NY
                seen.add((i,j))
                patch[j,i] = np.array([float(row[f"c{k}"]) for k in range(9)]).reshape(3,3)
    if not seen or not np.isfinite(patch).all():
        raise ValueError("Missing or nonfinite compact phase")
    return t, strips, patch


def check_matched_parameters(hold, drive):
    def values(node, path=()):
        for key, child in node.items():
            if isinstance(child, dict):
                if "value" in child:
                    yield (*path,key), child["value"]
                else:
                    yield from values(child, (*path,key))
    a,b = (dict(values(json.loads((folder/"parameters.json").read_text()))) for folder in (hold,drive))
    wall = ("Boundary velocity model","Function","Function expression")
    differences = {k for k in a if a[k] != b[k]}
    if a.keys() != b.keys() or differences != {("Output directory",),wall}:
        raise ValueError(f"Unexpected difference in resolved startup parameters: {differences}")


def physical_rows(case, folder, continuation=None):
    previous = None
    previous_t = work = 0.
    rows = []
    times = np.loadtxt(folder/"statistics",ndmin=2)
    assert list(times[:,0]) == list(range(9))
    steps = [(s,folder,False) for s in range(9)]
    if continuation:
        tags = sorted(int(p.stem.split("-")[-1]) for p in continuation.glob("granite-strips-*.csv"))
        assert tags and tags == list(range(9,tags[-1]+1))
        resumed_clocks = np.loadtxt(continuation/"statistics", ndmin=2)
        np.testing.assert_array_equal(resumed_clocks[:,0], np.arange(tags[-1]+1))
        np.testing.assert_array_equal(times[:,:3], resumed_clocks[:9,:3])
        steps.extend((s,continuation,True) for s in tags)
    for step, directory, compact in steps:
        extra = {}
        if compact:
            t, strips, patch = compact_frame(directory,step)
            np.testing.assert_allclose([t, t-previous_t], resumed_clocks[step,1:3], rtol=0, atol=1e-15)
            if step == 9:
                tref, ref, full, extra = full_frame(directory,step,.0005)
                assert t == tref
                np.testing.assert_array_equal(patch,full)
                for name in ("section","xi","formed","V","slip"):
                    np.testing.assert_array_equal(strips[name],ref[name])
        else:
            t, strips, patch, extra = full_frame(folder,step,.0005 if case == "drive" else 0.)
            np.testing.assert_allclose(t,times[step,1],rtol=0,atol=1e-15)
        phase = phase_metrics(patch.tobytes()).copy()
        if not step:
            start = phase.copy()
            seed = patch.copy()
        for label in ("001","005"):
            phase[f"advance_{label}_m"] = phase[f"tip_{label}_m"]-start[f"tip_{label}_m"]
        patch_change = np.max(abs(patch-seed),axis=(2,3))
        changed = np.argwhere(patch_change != 0.)
        phase["changed_phase_cells"] = len(changed)
        phase["max_abs_phase_coefficient_change"] = patch_change.max()
        if len(changed):
            phase["forward_edge_last_changed_cell_m"] = H*(1+changed[:,1].max())
        np.testing.assert_array_equal(strips["section"],np.arange(NX))
        np.testing.assert_array_equal(strips["xi"],H*(.5+np.arange(NX)))
        assert all(np.isfinite(v).all() for v in strips.values())
        x, rates, slips = (strips[k] for k in ("xi","V","slip"))
        formed = strips["formed"] > 0
        # Fixed material station just behind the original seed end. If a new
        # strip forms, retain this observable as well as the advancing tip ID.
        near = np.where(x < SEED)[0][-1]
        peak, peak_slip = np.argmax(abs(rates)), np.argmax(abs(slips))
        result = dict(case=case,step=step,time_s=t,**phase,**extra,
                      formed=int(sum(formed)),formed_ahead=int(sum(formed & (x>=SEED))),
                      tip_station=int(near),tip_station_x_m=x[near],tip_slip_m=slips[near],
                      max_abs_slip_m=abs(slips[peak_slip]),signed_strip_slip_m2=np.sum(slips)*H)
        if step:
            dt = t-previous_t
            assert dt > 0
            np.testing.assert_allclose(slips-previous["slip"],dt*rates,rtol=2e-11,atol=1e-22)
            result.update(midpoint_s=previous_t+dt/2,tip_V_m_s=rates[near],
                          peak_abs_V_m_s=abs(rates[peak]),peak_V_id=int(peak),peak_V_x_m=x[peak])
        if "interval_wall_work_J_m" in extra:
            work += extra["interval_wall_work_J_m"]
            result["wall_work_J_m"] = work
        rows.append(result)
        previous,previous_t = strips,t
    return rows


def write(rows,path):
    with path.open("w") as f:
        w = csv.DictWriter(f,list(dict.fromkeys(k for row in rows for k in row)),lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def step_times(files):
    times = []
    for f in files:
        start = None
        for line in f.read_text().splitlines():
            clock = float(line.split(maxsplit=1)[0])
            step = re.search(r"\*\*\* Timestep (\d+):",line)
            if step:
                start, n = clock, int(step[1])
            if start is not None and ("native section and point history" in line or "accepted strip rates, slips and phase" in line):
                if n > 0:
                    times.append(dict(step=n, elapsed_s=clock-start, measurement="timestep start to accepted output"))
                start = None
    return times


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("hold",type=Path)
    ap.add_argument("drive",type=Path)
    ap.add_argument("destination",type=Path)
    ap.add_argument("--continuation",type=Path)
    ap.add_argument("--drive-wall-logs",nargs="+",type=Path)
    ap.add_argument("--hold-wall-log",type=Path)
    opt = ap.parse_args()
    opt.destination.mkdir(parents=True,exist_ok=True)
    check_matched_parameters(opt.hold,opt.drive)
    hold = physical_rows("hold",opt.hold)
    drive = physical_rows("drive",opt.drive,opt.continuation)
    for h,d in zip(hold,drive):
        assert h["time_s"] == d["time_s"]
    initial_phase = []
    for case in (opt.hold,opt.drive):
        f = shards(case, "fields-0000", ("cell","component","local"))
        selected = f["component"] == field_names(case).index("phase_field")
        initial_phase.append(f["value"][selected])
    np.testing.assert_array_equal(*initial_phase)
    for tag in ("points-0000-r0.csv","points-0000-r1.csv","sections-0000.csv"):
        if (opt.hold/tag).read_bytes() != (opt.drive/tag).read_bytes():
            raise ValueError(f"Different native parent in {tag}")
    write(hold+drive,opt.destination/"trajectory.csv")
    if opt.drive_wall_logs:
        costs = step_times(opt.drive_wall_logs)
        # A signalled launcher can lose buffered timestamps even after a rank
        # has closed its accepted frame. Use adjacent original output mtimes
        # for those intervals and identify the timing basis in the cost CSV.
        observed = {r["step"] for r in costs}
        if opt.continuation:
            for row in drive:
                n = row["step"]
                if n > 9 and n not in observed:
                    def published(k):
                        return (opt.continuation/f"granite-strips-{k:04}.csv").stat().st_mtime_ns
                    duration_s = (published(n)-published(n-1))*1e-9
                    if duration_s <= 0:
                        raise ValueError(f"Unusable output timestamps for step {n}")
                    costs.append(dict(step=n,elapsed_s=duration_s,measurement="between compact output mtimes"))
        costs.sort(key=lambda row: row["step"])
        assert [r["step"] for r in costs] == [r["step"] for r in drive[1:]]
        write(costs,opt.destination/"drive-cost.csv")
    if opt.hold_wall_log:
        write(step_times([opt.hold_wall_log]),opt.destination/"hold-cost.csv")


if __name__ == "__main__":
    main()
