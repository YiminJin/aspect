"""Measure a fresh or resumed granite trajectory on the recorded physical cells.

Give directories in time order. A compact-only segment needs the preceding
frame (and the initial frame for displacement). On a turned graph, obtain the
angle of every newly formed station from a matching full-history replay.
"""
import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

from contours import seed_front
from trajectory import Case, check_interval, read, bernstein, field_names

PRESTRESS = 52500004.2031734


def compact(folder, step, ranks):
    tag = f"{step:04}"
    strips = read([folder/f"granite-strips-{tag}.csv"], ("section",))
    time = strips["time_s"][0]
    if not all(np.isfinite(data).all() for data in strips.values()) or not np.all(strips["time_s"]==time):
        raise ValueError(f"Inconsistent compact strip data at {tag}")
    pattern = f"granite-phase-{tag}-r*.csv"
    paths = sorted(folder.glob(pattern))
    if sorted(int(path.stem.split("-r")[-1]) for path in paths)!=list(range(ranks)):
        raise ValueError(f"Incomplete phase shards at {tag}")
    cells = []
    for path in paths:
        with path.open() as stream:
            reader = csv.reader(stream)
            if next(reader)!=["x0","y0","hx","hy",*(f"c{i}" for i in range(9))]:
                raise ValueError(f"Unexpected compact phase header in {path}")
            for record in reader:
                values = np.asarray(record, dtype=float)
                if len(values)!=13 or not np.isfinite(values).all() or np.min(values[2:4])<=0:
                    raise ValueError(f"Invalid phase patch in {path}")
                cells.append((values[:2],values[2:4],values[4:].reshape(3,3)))
    if len({tuple(lower) for lower,_,_ in cells})!=len(cells):
        raise ValueError(f"Overlapping phase owners at step {step}")
    return time,strips,cells


def native(case, step):
    pts, fields, strips, ledger = case.load(step)
    cells = case.phase_patches(pts,fields)
    area = pts["weight"].sum()
    np.testing.assert_allclose(area,np.prod(case.size),rtol=0,atol=1e-12)
    if step==0:
        np.testing.assert_array_equal(pts["tauxy"],PRESTRESS)
    if step:
        for role,wall_speed in (("velocity_x",.0005),("velocity_y",0.)):
            velocity = fields["value"][fields["component"]==field_names(case.folder).index(role)].reshape(-1,9)
            for side,indices in ((0,[0,1,6]),(1,[2,3,7])):
                boundary = [i for i,(base,width,_) in enumerate(cells)
                            if abs(base[1]+side*width[1]-case.origin[1]-side*case.size[1])<1e-12]
                if not boundary:
                    raise ValueError("Missing wall nodes")
                np.testing.assert_allclose(velocity[boundary][:,indices],(2*side-1)*wall_speed,rtol=0,atol=1e-14)
    extra = dict(cells=len(cells), mean_delta_tauxy_Pa=np.average(pts["tauxy"]-PRESTRESS,weights=pts["weight"]),
                 interval_wall_J_m=ledger["Wext"].item() if ledger else 0.,
                 fracture_J_m=ledger["fracture"].item() if ledger else None,
                 phase_work_J_m=ledger["phase_work"].item() if ledger else None)
    return ledger["time"].item() if ledger else 0.,strips,cells,extra


def graph(case, strips, cells):
    stations, formed = strips["xi"],strips["formed"].astype(bool)
    near = np.flatnonzero(stations<case.seed_s)[-1]
    # Prior to any new formation the graph through the prescribed seed is
    # exactly straight; the original compact recorder did not export beta.
    if np.any(formed[near+1:]) and "beta" not in strips:
        raise ValueError("A station beyond the seed has formed. Supply its full replay with --native-replay.")
    tip = tracked = case.seed.copy()
    xi,arclength = [case.seed_s],[0.]
    contiguous = True
    last = near
    for k,s in enumerate(stations):
        if s<=case.seed_s:
            if s>=case.seed_s-case.ell:
                contiguous &= Case.trace(cells,case.seed+(s-case.seed_s)*case.t)>=.05
            continue
        if not formed[k]:
            break
        segment = (s-xi[-1])*(case.t+np.tan(strips["beta"][k])*case.n)
        tip = tip+segment
        xi.append(s)
        arclength.append(arclength[-1]+np.linalg.norm(segment))
        contiguous &= Case.trace(cells,tip)>=.05
        if contiguous:
            tracked,last = tip,k
    # Distances are signed arclength coordinates on the accepted material
    # graph relative to the prescribed seed tip. Beyond the last knot the
    # graph continues on its final direction, as in accepted_graph().
    def along(s):
        if s<=xi[0] or len(xi)==1:
            return s-xi[0]
        if s>xi[-1]:
            return arclength[-1]+(s-xi[-1])*(arclength[-1]-arclength[-2])/(xi[-1]-xi[-2])
        return np.interp(s,xi,arclength)
    behind = np.flatnonzero(formed & (stations<=stations[last]))
    return tip,tracked,arclength[-1],along,behind[-1] if len(behind) else None


def phase_summary(case,cells,samples):
    out = {}
    active = [c for c in cells if np.any(c[2])]
    for level,col in ((.01,"001"),(.05,"005")):
        x,y,upper,res = seed_front(active,case.origin,case.seed-case.t*case.ell,level,samples)
        out.update({f"x{col}_m":x,f"y{col}_m":y,f"global_upper{col}_m":upper,
                    f"sampling_diagonal{col}_m":res})
    maximum = bound = absolute_bound = 0.
    z = bernstein(np.linspace(0,1,samples+1))
    for base,width,patch in active:
        if base[0]+width[0]<=case.seed[0]+1e-13*width[0]:
            continue
        # Restrict each quadratic to the portion ahead of the original tip.
        t = np.clip((case.seed[0]-base[0])/width[0],0,1)
        right = np.column_stack((patch@bernstein(np.array([t])).T,
                                 (1-t)*patch[:,1]+t*patch[:,2],patch[:,2]))
        absolute_bound = max(absolute_bound, np.abs(right).max())
        bound = max(bound,right.max())
        maximum = max(maximum,(z@right@z.T).max())
    out.update(ahead_phi_sample_max=maximum,ahead_phi_upper=bound,ahead_phi_abs_bound=absolute_bound)
    return out


def patches_key(cells):
    rows = [np.r_[base,width,patch.ravel()] for base,width,patch in cells if np.any(patch)]
    rows.sort(key=lambda r: (r[0],r[1]))
    return hashlib.sha256(np.asarray(rows).tobytes()).digest()


def measure(folders, samples, ranks, native_replay, last_step):
    previous = previous_clock = initial = previous_id = last = cached_key = None
    wall_work = 0.
    rows = []
    for folder in folders:
        case = Case(folder)
        replay = Case(native_replay) if native_replay else None
        tags = {int(p.stem.split("-")[-1]) for pat in ("sections-*.csv","granite-strips-*.csv")
                for p in folder.glob(pat)}
        if not tags:
            raise ValueError(f"No accepted frames in {folder}")
        for step in sorted(tags):
            if last_step is not None and step>last_step:
                break
            if last is not None and step!=last+1:
                raise ValueError(f"Gap or overlap between step {last} and {step}")
            tag = f"{step:04}"
            extra = {}
            has_native = (folder/f"sections-{tag}.csv").is_file()
            if (folder/f"granite-strips-{tag}.csv").is_file():
                clock,strips,cells = compact(folder,step,ranks)
                full = case if has_native else replay
                if full and (full.folder/f"sections-{tag}.csv").is_file():
                    t,ref,native_cells,extra = native(full,step)
                    edge=np.min([width for _,width,_ in native_cells],axis=0)
                    def index(patches):
                        records={}
                        for lo,width,coefficients in patches:
                            if np.any(coefficients):
                                position=np.r_[(lo-case.origin)/edge,width/edge]
                                key=tuple(np.rint(position).astype(int))
                                np.testing.assert_allclose(position,key,rtol=0,atol=1e-10)
                                if key in records:
                                    raise ValueError("Duplicate cell in phase overlap")
                                records[key]=coefficients
                        return records
                    sparse,dense=index(cells),index(native_cells)
                    if sparse.keys()!=dense.keys():
                        raise ValueError("Different nonzero phase footprints in compact and native outputs")
                    for cell in sparse:
                        np.testing.assert_array_equal(sparse[cell],dense[cell])
                    if t!=clock:
                        raise ValueError("Full and compact frames are at different clocks")
                    for col in ("section","xi","formed","V","slip"):
                        np.testing.assert_array_equal(ref[col],strips[col])
                    strips = ref
            else:
                clock,strips,cells,extra = native(case,step)
            if last is None and (step!=0 or clock!=0):
                raise ValueError("Initial endpoint required for contour displacements")
            if previous is not None:
                # Compact frames have no stored age. Only compare an age
                # increment when both native measurements are available.
                shared = {k:v for k,v in strips.items() if k in previous}
                check_interval(case,previous,shared,clock-previous_clock)
            key = patches_key(cells)
            if key!=cached_key:
                phase = phase_summary(case,cells,samples)
                cached_key = key
            tip,tracked,length,along,next_id = graph(case,strips,cells)
            near = np.flatnonzero(strips["xi"]<case.seed_s)[-1]
            peak = np.argmax(abs(strips["V"]))
            h = min(max(w) for _,w,_ in cells)
            row = dict(step=step,time_s=clock,midpoint_s=(clock+previous_clock)/2 if step else None,
                       peak_m_s=abs(strips["V"][peak]),peak_id=peak,peak_xi_m=strips["xi"][peak],
                       fixed_id=near,fixed_V_m_s=strips["V"][near],fixed_slip_m=strips["slip"][near],
                       max_slip_Dc=max(abs(strips["slip"]))/case.dc,formed=int(sum(strips["formed"])),
                       previous_tip_id=previous_id,previous_tip_V_m_s=strips["V"][previous_id] if previous_id is not None else None,
                       formed_tip_x_m=tip[0],formed_tip_y_m=tip[1],formed_tip_arc_m=length,
                       tracked_tip_x_m=tracked[0],tracked_tip_y_m=tracked[1],
                       tracked_tip_arc_m=along((tracked-case.origin)@case.t),finest_edge_m=h,**phase,**extra)
            if "age" in strips:
                row["fixed_age_s"]=strips["age"][near]
                row["fixed_resistance_Pa"]=strips["force"][near]
            wall_work = wall_work+extra["interval_wall_J_m"] if wall_work is not None and "interval_wall_J_m" in extra else None
            row["wall_work_J_m"] = wall_work
            initial = initial or row.copy()
            for col in ("001","005"):
                xy = np.array([row[f"x{col}_m"],row[f"y{col}_m"]])
                xy0 = np.array([initial[f"x{col}_m"],initial[f"y{col}_m"]])
                dx = xy[0]-xy0[0]
                row[f"advance{col}_x_m"]=dx
                row[f"advance{col}_h"]=dx/h
                row[f"advance{col}_arc_m"]=along((xy-case.origin)@case.t)-((xy0-case.seed)@case.t)
            rows.append(row)
            previous,previous_clock,previous_id,last = strips,clock,next_id,step
    return rows


def write_csv(rows,path):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w") as stream:
        writer = csv.DictWriter(stream,columns,lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("folders",type=Path,nargs="+")
    p.add_argument("--csv",type=Path,required=True)
    p.add_argument("--ranks",type=int,required=True)
    p.add_argument("--samples",type=int,default=16)
    p.add_argument("--last-step",type=int)
    p.add_argument("--native-replay",type=Path)
    args=p.parse_args()
    write_csv(measure(args.folders,args.samples,args.ranks,args.native_replay,args.last_step),args.csv)
