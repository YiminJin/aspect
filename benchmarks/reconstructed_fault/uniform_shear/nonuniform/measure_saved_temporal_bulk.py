#!/usr/bin/env python3
"""Missing same-mesh temporal bulk/normalization checks from accepted CSVs."""
import json
import math
from pathlib import Path
import resource
import time
import argparse
import numpy as np

start = time.monotonic()
parser = argparse.ArgumentParser()
parser.add_argument("--decompose-at",type=float)
parser.add_argument("--completed",action="store_true")
args = parser.parse_args()
base = Path(__file__).resolve().parent/"domain-convergence"
prior = json.loads((base/"temporal-partial-review.json").read_text())
if args.completed:
    for case in prior["cases"][1:]:
        directory=base.parent/"domain-convergence-completion"/case["name"]
        measurement=json.loads(directory.with_name(directory.name+"-measurements").joinpath("report.json").read_text())
        assert measurement["allowances_pass"] and measurement["completed_to_2_seconds"]
        case["directory"]=str(directory)
        case["accepted_times_s"]=[r["time_s"] for r in measurement["steps"]]
columns = ("x","y","weight","ux","uy","p","ux_y","uy_x","old_tau_xy","chi","V","history")

def bulk(directory, step):
    path = directory/f"bulk_{step}.csv"
    with path.open() as stream:
        header = stream.readline().strip().split(",")
        array = np.loadtxt(stream,delimiter=",",usecols=[header.index(k) for k in columns])
    assert np.isfinite(array).all()
    order = np.lexsort((array[:,1],array[:,0]))
    return {k:array[order,i] for i,k in enumerate(columns)}

def small(directory, stem, step):
    return np.atleast_1d(np.genfromtxt(directory/f"{stem}_{step}.csv",names=True,delimiter=","))

def norms(values, weights):
    return dict(rms=float(np.sqrt(np.average(values**2,weights=weights))),
                maximum=float(np.max(np.abs(values))),mean=float(np.average(values,weights=weights)))

report = dict(scope="accepted saved temporal states; no simulation or acceptance change",
              normalization=[], bulk_differences=[])
for case in ([] if args.decompose_at is not None or args.completed else prior["cases"][1:]):
    directory = base/case["name"]
    # Zero and first real step already have valid independent measurements.
    for step,t in list(enumerate(case["accepted_times_s"]))[2:]:
        data = bulk(directory,step)
        surface = small(directory,"surface",step)
        xs,index = np.unique(data["x"],return_inverse=True)
        w = data["weight"]
        integrated = data["chi"]*data["V"]+data["history"]
        column = np.bincount(index,weights=w*integrated)/np.bincount(index,weights=w)
        ratios = column/np.interp(xs,surface["x"],surface["V"])
        weights = np.r_[np.diff(surface["x"])/2,0]+np.r_[0,np.diff(surface["x"])/2]
        global_ratio = float(np.dot(w,integrated)/np.dot(weights,surface["V"]))
        error = float(max(abs(ratios-1)))
        assert error <= 1e-4 and abs(global_ratio-1) <= 1e-4
        report["normalization"].append(dict(case=case["name"],time_s=t,locations=len(xs),
                                            maximum_error=error,global_ratio=global_ratio))

for left,right in zip(prior["cases"],prior["cases"][1:]):
    common = sorted(set(left["accepted_times_s"]) & set(right["accepted_times_s"]))
    for t in common:
        if t == 0: continue # Byte-identical initial bulk exports are already verified.
        if args.decompose_at is not None and t != args.decompose_at: continue
        if args.completed:
            saved=json.loads((base/"temporal-saved-bulk.json").read_text())
            reused=[r for r in saved["bulk_differences"] if r["time_s"]==t and r["pair"]==left["name"]+"-"+right["name"]]
            if reused:
                report["bulk_differences"].extend(reused)
                continue
        states = []
        for case in (left,right):
            k = case["accepted_times_s"].index(t)
            directory = Path(case.get("directory",base/case["name"]))
            data = bulk(directory,k)
            dt = small(directory,"time",k)[0]["dt"]
            kappa = -1e8*math.expm1(-dt/100)
            data["tau_xy"] = kappa*(data["ux_y"]+data["uy_x"]-data["chi"]*data["V"]-data["history"])
            data["tau_xy"] += math.exp(-dt/100)*data["old_tau_xy"]
            if args.decompose_at is not None:
                data["strain"] = kappa*(data["ux_y"]+data["uy_x"])
                data["slip"] = -kappa*data["chi"]*data["V"]
                data["profile_history"] = -kappa*data["history"]
                data["old_stress"] = math.exp(-dt/100)*data["old_tau_xy"]
            states.append(data)
        c,f = states
        assert np.array_equal(c["x"],f["x"]) and np.array_equal(c["y"],f["y"])
        assert np.array_equal(c["weight"],f["weight"])
        y,index = np.unique(f["y"],return_inverse=True)
        w = f["weight"]
        sums = np.bincount(index,weights=w)
        fields = {}
        terms = ("strain","slip","profile_history","old_stress") if args.decompose_at is not None else ()
        anomalies = {}
        for name in ("ux","uy","p","tau_xy")+terms:
            delta = c[name]-f[name]
            mean_x = np.bincount(index,weights=w*delta)/sums
            fields[name] = dict(total=norms(delta,w),anomaly=norms(delta-mean_x[index],w))
            anomalies[name] = delta-mean_x[index]
        row = dict(pair=left["name"]+"-"+right["name"],time_s=t,fields=fields)
        if terms:
            gram = np.array([[np.average(anomalies[a]*anomalies[b],weights=w) for b in terms] for a in terms])
            row["decomposition"] = dict(order=terms,gram=gram.tolist(),
                closure_rms=norms(anomalies["tau_xy"]-sum(anomalies[a] for a in terms),w)["rms"])
        report["bulk_differences"].append(row)
report["wall_seconds"] = time.monotonic()-start
report["peak_rss_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
output = "temporal-bulk-decomposition.json" if args.decompose_at is not None else "temporal-saved-bulk.json"
destination=base/output
if args.completed:
    destination=base.parent/"domain-convergence-completion/temporal-bulk.json"
    report["scope"]="completed temporal sequence; validated saved common-time bulk comparisons reused"
destination.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
