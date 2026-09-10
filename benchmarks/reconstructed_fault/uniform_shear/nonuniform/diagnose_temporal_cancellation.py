#!/usr/bin/env python3
"""Bounded saved-state weak-balance decomposition; no fitted acceptance rule."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_convergence import samples

base = Path(__file__).resolve().parent/"domain-convergence"
destination = base/"temporal-cancellation.json"
previous = json.loads((base/"temporal-partial-review.json").read_text())
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--completed", action="store_true")
args = parser.parse_args()
if args.completed:
    destination = base.parent/"domain-convergence-completion/temporal-cancellation.json"
    for case in previous["cases"][1:]:
        directory = destination.parent/case["name"]
        report = json.loads(directory.with_name(directory.name+"-measurements").joinpath("report.json").read_text())
        assert report["completed_to_2_seconds"] and report["allowances_pass"]
        case["directory"] = str(directory)
        case["accepted_times_s"] = [r["time_s"] for r in report["steps"]]
names = ("q", "cohesive", "friction", "damping", "F", "V", "Theta", "C", "slip")
cases = []
kinematics = []
for case in previous["cases"]:
    directory = Path(case.get("directory", base/case["name"]))
    states = {}
    history = {}
    slip = None
    for k, t in enumerate(case["accepted_times_s"]):
        read = lambda stem: np.atleast_1d(np.genfromtxt(directory/f"{stem}_{k}.csv", names=True, delimiter=","))
        metadata, surface, weak = read("time"), read("surface"), read("surface_weak")
        assert metadata[0]["time"] == t
        s = np.r_[0., np.cumsum(np.hypot(np.diff(surface["x"]), np.diff(surface["y"])))]
        mass = np.diag(weak["Mdiag"])+np.diag(weak["Moff"][:-1],1)+np.diag(weak["Moff"][:-1],-1)
        loads = np.column_stack([weak[f] for f in ("q", "C", "friction", "damping", "F")])
        fields = np.linalg.solve(mass, loads)
        assert np.max(np.abs(loads[:,0]-loads[:,1:4].sum(axis=1)-loads[:,4])) < 1e-10
        if k == 0: slip = np.zeros(len(s))
        else: slip += metadata[0]["dt"]*surface["V"]
        history[t] = dict(dt=float(metadata[0]["dt"]), V=surface["V"],
                          Ih=surface["Ih"], C=surface["C"])
        fields = np.column_stack([fields, surface["V"], surface["Theta"], surface["C"], slip])
        states[t] = (s, fields)
    cases.append(states)
    kinematics.append(history)

rows = []
for pair in range(2):
    for t in sorted(set(cases[pair]) & set(cases[pair+1])):
        if t == 0: continue
        (sc,c), (sf,f) = cases[pair][t], cases[pair+1][t]
        x,w = samples(np.unique(np.r_[sc,sf,.0625,.1875]))
        delta = np.column_stack([np.interp(x,sc,c[:,j])-np.interp(x,sf,f[:,j]) for j in range(len(names))])
        mean = np.average(delta,axis=0,weights=w)
        anomaly = delta-mean
        inner = lambda a,b: float(np.average(a*b,weights=w))
        rms = lambda a: np.sqrt(inner(a,a))
        norms = {name:float(rms(anomaly[:,j])) for j,name in enumerate(names)}
        closure = anomaly[:,0]-anomaly[:,1:4].sum(axis=1)-anomaly[:,4]
        terms = anomaly[:,1:4]
        gram = np.array([[inner(terms[:,i],terms[:,j]) for j in range(3)] for i in range(3)])
        correlations = gram/np.sqrt(np.outer(np.diag(gram),np.diag(gram)))
        controls = np.array([0.,.125,.25])
        signed = {name:(np.interp(controls,sc,c[:,j])-np.interp(controls,sf,f[:,j])-mean[j]).tolist()
                  for j,name in enumerate(names)}
        rows.append(dict(pair=previous["cases"][pair]["name"]+"-"+previous["cases"][pair+1]["name"],
                         time_s=t, anomaly_rms=norms, signed_left_center_right=signed,
                         component_gram_Pa2=gram.tolist(), component_correlations=correlations.tolist(),
                         sum_component_norms_Pa=float(sum(norms[k] for k in ("cohesive","friction","damping"))),
                         friction_plus_damping_rms_Pa=float(rms(anomaly[:,2]+anomaly[:,3])),
                         balance_closure_rms_Pa=float(rms(closure)),
                         mean_differences=dict(zip(names,mean.tolist()))))
        # Exact temporal-accounting identities driven by accepted V, not an
        # independent per-vertex mechanical reference. Separate endpoint
        # solution differences from the chosen piecewise-constant time loads.
        if t in (.5,1.,1.5,2.):
            coarse, fine = kinematics[pair], kinematics[pair+1]
            accounting = {}
            for field in ("slip", "C"):
                endpoint = np.zeros(len(sc))
                time_load = np.zeros(len(sc))
                for end in sorted(time for time in coarse if 0 < time <= t):
                    dt = coarse[end]["dt"]
                    inside = sorted(time for time in fine if end-dt < time <= end)
                    assert abs(sum(fine[time]["dt"] for time in inside)-dt) < 1e-12
                    if field == "slip":
                        weight = dt
                        subweights = [fine[time]["dt"] for time in inside]
                    else:
                        # eta=1e8 Pa s, eta/G=100 s, and fixed full I_h.
                        weight = np.exp(-(t-end)/100)*(-1e8*np.expm1(-dt/100))/coarse[end]["Ih"]
                        subweights = [np.exp(-(t-time)/100)*(-1e8*np.expm1(-fine[time]["dt"]/100))/fine[time]["Ih"]
                                      for time in inside]
                    endpoint += weight*(coarse[end]["V"]-fine[end]["V"])
                    time_load += weight*fine[end]["V"]-sum(wi*fine[time]["V"] for wi,time in zip(subweights,inside))
                ep,tl = np.interp(x,sc,endpoint),np.interp(x,sc,time_load)
                ep -= np.average(ep,weights=w); tl -= np.average(tl,weights=w)
                actual = anomaly[:,names.index(field)]
                accounting[field] = dict(endpoint_rms=float(rms(ep)), time_load_rms=float(rms(tl)),
                    correlation=inner(ep,tl)/(rms(ep)*rms(tl)), reconstructed_rms=float(rms(ep+tl)),
                    measured_rms=float(rms(actual)), remainder_rms=float(rms(actual-ep-tl)))
            rows[-1]["accepted_V_accounting"] = accounting

report = dict(scope="accepted saved times only; decomposition, not a convergence pass",
              component_order=["cohesive","friction","damping"], rows=rows)
destination.write_text(json.dumps(report,indent=2)+"\n")
for row in rows:
    if row["time_s"] in (.5,1.,1.5,2.):
        print(json.dumps(row))
