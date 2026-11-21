"""Measure section formation, the native DG damage path and physical work on the matched 32x16 fixture."""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"first_free_turn"))
from analyze import RectBernstein, connectivity, contour_segments, gather, native, read
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("current", type=Path, help="Directory containing run.log and output/")
parser.add_argument("coulomb", type=Path)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(exist_ok=True, parents=True)
cases = {"Current": args.current, "Coulomb": args.coulomb}
accepted = {name: read(run/"output/bulk-accepted.csv") for name,run in cases.items()}
remote = {name: read(run/"output/tip-remote.csv") for name,run in cases.items()}
last = min(int(a.step[-1]) for a in accepted.values())
# Compare the longest common history before either prescribed integrity stop.
for r in remote.values():
    flagged = r.step[r.stop>0]
    if len(flagged):
        last = min(last,int(flagged[0])-1)
history, results, fields = [], {}, {}
reference_seed = None
fig, axes = plt.subplots(2,2,figsize=(10,6.8),layout="constrained")
colors = {"Current":"#145586","Coulomb":"#bd5429"}
for name,run in cases.items():
    path = run/"output"
    meta = json.loads((run/"command.json").read_text())
    a = accepted[name]
    a = a[a.step<=last]
    profile = read(path/"tip-profile.csv")
    initial = read(path/"tip-initial.csv")[0].fracture
    seed = gather(path,"tip-seed-fracture-r",["cell","i"])
    if reference_seed is None:
        reference_seed = seed.copy()
    else:
        np.testing.assert_array_equal(seed,reference_seed)
    previous = None
    births = {}
    rows = []
    for row in a:
        step = int(row.step)
        coeff = native(path,step)
        field = RectBernstein(coeff,.25,.25)
        planes = read(path/f"tip-planes-{step:04d}.csv")
        born = (planes.formed==1) & (planes.parent_formed==0)
        old = planes.parent_formed==1
        if previous is not None:
            np.testing.assert_array_equal(planes.parent_formed,previous.formed)
            np.testing.assert_array_equal(planes.angle[old],previous.angle[old])
            np.testing.assert_array_equal(planes.ds[old],previous.ds[old])
            np.testing.assert_array_equal(planes.parent_age,previous.age)
            np.testing.assert_array_equal(planes.parent_slip,previous.slip)
        np.testing.assert_allclose(planes.reference_dx,planes.ds*np.cos(planes.angle),rtol=1e-12)
        previous = planes
        gy = (np.arange(16)[:,None]+(np.polynomial.legendre.leggauss(6)[0]+1)/2).ravel()*.25
        for k in np.where(born)[0]:
            births[int(k)] = dict(step=step,x=float(planes.x[k]),angle_deg=float(np.rad2deg(planes.angle[k])),
                                  birth_phase=float(field.evaluate(planes.x[k],gy).max()))
        segments, bounds = contour_segments(field,64)
        vertices = np.concatenate(segments)
        right = vertices[vertices[:,0].argmax()]
        physical = row.Wext-row.energy_change-row.Db-row.Dc-row.Df
        rows.append(dict(rule=name,step=step,time=row.time,total_new=len(births),new=int(sum(born)),
                         right_x=right[0],right_y=right[1],added_fracture=row.fracture-initial,
                         physical_error=physical,ledger_error=row.ledger_error))
    ids = sorted(births)
    for k in ids:
        births[k]["end_phase"] = float(field.evaluate(planes.x[k],gy).max())
    history.extend(rows)
    color = colors[name]
    label = name
    for seg in segments:
        axes[0,0].plot(*seg.T,color=color,label=label,lw=1.6)
        label = None
    if name=="Current":
        for index,seg in enumerate(contour_segments(RectBernstein(coeff,.25,.25,seed.B),64)[0]):
            axes[0,0].plot(*seg.T,color="0.45",ls=":",lw=1.1,label="Equilibrated seed" if index==0 else None)
    right_ids = [k for k in ids if planes.x[k]>4]
    for appreciable in (True,False):
        group = [k for k in right_ids if (births[k]["end_phase"]>=1e-7)==appreciable]
        axes[0,1].plot(planes.x[group],np.rad2deg(planes.angle[group]),"o",ms=4,color=color,
                       markerfacecolor=color if appreciable else "none",label=name if appreciable else None)
    times = a.time*1000
    axes[1,0].plot(times,[r["added_fracture"] for r in rows],color=color,label=name)
    axes[1,1].plot(times,1e3*(np.array([r["right_x"] for r in rows])-6),color=color,label=name)
    seed_field = RectBernstein(coeff,.25,.25,seed.B)
    fields[name] = field
    nodes, gauss_weights = np.polynomial.legendre.leggauss(6)
    x = (.25*(np.arange(32)[:,None]+(1+nodes)/2)).ravel()
    y = (.25*(np.arange(16)[:,None]+(1+nodes)/2)).ravel()
    weights = np.outer(np.tile(gauss_weights,16),np.tile(gauss_weights,32))*(.25/2)**2
    damage = field.evaluate(x[None,:],y[:,None])-seed_field.evaluate(x[None,:],y[:,None])
    ahead = (x>=6)&(x<6.5)
    added = weights[:,ahead]*damage[:,ahead]
    power = read(path/"tip-power.csv")
    result = dict(exit_code=meta.get("exit_code"),wall_seconds=meta.get("seconds"),accepted_steps=int(accepted[name].step[-1]),
                  common_step=last,common_time=float(a.time[-1]),solver_seconds=float(profile.step_seconds[profile.step<=last].sum()),
                  new_sections=births,connectivity=connectivity(field),
                  ahead_added_phase_m2=float(added.sum()),
                  ahead_damage_centroid_y_m=float((added*y[:,None]).sum()/added.sum()) if added.sum()>0 else None,
                  max_power_adjoint_error=float(max(abs(power.power_adjoint_error[power.step<=last]))),
                  remote_phase=float(remote[name].max_phase[remote[name].step==last][0]),
                  integrity_trigger_steps=remote[name].step[remote[name].stop>0].tolist(),
                  right_front=[float(right[0]),float(right[1])],added_fracture=rows[-1]["added_fracture"],
                  signed_work_error=sum(r["physical_error"] for r in rows),absolute_work_error=sum(abs(r["physical_error"]) for r in rows),
                  max_ledger_error=float(np.max(abs(a.ledger_error))),max_mechanical=float(max(a.mechanical_residual)),
                  max_phase_kkt=float(max(a.phase_KKT)),executable=meta["binary_sha256"],checker=meta["checker_sha256"],
                  input_sha256=meta["input_sha256"])
    later = []
    for row in accepted[name][accepted[name].step>last]:
        step = int(row.step)
        poly = RectBernstein(native(path,step),.25,.25)
        seg, _ = contour_segments(poly,64)
        vertices = np.concatenate(seg)
        later.append(dict(step=step,time=float(row.time),right_front=vertices[vertices[:,0].argmax()].tolist(),
                          connectivity=connectivity(poly),remote_phase=float(remote[name].max_phase[remote[name].step==step][0])))
    result["unpaired_later_endpoints"] = later
    results[name] = result
difference = fields["Current"].evaluate(x[None,:],y[:,None])-fields["Coulomb"].evaluate(x[None,:],y[:,None])
results["phase_difference"] = dict(l2_m=float(np.sqrt(np.sum(weights*difference**2))),
                                   native_max=float(abs(difference).max()))
axes[0,0].set(xlim=(5.60,6.30),ylim=(1.82,2.18),aspect="equal",xlabel="x (m)",ylabel="y (m)",title=rf"Right tip at {a.time[-1]*1000:.0f} ms, $\phi=0.05$")
axes[0,1].set(xlabel="Right-hand section x (m)",ylabel="Material plane angle (degrees)",title=r"New sections (open if max $\phi<10^{-7}$)")
axes[1,0].set(xlabel="Time (ms)",ylabel="Added fracture energy (J/m)")
axes[1,1].set(xlabel="Time (ms)",ylabel="Right contour x − 6 m (mm)")
for ax in axes.flat:
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
fig.savefig(args.out/"turn-comparison.png",dpi=170)
(args.out/"summary.json").write_text(json.dumps(results,indent=2)+"\n")
with (args.out/"history.csv").open("w") as stream:
    writer = csv.DictWriter(stream,fieldnames=history[0].keys())
    writer.writeheader()
    writer.writerows(history)
