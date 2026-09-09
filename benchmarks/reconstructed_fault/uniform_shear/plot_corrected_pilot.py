#!/usr/bin/env python3
"""Export the accepted initial mechanical state, not the failed real timestep."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from vtkmodules.util.vtkConstants import VTK_LINE, VTK_VERTEX

from analyze import h
from diagnostics.visualize import write_vtu


def main():
    base = Path(__file__).resolve().parent/"diagnostics"
    output = base/"mechanical-pilot-view"
    output.mkdir(exist_ok=True)
    source = base/"mechanical-pilot"
    read = lambda name: np.atleast_1d(np.genfromtxt(source/f"{name}_0.csv",delimiter=",",names=True))
    times, bulk, surface, particles, nodes = [read(n) for n in ("time","bulk","surface","particles","phase")]
    if times[0]["time"] != 0 or not (source/"surface_0.csv").exists():
        raise ValueError("Missing accepted initial state")
    analysis = json.loads((base/"corrected-pilot-analysis.json").read_text())
    step = analysis["steps"][0]
    q = bulk["kappa"]*(bulk["ux_y"]+bulk["uy_x"]-bulk["chi"]*bulk["V"]-bulk["history"])+math.exp(-.02)*bulk["old_tau_xy"]
    for name, rows, fields in (
            ("accepted_t0_bulk_qp",bulk,{**{n:bulk[n] for n in ["ux","uy","p","phi","weight"]},"evaluated_tau_xy_Pa":q,
                                       "crack_shear_rate_s_inv":bulk["chi"]*bulk["V"]+bulk["history"]}),
            ("accepted_t0_particles",particles,{n:particles[n] for n in ["id","volume","H","tau_xy"]})):
        write_vtu(output/f"{name}.vtu",np.column_stack((rows["x"],rows["y"])),
                  [[i] for i in range(len(rows))],VTK_VERTEX,fields)
    write_vtu(output/"accepted_t0_fault.vtu",np.column_stack((surface["x"],surface["y"])),
              [[i,i+1] for i in range(len(surface)-1)],VTK_LINE,
              {n:surface[n] for n in ["V","Theta","C","Ih"]})

    # Approved finite-width velocity reference from the saved actual Q1 profile.
    # Keep q_old=1500 at timestep zero; do not use evaluated q as retained history.
    profile = np.unique(nodes[nodes["x"]==.125][["y","phi"]])
    profile = np.sort(profile,order="y")
    ys, phi = profile["y"], profile["phi"]
    gauss, weights = np.polynomial.legendre.leggauss(32)
    integral = analysis["profile"]["independent_Ih_m"]
    def accumulated(y):
        breaks = sorted(set([-.5,y]+[v for v in ys if -.5<v<y]))
        return sum((b-a)/2*np.dot(weights,h(np.interp((a+b)/2+(b-a)/2*gauss,ys,phi)))
                   for a,b in zip(breaks,breaks[1:]))/integral
    rows = []
    reference_by_y = {}
    for y in np.unique(bulk["y"]):
        mask = bulk["y"]==y
        reference = -step["U_m_s"]/2+(step["U_m_s"]-step["V_reference_m_s"])*(y+.5)+step["V_reference_m_s"]*accumulated(y)
        reference_by_y[y] = reference
        rows.append([y,np.mean(bulk["ux"][mask]),min(bulk["ux"][mask]),max(bulk["ux"][mask]),reference,
                     np.mean(q[mask]),min(q[mask]),max(q[mask]),step["evaluated_stress_reference_Pa"]])
    rows = np.array(rows)
    np.savetxt(output/"accepted_t0_transverse.csv",rows,delimiter=",",comments="",
               header="y_m,ux_mean_m_s,ux_min_m_s,ux_max_m_s,reference_ux_m_s,q_mean_Pa,q_min_Pa,q_max_Pa,reference_q_Pa")
    velocity_error = bulk["ux"]-np.array([reference_by_y[y] for y in bulk["y"]])
    surface_flux = float(np.trapezoid(surface["V"],surface["x"]))
    bulk_flux = float(np.dot(bulk["weight"],bulk["chi"]*bulk["V"]+bulk["history"]))
    summary = dict(time_s=0.,accepted_real_steps=0,
                   bulk_integrated_crack_rate_m2_s=bulk_flux,surface_integrated_V_m2_s=surface_flux,
                   actual_normalization_ratio=bulk_flux/surface_flux,
                   equivalent_rate_deficit_m_s=(surface_flux-bulk_flux)/.25,
                   history_rate_max_abs_s_inv=float(max(abs(bulk["history"]))),
                   velocity_reference_rms_error_m_s=float(np.sqrt(np.average(velocity_error**2,weights=bulk["weight"]))),
                   velocity_reference_max_error_m_s=float(max(abs(velocity_error))),
                   q_normal_range_Pa=[float(min(q)),float(max(q))],
                   q_max_along_x_range_Pa=float(max(np.ptp(q[bulk["y"]==y]) for y in np.unique(bulk["y"]))),
                   caveat="Initial mechanical comparison only; containment fails and the first real-step line search failed.")
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    fig,ax = plt.subplots(1,3,figsize=(14,4),layout="constrained")
    ax[0].plot(rows[:,0],rows[:,1],label="accepted t=0 FE QPs")
    ax[0].plot(rows[:,0],rows[:,4],"--",label="approved full-Ih reference")
    ax[0].set(xlabel="y [m]",ylabel="ux [m/s]"); ax[0].legend(fontsize=8)
    ax[1].plot(rows[:,0],rows[:,5],label="evaluated FE shear stress")
    ax[1].plot(rows[:,0],rows[:,8],"--",label="approved scalar q")
    ax[1].set(xlabel="y [m]",ylabel="q [Pa]"); ax[1].legend(fontsize=8)
    scan = np.genfromtxt(base/"support-resolution/width_scan.csv",names=True,delimiter=",")
    ax[2].semilogy(scan["half_width_m"][:-1],scan["max_omitted_fraction"][:-1],"o-")
    ax[2].axhline(1e-6,color="k",linestyle="--",label="unchanged containment target")
    ax[2].set(xlabel="association half-width [m]",ylabel="omitted fraction"); ax[2].legend(fontsize=8)
    fig.suptitle("Accepted initialization only — K1 gate remains unmet")
    fig.savefig(output/"initial_mechanics_and_width.png",dpi=160)
    plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
