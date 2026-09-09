#!/usr/bin/env python3
"""Cell-ID-resolved raw initial-stress audit; no stress smoothing or replacement."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from analyze import h
from reference import Histories, solve_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--cell-map",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    read = lambda name: np.atleast_1d(np.genfromtxt(args.directory/f"{name}_0.csv",delimiter=",",names=True))
    b, surface, particles, nodes = [read(n) for n in ["bulk","surface","particles","phase"]]
    with (args.cell_map or args.directory/"bulk_cell_ids_0.csv").open() as f:
        provenance = list(csv.DictReader(f))
    if len(provenance) != len(b):
        raise ValueError("QP provenance size mismatch")
    q = b["kappa"]*(b["ux_y"]+b["uy_x"]-b["chi"]*b["V"]-b["history"])+math.exp(-.02)*b["old_tau_xy"]
    global_mean = np.average(q,weights=b["weight"])
    g,w = np.polynomial.legendre.leggauss(3)
    expected = np.array([[x,y] for y in g for x in g])
    xs = np.unique(nodes["x"]); ys = np.unique(nodes["y"])
    mean_moments, cell_rows, particle_rows = [], [], []
    # Reconstruct the Q2 gradient at actual particle locations, not the stress
    # at Gauss points. Nonlinear h(phi_h) is reevaluated at the particle itself.
    def basis(z):
        return np.array([np.prod([(z-g[j])/(g[i]-g[j]) for j in range(3) if j!=i]) for i in range(3)])
    for row in range(0,len(b),9):
        a = b[row:row+9]; meta = provenance[row:row+9]
        if len({m["cell_id"] for m in meta})!=1 or [int(m["q"]) for m in meta]!=list(range(9)):
            raise ValueError("Not one actual cell with ordered 3x3 quadrature")
        if [int(m["row"]) for m in meta] != list(range(row,row+9)):
            raise ValueError("Incorrect QP row mapping")
        center = np.array([np.mean(a["x"]),np.mean(a["y"])])
        ix,iy = np.searchsorted(xs,center[0])-1,np.searchsorted(ys,center[1])-1
        lower, upper = np.array([xs[ix],ys[iy]]),np.array([xs[ix+1],ys[iy+1]])
        unit = 2*(np.column_stack((a["x"],a["y"]))-lower)/(upper-lower)-1
        if np.max(abs(unit-expected))>1e-12 or np.max(abs(a["weight"]/np.prod(upper-lower)-np.outer(w,w).ravel()/4))>1e-12:
            raise ValueError("Actual coordinates/weights do not match the cell's 3x3 Gauss rule")
        stress = q[row:row+9]; mean = np.average(stress,weights=a["weight"])
        error = stress-mean
        moments = [np.average(stress-global_mean,weights=a["weight"]),
                   np.average((stress-global_mean)*unit[:,0],weights=a["weight"]),
                   np.average((stress-global_mean)*unit[:,1],weights=a["weight"])]
        transverse = error.reshape(3,3).mean(axis=1)
        pattern_error = np.max(abs(transverse-transverse[1]*np.array([-.8,1.,-.8])))
        cell_rows.append([meta[0]["cell_id"],*[int(meta[0][f"v{i}"]) for i in range(4)],*center,mean,
                          *moments,*transverse,pattern_error])
        mean_moments.append(moments)
        selection = np.flatnonzero((particles["x"]>=lower[0])&(particles["x"]<upper[0])&
                                  (particles["y"]>=lower[1])&(particles["y"]<upper[1])&(particles["active"]==1))
        for p in selection:
            point = particles[p]
            unit_p = 2*(np.array([point["x"],point["y"]])-lower)/(upper-lower)-1
            shape = np.outer(basis(unit_p[1]),basis(unit_p[0])).ravel()
            gradient = shape@(a["ux_y"]+a["uy_x"])
            phi = shape@a["phi"]
            s,xi = int(point["segment"]),point["xi"]
            interpolate = lambda name: (1-xi)*surface[name][s]+xi*surface[name][s+1]
            V,Ih,C,theta = [interpolate(n) for n in ["V","Ih","C","Theta"]]
            kappa = float(a["kappa"][0]); beta=math.exp(-.02)
            stress_p = kappa*(gradient-float(h(phi))*V/Ih)+beta*point["tau_xy"]
            cohesive = kappa*V/Ih+beta*C
            mu = .025*math.asinh(V/(2e-5)*math.exp((.6+.013*math.log(theta*1e-5/.001))/.025))
            residual = stress_p-cohesive-1000*mu-1e5*V
            particle_rows.append([int(point["id"]),meta[0]["cell_id"],point["volume"],s,xi,phi,
                                  gradient,stress_p,cohesive,mu,residual])

    names="cell_id,v0,v1,v2,v3,x_center,y_center,q_mean_Pa,constant_moment_Pa,x_linear_moment_Pa,y_linear_moment_Pa,error_y_low_Pa,error_y_mid_Pa,error_y_high_Pa,gauss_585_pattern_error_Pa"
    with (args.output/"cell_stress_moments.csv").open("w") as f:
        writer=csv.writer(f);writer.writerow(names.split(","));writer.writerows(cell_rows)
    with (args.output/"particle_surface_traction.csv").open("w") as f:
        writer=csv.writer(f);writer.writerow("particle_id,cell_id,volume_m2,segment,xi,phi,engineering_strain_rate_s_inv,evaluated_tau_xy_Pa,evaluated_C_Pa,mu,surface_residual_Pa".split(","));writer.writerows(particle_rows)
    p = np.array([r[2:] for r in particle_rows],float)
    mass = np.zeros((len(surface),len(surface))); rhs=np.zeros(len(surface))
    for weight,s,xi,*_,residual in p:
        s=int(s);shape=np.array([1-xi,xi]);mass[s:s+2,s:s+2]+=weight*np.outer(shape,shape);rhs[s:s+2]+=weight*residual*shape
    strong=np.linalg.solve(mass,rhs)
    norm=math.sqrt(float(rhs@strong)/mass.sum())

    # Independent full-Ih and initial scalar reference for each resolution.
    center_profile=np.unique(nodes[nodes["x"]==xs[len(xs)//2]][["y","phi"]]);center_profile=np.sort(center_profile,order="y")
    gg,ww=np.polynomial.legendre.leggauss(32)
    integral=sum((bb-aa)/2*np.dot(ww,h(np.interp((aa+bb)/2+(bb-aa)/2*gg,center_profile["y"],center_profile["phi"]))) for aa,bb in zip(ys,ys[1:]))
    history=Histories(1500.,float(np.mean(surface["C"])),float(np.mean(surface["Theta"])))
    reference,_=solve_step(history,2.,1e-4,integral,integral,initial=True)
    with (args.output/"raw_stress_error_qps.csv").open("w") as f:
        writer=csv.writer(f)
        writer.writerow(["cell_id","q","x_m","y_m","weight_m2","evaluated_tau_xy_Pa","reference_q_Pa","raw_error_Pa"])
        for i,point in enumerate(b):
            writer.writerow([provenance[i]["cell_id"],provenance[i]["q"],point["x"],point["y"],point["weight"],q[i],reference["stress"],q[i]-reference["stress"]])
    summary=dict(cells=len(cell_rows),qps=len(b),active_particles=len(particle_rows),
                 raw_q_min_Pa=float(min(q)),raw_q_max_Pa=float(max(q)),q_mean_Pa=float(global_mean),
                 cell_mean_range_Pa=float(np.ptp([r[7] for r in cell_rows])),
                 max_abs_constant_moment_about_global_mean_Pa=float(np.max(abs(np.array(mean_moments)[:,0]))),
                 max_abs_linear_moment_Pa=float(np.max(abs(np.array(mean_moments)[:,1:]))),
                 max_585_pattern_error_Pa=float(max(r[-1] for r in cell_rows)),
                 particle_traction_mean_Pa=float(np.average(p[:,5],weights=p[:,0])),
                 particle_surface_residual_strong_rms_Pa=norm,
                 independent_full_Ih_m=float(integral),reference_V_m_s=reference["V"],reference_q_Pa=reference["stress"],
                 observed_mean_V_m_s=float(np.mean(surface["V"])),
                 raw_q_reference_rms_error_Pa=float(np.sqrt(np.average((q-reference["stress"])**2,weights=b["weight"]))),
                 raw_q_reference_max_error_Pa=float(max(abs(q-reference["stress"]))),
                 raw_q_within_cell_rms_Pa=float(np.sqrt(np.average((q-np.repeat([r[7] for r in cell_rows],9))**2,weights=b["weight"]))),
                 raw_q_within_cell_max_Pa=float(max(abs(q-np.repeat([r[7] for r in cell_rows],9)))))
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
