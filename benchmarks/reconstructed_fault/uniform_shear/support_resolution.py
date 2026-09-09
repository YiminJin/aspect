#!/usr/bin/env python3
"""Saved-profile support study and separately labeled retained-fraction diagnostic.

The approved reference stays unchanged. The diagnostic changes only U-V to
U-r*V in the scalar bulk balance; the cohesive law retains the FULL I_h.
Neither calculation changes production geometry, histories or support policy.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

from analyze import h
from reference import Histories, ideal_profile, residual, solve_step


def retained_step(histories, dt, loading, integral, retained, initial=False):
    beta, kappa = math.exp(-dt/100), -1e8*math.expm1(-dt/100)
    function = lambda v: residual(v,histories,dt,loading,integral,integral)+kappa*(1-retained)*v
    lower, upper = 1e-12, 1e-4
    if function(lower) <= 0:
        raise ValueError("No admissible interior retained-fraction root")
    while function(upper) > 0:
        upper *= 2
    v = brentq(function,lower,upper,xtol=1e-16,rtol=1e-14)
    if abs(function(v)) > 1e-7:
        raise ArithmeticError("Diagnostic traction root failed existing reference accuracy")
    stress = beta*histories.stress+kappa*(loading-retained*v)
    cohesive = beta*histories.cohesive+kappa/integral*v
    response = dict(V=v,stress=stress,cohesive=cohesive)
    theta = histories.theta*math.exp(-v*dt/.001)+.001/v*(-math.expm1(-v*dt/.001))
    return response, histories if initial else Histories(stress,cohesive,theta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent/"diagnostics"
    parser.add_argument("--accepted",type=Path,help="Use only this actual accepted time/load sequence")
    parser.add_argument("--output",type=Path,default=base/"support-resolution")
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    nodes = np.genfromtxt(base/"results-corrected/raw/pre_mechanics_nodes.csv",names=True,delimiter=",")
    xs, ys = np.unique(nodes["x"]), np.unique(nodes["y"])
    profiles = np.empty((len(xs),len(ys)))
    profiles[np.searchsorted(xs,nodes["x"]),np.searchsorted(ys,nodes["y"])] = nodes["phi"]
    if profiles.min()<0 or profiles.max()>=1 or not np.isfinite(profiles).all():
        raise ValueError("Invalid saved K1 profile")
    gauss, weights = np.polynomial.legendre.leggauss(32)
    def integrals(width):
        breaks = sorted(set([-width,width]+[y for y in ys if -width<y<width]))
        totals = np.zeros(len(xs))
        for a,b in zip(breaks,breaks[1:]):
            z = (a+b)/2+(b-a)/2*gauss
            values = np.array([np.interp(z,ys,p) for p in profiles])
            totals += (b-a)/2*(h(values)@weights)
        return totals
    full = integrals(.5)
    half_width = ideal_profile()[0]
    omitted = lambda w: float(max((full-integrals(w))/full))
    minimum = brentq(lambda w: omitted(w)-1e-6,half_width,.5,xtol=1e-13)
    # A mesh-face-aligned candidate gives modest room below the target. This
    # is a recommendation to review, not a write to manager-owned geometry.
    candidate = float(ys[np.searchsorted(ys,minimum)])
    widths = sorted(set([half_width,minimum,candidate,.3125,.328125,.34375,.359375,.375,.4,.5]))
    particles = np.genfromtxt(base/"results-corrected/raw/before_phase_particles.csv",names=True,delimiter=",")
    # For this x-uniform regular layout the consistent surface projection is
    # constant and equals this volume-weighted transverse mean. This exposes
    # the initial-C side effect of widening the shared particle/QP support.
    q0 = np.sqrt(2e6*particles["H"])/(1+h(np.interp(particles["y"],ys,profiles[len(xs)//2])))
    def widened_initial_C(width):
        active = abs(particles["y"])<=width
        return float(np.average(q0[active],weights=particles["volume"][active]))
    curve = np.array([[w,omitted(w),float(np.mean(integrals(w))),w/half_width,
                       np.count_nonzero(abs(particles["y"])<=w),widened_initial_C(w)] for w in widths])
    np.savetxt(args.output/"width_scan.csv",curve,delimiter=",",comments="",
               header="half_width_m,max_omitted_fraction,mean_retained_integral_m,width_cost_ratio,associated_particles,estimated_initialized_C_Pa")

    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(base/"results-corrected/pre_mechanics_fault.vtu"))
    reader.Update()
    fault = reader.GetOutput()
    C0 = float(np.mean(vtk_to_numpy(fault.GetPointData().GetArray("phase field fault cohesive traction"))))
    theta0 = float(np.mean(vtk_to_numpy(fault.GetPointData().GetArray("phase field fault state"))))
    integral, retained = float(np.mean(full)), 1-omitted(half_width)
    if args.accepted:
        times = []
        for step in range(100):
            path = args.accepted/f"time_{step}.csv"
            if not path.exists():
                break
            row = np.atleast_1d(np.genfromtxt(path,names=True,delimiter=","))[0]
            times.append((float(row["time"]),float(row["dt"]),float(row["U"])))
        if not times or times[0][0] != 0:
            raise ValueError("Missing accepted initial state")
    else:
        times = [(t,2.,1e-4*(1+.2*min(t/4,1))) for t in [0.,2.,4.,6.]]
    # Both paths start once from retained initialized histories. Evaluated
    # timestep-zero responses do not replace those histories in either path.
    approved = diagnostic = Histories(1500.,C0,theta0)
    slip_approved = slip_diagnostic = 0.0
    trajectory = []
    for step,(time,dt,loading) in enumerate(times):
        if step and abs(time-times[step-1][0]-dt)>1e-12:
            raise ValueError("Inconsistent accepted sequence")
        response, approved = solve_step(approved,dt,loading,integral,integral,initial=step==0)
        altered, diagnostic = retained_step(diagnostic,dt,loading,integral,retained,initial=step==0)
        if step:
            slip_approved += dt*response["V"]
            slip_diagnostic += dt*altered["V"]
        trajectory.append([time,dt,loading,response["V"],altered["V"],response["stress"],altered["stress"],
                           approved.cohesive,diagnostic.cohesive,approved.theta,diagnostic.theta,
                           slip_approved,slip_diagnostic])
    np.savetxt(args.output/"retained_fraction_trajectory.csv",trajectory,delimiter=",",comments="",
               header="time_s,dt_s,U_m_s,approved_V_m_s,diagnostic_V_m_s,approved_evaluated_q_Pa,diagnostic_evaluated_q_Pa,approved_retained_C_Pa,diagnostic_retained_C_Pa,approved_Theta_s,diagnostic_Theta_s,approved_slip_m,diagnostic_slip_m")
    report = dict(containment_gate_passed=False,production_support_unchanged=True,
                  sequence="actual accepted" if args.accepted else "prospective nominal; not production evidence",
                  full_Ih_m=integral,retained_fraction=retained,
                  current_half_width_m=half_width,current_omitted_fraction=omitted(half_width),
                  minimum_half_width_for_1e_6_m=minimum,minimum_extension_each_side_m=minimum-half_width,
                  candidate_half_width_m=candidate,candidate_omitted_fraction=omitted(candidate),
                  estimated_band_work_ratio=candidate/half_width,
                  current_initial_associated_particles=int(np.count_nonzero(abs(particles["y"])<=half_width)),
                  candidate_initial_associated_particles=int(np.count_nonzero(abs(particles["y"])<=candidate)),
                  support_extension_initial_C_warning=dict(current_Pa=widened_initial_C(half_width),
                                                          minimum_width_Pa=widened_initial_C(minimum),
                                                          candidate_Pa=widened_initial_C(candidate)),
                  retained_initial_histories=dict(stress_Pa=1500.,C_Pa=C0,Theta_s=theta0),
                  scalar_diagnostic="q=beta*q_old+kappa*(U-r*V), C=beta*C_old+kappa*V/FULL_Ih; not the approved reference",
                  trajectory=trajectory)
    (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
