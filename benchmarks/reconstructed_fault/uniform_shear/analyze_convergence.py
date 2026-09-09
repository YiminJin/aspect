#!/usr/bin/env python3
"""Conditional K1 trajectory, unsmoothed stress and velocity-profile errors.

The input analysis initializes its reference ONCE. This second pass uses its
saved reference trajectory, never production histories as subsequent inputs.
It separates raw point errors from means and resolution-dependent initial data.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from analyze import h, read
from reference import Histories, solve_step


def profile_primitive(ys, phi, samples):
    """Integrate h of Q1 phi, split at FE nodes; no truncation/renormalization."""
    gauss, weights = np.polynomial.legendre.leggauss(32)
    a, b = ys[:-1], ys[1:]
    def panels(left, right):
        positions = (left[:, None]+right[:, None])/2 + (right-left)[:, None]*gauss/2
        return (right-left)/2 * (h(np.interp(positions, ys, phi)) @ weights)
    totals = np.r_[0., np.cumsum(panels(a, b))]
    index = np.clip(np.searchsorted(ys, samples, side="right")-1, 0, len(ys)-2)
    return totals[index]+panels(ys[index], samples), float(totals[-1])


def scalar_limit(integral, initial, dt):
    """One independent small-dt trajectory, initialized once from fixed data."""
    history, slip, result = initial, 0., []
    for k in range(round(6/dt)):
        time = (k+1)*dt
        response, history = solve_step(history, dt, 1e-4*(1+.2*min(time/4, 1.)),
                                       integral, integral)
        slip += dt*response["V"]
        if any(abs(time-target) < 1e-12 for target in (2., 4., 6.)):
            result.append(dict(time_s=time, V=response["V"], Theta=history.theta,
                               C=history.cohesive, q=history.stress, slip=slip))
    return result


def analyze(directory, analysis):
    nodes = read(directory, "phase", 0, ("x", "y", "phi"))
    xs = np.unique(nodes["x"])
    profile = np.sort(np.unique(nodes[nodes["x"] == xs[len(xs)//2]][["y", "phi"]]), order="y")
    ys, phi = profile["y"], profile["phi"]
    integral = analysis["profile"]["independent_Ih_m"]
    output = directory.parent/(directory.name+"-errors")
    output.mkdir(exist_ok=True)
    rows = []
    for item in analysis["steps"]:
        step, dt = item["step"], item["dt_s"]
        bulk = read(directory, "bulk", step, ("x", "y", "weight", "ux", "p", "old_tau_xy"))
        surface = read(directory, "surface", step, ("x", "V", "Theta", "C"))
        if analysis["geometry"]["max_angle_rad"] > 1e-6:
            raise ValueError("Horizontal scalar reference is not applicable")
        # For the verified horizontal K1 line 2*S_xy=1. This is the full
        # production Maxwell decomposition, not the strain-only diagnostic.
        q = bulk["kappa"]*(bulk["ux_y"]+bulk["uy_x"]
                            -bulk["chi"]*bulk["V"]-bulk["history"])
        q += math.exp(-dt/100)*bulk["old_tau_xy"]
        reference_q = item["evaluated_stress_reference_Pa"]
        yq, index = np.unique(bulk["y"], return_inverse=True)
        primitive, check_integral = profile_primitive(ys, phi, yq)
        if abs(check_integral-integral) > 1e-8:
            raise ValueError("Center profile disagrees with full normal-profile quadrature")
        reference_u = (-item["U_m_s"]/2
                       +(item["U_m_s"]-item["V_reference_m_s"])*(yq+.5)
                       +item["V_reference_m_s"]*primitive/integral)
        qe, ue = q-reference_q, bulk["ux"]-reference_u[index]
        rms = lambda values: float(np.sqrt(np.average(values**2, weights=bulk["weight"])))
        measured = dict(time_s=item["time_s"], dt_s=dt,
                        raw_q_rms_error_Pa=rms(qe), raw_q_max_error_Pa=float(max(abs(qe))),
                        raw_q_min_Pa=float(min(q)), raw_q_max_Pa=float(max(q)),
                        velocity_rms_error_m_s=rms(ue), velocity_max_error_m_s=float(max(abs(ue))),
                        pressure_min_Pa=float(min(bulk["p"])), pressure_max_Pa=float(max(bulk["p"])),
                        pressure_mean_Pa=float(np.average(bulk["p"], weights=bulk["weight"])),
                        raw_q_along_fault_range_Pa=float(max(np.ptp(q[index == j]) for j in range(len(yq)))),
                        normalization_ratio=item["integrated_slip_normalization"],
                        normalization_rate_error_m_s=abs(1-item["integrated_slip_normalization"])*item["V_mean_m_s"])
        # Existing resolved conditional criterion: 0.2% relative plus 1e-5
        # of the documented dimensional scale. Near-zero velocities use the
        # absolute allowance, not a replacement machine-minimum denominator.
        checks = dict(raw_q=measured["raw_q_max_error_Pa"] <= .015+.002*abs(reference_q),
                      velocity_max=bool(np.all(abs(ue) <= 1e-9+.002*abs(reference_u[index]))),
                      velocity_rms=rms(ue) <= 1e-9+.002*rms(reference_u[index]))
        for field, expected, scale in (("V", "V_reference_m_s", 1e-4),
                                       ("Theta", "Theta_reference_s", 200.),
                                       ("C", "C_reference_Pa", 1500.)):
            data, ref = surface[field], item[expected]
            measured[field] = dict(min=float(min(data)), max=float(max(data)),
                                   mean=item[{"V":"V_mean_m_s", "Theta":"Theta_mean_s", "C":"C_mean_Pa"}[field]],
                                   reference=ref, max_error=float(max(abs(data-ref))),
                                   endpoint_values=[float(data[0]), float(data[-1])])
            checks[field] = bool(max(abs(data-ref)) <= 1e-5*scale+.002*abs(ref))
        checks["slip"] = abs(item["accumulated_slip_m"]-item["accumulated_reference_slip_m"]) <= 6e-9+.002*abs(item["accumulated_reference_slip_m"])
        measured["log10_V_ratio_max"] = float(max(abs(np.log10(surface["V"]/item["V_reference_m_s"]))))
        measured["resolved_conditional_checks"] = checks
        measured["accumulated_slip_m"] = item["accumulated_slip_m"]
        measured["reference_slip_m"] = item["accumulated_reference_slip_m"]
        rows.append(measured)
        # Preserve raw transverse extrema and all native quadrature-point
        # errors. Neither export filters stress or changes an accepted state.
        np.savetxt(output/f"raw_qps_{step}.csv", np.column_stack((bulk["x"], bulk["y"], bulk["weight"], q, qe, bulk["ux"], reference_u[index], ue)),
                   delimiter=",", header="x,y,weight,q_Pa,q_error_Pa,ux,reference_ux,ux_error", comments="")
        samples = [[y, float(np.mean(bulk["ux"][index == j])), reference_u[j],
                    float(min(q[index == j])), float(np.mean(q[index == j])), float(max(q[index == j])), reference_q]
                   for j,y in enumerate(yq)]
        np.savetxt(output/f"transverse_{step}.csv", samples, delimiter=",",
                   header="y,ux_mean,reference_ux,q_min,q_mean,q_max,reference_q", comments="")
    initial = Histories(1500., analysis["steps"][0]["C_mean_Pa"], analysis["steps"][0]["Theta_mean_s"])
    limit = scalar_limit(integral, initial, 1/256)
    finer = scalar_limit(integral, initial, 1/512)
    return dict(initialization=dict(mesh=[len(xs)-1,len(ys)-1], phi_center=float(np.interp(0.,ys,phi)),
                                    phi_raw_min=float(min(phi)), Ih_m=integral, C0_Pa=initial.cohesive,
                                    Theta0_s=initial.theta, q0_retained_Pa=initial.stress),
                completed_to_6_seconds=analysis["completed_to_6_seconds"],
                containment=analysis["containment_allowance"],
                omitted_fraction=analysis["profile"]["max_omitted_strip_fraction"],
                assumption_checks=analysis["assumption_checks"], steps=rows,
                resolved_conditional_pass=analysis["completed_to_6_seconds"] and all(
                    all(row["resolved_conditional_checks"].values()) for row in rows),
                temporal_reference_dt_s=1/512, temporal_reference=finer,
                temporal_reference_self_difference={key:max(abs(a[key]-b[key]) for a,b in zip(limit,finer))
                                                    for key in ("V","Theta","C","q","slip")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("analysis", type=Path)
    args = parser.parse_args()
    result = analyze(args.directory, json.loads(args.analysis.read_text()))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
