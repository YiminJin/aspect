#!/usr/bin/env python3
"""Independent FE-profile and scalar analysis of serial/distributed K1 output.

Reports assumption failures separately from conditional trajectory errors.
Does not fit coefficients, reset histories, or renormalize truncated slip.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq

from reference import Histories, solve_step


def read(directory, name, step, required):
    path = directory / f"{name}_{step}.csv"
    paths = [path] if path.exists() else sorted(directory.glob(f"{name}_rank*_{step}.csv"))
    if not paths:
        raise ValueError(f"Missing diagnostic data: {path}")
    ranks = [np.atleast_1d(np.genfromtxt(p, delimiter=",", names=True)) for p in paths]
    if name in ("surface", "surface_weak", "segments", "time"):
        if any(not np.array_equal(ranks[0], other) for other in ranks[1:]):
            raise ValueError(f"Inconsistent replicated {name} data at step {step}")
        values = ranks[0]
    else:
        values = np.concatenate(ranks)
    if not set(required).issubset(values.dtype.names or ()):
        raise ValueError(f"Missing columns in {path}")
    if any(not np.isfinite(values[field]).all() for field in values.dtype.names):
        raise ValueError(f"Non-finite diagnostic data in {path}")
    return values


def h(phi):
    effective = np.maximum(phi, 0.0)
    if np.any(effective >= 1):
        raise ValueError("Singular I_h profile")
    return 128.0 * effective * (1.0 + effective) / (1.0 - effective)**2


def boundary_error(bulk, loading):
    """Recover exact Q2 boundary traces from each cell's 3x3 Gauss samples.

    Evaluate at the tangential Gauss coordinates; three samples determine
    each quadratic trace. This is not extrapolation of a fitted bulk model.
    """
    error = 0.0
    for x in np.unique(bulk["x"]):
        column = np.sort(bulk[bulk["x"] == x], order="y")
        for samples, boundary in ((column[:3], -.5), (column[-3:], .5)):
            ys = samples["y"]
            basis = np.array([np.prod([(boundary-ys[j])/(ys[i]-ys[j])
                                       for j in range(3) if j != i]) for i in range(3)])
            error = max(error, abs(np.dot(basis, samples["ux"])-boundary*loading),
                        abs(np.dot(basis, samples["uy"])))
    return float(error)


def initial_history_audit(particles):
    # Invert the stationary AT1 first integral at each original transverse
    # coordinate. This is independent of both the FE solve and initializer.
    core, ell = .6, .15625
    def jacobian(t):
        p = core*math.sin(t)**2
        # Cancel sqrt(phi*(core-phi)) analytically after phi=core*sin²(t).
        return 2*ell*math.sqrt(1+core)*(1-p)/math.sqrt(3-core-(1+core)*p)
    def distance(t):
        return quad(jacobian, t, math.pi/2, epsabs=1e-12, epsrel=1e-12)[0]
    half_width = distance(0.)
    error, variation = 0., 0.
    for y in np.unique(particles["y"]):
        column = particles[particles["y"] == y]
        p = 0. if abs(y) >= half_width else core*math.sin(
            brentq(lambda t: distance(t)-abs(y), 0., math.pi/2, xtol=1e-13))**2
        expected = 64*core/h(core)*(1+h(p))**2 if p > .1 else .5
        error = max(error, float(np.max(abs(column["H"]-expected)/expected)))
        variation = max(variation, float(np.ptp(column["H"])))
    return dict(max_relative_error=error, max_along_fault_range_Pa=variation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--provisional-k1-containment", action="store_true",
                        help="Apply the reviewed K1-only 1e-4 containment allowance; no reference or normalization change")
    args = parser.parse_args()
    directory = args.directory
    nodes = read(directory, "phase", 0, ("x", "y", "phi"))
    xs, ys = np.unique(nodes["x"]), np.unique(nodes["y"])
    grid = np.full((len(xs), len(ys)), np.nan)
    grid[np.searchsorted(xs, nodes["x"]), np.searchsorted(ys, nodes["y"])] = nodes["phi"]
    if not np.isfinite(grid).all() or grid.min() < -1e-4:
        raise ValueError("Invalid rectangular Q1 phase profile")
    phi = RegularGridInterpolator((xs, ys), grid, method="linear", bounds_error=True)

    # Independent integration along actual normals, split at every crossed FE
    # grid line. This neither calls the production kernel nor uses its I_h.
    def integrate(origin, normal, interval=None, tolerance=1e-11):
        crossings = []
        lower, upper = -math.inf, math.inf
        for d, coordinates in enumerate((xs, ys)):
            if abs(normal[d]) > 1e-14:
                candidates = (coordinates-origin[d])/normal[d]
                lower = max(lower, candidates.min())
                upper = min(upper, candidates.max())
                crossings.extend(candidates)
        if interval is not None:
            lower, upper = max(lower, interval[0]), min(upper, interval[1])
        breaks = sorted(set([lower, upper]+[z for z in crossings if lower < z < upper]))
        return sum(quad(lambda z: float(h(phi(np.clip(origin+z*normal,
                                                       [xs[0], ys[0]], [xs[-1], ys[-1]]))[0])),
                        a, b, epsabs=tolerance/len(breaks), epsrel=tolerance)[0]
                   for a, b in zip(breaks, breaks[1:]))

    surface0 = read(directory, "surface", 0, ("fault", "node", "x", "y", "V", "Theta", "C", "Ih"))
    if np.any(surface0["fault"] != 0):
        raise ValueError("K1 requires one fault")
    points = np.column_stack((surface0["x"], surface0["y"]))
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    weights = np.zeros(len(points))
    weights[:-1] += lengths/2; weights[1:] += lengths/2
    length = lengths.sum()
    segments = read(directory, "segments", 0, ("x", "y", "nx", "ny", "half_width_minus", "half_width_plus"))
    independent = []
    omitted = []
    quadrature_changes = []
    for row in segments:
        origin, normal = np.array([row["x"], row["y"]]), np.array([row["nx"], row["ny"]])
        integral = integrate(origin, normal)
        independent.append(integral)
        quadrature_changes.append(abs(integral-integrate(origin, normal, tolerance=5e-13)))
        strip = integrate(origin, normal, (-row["half_width_minus"], row["half_width_plus"]))
        omitted.append((integral-strip)/integral)
    independent = np.array(independent)
    integral = float(np.dot(lengths, independent)/length)
    projected_midpoints = (surface0["Ih"][:-1]+surface0["Ih"][1:])/2

    # Match the production surface sampling/projection, but integrate each
    # Q1 FE normal profile independently. Midpoint comparisons alone mix
    # profile quadrature error with surface projection/interpolation error.
    mass, load = np.zeros((len(points), len(points))), np.zeros(len(points))
    abscissae, gauss_weights = np.polynomial.legendre.leggauss(3)
    for s, row in enumerate(segments):
        normal = np.array([row["nx"], row["ny"]])
        for xi, weight in zip((abscissae+1)/2, gauss_weights/2):
            shape = np.array([1-xi, xi])
            origin = shape @ points[s:s+2]
            value = integrate(origin, normal)
            mass[s:s+2, s:s+2] += lengths[s]*weight*np.outer(shape, shape)
            load[s:s+2] += lengths[s]*weight*value*shape
    independent_projected_Ih = np.linalg.solve(mass, load)

    # Retain the independent OLD point-rule reconstruction for old artifacts.
    # For domain-rule output this is a discretization comparison, not an
    # independent check of the new domain-integrated initial projection.
    particles0 = read(directory, "particles", 0, ("id", "x", "y", "volume", "H", "active", "segment", "xi"))
    matrix, rhs = np.zeros((len(points), len(points))), np.zeros(len(points))
    for particle in particles0[particles0["active"] == 1]:
        s, xi = int(particle["segment"]), particle["xi"]
        shape = np.array([1-xi, xi])
        pphi = float(phi([[particle["x"], particle["y"]]])[0])
        q = math.sqrt(2e6*particle["H"])/(1+float(h(pphi)))
        matrix[s:s+2, s:s+2] += particle["volume"]*np.outer(shape, shape)
        rhs[s:s+2] += particle["volume"]*q*shape
    projected_C = np.linalg.solve(matrix, rhs)
    domain_rule = ((directory / "surface_weak_0.csv").exists()
                   or any(directory.glob("surface_weak_rank*_0.csv")))
    projection_difference = float(np.max(abs(projected_C-surface0["C"])))
    initial_projection = dict(surface_rule="point volume",
                              C_projection_max_error_Pa=projection_difference,
                              endpoint_projection_mass_m2=np.diag(matrix)[[0, -1]].tolist())
    if domain_rule:
        weak = read(directory, "surface_weak", 0, ("Mdiag",))
        initial_projection = dict(surface_rule="domain integrated",
            legacy_point_projection_difference_Pa=projection_difference,
            independent_domain_initial_projection_check="covered by focused moment/projection regressions",
            endpoint_projection_mass_m2=weak["Mdiag"][[0, -1]].tolist())

    report = dict(
        geometry=dict(length_m=float(length), vertices=len(points),
                      max_abs_y_m=float(abs(points[:, 1]).max()),
                      max_angle_rad=float(np.max(abs(np.arctan2(np.diff(points[:, 1]), np.diff(points[:, 0]))))),
                      endpoints=points[[0, -1]].tolist(),
                      normal_half_widths_m=[float(segments["half_width_minus"].min()), float(segments["half_width_plus"].max())]),
        profile=dict(raw_min=float(grid.min()), raw_max=float(grid.max()),
                     max_along_fault_phi_range=float(np.ptp(grid, axis=0).max()),
                     independent_Ih_m=integral,
                     independent_Ih_range_m=[float(independent.min()), float(independent.max())],
                     max_Ih_quadrature_change_m=float(max(quadrature_changes)),
                     max_midpoint_Ih_relative_error=float(np.max(abs(projected_midpoints-independent)/independent)),
                     max_projected_Ih_relative_error=float(np.max(abs(surface0["Ih"]-independent_projected_Ih)/independent_projected_Ih)),
                     max_omitted_strip_fraction=float(max(omitted))),
        initialization=dict(C_mean_Pa=float(np.dot(weights, surface0["C"])/length),
                            H=initial_history_audit(particles0),
                            **initial_projection),
        steps=[])
    # Initialize ONCE from the retained initial histories. Subsequent ASPECT
    # values are observations, never inputs resetting the reference history.
    history = Histories(stress=1500., cohesive=report["initialization"]["C_mean_Pa"],
                        theta=float(np.dot(weights, surface0["Theta"])/length))
    reference_slip = observed_slip = 0.0
    previous_time = 0.0
    vertex_theta = surface0["Theta"].copy()
    for step in range(100):
        if not ((directory / f"time_{step}.csv").exists()
                or (directory / f"time_rank0_{step}.csv").exists()):
            break
        time_row = read(directory, "time", step, ("time", "dt", "U"))[0]
        time, dt, loading = map(float, (time_row["time"], time_row["dt"], time_row["U"]))
        if step > 0 and (time <= previous_time or abs(time-previous_time-dt) > 1e-12):
            raise ValueError("Inconsistent accepted timestep sequence")
        surface = read(directory, "surface", step, ("V", "Theta", "C"))
        bulk = read(directory, "bulk", step, ("weight", "ux", "uy", "ux_y", "uy_x", "chi", "V", "history", "old_tau_xy"))
        particles = read(directory, "particles", step, ("id", "volume", "tau_xy"))
        mean = lambda field: float(np.dot(weights, surface[field])/length)
        old_theta = history.theta
        response, history = solve_step(history, dt, loading, integral, integral, initial=step == 0)
        kappa, beta = -1e8*math.expm1(-dt/100), math.exp(-dt/100)
        # Resolve the actual tangent, not the intended horizontal line. Verify
        # recovered associations against exported production activity and V
        # before using 2*S_xy in the physical Maxwell shear response.
        qp = np.column_stack((bulk["x"], bulk["y"]))
        distances = np.full((len(qp), len(lengths)), np.inf)
        coordinates = np.zeros_like(distances)
        factors = np.zeros(len(lengths))
        for s, segment_length in enumerate(lengths):
            tangent = (points[s+1]-points[s])/segment_length
            normal = np.array([-tangent[1], tangent[0]])
            offset = qp-points[s]
            xi, z = offset @ tangent / segment_length, offset @ normal
            active = (xi >= 0) & (xi <= 1) & (abs(z) <= segments[s]["half_width_plus"])
            distances[active, s] = abs(z[active])
            coordinates[:, s] = xi
            factors[s] = tangent[0]*normal[1]+tangent[1]*normal[0]
        nearest = np.argmin(distances, axis=1)
        active = np.isfinite(distances[np.arange(len(qp)), nearest])
        xi = coordinates[np.arange(len(qp)), nearest]
        recovered_V = np.where(active, (1-xi)*surface["V"][nearest]+xi*surface["V"][nearest+1], 0.)
        if not np.array_equal(active, bulk["active"] == 1) or np.max(abs(recovered_V-bulk["V"])) > 1e-15:
            raise ValueError("Independent QP association disagrees with production")
        shear_factor = np.where(active, factors[nearest], 0.)
        evaluated_stress = (kappa*(bulk["ux_y"]+bulk["uy_x"]
                                   -(bulk["chi"]*bulk["V"]+bulk["history"])*shear_factor)
                            + beta*bulk["old_tau_xy"])
        stress_mean = float(np.average(evaluated_stress, weights=bulk["weight"]))
        norm = float(np.dot(bulk["weight"], bulk["chi"]*bulk["V"]+bulk["history"])/(length*mean("V")))
        if step > 0:
            reference_slip += dt*response["V"]
            observed_slip += dt*mean("V")
            decay = np.exp(-surface["V"]*dt/.001)
            vertex_theta = vertex_theta*decay + .001/surface["V"]*(-np.expm1(-surface["V"]*dt/.001))
        report["steps"].append(dict(
            step=step, time_s=time, dt_s=dt, U_m_s=loading,
            V_mean_m_s=mean("V"), V_reference_m_s=response["V"],
            V_error_m_s=mean("V")-response["V"],
            V_along_fault_range_over_scale=float(np.ptp(surface["V"])/1e-4),
            C_mean_Pa=mean("C"), C_reference_Pa=history.cohesive,
            C_along_fault_range_over_scale=float(np.ptp(surface["C"])/1500.),
            Theta_mean_s=mean("Theta"), Theta_reference_s=history.theta,
            Theta_reference_increment_s=0.0 if step == 0 else history.theta-old_theta,
            Theta_along_fault_range_over_scale=float(np.ptp(surface["Theta"])/200.),
            Theta_vertex_update_error_s=float(np.max(abs(vertex_theta-surface["Theta"]))),
            prescribed_velocity_max_error_m_s=boundary_error(bulk, loading),
            evaluated_stress_mean_Pa=stress_mean, evaluated_stress_reference_Pa=response["stress"],
            evaluated_stress_range_Pa=float(np.ptp(evaluated_stress)),
            retained_particle_stress_mean_Pa=float(np.average(particles["tau_xy"], weights=particles["volume"])),
            integrated_slip_normalization=norm,
            particle_volume_sum_m2=float(particles["volume"].sum()),
            particle_volume_relative_extrema=(particles["volume"][[np.argmin(particles["volume"]),np.argmax(particles["volume"])]]/(.25/len(particles0))).tolist(),
            max_transverse_velocity_m_s=float(np.max(abs(bulk["uy"]))),
            transverse_velocity_rms_m_s=float(np.sqrt(np.average(bulk["uy"]**2, weights=bulk["weight"]))),
            divergence_rms_s_inv=float(np.sqrt(np.average((bulk["ux_x"]+bulk["uy_y"])**2, weights=bulk["weight"]))),
            accumulated_slip_m=observed_slip, accumulated_reference_slip_m=reference_slip))
        previous_time = time
    report["completed_to_6_seconds"] = abs(previous_time-6.) < 1e-12
    report["containment_allowance"] = dict(
        value=1e-4 if args.provisional_k1_containment else 1e-6,
        scope="provisional K1 only" if args.provisional_k1_containment else "original gate",
        full_Ih_reference_unchanged=True,
        completed_refinement_campaign=False)
    report["assumption_checks"] = dict(
        straight_centered_fault=report["geometry"]["max_abs_y_m"]/.15625 <= 1e-6
                                and report["geometry"]["max_angle_rad"] <= 1e-6,
        contained_profiles=max(omitted) <= report["containment_allowance"]["value"],
        projected_Ih=report["profile"]["max_projected_Ih_relative_error"] <= 1e-6,
        integrated_slip=all(abs(step["integrated_slip_normalization"]-1) <= 1e-4 for step in report["steps"]),
        along_fault_uniformity=all(step["V_along_fault_range_over_scale"] <= 1e-4 for step in report["steps"]),
        transverse_velocity=all(step["transverse_velocity_rms_m_s"] <= 1e-8 for step in report["steps"]),
        divergence=all(step["divergence_rms_s_inv"] <= 1e-8 for step in report["steps"]),
        prescribed_velocities=all(step["prescribed_velocity_max_error_m_s"] <= 1e-13 for step in report["steps"]))
    report["trajectory_comparison_valid"] = all(report["assumption_checks"].values())
    print(json.dumps(report, indent=2))
    return 0 if report["completed_to_6_seconds"] and report["trajectory_comparison_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
