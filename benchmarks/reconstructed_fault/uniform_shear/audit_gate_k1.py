#!/usr/bin/env python3
"""Audit missing K1 initialization prerequisites from saved production exports.

This does not run mechanics or certify Gate K1. Targets are the existing K0/K1
targets, and the finite association strip is never renormalized to obtain a pass.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

from analyze import h
from reference import ideal_profile


def read_csv(path, columns):
    rows = np.atleast_1d(np.genfromtxt(path, names=True, delimiter=","))
    if not set(columns).issubset(rows.dtype.names or ()):
        raise ValueError(f"Missing columns in {path}")
    if any(not np.isfinite(rows[name]).all() for name in rows.dtype.names):
        raise ValueError(f"Non-finite input in {path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent/"diagnostics"
    parser.add_argument("--packet", type=Path, default=base/"results-corrected")
    parser.add_argument("--output", type=Path, default=base/"gate-k1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    nodes = read_csv(args.packet/"raw/pre_mechanics_nodes.csv", ("x", "y", "phi"))
    particles = read_csv(args.packet/"raw/before_phase_particles.csv", ("x", "y", "H", "volume"))
    xs, ys = np.unique(nodes["x"]), np.unique(nodes["y"])
    grid = np.full((len(xs), len(ys)), np.nan)
    grid[np.searchsorted(xs,nodes["x"]), np.searchsorted(ys,nodes["y"])] = nodes["phi"]
    if not np.isfinite(grid).all() or grid.min() < 0 or grid.max() >= 1:
        raise ValueError("This audit requires the saved physical, nonsingular rectangular K1 profile")
    phi = RegularGridInterpolator((xs,ys), grid, bounds_error=True)
    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(args.packet/"pre_mechanics_fault.vtu"))
    reader.Update()
    fault = reader.GetOutput()
    points = vtk_to_numpy(fault.GetPoints().GetData())[:,:2]
    def property_values(name):
        values = fault.GetPointData().GetArray(name)
        if values is None:
            raise ValueError(f"Missing fault property {name}")
        return vtk_to_numpy(values)
    production_Ih = property_values("phase field fault previous I h")
    production_C = property_values("phase field fault cohesive traction")
    theta = property_values("phase field fault state")
    differences = np.diff(points,axis=0)
    lengths = np.linalg.norm(differences,axis=1)
    tangents = differences/lengths[:,None]
    normals = np.column_stack((-tangents[:,1],tangents[:,0]))

    # The unchanged manager obtains this strip from the prescribed stationary
    # profile, not from the solved FE tail. Prior production queries measured
    # the same value. Recompute it independently; do not fit it to corrected phi.
    half_width, _ = ideal_profile()
    def integrate(origin, normal, extent=None, tolerance=1e-11):
        lower, upper = -np.inf, np.inf
        crossings = []
        for d, coordinates in enumerate((xs,ys)):
            if abs(normal[d]) > 1e-14:
                values = (coordinates-origin[d])/normal[d]
                lower, upper = max(lower,min(values)), min(upper,max(values))
                crossings.extend(values)
        if extent is not None:
            lower, upper = max(lower,-extent), min(upper,extent)
        breaks = sorted(set([lower,upper]+[z for z in crossings if lower<z<upper]))
        # Split at every Q1 knot; tolerances constrain integration error only.
        return sum(quad(lambda z: float(h(phi([np.clip(origin+z*normal,
                                                       [xs[0],ys[0]], [xs[-1],ys[-1]])])[0])),
                        a,b,epsabs=tolerance/len(breaks),epsrel=tolerance)[0]
                   for a,b in zip(breaks,breaks[1:]))

    mass, load = np.zeros((len(points),len(points))), np.zeros(len(points))
    samples = []
    abscissae, weights = np.polynomial.legendre.leggauss(3)
    for s, length in enumerate(lengths):
        for xi, weight in zip((abscissae+1)/2,weights/2):
            shape = np.array([1-xi,xi])
            origin = shape@points[s:s+2]
            full = integrate(origin,normals[s])
            tight = integrate(origin,normals[s],tolerance=5e-13)
            strip = integrate(origin,normals[s],extent=half_width)
            samples.append([s,xi,*origin,full,strip,full-strip,(full-strip)/full,abs(full-tight)])
            mass[s:s+2,s:s+2] += length*weight*np.outer(shape,shape)
            load[s:s+2] += length*weight*full*shape
    samples = np.array(samples)
    independent_Ih = np.linalg.solve(mass,load)

    # Cross-check adaptive integration with fixed Gauss rules on the center
    # Q1 column. In particular, resolve the strip endpoint within its FE cell.
    def fixed_gauss_integral(extent, order):
        breaks = sorted(set([-extent,extent]+[y for y in ys if -extent<y<extent]))
        abscissae, weights = np.polynomial.legendre.leggauss(order)
        total = 0.0
        for a,b in zip(breaks,breaks[1:]):
            y = (a+b)/2+(b-a)/2*abscissae
            values = phi(np.column_stack((np.full(order,(xs[0]+xs[-1])/2),y)))
            total += (b-a)/2*np.dot(weights,h(values))
        return total
    gauss_full = fixed_gauss_integral(.5,32)
    gauss_strip = fixed_gauss_integral(half_width,32)
    gauss_change = max(abs(gauss_full-fixed_gauss_integral(.5,16)),
                       abs(gauss_strip-fixed_gauss_integral(half_width,16)))

    # Independent finite-strip associations and particle-volume Q1 projection.
    # At an internal vertex select only one segment, as in the production policy.
    locations = np.column_stack((particles["x"],particles["y"]))
    distances = np.full((len(particles),len(lengths)),np.inf)
    coordinates = np.zeros_like(distances)
    for s, length in enumerate(lengths):
        offset = locations-points[s]
        xi, z = offset@tangents[s]/length, offset@normals[s]
        active = (xi>=0)&(xi<=1)&(abs(z)<=half_width)
        distances[active,s] = abs(z[active])
        coordinates[:,s] = xi
    selected = np.argmin(distances,axis=1)
    active = np.isfinite(distances[np.arange(len(particles)),selected])
    projection_mass, rhs = np.zeros_like(mass), np.zeros_like(load)
    q = np.sqrt(2e6*particles["H"])/(1+h(phi(locations)))
    for p in np.flatnonzero(active):
        s = selected[p]
        xi = coordinates[p,s]
        shape = np.array([1-xi,xi])
        projection_mass[s:s+2,s:s+2] += particles["volume"][p]*np.outer(shape,shape)
        rhs[s:s+2] += particles["volume"][p]*q[p]*shape
    independent_C = np.linalg.solve(projection_mass,rhs)

    report = dict(
        gate_K1_satisfied=False,
        reason="Association-strip containment fails; corrected accepted mechanics and convergence remain unmeasured.",
        provenance="Saved corrected pre-mechanics exports; no accepted kinematic state or real timestep in this packet.",
        geometry=dict(length_m=float(sum(lengths)),
                      max_abs_y_over_ell=float(max(abs(points[:,1]))/.15625),
                      max_segment_angle_rad=float(max(abs(np.arctan2(differences[:,1],differences[:,0])))),
                      endpoints_m=points[[0,-1]].tolist()),
        normalization=dict(independent_Ih_range_m=[float(min(samples[:,4])),float(max(samples[:,4]))],
                           production_Ih_range_m=[float(min(production_Ih)),float(max(production_Ih))],
                           max_projected_relative_error=float(max(abs(independent_Ih-production_Ih)/independent_Ih)),
                           max_quadrature_self_change_m=float(max(samples[:,8])),
                           fixed_gauss_16_to_32_change_m=float(gauss_change),
                           fixed_gauss_center_full_m=float(gauss_full),
                           fixed_gauss_center_strip_m=float(gauss_strip)),
        containment=dict(half_width_m=half_width,
                         half_width_provenance="Independent stationary support; unchanged manager policy and prior production measurement agree.",
                         max_omitted_integral_m=float(max(samples[:,6])),
                         max_omitted_fraction=float(max(samples[:,7])),target=1e-6,passed=False,
                         exterior_domain_tail="Unmeasured; no extrapolation beyond the saved FE domain is justified."),
        initial_surface=dict(endpoint_projection_mass_m2=np.diag(projection_mass)[[0,-1]].tolist(),
                             projected_C_max_error_Pa=float(max(abs(independent_C-production_C))),
                             C_range_Pa=[float(min(production_C)),float(max(production_C))],
                             theta_range_s=[float(min(theta)),float(max(theta))]),
        inputs_sha256={name:hashlib.sha256((args.packet/name).read_bytes()).hexdigest()
                       for name in ["raw/pre_mechanics_nodes.csv","raw/before_phase_particles.csv","pre_mechanics_fault.vtu"]})
    report["containment"]["passed"] = report["containment"]["max_omitted_fraction"] <= 1e-6
    np.savetxt(args.output/"normal_profile_integrals.csv",samples,delimiter=",",comments="",
               header="segment,xi,x_m,y_m,Ih_full_m,Ih_strip_m,Ih_omitted_m,omitted_fraction,quadrature_self_change_m")
    np.savetxt(args.output/"initial_surface_projection.csv",
               np.column_stack((points,production_Ih,independent_Ih,production_C,independent_C)),
               delimiter=",",comments="",header="x_m,y_m,production_Ih_m,independent_Ih_m,production_C_Pa,independent_C_Pa")
    (args.output/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return 0 if report["gate_K1_satisfied"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
