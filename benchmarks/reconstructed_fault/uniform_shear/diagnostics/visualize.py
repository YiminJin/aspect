#!/usr/bin/env python3
"""Export unsmoothed K1 initialization diagnostics, never accepted mechanics."""
import argparse
import json
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq
from scipy.sparse.linalg import spsolve
from vtkmodules.vtkCommonCore import vtkPoints, vtkIdList
from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter, vtkXMLUnstructuredGridReader
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.util.vtkConstants import VTK_QUAD, VTK_VERTEX, VTK_LINE


def csv(path):
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))


def write_vtu(path, points, cells, cell_type, fields, cell_fields=None):
    grid = vtkUnstructuredGrid()
    coordinates = vtkPoints()
    coordinates.SetData(numpy_to_vtk(np.column_stack((points, np.zeros(len(points)))), deep=True))
    grid.SetPoints(coordinates)
    for cell in cells:
        ids = vtkIdList()
        for i in cell:
            ids.InsertNextId(int(i))
        grid.InsertNextCell(cell_type, ids)
    for attributes, values in ((grid.GetPointData(), fields), (grid.GetCellData(), cell_fields or {})):
        for name, data in values.items():
            array = numpy_to_vtk(np.ascontiguousarray(data), deep=True)
            array.SetName(name)
            attributes.AddArray(array)
    writer = vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(grid)
    if writer.Write() != 1:
        raise OSError(path)
    # Read back the actual file, not just the in-memory object.
    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    assert reader.GetOutput().GetNumberOfPoints() == len(points)
    assert reader.GetOutput().GetNumberOfCells() == len(cells)


def intended_profile(y):
    core, ell = .6, .15625
    def jacobian(t):
        p = core*math.sin(t)**2
        return 2*ell*math.sqrt(1+core)*(1-p)/math.sqrt(3-core-(1+core)*p)
    def distance(t):
        return quad(jacobian, t, math.pi/2, epsabs=1e-12, epsrel=1e-12)[0]
    support = distance(0.)
    def value(z):
        if abs(z) >= support:
            return 0.
        if abs(z) < 1e-14:
            return core
        return core*math.sin(brentq(lambda t: distance(t)-abs(z), 0., math.pi/2))**2
    return np.array([value(z) for z in y])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    dest = args.output
    dest.mkdir(parents=True, exist_ok=True)
    raw = dest/"raw"
    raw.mkdir(exist_ok=True)
    for pattern in ("*.csv", "*.vtu", "*.prm", "log.txt"):
        for path in args.input.glob(pattern):
            shutil.copy2(path, raw/path.name)
    nodes = csv(raw/"pre_mechanics_nodes.csv")
    particles = csv(raw/"before_phase_particles.csv")
    stencils = csv(raw/"before_phase_cpdi.csv")
    nodes = np.sort(nodes, order="vertex")
    particles = np.sort(particles, order="id")
    cells = csv(raw/"pre_mechanics_cells.csv")
    topology = np.array([[int(row[c]) for c in ("v0", "v1", "v3", "v2")] for row in cells])
    topology = np.searchsorted(nodes["vertex"], topology)
    points = np.column_stack((nodes["x"], nodes["y"]))
    xs, ys = np.unique(nodes["x"]), np.unique(nodes["y"])
    nx, ny = len(xs)-1, len(ys)
    ix, iy = np.searchsorted(xs, nodes["x"]), np.searchsorted(ys, nodes["y"])
    phase_grid = np.empty((nx+1, ny))
    phase_grid[ix, iy] = nodes["phi"]
    phase = RegularGridInterpolator((xs, ys), phase_grid)

    # Build the actual transfer matrices and the exported homogeneous
    # periodic elimination. No field is symmetrized or written back to ASPECT.
    rows = np.searchsorted(particles["id"], stencils["id"])
    columns = np.searchsorted(nodes["vertex"], stencils["vertex"])
    assert np.array_equal(nodes["dof"][columns], stencils["dof"])
    constraints = csv(raw/"phase_constraints.csv")
    phase_dofs = set(nodes["dof"].astype(int))
    constraints = constraints[np.isin(constraints["dof"], list(phase_dofs))]
    dof_node = {int(dof): i for i, dof in enumerate(nodes["dof"])}
    for line in constraints:
        assert line["inhomogeneity"] == 0 and line["weight"] == 1
        a, b = dof_node[int(line["dof"])], dof_node[int(line["master"])]
        assert iy[a] == iy[b] and {ix[a], ix[b]} == {0, nx}
    assert len(constraints) == ny
    canonical = (ix % nx)*ny+iy
    W, X, Y = [sparse.coo_matrix((stencils[field], (rows, columns)),
                                 shape=(len(particles), len(nodes))).tocsr()
               for field in ("w", "gx", "gy")]
    eliminate = sparse.coo_matrix((np.ones(len(nodes)), (np.arange(len(nodes)), canonical)),
                                   shape=(len(nodes), nx*ny)).tocsr()
    weights, gx, gy = W@eliminate, X@eliminate, Y@eliminate
    volume, H = particles["volume"], particles["H"]
    def residual(values):
        p = weights@values
        numerator, denominator = (1-p)**2, (1-p)**2+128*p*(1+p)
        derivative = -2*(1-p)/denominator-numerator*(126+258*p)/denominator**2
        return (weights.T@(volume*(H*derivative+64))
                +3.125*(gx.T@(volume*(gx@values))+gy.T@(volume*(gy@values))))
    initial_residual = residual(np.zeros(nx*ny))
    jacobian = (weights.T@sparse.diags(volume*H*32000)@weights
                +3.125*(gx.T@sparse.diags(volume)@gx+gy.T@sparse.diags(volume)@gy))
    first_direction = spsolve(jacobian, -initial_residual)
    actual = phase_grid[:-1].ravel()
    final_residual = residual(actual)
    sums = np.asarray(W.sum(axis=1)).ravel()
    gradient_sum = np.column_stack((np.asarray(X.sum(axis=1)).ravel(), np.asarray(Y.sum(axis=1)).ravel()))
    linear_y_gradient = np.column_stack((X@nodes["y"], Y@nodes["y"]))
    bad_weights = abs(sums-1) > 1e-10
    bad_gradients = np.linalg.norm(gradient_sum, axis=1) > 1e-8
    # For a translationally invariant stencil, shifting a full cell along x
    # must commute with both transfer and residual assembly after periodicity.
    shifted = np.roll(actual.reshape(nx, ny), 1, axis=0).ravel()
    commutator = residual(shifted)-np.roll(final_residual.reshape(nx, ny), 1, axis=0).ravel()
    initial = csv(raw/"before_phase_nodes.csv")
    report = dict(
        phase_before_solve_max=float(np.max(abs(initial["phi"]))),
        phase_min=float(nodes["phi"].min()), phase_max=float(nodes["phi"].max()),
        phase_max_along_x_range=float(np.ptp(phase_grid, axis=0).max()),
        partition_of_unity_max_error=float(np.max(abs(sums-1))),
        partition_of_unity_failing_particles=int(np.sum(bad_weights)),
        constant_gradient_failing_particles=int(np.sum(bad_gradients)),
        failing_particle_y_coordinates=np.unique(particles["y"][bad_weights]).tolist(),
        constant_gradient_max=float(np.max(np.linalg.norm(gradient_sum, axis=1))),
        linear_y_gradient_max_error=float(np.max(np.linalg.norm(linear_y_gradient-[0,1], axis=1))),
        initial_residual_norm=float(np.linalg.norm(initial_residual)),
        initial_residual_along_x_range=float(np.ptp(initial_residual.reshape(nx, ny), axis=0).max()),
        independent_first_newton_direction_along_x_range=float(np.ptp(first_direction.reshape(nx, ny), axis=0).max()),
        independently_reassembled_final_relative_residual=float(np.linalg.norm(final_residual)/np.linalg.norm(initial_residual)),
        residual_translation_commutator_relative=float(np.linalg.norm(commutator)/np.linalg.norm(initial_residual)),
        phase_periodic_constraints_verified=int(len(constraints)),
        initial_particle_volume_sum=float(volume.sum()),
        accepted_mechanics_available=False)
    # Preserve sample-level evidence, including failures in interior columns.
    np.savetxt(dest/"particle_stencil_checks.csv",
               np.column_stack((particles["id"],particles["x"],particles["y"],
                                sums,gradient_sum,linear_y_gradient)), delimiter=",",
               header="id,x,y,sum_w,sum_gx,sum_gy,gradient_x_of_y,gradient_y_of_y", comments="")

    # Save native unsmoothed nodal data and cellwise exact Q1 center gradients.
    gradients = []
    for c in topology:
        p, xy = nodes["phi"][c], points[c]
        gradients.append([(p[1]+p[2]-p[0]-p[3])/(2*(xy[1,0]-xy[0,0])),
                          (p[2]+p[3]-p[0]-p[1])/(2*(xy[3,1]-xy[0,1])), 0.])
    write_vtu(dest/"pre_mechanics_bulk.vtu", points, topology, VTK_QUAD,
              {"phi": nodes["phi"], "vertex_id": nodes["vertex"].astype(np.int64)},
              {"Q1_gradient_at_cell_center": np.array(gradients)})
    write_vtu(dest/"independent_discrete_diagnostics.vtu", points, topology, VTK_QUAD,
              {"initial_residual_after_periodic_elimination": initial_residual[canonical],
               "independent_first_Newton_direction_NOT_a_production_iterate": first_direction[canonical]})
    particle_xy = np.column_stack((particles["x"], particles["y"]))
    write_vtu(dest/"initial_particles.vtu", particle_xy, np.arange(len(particles))[:,None], VTK_VERTEX,
              {"particle_id": particles["id"].astype(np.int64), "H": H, "domain_volume": volume,
               "CPDI_sum_w": sums, "CPDI_constant_gradient_norm": np.linalg.norm(gradient_sum, axis=1)})
    shutil.copy2(raw/"pre_mechanics_fault.vtu", dest/"pre_mechanics_fault.vtu")
    fault = ET.parse(raw/"pre_mechanics_fault.vtu").getroot()
    fault_points = np.fromstring(fault.find(".//Points/DataArray").text, sep=" ").reshape(-1,3)[:,:2]
    normals = np.diff(fault_points, axis=0)
    normals = np.column_stack((-normals[:,1], normals[:,0]))/np.linalg.norm(normals, axis=1)[:,None]
    write_vtu(dest/"fault_normals.vtu", fault_points, np.column_stack((np.arange(len(normals)), np.arange(1,len(normals)+1))), VTK_LINE, {},
              {"segment_normal": np.column_stack((normals, np.zeros(len(normals))))})
    write_vtu(dest/"nominal_y_zero.vtu", np.array([[0.,0.],[.25,0.]]), [[0,1]], VTK_LINE, {})

    # Profiles contain native support/particle points, not only dense curves.
    intended = intended_profile(ys)
    locations = [0., xs[1], .125, xs[-2], .25]
    profiles = np.column_stack((ys, intended, *[phase(np.column_stack((np.full(ny,x), ys))) for x in locations]))
    np.savetxt(dest/"transverse_phi.csv", profiles, delimiter=",", header="y,intended_stationary_phi,"+",".join(f"FE_phi_x_{x:g}" for x in locations), comments="")
    np.savetxt(dest/"along_fault.csv", np.column_stack((fault_points, phase(fault_points))), delimiter=",", header="x,y,FE_phi_at_fault_vertex", comments="")
    fault_x = np.unique(np.concatenate((xs,fault_points[:,0])))
    fault_samples = np.column_stack((fault_x,np.interp(fault_x,fault_points[:,0],fault_points[:,1])))
    np.savetxt(dest/"along_fault_mesh_samples.csv", np.column_stack((fault_samples,phase(fault_samples))), delimiter=",",
               header="x,y,FE_phi_on_actual_polyline_at_bulk_and_fault_x_coordinates", comments="")
    center = phase(np.column_stack((xs, np.zeros_like(xs))))
    np.savetxt(dest/"centerline.csv", np.column_stack((xs, center)), delimiter=",", header="x,FE_phi_y_zero", comments="")
    particle_ys = np.unique(particles["y"])
    reference_phi = intended_profile(particle_ys)
    href = lambda p: 128*p*(1+p)/(1-p)**2
    reference_H = np.where(reference_phi>.1, 64*.6/href(.6)*(1+href(reference_phi))**2, .5)
    H_columns = [H[particles["y"] == y] for y in particle_ys]
    Hmin, Hmax = np.array([a.min() for a in H_columns]), np.array([a.max() for a in H_columns])
    report["initial_H_relative_reference_error"] = float(np.max(np.maximum(abs(Hmin-reference_H),abs(Hmax-reference_H))/reference_H))
    report["initial_H_along_x_range"] = float(np.max(Hmax-Hmin))
    np.savetxt(dest/"initial_H.csv", np.column_stack((particle_ys,Hmin,Hmax,reference_H)), delimiter=",", header="y,H_min_x,H_max_x,intended_initializer_H", comments="")
    (dest/"measurements.json").write_text(json.dumps(report,indent=2)+"\n")

    fig, ax = plt.subplots(2,2,figsize=(12,8), constrained_layout=True)
    m = ax[0,0].pcolormesh(xs,ys,phase_grid.T, shading="nearest", cmap="viridis")
    for x in xs: ax[0,0].axvline(x,color="w",lw=.15,alpha=.35)
    for y in ys: ax[0,0].axhline(y,color="w",lw=.15,alpha=.35)
    ax[0,0].plot(fault_points[:,0],fault_points[:,1],"r.-",label="reconstructed")
    ax[0,0].axhline(0,color="k",ls="--",lw=.7)
    fig.colorbar(m,ax=ax[0,0],label="Q1 nodal phi (unsmoothed)")
    ax[0,0].set(xlabel="x [m]",ylabel="y [m]",title="Converged pre-mechanics state")
    ax[0,1].plot(xs,center,"o-",label="phi(x,0), native x nodes")
    ax[0,1].plot(fault_points[:,0],phase(fault_points),"s-",label="phi at actual fault vertices")
    ax[0,1].set(xlabel="x [m]",ylabel="phi"); ax[0,1].legend(fontsize=8)
    for k,x in enumerate(locations): ax[1,0].plot(ys,profiles[:,k+2],".-",ms=3,label=f"FE x={x:g}")
    ax[1,0].plot(ys,intended,"k--",label="intended stationary reference (not FE constraint)")
    ax[1,0].set(xlabel="y [m]",ylabel="phi"); ax[1,0].legend(fontsize=7)
    ax[1,1].plot(fault_points[:,0],fault_points[:,1],"o-",label="actual vertices")
    ax[1,1].axhline(0,color="k",ls="--",label="nominal")
    ax[1,1].set(xlabel="x [m]",ylabel="fault y [m]"); ax[1,1].legend()
    fig.savefig(dest/"phase_and_fault.png",dpi=170); plt.close(fig)
    fig, ax = plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    ax[0].plot(particle_ys,Hmin,".",label="initial H, particle rows")
    ax[0].plot(particle_ys,reference_H,"k--",label="independent intended H")
    ax[0].set(xlabel="y [m]",ylabel="H [Pa]"); ax[0].legend(fontsize=7)
    m=ax[1].scatter(particles["x"],particles["y"],c=np.linalg.norm(gradient_sum,axis=1),s=2)
    fig.colorbar(m,ax=ax[1],label="|sum CPDI gradients| [1/m]")
    ax[1].set(xlabel="x [m]",ylabel="y [m]",title="Actual particle locations")
    ax[1].set_ylim(-.016,.016)
    for x in xs: ax[1].axvline(x,color="k",lw=.3,alpha=.3)
    ax[1].axhline(0,color="k",lw=.3)
    residual_grid=initial_residual.reshape(nx,ny)
    m=ax[2].pcolormesh(xs[:-1],ys,(residual_grid-residual_grid.mean(axis=0)).T,shading="nearest",cmap="coolwarm")
    fig.colorbar(m,ax=ax[2],label="initial residual minus its x-row mean")
    ax[2].set(xlabel="x [m]",ylabel="y [m]",title="x-dependent residual (diagnostic only)",ylim=(-.0625,.0625))
    fig.savefig(dest/"particles_and_discrete_problem.png",dpi=170); plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
