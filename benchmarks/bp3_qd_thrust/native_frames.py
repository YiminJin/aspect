"""Read the native PVD/PVTU, retaining the four ordered corner traces of each patch."""
import argparse
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOXML import vtkXMLPUnstructuredGridReader


def frames(directory):
    entries = ET.parse(directory / "solution.pvd").findall(".//DataSet")
    result = {float(e.attrib["timestep"]): directory/e.attrib["file"] for e in entries}
    if not entries or len(entries) != len(result) or list(result) != sorted(result):
        raise ValueError(f"Empty, repeated or unordered frames in {directory}")
    return result


def read(path):
    pieces = [path.parent/e.attrib["Source"] for e in ET.parse(path).findall(".//Piece")]
    if not pieces or not all(f.is_file() for f in pieces):
        raise ValueError(f"Missing piece of {path}")
    reader = vtkXMLPUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()
    cells = grid.GetCells()
    offsets = vtk_to_numpy(cells.GetOffsetsArray())
    np.testing.assert_array_equal(np.diff(offsets), 4)
    corners = vtk_to_numpy(cells.GetConnectivityArray()).reshape(-1,4)
    xyz = vtk_to_numpy(grid.GetPoints().GetData())[corners]
    # Discontinuous values sharing a coordinate must retain their cell side.
    boxes = np.column_stack((xyz[:,:,:2].min(axis=1),xyz[:,:,:2].max(axis=1)))
    order = np.lexsort(boxes.T)
    keys = boxes[order]
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError("Duplicate native patch")
    arrays = {}
    for name in ("phase_field","velocity","p","fault_tau_xx","fault_tau_yy","fault_tau_xy"):
        array = grid.GetPointData().GetArray(name)
        if array is None:
            raise ValueError(f"Missing {name} in {path}")
        values = vtk_to_numpy(array)[corners][order]
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite {name} in {path}")
        arrays[name] = values.astype(float)
    return keys, arrays, sum(f.stat().st_size for f in pieces)+path.stat().st_size


def compare(first, second, mesh_only=False):
    a,b = frames(first),frames(second)
    if not mesh_only and list(a) != list(b):
        raise ValueError(f"Different frame clocks: {list(a)}, {list(b)}")
    for time in ([0.] if mesh_only else a):
        ka,va,_ = read(a[time])
        kb,vb,size = read(b[time])
        np.testing.assert_array_equal(ka,kb)
        errors = {}
        for name in (["phase_field"] if mesh_only else va):
            tol = {"phase_field":0., "velocity":1.e-16, "p":1.e-5}.get(name,1.e-5)
            np.testing.assert_allclose(va[name],vb[name],atol=tol,rtol=0. if mesh_only else 2.e-6)
            errors[name] = np.max(np.abs(va[name]-vb[name])).item()
        print(f"{time:g} s, {len(ka)} ordered patches, {size} bytes, max errors {errors}")
        if mesh_only:
            widths,counts = np.unique(ka[:,2]-ka[:,0],return_counts=True)
            print("Q2 mesh by active cell width (m), count:", list(zip(2*widths, counts//4)))
            print("Finest cell height from Float32 coordinates (m):", 2*np.min(ka[:,3]-ka[:,1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first",type=Path)
    parser.add_argument("second",type=Path)
    parser.add_argument("--mesh-only",action="store_true")
    args = parser.parse_args()
    compare(args.first,args.second,args.mesh_only)
