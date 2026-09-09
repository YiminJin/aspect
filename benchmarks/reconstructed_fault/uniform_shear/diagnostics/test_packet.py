"""Integrity checks for this saved diagnostic packet, not production acceptance."""
import json
from pathlib import Path
import unittest

import numpy as np
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
from vtkmodules.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).parent/"results"


class DiagnosticPacket(unittest.TestCase):
    def test_paraview_files_are_readable_and_finite(self):
        expected = {"pre_mechanics_bulk": (1105,1024),
                    "independent_discrete_diagnostics": (1105,1024),
                    "initial_particles": (9216,9216),
                    "pre_mechanics_fault": (9,8), "fault_normals": (9,8),
                    "nominal_y_zero": (2,1)}
        for name, counts in expected.items():
            reader = vtkXMLUnstructuredGridReader()
            reader.SetFileName(str(ROOT/(name+".vtu")))
            reader.Update()
            data = reader.GetOutput()
            self.assertEqual((data.GetNumberOfPoints(),data.GetNumberOfCells()),counts)
            for attributes in (data.GetPointData(),data.GetCellData()):
                for i in range(attributes.GetNumberOfArrays()):
                    self.assertTrue(np.isfinite(vtk_to_numpy(attributes.GetArray(i))).all())

    def test_initial_inputs_remained_unchanged_through_phase_solve(self):
        for name in ("particles","cpdi","cells"):
            before=(ROOT/"raw"/f"before_phase_{name}.csv").read_bytes()
            after=(ROOT/"raw"/f"pre_mechanics_{name}.csv").read_bytes()
            self.assertEqual(before,after)

    def test_recorded_diagnostic_not_an_accepted_mechanical_state(self):
        report=json.loads((ROOT/"measurements.json").read_text())
        self.assertFalse(report["accepted_mechanics_available"])
        self.assertEqual(report["phase_before_solve_max"],0.)
        self.assertLess(report["independently_reassembled_final_relative_residual"],1e-8)
        self.assertEqual(report["phase_periodic_constraints_verified"],65)
        # Verify that the packet actually contains the reported defect.
        # These are reproduction checks, not acceptable CPDI error limits.
        self.assertEqual(report["partition_of_unity_failing_particles"],68)
        self.assertGreater(report["partition_of_unity_max_error"],.3)


if __name__ == "__main__":
    unittest.main()
