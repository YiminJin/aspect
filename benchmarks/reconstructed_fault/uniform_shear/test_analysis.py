"""Small checks of the independent diagnostic calculations, not ASPECT runs."""
import unittest
import tempfile
from pathlib import Path

import numpy as np

from analyze import boundary_error, read


class AnalysisChecks(unittest.TestCase):
    def test_distributed_rows_and_replicated_fields_are_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for rank in range(2):
                np.savetxt(directory/f"bulk_rank{rank}_0.csv", [[rank, 3.]],
                           delimiter=",", header="x,value", comments="")
                np.savetxt(directory/f"surface_rank{rank}_0.csv", [[0., 7.]],
                           delimiter=",", header="x,value", comments="")
                np.savetxt(directory/f"surface_weak_rank{rank}_0.csv", [[0., .1, 7.]],
                           delimiter=",", header="node,Mdiag,q", comments="")
            self.assertEqual(len(read(directory,"bulk",0,("x",))),2)
            self.assertEqual(len(read(directory,"surface",0,("x",))),1)
            self.assertEqual(len(read(directory,"surface_weak",0,("Mdiag",))),1)
            np.savetxt(directory/"surface_rank1_0.csv", [[0., 8.]],
                       delimiter=",", header="x,value", comments="")
            with self.assertRaisesRegex(ValueError,"Inconsistent replicated"):
                read(directory,"surface",0,("x",))
            np.savetxt(directory/"surface_weak_rank1_0.csv", [[0., .2, 7.]],
                       delimiter=",", header="node,Mdiag,q", comments="")
            with self.assertRaisesRegex(ValueError,"Inconsistent replicated"):
                read(directory,"surface_weak",0,("Mdiag",))

    def test_boundary_trace_uses_q2_not_nearest_sample(self):
        gauss = np.polynomial.legendre.leggauss(3)[0]
        data = np.zeros(18, dtype=[(field, float) for field in ("x", "y", "ux", "uy")])
        loading = 1e-4
        i = 0
        for x in (gauss+1)/2:
            for bottom in (-.5, .4):
                for eta in (gauss+1)/2:
                    y = bottom+.1*eta
                    data[i] = (x, y, y*loading+(y*y-.25)*.002, (y*y-.25)*.003)
                    i += 1
        self.assertLess(boundary_error(data, loading), 1e-18)
        data["ux"] += 2e-5
        self.assertAlmostEqual(boundary_error(data, loading), 2e-5, delta=1e-18)


if __name__ == "__main__":
    unittest.main()
