"""Small offline quadrature/width checks, independent of an ASPECT run."""
from pathlib import Path
import unittest

import numpy as np

from diagnose_timestep_support import ProfileIntegral, required_width
from reference import Reference, read_parameters


class SupportChecks(unittest.TestCase):
    def test_partial_cell_constant_profile(self):
        parameters = Path(__file__).resolve().parent/'smoke/parameters.prm'
        model = Reference(read_parameters(parameters), 128, 6, .00225)
        phi = np.full_like(model.y, .3)
        integral = ProfileIntegral(model, phi)
        _, h = model.degradation(.3)
        for width in (0., .0012345, model.support, .4321, .5):
            self.assertAlmostEqual(integral.strip(width), 2*width*h, places=11)

    def test_signed_nonmonotone_width_search(self):
        # A temporarily passing narrow interval must not hide a wider failure.
        error = lambda w: abs((w-.3)*(.5-w))
        width = required_width(error, .2, .001)
        self.assertGreater(width, .49)
        self.assertAlmostEqual(error(width), .001, places=11)


if __name__ == '__main__':
    unittest.main()
