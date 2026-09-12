"""Cheap diagnostic regressions, including the preserved first K4 attempt."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from production import HERE, check
from analyze_production import q2_profile, residual_checks
from verify_exports import boundary_error


class Diagnostics(unittest.TestCase):
    def test_q2_reconstruction(self):
        x,w=np.polynomial.legendre.leggauss(3)
        y=(np.arange(8)[:,None]+(1+x)/2)/8-.5
        samples=np.linspace(-.5,.5,1001)
        np.testing.assert_allclose(q2_profile(y.ravel(),(1+2*y+3*y*y).ravel(),samples),
                                   1+2*samples+3*samples*samples,atol=2e-15)

    def test_boundary_trace(self):
        g,_=np.polynomial.legendre.leggauss(3)
        y=(np.arange(8)[:,None]+(1+g)/2)/8-.5
        xx,yy=np.meshgrid([.01,.08,.2],y.ravel())
        data=np.c_[xx.ravel(),yy.ravel(),(yy*1e-4+xx*(yy*yy-.25)).ravel(),np.zeros(xx.size)]
        self.assertLess(boundary_error(data,1e-4),1e-16)
        data[:,2]+=1e-7
        self.assertGreater(boundary_error(data,1e-4),.99e-7)

    def test_preserved_attempt(self):
        source=HERE/'attempt1-volume-guard/k42_A'
        if not source.exists(): self.skipTest('Local raw evidence not installed')
        with tempfile.TemporaryDirectory() as temporary:
            target=Path(temporary)/'k42_A'
            target.mkdir()
            for file in source.glob('*.csv'): (target/file.name).symlink_to(file)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(check(target,0),0)
                self.assertEqual(check(target,1),0)
            report=json.loads((target/'k4_guard_1.json').read_text())
            self.assertLess(report['max_normalization_error'],1e-4)
            self.assertGreater(abs(report['particle_volume_relative_error']),1e-10)
            self.assertTrue(report['checks']['positive_particle_measure'])
        result=residual_checks(source.with_suffix('.log'))
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['final']),2)


if __name__=='__main__': unittest.main()
