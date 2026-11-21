"""Check a retained or replayed native closeout control against its frozen references."""
import argparse
from pathlib import Path

import numpy as np

from compare import fields, read


def mature(expected, output):
    largest = 0.
    for path in sorted(expected.glob('*.csv')):
        old, new = read(path), read(output/path.name)
        assert old.dtype.names == new.dtype.names and len(old) == len(new)
        for name in old.dtype.names:
            if name == 'rank_peak_rss_kib':
                continue
            if old[name].dtype.kind in 'OUS':
                np.testing.assert_array_equal(old[name],new[name])
                continue
            if old[name].dtype.kind in 'iu':
                np.testing.assert_array_equal(old[name],new[name])
            else:
                np.testing.assert_allclose(old[name],new[name],rtol=1e-8,atol=1e-10)
            largest = max(largest,max(abs(old[name]-new[name])))
    print(f'Mature control passed. Largest absolute column difference {largest:.6g}.')


def propagation(expected, output):
    row = read(output/'bulk-accepted.csv')
    np.testing.assert_array_equal(row['step'],[1,2,3])
    np.testing.assert_array_equal(row['step'],row['publications'])
    assert max(row['mechanical_residual']) <= 1e-9 and max(row['phase_KKT']) <= 1e-7
    assert max(abs(read(output/'tip-power.csv')['power_adjoint_error'])) < 1e-12
    assert read(output/'bulk-tangent.csv')['relative_action_error'][0] < 2e-5
    for label in ('trial keeps complete accepted parent','rollback restores fields and all histories',
                  'retry is deterministic','second rollback unchanged','single publication'):
        assert label+',1' in (output/'tip-checks.csv').read_text().splitlines()
    p, q = [read(f/'tip-planes-0003.csv') for f in (expected,output)]
    for name in ('section','formed','parent_formed'):
        np.testing.assert_array_equal(p[name],q[name])
    for name in ('xi','ds','age','slip'):
        np.testing.assert_allclose(p[name],q[name],rtol=1e-7,atol=1e-11)
    old_points, new_points = (np.sort(fields(f,3),order=['cell','q']) for f in (expected,output))
    for name in ('cell','q','section'):
        np.testing.assert_array_equal(old_points[name],new_points[name])
    for name, tol in [('phi',1e-9),('tauxx',1e-6),('tauyy',1e-6),('tauzz',1e-6),('tauxy',1e-6),
                      ('Axx',1e-10),('Ayy',1e-10),('Azz',1e-10),('Axy',1e-10)]:
        np.testing.assert_allclose(old_points[name],new_points[name],rtol=1e-7,atol=tol)
    old = read(expected/'bulk-accepted.csv')
    np.testing.assert_allclose(row['time'],old['time'],atol=1e-15,rtol=0)
    for key in ('bulk','cohesive','Wext','Df','phase_remainder'):
        np.testing.assert_allclose(row[key],old[key],rtol=1e-6,atol=1e-11)
    # Test added fracture at its own scale. Total stored fracture is O(1).
    np.testing.assert_allclose(np.diff(row['fracture']),np.diff(old['fracture']),rtol=1e-6,atol=1e-11)
    print('Localized propagation tangent, rollback, power and endpoint control passed.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference',type=Path)
    p.add_argument('case',choices=['mature','localized-onecut'])
    p.add_argument('output',type=Path)
    a = p.parse_args()
    if a.case == 'mature':
        mature(a.reference/'mature/expected',a.output)
    else:
        propagation(a.reference/'runs'/a.case/'output',a.output)
