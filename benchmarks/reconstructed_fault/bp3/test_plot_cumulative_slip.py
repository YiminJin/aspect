"""Small plotting-data checks; no ASPECT simulation or graphical backend needed."""
import csv
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from plot_cumulative_slip import HEADER, Profile, read_profiles, select_profiles
from export_profile_slip import export
from slip_history import restore_profile_payloads
from check_cumulative_slip import check


class SlipContours(unittest.TestCase):
    def test_sparse_profiles_velocity_export_and_restart_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'profiles').mkdir()
            (root/'profiles.csv').write_text('step,time_s,file\n0,0,profiles/fault_0.csv\n'
                                             '9,1000,profiles/fault_9.csv\n')
            for step,time in [(0,0),(9,1000)]:
                with (root/f'profiles/fault_{step}.csv').open('w') as out:
                    out.write('step,time_s,fault,node,xd_m,x_m,y_m,slip_m,V_m_per_s\n')
                    for node in range(3):
                        # Average slip/dt is 1e-5, instantaneous V is seismic.
                        out.write(f'{step},{time},0,{node},{node*1000},{node*1000},0,'
                                  f'{.01 if step else 0},{.1 if step else 1e-9}\n')
            profiles=list(read_profiles(root))
            _,selected,summary=select_profiles(profiles)
            self.assertEqual(selected[-1]['regime'],'coseismic')
            self.assertEqual(selected[-1]['max_rate_m_s'],.1)
            self.assertTrue(summary['saved_profiles_only'])
            (root/'accepted_steps.csv').write_text('step,time,dt,max_V\n'
                                                  '0,0,0,1e-9\n9,1000,10,.1\n')
            with contextlib.redirect_stdout(io.StringIO()):
                check(root)
            verification=json.loads((root/'profile_verification.json').read_text())
            self.assertEqual(verification['saved_profiles'],2)
            self.assertEqual(verification['adjacent_slip_updates_checked'],0)
            self.assertFalse(verification['unsaved_slip_updates_checked'])
            legacy=root/'legacy.csv'
            self.assertEqual(export(root/'profiles.csv',legacy),2)
            _,saved,summary=select_profiles(read_profiles(legacy))
            self.assertTrue(all(p['regime']=='unknown' for p in saved))
            self.assertTrue(summary['saved_profiles_only'])
            self.assertFalse(summary['instantaneous_velocity_available'])
            branch=root/'branch';branch.mkdir()
            (branch/'profiles.csv').write_text('step,time_s,file\n0,0,profiles/fault_0.csv\n')
            restore_profile_payloads(root,branch,4)
            self.assertTrue((branch/'profiles/fault_0.csv').is_file())
            self.assertFalse((branch/'profiles/fault_9.csv').exists())
            with self.assertRaisesRegex(ValueError,'overwrite'):
                restore_profile_payloads(root,branch,4)
            (branch/'profiles.csv').write_text((root/'profiles.csv').read_text())
            with self.assertRaisesRegex(ValueError,'newer'):
                restore_profile_payloads(root,branch,4)
            (root/'profiles/fault_9.csv').unlink()
            with self.assertRaisesRegex(ValueError,'Missing indexed profile'):
                list(read_profiles(root))

    def test_compact_slip_with_full_precision_time_and_geometry(self):
        # At event times separated by milliseconds after centuries, rounding
        # time to the slip precision would merge distinct accepted profiles.
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'slip.csv'
            original=[]
            with path.open('w') as stream:
                stream.write(','.join(HEADER)+'\n')
                for step in range(3):
                    slips=[1.234567890123+step*1e-6,2.345678901234+step*2e-6]
                    original.append(slips)
                    for node,slip in enumerate(slips):
                        stream.write(f'{step},{(5e9+step*.001):.17g},0,{node},'
                                     f'{(node*19.9821234567):.17g},{(node*19.9821234567):.17g},'
                                     f'{slip:.10g}\n')
            profiles=list(read_profiles(path))
            self.assertEqual(len(profiles),3)
            for profile,slips in zip(profiles,original):
                np.testing.assert_allclose(profile.slip,slips,rtol=5e-10,atol=0.)
            self.assertGreater(profiles[1].time,profiles[0].time)

    def profile(self, step, time, slip):
        return Profile(step, time, np.arange(3), np.array([50000., 20000., 0.]), np.array(slip))

    def test_increment_not_time_or_change_of_maximum(self):
        profiles = [self.profile(0, 0., [0., 0., 0.]),
                    self.profile(1, 1., [0., 1., 0.]),
                    self.profile(2, 2., [0., 1., 0.15]),
                    self.profile(3, 100., [0., 1., 0.16])]
        _, chosen, _ = select_profiles(profiles, increment=.1, seismic_rate=10.)
        self.assertEqual([p['step'] for p in chosen], [0, 1, 2, 3])

    def test_seismic_classification_uses_full_fault_and_keeps_transition(self):
        profiles = [self.profile(0, 0., [0., 0., 0.]),
                    self.profile(1, 100., [.01, .01, .01]),
                    self.profile(2, 200., [.02, .02, .02]),
                    self.profile(3, 201., [.03, .021, .021]),
                    self.profile(4, 301., [.04, .022, .022])]
        _, chosen, summary = select_profiles(profiles, increment=1.)
        self.assertEqual([p['step'] for p in chosen], [0, 1, 2, 3, 4])
        self.assertEqual(chosen[3]['regime'], 'coseismic')
        self.assertEqual(summary['coseismic_accepted_profiles'], 1)
        self.assertAlmostEqual(chosen[3]['max_rate_m_s'], .01)

    def test_skips_small_changes_but_keeps_last_and_exact_recorded_values(self):
        profiles = [self.profile(i, i*1000., [i*.04, i*.04, i*.02]) for i in range(7)]
        xd, chosen, _ = select_profiles(profiles, increment=.1)
        self.assertEqual([p['step'] for p in chosen], [0, 1, 4, 6])
        np.testing.assert_array_equal(xd, [0., 20000.])
        np.testing.assert_array_equal(chosen[-1]['slip'], [.12, .24])

    def test_stream_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'slip.csv'
            rows = [[k, k*100, 0, n, n*10, 20-n*10, k*.1] for k in range(3) for n in range(3)]
            def write(data):
                with path.open('w', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(HEADER)
                    writer.writerows(data)
            write(rows)
            self.assertEqual(len(list(read_profiles(path))), 3)
            write(rows[:3] + rows)
            with self.assertWarnsRegex(UserWarning, 'identical duplicate'):
                self.assertEqual(len(list(read_profiles(path))), 3)
            conflicting = [r.copy() for r in rows[:3]]
            conflicting[0][-1] = 4.
            write(conflicting + rows)
            with self.assertRaisesRegex(ValueError, 'Conflicting duplicate'):
                list(read_profiles(path))
            write(rows[:-1])
            with self.assertRaisesRegex(ValueError, 'Incomplete profile'):
                list(read_profiles(path))
            write(rows[:3] + rows[6:])
            with self.assertRaisesRegex(ValueError, 'consecutive steps'):
                list(read_profiles(path))
            write(rows + rows[3:6])
            with self.assertRaisesRegex(ValueError, 'consecutive steps'):
                list(read_profiles(path))


if __name__ == '__main__':
    unittest.main()
