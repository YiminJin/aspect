"""Checks for matched signed differences and the bounded scalar extension."""
import json
import unittest

import numpy as np

from temporal_width import DATA, OUT, load
from test_reference_check import k1


class Checks(unittest.TestCase):
    def test_new_scalar_level_and_initial_retention(self):
        for name in ('ell0','half'):
            init=json.loads((DATA/f'{name}-4096/initialization.json').read_text())['retained_initial']
            rows,u=load(name,.0625)
            self.assertEqual(len(rows),97)
            old=k1.Histories(init['q'],init['C'],init['Theta'])
            for r in rows:
                evaluated,updated=k1.solve_step(old,.0625 if r['step'] else 2.,r['U'],
                                                init['Ih'],init['Ih'],initial=r['step']==0)
                np.testing.assert_allclose([r['V'],r['q'],r['C'],r['Theta']],
                    [evaluated['V'],evaluated['stress'],evaluated['cohesive'],updated.theta],rtol=1e-12,atol=1e-12)
                if not r['step']: self.assertEqual(updated,old)
                self.assertLess(r['supported_normalization_error'],1e-4)
                self.assertEqual(r['history_integral'],0.)
                old=updated
            np.testing.assert_allclose(u[:,0],[-r['U']/2 for r in rows],atol=1e-12,rtol=0)
            np.testing.assert_allclose(u[:,-1],[r['U']/2 for r in rows],atol=1e-12,rtol=0)

    def test_signed_difference_before_norm(self):
        report=json.loads((OUT/'comparison.json').read_text())
        for pair in (report['saved_pair'],report['new_pair']):
            c={n:load(n,pair['coarse_dt']) for n in ('ell0','half')}
            f={n:load(n,pair['fine_dt']) for n in ('ell0','half')}
            for row in pair['rows']:
                i=round(row['time_s']/pair['coarse_dt']); j=2*i
                e0=c['ell0'][1][i]-f['ell0'][1][j]
                e1=c['half'][1][i]-f['half'][1][j]
                self.assertEqual(float(np.max(abs(e1-e0))),row['metrics']['velocity_max']['matched_temporal_change'])
                for key in ('V','q','C','Theta','slip'):
                    change=(c['half'][0][i][key]-c['ell0'][0][i][key])-(f['half'][0][j][key]-f['ell0'][0][j][key])
                    self.assertEqual(abs(change),row['metrics'][key]['matched_temporal_change'])

    def test_contraction_uses_identical_times_and_no_readiness_override(self):
        report=json.loads((OUT/'comparison.json').read_text())
        self.assertEqual(len(report['saved_pair']['rows']),25)
        self.assertEqual(len(report['new_pair']['rows']),49)
        for values in report['contraction_same_times'].values():
            self.assertEqual([r['time_s'] for r in values['pointwise']],[k*.25 for k in range(25)])
            self.assertEqual(values['new_max_same_times'],max(r['new_change'] for r in values['pointwise']))
        self.assertFalse(report['original_K41_ready'])
        self.assertFalse(json.loads((DATA/'decision.json').read_text())['ready'])
        self.assertTrue(report['proposed_criterion_only'])


if __name__=='__main__': unittest.main()
