"""Test persistent qualification without a first-pass or final-time shortcut."""
import json
import unittest

from post_transient import OUT, persistent_start


class Checks(unittest.TestCase):
    def test_persistent_suffix_not_first_pass(self):
        self.assertEqual(persistent_start([0,1,2,3],[0.,.8,1.2,.9]),3)
        self.assertEqual(persistent_start([0,1,2],[1.,.9,0.]),0)
        self.assertIsNone(persistent_start([0,1,2],[0.,.9,1.1]))

    def test_every_saved_time_after_start(self):
        result=json.loads((OUT/'post-transient.json').read_text())
        rows=json.loads((OUT/'comparison.json').read_text())['new_pair']['rows']
        for key,value in result['qualification'].items():
            if key=='Ih': continue
            start=value['t_qual_s']
            self.assertTrue(all(r['metrics'][key]['proposed_ratio']<=1 for r in rows if r['time_s']>=start))
            if start:
                preceding=[r for r in rows if r['time_s']<start][-1]
                self.assertGreater(preceding['metrics'][key]['proposed_ratio'],1)
        self.assertEqual(result['common_interval_s'],[4.,6.])
        self.assertEqual(result['samples_in_common_interval'],17)
        self.assertFalse(result['original_K41_all_time_pass'])
        self.assertIsNone(result['older_pair_qualification']['slip'])


if __name__=='__main__': unittest.main()
