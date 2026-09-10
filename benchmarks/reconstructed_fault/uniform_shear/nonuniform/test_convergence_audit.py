import unittest

import numpy as np

from audit_convergence import FIELDS, differences, samples


def field(x, offset, slope):
    data = np.zeros(len(x), dtype=[('s', float)]+[(name, float) for name in FIELDS])
    data['s'] = x
    for name in FIELDS:
        data[name] = offset+slope*x
    return data


class ConvergenceAudit(unittest.TestCase):
    def test_exact_second_moment_not_squared_mean(self):
        x, w = samples(np.array([0., .03, .25]))
        self.assertAlmostEqual(np.sum(w), .25)
        self.assertAlmostEqual(np.dot(w, x*x), .25**3/3)
        self.assertGreater(np.dot(w, x*x)/sum(w), (np.dot(w, x)/sum(w))**2)

    def test_initial_offset_and_endpoint_changes_are_separate(self):
        a = np.array([0., .1, .25]); b = np.array([0., .04, .15, .25])
        result = differences(field(a, 5, 4), field(b, 1, 2),
                             field(a, 3, 4), field(b, 1, 2))
        for value in result.values():
            self.assertAlmostEqual(value['total']['mean'], 4.25)
            self.assertAlmostEqual(value['anomaly']['L2'], .5/np.sqrt(12))
            self.assertAlmostEqual(value['change_from_initial_difference']['L2'], 2.)
            self.assertAlmostEqual(value['anomaly_change_from_initial_difference']['L2'], 0.)
            self.assertAlmostEqual(value['controls'][0]['total'], 4.)
            self.assertAlmostEqual(value['controls'][-1]['anomaly'], .25)
        self.assertAlmostEqual(result['V']['maximum_log10_ratio'], np.log10(5.))

    def test_logarithmic_rate_diagnostic_rejects_nonpositive_states(self):
        x = np.array([0., .25])
        with self.assertRaisesRegex(ValueError, 'positive states'):
            differences(field(x, 0, 1), field(x, 1, 0),
                        field(x, 1, 0), field(x, 1, 0))


if __name__ == '__main__':
    unittest.main()
