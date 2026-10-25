import unittest

from evaluation.solution import quality_gate, score


class StrictEdges(unittest.TestCase):
    def rows(self, n=50):
        return [dict(id=str(i), expected='A', choices=['A', 'B'], slices=[f's{i%3}']) for i in range(n)]

    def report(self):
        rows = self.rows()
        result = score(rows, [dict(id=r['id'], raw='{"answer":"A"}', error=None) for r in rows])
        result.update(status='complete', model=dict(kind='base', name='fixture', revision='a'*40),
                      protocol_sha256='p', golden_sha256='g', baseline_spec_sha256='b',
                      behavioral=score(self.rows(3), [dict(id=str(i), raw='{"answer":"A"}', error=None) for i in range(3)]))
        return result

    def test_duplicate_json_key_rejected(self):
        result = score(self.rows(1), [dict(id='0', raw='{"answer":"B","answer":"A"}', error=None)])
        self.assertEqual(result['overall']['valid'], 0)

    def test_generation_error_with_correct_json_stays_in_denominator(self):
        result = score(self.rows(1), [dict(id='0', raw='{"answer":"A"}', error='timeout')])
        self.assertEqual((result['overall']['n'], result['overall']['correct'], result['overall']['errors']), (1, 0, 1))

    def test_changed_group_membership_rejected_with_same_group_counts(self):
        baseline, candidate = self.report(), self.report()
        candidate['per_item'][0]['slices'], candidate['per_item'][1]['slices'] = candidate['per_item'][1]['slices'], candidate['per_item'][0]['slices']
        with self.assertRaises(ValueError):
            quality_gate(baseline, candidate, ['s0'])

    def test_unpinned_model_rejected(self):
        baseline, candidate = self.report(), self.report()
        candidate['model']['revision'] = 'main'
        with self.assertRaises(ValueError):
            quality_gate(baseline, candidate, ['s0'])

    def test_infinite_and_fractional_counts_rejected(self):
        for value in [float('inf'), 50.0, True]:
            baseline, candidate = self.report(), self.report()
            candidate['overall']['n'] = value
            with self.assertRaises(ValueError):
                quality_gate(baseline, candidate, ['s0'])


if __name__ == '__main__':
    unittest.main()
