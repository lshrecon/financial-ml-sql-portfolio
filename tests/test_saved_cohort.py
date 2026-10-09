import unittest
import numpy as np
import pandas as pd
from evaluate_saved_cohort import prepare_cohort, select_winner


class SavedCohortTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame([
            {'연도': 2002, '기업명': 'unknown', '매수가': 100, '매도가': np.nan, '구매주식수': 9, '투자원금': 1000, '매수일': '2002-04-01', '매도일': '2003-03-31', 'feature': 1},
            {'연도': 2002, '기업명': 'flat', '매수가': 100, '매도가': 100, '구매주식수': 9, '투자원금': 1000, '매수일': '2002-04-01', '매도일': '2003-03-31', 'feature': 2}
        ])
        self.protocol = {'expected_missing': [[2002, 'unknown']], 'features': ['feature']}

    def test_unknown_is_quarantined_flat_is_retained_as_nonpositive(self):
        result, rejected = prepare_cohort(self.raw, self.protocol)
        self.assertEqual(rejected['기업명'].tolist(), ['unknown'])
        self.assertEqual(result['기업명'].tolist(), ['flat'])
        self.assertEqual(result['수익률_양음'].tolist(), [0])
        self.assertEqual(result['수익'].tolist(), [0])
        self.assertTrue(pd.isna(self.raw.loc[0, '매도가']))

    def test_unreviewed_missing_exit_stops(self):
        self.raw.loc[1, '매도가'] = np.nan
        with self.assertRaisesRegex(ValueError, 'differ'):
            prepare_cohort(self.raw, self.protocol)

    def test_infinite_feature_stops_instead_of_dropping(self):
        self.raw['feature'] = self.raw['feature'].astype(float)
        self.raw.loc[1, 'feature'] = np.inf
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            prepare_cohort(self.raw, self.protocol)

    def test_unknown_dates_stop(self):
        for value in ['not a date', None]:
            with self.subTest(value=value):
                self.raw.loc[1, '매도일'] = value
                with self.assertRaises(ValueError):
                    prepare_cohort(self.raw, self.protocol)

    def test_predeclared_selection_metric_not_accuracy(self):
        values = {'baseline': {'f1': .60, 'accuracy': .44}, 'tree': {'f1': .53, 'accuracy': .57}}
        self.assertEqual(select_winner(values, ['baseline', 'tree']), 'baseline')


if __name__ == '__main__':
    unittest.main()
