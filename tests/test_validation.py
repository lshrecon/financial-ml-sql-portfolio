import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_validation import recalculate_returns, temporal_split, expanding_year_folds


class AccountingInputs(unittest.TestCase):
    def row(self, sale=27., shares=3, capital=100., buy=30.):
        return pd.DataFrame({'매수가': [buy], '매도가': [sale],
                             '구매주식수': [shares], '투자원금': [capital]})

    def test_old_positive_label_becomes_loss(self):
        result = recalculate_returns(self.row()).iloc[0]
        self.assertEqual(result['매출'], 91.)
        self.assertEqual(result['수익'], -9.)
        self.assertEqual(result['수익률'], -9.)
        self.assertEqual(result['수익률_양음'], 0)

    def test_uninvested_portion_is_cash_not_profit(self):
        result = recalculate_returns(self.row(shares=0)).iloc[0]
        self.assertEqual(result['매출'], 100.)
        self.assertEqual(result['수익'], 0.)

    def test_exact_allocation_and_zero_terminal_price(self):
        result = recalculate_returns(self.row(sale=0., capital=90.)).iloc[0]
        self.assertEqual(result['잔금'], 0.)
        self.assertEqual(result['수익률'], -100.)

    def test_invalid_values_are_not_silently_labelled_as_losses(self):
        for changes in ({'buy': 0.}, {'sale': np.nan}, {'sale': np.inf},
                        {'shares': 3.5}, {'shares': -1}, {'capital': 80.}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                recalculate_returns(self.row(**changes))


class TemporalEvaluation(unittest.TestCase):
    def frame(self):
        years = np.repeat(np.arange(2010, 2024), 2)
        return pd.DataFrame({'연도': years, '기업명': ['A', 'B'] * 14,
                             '매수일': [f'{y}-01-04' for y in years],
                             '매도일': [f'{y}-12-30' for y in years]})

    def test_three_periods_are_disjoint_and_cover_all_rows(self):
        frame = self.frame()
        train, valid, test = temporal_split(frame)
        self.assertEqual(len(train) + len(valid) + len(test), len(frame))
        self.assertLess(train['연도'].max(), valid['연도'].min())
        self.assertLess(valid['연도'].max(), test['연도'].min())
        self.assertEqual(test['연도'].min(), 2021)

    def test_year_fold_never_sees_same_or_future_year(self):
        frame = self.frame().sample(frac=1, random_state=7)
        for train, valid in expanding_year_folds(frame):
            self.assertLess(frame.iloc[train]['연도'].max(), frame.iloc[valid]['연도'].min())
            self.assertEqual(frame.iloc[valid]['연도'].nunique(), 1)
            self.assertTrue(set(train).isdisjoint(valid))

    def test_label_not_yet_observable_is_rejected(self):
        frame = self.frame()
        frame.loc[frame['연도'] == 2016, '매도일'] = '2017-03-31'
        with self.assertRaisesRegex(ValueError, 'overlap'):
            temporal_split(frame)

    def test_duplicate_missing_year_and_missing_period_are_rejected(self):
        frame = self.frame()
        broken_year = frame.copy()
        broken_year.loc[0, '연도'] = np.nan
        for broken in (pd.concat([frame, frame.iloc[:1]]), broken_year,
                       frame[frame['연도'] <= 2020]):
            with self.subTest(rows=len(broken)), self.assertRaises(ValueError):
                temporal_split(broken)


if __name__ == '__main__':
    unittest.main()
