"""Regressions found when reviewing the 2026-10-08 feature changes."""
import unittest

import numpy as np
import pandas as pd

import value_factor_backtest as functions
from ml_validation import add_price_direction_targets, recalculate_returns


class IndependentReviewRegressions(unittest.TestCase):
    def test_no_observed_features_preserves_empty_targets(self):
        features = pd.DataFrame({'Year': [2020], 'A': [np.nan]})
        selected = functions.top_companies_by_inverse_Rank(features, 2020, 1)
        prices = pd.DataFrame({
            'Symbol Name': pd.to_datetime(['2020-04-01', '2021-03-31']),
            'A': [100., 120.],
        })
        purchase = functions.Price_for_Purchase_Sale(2020, 4, 2021, 3, selected, prices)
        shares = functions.Purchase_share(purchase, selected, 5_000_000 * len(selected))
        positions = pd.merge(shares, purchase.drop_duplicates(['기업명', '매수일']),
                             on='기업명', how='left')
        accounted = recalculate_returns(positions)
        before = accounted.copy(deep=True)
        result = add_price_direction_targets(accounted)
        self.assertEqual(len(result), 0)
        self.assertTrue({'가격변화율', '가격상승여부'}.issubset(result.columns))
        self.assertTrue(pd.api.types.is_float_dtype(result['가격변화율']))
        self.assertTrue(pd.api.types.is_integer_dtype(result['가격상승여부']))
        pd.testing.assert_frame_equal(accounted, before)

    def test_pbr_preserves_own_years_when_per_range_differs(self):
        bps = pd.DataFrame({'Name': ['A', 'A'], 'Year': [2020, 2021],
                            'BPS': [20., 30.]})
        prices = pd.DataFrame({'Year': [2021, 2022], 'A': [100., 240.]})
        result = functions.calculate_price_ratios(bps, 'BPS', prices, ['A'])
        expected = pd.DataFrame({'Year': [2021, 2022], 'A': [5., 8.]})
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)


if __name__ == '__main__':
    unittest.main()
