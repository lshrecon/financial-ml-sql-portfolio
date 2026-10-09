import unittest
import numpy as np
import pandas as pd
import KSE_가치팩터_functions as functions


class FeatureAlignment(unittest.TestCase):
    def test_average_matches_year_keys_not_row_positions(self):
        a = pd.DataFrame({'Year': [2020, 2021], 'A': [10., 20.]})
        b = pd.DataFrame({'Year': [2021, 2020], 'A': [40., 30.]})
        actual = functions.calculate_weighted_average([a, b])
        self.assertEqual(actual['Year'].tolist(), [2020, 2021])
        self.assertEqual(actual['A'].tolist(), [20., 30.])
        self.assertEqual(b['Year'].tolist(), [2021, 2020])

    def test_average_rejects_unequal_year_or_company_coverage(self):
        a = pd.DataFrame({'Year': [2020, 2021], 'A': [10., 20.]})
        for b in [pd.DataFrame({'Year': [2020], 'A': [30.]}),
                  pd.DataFrame({'Year': [2020, 2021], 'B': [30., 40.]}),
                  pd.DataFrame({'Year': [2020, 2020], 'A': [30., 40.]})]:
            with self.subTest(columns=list(b)), self.assertRaises(ValueError):
                functions.calculate_weighted_average([a, b])

    def test_endpoint_average_uses_dates_not_storage_order(self):
        prices = pd.DataFrame({'Symbol Name': pd.to_datetime(['2020-01-31','2020-01-02','2020-03-31','2020-03-02']),
                               'A': [150., 100., 160., 180.]})
        result = functions.calculate_q1_average_for_all_companies(prices, 2020, ['A'])
        self.assertEqual(result['A'], 130.)

    def test_missing_feature_is_not_ranked_as_if_observed(self):
        frame = pd.DataFrame({'Year': [2020], 'A': [2.], 'B': [np.nan]})
        result = functions.top_companies_by_inverse_Rank(frame, 2020, 10)
        self.assertEqual(result['Company'].tolist(), ['A'])
        self.assertEqual(result['Rank'].tolist(), [2.])

    def test_duplicate_feature_year_is_not_silently_first_row(self):
        frame = pd.DataFrame({'Year': [2020, 2020], 'A': [2., 100.]})
        with self.assertRaises(ValueError):
            functions.top_companies_by_inverse_Rank(frame, 2020, 1)


class HistoricalMetricDefinitions(unittest.TestCase):
    def test_annual_return_can_include_initial_capital_explicitly(self):
        result = functions.calculate_annual_compound_return({2020:110, 2021:121}, initial_value=100, periods=2)
        self.assertAlmostEqual(result, .1)

    def test_drawdown_follows_years_not_dictionary_insertion_order(self):
        result, year = functions.calculate_mdd_with_year({2021:80, 2020:100})
        self.assertAlmostEqual(result, .2)
        self.assertEqual(year, 2021)


if __name__ == '__main__':
    unittest.main()
