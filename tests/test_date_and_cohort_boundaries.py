import sys
import importlib
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
kf = importlib.import_module('KSE_가치팩터_functions')


class BoundaryCases(unittest.TestCase):
    def rank(self, year=2020):
        return pd.DataFrame({'Company': ['A'], 'Year': [year], 'Rank': [27]})

    def test_feature_year_can_precede_purchase_year_without_losing_capital(self):
        prices = pd.DataFrame({'A': [100., 120.]}, index=pd.to_datetime(['2021-04-01', '2022-03-31']))
        totals, positions = kf.process_financial_indicator([2020], self.rank(), 1000.,
                                                          (2021, 4), (2022, 3), prices)
        self.assertEqual(totals[2020], 1200.)
        self.assertEqual(positions[2020]['구매주식수'].tolist(), [10])

    def test_date_column_input_uses_converted_index(self):
        prices = pd.DataFrame({'Date': ['2021-04-01', '2022-03-31'], 'A': [100., 120.]})
        result = kf.Price_for_Purchase_Sale(2021, 4, 2022, 3, self.rank(), prices)
        self.assertEqual(result['매수가'].iloc[0], 100.)

    def test_absent_month_missing_price_and_duplicate_date_are_rejected(self):
        for dates, values in [(['2021-03-31', '2022-03-31'], [100., 120.]),
                              (['2021-04-01', '2022-03-31'], [100., np.nan]),
                              (['2021-04-01', '2021-04-01', '2022-03-31'], [100., 101., 120.])]:
            prices = pd.DataFrame({'A': values}, index=pd.to_datetime(dates))
            with self.subTest(dates=dates), self.assertRaises(ValueError):
                kf.Price_for_Purchase_Sale(2021, 4, 2022, 3, self.rank(), prices)


if __name__ == '__main__':
    unittest.main()
