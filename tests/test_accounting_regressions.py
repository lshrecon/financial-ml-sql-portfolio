import importlib.util
import os
from pathlib import Path
import sys
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
target = Path(os.environ.get("ML_HELPER_PATH", ROOT / "KSE_가치팩터_functions.py"))
spec = importlib.util.spec_from_file_location("accounting_under_test", target)
kf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kf)

class ExistingBehaviorRegression(unittest.TestCase):
    def rank(self, year=2021, rank=1):
        return pd.DataFrame({"Company": ["A"], "Year": [year], "Rank": [rank]})
    def close(self, dates, prices):
        return pd.DataFrame({"A": prices}, index=pd.to_datetime(dates))
    def test_flat_price_has_zero_profit_despite_unspent_cash(self):
        kf.start_year = 2021  # Original module relied on this undeclared global.
        close = self.close(["2021-01-04", "2021-03-31"], [30., 30.])
        total, positions = kf.process_year_data(2021, self.rank(), 100., 2021, 1, 2021, 3, close)
        self.assertAlmostEqual(total, 100.)
        self.assertAlmostEqual(positions["수익"].iloc[0], 0.)
    def test_purchase_never_moves_before_requested_month(self):
        close = self.close(["2020-12-31", "2021-01-04", "2021-03-31"], [20., 30., 33.])
        prices = kf.Price_for_Purchase_Sale(2021, 1, 2021, 3, self.rank(), close)
        self.assertEqual(prices["매수일"].iloc[0], pd.Timestamp("2021-01-04"))
    def test_february_sale_uses_last_observed_day_in_month(self):
        close = self.close(["2021-01-04", "2021-02-26", "2021-03-01"], [30., 33., 50.])
        prices = kf.Price_for_Purchase_Sale(2021, 1, 2021, 2, self.rank(), close)
        self.assertEqual(prices["매도일"].iloc[0], pd.Timestamp("2021-02-26"))
    def test_selected_company_is_not_dropped_by_its_original_rank(self):
        prices = pd.DataFrame({"기업명":["A"],"매수일":[pd.Timestamp("2021-01-04")],"매수가":[30.]})
        shares = kf.Purchase_share(prices, self.rank(rank=27), 100.)
        self.assertEqual(shares["기업명"].tolist(), ["A"])
        self.assertEqual(shares["구매주식수"].tolist(), [3])
    def test_empty_selection_preserves_cash(self):
        kf.start_year = 2021
        rank = self.rank().iloc[:0]
        total, positions = kf.process_year_data(2021, rank, 100., 2021, 1, 2021, 3,
                                                 self.close(["2021-01-04","2021-03-31"],[30.,30.]))
        self.assertEqual(total,100.)
        self.assertTrue(positions.empty)

if __name__ == "__main__":
    unittest.main()
