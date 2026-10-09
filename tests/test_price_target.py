import unittest
import numpy as np
import pandas as pd
from ml_validation import add_price_direction_targets, recalculate_returns


class PriceTargetTests(unittest.TestCase):
    def test_zero_shares_does_not_mean_price_did_not_rise(self):
        source = pd.DataFrame({'매수가':[600], '매도가':[660], '구매주식수':[0], '투자원금':[500]})
        result = add_price_direction_targets(recalculate_returns(source))
        self.assertEqual(result.loc[0, '수익'], 0)
        self.assertEqual(result.loc[0, '수익률_양음'], 0)
        self.assertEqual(result.loc[0, '가격상승여부'], 1)
        self.assertAlmostEqual(result.loc[0, '가격변화율'], 10)
        self.assertEqual(result.loc[0, '잔금'], 500)

    def test_scale_invariance_and_flat_and_zero_exit(self):
        source = pd.DataFrame({'매수가':[100, 10000, 100, 100], '매도가':[110, 11000, 100, 0]})
        result = add_price_direction_targets(source)
        self.assertEqual(result['가격상승여부'].tolist(), [1,1,0,0])
        self.assertAlmostEqual(result.loc[0, '가격변화율'], result.loc[1, '가격변화율'])
        self.assertEqual(result.loc[3, '가격변화율'], -100)
        self.assertNotIn('가격상승여부', source)

    def test_missing_or_invalid_prices_do_not_become_zero_class(self):
        for buy, sell in [(100,np.nan),(100,np.inf),(0,100),(100,-1)]:
            with self.subTest(buy=buy,sell=sell), self.assertRaises(ValueError):
                add_price_direction_targets(pd.DataFrame({'매수가':[buy], '매도가':[sell]}))
