import unittest
from unittest import mock

import numpy as np
import pandas as pd

import value_factor_backtest as functions


class IndicatorFunctions(unittest.TestCase):
    def test_extract_per_share_values_keeps_one_column_sorted_by_year(self):
        df_fin = pd.DataFrame(
            {
                'Name': ['B', 'A', 'C'],
                '회계년': [2021, 2020, 2022],
                'EPS': [3.0, 1.0, 5.0],
                'BPS': [9.0, 8.0, 7.0],
            }
        )
        values = functions.extract_per_share_values(df_fin, 'EPS')
        self.assertEqual(list(values.columns), ['Name', 'Year', 'EPS'])
        self.assertEqual(values['Year'].tolist(), [2020, 2021, 2022])
        self.assertEqual(values['Name'].tolist(), ['A', 'B', 'C'])
        self.assertEqual(values['EPS'].tolist(), [1.0, 3.0, 5.0])
        self.assertIn('회계년', df_fin.columns)

    def test_calculate_price_ratio_rank_orders_companies_each_year(self):
        inverse_ratios = pd.DataFrame(
            {'Year': [2001, 2002], 'A': [0.2, 0.3], 'B': [0.5, 0.1]}
        )
        ranked = functions.calculate_price_ratio_rank(
            inverse_ratios, ['A', 'B']
        )
        self.assertEqual(
            ranked[['Year', 'Company', 'Rank']].values.tolist(),
            [[2001, 'B', 1], [2001, 'A', 2], [2002, 'A', 1], [2002, 'B', 2]],
        )

    def test_merge_values_puts_each_company_year_value_in_its_row(self):
        df_profit = pd.DataFrame(
            {'기업명': ['A', 'B', 'A'], '연도': [2001, 2001, 2002]}
        )
        df_inverse = pd.DataFrame(
            {'Year': [2001, 2002], 'A': [0.2, 0.3], 'B': [0.5, 0.1]}
        )
        merged = functions.merge_values(df_profit, df_inverse, '1/PER')
        self.assertEqual(merged['1/PER'].tolist(), [0.2, 0.5, 0.3])
        self.assertNotIn('1/PER', df_profit.columns)

    def test_calculate_combined_profit_buys_in_april_sells_next_march(self):
        dates = pd.to_datetime(
            [
                '2001-04-02',
                '2001-04-03',
                '2002-03-28',
                '2002-03-29',
                '2002-04-01',
                '2003-03-31',
            ]
        )
        adjusted_closes = pd.DataFrame(
            {
                'A': [1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1260.0],
                'B': [500.0, np.nan, np.nan, 400.0, np.nan, np.nan],
            },
            index=dates,
        )
        ranked = pd.DataFrame(
            {'Company': ['A', 'B', 'A'], 'Year': [2001, 2001, 2002]}
        )
        profit = functions.calculate_combined_profit(
            ranked, 2001, 2002, 2001, 4, 2002, 3, adjusted_closes
        )
        self.assertEqual(profit['연도'].tolist(), [2001, 2001, 2002])
        self.assertEqual(profit['기업명'].tolist(), ['A', 'B', 'A'])
        self.assertEqual(profit['매수가'].tolist(), [1000.0, 500.0, 1400.0])
        self.assertEqual(profit['매도가'].tolist(), [1300.0, 400.0, 1260.0])
        self.assertEqual(profit['구매주식수'].tolist(), [5000, 10000, 3571])
        self.assertEqual(profit['투자원금'].tolist(), [5000000.0] * 3)
        self.assertEqual(profit['가격상승여부'].tolist(), [1, 0, 0])

    def test_calculate_portfolio_profits_sums_profit_of_both_sheets(self):
        sheets = {
            0: pd.DataFrame(
                {
                    '매수가': [1000.0, 500.0],
                    '매도가': [1100.0, 400.0],
                    '구매주식수': [10, 20],
                    '투자원금': [10000.0, 10000.0],
                }
            ),
            1: pd.DataFrame(
                {
                    '매수가': [100.0],
                    '매도가': [150.0],
                    '구매주식수': [50],
                    '투자원금': [5000.0],
                }
            ),
        }
        with mock.patch.object(
            functions.pd,
            'read_excel',
            side_effect=lambda path, sheet_name: sheets[sheet_name].copy(),
        ) as read_excel:
            profits = functions.calculate_portfolio_profits('backtest.xlsx')
        self.assertEqual(profits, (-1000.0, 2500.0))
        self.assertEqual(
            [call.kwargs['sheet_name'] for call in read_excel.call_args_list],
            [0, 1],
        )


if __name__ == '__main__':
    unittest.main()
