"""Regressions found when reviewing the 2026-10-08 feature changes."""
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

import KSE_가치팩터_functions as functions
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

    def test_pbr_cell_preserves_own_years_when_per_range_differs(self):
        notebook = Path(__file__).resolve().parents[1] / '머신러닝_전처리.ipynb'
        cells = json.loads(notebook.read_text(encoding='utf-8'))['cells']
        sources = [''.join(cell.get('source', [])) for cell in cells
                   if cell.get('cell_type') == 'code']
        pbr_cells = [source for source in sources
                     if 'BPS_Pure_df_pivot =' in source and 'df_pbr = pd.DataFrame()' in source]
        self.assertEqual(len(pbr_cells), 1)
        per = pd.DataFrame({'Year': [2020, 2021, 2022], 'A': [1., 2., 3.]})
        namespace = {
            'pd': pd,
            'company_columns': ['A'],
            'BPS_Pure_df': pd.DataFrame({'Name': ['A', 'A'], 'Year': [2020, 2021],
                                        'BPS': [20., 30.]}),
            'average_prices_df': pd.DataFrame({'Year': [2021, 2022], 'A': [100., 240.]}),
            'df_per': per.copy(deep=True),
        }
        # Execute the real calculation cell with synthetic inputs; it performs no I/O.
        exec(compile(pbr_cells[0], str(notebook) + ':PBR', 'exec'), namespace)
        expected = pd.DataFrame({'Year': [2021, 2022], 'A': [5., 8.]})
        pd.testing.assert_frame_equal(namespace['df_pbr'], expected, check_dtype=False)
        pd.testing.assert_frame_equal(namespace['df_per'], per)


if __name__ == '__main__':
    unittest.main()
