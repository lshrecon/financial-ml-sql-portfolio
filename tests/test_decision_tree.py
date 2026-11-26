import contextlib
import io
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import decision_tree


def made_up_profit_table():
    rows = []
    companies = [('A', 0.9), ('B', 0.7), ('C', 0.3), ('D', 0.1)]
    for year in range(2010, 2023):
        for rank, (company, inverse_per) in enumerate(companies, start=1):
            rows.append(
                {
                    '연도': year,
                    '기업명': company,
                    '1/PER': inverse_per,
                    '1/PER Rank': rank,
                    '매수가': 100.0,
                    '매도가': 120.0 if inverse_per > 0.5 else 80.0,
                    '구매주식수': 10,
                    '투자원금': 1000.0,
                }
            )
    return pd.DataFrame(rows)


class PriceDirectionTrees(unittest.TestCase):
    def test_tree_picks_the_companies_it_predicts_to_rise(self):
        with mock.patch.object(
            decision_tree.pd, 'read_excel', return_value=made_up_profit_table()
        ):
            tree = decision_tree.PriceDirectionTree(
                'profit.xlsx', ['1/PER', '1/PER Rank']
            )
        tree.train(max_depth=1)
        top_2021 = tree.top_rising_companies('1/PER', 2021)
        self.assertEqual(top_2021['기업명'].tolist(), ['A', 'B'])
        self.assertEqual(top_2021['예측_가격상승여부'].tolist(), [1, 1])
        every_year = tree.top_rising_companies('1/PER')
        self.assertEqual(every_year['연도'].tolist(), [2021, 2021, 2022, 2022])
        self.assertEqual(every_year['기업명'].tolist(), ['A', 'B', 'A', 'B'])
        self.assertNotIn('예측_가격상승여부', tree.test_data.columns)

    def test_compare_trees_prints_accuracy_and_confusion_matrices(self):
        validation_data = pd.DataFrame({'가격상승여부': [1, 0, 1, 0]})
        original = types.SimpleNamespace(
            validation_data=validation_data,
            predict=lambda data: np.array([1, 0, 0, 0]),
        )
        simplified = types.SimpleNamespace(
            validation_data=validation_data,
            predict=lambda data: np.array([1, 1, 1, 1]),
        )
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            decision_tree.compare_trees(original, simplified)
        self.assertEqual(
            printed.getvalue().splitlines(),
            [
                '원본 모델 정확도: 0.75',
                '단순화된 모델 정확도: 0.5',
                '원본 모델의 혼동 행렬:',
                '[[2 0]',
                ' [1 1]]',
                '',
                '단순화된 모델의 혼동 행렬:',
                '[[0 2]',
                ' [0 2]]',
            ],
        )


if __name__ == '__main__':
    unittest.main()
