"""Build and compare decision trees for price direction."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree

from ml_validation import (
    add_price_direction_targets,
    recalculate_returns,
    temporal_split,
)


class PriceDirectionTree:
    def __init__(self, excel_file_path, features):
        excel_data = add_price_direction_targets(
            recalculate_returns(pd.read_excel(excel_file_path))
        )
        self.features = features
        self.train_data, self.validation_data, self.test_data = temporal_split(
            excel_data
        )
        self._model = None

    def train(self, max_depth=None):
        self._model = DecisionTreeClassifier(
            max_depth=max_depth, random_state=42
        )
        self._model.fit(
            self.train_data[self.features], self.train_data['가격상승여부']
        )

    def predict(self, data):
        return self._model.predict(data[self.features])

    def plot(self):
        plt.figure(figsize=(20, 10))
        plot_tree(
            self._model,
            filled=True,
            feature_names=self.features,
            class_names=['Not rising', 'Rising'],
        )
        plt.show()

    def top_rising_companies(self, column, year=None):
        test_data = self.test_data.copy()
        test_data['예측_가격상승여부'] = self.predict(test_data)
        rising = test_data[test_data['예측_가격상승여부'] == 1]
        if year is None:
            return (
                rising.groupby('연도')[list(rising.columns)]
                .apply(lambda x: x.nlargest(20, column))
                .reset_index(drop=True)
            )
        return (
            rising[rising['연도'] == year]
            .sort_values(column, ascending=False)
            .head(20)
        )


def compare_trees(original_tree, simplified_tree):
    y_validation = original_tree.validation_data['가격상승여부']
    original_predictions_validation = original_tree.predict(
        original_tree.validation_data
    )
    simplified_predictions_validation = simplified_tree.predict(
        original_tree.validation_data
    )

    original_accuracy = accuracy_score(
        y_validation, original_predictions_validation
    )
    simplified_accuracy = accuracy_score(
        y_validation, simplified_predictions_validation
    )

    print(f'원본 모델 정확도: {original_accuracy}')
    print(f'단순화된 모델 정확도: {simplified_accuracy}')

    original_conf_matrix = confusion_matrix(
        y_validation, original_predictions_validation
    )
    simplified_conf_matrix = confusion_matrix(
        y_validation, simplified_predictions_validation
    )

    print('원본 모델의 혼동 행렬:')
    print(original_conf_matrix)

    print('\n단순화된 모델의 혼동 행렬:')
    print(simplified_conf_matrix)
