"""Price-direction v2: target fix for zero-share rows; historical diagnostic only."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from ml_validation import add_price_direction_targets, expanding_year_folds, recalculate_returns, temporal_split


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_cohort(raw, protocol):
    """Quarantine known unavailable outcomes, fail on unexpected quality issues."""
    missing = raw['매도가'].isna()
    expected = {(int(y), name) for y, name in protocol['expected_missing']}
    actual = set(zip(raw.loc[missing, '연도'].astype(int), raw.loc[missing, '기업명']))
    if actual != expected or int(missing.sum()) != len(expected):
        raise ValueError('Missing exits differ from the reviewed source trace')
    rejected = raw.loc[missing, ['연도', '기업명', '매수일', '매도일']].copy()
    rejected['reason'] = 'Exit unavailable in original price source; outcome unknown'
    cohort = add_price_direction_targets(recalculate_returns(raw.loc[~missing]))
    features = cohort[protocol['features']].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError('Nonfinite feature: stop rather than silently impute/drop')
    cohort[protocol['features']] = features
    for column in ['매수일', '매도일']:
        cohort[column] = pd.to_datetime(cohort[column], errors='raise')
        if cohort[column].isna().any():
            raise ValueError('Missing holding date')
    if (cohort['매수일'] >= cohort['매도일']).any():
        raise ValueError('Nonpositive holding period')
    return cohort, rejected


def metrics(model, part, features):
    truth = part['가격상승여부'].to_numpy()
    prediction = model.predict(part[features])
    classes = list(model.classes_)
    probability = model.predict_proba(part[features])[:, classes.index(1)] if 1 in classes else np.zeros(len(part))
    return {
        'n': len(part), 'positive_rate': float(truth.mean()),
        'predicted_positive_rate': float(prediction.mean()),
        'accuracy': float(accuracy_score(truth, prediction)),
        'balanced_accuracy': float(balanced_accuracy_score(truth, prediction)),
        'precision': float(precision_score(truth, prediction, zero_division=0)),
        'recall': float(recall_score(truth, prediction, zero_division=0)),
        'f1': float(f1_score(truth, prediction, zero_division=0)),
        'roc_auc': float(roc_auc_score(truth, probability)) if len(np.unique(truth)) == 2 else None,
        'confusion_matrix_TN_FP_FN_TP': confusion_matrix(truth, prediction, labels=[0, 1]).ravel().tolist()
    }


def select_winner(validation, ordered_names):
    # max returns the first tied item; test outcomes are deliberately absent.
    return max(ordered_names, key=lambda name: validation[name]['f1'])


def run(source, protocol_path, output):
    protocol = json.loads(Path(protocol_path).read_text(encoding='utf-8'))
    if sha256(source) != protocol['source_sha256']:
        raise ValueError('Source changed since protocol was frozen')
    raw = pd.read_excel(source)
    cohort, rejected = prepare_cohort(raw, protocol)
    train, validation, historical_test = temporal_split(cohort, protocol['train_end'], protocol['validation_end'])
    features = protocol['features']
    folds = expanding_year_folds(train, protocol['cv_folds'])
    models = {
        'dummy_majority': DummyClassifier(strategy='most_frequent'),
        'dummy_all_positive': DummyClassifier(strategy='constant', constant=1),
    }
    for model in models.values():
        model.fit(train[features], train['가격상승여부'])
    cv_results = {}
    for name, estimator, grid in [
        ('decision_tree_cv', DecisionTreeClassifier(random_state=protocol['seed']), protocol['decision_tree_grid']),
        ('random_forest_cv', RandomForestClassifier(random_state=protocol['seed'], n_jobs=1), protocol['random_forest_grid'])
    ]:
        search = GridSearchCV(estimator, grid, scoring=make_scorer(f1_score, zero_division=0), cv=folds, error_score='raise', n_jobs=2, refit=True)
        search.fit(train[features], train['가격상승여부'])
        models[name] = search.best_estimator_
        cv_results[name] = {'best_parameters': search.best_params_, 'best_mean_cv_f1': float(search.best_score_), 'candidates': [
            {'parameters': p, 'mean_f1': float(score)} for p, score in zip(search.cv_results_['params'], search.cv_results_['mean_test_score'])]}
        print(f'Completed train-only CV: {name}', flush=True)
    validation_metrics = {name: metrics(models[name], validation, features) for name in protocol['candidates']}
    selected = select_winner(validation_metrics, protocol['candidates'])
    # Freeze winner before accessing historical test labels for scoring.
    selection = {'selected': selected, 'selection_source': '2017-2020 validation F1', 'validation': validation_metrics, 'protocol_sha256': sha256(protocol_path)}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'selection-before-test.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding='utf-8')
    evaluated_names = list(dict.fromkeys([selected, 'dummy_majority', 'dummy_all_positive']))
    report = {
        'protocol': protocol, 'protocol_sha256': sha256(protocol_path), 'source_sha256': sha256(source),
        'script_sha256': sha256(__file__),
        'validation_helper_sha256': sha256(Path(__file__).with_name('ml_validation.py')),
        'zero_share_rows': int(cohort['구매주식수'].eq(0).sum()),
        'price_target_differs_from_account_profit': int((cohort['가격상승여부'] != (cohort['수익'] > 0).astype(int)).sum()),
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scikit_learn': sklearn.__version__},
        'raw_rows': len(raw), 'diagnostic_rows': len(cohort),
        'quarantined': json.loads(rejected.to_json(orient='records', date_format='iso', force_ascii=False)),
        'price_target_flips_from_saved_return': int(((raw.loc[cohort.index, '수익률'] > 0).astype(int) != cohort['가격상승여부']).sum()),
        'splits': {name: {'n': len(part), 'year_min': int(part['연도'].min()), 'year_max': int(part['연도'].max()), 'first_purchase': str(part['매수일'].min().date()), 'last_exit': str(part['매도일'].max().date()), 'positive_rate': float(part['가격상승여부'].mean())} for name, part in [('train', train), ('validation', validation), ('historical_test', historical_test)]},
        'folds': [{'train_end_year': int(train.iloc[a]['연도'].max()), 'validation_year': int(train.iloc[b]['연도'].min()), 'train_n': len(a), 'validation_n': len(b)} for a, b in folds],
        'cross_validation': cv_results, **selection,
        'historical_test': {name: metrics(models[name], historical_test, features) for name in evaluated_names},
        'historical_test_by_year': {str(int(year)): metrics(models[selected], part, features) for year, part in historical_test.groupby('연도')},
        'limitations': [
            'Previously used historical test data: not new independent evidence',
            'Complete-case diagnostic excludes 3 unknown outcomes; may bias the dataset',
            'Saved features were not regenerated from point-in-time source financial statements',
            'No publication dates in source workbook; look-ahead and revision bias are unresolved',
            'Original historical investment universe and survivorship bias unresolved',
            'Adjusted-price direction only; adjustment convention unverified, not raw-price or total-return validation',
            'Integer share quantities from adjusted prices are not verified executable historical positions',
            'Classification only; no investable return, trading-cost or dividend validation',
            'This bounded diagnostic does not execute the entire original notebook/grid'
        ]
    }
    (output / 'evaluation-results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'selected': selected, 'validation': validation_metrics, 'historical_test': report['historical_test']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('--protocol', default='verification/price-direction-v2/protocol.json')
    parser.add_argument('--output', default='verification/price-direction-v2')
    args = parser.parse_args()
    run(args.source, args.protocol, args.output)
