"""Backtesting tools using adjusted closes and yearly features."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ml_validation import add_price_direction_targets, recalculate_returns


class AdjustedClosesByCompany:
    """Adjusted closes taken from a table and kept sorted by date."""

    DATE_COLUMNS = ('Symbol Name', 'Date')

    def __init__(self, adjusted_closes):
        date_indexed_closes = adjusted_closes.copy()
        if not isinstance(date_indexed_closes.index, pd.DatetimeIndex):
            date_column = next(
                (
                    name
                    for name in self.DATE_COLUMNS
                    if name in date_indexed_closes.columns
                ),
                None,
            )
            if date_column is None:
                raise ValueError(f'Missing date column: {self.DATE_COLUMNS}')
            date_indexed_closes = date_indexed_closes.set_index(date_column)
            date_indexed_closes.index = pd.to_datetime(
                date_indexed_closes.index, errors='coerce'
            )
        if (
            date_indexed_closes.index.isna().any()
            or date_indexed_closes.index.has_duplicates
        ):
            raise ValueError('Price dates must be valid and unique')
        self._date_sorted_closes = date_indexed_closes.sort_index()

    def search_first_date(self, year, month):
        return next(
            (
                date
                for date in self._date_sorted_closes.index
                if (date.year, date.month) == (year, month)
            ),
            None,
        )

    def search_last_date(self, year, month):
        return next(
            (
                date
                for date in reversed(self._date_sorted_closes.index)
                if (date.year, date.month) == (year, month)
            ),
            None,
        )


def calculate_q1_endpoint_mean(df, year, company_columns):
    """Mean of January's first and March's last observed prices.

    Not Q1 daily mean. Missing endpoints stay unknown. Do not move to
    another date to make a price.
    """
    if not isinstance(year, (int, np.integer)) or isinstance(year, bool):
        raise ValueError('Year must be an integer')
    if df.columns.duplicated().any() or len(company_columns) != len(
        set(company_columns)
    ):
        raise ValueError('Duplicate company columns')
    if not set(company_columns).issubset(df.columns):
        raise ValueError('Missing company columns')
    closes = AdjustedClosesByCompany(df)
    prices = closes._date_sorted_closes.loc[:, company_columns].apply(
        pd.to_numeric, errors='raise'
    )
    first_date_q1 = closes.search_first_date(year, 1)
    last_date_q1 = closes.search_last_date(year, 3)
    if first_date_q1 is None or last_date_q1 is None:
        return {name: np.nan for name in company_columns}
    ends = prices.loc[[first_date_q1, last_date_q1]]
    if np.isinf(ends.to_numpy()).any() or (ends <= 0).any().any():
        raise ValueError(
            'Observed endpoint prices must be finite and positive'
        )
    # skipna=False: one observed endpoint is not a substitute for
    # both endpoints.
    return ends.mean(axis=0, skipna=False).to_dict()


def calculate_q1_average_for_all_companies(df, year, company_columns):
    """Compatibility alias.

    This is the two-endpoint mean, not all daily prices.
    """
    return calculate_q1_endpoint_mean(df, year, company_columns)


def extract_per_share_values(df_fin, column):
    return (
        df_fin[['Name', '회계년', column]]
        .sort_values(by=['회계년'])
        .rename(columns={'회계년': 'Year'})
    )


def calculate_price_ratios(
    per_share_values, column, average_prices, company_columns
):
    values_by_year = per_share_values.pivot(
        index='Year', columns='Name', values=column
    )
    ratios = pd.DataFrame()
    for year in range(
        per_share_values['Year'].min() + 1, average_prices['Year'].max() + 1
    ):
        if (
            year in average_prices['Year'].values
            and (year - 1) in values_by_year.index
        ):
            previous_year_values = values_by_year.loc[year - 1]
            current_year_prices = average_prices[
                average_prices['Year'] == year
            ].iloc[0]
            row = {'Year': year}
            for company in company_columns:
                value = previous_year_values.get(company)
                price = current_year_prices.get(company)
                if value and price and value != 0:
                    ratio = price / value
                else:
                    ratio = None
                row[company] = ratio
            ratios = pd.concat(
                [ratios, pd.DataFrame([row])], ignore_index=True
            )
    ratios['Year'] = ratios['Year'].astype(int)
    return ratios


def calculate_inverse_price_ratios(price_ratios, company_columns):
    inverse_ratios = price_ratios.copy()
    for company in company_columns:
        inverse_ratios[company] = 1 / inverse_ratios[company]
    return inverse_ratios


def _feature_table(frame):
    if 'Year' not in frame or frame.columns.duplicated().any():
        raise ValueError(
            'Features need one Year column and unique company columns'
        )
    result = frame.copy()
    year = pd.to_numeric(result['Year'], errors='raise')
    if (
        not np.isfinite(year).all()
        or (year != np.floor(year)).any()
        or year.duplicated().any()
    ):
        raise ValueError('Feature years must be finite, unique integers')
    result['Year'] = year.astype(int)
    result = result.set_index('Year').sort_index()
    result = result.apply(pd.to_numeric, errors='raise')
    if np.isinf(result.to_numpy()).any():
        raise ValueError('Infinite financial feature')
    return result


def top_companies_by_inverse_rank(df, year, top_n):
    """Rank observed inverse-factor SCORES descending.

    Historical name is retained. Rank in this function's result is the
    score, until the notebook assigns ordinal ranks. Missing features
    cannot qualify for selection.
    """
    if (
        not isinstance(top_n, (int, np.integer))
        or isinstance(top_n, bool)
        or top_n < 0
    ):
        raise ValueError('top_n must be a nonnegative integer')
    if not isinstance(year, (int, np.integer)) or isinstance(year, bool):
        raise ValueError('Year must be an integer')
    table = _feature_table(df)
    if year not in table.index:
        return pd.DataFrame(columns=['Company', 'Rank'])
    scores = table.loc[year].dropna()
    result = scores.rename_axis('Company').reset_index(name='Rank')
    # Stable tie handling makes source-column order irrelevant.
    return (
        result.sort_values(
            ['Rank', 'Company'], ascending=[False, True], kind='stable'
        )
        .head(top_n)
        .reset_index(drop=True)
    )


top_companies_by_inverse_Rank = top_companies_by_inverse_rank


def calculate_price_ratio_rank(inverse_ratios, company_columns):
    ranked_companies_by_year = pd.DataFrame()
    for year in range(2001, 2024):
        top_companies = top_companies_by_inverse_rank(
            inverse_ratios, year, len(company_columns)
        )
        top_companies['Year'] = year
        top_companies['Rank'] = range(1, len(top_companies) + 1)
        ranked_companies_by_year = pd.concat(
            [ranked_companies_by_year, top_companies], ignore_index=True
        )
    return ranked_companies_by_year


def calculate_weighted_average(dfs):
    """Equal-weight mean of matching year/company cells.

    No scale normalization. Any missing component remains missing. All
    tables must cover the same keys. The legacy name does not mean
    caller-configurable weights.
    """
    if not dfs:
        raise ValueError('At least one feature table is required')
    tables = [_feature_table(frame) for frame in dfs]
    first = tables[0]
    for table in tables[1:]:
        if not table.index.equals(first.index) or set(table.columns) != set(
            first.columns
        ):
            raise ValueError(
                'Feature tables must have identical year/company coverage'
            )
    arrays = [
        table.reindex(columns=first.columns).to_numpy(dtype=float)
        for table in tables
    ]
    values = np.mean(np.stack(arrays), axis=0)
    return pd.DataFrame(
        values, index=first.index, columns=first.columns
    ).reset_index()


def _annual_observations(values):
    frame = (
        values[['Year', 'Total Value']].copy()
        if isinstance(values, pd.DataFrame)
        else pd.DataFrame(
            list(values.items()), columns=['Year', 'Total Value']
        )
    )
    if frame.empty:
        raise ValueError('At least one portfolio observation is required')
    frame = frame.apply(pd.to_numeric, errors='raise')
    if (
        not np.isfinite(frame.to_numpy()).all()
        or (frame['Year'] != np.floor(frame['Year'])).any()
        or frame['Year'].duplicated().any()
        or (frame['Total Value'] < 0).any()
    ):
        raise ValueError(
            'Need unique integer years and finite nonnegative asset values'
        )
    frame['Year'] = frame['Year'].astype(int)
    return frame.sort_values('Year').reset_index(drop=True)


def calculate_annual_compound_return(
    values, *, initial_value=None, periods=None
):
    """CAGR between observations, or from EXPLICIT initial capital/time span.

    Without both keyword arguments, only first-to-last observed year
    is measured. Year-end values alone cannot reveal initial capital or
    the first holding year.
    """
    frame = _annual_observations(values)
    if (initial_value is None) != (periods is None):
        raise ValueError('Provide initial_value and periods together')
    if initial_value is None:
        initial_value = frame['Total Value'].iloc[0]
        periods = frame['Year'].iloc[-1] - frame['Year'].iloc[0]
    if (
        not np.isfinite(initial_value)
        or initial_value <= 0
        or not np.isfinite(periods)
        or periods <= 0
    ):
        raise ValueError(
            'Initial value and elapsed years must be finite and positive'
        )
    return (frame['Total Value'].iloc[-1] / initial_value) ** (1 / periods) - 1


def calculate_mdd_with_year(values):
    """Drawdown of the provided annual observations.

    Not daily maximum drawdown.
    """
    frame = _annual_observations(values)
    if frame['Total Value'].iloc[0] <= 0:
        raise ValueError('First observed asset value must be positive')
    peaks = frame['Total Value'].cummax()
    drawdowns = (peaks - frame['Total Value']) / peaks
    position = int(drawdowns.idxmax())
    value = float(drawdowns.iloc[position])
    return value, int(frame['Year'].iloc[position]) if value > 0 else None


def plot_total_values(values, title):
    frame = _annual_observations(values)
    plt.figure(figsize=(10, 6))
    plt.plot(frame['Year'], frame['Total Value'], marker='o')
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Total Value')
    plt.grid(True)
    plt.show()


def _check_purchase_prices(prices):
    prices = pd.to_numeric(prices, errors='coerce')
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError('Purchase prices must be finite and positive')
    return prices


def price_for_purchase_sale(
    purchase_year,
    purchase_month,
    sale_year,
    sale_month,
    ranked_companies_for_year,
    adjusted_closes,
):
    """Use first/last observed date WITHIN each requested month.

    Never nearest outside it. This is a month-boundary rule, not a
    complete exchange calendar or execution model. Missing endpoint
    prices are rejected rather than silently shifted to another date.
    """
    columns = ['기업명', '매수일', '매수가', '매도일', '매도가']
    symbols = ranked_companies_for_year['Company'].drop_duplicates().tolist()
    if not symbols:
        return pd.DataFrame(columns=columns)
    closes = AdjustedClosesByCompany(adjusted_closes)
    date_sorted_closes = closes._date_sorted_closes
    purchase_date = closes.search_first_date(purchase_year, purchase_month)
    sale_date = closes.search_last_date(sale_year, sale_month)
    if purchase_date is None or sale_date is None:
        raise ValueError(
            'No observed price dates in the requested purchase or sale month'
        )
    if sale_date < purchase_date:
        raise ValueError('Sale date cannot precede purchase date')
    if not set(symbols).issubset(date_sorted_closes.columns):
        raise ValueError('Selected companies are missing from price data')
    purchase_prices = _check_purchase_prices(
        date_sorted_closes.loc[purchase_date, symbols]
    )
    sale_prices = pd.to_numeric(
        date_sorted_closes.loc[sale_date, symbols], errors='coerce'
    )
    if not np.isfinite(sale_prices).all() or (sale_prices < 0).any():
        raise ValueError('Sale prices must be finite and nonnegative')
    return pd.DataFrame(
        {
            '기업명': symbols,
            '매수일': purchase_date,
            '매수가': purchase_prices.values,
            '매도일': sale_date,
            '매도가': sale_prices.values,
        }
    )


Price_for_Purchase_Sale = price_for_purchase_sale


def purchase_share(purchase, ranked_companies_for_year, investment):
    """Allocate equally across the already-selected companies.

    Without re-ranking.
    """
    columns = ['기업명', '구매주식수', '투자원금']
    if not np.isfinite(investment) or investment < 0:
        raise ValueError('Investment must be finite and nonnegative')
    if purchase.empty or investment == 0:
        return pd.DataFrame(columns=columns)
    purchase_years = pd.to_datetime(
        purchase['매수일'], errors='coerce'
    ).dt.year
    if purchase_years.isna().any() or purchase_years.nunique() != 1:
        raise ValueError('Purchases must belong to one valid year')
    # ranked_companies_for_year is the caller's already-selected
    # cohort. Its label/feature year can precede the purchase year;
    # filtering it again would silently lose capital.
    selected = ranked_companies_for_year['Company'].drop_duplicates()
    filtered = purchase[purchase['기업명'].isin(selected)].copy()
    if filtered['기업명'].duplicated().any():
        raise ValueError('Duplicate purchase company')
    if set(selected) != set(filtered['기업명']):
        raise ValueError('Missing prices for selected companies')
    if filtered.empty:
        return pd.DataFrame(columns=columns)
    prices = _check_purchase_prices(filtered['매수가'])
    allocation = investment / len(filtered)
    return pd.DataFrame(
        {
            '기업명': filtered['기업명'].to_numpy(),
            '구매주식수': np.floor(allocation / prices).astype(int).to_numpy(),
            '투자원금': allocation,
        }
    )


Purchase_share = purchase_share


def process_year_data(
    year,
    ranked_companies_for_year,
    previous_total_value,
    base_purchase_year,
    base_purchase_month,
    base_sale_year,
    base_sale_month,
    adjusted_closes,
    base_year=None,
):
    if not np.isfinite(previous_total_value) or previous_total_value < 0:
        raise ValueError('Portfolio value must be finite and nonnegative')
    if ranked_companies_for_year.empty or previous_total_value == 0:
        return previous_total_value, pd.DataFrame(
            columns=[
                '기업명',
                '매수일',
                '매도가',
                '매수가',
                '매도일',
                '구매주식수',
                '투자원금',
                '잔금',
                '매출',
                '수익',
                '수익률',
                '수익률_양음',
            ]
        )
    anchor = base_purchase_year if base_year is None else base_year
    purchase_current = price_for_purchase_sale(
        base_purchase_year + (year - anchor),
        base_purchase_month,
        base_sale_year + (year - anchor),
        base_sale_month,
        ranked_companies_for_year,
        adjusted_closes,
    )

    share_current = purchase_share(
        purchase_current, ranked_companies_for_year, previous_total_value
    )

    purchase_current_first = purchase_current.drop_duplicates(
        subset=['기업명', '매수일'], keep='first'
    )
    purchase_current_share = pd.merge(
        share_current, purchase_current_first, on='기업명', how='left'
    )

    purchase_current_share = recalculate_returns(purchase_current_share)

    total_value_current = purchase_current_share['매출'].sum()

    return total_value_current, purchase_current_share


def calculate_combined_profit(
    ranked_companies_by_year,
    start_year,
    end_year,
    base_purchase_year,
    base_purchase_month,
    base_sale_year,
    base_sale_month,
    adjusted_closes,
):
    profits = []
    for year in range(start_year, end_year + 1):
        ranked_companies_for_year = ranked_companies_by_year[
            ranked_companies_by_year['Year'] == year
        ]
        _, profit = process_year_data(
            year,
            ranked_companies_for_year,
            5000000 * len(ranked_companies_for_year),
            base_purchase_year,
            base_purchase_month,
            base_sale_year,
            base_sale_month,
            adjusted_closes,
            base_year=start_year,
        )
        profit = add_price_direction_targets(profit)
        profit['연도'] = year
        profits.append(profit)
    return pd.concat(profits)


def merge_values(df_profit, df_inverse, column):
    df_merged = df_profit.copy()
    for index, row in df_profit.iterrows():
        year = row['연도']
        company = row['기업명']
        value = df_inverse.loc[df_inverse['Year'] == year, company].iloc[0]
        df_merged.at[index, column] = value
    return df_merged


def save_to_excel(data, filename):
    data.to_excel(filename, index=False)
    print(f'Data saved to {filename}')


def process_financial_indicator(
    year_range,
    data_set,
    initial_investment,
    purchase_params,
    sale_params,
    adjusted_closes,
):
    year_range = list(year_range)
    if year_range != sorted(set(year_range)):
        raise ValueError('Year range must be strictly increasing and unique')
    total_values = {}
    profits = {}
    previous_total_value = initial_investment
    for year in year_range:
        ranked_companies_for_year = data_set[data_set['Year'] == year]
        total_value_current, profit_df = process_year_data(
            year,
            ranked_companies_for_year,
            previous_total_value,
            *purchase_params,
            *sale_params,
            adjusted_closes,
            base_year=year_range[0],
        )

        total_values[year] = total_value_current
        profits[year] = profit_df
        previous_total_value = total_value_current

    return total_values, profits
