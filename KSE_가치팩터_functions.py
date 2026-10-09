import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ml_validation import recalculate_returns


def calculate_q1_endpoint_mean(df, year, company_columns):
    """Mean of January's first and March's last observed prices, not Q1 daily mean.

    Missing endpoints stay unknown. Do not move to another date to make a price.
    """
    if not isinstance(year, (int, np.integer)) or isinstance(year, bool):
        raise ValueError('Year must be an integer')
    if df.columns.duplicated().any() or len(company_columns) != len(set(company_columns)):
        raise ValueError('Duplicate company columns')
    if 'Symbol Name' not in df or not set(company_columns).issubset(df.columns):
        raise ValueError('Missing date or company columns')
    dates = pd.to_datetime(df['Symbol Name'], errors='raise')
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError('Price dates must be valid and unique')
    prices = df.loc[:, company_columns].apply(pd.to_numeric, errors='raise').copy()
    prices.index = pd.DatetimeIndex(dates)
    prices = prices.sort_index()
    january = prices.loc[(prices.index.year == year) & (prices.index.month == 1)]
    march = prices.loc[(prices.index.year == year) & (prices.index.month == 3)]
    if january.empty or march.empty:
        return {name: np.nan for name in company_columns}
    ends = pd.concat([january.iloc[[0]], march.iloc[[-1]]])
    if np.isinf(ends.to_numpy()).any() or (ends <= 0).any().any():
        raise ValueError('Observed endpoint prices must be finite and positive')
    # skipna=False: one observed endpoint is not a substitute for both endpoints.
    return ends.mean(axis=0, skipna=False).to_dict()


def calculate_q1_average_for_all_companies(df, year, company_columns):
    """Compatibility alias; this is the two-endpoint mean, not all daily prices."""
    return calculate_q1_endpoint_mean(df, year, company_columns)


def _feature_table(frame):
    if 'Year' not in frame or frame.columns.duplicated().any():
        raise ValueError('Features need one Year column and unique company columns')
    result = frame.copy()
    year = pd.to_numeric(result['Year'], errors='raise')
    if not np.isfinite(year).all() or (year != np.floor(year)).any() or year.duplicated().any():
        raise ValueError('Feature years must be finite, unique integers')
    result['Year'] = year.astype(int)
    result = result.set_index('Year').sort_index()
    result = result.apply(pd.to_numeric, errors='raise')
    if np.isinf(result.to_numpy()).any():
        raise ValueError('Infinite financial feature')
    return result


def top_companies_by_inverse_Rank(df, year, top_n):
    """Rank observed inverse-factor SCORES descending. Historical name is retained.

    Rank in this function's result is the score, until the notebook assigns
    ordinal ranks. Missing features cannot qualify for selection.
    """
    if not isinstance(top_n, (int, np.integer)) or isinstance(top_n, bool) or top_n < 0:
        raise ValueError('top_n must be a nonnegative integer')
    if not isinstance(year, (int, np.integer)) or isinstance(year, bool):
        raise ValueError('Year must be an integer')
    table = _feature_table(df)
    if year not in table.index:
        return pd.DataFrame(columns=['Company', 'Rank'])
    scores = table.loc[year].dropna()
    result = scores.rename_axis('Company').reset_index(name='Rank')
    # Stable tie handling makes source-column order irrelevant.
    return result.sort_values(['Rank', 'Company'], ascending=[False, True], kind='stable').head(top_n).reset_index(drop=True)


def calculate_weighted_average(dfs):
    """Equal-weight mean of matching year/company cells; no scale normalization.

    Any missing component remains missing. All tables must cover the same keys.
    The legacy name does not mean caller-configurable weights.
    """
    if not dfs:
        raise ValueError('At least one feature table is required')
    tables = [_feature_table(frame) for frame in dfs]
    first = tables[0]
    for table in tables[1:]:
        if not table.index.equals(first.index) or set(table.columns) != set(first.columns):
            raise ValueError('Feature tables must have identical year/company coverage')
    arrays = [table.reindex(columns=first.columns).to_numpy(dtype=float) for table in tables]
    values = np.mean(np.stack(arrays), axis=0)
    return pd.DataFrame(values, index=first.index, columns=first.columns).reset_index()


def _annual_observations(values):
    frame = (values[['Year', 'Total Value']].copy() if isinstance(values, pd.DataFrame)
             else pd.DataFrame(list(values.items()), columns=['Year', 'Total Value']))
    if frame.empty:
        raise ValueError('At least one portfolio observation is required')
    frame = frame.apply(pd.to_numeric, errors='raise')
    if (not np.isfinite(frame.to_numpy()).all()
            or (frame['Year'] != np.floor(frame['Year'])).any()
            or frame['Year'].duplicated().any() or (frame['Total Value'] < 0).any()):
        raise ValueError('Need unique integer years and finite nonnegative asset values')
    frame['Year'] = frame['Year'].astype(int)
    return frame.sort_values('Year').reset_index(drop=True)


def calculate_annual_compound_return(values, *, initial_value=None, periods=None):
    """CAGR between observations, or from EXPLICIT initial capital/time span.

    Without both keyword arguments, only first-to-last observed year is measured.
    Year-end values alone cannot reveal initial capital or the first holding year.
    """
    frame = _annual_observations(values)
    if (initial_value is None) != (periods is None):
        raise ValueError('Provide initial_value and periods together')
    if initial_value is None:
        initial_value = frame['Total Value'].iloc[0]
        periods = frame['Year'].iloc[-1] - frame['Year'].iloc[0]
    if not np.isfinite(initial_value) or initial_value <= 0 or not np.isfinite(periods) or periods <= 0:
        raise ValueError('Initial value and elapsed years must be finite and positive')
    return (frame['Total Value'].iloc[-1] / initial_value) ** (1 / periods) - 1


def calculate_mdd_with_year(values):
    """Drawdown of the provided annual observations, not daily maximum drawdown."""
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

def Price_for_Purchase_Sale(purchase_year, purchase_month, sale_year, sale_month, Rank_df, close_df):
    """Use first/last observed date WITHIN each requested month, never nearest outside it.

    This is a month-boundary rule, not a complete exchange calendar or execution model.
    Missing endpoint prices are rejected rather than silently shifted to another date.
    """
    columns = ['기업명', '매수일', '매수가', '매도일', '매도가']
    symbols = Rank_df['Company'].drop_duplicates().tolist()
    if not symbols:
        return pd.DataFrame(columns=columns)
    close_1 = close_df.copy()
    if not isinstance(close_1.index, pd.DatetimeIndex):
        close_1 = close_1.set_index(close_1.columns[0])
        close_1.index = pd.to_datetime(close_1.index, errors='coerce')
    if close_1.index.isna().any() or close_1.index.has_duplicates:
        raise ValueError('Price dates must be valid and unique')
    close_1 = close_1.sort_index()
    purchase_period = pd.Period(year=purchase_year, month=purchase_month, freq='M')
    sale_period = pd.Period(year=sale_year, month=sale_month, freq='M')
    periods = close_1.index.to_period('M')
    purchase_dates = close_1.index[periods == purchase_period]
    sale_dates = close_1.index[periods == sale_period]
    if purchase_dates.empty or sale_dates.empty:
        raise ValueError('No observed price dates in the requested purchase or sale month')
    purchase_date, sale_date = purchase_dates.min(), sale_dates.max()
    if sale_date < purchase_date:
        raise ValueError('Sale date cannot precede purchase date')
    if not set(symbols).issubset(close_1.columns):
        raise ValueError('Selected companies are missing from price data')
    purchase_prices = pd.to_numeric(close_1.loc[purchase_date, symbols], errors='coerce')
    sale_prices = pd.to_numeric(close_1.loc[sale_date, symbols], errors='coerce')
    if (not np.isfinite(purchase_prices).all() or not np.isfinite(sale_prices).all()
            or (purchase_prices <= 0).any() or (sale_prices < 0).any()):
        raise ValueError('Endpoint prices must be finite; purchase > 0 and sale >= 0')
    return pd.DataFrame({
        '기업명': symbols,
        '매수일': purchase_date,
        '매수가': purchase_prices.values,
        '매도일': sale_date,
        '매도가': sale_prices.values
    })

def Purchase_share(Purchase, Rank, investment):
    """Allocate equally across the already-selected companies, without re-ranking."""
    columns = ['기업명', '구매주식수', '투자원금']
    if not np.isfinite(investment) or investment < 0:
        raise ValueError('Investment must be finite and nonnegative')
    if Purchase.empty or investment == 0:
        return pd.DataFrame(columns=columns)
    purchase_years = pd.to_datetime(Purchase['매수일'], errors='coerce').dt.year
    if purchase_years.isna().any() or purchase_years.nunique() != 1:
        raise ValueError('Purchases must belong to one valid year')
    # Rank is the caller's already-selected cohort. Its label/feature year can
    # precede the purchase year; filtering it again would silently lose capital.
    selected = Rank['Company'].drop_duplicates()
    filtered = Purchase[Purchase['기업명'].isin(selected)].copy()
    if filtered['기업명'].duplicated().any():
        raise ValueError('Duplicate purchase company')
    if set(selected) != set(filtered['기업명']):
        raise ValueError('Missing prices for selected companies')
    if filtered.empty:
        return pd.DataFrame(columns=columns)
    prices = pd.to_numeric(filtered['매수가'], errors='coerce')
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError('Purchase prices must be finite and positive')
    allocation = investment / len(filtered)
    return pd.DataFrame({'기업명': filtered['기업명'].to_numpy(),
                         '구매주식수': np.floor(allocation / prices).astype(int).to_numpy(),
                         '투자원금': allocation})



def process_year_data(year, data, previous_total_value, purchase_year_imsi, purchase_month_imsi, sale_year_imsi, sale_month_imsi, close_1, base_year=None):
    if not np.isfinite(previous_total_value) or previous_total_value < 0:
        raise ValueError('Portfolio value must be finite and nonnegative')
    if data.empty or previous_total_value == 0:
        return previous_total_value, pd.DataFrame(columns=['기업명', '매수일', '매도가', '매수가', '매도일', '구매주식수', '투자원금', '잔금', '매출', '수익', '수익률', '수익률_양음'])
    # Explicit period anchor replaces the undeclared global start_year.
    anchor = purchase_year_imsi if base_year is None else base_year
    Purchase_current = Price_for_Purchase_Sale(purchase_year_imsi + (year - anchor), purchase_month_imsi,
                                               sale_year_imsi + (year - anchor), sale_month_imsi,
                                               data, close_1)

    share_current = Purchase_share(Purchase_current, data, previous_total_value)

    Purchase_current_first = Purchase_current.drop_duplicates(subset=['기업명', '매수일'], keep='first')
    Purchase_current_share = pd.merge(share_current, Purchase_current_first, on='기업명', how='left')

    Purchase_current_share = recalculate_returns(Purchase_current_share)

    total_value_current = Purchase_current_share['매출'].sum()

    return total_value_current, Purchase_current_share




def save_to_excel(data, filename):
    data.to_excel(filename, index=False)
    print(f"Data saved to {filename}")
def process_financial_indicator(year_range, data_set, initial_investment, purchase_params, sale_params, close_data):
    year_range = list(year_range)
    if year_range != sorted(set(year_range)):
        raise ValueError('Year range must be strictly increasing and unique')
    total_values = {}
    profits = {}
    previous_total_value = initial_investment
    for year in year_range:
        current_data = data_set[data_set['Year'] == year]
        total_value_current, profit_df = process_year_data(year, current_data, previous_total_value, *purchase_params, *sale_params, close_data, base_year=year_range[0])
        
        total_values[year] = total_value_current
        profits[year] = profit_df
        previous_total_value = total_value_current

    return total_values, profits





# 모든 데이터 프레임을 동일한 형식으로 만듭니다 (예: 동일한 기업명 순서)
# 이를 위해서는 각 데이터 프레임의 컬럼을 동일하게 정렬해야 합니다.
# 또한 모든 데이터 프레임에서 동일한 연도의 데이터만을 사용해야 합니다.

# 가중치를 적용하여 평균을 계산하는 함수


# 연복리 수익률 계산 함수
