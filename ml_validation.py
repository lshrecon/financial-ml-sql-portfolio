"""Offline accounting and time-split checks; no model fitting or I/O."""
import numpy as np
import pandas as pd

POSITION_COLUMNS = ["매수가", "매도가", "구매주식수", "투자원금"]


def add_price_direction_targets(frame):
    """Price-series direction is independent of allocated cash and integer quantity.

    These inputs come from the source's adjusted-price series. Its adjustment
    convention is not verified; this is not an investable total-return claim.
    Keep the existing position-accounting fields and labels unchanged.
    """
    result = frame.copy()
    prices = result[['매수가', '매도가']].apply(pd.to_numeric, errors='raise')
    if prices.empty:
        # A year with no eligible companies has no targets, not a negative label.
        # Empty merged price columns can have object dtype; avoid isfinite on them.
        result['가격변화율'] = pd.Series(index=result.index, dtype=float)
        result['가격상승여부'] = pd.Series(index=result.index, dtype=int)
        return result
    if not np.isfinite(prices.to_numpy()).all() or (prices['매수가'] <= 0).any() or (prices['매도가'] < 0).any():
        raise ValueError('Price target requires finite entry > 0 and exit >= 0')
    result['가격변화율'] = (prices['매도가'] / prices['매수가'] - 1) * 100
    result['가격상승여부'] = (prices['매도가'] > prices['매수가']).astype(int)
    return result

def recalculate_returns(frame):
    """Gross P&L in currency; return in percent of allocated capital, including cash."""
    missing = set(POSITION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing position columns: {sorted(missing)}; regenerate labels first")
    result = frame.copy()
    if result.empty:
        for name in ("잔금", "매출", "수익", "수익률", "수익률_양음"):
            result[name] = pd.Series(index=result.index, dtype=float)
        return result
    values = result[POSITION_COLUMNS].apply(pd.to_numeric, errors="coerce")
    invalid = (~np.isfinite(values)).any(axis=1)
    invalid |= (values["매수가"] <= 0) | (values["매도가"] < 0)
    invalid |= (values["구매주식수"] < 0) | (values["투자원금"] <= 0)
    invalid |= values["구매주식수"] != np.floor(values["구매주식수"])
    cash = values["투자원금"] - values["매수가"] * values["구매주식수"]
    invalid |= cash < -1e-8 * values["투자원금"]
    if invalid.any():
        raise ValueError(f"Invalid accounting inputs in {int(invalid.sum())} rows; first row indices: {result.index[invalid].tolist()[:5]}")
    result[POSITION_COLUMNS] = values
    result["잔금"] = cash.clip(lower=0)
    result["매출"] = values["매도가"] * values["구매주식수"] + result["잔금"]
    result["수익"] = result["매출"] - values["투자원금"]
    result["수익률"] = result["수익"] / values["투자원금"] * 100.0
    result["수익률_양음"] = (result["수익"] > 0).astype(int)
    return result

def _years(frame):
    if "연도" not in frame:
        raise ValueError("Missing year column")
    year = pd.to_numeric(frame["연도"], errors="coerce")
    if not np.isfinite(year).all() or (year != np.floor(year)).any():
        raise ValueError("Years must be finite integers")
    if "기업명" in frame and frame.duplicated(["연도", "기업명"]).any():
        raise ValueError("Duplicate company/year observations")
    return year.astype(int)

def _available_before(earlier, later):
    """All earlier labels must be observable before the next period's first purchase."""
    if earlier.empty or later.empty:
        raise ValueError("Each temporal period needs observations")
    if "매도일" in earlier and "매수일" in later:
        end = pd.to_datetime(earlier["매도일"], errors="coerce")
        start = pd.to_datetime(later["매수일"], errors="coerce")
        if end.isna().any() or start.isna().any():
            raise ValueError("Missing or invalid holding dates")
        if end.max() >= start.min():
            raise ValueError("Holding-period labels overlap the next evaluation period")

def temporal_split(frame, train_end=2016, validation_end=2020):
    if train_end >= validation_end:
        raise ValueError("Training must end before validation")
    year = _years(frame)
    order = ["연도"] + (["기업명"] if "기업명" in frame else [])
    masks = [year <= train_end, (year > train_end) & (year <= validation_end), year > validation_end]
    parts = tuple(frame.loc[m].sort_values(order).copy().reset_index(drop=True) for m in masks)
    _available_before(parts[0], parts[1])
    _available_before(parts[1], parts[2])
    return parts

def expanding_year_folds(frame, n_splits=3):
    """Last n distinct years form single-year validation folds; indices are positional."""
    if not isinstance(n_splits, int) or n_splits < 1:
        raise ValueError("n_splits must be a positive integer")
    year = _years(frame)
    unique = sorted(year.unique())
    if len(unique) <= n_splits:
        raise ValueError("Need at least n_splits + 1 distinct years")
    result = []
    for validation_year in unique[-n_splits:]:
        train = np.flatnonzero((year < validation_year).to_numpy())
        valid = np.flatnonzero((year == validation_year).to_numpy())
        _available_before(frame.iloc[train], frame.iloc[valid])
        result.append((train, valid))
    return result
