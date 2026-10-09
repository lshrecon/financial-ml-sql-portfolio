"""Validate stock-price rows before exporting the MySQL LOAD DATA input.

This module never connects to a database. CSV output uses UTF-8 without BOM,
LF line endings, and doubled quote escaping, matching the accompanying SQL.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
import io
import re
from pathlib import Path

import pandas as pd


COLUMNS = ["entity", "attribute", "value"]
MAX_PRICE = Decimal("999999999999999999.99")  # DECIMAL(20, 2)
MISSING_PRICE_TEXT = {"", "-", "--", "na", "n/a", "null", "nan"}
NUMBER = re.compile(r"[+]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?")


class DataValidationError(ValueError):
    """Input cannot be represented faithfully by the target price table."""


@dataclass(frozen=True)
class PreparationReport:
    source_price_cells: int
    missing_prices_removed: int
    output_rows: int


def _date(value: object) -> str:
    if pd.isna(value):
        raise DataValidationError("Date is missing")
    if isinstance(value, datetime):
        if value.tzinfo is not None or value.time() != time() or getattr(value, "nanosecond", 0) != 0:
            raise DataValidationError("Use a date or timezone-free midnight, not an intraday timestamp")
        parsed = value.date()
    elif isinstance(value, date):
        parsed = value
    elif isinstance(value, str) and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value.strip()):
        try:
            parsed = date.fromisoformat(value.strip())
        except ValueError as exc:
            raise DataValidationError("Invalid calendar date") from exc
    else:
        raise DataValidationError("Date must be YYYY-MM-DD or an Excel date value")
    if parsed.year < 1000:
        raise DataValidationError("Date is outside the MySQL DATE range")
    return parsed.isoformat()


def _company(value: object) -> str:
    if not isinstance(value, str):
        raise DataValidationError("Company name must be text")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise DataValidationError("Company name contains a control character")
    cleaned = value.strip()
    if not cleaned or len(cleaned) > 100:
        raise DataValidationError("Company name must contain 1 to 100 characters")
    try:
        cleaned.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise DataValidationError("Company name cannot be encoded as valid UTF-8") from exc
    return cleaned


def _missing_price(value: object) -> bool:
    return bool(pd.isna(value)) or (
        isinstance(value, str) and value.strip().lower() in MISSING_PRICE_TEXT
    )


def _price(value: object) -> str:
    if isinstance(value, bool) or _missing_price(value):
        raise DataValidationError("Price is missing or boolean")
    text = str(value).strip()
    if not NUMBER.fullmatch(text):
        raise DataValidationError("Price must be a finite positive number with valid comma grouping")
    try:
        number = Decimal(text.replace(",", ""))
        if not number.is_finite() or not (Decimal("0") < number <= MAX_PRICE):
            raise DataValidationError("Price must be positive and fit DECIMAL(20, 2)")
        cents = number.quantize(Decimal("0.01"))
    except InvalidOperation as exc:
        raise DataValidationError("Price cannot be represented as DECIMAL(20, 2)") from exc
    if number != cents:
        raise DataValidationError("Price has more than two decimal places; an explicit rounding policy is required")
    return format(cents, ".2f")


def validate_stock_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Return canonical rows, rejecting invalid values and duplicate date/company keys."""
    if list(rows.columns) != COLUMNS or rows.empty:
        raise DataValidationError("Expected a nonempty table with entity, attribute, value columns")
    records = []
    seen = set()
    for position, values in enumerate(rows.itertuples(index=False, name=None), start=1):
        try:
            day, company, price = _date(values[0]), _company(values[1]), _price(values[2])
        except DataValidationError as exc:
            raise DataValidationError(f"Row {position}: {exc}") from exc
        key = (day, company)
        if key in seen:
            raise DataValidationError(f"Row {position}: duplicate date/company key; resolve it explicitly")
        seen.add(key)
        records.append((day, company, price))
    return pd.DataFrame(records, columns=COLUMNS)


def prepare_stock_prices(wide: pd.DataFrame, date_column: str = "Date") -> tuple[pd.DataFrame, PreparationReport]:
    """Reshape a date-by-company table; remove only explicitly missing prices.

    Invalid dates/prices and all duplicate keys fail instead of being silently
    dropped, rounded, or converted. Missing-price removals are counted.
    """
    if not wide.columns.is_unique or date_column not in wide.columns or len(wide.columns) < 2 or wide.empty:
        raise DataValidationError("Expected unique columns, a Date column, and at least one company")
    companies = [column for column in wide.columns if column != date_column]
    cleaned_names = [_company(column) for column in companies]
    if len(set(cleaned_names)) != len(cleaned_names):
        raise DataValidationError("Company columns collide after whitespace normalization")
    # Keep the date under a private, collision-free name while reshaping. A
    # company called "Date", or a custom date column called "attribute", must
    # not replace the date or collide with the three output column names.
    normalized = wide[companies].copy()
    normalized.columns = cleaned_names
    temporary_date = "__stock_date__"
    while temporary_date in normalized.columns:
        temporary_date += "_"
    normalized.insert(0, temporary_date, wide[date_column].map(_date))
    temporary_value = "__stock_price_value__"
    while temporary_value in normalized.columns:
        temporary_value += "_"
    long = normalized.melt(id_vars=[temporary_date], var_name="attribute", value_name=temporary_value)
    long = long.rename(columns={temporary_date: "entity", temporary_value: "value"})[COLUMNS]
    missing = long["value"].map(_missing_price)
    clean = validate_stock_rows(long.loc[~missing])
    return clean, PreparationReport(len(long), int(missing.sum()), len(clean))


def _render_csv(rows: pd.DataFrame) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, quoting=csv.QUOTE_ALL, doublequote=True, lineterminator="\n")
    writer.writerow(COLUMNS)
    writer.writerows(rows.itertuples(index=False, name=None))
    return buffer.getvalue()


def write_stock_csv(rows: pd.DataFrame, path: str | Path, *, overwrite: bool = False) -> None:
    clean = validate_stock_rows(rows)
    # Render and encode before opening the destination: encoding failures must
    # not truncate an existing file. This is not an atomic filesystem write.
    payload = _render_csv(clean).encode("utf-8")
    with Path(path).open("wb" if overwrite else "xb") as handle:
        handle.write(payload)


def validate_stock_csv(path: str | Path) -> pd.DataFrame:
    """Require the canonical CSV format used by the MySQL loader, not just parsable CSV."""
    try:
        text = Path(path).read_bytes().decode("utf-8")
        parsed = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise DataValidationError("CSV must be valid UTF-8 and correctly quoted") from exc
    if not parsed or parsed[0] != COLUMNS or any(len(row) != 3 for row in parsed[1:]):
        raise DataValidationError("CSV header or column count does not match the target table")
    clean = validate_stock_rows(pd.DataFrame(parsed[1:], columns=COLUMNS))
    if text != _render_csv(clean):
        raise DataValidationError("CSV is not canonical: re-export with write_stock_csv (UTF-8, LF, quoted fields)")
    return clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a stock CSV without connecting to MySQL")
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()
    checked = validate_stock_csv(args.csv_path)
    print(f"Validated {len(checked)} unique stock-price rows; no database operation performed.")
