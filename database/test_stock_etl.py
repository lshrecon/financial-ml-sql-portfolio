"""Offline regressions only: these tests do not run or emulate MySQL."""

import tempfile
from pathlib import Path
import unittest

import pandas as pd

from stock_etl import (
    COLUMNS, DataValidationError, prepare_stock_prices,
    validate_stock_csv, validate_stock_rows, write_stock_csv,
)


def legacy_rows(wide):
    """The original notebook cell 1's melt / mapping / price-dropna, without file I/O."""
    long = pd.melt(wide, id_vars=["Date"], var_name="Company", value_name="Closing_Price")
    return pd.DataFrame({
        "entity": long["Date"], "attribute": long["Company"], "value": long["Closing_Price"],
    }).dropna(subset=["value"])


class StockEtlTests(unittest.TestCase):
    def test_legacy_keeps_invalid_date_but_repair_rejects_it(self):
        wide = pd.DataFrame({"Date": ["2023-02-30"], "A": [100]})
        self.assertEqual(legacy_rows(wide).iloc[0]["entity"], "2023-02-30")
        with self.assertRaisesRegex(DataValidationError, "calendar date"):
            prepare_stock_prices(wide)

    def test_legacy_keeps_missing_date_but_repair_rejects_nat(self):
        bad_date = pd.to_datetime("invalid-date", errors="coerce")
        wide = pd.DataFrame({"Date": [bad_date], "A": [100]})
        self.assertTrue(pd.isna(legacy_rows(wide).iloc[0]["entity"]))
        with self.assertRaisesRegex(DataValidationError, "Date is missing"):
            prepare_stock_prices(wide)

    def test_legacy_keeps_duplicate_key_but_repair_rejects_same_and_conflicting_prices(self):
        for prices in ([100, 100], [100, 110]):
            with self.subTest(prices=prices):
                wide = pd.DataFrame({"Date": ["2023-01-02", "2023-01-02"], "A": prices})
                self.assertEqual(len(legacy_rows(wide)), 2)
                with self.assertRaisesRegex(DataValidationError, "duplicate date/company"):
                    prepare_stock_prices(wide)

    def test_valid_reshape_and_missing_price_count(self):
        wide = pd.DataFrame({"Date": [pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-03")],
                             " A ": ["1,200.50", None], "B": [200, 210]})
        rows, report = prepare_stock_prices(wide)
        self.assertEqual(rows.to_dict("records"), [
            {"entity": "2023-01-02", "attribute": "A", "value": "1200.50"},
            {"entity": "2023-01-02", "attribute": "B", "value": "200.00"},
            {"entity": "2023-01-03", "attribute": "B", "value": "210.00"},
        ])
        self.assertEqual((report.source_price_cells, report.missing_prices_removed, report.output_rows), (4, 1, 3))

    def test_invalid_prices_do_not_become_valid_numbers(self):
        for value in [0, -1, True, float("inf"), "abc100", "12,34", "1.001", "1000000000000000000"]:
            with self.subTest(value=value):
                wide = pd.DataFrame({"Date": ["2023-01-02"], "A": [value]})
                self.assertEqual(len(legacy_rows(wide)), 1)
                with self.assertRaises(DataValidationError):
                    prepare_stock_prices(wide)

    def test_no_ambiguous_intraday_or_zero_dates(self):
        for value in ["01/02/2023", "0000-00-00", "2023-13-01", "2023-01-01T12:00:00",
                      pd.Timestamp("2023-01-01 00:01:00"), pd.Timestamp("2023-01-01", tz="UTC")]:
            with self.subTest(value=value):
                with self.assertRaises(DataValidationError):
                    prepare_stock_prices(pd.DataFrame({"Date": [value], "A": [100]}))

    def test_valid_leap_day_and_decimal_boundary(self):
        rows, _ = prepare_stock_prices(pd.DataFrame({"Date": ["2024-02-29"], "A": ["999999999999999999.99"]}))
        self.assertEqual(rows.iloc[0]["value"], "999999999999999999.99")

    def test_company_validation_and_normalization_collision(self):
        for name in ["", " ", "A\nB", "\tA", "A" * 101]:
            with self.subTest(name=name):
                with self.assertRaises(DataValidationError):
                    prepare_stock_prices(pd.DataFrame({"Date": ["2023-01-02"], name: [100]}))
        with self.assertRaisesRegex(DataValidationError, "collide"):
            prepare_stock_prices(pd.DataFrame({"Date": ["2023-01-02"], "A": [100], " A ": [100]}))

    def test_company_named_value_is_not_a_pandas_melt_collision(self):
        rows, _ = prepare_stock_prices(pd.DataFrame({"Date": ["2023-01-02"], "value": [100]}))
        self.assertEqual(rows.iloc[0]["attribute"], "value")

    def test_company_normalization_does_not_overwrite_the_date_column(self):
        rows, report = prepare_stock_prices(pd.DataFrame({"Date": ["2023-01-02"], " Date ": [100]}))
        self.assertEqual(rows.to_dict("records"), [
            {"entity": "2023-01-02", "attribute": "Date", "value": "100.00"}])
        self.assertEqual(report.output_rows, 1)

    def test_custom_date_column_does_not_collide_with_output_columns(self):
        for date_column in ["attribute", "entity", "value", "__stock_date__"]:
            with self.subTest(date_column=date_column):
                rows, _ = prepare_stock_prices(
                    pd.DataFrame({date_column: ["2023-01-02"], "A": [100]}), date_column=date_column)
                self.assertEqual(rows.to_dict("records"), [
                    {"entity": "2023-01-02", "attribute": "A", "value": "100.00"}])

    def test_nanosecond_timestamp_is_not_silently_truncated_to_a_date(self):
        with self.assertRaisesRegex(DataValidationError, "intraday"):
            prepare_stock_prices(pd.DataFrame({
                "Date": [pd.Timestamp("2023-01-02 00:00:00.000000001")], "A": [100]}))

    def test_invalid_unicode_cannot_erase_an_existing_csv(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix=".test-stock-") as directory:
            path = Path(directory) / "prices.csv"
            good = pd.DataFrame([["2023-01-02", "A", "100.00"]], columns=COLUMNS)
            write_stock_csv(good, path)
            original = path.read_bytes()
            bad = pd.DataFrame([["2023-01-02", "A" + chr(0xD800), "100.00"]], columns=COLUMNS)
            with self.assertRaises(DataValidationError):
                write_stock_csv(bad, path, overwrite=True)
            self.assertEqual(path.read_bytes(), original)

    def test_csv_round_trip_preserves_unicode_comma_quote_and_backslash(self):
        company = '회사, "특별"\\A'
        rows = pd.DataFrame([["2023-01-02", company, "100.00"]], columns=COLUMNS)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix=".test-stock-") as directory:
            self.assertEqual(Path(directory).resolve().parent, Path(__file__).resolve().parent)
            path = Path(directory) / "prices.csv"
            write_stock_csv(rows, path)
            raw = path.read_bytes()
            self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
            self.assertNotIn(b"\r", raw)
            pd.testing.assert_frame_equal(rows, validate_stock_csv(path))

    def test_csv_tampering_is_rejected(self):
        rows = pd.DataFrame([["2023-01-02", "A", "100.00"]], columns=COLUMNS)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix=".test-stock-") as directory:
            self.assertEqual(Path(directory).resolve().parent, Path(__file__).resolve().parent)
            path = Path(directory) / "prices.csv"
            write_stock_csv(rows, path)
            valid = path.read_text(encoding="utf-8")
            variants = [valid.replace("100.00", "-1.00"), valid.replace("2023-01-02", "2023-02-30"),
                        valid + valid.splitlines(keepends=True)[1], valid.replace("\n", "\r\n"),
                        "\ufeff" + valid, valid.replace('"value"', '"price"'), valid.replace('"100.00"', '"100.00","extra"')]
            for bad in variants:
                with self.subTest(bad=bad):
                    path.write_bytes(bad.encode("utf-8"))
                    with self.assertRaises(DataValidationError):
                        validate_stock_csv(path)

    def test_validation_failure_does_not_overwrite_existing_file(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix=".test-stock-") as directory:
            self.assertEqual(Path(directory).resolve().parent, Path(__file__).resolve().parent)
            path = Path(directory) / "prices.csv"
            path.write_bytes(b"keep this file")
            bad = pd.DataFrame([["2023-01-02", "A", -1]], columns=COLUMNS)
            with self.assertRaises(DataValidationError):
                write_stock_csv(bad, path, overwrite=True)
            self.assertEqual(path.read_bytes(), b"keep this file")
            good = pd.DataFrame([["2023-01-02", "A", 100]], columns=COLUMNS)
            with self.assertRaises(FileExistsError):
                write_stock_csv(good, path)

    def test_empty_input_or_all_missing_prices_is_not_exported(self):
        for wide in [pd.DataFrame({"Date": [], "A": []}),
                     pd.DataFrame({"Date": ["2023-01-02"], "A": [None]})]:
            with self.subTest(wide=wide):
                with self.assertRaises(DataValidationError):
                    prepare_stock_prices(wide)


if __name__ == "__main__":
    unittest.main()
