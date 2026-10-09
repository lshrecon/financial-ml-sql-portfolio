"""Synthetic MySQL checks. Called only by run_isolated_mysql.py's owned server."""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import subprocess
import unittest

import pandas as pd
from stock_etl import DataValidationError, validate_stock_rows, write_stock_csv


class MySQLChecks(unittest.TestCase):
    client: Path
    memory_name: str
    uploads: Path
    source_dir: Path
    child_env: dict
    trace: list

    def query(self, sql: str, expected_error: int | None = None) -> str:
        command = [str(self.client), '--no-defaults', '--no-login-paths',
                   '--protocol=MEMORY', '--shared-memory-base-name=' + self.memory_name,
                   '--user=root', '--default-character-set=utf8mb4',
                   '--batch', '--raw', '--skip-column-names', '--connect-timeout=3']
        result = subprocess.run(command, input=sql, encoding='utf-8', errors='replace',
                                capture_output=True, env=self.child_env, timeout=30)
        self.trace.append({'sql': sql, 'exit_code': result.returncode,
                           'stdout': result.stdout, 'stderr': result.stderr})
        if expected_error is None:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, 'Expected MySQL error but SQL succeeded')
            self.assertIn('ERROR ' + str(expected_error) + ' ', result.stderr)
        return result.stdout.strip()

    def rows(self, database: str) -> str:
        return self.query(f'USE {database}; SELECT id, entity, HEX(attribute), value '
                          'FROM stock_data_eav ORDER BY id;')

    def csv_file(self, name: str, rows: list, validate: bool = True) -> Path:
        path = self.uploads / (name + '.csv')
        if validate:
            write_stock_csv(pd.DataFrame(rows, columns=['entity', 'attribute', 'value']), path)
        else:
            with path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream, quoting=csv.QUOTE_ALL, lineterminator='\n')
                writer.writerow(['entity', 'attribute', 'value'])
                writer.writerows(rows)
        return path

    def new_schema(self, database: str, path: Path) -> str:
        sql = (self.source_dir / '주식데이터베이스sql.sql').read_text(encoding='utf-8')
        # Names below are fixed synthetic test identifiers, never user input.
        sql = sql.replace('stock_data CHARACTER', database + ' CHARACTER')
        sql = sql.replace('USE stock_data;', f'USE {database};')
        return sql.replace('/path/to/mysql-uploads/stock_data_eav.csv', path.as_posix())

    def load(self, database: str, path: Path) -> str:
        text = self.new_schema(database, path)
        return 'USE ' + database + ';\n' + text[:text.index('CREATE DATABASE')] + \
               text[text.index('START TRANSACTION;'):text.index('-- Preview exactly')]

    def seed(self, database: str) -> None:
        path = self.csv_file(database + '_seed', [
            ['2023-01-02', 'A', '100.00'], ['2023-01-02', 'B', '200.00'],
            ['2023-01-03', 'A', '110.00']])
        self.query(self.new_schema(database, path))
        self.assertEqual(self.query(f'SELECT COUNT(*) FROM {database}.stock_data_eav;'), '3')

    def legacy(self, database: str, values: str) -> None:
        self.query(f"CREATE DATABASE {database} CHARACTER SET utf8mb4; USE {database}; "
                   "CREATE TABLE stock_data_eav (id INT NOT NULL AUTO_INCREMENT PRIMARY KEY, "
                   "entity DATE, attribute VARCHAR(100), value DECIMAL(20,2)) ENGINE=InnoDB; "
                   "INSERT INTO stock_data_eav(entity,attribute,value) VALUES " + values + ';')

    def migration(self, database: str) -> str:
        return (self.source_dir / 'migration_existing_stock_data.sql').read_text(
            encoding='utf-8').replace('USE stock_data;', f'USE {database};')

    def test_01_original_allows_repeated_three_rows(self):
        self.legacy('legacy_repeated', "('2023-01-02','A',100),('2023-01-02','B',200),('2023-01-03','A',110)")
        self.query("INSERT INTO legacy_repeated.stock_data_eav(entity,attribute,value) "
                   "VALUES ('2023-01-02','A',100),('2023-01-02','B',200),('2023-01-03','A',110);")
        self.assertEqual(self.query('SELECT COUNT(*) FROM legacy_repeated.stock_data_eav;'), '6')

    def test_02_new_ddl_and_duplicate_load_preserves_rows(self):
        database = 'repaired_duplicate'
        self.seed(database)
        before = self.rows(database)
        # First row could be valid: verify the entire failing LOAD is rolled back.
        path = self.csv_file('duplicate_load', [['2023-02-01','C','300'], ['2023-01-02','A','105']])
        self.query(self.load(database, path), expected_error=1062)
        self.assertEqual(self.rows(database), before)

    def test_03_invalid_loads_preserve_preexisting_and_earlier_rows(self):
        database = 'repaired_invalid'
        self.seed(database)
        before = self.rows(database)
        cases = [('negative', '2023-02-03', '-1', 3819),
                 ('zero', '2023-02-03', '0', 3819),
                 ('text_price', '2023-02-03', 'abc', 1366),
                 ('invalid_date', '2023-02-30', '100', 1292),
                 ('missing_date', '', '100', 1292),
                 ('missing_price', '2023-02-03', '', 1366),
                 ('zero_date', '0000-00-00', '100', 1292)]
        for name, day, price, error in cases:
            with self.subTest(name=name):
                path = self.csv_file(name, [['2023-02-01','C','300'],[day,'D',price]], validate=False)
                self.query(self.load(database, path), expected_error=error)
                self.assertEqual(self.rows(database), before)

    def test_04_unicode_quotes_commas_and_backslashes_roundtrip(self):
        database = 'quoted_roundtrip'
        companies = ['가상, "인용" \\ 기업 🎵', '가상 ", 경계', '"앞뒤"', r'\N', 'NULL', 'x" ,y']
        path = self.csv_file('quoted', [['2023-01-02', company, '1200.50'] for company in companies])
        self.query(self.new_schema(database, path))
        actual = self.query(f'SELECT HEX(attribute), value FROM {database}.stock_data_eav ORDER BY id;')
        expected = '\n'.join(company.encode('utf-8').hex().upper() + '\t1200.50' for company in companies)
        self.assertEqual(actual, expected)

    def test_05_null_empty_company_and_spaces_are_rejected(self):
        database = 'null_and_company'
        self.seed(database)
        before = self.rows(database)
        values = [("NULL,'D',100",1048), ("'2023-02-01',NULL,100",1048),
                  ("'2023-02-01','D',NULL",1048), ("'2023-02-01','',100",3819),
                  ("'2023-02-01',' D ',100",3819)]
        for row, error in values:
            with self.subTest(row=row):
                self.query(f'INSERT INTO {database}.stock_data_eav(entity,attribute,value) VALUES ({row});', error)
                self.assertEqual(self.rows(database), before)

    def test_06_date_range_and_ten_row_preview(self):
        database = 'date_range'
        self.seed(database)
        self.query(f"INSERT INTO {database}.stock_data_eav(entity,attribute,value) VALUES "
                   "('2022-12-31','A',99),('2024-01-01','A',111);")
        self.assertEqual(self.query(f"SELECT COUNT(*) FROM {database}.stock_data_eav "
                                   "WHERE entity >= '2023-01-01' AND entity < '2024-01-01';"), '3')
        extra = ','.join(f"('2025-01-{day:02}','A',100)" for day in range(1, 11))
        self.query(f'INSERT INTO {database}.stock_data_eav(entity,attribute,value) VALUES ' + extra + ';')
        preview = self.query(f'SELECT entity, attribute, value FROM {database}.stock_data_eav '
                             'ORDER BY entity, attribute LIMIT 10;')
        self.assertEqual(len(preview.splitlines()), 10)
        self.assertTrue(preview.splitlines()[0].startswith('2022-12-31\tA\t'))

    def test_07_clean_migration_keeps_values_and_adds_constraints(self):
        database = 'migration_clean'
        self.legacy(database, "('2023-01-02','A',100),('2023-01-02','B',200),('2023-01-03','A',110)")
        before = self.rows(database)
        self.query(self.migration(database))
        self.assertEqual(self.rows(database), before)
        self.assertEqual(self.query("SELECT COUNT(*) FROM information_schema.table_constraints "
                                   f"WHERE table_schema='{database}' AND table_name='stock_data_eav';"), '5')
        self.query(f"INSERT INTO {database}.stock_data_eav(entity,attribute,value) VALUES ('2023-01-02','A',100);", 1062)
        self.assertEqual(self.rows(database), before)

    def test_08_duplicate_migration_fails_without_deleting_rows(self):
        database = 'migration_duplicate'
        self.legacy(database, "('2023-01-02','A',100),('2023-01-02','A',105)")
        before = self.rows(database)
        self.query(self.migration(database), expected_error=1062)
        self.assertEqual(self.rows(database), before)
        self.assertEqual(self.query("SELECT COUNT(*) FROM information_schema.table_constraints "
                                   f"WHERE table_schema='{database}' AND table_name='stock_data_eav';"), '1')

    def test_09_invalid_migration_fails_without_deleting_rows(self):
        database = 'migration_invalid'
        self.legacy(database, "('2023-01-02','A',-1),('2023-01-03','A',110)")
        before = self.rows(database)
        self.query(self.migration(database), expected_error=3819)
        self.assertEqual(self.rows(database), before)
        self.assertEqual(self.query("SELECT COUNT(*) FROM information_schema.table_constraints "
                                   f"WHERE table_schema='{database}' AND table_name='stock_data_eav';"), '1')

    def test_10_decimal_precision_is_checked_before_mysql(self):
        frame = pd.DataFrame([['2023-01-02','A','1.001']], columns=['entity','attribute','value'])
        with self.assertRaises(DataValidationError):
            validate_stock_rows(frame)
        # MySQL DECIMAL may round fractional digits even with strict SQL mode.
        # Establish why the Python gate is required, rather than claiming DB rejects it.
        database = 'decimal_boundary'
        self.seed(database)
        self.query(f"INSERT INTO {database}.stock_data_eav(entity,attribute,value) VALUES ('2023-03-01','C',1.001);")
        self.assertEqual(self.query(f"SELECT value FROM {database}.stock_data_eav WHERE attribute='C';"), '1.00')

    def test_11_previous_quote_escape_silently_loaded_zero_rows(self):
        database = 'previous_quote_bug'
        path = self.csv_file('previous_quotes', [['2023-01-02','A','100'],
                                                 ['2023-01-02','B','200'],
                                                 ['2023-01-03','A','110']])
        old_sql = self.new_schema(database, path).replace("ESCAPED BY ''\nLINES", "ESCAPED BY '\"'\nLINES")
        self.query(old_sql)
        self.assertEqual(self.query(f'SELECT COUNT(*) FROM {database}.stock_data_eav;'), '0')


def run_checks(client, memory_name, uploads, source_dir, child_env, trace):
    for key, value in dict(client=client, memory_name=memory_name, uploads=uploads,
                           source_dir=source_dir, child_env=child_env, trace=trace).items():
        setattr(MySQLChecks, key, value)
    output = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MySQLChecks)
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    return result, output.getvalue()
