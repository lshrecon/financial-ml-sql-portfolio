-- MySQL 8.0.16+ (enforced CHECK constraints).
-- Synthetic integration checks passed on MySQL 8.4.11 (2026-10-07); no user DB was changed.
-- For a NEW table only. For an existing table, inspect migration_existing_stock_data.sql first.
-- Validate the CSV with: python stock_etl.py stock_data_eav.csv
-- Do not run the MySQL client with --force: an error must stop the import.
SET SESSION sql_mode = 'STRICT_ALL_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';
SET NAMES utf8mb4;

CREATE DATABASE IF NOT EXISTS stock_data CHARACTER SET utf8mb4 COLLATE utf8mb4_bin;
USE stock_data;

CREATE TABLE stock_data_eav (
    id INT NOT NULL AUTO_INCREMENT,
    entity DATE NOT NULL,
    attribute VARCHAR(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
    value DECIMAL(20, 2) NOT NULL,
    PRIMARY KEY (id),
    CONSTRAINT uq_stock_date_company UNIQUE (entity, attribute),
    CONSTRAINT ck_stock_price_positive CHECK (value > 0),
    CONSTRAINT ck_stock_company_nonempty CHECK (
        CHAR_LENGTH(attribute) > 0 AND CHAR_LENGTH(attribute) = CHAR_LENGTH(TRIM(attribute))
    ),
    CONSTRAINT ck_stock_valid_date CHECK (
        YEAR(entity) >= 1000 AND MONTH(entity) BETWEEN 1 AND 12
        AND LAST_DAY(entity) IS NOT NULL
        AND DAY(entity) BETWEEN 1 AND DAY(LAST_DAY(entity))
    )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

SHOW VARIABLES LIKE 'secure_file_priv';

-- Replace with the SERVER-SIDE CSV path permitted by secure_file_priv.
-- The isolated Windows check used an ASCII upload path; verify filename encoding on your host.
-- CSV contract: UTF-8 without BOM, LF, all fields quoted, internal quotes doubled.
-- No LOCAL / IGNORE / REPLACE: invalid data or repeated keys must not be silently accepted.
START TRANSACTION;
LOAD DATA INFILE '/path/to/mysql-uploads/stock_data_eav.csv'
INTO TABLE stock_data_eav
CHARACTER SET utf8mb4
-- CSV doubles internal quotes; do not treat quotes/backslashes as escape prefixes.
-- ESCAPED BY '"' plus IGNORE 1 LINES skipped this quoted file entirely in MySQL 8.4.11.
FIELDS TERMINATED BY ',' ENCLOSED BY '"' ESCAPED BY ''
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(entity, attribute, value);
COMMIT;

-- Preview exactly 10 rows in a stable order.
SELECT entity, attribute, value
FROM stock_data_eav
ORDER BY entity, attribute
LIMIT 10;

SELECT COUNT(*) AS row_count FROM stock_data_eav;

-- DATE range, avoiding conversion of the DATE column to a LIKE string.
SELECT entity, attribute, value
FROM stock_data_eav
WHERE entity >= '2023-01-01' AND entity < '2024-01-01'
ORDER BY entity, attribute;
