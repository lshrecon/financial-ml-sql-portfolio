-- Existing MySQL table migration: review and run separately; never run automatically.
-- Requires MySQL 8.0.16+. No row deletion or automatic deduplication.
-- Back up and identify any other tables/applications depending on this table first.
-- Clean/invalid synthetic-table cases were checked on MySQL 8.4.11 (2026-10-07).
-- This does not authorize or validate migration of any existing user database.
USE stock_data;
SET SESSION sql_mode = 'STRICT_ALL_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';
SET NAMES utf8mb4;

SHOW CREATE TABLE stock_data_eav;

-- 1. Duplicate business keys under the target case-sensitive collation.
SELECT entity, CONVERT(attribute USING utf8mb4) COLLATE utf8mb4_bin AS company_key,
       COUNT(*) AS duplicate_count, MIN(value) AS minimum_price, MAX(value) AS maximum_price
FROM stock_data_eav
GROUP BY entity, company_key
HAVING COUNT(*) > 1;

-- 2. Invalid rows. If either query returns rows, STOP and inspect the source data.
-- Equal prices do not authorize deleting duplicates; differing prices require a source decision.
SELECT id, entity, attribute, value
FROM stock_data_eav
WHERE entity IS NULL OR YEAR(entity) < 1000 OR MONTH(entity) NOT BETWEEN 1 AND 12
   OR LAST_DAY(entity) IS NULL OR DAY(entity) NOT BETWEEN 1 AND DAY(LAST_DAY(entity))
   OR attribute IS NULL OR CHAR_LENGTH(attribute) = 0
   OR CHAR_LENGTH(attribute) <> CHAR_LENGTH(TRIM(attribute))
   OR value IS NULL OR value <= 0;

-- 3. Run ONLY after both checks return no rows and SHOW CREATE confirms these
-- constraints do not already exist. ALTER changes constraints, never removes rows.
-- If duplicates/invalid rows remain, strict mode and the added constraints must reject the ALTER.
ALTER TABLE stock_data_eav
    MODIFY entity DATE NOT NULL,
    MODIFY attribute VARCHAR(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
    MODIFY value DECIMAL(20, 2) NOT NULL,
    ADD CONSTRAINT uq_stock_date_company UNIQUE (entity, attribute),
    ADD CONSTRAINT ck_stock_price_positive CHECK (value > 0),
    ADD CONSTRAINT ck_stock_company_nonempty CHECK (
        CHAR_LENGTH(attribute) > 0 AND CHAR_LENGTH(attribute) = CHAR_LENGTH(TRIM(attribute))
    ),
    ADD CONSTRAINT ck_stock_valid_date CHECK (
        YEAR(entity) >= 1000 AND MONTH(entity) BETWEEN 1 AND 12
        AND LAST_DAY(entity) IS NOT NULL
        AND DAY(entity) BETWEEN 1 AND DAY(LAST_DAY(entity))
    );

-- Verify the installed constraints before importing further files.
SHOW CREATE TABLE stock_data_eav;
