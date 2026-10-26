from pathlib import Path
import os
import re
import time
import unicodedata
import shutil

import duckdb
import pandas as pd
from unidecode import unidecode


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(
    r"C:\Users\dhath\OneDrive\Documents\ML_Challenge"
)

TEST_DIR = BASE_DIR / "test_cleaned_data"
OUTPUT_DIR = BASE_DIR / "output"

S1_PATH = TEST_DIR / "source1_clean.tsv"
S2_PATH = TEST_DIR / "source2_clean.tsv"
S3_PATH = TEST_DIR / "source3_clean.tsv"

DB_PATH = OUTPUT_DIR / "blocking_test_v4_1.duckdb"
TEMP_DIR = OUTPUT_DIR / "duckdb_temp_test_v4_1"

FINAL_PATH = OUTPUT_DIR / "candidate_pairs_test_v4_1.tsv"
TEMP_FINAL_PATH = OUTPUT_DIR / "candidate_pairs_test_v4_1.tmp.tsv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

THREADS = max(4, min(12, os.cpu_count() or 8))
MEMORY_LIMIT = "8GB"

CHUNK_SIZE = 100_000

# V3 validated settings
TOP_RARE_TOKENS = 3

MAX_KEY_FREQUENCY = 100

BLOCK_LIMITS = {
    "exact_name": 300,
    "exact_address": 500,
    "name_token": 250,
    "address_token": 300,
    "name_pair": 200,
    "address_pair": 250,
    "name_prefix": 200,
    "name_suffix": 200,
    "address_prefix": 250,
    "address_suffix": 250,
    "address_num_token": 250,
    "postal": 200,
}

# V4.1 validated settings
V4_MAX_KEY_FREQUENCY = 50
V4_MAX_ROWS_PER_KEY = 50
V4_MAX_TOKEN_BLOCKS_PER_RECORD = 2
V4_MIN_TOKEN_LENGTH = 4
V4_PREFIX_LEN = 6
V4_SUFFIX_LEN = 6


# ============================================================
# 3. BASIC CHECKS
# ============================================================

print("=" * 75)
print("BUSINESS ENTITY RESOLUTION")
print("TEST DATA BLOCKING V4.1")
print("=" * 75)

for path in [S1_PATH, S2_PATH, S3_PATH]:

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

print()
print("S1:", S1_PATH)
print("S2:", S2_PATH)
print("S3:", S3_PATH)
print()
print("DuckDB threads:", THREADS)
print("DuckDB memory:", MEMORY_LIMIT)


# ============================================================
# 4. NORMALIZATION HELPERS
# ============================================================

def clean_text(value):

    if value is None:
        return ""

    value = str(value).strip().casefold()

    if not value:
        return ""

    value = value.replace("&", " and ")

    value = unicodedata.normalize(
        "NFKC",
        value
    )

    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


def transliterate_text(value):

    if not value:
        return ""

    value = unidecode(str(value))

    value = value.casefold()

    value = value.replace("&", " and ")

    value = re.sub(
        r"[^a-z0-9\s]",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


def is_non_ascii(value):

    if not value:
        return False

    return any(
        ord(c) > 127
        for c in str(value)
    )


def get_tokens(value):

    if not value:
        return []

    result = []

    for token in str(value).split():

        if len(token) < V4_MIN_TOKEN_LENGTH:
            continue

        if token.isdigit():
            continue

        result.append(token)

    return list(set(result))


def make_v4_blocks(name):

    """
    V4.1 transliteration blocks.

    These are used only for cross-script matching.
    """

    name = clean_text(name)

    if not name:
        return []

    translit = transliterate_text(name)

    if not translit:
        return []

    blocks = []

    # Full transliterated name
    blocks.append(
        ("translit_name", translit)
    )

    compact = re.sub(
        r"[^a-z0-9]",
        "",
        translit
    )

    if len(compact) >= V4_PREFIX_LEN:

        blocks.append(
            (
                "translit_prefix",
                compact[:V4_PREFIX_LEN]
            )
        )

    if len(compact) >= V4_SUFFIX_LEN:

        blocks.append(
            (
                "translit_suffix",
                compact[-V4_SUFFIX_LEN:]
            )
        )

    tokens = get_tokens(translit)

    tokens = sorted(
        tokens,
        key=lambda x: (-len(x), x)
    )

    for token in tokens[
        :V4_MAX_TOKEN_BLOCKS_PER_RECORD
    ]:

        blocks.append(
            (
                "translit_token",
                token
            )
        )

    return list(set(blocks))


# ============================================================
# 5. CONNECT TO DUCKDB
# ============================================================

print()
print("Opening DuckDB...")

con = duckdb.connect(
    str(DB_PATH)
)

con.execute(
    f"SET threads TO {THREADS}"
)

con.execute(
    f"SET memory_limit = '{MEMORY_LIMIT}'"
)

con.execute(
    f"SET temp_directory = '{TEMP_DIR.as_posix()}'"
)

print("DuckDB ready.")


# ============================================================
# 6. CREATE SOURCE VIEWS
# ============================================================

print()
print("Loading test source views...")

con.execute(
    f"""
    CREATE OR REPLACE VIEW source1 AS
    SELECT
        entity_id,
        business_name_clean,
        business_address_clean,
        lower(trim(country)) AS country
    FROM read_csv(
        '{S1_PATH.as_posix()}',
        delim='\\t',
        header=true,
        all_varchar=true,
        ignore_errors=false
    )
    """
)

con.execute(
    f"""
    CREATE OR REPLACE VIEW source2 AS
    SELECT
        entity_id,
        business_name_clean,
        business_address_clean,
        lower(trim(country)) AS country
    FROM read_csv(
        '{S2_PATH.as_posix()}',
        delim='\\t',
        header=true,
        all_varchar=true,
        ignore_errors=false
    )
    """
)

con.execute(
    f"""
    CREATE OR REPLACE VIEW source3 AS
    SELECT
        entity_id,
        business_name_clean,
        business_address_clean,
        lower(trim(country)) AS country
    FROM read_csv(
        '{S3_PATH.as_posix()}',
        delim='\\t',
        header=true,
        all_varchar=true,
        ignore_errors=false
    )
    """
)

con.execute(
    """
    CREATE OR REPLACE VIEW target AS
    SELECT * FROM source2
    UNION ALL
    SELECT * FROM source3
    """
)


# ============================================================
# 7. COUNTS
# ============================================================

print()
print("Counting records...")

s1_count = con.execute(
    "SELECT COUNT(*) FROM source1"
).fetchone()[0]

s2_count = con.execute(
    "SELECT COUNT(*) FROM source2"
).fetchone()[0]

s3_count = con.execute(
    "SELECT COUNT(*) FROM source3"
).fetchone()[0]

target_count = s2_count + s3_count

print()
print("S1 records :", f"{s1_count:,}")
print("S2 records :", f"{s2_count:,}")
print("S3 records :", f"{s3_count:,}")
print("Target     :", f"{target_count:,}")


# ============================================================
# 8. CREATE TOKEN TABLES
# ============================================================

print()
print("=" * 75)
print("BUILDING V3 BLOCKS")
print("=" * 75)

start_v3 = time.time()

for table in [
    "s1_name_tokens",
    "target_name_tokens",
    "s1_address_tokens",
    "target_address_tokens",
    "s1_rare_name",
    "target_rare_name",
    "s1_rare_address",
    "target_rare_address",
]:
    con.execute(
        f"DROP TABLE IF EXISTS {table}"
    )


# ------------------------------------------------------------
# NAME TOKENS
# ------------------------------------------------------------

con.execute(
    """
    CREATE TABLE s1_name_tokens AS
    SELECT DISTINCT
        s.entity_id,
        s.country,
        trim(u.token) AS token
    FROM source1 s
    CROSS JOIN UNNEST(
        string_split(
            coalesce(s.business_name_clean, ''),
            ' '
        )
    ) AS u(token)
    WHERE length(trim(u.token)) >= 4
      AND NOT regexp_full_match(
          trim(u.token),
          '^[0-9]+$'
      )
      AND trim(u.token) <> ''
    """
)

con.execute(
    """
    CREATE TABLE target_name_tokens AS
    SELECT DISTINCT
        t.entity_id,
        t.country,
        trim(u.token) AS token
    FROM target t
    CROSS JOIN UNNEST(
        string_split(
            coalesce(t.business_name_clean, ''),
            ' '
        )
    ) AS u(token)
    WHERE length(trim(u.token)) >= 4
      AND NOT regexp_full_match(
          trim(u.token),
          '^[0-9]+$'
      )
      AND trim(u.token) <> ''
    """
)


# ------------------------------------------------------------
# ADDRESS TOKENS
# ------------------------------------------------------------

con.execute(
    """
    CREATE TABLE s1_address_tokens AS
    SELECT DISTINCT
        s.entity_id,
        s.country,
        trim(u.token) AS token
    FROM source1 s
    CROSS JOIN UNNEST(
        string_split(
            coalesce(s.business_address_clean, ''),
            ' '
        )
    ) AS u(token)
    WHERE length(trim(u.token)) >= 3
      AND NOT regexp_full_match(
          trim(u.token),
          '^[0-9]+$'
      )
      AND trim(u.token) <> ''
    """
)

con.execute(
    """
    CREATE TABLE target_address_tokens AS
    SELECT DISTINCT
        t.entity_id,
        t.country,
        trim(u.token) AS token
    FROM target t
    CROSS JOIN UNNEST(
        string_split(
            coalesce(t.business_address_clean, ''),
            ' '
        )
    ) AS u(token)
    WHERE length(trim(u.token)) >= 3
      AND NOT regexp_full_match(
          trim(u.token),
          '^[0-9]+$'
      )
      AND trim(u.token) <> ''
    """
)


# ============================================================
# 9. TOKEN FREQUENCIES
# ============================================================

print("Calculating token frequencies...")

con.execute(
    """
    CREATE OR REPLACE TABLE s1_name_token_freq AS
    SELECT
        country,
        token,
        COUNT(*) AS freq
    FROM s1_name_tokens
    GROUP BY country, token
    """
)

con.execute(
    """
    CREATE OR REPLACE TABLE target_name_token_freq AS
    SELECT
        country,
        token,
        COUNT(*) AS freq
    FROM target_name_tokens
    GROUP BY country, token
    """
)

con.execute(
    """
    CREATE OR REPLACE TABLE s1_address_token_freq AS
    SELECT
        country,
        token,
        COUNT(*) AS freq
    FROM s1_address_tokens
    GROUP BY country, token
    """
)

con.execute(
    """
    CREATE OR REPLACE TABLE target_address_token_freq AS
    SELECT
        country,
        token,
        COUNT(*) AS freq
    FROM target_address_tokens
    GROUP BY country, token
    """
)


# ============================================================
# 10. SELECT THREE RAREST TOKENS PER RECORD
# ============================================================

print("Selecting rare tokens...")

con.execute(
    f"""
    CREATE TABLE s1_rare_name AS
    SELECT
        x.entity_id,
        x.country,
        x.token,
        x.freq
    FROM (
        SELECT
            t.entity_id,
            t.country,
            t.token,
            f.freq,
            ROW_NUMBER() OVER (
                PARTITION BY t.entity_id
                ORDER BY f.freq, length(t.token) DESC, t.token
            ) AS rn
        FROM s1_name_tokens t
        INNER JOIN s1_name_token_freq f
            ON t.country = f.country
           AND t.token = f.token
    ) x
    WHERE x.rn <= {TOP_RARE_TOKENS}
    """
)

con.execute(
    f"""
    CREATE TABLE target_rare_name AS
    SELECT
        x.entity_id,
        x.country,
        x.token,
        x.freq
    FROM (
        SELECT
            t.entity_id,
            t.country,
            t.token,
            f.freq,
            ROW_NUMBER() OVER (
                PARTITION BY t.entity_id
                ORDER BY f.freq, length(t.token) DESC, t.token
            ) AS rn
        FROM target_name_tokens t
        INNER JOIN target_name_token_freq f
            ON t.country = f.country
           AND t.token = f.token
    ) x
    WHERE x.rn <= {TOP_RARE_TOKENS}
    """
)

con.execute(
    f"""
    CREATE TABLE s1_rare_address AS
    SELECT
        x.entity_id,
        x.country,
        x.token,
        x.freq
    FROM (
        SELECT
            t.entity_id,
            t.country,
            t.token,
            f.freq,
            ROW_NUMBER() OVER (
                PARTITION BY t.entity_id
                ORDER BY f.freq, length(t.token) DESC, t.token
            ) AS rn
        FROM s1_address_tokens t
        INNER JOIN s1_address_token_freq f
            ON t.country = f.country
           AND t.token = f.token
    ) x
    WHERE x.rn <= {TOP_RARE_TOKENS}
    """
)

con.execute(
    f"""
    CREATE TABLE target_rare_address AS
    SELECT
        x.entity_id,
        x.country,
        x.token,
        x.freq
    FROM (
        SELECT
            t.entity_id,
            t.country,
            t.token,
            f.freq,
            ROW_NUMBER() OVER (
                PARTITION BY t.entity_id
                ORDER BY f.freq, length(t.token) DESC, t.token
            ) AS rn
        FROM target_address_tokens t
        INNER JOIN target_address_token_freq f
            ON t.country = f.country
           AND t.token = f.token
    ) x
    WHERE x.rn <= {TOP_RARE_TOKENS}
    """
)


# ============================================================
# 11. BUILD BLOCK TABLES
# ============================================================

print("Building S1 block table...")

con.execute(
    "DROP TABLE IF EXISTS s1_blocks_raw"
)

con.execute(
    """
    CREATE TABLE s1_blocks_raw AS

    -- EXACT NAME
    SELECT
        entity_id,
        country,
        'exact_name' AS block_type,
        business_name_clean AS block_key
    FROM source1
    WHERE business_name_clean <> ''

    UNION ALL

    -- EXACT ADDRESS
    SELECT
        entity_id,
        country,
        'exact_address',
        business_address_clean
    FROM source1
    WHERE business_address_clean <> ''

    UNION ALL

    -- RARE NAME TOKEN
    SELECT
        entity_id,
        country,
        'name_token',
        token
    FROM s1_rare_name

    UNION ALL

    -- RARE ADDRESS TOKEN
    SELECT
        entity_id,
        country,
        'address_token',
        token
    FROM s1_rare_address

    UNION ALL

    -- NAME TOKEN PAIRS
    SELECT
        a.entity_id,
        a.country,
        'name_pair',
        a.token || '|' || b.token
    FROM s1_rare_name a
    INNER JOIN s1_rare_name b
        ON a.entity_id = b.entity_id
       AND a.token < b.token

    UNION ALL

    -- ADDRESS TOKEN PAIRS
    SELECT
        a.entity_id,
        a.country,
        'address_pair',
        a.token || '|' || b.token
    FROM s1_rare_address a
    INNER JOIN s1_rare_address b
        ON a.entity_id = b.entity_id
       AND a.token < b.token

    UNION ALL

    -- NAME TOKEN PREFIX
    SELECT DISTINCT
        entity_id,
        country,
        'name_prefix',
        left(token, 4)
    FROM s1_name_tokens
    WHERE length(token) >= 4

    UNION ALL

    -- NAME TOKEN SUFFIX
    SELECT DISTINCT
        entity_id,
        country,
        'name_suffix',
        right(token, 4)
    FROM s1_name_tokens
    WHERE length(token) >= 4

    UNION ALL

    -- ADDRESS TOKEN PREFIX
    SELECT DISTINCT
        entity_id,
        country,
        'address_prefix',
        left(token, 4)
    FROM s1_address_tokens
    WHERE length(token) >= 4

    UNION ALL

    -- ADDRESS TOKEN SUFFIX
    SELECT DISTINCT
        entity_id,
        country,
        'address_suffix',
        right(token, 4)
    FROM s1_address_tokens
    WHERE length(token) >= 4

    UNION ALL

    -- ADDRESS NUMBER + TOKEN
    SELECT DISTINCT
        a.entity_id,
        a.country,
        'address_num_token',
        regexp_extract(
            s.business_address_clean,
            '\\b[0-9]{1,6}\\b',
            0
        ) || '|' || a.token
    FROM s1_address_tokens a
    INNER JOIN source1 s
        ON a.entity_id = s.entity_id
    WHERE regexp_matches(
        s.business_address_clean,
        '\\b[0-9]{1,6}\\b'
    )

    UNION ALL

    -- POSTAL
    SELECT DISTINCT
        entity_id,
        country,
        'postal',
        regexp_extract(
            business_address_clean,
            '\\b[0-9]{5,6}\\b',
            0
        )
    FROM source1
    WHERE regexp_matches(
        business_address_clean,
        '\\b[0-9]{5,6}\\b'
    )
    """
)


print("Building target block table...")

con.execute(
    "DROP TABLE IF EXISTS target_blocks_raw"
)

con.execute(
    """
    CREATE TABLE target_blocks_raw AS

    SELECT
        entity_id,
        country,
        'exact_name' AS block_type,
        business_name_clean AS block_key
    FROM target
    WHERE business_name_clean <> ''

    UNION ALL

    SELECT
        entity_id,
        country,
        'exact_address',
        business_address_clean
    FROM target
    WHERE business_address_clean <> ''

    UNION ALL

    SELECT
        entity_id,
        country,
        'name_token',
        token
    FROM target_rare_name

    UNION ALL

    SELECT
        entity_id,
        country,
        'address_token',
        token
    FROM target_rare_address

    UNION ALL

    SELECT
        a.entity_id,
        a.country,
        'name_pair',
        a.token || '|' || b.token
    FROM target_rare_name a
    INNER JOIN target_rare_name b
        ON a.entity_id = b.entity_id
       AND a.token < b.token

    UNION ALL

    SELECT
        a.entity_id,
        a.country,
        'address_pair',
        a.token || '|' || b.token
    FROM target_rare_address a
    INNER JOIN target_rare_address b
        ON a.entity_id = b.entity_id
       AND a.token < b.token

    UNION ALL

    SELECT DISTINCT
        entity_id,
        country,
        'name_prefix',
        left(token, 4)
    FROM target_name_tokens
    WHERE length(token) >= 4

    UNION ALL

    SELECT DISTINCT
        entity_id,
        country,
        'name_suffix',
        right(token, 4)
    FROM target_name_tokens
    WHERE length(token) >= 4

    UNION ALL

    SELECT DISTINCT
        entity_id,
        country,
        'address_prefix',
        left(token, 4)
    FROM target_address_tokens
    WHERE length(token) >= 4

    UNION ALL

    SELECT DISTINCT
        entity_id,
        country,
        'address_suffix',
        right(token, 4)
    FROM target_address_tokens
    WHERE length(token) >= 4

    UNION ALL

    SELECT DISTINCT
        a.entity_id,
        a.country,
        'address_num_token',
        regexp_extract(
            t.business_address_clean,
            '\\b[0-9]{1,6}\\b',
            0
        ) || '|' || a.token
    FROM target_address_tokens a
    INNER JOIN target t
        ON a.entity_id = t.entity_id
    WHERE regexp_matches(
        t.business_address_clean,
        '\\b[0-9]{1,6}\\b'
    )

    UNION ALL

    SELECT DISTINCT
        entity_id,
        country,
        'postal',
        regexp_extract(
            business_address_clean,
            '\\b[0-9]{5,6}\\b',
            0
        )
    FROM target
    WHERE regexp_matches(
        business_address_clean,
        '\\b[0-9]{5,6}\\b'
    )
    """
)


# ============================================================
# 12. REMOVE DUPLICATE BLOCK ROWS
# ============================================================

print("Deduplicating block rows...")

con.execute(
    """
    CREATE OR REPLACE TABLE s1_blocks AS
    SELECT DISTINCT
        entity_id,
        country,
        block_type,
        block_key
    FROM s1_blocks_raw
    WHERE block_key <> ''
    """
)

con.execute(
    """
    CREATE OR REPLACE TABLE target_blocks AS
    SELECT DISTINCT
        entity_id,
        country,
        block_type,
        block_key
    FROM target_blocks_raw
    WHERE block_key <> ''
    """
)


# ============================================================
# 13. LIMIT BLOCK FREQUENCY
# ============================================================

print("Applying V3 block-frequency limits...")

con.execute(
    """
    CREATE OR REPLACE TABLE s1_blocks_limited AS
    SELECT *
    FROM (
        SELECT
            b.*,
            ROW_NUMBER() OVER (
                PARTITION BY
                    country,
                    block_type,
                    block_key
                ORDER BY entity_id
            ) AS rn
        FROM s1_blocks b
    )
    WHERE
        rn <= CASE block_type
            WHEN 'exact_name' THEN 300
            WHEN 'exact_address' THEN 500
            WHEN 'name_token' THEN 250
            WHEN 'address_token' THEN 300
            WHEN 'name_pair' THEN 200
            WHEN 'address_pair' THEN 250
            WHEN 'name_prefix' THEN 200
            WHEN 'name_suffix' THEN 200
            WHEN 'address_prefix' THEN 250
            WHEN 'address_suffix' THEN 250
            WHEN 'address_num_token' THEN 250
            WHEN 'postal' THEN 200
            ELSE 100
        END
    """
)

con.execute(
    f"""
    CREATE OR REPLACE TABLE target_blocks_limited AS
    SELECT *
    FROM (
        SELECT
            b.*,
            COUNT(*) OVER (
                PARTITION BY
                    country,
                    block_type,
                    block_key
            ) AS block_freq,
            ROW_NUMBER() OVER (
                PARTITION BY
                    country,
                    block_type,
                    block_key
                ORDER BY entity_id
            ) AS rn
        FROM target_blocks b
    )
    WHERE block_freq <= {MAX_KEY_FREQUENCY}
      AND rn <= CASE block_type
            WHEN 'exact_name' THEN 300
            WHEN 'exact_address' THEN 500
            WHEN 'name_token' THEN 250
            WHEN 'address_token' THEN 300
            WHEN 'name_pair' THEN 200
            WHEN 'address_pair' THEN 250
            WHEN 'name_prefix' THEN 200
            WHEN 'name_suffix' THEN 200
            WHEN 'address_prefix' THEN 250
            WHEN 'address_suffix' THEN 250
            WHEN 'address_num_token' THEN 250
            WHEN 'postal' THEN 200
            ELSE 100
        END
    """
)


# ============================================================
# 14. V3 CANDIDATES
# ============================================================

print()
print("Generating V3 candidate pairs...")

con.execute(
    "DROP TABLE IF EXISTS candidate_pairs_v3"
)

con.execute(
    """
    CREATE TABLE candidate_pairs_v3 AS
    SELECT DISTINCT
        s.entity_id AS source1_entity_id,
        t.entity_id AS matched_entity_id
    FROM s1_blocks_limited s
    INNER JOIN target_blocks_limited t
        ON s.country = t.country
       AND s.block_type = t.block_type
       AND s.block_key = t.block_key
    WHERE s.entity_id LIKE 'S1-%'
      AND (
          t.entity_id LIKE 'S2-%'
          OR
          t.entity_id LIKE 'S3-%'
      )
    """
)

v3_count = con.execute(
    "SELECT COUNT(*) FROM candidate_pairs_v3"
).fetchone()[0]

print()
print("V3 candidate pairs:", f"{v3_count:,}")


# ============================================================
# 15. V4.1 CROSS-SCRIPT BLOCKS
# ============================================================

print()
print("=" * 75)
print("BUILDING V4.1 CROSS-SCRIPT BLOCKS")
print("=" * 75)

v4_start = time.time()


def process_v4_file(
    path,
    table_name,
    source_label
):

    print()
    print("Processing", source_label)

    all_rows = []

    total_rows = 0
    total_blocks = 0

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        rows = []

        for row in chunk.itertuples(
            index=False
        ):

            entity_id = row.entity_id
            country = str(
                row.country
            ).casefold().strip()

            name = getattr(
                row,
                "business_name_clean",
                ""
            )

            if not name:
                continue

            non_ascii = is_non_ascii(name)

            blocks = make_v4_blocks(name)

            for block_type, block_key in blocks:

                rows.append(
                    (
                        str(entity_id),
                        country,
                        block_type,
                        block_key,
                        non_ascii
                    )
                )

        if rows:

            df = pd.DataFrame(
                rows,
                columns=[
                    "entity_id",
                    "country",
                    "block_type",
                    "block_key",
                    "is_non_ascii"
                ]
            )

            df.drop_duplicates(
                inplace=True
            )

            con.register(
                "v4_chunk",
                df
            )

            con.execute(
                f"""
                INSERT INTO {table_name}
                SELECT *
                FROM v4_chunk
                """
            )

            con.unregister(
                "v4_chunk"
            )

            total_blocks += len(df)

        total_rows += len(chunk)

        if chunk_no % 5 == 0:

            elapsed = (
                time.time() - v4_start
            ) / 60

            print(
                f"{source_label}: "
                f"{total_rows:,} rows | "
                f"{total_blocks:,} blocks | "
                f"{elapsed:.1f} min"
            )

    print(
        source_label,
        "complete:",
        f"{total_rows:,} rows,",
        f"{total_blocks:,} blocks"
    )


con.execute(
    "DROP TABLE IF EXISTS v4_s1_blocks"
)

con.execute(
    "DROP TABLE IF EXISTS v4_target_blocks"
)

con.execute(
    """
    CREATE TABLE v4_s1_blocks (
        entity_id VARCHAR,
        country VARCHAR,
        block_type VARCHAR,
        block_key VARCHAR,
        is_non_ascii BOOLEAN
    )
    """
)

con.execute(
    """
    CREATE TABLE v4_target_blocks (
        entity_id VARCHAR,
        country VARCHAR,
        block_type VARCHAR,
        block_key VARCHAR,
        is_non_ascii BOOLEAN
    )
    """
)


process_v4_file(
    S1_PATH,
    "v4_s1_blocks",
    "S1"
)

process_v4_file(
    S2_PATH,
    "v4_target_blocks",
    "S2"
)

process_v4_file(
    S3_PATH,
    "v4_target_blocks",
    "S3"
)


# ============================================================
# 16. V4.1 BLOCK FREQUENCY LIMIT
# ============================================================

print()
print("Applying V4.1 frequency limits...")

con.execute(
    """
    CREATE OR REPLACE TABLE v4_target_limited AS
    SELECT
        entity_id,
        country,
        block_type,
        block_key,
        is_non_ascii
    FROM (
        SELECT
            t.*,
            COUNT(*) OVER (
                PARTITION BY
                    country,
                    block_type,
                    block_key
            ) AS freq,
            ROW_NUMBER() OVER (
                PARTITION BY
                    country,
                    block_type,
                    block_key
                ORDER BY entity_id
            ) AS rn
        FROM v4_target_blocks t
    )
    WHERE freq <= 50
      AND rn <= 50
    """
)

con.execute(
    """
    CREATE OR REPLACE TABLE v4_s1_limited AS
    SELECT
        entity_id,
        country,
        block_type,
        block_key,
        is_non_ascii
    FROM (
        SELECT
            s.*,
            COUNT(*) OVER (
                PARTITION BY
                    country,
                    block_type,
                    block_key
            ) AS freq,
            ROW_NUMBER() OVER (
                PARTITION BY
                    country,
                    block_type,
                    block_key
                ORDER BY entity_id
            ) AS rn
        FROM v4_s1_blocks s
    )
    WHERE freq <= 50
      AND rn <= 50
    """
)


# ============================================================
# 17. V4.1 ADDITIONAL CANDIDATES
# ============================================================

print()
print("Generating V4.1 cross-script candidates...")

con.execute(
    "DROP TABLE IF EXISTS v4_additional_raw"
)

con.execute(
    """
    CREATE TABLE v4_additional_raw AS
    SELECT DISTINCT
        s.entity_id AS source1_entity_id,
        t.entity_id AS matched_entity_id
    FROM v4_s1_limited s
    INNER JOIN v4_target_limited t
        ON s.country = t.country
       AND s.block_type = t.block_type
       AND s.block_key = t.block_key
       AND s.is_non_ascii <> t.is_non_ascii
    WHERE s.entity_id LIKE 'S1-%'
      AND (
          t.entity_id LIKE 'S2-%'
          OR
          t.entity_id LIKE 'S3-%'
      )
    """
)

raw_v4_count = con.execute(
    """
    SELECT COUNT(*)
    FROM v4_additional_raw
    """
).fetchone()[0]

print(
    "Raw V4.1 candidates:",
    f"{raw_v4_count:,}"
)


# ============================================================
# 18. REMOVE V3 OVERLAP
# ============================================================

print()
print("Removing V3/V4.1 overlap...")

con.execute(
    "DROP TABLE IF EXISTS v4_new"
)

con.execute(
    """
    CREATE TABLE v4_new AS
    SELECT
        v.source1_entity_id,
        v.matched_entity_id
    FROM v4_additional_raw v
    ANTI JOIN candidate_pairs_v3 c
        ON v.source1_entity_id =
           c.source1_entity_id
       AND v.matched_entity_id =
           c.matched_entity_id
    """
)

new_count = con.execute(
    "SELECT COUNT(*) FROM v4_new"
).fetchone()[0]

overlap_count = raw_v4_count - new_count

print(
    "V3/V4.1 overlap:",
    f"{overlap_count:,}"
)

print(
    "Genuinely new V4.1 candidates:",
    f"{new_count:,}"
)


# ============================================================
# 19. SAFETY CHECKS
# ============================================================

print()
print("=" * 75)
print("SAFETY CHECKS")
print("=" * 75)

duplicate_new = con.execute(
    """
    SELECT COUNT(*)
    FROM (
        SELECT
            source1_entity_id,
            matched_entity_id,
            COUNT(*) AS c
        FROM v4_new
        GROUP BY
            source1_entity_id,
            matched_entity_id
        HAVING COUNT(*) > 1
    )
    """
).fetchone()[0]

invalid_v3 = con.execute(
    """
    SELECT COUNT(*)
    FROM candidate_pairs_v3
    WHERE source1_entity_id NOT LIKE 'S1-%'
       OR (
           matched_entity_id NOT LIKE 'S2-%'
           AND matched_entity_id NOT LIKE 'S3-%'
       )
    """
).fetchone()[0]

invalid_v4 = con.execute(
    """
    SELECT COUNT(*)
    FROM v4_new
    WHERE source1_entity_id NOT LIKE 'S1-%'
       OR (
           matched_entity_id NOT LIKE 'S2-%'
           AND matched_entity_id NOT LIKE 'S3-%'
       )
    """
).fetchone()[0]

print(
    "Duplicate new pairs:",
    f"{duplicate_new:,}"
)

print(
    "Invalid V3 pairs:",
    f"{invalid_v3:,}"
)

print(
    "Invalid V4.1 pairs:",
    f"{invalid_v4:,}"
)

if duplicate_new != 0:
    raise RuntimeError(
        "Duplicate V4.1 candidate pairs detected."
    )

if invalid_v3 != 0 or invalid_v4 != 0:
    raise RuntimeError(
        "Invalid source pair detected."
    )


# ============================================================
# 20. FINAL EXPECTED COUNT
# ============================================================

expected_final = v3_count + new_count

print()
print(
    "Expected final candidates:",
    f"{expected_final:,}"
)


# ============================================================
# 21. EXPORT V3 DIRECTLY TO TEMP FINAL
# ============================================================

print()
print("=" * 75)
print("EXPORTING FINAL CANDIDATE FILE")
print("=" * 75)

if TEMP_FINAL_PATH.exists():
    TEMP_FINAL_PATH.unlink()

if FINAL_PATH.exists():
    FINAL_PATH.unlink()

print(
    "Writing V3 candidates..."
)

con.execute(
    f"""
    COPY (
        SELECT
            source1_entity_id,
            matched_entity_id
        FROM candidate_pairs_v3
    )
    TO '{TEMP_FINAL_PATH.as_posix()}'
    (
        HEADER,
        DELIMITER '\\t'
    )
    """
)


# ============================================================
# 22. APPEND ONLY NEW V4.1 CANDIDATES
# ============================================================

print(
    "Appending new V4.1 candidates..."
)

# DuckDB writes a temporary file for the new part.
NEW_PART_PATH = OUTPUT_DIR / "candidate_pairs_test_v4_new_part.tsv"

if NEW_PART_PATH.exists():
    NEW_PART_PATH.unlink()

con.execute(
    f"""
    COPY (
        SELECT
            source1_entity_id,
            matched_entity_id
        FROM v4_new
    )
    TO '{NEW_PART_PATH.as_posix()}'
    (
        HEADER,
        DELIMITER '\\t'
    )
    """
)


# ============================================================
# 23. FILE-LEVEL MERGE
# ============================================================

print(
    "Combining V3 + new V4.1 candidates..."
)

with open(
    TEMP_FINAL_PATH,
    "ab"
) as out_file:

    with open(
        NEW_PART_PATH,
        "rb"
    ) as in_file:

        # Skip header of second file
        in_file.readline()

        shutil.copyfileobj(
            in_file,
            out_file,
            length=16 * 1024 * 1024
        )


# ============================================================
# 24. FINALIZE
# ============================================================

os.replace(
    TEMP_FINAL_PATH,
    FINAL_PATH
)

if NEW_PART_PATH.exists():
    NEW_PART_PATH.unlink()


# ============================================================
# 25. FINAL CHECK
# ============================================================

file_size_gb = (
    FINAL_PATH.stat().st_size
    / (1024 ** 3)
)

print()
print("=" * 75)
print("FINAL TEST BLOCKING RESULT")
print("=" * 75)

print(
    "V3 candidates       :",
    f"{v3_count:,}"
)

print(
    "V4.1 new candidates :",
    f"{new_count:,}"
)

print(
    "Expected final      :",
    f"{expected_final:,}"
)

print(
    "Final file size     :",
    f"{file_size_gb:.2f} GB"
)

print()
print(
    "FINAL FILE:"
)

print(
    FINAL_PATH
)

print()
print("SAFETY CHECK: PASSED")

print()
print(
    "IMPORTANT: This is the candidate/blocking file."
)

print(
    "Next step is matching/scoring, not final submission."
)


# ============================================================
# 26. CLOSE
# ============================================================

con.close()

print()
print("DuckDB closed.")
print("DONE.")