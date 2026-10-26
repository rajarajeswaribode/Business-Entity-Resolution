# ============================================================
# V4.1 TARGETED CROSS-SCRIPT / TRANSLITERATION BLOCKING
# ============================================================
#
# Purpose:
#   Keep ALL V3 candidates and add candidates specifically for
#   cross-script / transliteration misses.
#
# Existing V3:
#   610,617,498 candidates
#   95.4793% blocking recall
#
# This script DOES NOT use ground truth to generate candidates.
#
# ============================================================

from pathlib import Path
import os
import re
import time
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

CLEAN_DIR = BASE_DIR / "cleaned_data"
OUTPUT_DIR = BASE_DIR / "output"

SOURCE1_PATH = CLEAN_DIR / "source1_clean.tsv"
SOURCE2_PATH = CLEAN_DIR / "source2_clean.tsv"
SOURCE3_PATH = CLEAN_DIR / "source3_clean.tsv"

V3_CANDIDATES = OUTPUT_DIR / "candidate_pairs_v3.tsv"

V4_ADDITIONAL = OUTPUT_DIR / "candidate_pairs_v4_1_additional.tsv"
V4_FINAL = OUTPUT_DIR / "candidate_pairs_v4_1.tsv"

DB_PATH = OUTPUT_DIR / "blocking_v4_1.duckdb"
TEMP_DIR = OUTPUT_DIR / "duckdb_temp_v4_1"


# ============================================================
# 2. SETTINGS
# ============================================================

CHUNK_SIZE = 100_000

THREADS = max(
    4,
    min(12, os.cpu_count() or 8)
)

MEMORY_LIMIT = "8GB"

# Only keep blocks occurring at most this many times
MAX_KEY_FREQUENCY = 50

# From each block, take at most this many S1 rows
# and this many target rows.
MAX_ROWS_PER_KEY = 50

# Maximum number of transliterated tokens stored per record
MAX_TOKEN_BLOCKS_PER_RECORD = 2

MIN_TOKEN_LENGTH = 4

# Prefix/suffix lengths
PREFIX_LEN = 6
SUFFIX_LEN = 6


# ============================================================
# 3. BASIC CHECKS
# ============================================================

start_time = time.time()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

for p in [
    SOURCE1_PATH,
    SOURCE2_PATH,
    SOURCE3_PATH,
    V3_CANDIDATES
]:
    if not p.exists():
        raise FileNotFoundError(f"Missing file:\n{p}")

print("=" * 70)
print("V4.1 TARGETED CROSS-SCRIPT BLOCKING")
print("=" * 70)

print(f"S1 : {SOURCE1_PATH}")
print(f"S2 : {SOURCE2_PATH}")
print(f"S3 : {SOURCE3_PATH}")
print(f"V3 : {V3_CANDIDATES}")
print()


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def has_non_ascii(value):
    if value is None:
        return False

    value = str(value)

    return any(ord(ch) > 127 for ch in value)


def transliterate(value):
    if value is None:
        return ""

    value = str(value)

    if not value:
        return ""

    # If already ASCII, don't waste time transliterating it.
    if value.isascii():
        return value

    value = unidecode(value)

    value = value.casefold()

    value = re.sub(r"[^a-z0-9]+", " ", value)

    value = re.sub(r"\s+", " ", value).strip()

    return value


def make_tokens(value):
    if not value:
        return []

    return [
        x
        for x in value.split()
        if len(x) >= MIN_TOKEN_LENGTH
    ]


def make_blocks(name):
    """
    Generate a SMALL number of transliteration blocks.

    Blocks:
      N_FULL   = full transliterated name
      N_PREF   = first 6 characters
      N_SUFF   = last 6 characters
      N_TOKEN  = at most 2 longest tokens

    Important:
      We only create these blocks for records where the
      original name contains non-ASCII OR where needed as
      an ASCII counterpart.
    """

    t = transliterate(name)

    if not t:
        return []

    blocks = []

    # Full transliterated name
    blocks.append(("N_FULL", t))

    # Prefix
    compact = t.replace(" ", "")

    if len(compact) >= PREFIX_LEN:
        blocks.append(
            ("N_PREF", compact[:PREFIX_LEN])
        )

    # Suffix
    if len(compact) >= SUFFIX_LEN:
        blocks.append(
            ("N_SUFF", compact[-SUFFIX_LEN:])
        )

    # Longest tokens only
    tokens = make_tokens(t)

    tokens = sorted(
        set(tokens),
        key=lambda x: (-len(x), x)
    )

    for token in tokens[:MAX_TOKEN_BLOCKS_PER_RECORD]:
        blocks.append(
            ("N_TOKEN", token)
        )

    return blocks


# ============================================================
# 5. CONNECT TO DUCKDB
# ============================================================

if DB_PATH.exists():
    print("Removing previous V4.1 database...")
    DB_PATH.unlink()

if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)

TEMP_DIR.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(str(DB_PATH))

con.execute(
    f"PRAGMA threads={THREADS}"
)

con.execute(
    f"PRAGMA memory_limit='{MEMORY_LIMIT}'"
)

con.execute(
    f"PRAGMA temp_directory='{TEMP_DIR.as_posix()}'"
)


# ============================================================
# 6. CREATE BLOCK TABLE
# ============================================================

con.execute("""
CREATE TABLE blocks (
    source_type VARCHAR,
    entity_id VARCHAR,
    country VARCHAR,
    block_type VARCHAR,
    block_key VARCHAR,
    is_non_ascii BOOLEAN
)
""")


# ============================================================
# 7. PROCESS ONE SOURCE
# ============================================================

def process_source(path, source_type):
    print()
    print("-" * 70)
    print(f"Processing {source_type}")
    print(path)
    print("-" * 70)

    total_rows = 0
    block_rows = 0
    non_ascii_rows = 0

    for chunk_no, df in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        required = [
            "entity_id",
            "business_name_clean",
            "country"
        ]

        missing = [
            c for c in required
            if c not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{source_type}: missing columns {missing}"
            )

        rows = []

        for row in df.itertuples(index=False):

            entity_id = getattr(row, "entity_id")
            name = getattr(row, "business_name_clean")
            country = getattr(row, "country")

            if not name:
                continue

            non_ascii = has_non_ascii(name)

            if non_ascii:
                non_ascii_rows += 1

            blocks = make_blocks(name)

            for block_type, block_key in blocks:

                if not block_key:
                    continue

                rows.append(
                    (
                        source_type,
                        entity_id,
                        country,
                        block_type,
                        block_key,
                        non_ascii
                    )
                )

        if rows:

            block_df = pd.DataFrame(
                rows,
                columns=[
                    "source_type",
                    "entity_id",
                    "country",
                    "block_type",
                    "block_key",
                    "is_non_ascii"
                ]
            )

            con.register(
                "current_blocks",
                block_df
            )

            con.execute("""
                INSERT INTO blocks
                SELECT
                    source_type,
                    entity_id,
                    country,
                    block_type,
                    block_key,
                    is_non_ascii
                FROM current_blocks
            """)

            con.unregister("current_blocks")

            block_rows += len(rows)

        total_rows += len(df)

        if chunk_no % 5 == 0:
            print(
                f"{source_type}: "
                f"{total_rows:,} rows | "
                f"{block_rows:,} blocks"
            )

    print(
        f"{source_type} complete: "
        f"{total_rows:,} rows | "
        f"{non_ascii_rows:,} non-ASCII names | "
        f"{block_rows:,} blocks"
    )


# ============================================================
# 8. BUILD BLOCKS
# ============================================================

process_source(
    SOURCE1_PATH,
    "S1"
)

process_source(
    SOURCE2_PATH,
    "S2"
)

process_source(
    SOURCE3_PATH,
    "S3"
)


# ============================================================
# 9. INDEX
# ============================================================

print()
print("Creating DuckDB block index...")

con.execute("""
CREATE INDEX idx_blocks_lookup
ON blocks(country, block_type, block_key)
""")


# ============================================================
# 10. BLOCK FREQUENCY
# ============================================================

print("Calculating block frequencies...")

con.execute("""
CREATE TABLE valid_blocks AS
SELECT
    country,
    block_type,
    block_key
FROM blocks
GROUP BY
    country,
    block_type,
    block_key
HAVING COUNT(*) <= ?
""", [MAX_KEY_FREQUENCY])


print(
    "Valid blocks:",
    con.execute(
        "SELECT COUNT(*) FROM valid_blocks"
    ).fetchone()[0]
)


# ============================================================
# 11. RANK ROWS INSIDE EACH BLOCK
# ============================================================
#
# We deliberately cap each side independently.
#
# This prevents:
#
#     1 S1 x 10,000 target
#
# from becoming a huge Cartesian product.
#
# ============================================================

print("Ranking rows inside blocks...")

con.execute("""
CREATE TABLE ranked_blocks AS
WITH filtered AS (
    SELECT b.*
    FROM blocks b
    INNER JOIN valid_blocks v
        ON  b.country = v.country
        AND b.block_type = v.block_type
        AND b.block_key = v.block_key
),

ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY
                source_type,
                country,
                block_type,
                block_key
            ORDER BY entity_id
        ) AS rn
    FROM filtered
)

SELECT
    source_type,
    entity_id,
    country,
    block_type,
    block_key,
    is_non_ascii
FROM ranked
WHERE rn <= ?
""", [MAX_ROWS_PER_KEY])


# ============================================================
# 12. GENERATE ONLY CROSS-SCRIPT CANDIDATES
# ============================================================
#
# We DO NOT add ASCII <-> ASCII candidates here.
#
# V3 already handles the normal ASCII case.
#
# New candidates are only:
#
#   S1 non-ASCII -> target ASCII
#   S1 ASCII     -> target non-ASCII
#
# This is the important reduction.
#
# ============================================================

print()
print("Generating targeted cross-script candidates...")

con.execute("""
CREATE TABLE v4_additional_raw AS

SELECT DISTINCT
    s.entity_id AS source1_entity_id,
    t.entity_id AS matched_entity_id

FROM ranked_blocks s

INNER JOIN ranked_blocks t
    ON  s.country = t.country
    AND s.block_type = t.block_type
    AND s.block_key = t.block_key

WHERE
    s.source_type = 'S1'
    AND t.source_type IN ('S2', 'S3')

    AND (
        (
            s.is_non_ascii = TRUE
            AND t.is_non_ascii = FALSE
        )
        OR
        (
            s.is_non_ascii = FALSE
            AND t.is_non_ascii = TRUE
        )
    )
""")


raw_additional = con.execute("""
SELECT COUNT(*)
FROM v4_additional_raw
""").fetchone()[0]

print(
    f"Raw V4.1 additional candidates: "
    f"{raw_additional:,}"
)


# ============================================================
# 13. REMOVE V3 DUPLICATES
# ============================================================

print()
print("Removing candidates already present in V3...")


con.execute("""
CREATE TABLE v3_candidates AS

SELECT
    source1_entity_id,
    matched_entity_id

FROM read_csv(
    ?,
    delim='\t',
    header=true,
    all_varchar=true,
    ignore_errors=false
)
""", [str(V3_CANDIDATES)])


v3_count = con.execute("""
SELECT COUNT(*)
FROM v3_candidates
""").fetchone()[0]


print(
    f"V3 candidates loaded: {v3_count:,}"
)


con.execute("""
CREATE TABLE v4_new AS

SELECT
    r.source1_entity_id,
    r.matched_entity_id

FROM v4_additional_raw r

ANTI JOIN v3_candidates v

    ON  r.source1_entity_id = v.source1_entity_id
    AND r.matched_entity_id = v.matched_entity_id
""")


new_count = con.execute("""
SELECT COUNT(*)
FROM v4_new
""").fetchone()[0]


print(
    f"Genuinely new candidates: {new_count:,}"
)


# ============================================================
# 14. FINAL V4.1
# ============================================================

print()
print("Building final V4.1 candidate file...")

con.execute("""
CREATE TABLE v4_final AS

SELECT
    source1_entity_id,
    matched_entity_id
FROM v3_candidates

UNION

SELECT
    source1_entity_id,
    matched_entity_id
FROM v4_new
""")


final_count = con.execute("""
SELECT COUNT(*)
FROM v4_final
""").fetchone()[0]


print(
    f"Final V4.1 candidates: {final_count:,}"
)


# ============================================================
# 15. SAFETY CHECKS
# ============================================================

print()
print("=" * 70)
print("SAFETY CHECKS")
print("=" * 70)


# ------------------------------------------------------------
# Duplicate pairs
# ------------------------------------------------------------

duplicate_groups = con.execute("""
SELECT COUNT(*)
FROM (
    SELECT
        source1_entity_id,
        matched_entity_id,
        COUNT(*) AS c
    FROM v4_final
    GROUP BY
        source1_entity_id,
        matched_entity_id
    HAVING COUNT(*) > 1
)
""").fetchone()[0]


print(
    f"Duplicate pair groups: {duplicate_groups:,}"
)


# ------------------------------------------------------------
# Invalid source pairs
# ------------------------------------------------------------

invalid_pairs = con.execute("""
SELECT COUNT(*)
FROM v4_final
WHERE
    NOT starts_with(source1_entity_id, 'S1-')
    OR
    NOT (
        starts_with(matched_entity_id, 'S2-')
        OR
        starts_with(matched_entity_id, 'S3-')
    )
""").fetchone()[0]


print(
    f"Invalid source pairs: {invalid_pairs:,}"
)


# ------------------------------------------------------------
# NULL / empty pairs
# ------------------------------------------------------------

null_pairs = con.execute("""
SELECT COUNT(*)
FROM v4_final
WHERE
    source1_entity_id IS NULL
    OR matched_entity_id IS NULL
    OR source1_entity_id = ''
    OR matched_entity_id = ''
""").fetchone()[0]


print(
    f"NULL/empty pairs: {null_pairs:,}"
)


# ------------------------------------------------------------
# V4 must contain at least all V3 candidates
# ------------------------------------------------------------

if final_count < v3_count:
    raise RuntimeError(
        "FATAL ERROR: V4.1 has fewer candidates than V3."
    )


# ------------------------------------------------------------
# Every V3 pair must exist in V4
# ------------------------------------------------------------

missing_v3 = con.execute("""
SELECT COUNT(*)
FROM v3_candidates v

ANTI JOIN v4_final f

    ON  v.source1_entity_id = f.source1_entity_id
    AND v.matched_entity_id = f.matched_entity_id
""").fetchone()[0]


print(
    f"V3 pairs missing from final: {missing_v3:,}"
)


# ============================================================
# 16. EXPORT ADDITIONAL
# ============================================================

print()
print("Exporting additional candidates...")

con.execute("""
COPY (
    SELECT
        source1_entity_id,
        matched_entity_id
    FROM v4_new
    ORDER BY
        source1_entity_id,
        matched_entity_id
)
TO ?
(
    HEADER,
    DELIMITER '\t'
)
""", [str(V4_ADDITIONAL)])


# ============================================================
# 17. EXPORT FINAL
# ============================================================

print("Exporting final V4.1 candidates...")

con.execute("""
COPY (
    SELECT
        source1_entity_id,
        matched_entity_id
    FROM v4_final
    ORDER BY
        source1_entity_id,
        matched_entity_id
)
TO ?
(
    HEADER,
    DELIMITER '\t'
)
""", [str(V4_FINAL)])


# ============================================================
# 18. FINAL SAFETY RESULT
# ============================================================

if (
    duplicate_groups != 0
    or
    invalid_pairs != 0
    or
    null_pairs != 0
    or
    missing_v3 != 0
):
    print()
    print("SAFETY CHECK: FAILED")
    con.close()
    raise RuntimeError(
        "Safety checks failed. Do NOT use the V4.1 file."
    )


print()
print("=" * 70)
print("SAFETY CHECK: PASSED")
print("=" * 70)


# ============================================================
# 19. FILE INFORMATION
# ============================================================

file_size_mb = (
    V4_FINAL.stat().st_size / (1024 * 1024)
)

elapsed_minutes = (
    time.time() - start_time
) / 60


print()
print("RESULTS")
print("-" * 70)

print(
    f"V3 candidates          : {v3_count:,}"
)

print(
    f"Raw additional         : {raw_additional:,}"
)

print(
    f"Genuinely new          : {new_count:,}"
)

print(
    f"Final V4.1 candidates  : {final_count:,}"
)

print(
    f"Duplicate groups       : {duplicate_groups:,}"
)

print(
    f"Invalid pairs          : {invalid_pairs:,}"
)

print(
    f"NULL/empty pairs       : {null_pairs:,}"
)

print(
    f"V3 pairs missing       : {missing_v3:,}"
)

print(
    f"Final file size        : {file_size_mb:.2f} MB"
)

print(
    f"Total time             : {elapsed_minutes:.2f} minutes"
)

print()
print("FINAL FILE:")
print(V4_FINAL)

print()
print("ADDITIONAL FILE:")
print(V4_ADDITIONAL)

print()
print("=" * 70)
print("V4.1 COMPLETE")
print("=" * 70)

con.close()